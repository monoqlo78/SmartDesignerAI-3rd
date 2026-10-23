"""Bounded offline factual knowledge packs for exact, cited closed-book facts."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

SCHEMA = "synapsemesh.factual-knowledge.v1"
MAX_PACK_BYTES = 1_048_576
MAX_FACTS = 512
_SPACE_PUNCT = re.compile(r"[\s、。！？!?・,.]+")
_NEGATION_CUES = ("ではない", "じゃない", "でない", "以外", "除く", "除いて")


class FactualKnowledgeError(ValueError):
    """The factual pack is malformed or exceeds a declared bound."""


@dataclass(frozen=True)
class FactSource:
    provider: str
    subject_id: str
    property_id: str
    object_id: str
    url: str

    @property
    def citation(self) -> str:
        return f"{self.provider}:{self.subject_id}/{self.property_id}/{self.object_id}"


@dataclass(frozen=True)
class FactRecord:
    subject: str
    subject_aliases: tuple[str, ...]
    relation: str
    relation_aliases: tuple[str, ...]
    object: str
    source: FactSource

    def answer(self) -> str:
        return f"{self.subject}の{self.relation}は{self.object}です。"

    def reverse_answer(self) -> str:
        return f"{self.object}は{self.subject}の{self.relation}です。"


@dataclass(frozen=True)
class FactualAnswer:
    text: str
    citation: str
    pack_id: str


def _normalise(text: str) -> str:
    return _SPACE_PUNCT.sub("", unicodedata.normalize("NFKC", text).casefold())


def _strings(value: object, field: str) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(v, str) and v.strip() for v in value)
    ):
        raise FactualKnowledgeError(f"{field} must be a non-empty string list")
    return tuple(value)


class FactualKnowledgePack:
    """Data-only exact fact lookup; ambiguous and unsupported requests abstain."""

    def __init__(
        self, pack_id: str, license_name: str, facts: tuple[FactRecord, ...], digest: str
    ) -> None:
        self.pack_id = pack_id
        self.license = license_name
        self.facts = facts
        self.sha256 = digest

    @classmethod
    def load(cls, path: str | Path) -> FactualKnowledgePack:
        target = Path(path)
        raw = target.read_bytes()
        if len(raw) > MAX_PACK_BYTES:
            raise FactualKnowledgeError("factual pack exceeds byte bound")
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise FactualKnowledgeError("factual pack is not valid UTF-8 JSON") from exc
        if not isinstance(payload, dict) or set(payload) != {
            "schema",
            "pack_id",
            "license",
            "facts",
        }:
            raise FactualKnowledgeError("invalid factual pack envelope")
        if payload["schema"] != SCHEMA:
            raise FactualKnowledgeError("unsupported factual pack schema")
        pack_id, license_name = payload["pack_id"], payload["license"]
        if not isinstance(pack_id, str) or not pack_id.strip():
            raise FactualKnowledgeError("pack_id must be non-empty")
        if not isinstance(license_name, str) or not license_name.strip():
            raise FactualKnowledgeError("license must be non-empty")
        rows = payload["facts"]
        if not isinstance(rows, list) or not rows or len(rows) > MAX_FACTS:
            raise FactualKnowledgeError("facts must be a bounded non-empty list")
        facts: list[FactRecord] = []
        keys: set[tuple[str, str]] = set()
        for row in rows:
            if not isinstance(row, dict) or set(row) != {
                "subject",
                "subject_aliases",
                "relation",
                "relation_aliases",
                "object",
                "source",
            }:
                raise FactualKnowledgeError("invalid fact record")
            source_raw = row["source"]
            if not isinstance(source_raw, dict) or set(source_raw) != {
                "provider",
                "subject_id",
                "property_id",
                "object_id",
                "url",
            }:
                raise FactualKnowledgeError("invalid fact source")
            scalar_values = (row["subject"], row["relation"], row["object"], *source_raw.values())
            if not all(isinstance(value, str) and value.strip() for value in scalar_values):
                raise FactualKnowledgeError("fact scalar fields must be non-empty strings")
            if not source_raw["url"].startswith("https://"):
                raise FactualKnowledgeError("fact source URL must use HTTPS")
            fact = FactRecord(
                subject=row["subject"],
                subject_aliases=_strings(row["subject_aliases"], "subject_aliases"),
                relation=row["relation"],
                relation_aliases=_strings(row["relation_aliases"], "relation_aliases"),
                object=row["object"],
                source=FactSource(**source_raw),
            )
            key = (_normalise(fact.subject), _normalise(fact.relation))
            if key in keys:
                raise FactualKnowledgeError("duplicate subject/relation fact")
            keys.add(key)
            facts.append(fact)
        return cls(pack_id, license_name, tuple(facts), hashlib.sha256(raw).hexdigest())

    def answer(self, messages: list[dict[str, str]]) -> FactualAnswer | None:
        if not messages or messages[-1].get("role") != "user":
            return None
        current = _normalise(messages[-1]["content"])
        if any(_normalise(cue) in current for cue in _NEGATION_CUES):
            return None

        def mentions_subject(text: str, fact: FactRecord) -> bool:
            aliases = tuple(_normalise(value) for value in (fact.subject, *fact.subject_aliases))
            return any(alias in text for alias in aliases)

        def asks_relation_of_object(text: str, fact: FactRecord) -> bool:
            relations = (fact.relation, *fact.relation_aliases)
            return any(_normalise(f"{fact.object}の{relation}") in text for relation in relations)

        reverse_scope = [fact for fact in self.facts if asks_relation_of_object(current, fact)]
        if reverse_scope:
            subject_scope = [fact for fact in self.facts if _normalise(fact.object) in current]
        else:
            subject_scope = []
        reverse = bool(reverse_scope)
        if not subject_scope:
            explicit = [fact for fact in self.facts if mentions_subject(current, fact)]
            subject_scope = explicit
        if not subject_scope:
            for message in reversed(messages[:-1]):
                if message.get("role") != "user":
                    continue
                previous = _normalise(message["content"])
                inherited = [fact for fact in self.facts if mentions_subject(previous, fact)]
                if inherited:
                    subject_scope = inherited
                    break
        candidates: list[FactRecord] = []
        for fact in subject_scope:
            relations = tuple(
                _normalise(value) for value in (fact.relation, *fact.relation_aliases)
            )
            if any(relation in current for relation in relations):
                candidates.append(fact)
        if len(candidates) != 1:
            return None
        fact = candidates[0]
        text = fact.reverse_answer() if reverse else fact.answer()
        return FactualAnswer(text, fact.source.citation, self.pack_id)
