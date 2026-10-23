"""HippoRAG 2 style Non-Parametric Associative Graph Memory (ADR-0727 / T-11578).

Implements a deterministic, zero-network Entity-Relation Knowledge Graph with:
1. **Personalized PageRank (PPR)** with restart (ICML 2025 HippoRAG 2 formulation)
   over entity nodes and relation edges to surface associative context.
2. **Multi-Hop Bridge Reasoning (1-hop, 2-hop, 3-hop)**:
   Resolves compositional queries such as:
   - "『源氏物語』の作者が生まれた都市の現在の都道府県庁所在地は？"
   - "日本で最も広い湖がある都道府県の県庁所在地は？"
   - "Transformer論文の発表年の翌年にノーベル生理学・医学賞を受賞した日本人科学者は？"
   - "Pythonの開発者が生まれた国の首都は？"
3. **Citations & Provenance Chain**:
   Every hop records its Wikidata / standard source citation so the final synthesized
   response has a complete, machine-verifiable provenance chain.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

GRAPH_SCHEMA = "synapsemesh.hipporag2-graph.v3"
MAX_GRAPH_BYTES = 2_097_152
_PUNCT_RE = re.compile(r"[\s、。！？!?・,.\"'「」『』（）()\[\]]+")
_HOP_SPLIT_RE = re.compile(r"の")


def normalise_key(text: str) -> str:
    """NFKC + casefold + punctuation strip for deterministic entity/relation matching."""
    return _PUNCT_RE.sub("", unicodedata.normalize("NFKC", text).casefold())


@dataclass(frozen=True)
class GraphEdge:
    subject: str
    subject_aliases: tuple[str, ...]
    relation: str
    relation_aliases: tuple[str, ...]
    object: str
    object_aliases: tuple[str, ...]
    domain: str
    citation: str

    def all_subject_forms(self) -> tuple[str, ...]:
        return (self.subject, *self.subject_aliases)

    def all_relation_forms(self) -> tuple[str, ...]:
        return (self.relation, *self.relation_aliases)

    def all_object_forms(self) -> tuple[str, ...]:
        return (self.object, *self.object_aliases)


@dataclass(frozen=True)
class GraphHopTrace:
    hop_index: int
    subject: str
    relation: str
    object: str
    citation: str


@dataclass(frozen=True)
class GraphReasoningResult:
    answer_object: str
    formatted_answer: str
    hops: tuple[GraphHopTrace, ...]
    ppr_top_entities: tuple[tuple[str, float], ...]
    citations: tuple[str, ...]
    pack_id: str


class HippoRagGraphMemory:
    """Non-parametric associative knowledge graph with Personalized PageRank and multi-hop."""

    def __init__(
        self,
        pack_id: str,
        license_name: str,
        edges: tuple[GraphEdge, ...],
        sha256: str,
    ) -> None:
        self.pack_id = pack_id
        self.license = license_name
        self.edges = edges
        self.sha256 = sha256

        # Build canonical entity list and adjacency structures
        entity_set: dict[str, str] = {}
        self._alias_to_canonical: dict[str, str] = {}
        self._out_edges: dict[str, list[GraphEdge]] = {}
        self._in_edges: dict[str, list[GraphEdge]] = {}

        for edge in edges:
            subj_canon = edge.subject
            obj_canon = edge.object
            entity_set[normalise_key(subj_canon)] = subj_canon
            entity_set[normalise_key(obj_canon)] = obj_canon

            for s_form in edge.all_subject_forms():
                self._alias_to_canonical[normalise_key(s_form)] = subj_canon
            for o_form in edge.all_object_forms():
                self._alias_to_canonical[normalise_key(o_form)] = obj_canon

            self._out_edges.setdefault(subj_canon, []).append(edge)
            self._in_edges.setdefault(obj_canon, []).append(edge)

        self.entities: tuple[str, ...] = tuple(sorted(set(entity_set.values())))

    @classmethod
    def load(cls, path: str | Path) -> HippoRagGraphMemory:
        target = Path(path)
        raw = target.read_bytes()
        if len(raw) > MAX_GRAPH_BYTES:
            raise ValueError("graph memory file exceeds byte bound")
        payload = json.loads(raw.decode("utf-8"))
        if not isinstance(payload, dict) or payload.get("schema") != GRAPH_SCHEMA:
            raise ValueError(f"unsupported graph schema: {payload.get('schema')}")
        pack_id = str(payload["pack_id"])
        license_name = str(payload["license"])
        raw_edges = payload.get("edges", [])
        if not isinstance(raw_edges, list) or not raw_edges:
            raise ValueError("edges must be a non-empty list")

        edges: list[GraphEdge] = []
        for row in raw_edges:
            edges.append(
                GraphEdge(
                    subject=str(row["subject"]),
                    subject_aliases=tuple(str(x) for x in row.get("subject_aliases", [])),
                    relation=str(row["relation"]),
                    relation_aliases=tuple(str(x) for x in row.get("relation_aliases", [])),
                    object=str(row["object"]),
                    object_aliases=tuple(str(x) for x in row.get("object_aliases", [])),
                    domain=str(row.get("domain", "general")),
                    citation=str(row["citation"]),
                )
            )
        digest = hashlib.sha256(raw).hexdigest()
        return cls(pack_id, license_name, tuple(edges), digest)

    def detect_seed_entities(self, query: str) -> list[str]:
        """Find canonical entities whose aliases appear in the query, longest match first."""
        norm_q = normalise_key(query)
        matches: list[tuple[int, int, str]] = []
        for alias_norm, canon in self._alias_to_canonical.items():
            if not alias_norm or len(alias_norm) < 2:
                continue
            pos = norm_q.find(alias_norm)
            if pos != -1:
                matches.append((pos, -len(alias_norm), canon))
        # Sort by earliest occurrence in the query, then longest alias match
        matches.sort()
        seen: list[str] = []
        for _, _, canon in matches:
            if canon not in seen:
                seen.append(canon)
        return seen

    def personalized_pagerank(
        self,
        seed_entities: list[str],
        *,
        damping: float = 0.85,
        max_iter: int = 30,
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """Run deterministic Personalized PageRank (PPR) with restart from seed_entities."""
        if not self.entities or not seed_entities:
            return []
        valid_seeds = [s for s in seed_entities if s in self.entities]
        if not valid_seeds:
            return []

        n = len(self.entities)
        idx_map = {ent: i for i, ent in enumerate(self.entities)}
        teleport = [0.0] * n
        seed_weight = 1.0 / len(valid_seeds)
        for s in valid_seeds:
            teleport[idx_map[s]] = seed_weight

        # Build undirected/bidirectional transition list for associative spreading
        neighbors: list[list[int]] = [[] for _ in range(n)]
        for edge in self.edges:
            u = idx_map[edge.subject]
            v = idx_map[edge.object]
            if v not in neighbors[u]:
                neighbors[u].append(v)
            if u not in neighbors[v]:
                neighbors[v].append(u)

        scores = list(teleport)
        for _ in range(max_iter):
            next_scores = [(1.0 - damping) * teleport[i] for i in range(n)]
            for u in range(n):
                deg = len(neighbors[u])
                if deg > 0:
                    share = damping * scores[u] / deg
                    for v in neighbors[u]:
                        next_scores[v] += share
                else:
                    next_scores[u] += damping * scores[u]
            scores = next_scores

        ranked = sorted(
            ((self.entities[i], round(scores[i], 6)) for i in range(n)),
            key=lambda item: (-item[1], item[0]),
        )
        return ranked[:top_k]

    def _match_relation_in_text(
        self, current_entity: str, text_segment: str
    ) -> tuple[GraphEdge, int] | None:
        """Find outgoing edge from current_entity whose relation alias appears earliest."""
        norm_seg = normalise_key(text_segment)
        candidates: list[tuple[int, int, GraphEdge]] = []
        for edge in self._out_edges.get(current_entity, []):
            for rel_form in edge.all_relation_forms():
                rel_norm = normalise_key(rel_form)
                if not rel_norm:
                    continue
                pos = norm_seg.find(rel_norm)
                if pos != -1:
                    candidates.append((pos, -len(rel_norm), edge))
        if not candidates:
            return None
        candidates.sort(key=lambda item: (item[0], item[1]))
        best_pos, neg_len, best_edge = candidates[0]
        return best_edge, best_pos + (-neg_len)

    def resolve_multihop(self, query: str, *, max_hops: int = 4) -> GraphReasoningResult | None:
        """Resolve 1-hop, 2-hop, or 3-hop entity-relation chains deterministically."""
        norm_q = normalise_key(query)
        seeds = self.detect_seed_entities(query)
        if not seeds:
            return None

        ppr_top = tuple(self.personalized_pagerank(seeds, top_k=5))

        # Try starting from each detected seed entity and greedily following relation mentions
        best_hops: list[GraphHopTrace] = []
        for seed in seeds:
            # Find where the seed alias ends in norm_q so we follow relations in forward order
            seed_end = 0
            for edge in self._out_edges.get(seed, []):
                for s_form in edge.all_subject_forms():
                    s_norm = normalise_key(s_form)
                    pos = norm_q.find(s_norm)
                    if pos != -1:
                        seed_end = max(seed_end, pos + len(s_norm))

            remaining_text = norm_q[seed_end:] if seed_end > 0 else norm_q
            curr_ent = seed
            hops: list[GraphHopTrace] = []
            for hop_idx in range(1, max_hops + 1):
                matched = self._match_relation_in_text(curr_ent, remaining_text)
                if matched is None:
                    # Also check full query if relation preceded subject (e.g. relative clause)
                    if hop_idx == 1:
                        matched = self._match_relation_in_text(curr_ent, norm_q)
                    if matched is None:
                        break
                    edge, _ = matched
                    remaining_text = ""
                else:
                    edge, consumed_end = matched
                    remaining_text = remaining_text[consumed_end:]

                hops.append(
                    GraphHopTrace(
                        hop_index=hop_idx,
                        subject=edge.subject,
                        relation=edge.relation,
                        object=edge.object,
                        citation=edge.citation,
                    )
                )
                curr_ent = edge.object

            if len(hops) > len(best_hops):
                best_hops = hops

        if not best_hops:
            return None

        final_obj = best_hops[-1].object
        citations = tuple(dict.fromkeys(h.citation for h in best_hops))
        if len(best_hops) == 1:
            h0 = best_hops[0]
            formatted = f"{h0.subject}の{h0.relation}は{h0.object}です。"
        else:
            chain_desc = " → ".join(
                f"{h.subject}の{h.relation}（{h.object}）" for h in best_hops[:-1]
            )
            last_h = best_hops[-1]
            last_str = f"{last_h.subject}の{last_h.relation}は{last_h.object}です。"
            formatted = (
                f"推論経路: {chain_desc} → {last_str}\n"
                f"答え: {final_obj}"
            )

        return GraphReasoningResult(
            answer_object=final_obj,
            formatted_answer=formatted,
            hops=tuple(best_hops),
            ppr_top_entities=ppr_top,
            citations=citations,
            pack_id=self.pack_id,
        )

    def retrieve_subgraph_facts(self, query: str, *, top_k_entities: int = 5) -> list[str]:
        """Return grounded fact sentences for the top PPR-activated entities."""
        seeds = self.detect_seed_entities(query)
        if not seeds:
            return []
        ranked = self.personalized_pagerank(seeds, top_k=top_k_entities)
        facts: list[str] = []
        for ent, _score in ranked:
            for edge in self._out_edges.get(ent, []):
                cite = edge.citation
                facts.append(
                    f"{edge.subject}の{edge.relation}は{edge.object}です。（出典: {cite}）"
                )
        return facts
