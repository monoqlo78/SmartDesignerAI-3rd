"""Loopback-only browser chat for v9qa plus bounded offline factual knowledge.

The model is loaded once in the background and reused across conversations. CPU is the default;
``--device cuda`` is an optional delivery accelerator. Stable exact facts may be answered by the
configured data-only Knowledge Pack before model generation. By default
the evaluated float32 weights are memory-mapped and decoded with a KV cache and sparse
MoE evaluation (ADR-0705; ~3.3 tokens/s on this PC versus ~0.15 for the first int8
build, artifacts/jawiki-v9/cpu-speed/summary.json); ``--weights int8`` keeps the T-11562
quantised path. A reply is a job: the browser posts the conversation, then polls the job
and shows the partial text as tokens arrive. One reply runs at a time.

No outbound network: the server binds to 127.0.0.1 only, refuses foreign Host headers,
and disables outbound socket connections in the process. Nothing here measures quality.

Usage:
    python -m src.llm.chat_web            # CPU, http://127.0.0.1:8791/
    python -m src.llm.chat_web --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import torch

from src.knowledge.factual import FactualKnowledgePack
from src.knowledge.hipporag_graph import HippoRagGraphMemory
from src.llm.byte_bpe import ByteBpe
from src.llm.chat_format_v2 import CHAT_FORMAT_V2, ChatFormatV2
from src.llm.native_distill import SovereignNativeCortex, build_distilled_trajectory_bank
from src.llm.synapse_lm_v2 import KVCache, SynapseLM
from src.toolbox import nextgen_cognitive, sol_cognitive_engine

MAX_REQUEST_BYTES = 64 * 1024
MAX_MESSAGE_CHARS = 2_000
MAX_MESSAGES = 8
MAX_NEW_TOKENS = 512
DEFAULT_NEW_TOKENS = 256
DEFAULT_FACTS = Path("data/knowledge/ja_core_facts_v2.json")
DEFAULT_SOL_GRAPH = Path("data/knowledge/ja_sol_knowledge_graph_v3.json")
DEFAULT_DISTILLED_CANDIDATE = Path(
    "artifacts/t11576/qwen3-4b-lora-20261019-run26/result/candidate.json"
)
SEED = 20261006
ROLES = ("user", "assistant")
IDENTITY_CUE = (
    "自己紹介",
    "あなたは誰",
    "あなたは何者",
    "モデル名",
    "who are you",
)
RETRY_CUES = (
    "もう一度",
    "もう一回",
    "やり直して",
    "再生成",
    "書き直して",
    "最初から",
    "続き",
    "retry",
    "regenerate",
)
IDENTITY_REPLY = (
    "私はSynapseMeshのローカル日本語モデル v9qa です。"
    "テキスト生成を担当し、検索・計算・動画解析などは別のツールや専門モデルと連携します。"
)
_DEFAULT_SOL_GRAPH_CACHE: HippoRagGraphMemory | None = None
_DEFAULT_DISTILLED_BANK_CACHE: dict[str, dict[str, str]] | None = None


def _is_retry_request(text: str) -> bool:
    stripped = text.strip()
    if not stripped or len(stripped) > 48:
        return False
    lowered = stripped.lower()
    return any(cue in lowered for cue in RETRY_CUES)


def prepare_conversation_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Sanitize multi-turn history while preserving context and resolving retry requests.

    - Preserves prior conversation turns so multi-turn follow-ups ('それをPythonにして', etc.)
      always carry over past context.
    - Removes poisoned turns where the assistant merely echoed a user prompt or retry phrase.
    - Auto-closes truncated markdown code fences in prior assistant turns so an earlier cutoff
      does not corrupt subsequent multi-turn generation.
    - When the latest user turn is a retry/regeneration request ('ごめん、もう一度生成して', etc.),
      replaces the failed/truncated prior attempt with the last substantive user prompt.
    """
    if not messages:
        return messages
    user_texts = {m["content"].strip() for m in messages if m.get("role") == "user"}
    cleaned: list[dict[str, str]] = []
    idx = 0
    n = len(messages)
    while idx < n:
        msg = messages[idx]
        if (
            msg["role"] == "user"
            and idx + 1 < n
            and messages[idx + 1]["role"] == "assistant"
        ):
            user_txt = msg["content"].strip()
            asst_txt = messages[idx + 1]["content"].strip()
            is_echo = asst_txt == user_txt or asst_txt in user_texts or _is_retry_request(asst_txt)
            if not is_echo and not _is_retry_request(user_txt):
                if asst_txt.count("```") % 2 == 1:
                    asst_txt = asst_txt + "\n```"
                cleaned.append({"role": "user", "content": user_txt})
                cleaned.append({"role": "assistant", "content": asst_txt})
            idx += 2
            continue
        cleaned.append({"role": msg["role"], "content": msg["content"].strip()})
        idx += 1

    last_user = messages[-1]["content"].strip()
    if _is_retry_request(last_user):
        prior_substantive = [
            m["content"].strip()
            for m in reversed(messages[:-1])
            if m.get("role") == "user" and not _is_retry_request(m["content"])
        ]
        if prior_substantive:
            target_prompt = prior_substantive[0]
            # Drop the immediately preceding assistant turn for target_prompt
            if (
                len(cleaned) >= 3
                and cleaned[-1]["role"] == "user"
                and cleaned[-2]["role"] == "assistant"
                and cleaned[-3]["role"] == "user"
                and cleaned[-3]["content"] == target_prompt
            ):
                cleaned.pop()  # retry user turn
                cleaned.pop()  # old assistant turn
            elif cleaned and cleaned[-1]["role"] == "user":
                cleaned[-1] = {"role": "user", "content": target_prompt}
            else:
                cleaned.append({"role": "user", "content": target_prompt})

    if not cleaned or cleaned[-1]["role"] != "user":
        return [{"role": "user", "content": last_user}]
    return cleaned


def _get_default_sol_graph() -> HippoRagGraphMemory | None:
    global _DEFAULT_SOL_GRAPH_CACHE
    if _DEFAULT_SOL_GRAPH_CACHE is None and DEFAULT_SOL_GRAPH.is_file():
        _DEFAULT_SOL_GRAPH_CACHE = HippoRagGraphMemory.load(DEFAULT_SOL_GRAPH)
    return _DEFAULT_SOL_GRAPH_CACHE


def _get_default_distilled_bank() -> dict[str, dict[str, str]]:
    global _DEFAULT_DISTILLED_BANK_CACHE
    if _DEFAULT_DISTILLED_BANK_CACHE is None:
        _DEFAULT_DISTILLED_BANK_CACHE = build_distilled_trajectory_bank(DEFAULT_DISTILLED_CANDIDATE)
    return _DEFAULT_DISTILLED_BANK_CACHE


def deterministic_reply(
    messages: list[dict[str, str]], knowledge: FactualKnowledgePack | None = None
) -> tuple[str, str] | None:
    """Exact high-confidence replies that should not be guessed by the language model."""
    question = messages[-1]["content"].strip()
    lowered = question.lower()
    if any(cue in lowered for cue in IDENTITY_CUE):
        return IDENTITY_REPLY, "identity"
    sol_verified = sol_cognitive_engine.resolve_sol_class_query(
        messages,
        knowledge=knowledge,
        sol_graph=_get_default_sol_graph(),
    )
    if sol_verified is not None:
        return sol_verified
    verified = nextgen_cognitive.verify_or_enhance_query(messages, knowledge=knowledge)
    if verified is not None:
        return verified
    bank = _get_default_distilled_bank()
    if bank:
        cortex = SovereignNativeCortex(
            knowledge=knowledge,
            sol_graph=_get_default_sol_graph(),
            distilled_bank=bank,
        )
        reply_text, source = cortex.reply(messages)
        if source != "native_synapse_lm_v2_safe_abstention":
            return reply_text, source
    return None


def stream_reply(
    model: SynapseLM,
    chat: ChatFormatV2,
    messages: list[dict[str, str]],
    *,
    max_new_tokens: int,
    temperature: float = 0.0,
    seed: int = SEED,
    top_k: int | None = None,
    on_text: Callable[[str, int], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    use_cache: bool = True,
) -> tuple[str, int, bool]:
    """``chat_v2_cli.reply`` one token at a time: same prompt, sampling and stop rule.

    Returns (text, generated tokens, stopped early). ``on_text`` receives the decoded
    partial reply after every token so a caller can show progress. ``use_cache`` keeps
    keys/values between tokens (T-11564); the logits equal the full-window recompute.
    """
    torch.manual_seed(seed)
    tokens, _ = chat.conversation_tokens(messages, training=False)
    context = model.config.context
    if len(tokens) >= context:
        tokens = tokens[-(context - 1) :]
    model.eval()
    device = next(model.parameters()).device
    sequence = torch.tensor([tokens], dtype=torch.long, device=device)
    produced: list[int] = []
    stopped = False
    kv = KVCache(model.config.depth) if use_cache else None
    with torch.no_grad():
        for _ in range(max_new_tokens):
            if should_stop is not None and should_stop():
                stopped = True
                break
            logits = model.next_logits(sequence, kv)
            if temperature <= 0.0:
                nxt = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    kth = logits.topk(min(top_k, logits.size(-1)), dim=-1).values[:, -1:]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                nxt = torch.multinomial(logits.softmax(dim=-1), num_samples=1)
            sequence = torch.cat([sequence, nxt], dim=1)
            produced.append(int(nxt))
            if on_text is not None:
                on_text(chat.decode_reply(produced), len(produced))
            if int(nxt) == chat.terminator_id:
                break
    return chat.decode_reply(produced), len(produced), stopped


def validate_messages(payload: object) -> list[dict[str, str]]:
    """Accept a short user/assistant history that ends with a user turn."""
    if not isinstance(payload, list) or not payload:
        raise ValueError("messages must be a non-empty list")
    messages: list[dict[str, str]] = []
    for item in payload[-MAX_MESSAGES:]:
        if not isinstance(item, dict):
            raise ValueError("each message must be an object")
        role, content = item.get("role"), item.get("content")
        if role not in ROLES or not isinstance(content, str) or not content.strip():
            raise ValueError("each message needs role user|assistant and non-empty content")
        if len(content) > MAX_MESSAGE_CHARS:
            raise ValueError(f"a message exceeds {MAX_MESSAGE_CHARS} characters")
        messages.append({"role": role, "content": content.strip()})
    if messages[-1]["role"] != "user":
        raise ValueError("the last message must be from the user")
    return messages


@dataclass
class Job:
    id: str
    messages: list[dict[str, str]]
    max_new_tokens: int
    status: str = "queued"
    text: str = ""
    tokens: int = 0
    error: str | None = None
    source: str = "model"
    started: float = field(default_factory=time.time)
    finished: float | None = None
    cancel: threading.Event = field(default_factory=threading.Event)

    def view(self) -> dict[str, Any]:
        end = self.finished or time.time()
        seconds = end - self.started
        return {
            "id": self.id,
            "status": self.status,
            "text": self.text,
            "tokens": self.tokens,
            "max_new_tokens": self.max_new_tokens,
            "seconds": round(seconds, 1),
            "tokens_per_second": round(self.tokens / seconds, 3) if seconds > 0 else None,
            "error": self.error,
            "source": self.source,
        }


Loader = Callable[[], tuple[SynapseLM, ChatFormatV2, dict[str, Any]]]


class ChatEngine:
    """Loads the model once (in the background) and runs one reply job at a time."""

    def __init__(self, loader: Loader, *, knowledge: FactualKnowledgePack | None = None) -> None:
        self._loader = loader
        self.knowledge = knowledge
        self.state = "loading"
        self.info: dict[str, Any] = {}
        self.error: str | None = None
        self.model: SynapseLM | None = None
        self.chat: ChatFormatV2 | None = None
        self._jobs: dict[str, Job] = {}
        self._busy: Job | None = None
        self._lock = threading.Lock()
        self._started = time.time()

    def load(self) -> None:
        try:
            self.model, self.chat, self.info = self._loader()
            self.info["load_seconds"] = round(time.time() - self._started, 1)
            self.state = "ready"
        except Exception as exc:  # surfaced to the browser as-is
            self.error = f"{type(exc).__name__}: {exc}"
            self.state = "error"

    def start_loading(self) -> threading.Thread:
        thread = threading.Thread(target=self.load, name="model-load", daemon=True)
        thread.start()
        return thread

    def status(self) -> dict[str, Any]:
        busy = self._busy
        return {
            "state": self.state,
            "error": self.error,
            "info": self.info,
            "busy": busy.id if busy is not None and busy.status == "running" else None,
            "seconds_since_start": round(time.time() - self._started, 1),
        }

    def submit(self, messages: list[dict[str, str]], max_new_tokens: int) -> Job:
        if self.state != "ready":
            raise RuntimeError(f"model is not ready ({self.state})")
        with self._lock:
            if self._busy is not None and self._busy.status in ("queued", "running"):
                raise BlockingIOError("another reply is still being generated")
            job = Job(uuid.uuid4().hex[:12], messages, max_new_tokens)
            self._jobs = {k: v for k, v in self._jobs.items() if v.status == "running"}
            self._jobs[job.id] = job
            self._busy = job
        threading.Thread(target=self._run, args=(job,), name=f"job-{job.id}", daemon=True).start()
        return job

    def _run(self, job: Job) -> None:
        assert self.model is not None and self.chat is not None
        job.status = "running"

        def progress(text: str, count: int) -> None:
            job.text, job.tokens = text, count

        try:
            effective_messages = prepare_conversation_messages(job.messages)
            guarded = deterministic_reply(effective_messages, self.knowledge)
            if guarded is not None:
                job.text, job.source = guarded
                job.tokens = 0
                job.status = "done"
                return
            use_multi_turn_base = len(effective_messages) > 1
            if hasattr(self.model, "bypass_lora"):
                self.model.bypass_lora = use_multi_turn_base  # type: ignore[attr-defined]
            try:
                text, count, stopped = stream_reply(
                    self.model,
                    self.chat,
                    effective_messages,
                    max_new_tokens=job.max_new_tokens,
                    on_text=progress,
                    should_stop=job.cancel.is_set,
                )
                user_inputs = {
                    m["content"].strip() for m in job.messages if m.get("role") == "user"
                }
                if not stopped and (
                    not text.strip()
                    or text.strip() in user_inputs
                    or _is_retry_request(text.strip())
                ):
                    if hasattr(self.model, "bypass_lora"):
                        self.model.bypass_lora = True  # type: ignore[attr-defined]
                    text, count, stopped = stream_reply(
                        self.model,
                        self.chat,
                        effective_messages,
                        max_new_tokens=job.max_new_tokens,
                        on_text=progress,
                        should_stop=job.cancel.is_set,
                    )
            finally:
                if hasattr(self.model, "bypass_lora"):
                    self.model.bypass_lora = False  # type: ignore[attr-defined]
            job.text, job.tokens = text, count
            job.status = "cancelled" if stopped else "done"
        except Exception as exc:
            job.error = f"{type(exc).__name__}: {exc}"
            job.status = "error"
        finally:
            job.finished = time.time()

    def job(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)


WEIGHTS = ("f32", "int8")
DEVICES = ("cpu", "cuda")
DEFAULT_THREADS = min(8, os.cpu_count() or 1)


class _HfTokenizerAdapter:
    """Adapt a Hugging Face tokenizer with Qwen3 non-thinking chat template to ChatEngine."""

    MULTI_TURN_SYSTEM = (
        "あなたは正確で簡潔な日本語AIアシスタントです。"
        "これまでの会話履歴の文脈を正確に引き継ぎ、要点を絞って簡潔・完結に回答してください。"
    )

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer
        self.terminator_id = int(tokenizer.eos_token_id)

    def conversation_tokens(
        self, messages: list[dict[str, str]], *, training: bool = False
    ) -> tuple[list[int], list[int]]:
        del training
        chat_messages = list(messages)
        if len(chat_messages) > 1 and chat_messages[0].get("role") != "system":
            chat_messages = [
                {"role": "system", "content": self.MULTI_TURN_SYSTEM},
                *chat_messages,
            ]
        prompt = self.tokenizer.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        ids = list(self.tokenizer(prompt, add_special_tokens=False)["input_ids"])
        return ids, [-100] * len(ids)

    def decode_reply(self, ids: list[int]) -> str:
        clean = [tok for tok in ids if tok != self.terminator_id]
        return str(self.tokenizer.decode(clean, skip_special_tokens=True))


class _HfCausalLMWrapper:
    """Wrap a Hugging Face / PEFT causal LM so ``stream_reply`` works with KV cache."""

    def __init__(self, model: Any, context: int = 4096) -> None:
        self.model = model
        self.bypass_lora = False
        self.config = type("_Cfg", (), {"context": context, "depth": 1})()

    def eval(self) -> _HfCausalLMWrapper:
        self.model.eval()
        return self

    def parameters(self) -> Any:
        return self.model.parameters()

    def next_logits(self, sequence: torch.Tensor, kv: KVCache | None = None) -> torch.Tensor:
        past = getattr(kv, "_hf_past", None) if kv is not None else None
        input_ids = sequence[:, -1:] if past is not None else sequence
        if self.bypass_lora and hasattr(self.model, "disable_adapter"):
            with self.model.disable_adapter():
                out = self.model(
                    input_ids=input_ids, past_key_values=past, use_cache=kv is not None
                )
        else:
            out = self.model(input_ids=input_ids, past_key_values=past, use_cache=kv is not None)
        if kv is not None:
            kv._hf_past = out.past_key_values  # type: ignore[attr-defined]
        return out.logits[:, -1, :]


def load_hf_lora(
    model_id: str,
    revision: str,
    adapter: Path | None = None,
    device: str = "cuda",
) -> Loader:
    """Load pinned Qwen3-4B (NF4 double-quantized on CUDA) with an optional PEFT LoRA adapter."""
    if device != "cuda":
        raise ValueError("NF4 QLoRA web serving requires --device cuda")

    def loader() -> tuple[Any, Any, dict[str, Any]]:
        import hashlib

        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but torch reports no CUDA device")
        local_Only = True
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, revision=revision, local_files_only=local_Only
        )
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        base = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            quantization_config=quant,
            device_map={"": 0},
            torch_dtype=torch.float16,
            local_files_only=local_Only,
        )
        run_label = f"{model_id}@{revision[:8]}"
        step = 0
        if adapter is not None:
            weights_file = adapter / "adapter_model.safetensors"
            if not weights_file.is_file():
                raise FileNotFoundError(f"no LoRA adapter weights at {weights_file}")
            adapter_sha = hashlib.sha256(weights_file.read_bytes()).hexdigest()[:12]
            base = PeftModel.from_pretrained(base, str(adapter), is_trainable=False)
            run_label = f"{adapter.as_posix()} (sha256:{adapter_sha})"
            step = 90
        base.eval()
        params = sum(p.numel() for p in base.parameters())
        info: dict[str, Any] = {
            "run": run_label,
            "step": step,
            "parameters": params,
            "decoding": "greedy, KV cache, Qwen3-4B NF4" + (" + LoRA" if adapter else ""),
            "threads": torch.get_num_threads(),
            "device": device,
            "inference_weights": "nf4_double + fp16 LoRA" if adapter else "nf4_double base",
            "cuda_device": torch.cuda.get_device_name(0),
        }
        return _HfCausalLMWrapper(base), _HfTokenizerAdapter(tokenizer), info

    return loader


def load_v9(run: Path, tokenizer: Path, weights: str = "f32", device: str = "cpu") -> Loader:
    """Fast CPU decode (ADR-0705): sparse MoE evaluation + KV cache.

    ``f32`` serves the evaluated float32 weights straight from the memory-mapped
    checkpoint (fastest measured, artifacts/jawiki-v9/cpu-speed/summary.json); ``int8``
    is the T-11562 weight-only quantisation for a PC that cannot page the checkpoint.
    """
    if weights not in WEIGHTS:
        raise ValueError(f"weights must be one of {WEIGHTS}")
    if device not in DEVICES:
        raise ValueError(f"device must be one of {DEVICES}")
    if device == "cuda" and weights == "int8":
        raise ValueError("int8 weight-only quantisation is a CPU path; use --weights f32 on cuda")

    def loader() -> tuple[SynapseLM, ChatFormatV2, dict[str, Any]]:
        from src.llm.quant_int8 import quantize_int8
        from src.llm.sample_lm import load_run

        if not (run / "checkpoint.pt").is_file():
            raise FileNotFoundError(f"no checkpoint at {run}; refusing to serve a stand-in")
        model, _tok, state = load_run(run, tokenizer, None)
        params = sum(p.numel() for p in model.parameters())
        if state.get("chat_format") != CHAT_FORMAT_V2:
            raise ValueError(f"{run} is not a {CHAT_FORMAT_V2} chat checkpoint")
        step = int(state.get("step", -1))
        state.clear()
        info: dict[str, Any] = {
            "run": run.as_posix(),
            "step": step,
            "parameters": params,
            "decoding": "greedy, KV cache, sparse MoE",
            "threads": torch.get_num_threads(),
            "device": device,
        }
        if weights == "int8":
            stats = quantize_int8(model)
            info["inference_weights"] = "int8 weight-only (per-row absmax)"
            info["int8_bytes"] = stats["int8_bytes"]
        else:
            info["inference_weights"] = "float32 (evaluated weights, memory-mapped)"
        if device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("--device cuda requested but torch reports no CUDA device")
            model = model.to("cuda")
            info["inference_weights"] = "float32 (evaluated weights, resident on cuda)"
            info["cuda_device"] = torch.cuda.get_device_name(0)
        model.set_sparse_eval(True)
        return model.eval(), ChatFormatV2(ByteBpe.load(tokenizer)), info

    return loader


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "SynapseChatWeb/1"
    sys_version = ""

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return

    @property
    def engine(self) -> ChatEngine:
        return self.server.engine  # type: ignore[attr-defined]

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
            "connect-src 'self'",
        )
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, payload: dict[str, Any]) -> None:
        self._send(
            code, json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json"
        )

    def _host_ok(self) -> bool:
        # Blocks DNS rebinding: only loopback names reach the API.
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0]
        if host in ("127.0.0.1", "localhost"):
            return True
        self._json(403, {"error": "loopback host only"})
        return False

    def do_GET(self) -> None:  # noqa: N802
        if not self._host_ok():
            return
        if self.path in ("/", "/index.html"):
            page = Path(__file__).with_name("chat_web.html").read_bytes()
            self._send(200, page, "text/html; charset=utf-8")
        elif self.path == "/status":
            self._json(200, self.engine.status())
        elif self.path.startswith("/job/"):
            job = self.engine.job(self.path.removeprefix("/job/"))
            if job is None:
                self._json(404, {"error": "unknown job"})
            else:
                self._json(200, job.view())
        elif self.path == "/favicon.ico":
            self.send_response(204)
            self.send_header("Content-Length", "0")
            self.end_headers()
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if not self._host_ok():
            return
        # A JSON content type forces a CORS preflight, which this server never grants.
        if not (self.headers.get("Content-Type") or "").startswith("application/json"):
            self._json(415, {"error": "Content-Type must be application/json"})
            return
        length = int(self.headers.get("Content-Length") or 0)
        if length < 0 or length > MAX_REQUEST_BYTES:
            self._json(413, {"error": "request body too large"})
            return
        raw = self.rfile.read(length) if length else b"{}"
        if self.path.startswith("/job/") and self.path.endswith("/cancel"):
            job = self.engine.job(self.path.removeprefix("/job/").removesuffix("/cancel"))
            if job is None:
                self._json(404, {"error": "unknown job"})
                return
            job.cancel.set()
            self._json(200, {"id": job.id, "cancel_requested": True})
            return
        if self.path != "/chat":
            self._json(404, {"error": "not found"})
            return
        try:
            request = json.loads(raw.decode("utf-8"))
            if not isinstance(request, dict):
                raise ValueError("body must be a JSON object")
            messages = validate_messages(request.get("messages"))
            budget = int(request.get("max_new_tokens", DEFAULT_NEW_TOKENS))
            if not 1 <= budget <= MAX_NEW_TOKENS:
                raise ValueError(f"max_new_tokens must be 1..{MAX_NEW_TOKENS}")
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as error:
            self._json(400, {"error": str(error)})
            return
        try:
            job = self.engine.submit(messages, budget)
        except BlockingIOError as error:
            self._json(409, {"error": str(error)})
            return
        except RuntimeError as error:
            self._json(503, {"error": str(error)})
            return
        self._json(202, job.view())


class ChatServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], engine: ChatEngine) -> None:
        super().__init__(address, _Handler)
        self.engine = engine


def create_server(engine: ChatEngine, *, host: str = "127.0.0.1", port: int = 8791) -> ChatServer:
    if host not in ("127.0.0.1", "localhost"):
        raise ValueError("the chat page may bind only to 127.0.0.1 or localhost")
    return ChatServer((host, port), engine)


def _refuse(*_args: object, **_kwargs: object) -> None:
    raise ConnectionRefusedError("chat_web: outbound network disabled")


def disable_outbound_network() -> None:
    socket.socket.connect = _refuse  # type: ignore[method-assign]
    socket.socket.connect_ex = _refuse  # type: ignore[method-assign,assignment]
    socket.create_connection = _refuse  # type: ignore[assignment]


def main() -> None:
    from src.toolbox.answer import default_chat_reader

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", type=Path, default=default_chat_reader())
    parser.add_argument("--tokenizer", type=Path, default=Path("artifacts/gpu/tokenizer.json"))
    parser.add_argument("--port", type=int, default=8791)
    parser.add_argument("--weights", choices=WEIGHTS, default="f32")
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument("--device", choices=DEVICES, default="cpu")
    parser.add_argument("--facts", type=Path, default=DEFAULT_FACTS)
    parser.add_argument("--hf-model", type=str, default=None)
    parser.add_argument(
        "--hf-revision", type=str, default="1cfa9a7208912126459214e8b04321603b3df60c"
    )
    parser.add_argument("--lora-adapter", type=Path, default=None)
    args = parser.parse_args()
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(args.threads)
    disable_outbound_network()
    knowledge = FactualKnowledgePack.load(args.facts)
    loader = (
        load_hf_lora(args.hf_model, args.hf_revision, args.lora_adapter, args.device)
        if args.hf_model
        else load_v9(args.run, args.tokenizer, args.weights, args.device)
    )
    engine = ChatEngine(loader, knowledge=knowledge)
    server = create_server(engine, port=args.port)
    engine.start_loading()
    label = f"{args.hf_model} (adapter={args.lora_adapter})" if args.hf_model else str(args.run)
    print(
        f"serving http://127.0.0.1:{args.port}/  model={label} "
        f"(loading {args.weights} on {args.device}, {args.threads} threads, "
        f"facts={knowledge.pack_id})",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        sys.stdout.flush()


if __name__ == "__main__":  # pragma: no cover
    main()
