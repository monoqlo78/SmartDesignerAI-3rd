"""SmartDesignerAI-3rd (SD-Sol-Class-R28) minimal HTTP client & cryptographic verifier.

Model weights, KL-GRPO training recipes, HippoRAG 2 memory, and Sol-Class cognitive runtime
are proprietary to Smart Designer.
For commercial licensing or authorized weight access, contact: msogabe@sdesigner.tokyo
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Any

EXPECTED_ADAPTER_SHA256 = "cae1902dac8a447179c155791d4d03755125746de392b10dd6e3c1a3b3c0a76b"
EXPECTED_COGNITIVE_RUNTIME_SHA256 = (
    "ebbdec659451fffefd7ffc4f465d776327b59b8c76a3f7476628362a40822a2f"
)


def verify_adapter_signature(adapter_file: Path) -> bool:
    """Verify that an authorized adapter_model.safetensors matches SmartDesignerAI-2nd."""
    digest = hashlib.sha256(adapter_file.read_bytes()).hexdigest()
    return digest == EXPECTED_ADAPTER_SHA256


def verify_cognitive_runtime_signature(runtime_file: Path) -> bool:
    """Verify that an authorized nextgen_cognitive.py matches SmartDesignerAI-2nd."""
    digest = hashlib.sha256(runtime_file.read_bytes()).hexdigest()
    return digest == EXPECTED_COGNITIVE_RUNTIME_SHA256


def query_smartdesigner_endpoint(
    prompt: str,
    *,
    endpoint: str = "http://127.0.0.1:8791",
    max_new_tokens: int = 128,
) -> dict[str, Any]:
    """Send a single-turn user prompt to an authorized SmartDesignerAI-2nd server."""
    payload = json.dumps(
        {
            "messages": [{"role": "user", "content": prompt}],
            "max_new_tokens": max_new_tokens,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{endpoint.rstrip('/')}/chat",
        data=payload,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return dict(json.loads(resp.read().decode("utf-8")))
