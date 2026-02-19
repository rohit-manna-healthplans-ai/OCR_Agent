from __future__ import annotations

import json
import os
import time
import urllib.request
import urllib.error
from typing import Any, Dict


DEFAULT_OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")


class OllamaError(RuntimeError):
    pass


def ollama_generate_json(
    prompt: str,
    model: str = "phi3",
    base_url: str = DEFAULT_OLLAMA_URL,
    timeout_s: int = 60,
    temperature: float = 0.0,
    top_p: float = 0.9,
    num_predict: int = 900,
) -> Dict[str, Any]:
    """Call Ollama /api/generate and parse STRICT JSON returned by the model."""
    url = f"{base_url.rstrip('/')}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": float(temperature),
            "top_p": float(top_p),
            "num_predict": int(num_predict),
        },
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.URLError as e:
        raise OllamaError(f"Ollama request failed: {e}") from e
    except Exception as e:
        raise OllamaError(f"Ollama request failed: {e}") from e

    try:
        envelope = json.loads(raw)
    except Exception as e:
        raise OllamaError(f"Invalid Ollama response JSON envelope: {e}. Raw={raw[:400]}") from e

    text = (envelope.get("response") or "").strip()
    if not text:
        raise OllamaError(f"Empty Ollama response. Keys={list(envelope.keys())}")

    try:
        return json.loads(text)
    except Exception:
        # salvage first {...}
        s = text.find("{")
        eidx = text.rfind("}")
        if s != -1 and eidx != -1 and eidx > s:
            try:
                return json.loads(text[s : eidx + 1])
            except Exception as e:
                raise OllamaError(f"Model did not return valid JSON: {e}. Text={text[:400]}") from e
        raise OllamaError(f"Model did not return valid JSON. Text={text[:400]}")
