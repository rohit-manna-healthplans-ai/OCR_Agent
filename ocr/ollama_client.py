from __future__ import annotations

import json
import os
import time
import urllib.request
import urllib.error
from typing import Any, Dict, Optional


DEFAULT_OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")


class OllamaError(RuntimeError):
    pass


def _strip_code_fences(s: str) -> str:
    s = (s or "").strip()
    # Handles ```json ... ``` or ``` ... ```
    if s.startswith("```"):
        # remove first fence line
        parts = s.split("\n", 1)
        s = parts[1] if len(parts) > 1 else ""
        # remove trailing fence
        s = s.rsplit("```", 1)[0].strip()
    return s.strip()


def ollama_generate_json(
    prompt: str,
    model: str = "phi3",
    base_url: str = DEFAULT_OLLAMA_URL,
    timeout_s: int = 300,
    temperature: float = 0.0,
    top_p: float = 0.9,
    num_predict: int = 700,
    retries: int = 2,
    retry_backoff_s: float = 1.25,
) -> Dict[str, Any]:
    """Call Ollama /api/generate and parse STRICT JSON returned by the model.

    Reliability improvements:
      - longer default timeout (300s)
      - small retry with backoff
      - strips markdown code fences if model returns them
      - salvage first {...} if needed
    """
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

    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    last_err: Optional[Exception] = None
    for attempt in range(max(1, int(retries) + 1)):
        try:
            with urllib.request.urlopen(req, timeout=float(timeout_s)) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
            last_err = None
            break
        except urllib.error.URLError as e:
            last_err = e
        except Exception as e:
            last_err = e

        if attempt < int(retries):
            time.sleep(retry_backoff_s * (2 ** attempt))

    if last_err is not None:
        raise OllamaError(f"Ollama request failed: {last_err}") from last_err

    try:
        envelope = json.loads(raw)
    except Exception as e:
        raise OllamaError(f"Invalid Ollama response JSON envelope: {e}. Raw={raw[:400]}") from e

    text = (envelope.get("response") or "").strip()
    if not text:
        raise OllamaError(f"Empty Ollama response. Keys={list(envelope.keys())}")

    text = _strip_code_fences(text)

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
