from __future__ import annotations

import os
from typing import Any, Dict, Optional

import requests


def _is_enabled(flag: bool) -> bool:
    return bool(flag) and bool(os.environ.get("AZURE_OPENAI_ENDPOINT") or os.environ.get("OPENAI_API_KEY"))


def _azure_chat_completion(prompt: str) -> str:
    """
    Azure OpenAI Chat Completions (REST). You must set:
    - AZURE_OPENAI_ENDPOINT (https://<resource>.openai.azure.com)
    - AZURE_OPENAI_API_KEY
    - AZURE_OPENAI_DEPLOYMENT (deployment name)
    - AZURE_OPENAI_API_VERSION (optional, default 2024-06-01)
    """
    endpoint = os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/")
    api_key = os.environ["AZURE_OPENAI_API_KEY"]
    deployment = os.environ["AZURE_OPENAI_DEPLOYMENT"]
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-06-01")

    url = f"{endpoint}/openai/deployments/{deployment}/chat/completions?api-version={api_version}"
    headers = {"api-key": api_key, "content-type": "application/json"}
    payload = {
        "messages": [
            {"role": "system", "content": "You are an OCR cleanup engine. Preserve meaning. Do not invent content."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
    }
    r = requests.post(url, headers=headers, json=payload, timeout=45)
    r.raise_for_status()
    data = r.json()
    return (data["choices"][0]["message"]["content"] or "").strip()


def maybe_llm_correct(text: str, enabled: bool = False) -> Dict[str, Any]:
    """
    If LLM env vars exist and enabled=True, returns corrected text.
    Otherwise returns original text with enabled=False.
    """
    if not _is_enabled(enabled):
        return {"enabled": False, "text": text, "provider": None}

    # Prefer Azure if configured
    if os.environ.get("AZURE_OPENAI_ENDPOINT") and os.environ.get("AZURE_OPENAI_API_KEY") and os.environ.get("AZURE_OPENAI_DEPLOYMENT"):
        prompt = (
            "Clean OCR text.\n"
            "- Fix obvious character mistakes (O/0, I/1) when confident.\n"
            "- Preserve line breaks as much as possible.\n"
            "- Do not add new information.\n\n"
            f"OCR TEXT:\n{text}\n"
        )
        try:
            out = _azure_chat_completion(prompt)
            return {"enabled": True, "text": out, "provider": "azure_openai"}
        except Exception as e:
            return {"enabled": False, "text": text, "provider": "azure_openai", "error": str(e)}

    return {"enabled": False, "text": text, "provider": None}
