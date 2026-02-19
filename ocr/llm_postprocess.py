
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Tuple

from .ollama_client import OllamaError, ollama_generate_json


def _tokenize_loose(s: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9]+", s or "")


def _safety_check(original: str, corrected: str) -> Tuple[bool, Dict[str, Any]]:
    o = original or ""
    c = corrected or ""

    if len(o) > 0 and len(c) < int(0.98 * len(o)):
        return False, {"reason": "text_too_short"}

    o_tokens = _tokenize_loose(o)
    c_tokens = set(_tokenize_loose(c))

    missing = [t for t in o_tokens if t not in c_tokens]
    if len(missing) / max(1, len(o_tokens)) > 0.01:
        return False, {"reason": "token_loss"}

    return True, {"reason": "ok"}


def postprocess_with_ollama(
    result: Dict[str, Any],
    model: str = "phi3",
    base_url: str = "http://127.0.0.1:11434",
    timeout_s: int = 60,
) -> Dict[str, Any]:

    original_text = (result.get("text") or "").strip()
    pages = result.get("pages") or []

    tables = []
    for pg in pages:
        for t in (pg.get("tables") or []):
            tables.append({
                "page_index": pg.get("page_index"),
                "cells": t.get("cells") or [],
                "bbox": t.get("bbox"),
            })

    prompt = {
        "task": "You are an advanced OCR structuring engine.",
        "rules": [
            "DO NOT remove or omit any text from full_text.",
            "corrected_text MUST retain every original word (you may fix spelling/grammar but do not delete content).",
            "Return STRICT JSON only.",
            "Identify header, footer, navigation panels, active tabs.",
            "Preserve tables accurately.",
            "Separate UI elements from main content."
        ],
        "input": {
            "full_text": original_text,
            "tables": tables
        },
        "output_schema": {
            "corrected_text": "string",
            "header": "string",
            "footer": "string",
            "active_tab": "string or null",
            "ui_elements": "list of strings",
            "main_content": "string",
            "tables": "list of structured tables"
        }
    }

    try:
        llm_out = ollama_generate_json(
            prompt=json.dumps(prompt, ensure_ascii=False),
            model=model,
            base_url=base_url,
            timeout_s=timeout_s,
            temperature=0.0,
        )

        ok, safety = _safety_check(original_text, llm_out.get("corrected_text", ""))
        if not ok:
            llm_out["corrected_text"] = original_text
            return {"available": True, "safe_applied": False, "result": llm_out}

        return {"available": True, "safe_applied": True, "result": llm_out}

    except OllamaError as e:
        return {"available": False, "error": str(e)}
