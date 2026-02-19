from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Tuple

from .ollama_client import OllamaError, ollama_generate_json


def _tokenize_loose(s: str) -> List[str]:
    # Loose tokens, used only for coarse safety checks
    return re.findall(r"[A-Za-z0-9]+", s or "")


def _coarse_safety_check(original: str, cleaned: str) -> Tuple[bool, Dict[str, Any]]:
    """Option-B safety check:

    We preserve original OCR text separately ALWAYS.
    This check only prevents catastrophic deletion in cleaned text.
    We do NOT enforce exact token preservation (because grammar/typo fixes change tokens).
    """
    o = (original or "").strip()
    c = (cleaned or "").strip()

    if not o:
        return True, {"reason": "no_original"}

    if not c:
        return False, {"reason": "empty_cleaned"}

    # Prevent huge truncation
    if len(c) < int(0.85 * len(o)):
        return False, {"reason": "cleaned_too_short"}

    # Prevent extreme token loss (very coarse)
    o_tokens = _tokenize_loose(o)
    c_tokens = _tokenize_loose(c)
    if o_tokens and len(c_tokens) < int(0.75 * len(o_tokens)):
        return False, {"reason": "token_count_drop"}

    return True, {"reason": "ok"}


def _pages_for_llm(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    pages = result.get("pages") or []
    out: List[Dict[str, Any]] = []
    for pg in pages:
        lines = pg.get("lines") or []
        out_lines = []
        for i, ln in enumerate(lines):
            out_lines.append(
                {
                    "line_idx": int(ln.get("line_idx") or i),
                    "text": (ln.get("text") or ""),
                    "bbox": ln.get("bbox") or None,
                    "conf": ln.get("conf") if ln.get("conf") is not None else None,
                }
            )

        out.append(
            {
                "page_index": int(pg.get("page_index") or 0),
                "width": int(pg.get("width") or 0),
                "height": int(pg.get("height") or 0),
                "header_footer": pg.get("header_footer") or {},
                "tables": pg.get("tables") or [],
                "lines": out_lines,
            }
        )
    return out


def postprocess_with_ollama(
    result: Dict[str, Any],
    model: str = "phi3",
    base_url: str = "http://127.0.0.1:11434",
    timeout_s: int = 300,
) -> Dict[str, Any]:
    """LLM postprocess for:
      - cleaned_text (grammar/typo improvements)
      - structured JSON in deterministic order per page:
          header -> main (includes tables & ui) -> footer

    IMPORTANT:
      - raw OCR text is never modified (always returned separately).
      - LLM operates on page lines (+bbox) from raw.json to improve accuracy.
    """

    raw_text_full = (result.get("text") or "").strip()
    pages_payload = _pages_for_llm(result)

    prompt = {
        "task": "You convert OCR raw output into a page-accurate structured JSON while preserving content.",
        "hard_rules": [
            "You MUST NOT invent text that is not present in the input lines.",
            "You MUST NOT drop any information: every input line must be assigned to exactly one of: header, main, footer, or noise.",
            "Return STRICT JSON only (no markdown).",
            "Provide BOTH raw and cleaned versions. Raw must match the input text exactly (do not rewrite raw).",            "For cleaned text: fix grammar/spelling/obvious OCR typos but do not change meaning.",
            "Tables: if a table exists in input.tables or can be inferred from column-like lines, output it in tables.",
        ],
        "notes": [
            "Browser chrome / OS taskbar often appear; classify those as noise and exclude from header/footer/main raw strings.",
            "Active tab may be inferred from nav items + page title; if unsure return null.",
        ],
        "input": {
            "raw_text_full": raw_text_full,
            "pages": pages_payload,
        },
        "output_schema": {
            "raw_text_full": "string (exactly as input.raw_text_full)",
            "cleaned_text_full": "string",
            "active_tab": "string or null",
            "pages": [
                {
                    "page_index": "int",
                    "header": {
                        "raw": "string",
                        "cleaned": "string",
                        "line_idxs": "list[int]"
                    },
                    "main": {
                        "raw": "string",
                        "cleaned": "string",
                        "blocks": [
                            {
                                "type": "text|ui|table",
                                "order": "int (reading order)",
                                "raw": "string",
                                "cleaned": "string",
                                "line_idxs": "list[int]",                                "bbox": "[x1,y1,x2,y2] or null",
                                "table": "optional structured table"
                            }
                        ]
                    },
                    "footer": {
                        "raw": "string",
                        "cleaned": "string",
                        "line_idxs": "list[int]"
                    },
                    "noise": {
                        "raw": "string",
                        "line_idxs": "list[int]"
                    }
                }
            ],
            "ui_elements": "list[string] (buttons, tabs, labels)",
            "tables": "list[structured tables]"
        }
    }

    try:
        llm_out = ollama_generate_json(
            prompt=json.dumps(prompt, ensure_ascii=False),
            model=model,
            base_url=base_url,
            timeout_s=timeout_s,
            temperature=0.0,
            num_predict=800,
            retries=2,
        )

        # Ensure raw preserved in output (never trust model fully)
        llm_out["raw_text_full"] = raw_text_full

        cleaned_text = (llm_out.get("cleaned_text_full") or "").strip()
        ok, safety = _coarse_safety_check(raw_text_full, cleaned_text)
        if not ok:
            llm_out["cleaned_text_full"] = raw_text_full
            llm_out["_clean_safety"] = {"applied": False, **safety}
            return {"available": True, "safe_applied": False, "result": llm_out}

        llm_out["_clean_safety"] = {"applied": True, **safety}
        return {"available": True, "safe_applied": True, "result": llm_out}

    except OllamaError as e:
        return {"available": False, "error": str(e)}
