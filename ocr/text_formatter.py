from typing import Any, Dict


def build_final_json(result: Dict[str, Any]) -> Dict[str, Any]:
    """Build UI-facing structured JSON.

    Preference order:
      1) Ollama structured result if available
      2) Fallback: minimal structure with full raw text in main
    """
    oll = result.get("ollama") or {}
    if oll.get("available") and isinstance(oll.get("result"), dict):
        structured = oll.get("result") or {}
        # If Ollama safe guard failed, cleaned_text_full may be raw_text_full (still ok)
        return {
            "raw_text_full": structured.get("raw_text_full") or (result.get("text") or ""),
            "cleaned_text_full": structured.get("cleaned_text_full") or (result.get("text") or ""),
            "active_tab": structured.get("active_tab"),
            "ui_elements": structured.get("ui_elements") or [],
            "tables": structured.get("tables") or [],
            "pages": structured.get("pages") or [],
        }

    # Fallback (no LLM)
    return {
        "raw_text_full": result.get("text") or "",
        "cleaned_text_full": result.get("text") or "",
        "active_tab": None,
        "ui_elements": [],
        "tables": [],
        "pages": [
            {
                "page_index": pg.get("page_index"),
                "header": {"raw": "", "cleaned": "", "line_idxs": []},
                "main": {
                    "raw": pg.get("text") or "",
                    "cleaned": pg.get("text") or "",
                    "blocks": [{"type": "text", "order": 1, "raw": pg.get("text") or "", "cleaned": pg.get("text") or "", "line_idxs": [], "bbox": None}],
                },
                "footer": {"raw": "", "cleaned": "", "line_idxs": []},
                "noise": {"raw": "", "line_idxs": []},
            }
            for pg in (result.get("pages") or [])
        ],
    }
