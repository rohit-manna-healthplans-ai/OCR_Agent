
from typing import Any, Dict

def build_final_json(result: Dict[str, Any]) -> Dict[str, Any]:
    oll = result.get("ollama") or {}
    if oll.get("available") and oll.get("safe_applied"):
        structured = oll.get("result") or {}
    else:
        structured = {
            "corrected_text": result.get("text"),
            "header": None,
            "footer": None,
            "active_tab": None,
            "ui_elements": [],
            "main_content": result.get("text"),
            "tables": []
        }

    return {
        "header": structured.get("header"),
        "footer": structured.get("footer"),
        "active_tab": structured.get("active_tab"),
        "ui_elements": structured.get("ui_elements"),
        "main_content": structured.get("main_content"),
        "tables": structured.get("tables"),
        "full_text": structured.get("corrected_text")
    }
