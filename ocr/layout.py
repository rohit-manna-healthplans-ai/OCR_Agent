from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

# Optional dependency (part of paddleocr): PPStructure
try:
    from paddleocr import PPStructure  # type: ignore
except Exception:
    PPStructure = None  # type: ignore


_LAYOUT_ENGINE: Optional[Any] = None


def _get_layout_engine() -> Optional[Any]:
    global _LAYOUT_ENGINE
    if _LAYOUT_ENGINE is not None:
        return _LAYOUT_ENGINE
    if PPStructure is None:
        return None
    # show_log False, lang=en. Use "en" where supported.
    try:
        _LAYOUT_ENGINE = PPStructure(show_log=False, lang="en")
    except TypeError:
        _LAYOUT_ENGINE = PPStructure(show_log=False)
    return _LAYOUT_ENGINE


def analyze_layout(img_rgb: np.ndarray) -> Dict[str, Any]:
    """
    Best-effort layout analysis using PaddleOCR PP-Structure.
    Returns blocks with type, bbox, and (when available) text/html.
    If PPStructure is unavailable, returns empty.
    """
    engine = _get_layout_engine()
    if engine is None:
        return {"available": False, "blocks": []}

    try:
        res = engine(img_rgb)
    except Exception:
        return {"available": False, "blocks": []}

    blocks: List[Dict[str, Any]] = []
    for item in (res or []):
        b = {
            "type": item.get("type"),
            "bbox": item.get("bbox"),
        }
        if "res" in item and isinstance(item["res"], dict):
            # Tables may contain html; text blocks contain text
            if "html" in item["res"]:
                b["html"] = item["res"].get("html")
            if "text" in item["res"]:
                b["text"] = item["res"].get("text")
        blocks.append(b)

    return {"available": True, "blocks": blocks}
