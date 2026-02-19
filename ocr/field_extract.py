from __future__ import annotations

from typing import Any, Dict, Callable, Optional, List, Tuple

import numpy as np

from ocr.boxed_fields import detect_char_boxes, group_boxes_into_rows, boxes_to_sequences, Box


def _crop(img_rgb: np.ndarray, box: Box) -> np.ndarray:
    H, W = img_rgb.shape[:2]
    b = box.pad(4, W, H)
    return img_rgb[b.y1:b.y2, b.x1:b.x2].copy()


def extract_boxed_block_text(
    img_rgb: np.ndarray,
    recognize_char_fn: Callable[[np.ndarray], Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Detects box sequences and reads each box as a single character.
    Returns a list of sequences with text + bbox.
    """
    boxes = detect_char_boxes(img_rgb)
    rows = group_boxes_into_rows(boxes)
    seqs = boxes_to_sequences(rows)

    out: List[Dict[str, Any]] = []
    for seq in seqs:
        chars = []
        confs = []
        x1 = min(b.x1 for b in seq)
        y1 = min(b.y1 for b in seq)
        x2 = max(b.x2 for b in seq)
        y2 = max(b.y2 for b in seq)
        for b in seq:
            crop = _crop(img_rgb, b)
            r = recognize_char_fn(crop) or {}
            ch = (r.get("text") or "").strip()
            if ch:
                chars.append(ch[0])
            else:
                chars.append("")  # keep position
            c = r.get("conf")
            if isinstance(c, (int, float)):
                confs.append(float(c))
        text = "".join(chars).strip()
        avg_conf = sum(confs) / max(1, len(confs)) if confs else 0.0
        if len(seq) >= 4:  # ignore tiny noise sequences
            out.append({"text": text, "conf": avg_conf, "bbox": [x1, y1, x2, y2], "n_boxes": len(seq)})
    return out


def extract_fields_from_page(
    page: Dict[str, Any],
    page_img_rgb: np.ndarray,
    recognizer_fn: Optional[Callable[[np.ndarray], str]] = None,
    recognize_char_fn: Optional[Callable[[np.ndarray], Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Your project already does key/value extraction from OCR lines.
    This function adds: boxed/block-letter extraction as extra signal.
    We DO NOT overwrite existing fields; we attach boxed sequences under "__boxed_sequences".
    """
    fields: Dict[str, Any] = {}

    # Existing heuristic key/value extraction using OCR lines (light)
    # We keep it minimal to avoid breaking your current behavior.
    lines = page.get("lines", []) or []
    for ln in lines:
        txt = (ln.get("text") or "").strip()
        if not txt:
            continue
        low = txt.lower()
        # Simple examples; your repo may already have better rules.
        if "policy" in low and "no" in low and "policy" not in fields:
            fields["Policy No"] = txt.split(":")[-1].strip() if ":" in txt else txt

        if ("name" in low) and ("full" not in low) and ("insured" in low or low.startswith("name")) and "Name" not in fields:
            fields["Name"] = txt.split(":")[-1].strip() if ":" in txt else txt

        if "address" in low and "Address" not in fields:
            fields["Address"] = txt.split(":")[-1].strip() if ":" in txt else txt

        if "pincode" in low and "Pincode" not in fields:
            fields["Pincode"] = txt.split(":")[-1].strip() if ":" in txt else txt

    # Add boxed sequences (block letters)
    if recognize_char_fn is not None:
        boxed = extract_boxed_block_text(page_img_rgb, recognize_char_fn=recognize_char_fn)
        fields["__boxed_sequences"] = boxed

    return fields
