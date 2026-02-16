from __future__ import annotations

import re
from typing import Dict, List, Optional, Any, Callable

from ocr.boxed_fields import read_boxed_sequence, read_boxed_grid


def _norm(s: str) -> str:
    s = (s or "").strip().lower()
    s = s.replace("’", "'").replace("“", '"').replace("”", '"')
    s = re.sub(r"[\s:]+", " ", s)
    return s


def _center_y(bbox: List[int]) -> float:
    return (bbox[1] + bbox[3]) / 2.0


def _overlap_y(a: List[int], b: List[int]) -> float:
    ay1, ay2 = a[1], a[3]
    by1, by2 = b[1], b[3]
    inter = max(0, min(ay2, by2) - max(ay1, by1))
    ha = max(1, ay2 - ay1)
    hb = max(1, by2 - by1)
    return inter / float(max(ha, hb))


def _clean_boxed_value(s: str) -> str:
    raw = (s or "").strip()
    if not raw:
        return raw
    parts = [p for p in raw.split() if p]
    if len(parts) >= 6 and sum(1 for p in parts if len(p) == 1) / len(parts) >= 0.7:
        return "".join(parts)
    if re.fullmatch(r"(?:[A-Za-z0-9]\s+){5,}[A-Za-z0-9]", raw):
        return raw.replace(" ", "")
    return raw


def _find_key_lines(lines: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    key_aliases = {
        "policy_no": ["policy no", "policy number", "policy no."],
        "sl_no_certificate_no": ["sl no/certificate no", "sl no certificate no", "certificate no", "sl no"],
        "company_tpa_id_no": ["company/tpa id no", "tpa id no", "company id no", "company/tpa id"],
        "name": ["name"],
        "address": ["address"],
        "city": ["city"],
        "pincode": ["pincode", "pin code", "pin"],
    }
    found: Dict[str, Dict[str, Any]] = {}
    for ln in lines:
        t = _norm(ln.get("text", ""))
        for k, aliases in key_aliases.items():
            if k in found:
                continue
            for a in aliases:
                if a in t:
                    found[k] = ln
                    break
    return found


def _pick_value_right_of_key(
    key_line: Dict[str, Any],
    lines: List[Dict[str, Any]],
    min_y_overlap: float = 0.20,
) -> Optional[Dict[str, Any]]:
    kbox = key_line.get("bbox") or [0, 0, 0, 0]
    kx2 = kbox[2]
    ky = _center_y(kbox)

    candidates = []
    for ln in lines:
        if ln is key_line:
            continue
        box = ln.get("bbox") or [0, 0, 0, 0]
        if box[0] <= kx2 + 5:
            continue
        if abs(_center_y(box) - ky) > max(20, 1.0 * (kbox[3] - kbox[1])):
            continue
        if _overlap_y(kbox, box) < min_y_overlap:
            continue
        conf = float(ln.get("conf") or 0.0)
        candidates.append((box[0], -conf, ln))

    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[0], x[1]))
    return candidates[0][2]


def _roi_right_of_key(img_w: int, img_h: int, key_bbox: List[int], pad_y: int = 22, pad_x: int = 10) -> List[int]:
    x1 = key_bbox[2] + pad_x
    y1 = max(0, key_bbox[1] - pad_y)
    x2 = img_w - 10
    y2 = min(img_h, key_bbox[3] + pad_y)
    return [x1, y1, x2, y2]


def extract_fields_from_page(
    page: Dict[str, Any],
    page_rgb_img,
    recognizer_fn: Callable,
) -> Dict[str, str]:
    """
    Field extraction using:
    1) normal OCR line-nearest-right (printed)
    2) boxed fallback:
       - read_boxed_sequence for single-line IDs
       - read_boxed_grid for multi-row blocks (Address)
    """
    lines = page.get("lines") or []
    if not lines:
        return {}

    H, W = page_rgb_img.shape[:2]
    key_lines = _find_key_lines(lines)
    out: Dict[str, str] = {}

    def get_seq(key: str) -> str:
        kl = key_lines.get(key)
        if not kl:
            return ""
        v = _pick_value_right_of_key(kl, lines)
        if v and (v.get("text") or "").strip():
            txt = _clean_boxed_value(v.get("text", ""))
            if len(txt) >= 2:
                return txt
        roi = _roi_right_of_key(W, H, kl.get("bbox") or [0, 0, 0, 0], pad_y=26)
        return read_boxed_sequence(page_rgb_img, roi, recognizer_fn=recognizer_fn)

    def get_grid(key: str) -> str:
        kl = key_lines.get(key)
        if not kl:
            return ""
        # grid blocks need larger vertical padding
        roi = _roi_right_of_key(W, H, kl.get("bbox") or [0, 0, 0, 0], pad_y=90, pad_x=8)
        return read_boxed_grid(page_rgb_img, roi, recognizer_fn=recognizer_fn)

    # Common boxed fields
    policy = get_seq("policy_no")
    if policy:
        out["Policy No"] = policy

    sl = get_seq("sl_no_certificate_no")
    if sl:
        out["SL No/Certificate No"] = sl

    cid = get_seq("company_tpa_id_no")
    if cid:
        out["Company/TPA ID No"] = cid

    nm = get_seq("name")
    if nm:
        out["Name"] = nm

    ct = get_seq("city")
    if ct:
        out["City"] = ct

    pin = get_seq("pincode")
    if pin:
        out["Pincode"] = pin

    # Address: try OCR first, else grid
    if "address" in key_lines:
        kl = key_lines["address"]
        v = _pick_value_right_of_key(kl, lines)
        addr = _clean_boxed_value(v.get("text", "")) if v else ""
        if len(addr) < 6:
            addr = get_grid("address")
        if addr:
            out["Address"] = addr

    return out
