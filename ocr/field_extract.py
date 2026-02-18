
"""
Enterprise Key-Value Extraction (stable, UI-safe)

Goals:
- Stop junk pairs from URL bar / taskbar
- Prefer form-like key/value from main content
- Keep backward compatible return shape

Important:
- Screen OCR captures *everything* in regions; this extractor is only for "fields" (business KV).
"""

from typing import Dict, Any, List
import re

# Broad but safe keywords (case-insensitive)
KEYWORDS = [
    # Generic
    "name", "address", "email", "phone", "mobile", "tel", "fax",
    "date", "dob", "gender", "age", "id", "member id", "policy", "claim", "status", "type",
    "provider", "doctor", "patient", "received", "assigned", "code", "description",
    # Billing/Amounts
    "invoice", "bill", "amount", "total", "subtotal", "tax", "gst", "vat", "due", "paid",
    # Banking/IDs
    "account", "ifsc", "swift", "utr", "pan", "aadhaar", "passport", "license",
]

_URL_RE = re.compile(r"(https?://|www\.)|(\.[a-z]{2,}/)", re.I)
_TIME_RE = re.compile(r"\b\d{1,2}:\d{2}\s?(am|pm)\b", re.I)
_TASKBAR_RE = re.compile(r"\b(qsearch|eng|mostly[- ]clear|wifi|battery)\b", re.I)


def _clean(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "").strip())


def _avg_conf(words: List[dict]) -> float:
    if not words:
        return 0.0
    return sum(float(w.get("conf", 0)) for w in words) / len(words)


def _contains_keyword(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in KEYWORDS)


def _digit_ratio(text: str) -> float:
    t = text or ""
    if not t:
        return 0.0
    return sum(c.isdigit() for c in t) / max(1, len(t))


def _is_ui_noise(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    if _URL_RE.search(t):
        return True
    if _TIME_RE.search(t):
        return True
    if _TASKBAR_RE.search(t):
        return True
    return False


def extract_fields_from_page(page: Dict[str, Any], image_rgb=None, recognize_char_fn=None) -> Dict[str, Any]:
    """
    Returns:
      {"pairs": [...], "total_pairs": n, "engine": "enterprise_kv_v1"}
    """
    lines = page.get("lines", [])
    pairs = []

    # PASS 1: Form-ish "Label: Value" only if label looks like a field (keyword or short alpha label)
    for line in lines:
        words = line.get("words", [])
        lt = _clean(" ".join((w.get("text") or "") for w in words))
        if not lt or _is_ui_noise(lt):
            continue

        if ":" in lt:
            k, v = lt.split(":", 1)
            key = _clean(k)
            val = _clean(v)
            if len(key) < 2 or len(val) < 1:
                continue
            if _is_ui_noise(key) or _is_ui_noise(val):
                continue
            if not (_contains_keyword(key) or (len(key) <= 22 and _digit_ratio(key) < 0.2)):
                continue

            conf = _avg_conf(words)
            pairs.append({"key": key, "value": val, "confidence": round(min(1.0, conf), 3), "method": "inline_colon"})

    # PASS 2: Right-aligned (only when left part has keyword-ish label)
    for line in lines:
        words = line.get("words", [])
        if len(words) < 3:
            continue
        lt = _clean(" ".join((w.get("text") or "") for w in words))
        if not lt or _is_ui_noise(lt):
            continue

        ws = sorted(words, key=lambda w: w["bbox"][0])
        mid_x = sum(w["bbox"][0] for w in ws) / len(ws)
        key_words = [w for w in ws if w["bbox"][2] < mid_x]
        val_words = [w for w in ws if w["bbox"][0] >= mid_x]
        if not key_words or not val_words:
            continue

        key = _clean(" ".join((w.get("text") or "") for w in key_words))
        val = _clean(" ".join((w.get("text") or "") for w in val_words))

        if len(key) < 2 or len(val) < 1:
            continue
        if _is_ui_noise(key) or _is_ui_noise(val):
            continue
        if not (_contains_keyword(key) or (len(key) <= 22 and _digit_ratio(key) < 0.2)):
            continue
        if _digit_ratio(key) > 0.5:
            continue

        conf = (_avg_conf(key_words) + _avg_conf(val_words)) / 2.0
        pairs.append({"key": key, "value": val, "confidence": round(min(1.0, conf), 3), "method": "right_aligned"})

    # Dedup
    unique = []
    seen = set()
    for p in pairs:
        sig = (p["key"].lower(), p["value"].lower())
        if sig in seen:
            continue
        seen.add(sig)
        unique.append(p)

    return {"pairs": unique, "total_pairs": len(unique), "engine": "enterprise_kv_v1"}
