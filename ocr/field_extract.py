
"""
Dynamic Key-Value Extraction Engine v3 (Production Grade)
----------------------------------------------------------
Features:
- No keyword dependency
- Colon / Dash / Equals support  (Key: Value, Key - Value, Key = Value)
- Right-side aligned detection
- Below-line form detection
- Multi-word value merging
- Regex-based value type detection (date, email, phone, amount, id)
- Confidence scoring (geometry + OCR + regex boost)
- Deduplication
"""

from typing import Dict, Any, List
import re


# -----------------------------
# Regex Patterns (Value Types)
# -----------------------------

REGEX_PATTERNS = {
    "email": re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b"),
    "phone": re.compile(r"(\+?\d[\d\- ]{8,}\d)"),
    "date": re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"),
    "amount": re.compile(r"₹?\s?\d{1,3}(,\d{3})*(\.\d{2})?"),
    "id": re.compile(r"\b[A-Z0-9]{6,}\b")
}


# -----------------------------
# Utilities
# -----------------------------

def _clean(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "").strip())


def _y_center(w):
    return (w["bbox"][1] + w["bbox"][3]) / 2


def _x_left(w):
    return w["bbox"][0]


def _x_right(w):
    return w["bbox"][2]


def _avg_conf(words: List[dict]) -> float:
    if not words:
        return 0.0
    return sum(float(w.get("conf", 0)) for w in words) / len(words)


def _digit_ratio(text: str) -> float:
    if not text:
        return 0.0
    digits = sum(c.isdigit() for c in text)
    return digits / len(text)


def _regex_boost(value: str) -> float:
    for pattern in REGEX_PATTERNS.values():
        if pattern.search(value):
            return 0.15
    return 0.0


# -----------------------------
# Main Extraction
# -----------------------------

def extract_fields_from_page(
    page: Dict[str, Any],
    image_rgb=None,
    recognize_char_fn=None
) -> Dict[str, Any]:

    lines = page.get("lines", [])
    pairs = []

    # ---------------------------------
    # PASS 1: Inline separator detection
    # ---------------------------------

    for line in lines:
        words = line.get("words", [])
        line_text = " ".join(w["text"] for w in words)

        for sep in [":", "-", "="]:
            if sep in line_text:
                parts = line_text.split(sep, 1)
                key = _clean(parts[0])
                value = _clean(parts[1])

                if len(key) < 2 or len(value) < 1:
                    continue

                base_conf = _avg_conf(words)
                boost = _regex_boost(value)

                pairs.append({
                    "key": key,
                    "value": value,
                    "confidence": round(min(1.0, base_conf + boost), 3),
                    "method": "inline_separator"
                })

    # ---------------------------------
    # PASS 2: Right-aligned spatial logic
    # ---------------------------------

    for line in lines:
        words = line.get("words", [])
        if len(words) < 2:
            continue

        words_sorted = sorted(words, key=_x_left)
        mid_x = sum(_x_left(w) for w in words_sorted) / len(words_sorted)

        key_words = [w for w in words_sorted if _x_right(w) < mid_x]
        value_words = [w for w in words_sorted if _x_left(w) >= mid_x]

        if not key_words or not value_words:
            continue

        key_text = _clean(" ".join(w["text"] for w in key_words))
        value_text = _clean(" ".join(w["text"] for w in value_words))

        if len(key_text) < 3 or len(value_text) < 1:
            continue

        # Heuristic: key should have lower digit ratio than value
        if _digit_ratio(key_text) > 0.5:
            continue

        base_conf = (_avg_conf(key_words) + _avg_conf(value_words)) / 2
        boost = _regex_boost(value_text)

        pairs.append({
            "key": key_text,
            "value": value_text,
            "confidence": round(min(1.0, base_conf + boost), 3),
            "method": "right_aligned"
        })

    # ---------------------------------
    # PASS 3: Below-line form detection
    # ---------------------------------

    for i in range(len(lines) - 1):
        curr_words = lines[i].get("words", [])
        next_words = lines[i + 1].get("words", [])

        if not curr_words or not next_words:
            continue

        key_text = _clean(" ".join(w["text"] for w in curr_words))
        value_text = _clean(" ".join(w["text"] for w in next_words))

        if len(key_text) < 3 or len(value_text) < 1:
            continue

        y_gap = abs(_y_center(curr_words[0]) - _y_center(next_words[0]))

        if 10 < y_gap < 60 and _digit_ratio(key_text) < 0.5:
            base_conf = (_avg_conf(curr_words) + _avg_conf(next_words)) / 2
            boost = _regex_boost(value_text)

            pairs.append({
                "key": key_text,
                "value": value_text,
                "confidence": round(min(1.0, base_conf + boost), 3),
                "method": "below_line"
            })

    # ---------------------------------
    # Deduplicate + Normalize
    # ---------------------------------

    unique = []
    seen = set()

    for p in pairs:
        signature = (p["key"].lower(), p["value"].lower())
        if signature not in seen:
            seen.add(signature)
            unique.append(p)

    return {
        "pairs": unique,
        "total_pairs": len(unique),
        "engine": "dynamic_spatial_v3"
    }
