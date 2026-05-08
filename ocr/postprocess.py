from __future__ import annotations

import re
from typing import Dict, Any, List, Optional

_MULTI_SP = re.compile(r" {2,}")
_WS_RE = re.compile(r"\s+")
_NON_ASCII_RE = re.compile(r"[^\x00-\x7F]+")


def clean_text(text: str) -> str:
    """Basic whitespace cleanup only. Never changes content."""
    if not text:
        return ""
    t = text.replace("\u00a0", " ")
    t = _WS_RE.sub(" ", t)
    return t.strip()


def fix_common_paddle_errors(text: str) -> str:
    """
    No corrections applied — raw OCR text is returned as-is.
    Downstream consumers extract and validate values themselves.
    """
    return text if text else ""


def normalize_linebreaks(text: str) -> str:
    if not text:
        return ""
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    t = "\n".join(line.rstrip() for line in t.split("\n"))
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


# ── Field validators (unchanged — used externally if needed) ──────────

PIN_RE = re.compile(r"\b\d{6}\b")
ALNUM_ID_RE = re.compile(r"\b[A-Za-z0-9][A-Za-z0-9\-\/]{5,}\b")
PAN_RE = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b", re.IGNORECASE)
IFSC_RE = re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(r"(?:\+91[\s-]?)?(?:0[\s-]?)?\b\d{10}\b")


def _first_match(pattern: re.Pattern, value: str) -> Optional[str]:
    if not value:
        return None
    m = pattern.search(value)
    return m.group(0) if m else None


def validate_fields(fields: Dict[str, Any]) -> Dict[str, Any]:
    """Field validators — called externally, never modifies OCR text."""
    if not fields:
        return fields
    out: Dict[str, Any] = dict(fields)
    for k, v in list(out.items()):
        if isinstance(v, str):
            out[k] = clean_text(v)
    return out
