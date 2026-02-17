from __future__ import annotations

import re
from typing import Dict, Any, Optional


# -------------------------
# Text cleanup
# -------------------------
_WS_RE = re.compile(r"\s+")
_NON_ASCII_RE = re.compile(r"[^\x00-\x7F]+")

# Common OCR confusions (very conservative)
_OCR_FIXES = [
    (re.compile(r"\b0([A-Za-z])"), r"O\1"),     # 0A -> OA (sometimes)
    (re.compile(r"\b([A-Za-z])0\b"), r"\1O"),  # A0 -> AO
]


def clean_text(text: str) -> str:
    """Lightweight cleanup without being too destructive."""
    if not text:
        return ""
    t = text.replace("\u00a0", " ")  # NBSP
    t = _NON_ASCII_RE.sub(" ", t)
    t = _WS_RE.sub(" ", t)
    return t.strip()


def normalize_linebreaks(text: str) -> str:
    """Keep paragraphs but normalize excessive blank lines."""
    if not text:
        return ""
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    t = "\n".join(line.rstrip() for line in t.split("\n"))
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


# -------------------------
# Field validation / normalization
# -------------------------
PIN_RE = re.compile(r"\b\d{6}\b")                           # India PIN
ALNUM_ID_RE = re.compile(r"\b[A-Za-z0-9][A-Za-z0-9\-\/]{5,}\b")  # IDs like ABC-123/45
PAN_RE = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b", re.IGNORECASE)     # PAN
IFSC_RE = re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b", re.IGNORECASE)   # IFSC
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(r"(?:\+91[\s-]?)?(?:0[\s-]?)?\b\d{10}\b")


def _first_match(pattern: re.Pattern, value: str) -> Optional[str]:
    if not value:
        return None
    m = pattern.search(value)
    return m.group(0) if m else None


def _apply_ocr_fixes(value: str) -> str:
    v = (value or "").strip()
    for pat, repl in _OCR_FIXES:
        v = pat.sub(repl, v)
    return v


def validate_fields(fields: Dict[str, Any]) -> Dict[str, Any]:
    """Best-effort validators. Never delete fields; only normalize when confident."""
    if not fields:
        return fields

    out: Dict[str, Any] = dict(fields)

    # Generic normalization
    for k, v in list(out.items()):
        if isinstance(v, str):
            out[k] = clean_text(v)

    # Pincode
    for key in ["Pincode", "Pin", "PIN", "Pin Code", "PIN Code"]:
        if key in out and isinstance(out[key], str):
            m = _first_match(PIN_RE, out[key])
            if m:
                out["Pincode"] = m

    # Policy / IDs (generic)
    for key in ["Policy No", "Policy Number", "Certificate No", "SL No/Certificate No", "Company/TPA ID No", "TPA ID"]:
        if key in out and isinstance(out[key], str):
            v = _apply_ocr_fixes(out[key]).replace(" ", "")
            m = _first_match(ALNUM_ID_RE, v)
            if m:
                out[key] = m

    # PAN
    if "PAN" in out and isinstance(out["PAN"], str):
        m = _first_match(PAN_RE, out["PAN"].upper().replace(" ", ""))
        if m:
            out["PAN"] = m.upper()

    # IFSC
    if "IFSC" in out and isinstance(out["IFSC"], str):
        m = _first_match(IFSC_RE, out["IFSC"].upper().replace(" ", ""))
        if m:
            out["IFSC"] = m.upper()

    # Email
    if "Email" in out and isinstance(out["Email"], str):
        m = _first_match(EMAIL_RE, out["Email"])
        if m:
            out["Email"] = m

    # Phone/Mobile
    for key in ["Mobile", "Phone", "Contact", "Mobile No", "Phone No"]:
        if key in out and isinstance(out[key], str):
            m = _first_match(PHONE_RE, out[key].replace(" ", ""))
            if m:
                out[key] = re.sub(r"\D", "", m)[-10:]

    return out
