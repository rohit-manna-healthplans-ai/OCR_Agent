from __future__ import annotations

import re
from typing import Dict, Any, Optional


# -------------------------
# Text cleanup
# -------------------------
_WS_RE = re.compile(r"\s+")
_NON_ASCII_RE = re.compile(r"[^\x00-\x7F]+")

# Expanded OCR confusion corrections for PaddleOCR on printed English
_OCR_FIXES = [
    # Zero vs letter O
    (re.compile(r"\b0([A-Za-z])"), r"O\1"),
    (re.compile(r"\b([A-Za-z])0\b"), r"\1O"),
    # l vs 1 (lowercase L confused with 1)
    (re.compile(r"\b1([a-z]{2,})"), r"l\1"),
    # rn vs m (very common paddle mistake)
    (re.compile(r"\brn\b"), "m"),
    (re.compile(r"rn([aeiou])"), r"m\1"),
    # ii vs u (double lowercase-i vs u)
    (re.compile(r"\bii([a-z])"), r"u\1"),
    # Stray pipe characters from table borders
    (re.compile(r"(?<!\s)\|(?!\s)"), " "),
    # Auth/Referral misspelling (f->t OCR confusion)
    (re.compile(r'\bAuth/Reterral\b'), 'Auth/Referral'),
    # Service code line-split recovery: "P-\n99203" or "P- 99203" -> "P-99203"
    (re.compile(r'\b(P|Q|G|H|S|T|V)-\s+(\d{4,5})\b'), r'\1-\2'),
    # Leading O1/O2 in outcome codes -> 01/02 (uppercase O misread as zero context)
    (re.compile(r'\bO(\d)\s+(DISCHARGED|EXPIRED|TRANSFERRED|HOSPICE|LEFT)\b'),
     r'0\1 \2'),
    # Diagnosis code space insertion: "J0 6.9" -> "J06.9", "C1 5.9" -> "C15.9"
    (re.compile(r'\b([A-Z])(\d)\s(\d+\.\d+)\b'), r'\1\2\3'),
    # "Totals:" followed by amounts - ensure space after colon
    (re.compile(r'\bTotals:(\S)'), r'Totals: \1'),
    # EDI Claim/Batch ID spacing
    (re.compile(r'\bEDIClaim#'), 'EDI Claim # '),
    (re.compile(r'\bEDIBatchID:'), 'EDI Batch ID: '),
]

_PADDLE_SPACE_FIXES = [
    # Fix missing space before capital in middle of word: "helloWorld" -> "hello World"
    (re.compile(r"([a-z])([A-Z][a-z])"), r"\1 \2"),
]


def clean_text(text: str) -> str:
    """Lightweight cleanup without being too destructive."""
    if not text:
        return ""
    t = text.replace("\u00a0", " ")  # NBSP
    t = _NON_ASCII_RE.sub(" ", t)
    t = _WS_RE.sub(" ", t)
    return t.strip()


def fix_common_paddle_errors(text: str) -> str:
    """Apply PaddleOCR-specific post-correction heuristics."""
    if not text:
        return ""
    t = text
    for pat, repl in _OCR_FIXES:
        t = pat.sub(repl, t)
    for pat, repl in _PADDLE_SPACE_FIXES:
        t = pat.sub(repl, t)
    return t


def normalize_linebreaks(text: str) -> str:
    """Keep paragraphs but normalize excessive blank lines, then apply paddle fixes."""
    if not text:
        return ""
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    t = "\n".join(line.rstrip() for line in t.split("\n"))
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = fix_common_paddle_errors(t)
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
