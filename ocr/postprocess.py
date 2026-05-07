from __future__ import annotations

import re
from typing import Dict, Any, List, Optional

_MULTI_SP = re.compile(r" {2,}")
_WS_RE = re.compile(r"\s+")
_NON_ASCII_RE = re.compile(r"[^\x00-\x7F]+")


def clean_text(text: str) -> str:
    if not text:
        return ""
    t = text.replace("\u00a0", " ")
    t = _NON_ASCII_RE.sub(" ", t)
    t = _WS_RE.sub(" ", t)
    return t.strip()


# ════════════════════════════════════════════════════════
#  OCR CORRECTION RULES
#  Order matters — specific before general.
#  Philosophy: fix only what we are CERTAIN about.
#  Never apply a rule that could corrupt valid data.
# ════════════════════════════════════════════════════════

_OCR_FIXES: List[tuple] = [

    # ── 1. Auth/Referral (common in insurance/healthcare) ──
    (re.compile(r"\bAuth/Reterral\b", re.I), "Auth/Referral"),
    (re.compile(r"\bAuth/RefStatus:"), "Auth/Ref Status:"),
    (re.compile(r"Auth/Referral\s*#\s*(\d)\s*(\d{2})\b"), r"Auth/Referral #\1(\2):"),

    # ── 2. Mid-word capitalisation (bold font anti-aliasing) ──
    (re.compile(r"\bDiagnOsis\b"), "Diagnosis"),
    (re.compile(r"\bOutCOme\b"), "Outcome"),
    (re.compile(r"\bOutcome:O\b"), "Outcome: 0"),

    # ── 3. Checkbox artifacts (☐ read as O or 0) ──
    (re.compile(r"\bOPCP\s+Encounters\b"), "0  PCP Encounters"),
    (re.compile(r"\bEncounters:O\b"), "Encounters: 0"),
    (re.compile(r"\bEncounters:(\d+)PCP\b"), r"Encounters: \1  PCP"),
    (re.compile(r"\bEncounters:(\d+)\s*PCP\b"), r"Encounters: \1  PCP"),

    # ── 4. Outcome code (O before known outcome words = zero) ──
    (re.compile(r"\bO(\d)\s+(DISCHARGED|EXPIRED|TRANSFERRED|HOSPICE|LEFT|HOME|STILL)\b"),
     r"0\1 \2"),

    # ── 5. Diagnosis code fixes ──
    (re.compile(r"\b([A-Z])(\d)\s+(\d+\.\d+)\b"), r"\1\2\3"),
    (re.compile(r"\b([A-Z])\s+(\d{1,3}\.\d{1,2})\b"), r"\1\2"),

    # ── 6. Service code line-split ──
    (re.compile(r"\b([A-Z])-\s+(\d{4,5})\b"), r"\1-\2"),

    # ── 7. Field name spacing ──
    (re.compile(r"\bMaritalStatus:"), "Marital Status:"),
    (re.compile(r"\bPlace\s+ofService\b"), "Place of Service"),
    (re.compile(r"\bFacility(\d)"), r"Facility: \1"),
    (re.compile(r"\bFacility([A-Z])"), r"Facility: \1"),
    (re.compile(r"\b#Of\s+Units"), "# Of Units"),
    (re.compile(r"\bPayer\s+Resp:(\S)"), r"Payer Resp: \1"),
    (re.compile(r"\bEOBIndicator\s*MPPR"), "EOB Indicator  MPPR"),
    (re.compile(r"\bEOBIndicator\b"), "EOB Indicator"),
    (re.compile(r"\bMPPRIndicator\b"), "MPPR Indicator"),
    (re.compile(r"\bldentity\b"), "Identity"),
    (re.compile(r"\bPOlProof\b"), "POI Proof"),
    (re.compile(r"\bPOl\b"), "POI"),

    # ── 8. EDI field spacing ──
    (re.compile(r"\bEDIBatch\s+ID:"), "EDI Batch ID:"),
    (re.compile(r"\bEDIBatchID:"), "EDI Batch ID:"),
    (re.compile(r"\bEDIClaim#?\s*"), "EDI Claim # "),
    (re.compile(r"\bEDI\s+Claim#\s*"), "EDI Claim # "),

    # ── 9. Gender field ──
    (re.compile(r"\bGender\s*\(?\d\)?\s*:?\s*([MF])\b"), r"Gender (\1)"),
    (re.compile(r"\bGender\s+\(?(\d)\)?\s*([MF])\b"), r"Gender (\1): \2"),
    (re.compile(r"\bGender\s+(\d)\s+([MF])\b"), r"Gender (\1): \2"),

    # ── 10. Member ID space ──
    (re.compile(r"\b([A-Z])\s+(\d{9,12})\b"), r"\1\2"),

    # ── 11. Degree symbol ──
    (re.compile(r"\b(\d{1,3})C\b(?![.\d])"), r"\1°C"),

    # ── 12. Time colon ──
    (re.compile(r"\b(1)(\d{2})\s*(AM|PM)\b"), r"\1:\2 \3"),
    (re.compile(r"\b(\d{1,2}:\d{2})(AM|PM)\b"), r"\1 \2"),

    # ── 13. Totals line spacing ──
    (re.compile(r"\bTotals:(\S)"), r"Totals: \1"),

    # ── 14. Common word merges (safe — only specific known patterns) ──
    (re.compile(r"\bEdit\s*with\s*Lovable\b", re.I), "Edit with Lovable"),

    # ── 15. Missing space after colon in label:VALUE pattern ──
    (re.compile(r"([A-Za-z]):([A-Z]{2})"), r"\1: \2"),

    # ── 16. Stray pipe from form/table borders ──
    (re.compile(r"(?<!\s)\|(?!\s)"), " "),
]

# Applied after _OCR_FIXES — space normalization only
_SPACE_FIXES: List[tuple] = [
    (re.compile(r"(\d)([A-DF-WYZ])(?![%°,.\d])"), r"\1 \2"),
]


def fix_common_paddle_errors(text: str) -> str:
    """Apply all OCR correction rules. Safe to call on any text."""
    if not text:
        return ""
    t = text
    for pat, repl in _OCR_FIXES:
        if callable(repl):
            t = pat.sub(repl, t)
        else:
            t = pat.sub(repl, t)
    for pat, repl in _SPACE_FIXES:
        t = pat.sub(repl, t)
    t = _MULTI_SP.sub(" ", t)
    return t


def normalize_linebreaks(text: str) -> str:
    if not text:
        return ""
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    t = "\n".join(line.rstrip() for line in t.split("\n"))
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = fix_common_paddle_errors(t)
    return t.strip()


# ════════════════════════════════════════════════════════
#  FIELD VALIDATORS
# ════════════════════════════════════════════════════════

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


def _apply_ocr_fixes(value: str) -> str:
    v = (value or "").strip()
    for pat, repl in _OCR_FIXES:
        if callable(repl):
            v = pat.sub(repl, v)
        else:
            v = pat.sub(repl, v)
    return v


def validate_fields(fields: Dict[str, Any]) -> Dict[str, Any]:
    if not fields:
        return fields
    out: Dict[str, Any] = dict(fields)
    for k, v in list(out.items()):
        if isinstance(v, str):
            out[k] = clean_text(fix_common_paddle_errors(v))
    for key in ["Pincode", "Pin", "PIN", "Pin Code", "PIN Code"]:
        if key in out and isinstance(out[key], str):
            m = _first_match(PIN_RE, out[key])
            if m:
                out["Pincode"] = m
    for key in ["Policy No", "Policy Number", "Certificate No",
                "SL No/Certificate No", "Company/TPA ID No", "TPA ID"]:
        if key in out and isinstance(out[key], str):
            v = _apply_ocr_fixes(out[key]).replace(" ", "")
            m = _first_match(ALNUM_ID_RE, v)
            if m:
                out[key] = m
    if "PAN" in out and isinstance(out["PAN"], str):
        m = _first_match(PAN_RE, out["PAN"].upper().replace(" ", ""))
        if m:
            out["PAN"] = m.upper()
    if "IFSC" in out and isinstance(out["IFSC"], str):
        m = _first_match(IFSC_RE, out["IFSC"].upper().replace(" ", ""))
        if m:
            out["IFSC"] = m.upper()
    if "Email" in out and isinstance(out["Email"], str):
        m = _first_match(EMAIL_RE, out["Email"])
        if m:
            out["Email"] = m
    for key in ["Mobile", "Phone", "Contact", "Mobile No", "Phone No"]:
        if key in out and isinstance(out[key], str):
            m = _first_match(PHONE_RE, out[key].replace(" ", ""))
            if m:
                out[key] = re.sub(r"\D", "", m)[-10:]
    return out
