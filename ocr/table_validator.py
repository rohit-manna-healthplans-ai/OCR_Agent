"""
Table validation / filtering heuristics for:
- scanned PDFs / invoices / statements (real tables)
- app/website screenshots (avoid UI/menu/toolbar false positives)

IMPORTANT: This version does NOT crop the screenshot. It instead estimates a
"main content region" and accepts tables primarily inside it.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple, Optional


_UI_KEYWORDS = {
    "dashboard","reports","settings","search","logout","login","profile",
    "claim","claims","queue","detail","home","menu","help","support",
    "notifications","upload","download","batch","viewer"
}

_NUM_RE = re.compile(r"(?:\d{1,4}[/-]\d{1,2}[/-]\d{2,4})|(?:\d[\d,]*\.\d+)|(?:\d+)")
_CURR_RE = re.compile(r"[$€£¥]\s*\d|(?:\bUSD\b|\bINR\b|\bEUR\b|\bGBP\b)", re.IGNORECASE)


def _flatten_cells(cells: List[List[str]]) -> List[str]:
    out: List[str] = []
    for row in cells:
        for c in row:
            if c is None:
                continue
            s = str(c).strip()
            if s:
                out.append(s)
    return out


def _bbox_area(b: List[int]) -> int:
    if not b or len(b) != 4:
        return 0
    x0,y0,x1,y1 = [int(v) for v in b]
    return max(0, x1-x0) * max(0, y1-y0)


def _bbox_iou(a: List[int], b: List[int]) -> float:
    if not a or not b or len(a) != 4 or len(b) != 4:
        return 0.0
    ax0,ay0,ax1,ay1 = [int(v) for v in a]
    bx0,by0,bx1,by1 = [int(v) for v in b]
    ix0, iy0 = max(ax0,bx0), max(ay0,by0)
    ix1, iy1 = min(ax1,bx1), min(ay1,by1)
    iw, ih = max(0, ix1-ix0), max(0, iy1-iy0)
    inter = iw*ih
    if inter <= 0:
        return 0.0
    union = _bbox_area(a) + _bbox_area(b) - inter
    return inter / max(union, 1)


def _bbox_overlap_ratio(inner: List[int], outer: List[int]) -> float:
    """How much of inner lies inside outer: intersection_area / inner_area."""
    if not inner or not outer or len(inner) != 4 or len(outer) != 4:
        return 0.0
    ix0 = max(int(inner[0]), int(outer[0]))
    iy0 = max(int(inner[1]), int(outer[1]))
    ix1 = min(int(inner[2]), int(outer[2]))
    iy1 = min(int(inner[3]), int(outer[3]))
    inter = max(0, ix1-ix0) * max(0, iy1-iy0)
    ia = _bbox_area(inner)
    return inter / max(ia, 1)


def is_likely_screenshot(page_w: int, page_h: int, page_text: str) -> bool:
    """
    Heuristic: desktop screenshots are often wide (16:9-ish) and contain UI terms.
    """
    if page_w <= 0 or page_h <= 0:
        return False
    ar = page_w / max(page_h, 1)
    wide = 1.55 <= ar <= 2.05 and page_w >= 1100
    if not wide:
        return False
    t = (page_text or "").lower()
    hits = sum(1 for k in _UI_KEYWORDS if k in t)
    return hits >= 2


def estimate_main_content_bbox(lines: List[Dict[str, Any]], page_w: int, page_h: int) -> Optional[List[int]]:
    """
    Estimate the main content region using OCR word boxes density.
    Works for screenshots (sidebar + main panel) and normal docs (whole page).

    Idea:
    - Build a histogram of word centers across X
    - Choose the densest contiguous band that spans >= 35% width
    - Expand to include most words in that band, returning its bbox
    """
    if not lines or page_w <= 0 or page_h <= 0:
        return None

    centers = []
    word_bboxes = []
    for ln in lines:
        for w in (ln.get("words") or []):
            bb = w.get("bbox")
            txt = (w.get("text") or "").strip()
            if not bb or len(bb) != 4 or not txt:
                continue
            x0,y0,x1,y1 = [int(v) for v in bb]
            cx = (x0+x1)/2.0
            centers.append(cx)
            word_bboxes.append([x0,y0,x1,y1])

    if len(centers) < 30:
        return None

    # 40 bins across width
    bins = 40
    hist = [0]*bins
    for cx in centers:
        bi = int(min(bins-1, max(0, (cx/page_w)*bins)))
        hist[bi] += 1

    # Find best contiguous segment length >= 35% width (~14 bins)
    min_len = max(10, int(0.35*bins))
    best = (0, 0, -1)  # (score, start, end)
    prefix = [0]
    for v in hist:
        prefix.append(prefix[-1]+v)

    for s in range(0, bins-min_len+1):
        for e in range(s+min_len, bins+1):
            score = prefix[e]-prefix[s]
            if score > best[0]:
                best = (score, s, e)

    _, s, e = best
    if e <= s:
        return None

    x_band0 = int((s/bins)*page_w)
    x_band1 = int((e/bins)*page_w)

    # Filter words whose center is inside band
    in_band = []
    for bb in word_bboxes:
        cx = (bb[0]+bb[2])/2.0
        if x_band0 <= cx <= x_band1:
            in_band.append(bb)

    if len(in_band) < 20:
        return None

    x0 = min(b[0] for b in in_band)
    y0 = min(b[1] for b in in_band)
    x1 = max(b[2] for b in in_band)
    y1 = max(b[3] for b in in_band)

    # Expand slightly
    pad_x = int(0.02*page_w)
    pad_y = int(0.02*page_h)
    x0 = max(0, x0 - pad_x)
    y0 = max(0, y0 - pad_y)
    x1 = min(page_w, x1 + pad_x)
    y1 = min(page_h, y1 + pad_y)

    # If the band basically covers whole width, treat as full page content (docs)
    if (x1-x0) >= 0.88*page_w:
        return [0, 0, page_w, page_h]

    return [x0,y0,x1,y1]


def validate_table(
    table: Dict[str, Any],
    page_w: int,
    page_h: int,
    page_text: str = "",
    main_content_bbox: Optional[List[int]] = None,
) -> Tuple[bool, float, List[str]]:
    """
    Returns: (accepted, score_0_1, reasons)
    Score is a heuristic confidence (higher is better).
    """

    reasons: List[str] = []

    cells = table.get("cells") or []
    n_rows = int(table.get("n_rows") or len(cells) or 0)
    n_cols = int(table.get("n_cols") or (len(cells[0]) if cells else 0))

    if n_rows < 2 or n_cols < 2:
        return False, 0.0, ["too_few_rows_or_cols"]

    bbox = table.get("bbox") or [0, 0, 0, 0]
    x0, y0, x1, y1 = [int(v) for v in bbox]
    w = max(0, x1 - x0)
    h = max(0, y1 - y0)

    # Content signal
    flat = _flatten_cells(cells)
    joined = " ".join(flat)
    tokens = joined.split()
    token_count = len(tokens)

    if token_count == 0:
        return False, 0.0, ["empty_cells"]

    num_hits = len(_NUM_RE.findall(joined))
    curr_hits = 1 if _CURR_RE.search(joined) else 0
    alpha_hits = sum(1 for ch in joined if ch.isalpha())

    numeric_ratio = num_hits / max(token_count, 1)
    alpha_ratio = alpha_hits / max(len(joined), 1)

    # UI keyword pressure inside table (menus)
    low = joined.lower()
    ui_hits = sum(1 for k in _UI_KEYWORDS if k in low)
    if ui_hits >= 2:
        reasons.append(f"ui_keywords:{ui_hits}")

    avg_chars_per_cell = sum(len(s) for s in flat) / max(len(flat), 1)
    if avg_chars_per_cell < 4 and numeric_ratio < 0.10 and curr_hits == 0:
        reasons.append("low_density_cells")

    # Main content region check (NO CROPPING)
    screenshot_mode = is_likely_screenshot(page_w, page_h, page_text)
    if screenshot_mode and main_content_bbox:
        inside = _bbox_overlap_ratio([x0,y0,x1,y1], main_content_bbox)
        # If a "table" lies mostly outside the main content area, it's likely sidebar/menu
        if inside < 0.55:
            reasons.append(f"outside_main_content:{inside:.2f}")

    # Score composition
    score = 0.0
    score += min(0.6, numeric_ratio * 1.8)      # numeric strength
    score += 0.15 if curr_hits else 0.0         # currency boost
    score += min(0.2, avg_chars_per_cell / 30.0)
    score += min(0.15, alpha_ratio * 0.25)

    # Penalties
    if any(r.startswith("ui_keywords") for r in reasons):
        score -= 0.25
    if "low_density_cells" in reasons:
        score -= 0.2
    if any(r.startswith("outside_main_content") for r in reasons):
        score -= 0.35  # strong penalty for screenshot side regions

    score = max(0.0, min(1.0, score))

    # Acceptance thresholds:
    threshold = 0.70 if screenshot_mode else 0.52
    accepted = score >= threshold

    if not accepted:
        reasons.append(f"score_below_threshold:{score:.3f}<{threshold:.2f}")

    return accepted, score, reasons