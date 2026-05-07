from __future__ import annotations

"""
Layout analysis engine — production-ready, general-purpose.
Works on any document type: forms, tables, dashboards, scans, screenshots.
"""

import re
import statistics
from typing import List, Dict, Any, Optional, Tuple

from ocr.schemas import Word, Line, Block, Column


# ════════════════════════════════════════════════════════
#  PRIMITIVES
# ════════════════════════════════════════════════════════


def _yc(w: Dict) -> float:
    return (w["bbox"][1] + w["bbox"][3]) / 2.0


def _h(w: Dict) -> float:
    return max(1, w["bbox"][3] - w["bbox"][1])


def _merge_bbox(bboxes: List[List[int]]) -> List[int]:
    return [
        min(b[0] for b in bboxes), min(b[1] for b in bboxes),
        max(b[2] for b in bboxes), max(b[3] for b in bboxes),
    ]


def _median_height(words: List[Dict]) -> float:
    heights = sorted(_h(w) for w in words)
    return heights[len(heights) // 2] if heights else 12.0


def _line_avg_conf(line: List[Dict]) -> float:
    if not line:
        return 0.0
    return sum(float(w.get("conf", 0)) for w in line) / len(line)


def _line_text(line: List[Dict]) -> str:
    return " ".join((w.get("text") or "").strip() for w in line).strip()


# ════════════════════════════════════════════════════════
#  STEP 1 — GROUP WORDS INTO LINES
# ════════════════════════════════════════════════════════


def group_into_lines(
    words: List[Dict],
    page_width: int,
    page_height: int,
) -> List[List[Dict]]:
    """
    Group word dicts into horizontal lines.
    - Adaptive y-threshold (60% of median char height)
    - Running average y-center per line
    - Splits wide lines at largest gap (separates multi-column content)
    """
    if not words:
        return []

    med_h = _median_height(words)
    thr = max(5.0, med_h * 0.6)
    ws = sorted(words, key=lambda w: (_yc(w), w["bbox"][0]))
    lines: List[List[Dict]] = []
    current: List[Dict] = []
    current_y: Optional[float] = None

    for w in ws:
        yc = _yc(w)
        if current_y is None:
            current, current_y = [w], yc
            continue
        if abs(yc - current_y) <= thr:
            current.append(w)
            n = len(current)
            current_y = current_y * (n - 1) / n + yc / n
        else:
            lines.append(sorted(current, key=lambda x: x["bbox"][0]))
            current, current_y = [w], yc
    if current:
        lines.append(sorted(current, key=lambda x: x["bbox"][0]))

    col_gap = page_width * 0.06
    result: List[List[Dict]] = []
    for line in lines:
        if len(line) < 2:
            result.append(line)
            continue
        line_span = line[-1]["bbox"][2] - line[0]["bbox"][0]
        if line_span < page_width * 0.55:
            result.append(line)
            continue
        gaps = [(line[i]["bbox"][0] - line[i - 1]["bbox"][2], i)
                for i in range(1, len(line))]
        max_gap, split_at = max(gaps, key=lambda g: g[0])
        if max_gap >= col_gap:
            left, right = line[:split_at], line[split_at:]
            if left:
                result.append(left)
            if right:
                result.append(right)
        else:
            result.append(line)
    return result


# ════════════════════════════════════════════════════════
#  STEP 2 — PAGE-LEVEL COLUMN DETECTION
# ════════════════════════════════════════════════════════


def detect_columns(
    lines: List[List[Dict]],
    page_width: int,
) -> List[Tuple[int, int]]:
    """
    Detect page-level layout columns (newspaper-style).
    NOT for data table columns — those are handled in reconstruct_tables().
    Returns [(0, page_width)] for single-column or full-width table pages.
    """
    if not lines or page_width <= 0:
        return [(0, page_width)]

    spans = [line[-1]["bbox"][2] - line[0]["bbox"][0] for line in lines if line]
    if spans and statistics.median(spans) > page_width * 0.65:
        return [(0, page_width)]

    profile = [0] * max(page_width, 1)
    for line in lines:
        for w in line:
            for x in range(max(0, w["bbox"][0]), min(page_width, w["bbox"][2])):
                if x < len(profile):
                    profile[x] += 1

    min_gutter = int(page_width * 0.06)
    in_gutter, gutter_start = False, 0
    gutters: List[Tuple[int, int]] = []
    for x, occ in enumerate(profile):
        if occ == 0 and not in_gutter:
            in_gutter, gutter_start = True, x
        elif occ > 0 and in_gutter:
            in_gutter = False
            if x - gutter_start >= min_gutter:
                gutters.append((gutter_start, x))

    if not gutters:
        return [(0, page_width)]

    cols, prev = [], 0
    for g_start, g_end in gutters:
        cols.append((prev, (g_start + g_end) // 2))
        prev = (g_start + g_end) // 2
    cols.append((prev, page_width))

    min_col_w = page_width * 0.12
    merged = []
    for col in cols:
        if merged and (col[1] - col[0]) < min_col_w:
            merged[-1] = (merged[-1][0], col[1])
        else:
            merged.append(col)
    return merged


def assign_lines_to_columns(
    lines: List[List[Dict]],
    col_ranges: List[Tuple[int, int]],
) -> List[List[List[Dict]]]:
    buckets: List[List[List[Dict]]] = [[] for _ in col_ranges]
    for line in lines:
        if not line:
            continue
        mid_x = (line[0]["bbox"][0] + line[-1]["bbox"][2]) / 2.0
        best = 0
        for i, (cs, ce) in enumerate(col_ranges):
            if cs <= mid_x < ce:
                best = i
                break
        buckets[best].append(line)
    return buckets


# ════════════════════════════════════════════════════════
#  STEP 3 — BLOCK CLASSIFICATION
# ════════════════════════════════════════════════════════

_UI_OVERLAYS = frozenset({
    "edit with lovable", "lovable", "edit with",
    "powered by", "beta", "preview",
})


def _is_noise(line: List[Dict]) -> bool:
    """
    Noise = clearly an OCR artifact. Very conservative.
    Real content (titles, KPIs, short labels) is NEVER noise.
    """
    if not line:
        return True
    texts = [(w.get("text") or "").strip() for w in line
             if (w.get("text") or "").strip()]
    if not texts:
        return True
    if _line_avg_conf(line) < 0.30:
        return True
    if all(len(t) == 1 and not t.isalnum() for t in texts):
        return True
    return False


def _is_heading(line: List[Dict], med_h: float) -> bool:
    """
    Detect section headings.
    Path 1: taller than body text (large/bold font)
    Path 2: short title-case line, high confidence, no colon, no digits
    UI overlays are excluded explicitly.
    """
    if not line:
        return False
    text = _line_text(line)
    words = text.split()
    if not words:
        return False

    avg_h = sum(_h(w) for w in line) / len(line)
    if avg_h >= med_h * 1.3:
        title_count = sum(1 for w in words if w and (w[0].isupper() or w.isupper()))
        if title_count >= max(1, len(words) * 0.6):
            lt = text.lower().strip()
            if lt not in _UI_OVERLAYS:
                return True

    if (2 <= len(words) <= 6
            and ":" not in text
            and not any(c.isdigit() for c in text)
            and all(w[0].isupper() for w in words if w and w[0].isalpha())
            and _line_avg_conf(line) >= 0.92
            and text.lower().strip() not in _UI_OVERLAYS):
        return True

    return False


def _is_table_row(
    line: List[Dict],
    page_width: int,
    prev_gaps: List[float],
) -> bool:
    if len(line) < 2:
        return False
    gaps = [line[i]["bbox"][0] - line[i - 1]["bbox"][2]
            for i in range(1, len(line))]
    large_gaps = sum(1 for g in gaps if g > page_width * 0.03)

    if large_gaps >= 3 and len(line) >= 6:
        return True
    if large_gaps >= 2:
        return True
    if prev_gaps and len(gaps) == len(prev_gaps):
        similarity = sum(1 for a, b in zip(gaps, prev_gaps)
                         if abs(a - b) < page_width * 0.03)
        if similarity >= len(gaps) - 1:
            return True
    return False


def _is_label_value(text: str) -> bool:
    if not text or len(text) > 200:
        return False
    if text.startswith("http"):
        return False
    if re.match(r"^\s*(totals?|subtotals?|grand\s+total)\b", text, re.IGNORECASE):
        return False
    if ":" not in text:
        return False
    before = text.split(":")[0].strip()
    return any(c.isalpha() for c in before)


def _merge_orphan_words(
    typed: List[Tuple[str, List[Dict]]],
    med_h: float,
) -> List[Tuple[str, List[Dict]]]:
    tol = max(8.0, med_h * 0.5)
    result = list(typed)
    for i, (btype, line) in enumerate(result):
        if btype != "paragraph" or not line:
            continue
        line_yc = sum(_yc(w) for w in line) / len(line)
        for j, (other_type, other_line) in enumerate(result):
            if i == j or other_type != "table_row" or not other_line:
                continue
            other_yc = sum(_yc(w) for w in other_line) / len(other_line)
            if abs(line_yc - other_yc) <= tol:
                merged = sorted(other_line + line, key=lambda w: w["bbox"][0])
                result[j] = ("table_row", merged)
                result[i] = ("table_row", [])
                break
    return [(bt, ln) for bt, ln in result if ln]


def _make_block(
    btype: str,
    group: List[List[Dict]],
    reading_order: int,
) -> Optional[Block]:
    all_words = [w for line in group for w in line]
    if not all_words:
        return None
    bbox = _merge_bbox([w["bbox"] for w in all_words])
    schema_lines = []
    for li, raw_line in enumerate(group):
        sl = sorted(raw_line, key=lambda w: w["bbox"][0])
        if not sl:
            continue
        schema_lines.append(Line(
            text=_line_text(sl),
            conf=_line_avg_conf(sl),
            bbox=_merge_bbox([w["bbox"] for w in sl]),
            words=[Word(text=w.get("text", ""),
                        conf=float(w.get("conf", 0.0)),
                        bbox=w["bbox"]) for w in sl],
            line_index=li,
        ))
    if not schema_lines:
        return None
    return Block(block_type=btype, lines=schema_lines,
                 bbox=bbox, reading_order=reading_order)


def classify_blocks(
    col_lines: List[List[Dict]],
    page_width: int,
    med_h: float,
) -> List[Block]:
    """
    KEY RULE: table_row lines are NEVER merged together into one block.
    Each table_row line becomes its own Block with one Line.
    """
    if not col_lines:
        return []

    typed: List[Tuple[str, List[Dict]]] = []
    prev_gaps: List[float] = []

    for line in col_lines:
        if _is_noise(line):
            typed.append(("noise", line))
            prev_gaps = []
            continue
        text = _line_text(line)
        if not text:
            continue
        if _is_table_row(line, page_width, prev_gaps):
            typed.append(("table_row", line))
            prev_gaps = [line[i]["bbox"][0] - line[i - 1]["bbox"][2]
                         for i in range(1, len(line))]
        elif _is_heading(line, med_h):
            typed.append(("heading", line))
            prev_gaps = []
        elif _is_label_value(text):
            typed.append(("label_value", line))
            prev_gaps = []
        else:
            typed.append(("paragraph", line))
            prev_gaps = []

    typed = _merge_orphan_words(typed, med_h)

    blocks: List[Block] = []
    reading_order = 0
    i = 0

    while i < len(typed):
        btype, first_line = typed[i]
        j = i + 1

        if btype == "table_row":
            b = _make_block(btype, [first_line], reading_order)
            if b:
                blocks.append(b)
                reading_order += 1
            i = j
            continue

        if btype in ("heading", "noise"):
            b = _make_block(btype, [first_line], reading_order)
            if b:
                blocks.append(b)
                reading_order += 1
            i = j
            continue

        group = [first_line]
        while j < len(typed) and typed[j][0] == btype:
            group.append(typed[j][1])
            j += 1

        b = _make_block(btype, group, reading_order)
        if b:
            blocks.append(b)
            reading_order += 1
        i = j

    return blocks


# ════════════════════════════════════════════════════════
#  STEP 4 — TABLE RECONSTRUCTION
# ════════════════════════════════════════════════════════


def _detect_two_row_header(
    table_row_blocks: List[Block],
) -> Tuple[Block, int]:
    if len(table_row_blocks) < 2:
        return table_row_blocks[0], 1

    r0, r1 = table_row_blocks[0], table_row_blocks[1]
    gap = r1.bbox[1] - r0.bbox[3]
    r0_h = max(1, r0.bbox[3] - r0.bbox[1])

    if gap >= r0_h * 1.2:
        return r0, 1

    r0_xs = set(range(r0.bbox[0], r0.bbox[2]))
    r1_xs = set(range(r1.bbox[0], r1.bbox[2]))
    overlap = len(r0_xs & r1_xs) / max(1, min(len(r0_xs), len(r1_xs)))

    if overlap < 0.4:
        return r0, 1

    r0_words = sorted([w for ln in r0.lines for w in ln.words], key=lambda w: w.bbox[0])
    r1_words = sorted([w for ln in r1.lines for w in ln.words], key=lambda w: w.bbox[0])
    all_hw = sorted(r0_words + r1_words, key=lambda w: w.bbox[0])

    if not all_hw:
        return r0, 1

    merged_line = Line(
        text=" ".join(w.text for w in all_hw),
        conf=min(w.conf for w in all_hw),
        bbox=_merge_bbox([w.bbox for w in all_hw]),
        words=all_hw,
        line_index=0,
    )
    merged_block = Block(
        block_type="table_row",
        lines=[merged_line],
        bbox=_merge_bbox([r0.bbox, r1.bbox]),
        reading_order=r0.reading_order,
    )
    return merged_block, 2


def reconstruct_tables(
    blocks: List[Block],
    page_width: int,
) -> List[Dict[str, Any]]:
    if not blocks:
        return []

    table_row_blocks = sorted(
        [b for b in blocks if b.block_type == "table_row"],
        key=lambda b: b.bbox[1],
    )
    if not table_row_blocks:
        return []

    header_block, data_start = _detect_two_row_header(table_row_blocks)
    data_row_blocks = table_row_blocks[data_start:]

    header_words = sorted(
        [w for ln in header_block.lines for w in ln.words],
        key=lambda w: w.bbox[0],
    )
    col_anchors = [(w.bbox[0] + w.bbox[2]) // 2 for w in header_words]
    n_cols = len(col_anchors)

    if n_cols < 1:
        return []

    def word_to_col(bbox: List[int]) -> int:
        xc = (bbox[0] + bbox[2]) // 2
        return min(range(n_cols), key=lambda i: abs(col_anchors[i] - xc))

    table_y_top = header_block.bbox[1]
    table_y_bot = data_row_blocks[-1].bbox[3] if data_row_blocks else header_block.bbox[3]

    header_cells: List[List[str]] = [[] for _ in range(n_cols)]
    for w in header_words:
        header_cells[word_to_col(w.bbox)].append(w.text)

    row_grid: List[List[List[str]]] = []
    for b in data_row_blocks:
        cells: List[List[str]] = [[] for _ in range(n_cols)]
        for ln in b.lines:
            for w in sorted(ln.words, key=lambda x: x.bbox[0]):
                cells[word_to_col(w.bbox)].append(w.text)
        if any(any(c) for c in cells):
            row_grid.append(cells)

    continuation = [
        b for b in blocks
        if b.block_type in ("paragraph", "noise", "label_value")
        and b.bbox[1] >= table_y_top
        and b.bbox[3] <= table_y_bot + 80
    ]

    for cb in continuation:
        cb_yc = (cb.bbox[1] + cb.bbox[3]) / 2.0
        best_ri, best_dist = None, float("inf")
        for ri, rb in enumerate(data_row_blocks):
            if rb.bbox[3] <= cb_yc + 10:
                dist = abs(cb_yc - rb.bbox[3])
                if dist < best_dist:
                    best_dist, best_ri = dist, ri
        if best_ri is None or best_ri >= len(row_grid):
            continue
        for ln in cb.lines:
            for w in sorted(ln.words, key=lambda x: x.bbox[0]):
                ci = word_to_col(w.bbox)
                xc = (w.bbox[0] + w.bbox[2]) // 2
                if abs(xc - col_anchors[ci]) < page_width * 0.06:
                    row_grid[best_ri][ci].append(w.text)
        cb.block_type = "table"

    def cells_to_str(cells: List[List[str]]) -> List[str]:
        return [" ".join(c).strip() for c in cells]

    header_row = cells_to_str(header_cells)
    data_rows = [cells_to_str(r) for r in row_grid]

    def md_row(cells: List[str]) -> str:
        return "| " + " | ".join(c or " " for c in cells) + " |"

    separator = "| " + " | ".join("---" for _ in header_row) + " |"
    md_lines = [md_row(header_row), separator] + [md_row(r) for r in data_rows]

    for b in table_row_blocks:
        b.block_type = "table"

    return [{
        "bbox": _merge_bbox([b.bbox for b in table_row_blocks]),
        "headers": header_row,
        "rows": data_rows,
        "text": "\n".join(md_lines),
    }]


# ════════════════════════════════════════════════════════
#  STEP 5 — ZONE DETECTION
# ════════════════════════════════════════════════════════


def detect_zones(
    lines: List[List[Dict]],
    page_width: int,
    page_height: int,
) -> Dict[str, List[List[Dict]]]:
    zones: Dict[str, List[List[Dict]]] = {
        "HEADER": [], "FOOTER": [], "SIDEBAR": [], "BODY": []
    }
    if not lines:
        return zones

    line_ycs = sorted(
        statistics.median([_yc(w) for w in line])
        for line in lines if line
    )

    header_boundary = page_height * 0.10
    footer_boundary = page_height * 0.93

    if len(line_ycs) >= 2:
        spacings = [line_ycs[i + 1] - line_ycs[i] for i in range(len(line_ycs) - 1)]
        med_sp = statistics.median(spacings) if spacings else 20

        for i, sp in enumerate(spacings):
            if line_ycs[i] > page_height * 0.25:
                break
            if sp > med_sp * 2.5:
                header_boundary = min(
                    line_ycs[i] + sp * 0.3,
                    page_height * 0.13,
                )
                break

        for i in range(len(spacings) - 1, -1, -1):
            if line_ycs[i] < page_height * 0.75:
                break
            if spacings[i] > med_sp * 2.5:
                footer_boundary = max(
                    line_ycs[i] + spacings[i] * 0.7,
                    page_height * 0.90,
                )
                break

    sidebar_boundary = -1
    if page_width > 0:
        profile = [0] * page_width
        for line in lines:
            for w in line:
                for x in range(max(0, w["bbox"][0]), min(page_width, w["bbox"][2])):
                    if x < len(profile):
                        profile[x] += 1
        zero_run, best_run, best_end = 0, 0, -1
        for x in range(int(page_width * 0.10), int(page_width * 0.28)):
            if x < len(profile) and profile[x] == 0:
                zero_run += 1
                if zero_run > best_run:
                    best_run, best_end = zero_run, x
            else:
                zero_run = 0
        if best_run > page_width * 0.025:
            sidebar_boundary = best_end - best_run // 2

    for line in lines:
        if not line:
            continue
        yc = statistics.median(_yc(w) for w in line)
        xc = (line[0]["bbox"][0] + line[-1]["bbox"][2]) / 2.0
        if yc < header_boundary:
            zones["HEADER"].append(line)
        elif yc > footer_boundary:
            zones["FOOTER"].append(line)
        elif sidebar_boundary > 0 and xc < sidebar_boundary:
            zones["SIDEBAR"].append(line)
        else:
            zones["BODY"].append(line)

    return zones


# ════════════════════════════════════════════════════════
#  STEP 6 — READING ORDER
# ════════════════════════════════════════════════════════


def reading_order_sort(blocks: List[Block]) -> List[Block]:
    if not blocks:
        return []
    sorted_b = sorted(blocks, key=lambda b: (b.bbox[1], b.bbox[0]))
    bands: List[List[Block]] = []
    for b in sorted_b:
        placed = False
        for band in bands:
            band_y1 = min(bb.bbox[1] for bb in band)
            band_y2 = max(bb.bbox[3] for bb in band)
            overlap = min(b.bbox[3], band_y2) - max(b.bbox[1], band_y1)
            height = max(1, b.bbox[3] - b.bbox[1])
            if overlap / height > 0.4:
                band.append(b)
                placed = True
                break
        if not placed:
            bands.append([b])
    ordered: List[Block] = []
    for band in bands:
        ordered.extend(sorted(band, key=lambda b: b.bbox[0]))
    for i, b in enumerate(ordered):
        b.reading_order = i
    return ordered


def analyse_layout(
    words: List[Dict],
    page_width: int,
    page_height: int,
) -> Tuple[List[Block], List[Column], List[Dict[str, Any]]]:
    if not words or page_width <= 0 or page_height <= 0:
        return [], [], []

    med_h = _median_height(words)
    raw_lines = group_into_lines(words, page_width, page_height)
    zones = detect_zones(raw_lines, page_width, page_height)
    body_lines = zones["BODY"]
    col_ranges = detect_columns(body_lines, page_width)
    col_buckets = assign_lines_to_columns(body_lines, col_ranges)

    columns: List[Column] = []
    all_body_blocks: List[Block] = []

    for ci, (col_lines, col_range) in enumerate(zip(col_buckets, col_ranges)):
        col_blocks = classify_blocks(col_lines, page_width, med_h)
        columns.append(Column(col_index=ci, x_range=list(col_range), blocks=col_blocks))
        all_body_blocks.extend(col_blocks)

    zone_blocks: List[Block] = []
    for zone_name in ("HEADER", "SIDEBAR", "FOOTER"):
        zlines = zones[zone_name]
        if not zlines:
            continue
        zblocks = classify_blocks(zlines, page_width, med_h)
        for b in zblocks:
            b.block_type = zone_name.lower()
        zone_blocks.extend(zblocks)

    tables = reconstruct_tables(all_body_blocks, page_width)

    for col in columns:
        col.blocks = [b for b in col.blocks if b.block_type != "table"]

    all_blocks = reading_order_sort(zone_blocks + all_body_blocks)
    return all_blocks, columns, tables
