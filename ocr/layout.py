from __future__ import annotations

"""
Layout analysis engine.

Converts a flat list of word bboxes (from PaddleOCR) into a structured
page layout: columns -> blocks -> lines, with block-type classification
and table reconstruction.

Design principles
-----------------
- Works purely from bbox coordinates and confidence scores - no image needed.
- No ML models - fast, CPU-only, deterministic.
- Handles: single-column docs, two-column docs, forms with label/value pairs,
  tables, UI screenshots with sidebars, mixed layouts.
- All thresholds are relative to page dimensions or median character height -
  never hardcoded pixel values.
"""

from typing import List, Dict, Any, Optional, Tuple
import statistics
import re

from ocr.schemas import Word, Line, Block, Column, PageResult


# -- Primitive helpers -----------------------------------------------------

def _yc(w: Dict) -> float:
    return (w["bbox"][1] + w["bbox"][3]) / 2.0


def _xc(w: Dict) -> float:
    return (w["bbox"][0] + w["bbox"][2]) / 2.0


def _h(w: Dict) -> float:
    return max(1, w["bbox"][3] - w["bbox"][1])


def _w(w: Dict) -> float:
    return max(1, w["bbox"][2] - w["bbox"][0])


def _merge_bbox(bboxes: List[List[int]]) -> List[int]:
    return [
        min(b[0] for b in bboxes),
        min(b[1] for b in bboxes),
        max(b[2] for b in bboxes),
        max(b[3] for b in bboxes),
    ]


def _median_height(words: List[Dict]) -> float:
    heights = sorted(_h(w) for w in words)
    return heights[len(heights) // 2] if heights else 12.0


# -- Step 1: Group words into lines ---------------------------------------

def group_into_lines(
    words: List[Dict],
    page_width: int,
    page_height: int,
) -> List[List[Dict]]:
    """
    Group word dicts into horizontal lines using adaptive y-threshold.

    Key improvements over the baseline:
    - Threshold = 60% of median char height (adapts to font size on the page).
    - Running average y-center per line (keeps slightly slanted text together).
    - After grouping, splits any line wider than 70% of page_width at the
      largest internal gap - this separates left-column from right-column
      content that was incorrectly merged.
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
            current = [w]
            current_y = yc
            continue
        if abs(yc - current_y) <= thr:
            current.append(w)
            n = len(current)
            current_y = current_y * (n - 1) / n + yc / n
        else:
            lines.append(sorted(current, key=lambda x: x["bbox"][0]))
            current = [w]
            current_y = yc

    if current:
        lines.append(sorted(current, key=lambda x: x["bbox"][0]))

    # Column-gap splitting
    col_gap = page_width * 0.07   # 7% of page = inter-column whitespace
    result: List[List[Dict]] = []
    for line in lines:
        if len(line) < 2:
            result.append(line)
            continue
        line_span = line[-1]["bbox"][2] - line[0]["bbox"][0]
        if line_span < page_width * 0.55:
            result.append(line)
            continue
        # Find the widest gap
        gaps = [
            (line[i]["bbox"][0] - line[i - 1]["bbox"][2], i)
            for i in range(1, len(line))
        ]
        max_gap, split_at = max(gaps, key=lambda g: g[0])
        if max_gap >= col_gap:
            result.append(line[:split_at])
            result.append(line[split_at:])
        else:
            result.append(line)

    return result


# -- Step 2: Detect column structure --------------------------------------

def detect_columns(
    lines: List[List[Dict]],
    page_width: int,
) -> List[Tuple[int, int]]:
    """
    Detect page-level layout columns (e.g. two-column newspaper layout).
    Returns [(0, page_width)] for single-column pages and full-width tables.

    Note: data table columns are NOT detected here - they are handled by
    reconstruct_tables() using word x-anchors from the header row.
    """
    if not lines or page_width <= 0:
        return [(0, page_width)]

    # If lines span > 85% of page width, this is a full-width table or
    # single-column document - no page-level column split needed.
    line_spans = []
    for line in lines:
        if line:
            span = line[-1]["bbox"][2] - line[0]["bbox"][0]
            line_spans.append(span)
    if line_spans:
        median_span = sorted(line_spans)[len(line_spans) // 2]
        if median_span > page_width * 0.70:
            return [(0, page_width)]

    # Build occupancy profile
    profile = [0] * max(page_width, 1)
    for line in lines:
        for w in line:
            x1 = max(0, w["bbox"][0])
            x2 = min(page_width - 1, w["bbox"][2])
            for x in range(x1, x2 + 1):
                if x < len(profile):
                    profile[x] += 1

    min_gutter = int(page_width * 0.06)
    in_gutter = False
    gutter_start = 0
    gutters: List[Tuple[int, int]] = []

    for x, occ in enumerate(profile):
        if occ == 0 and not in_gutter:
            in_gutter = True
            gutter_start = x
        elif occ > 0 and in_gutter:
            in_gutter = False
            width = x - gutter_start
            if width >= min_gutter:
                gutters.append((gutter_start, x))

    if not gutters:
        return [(0, page_width)]

    cols = []
    prev = 0
    for g_start, g_end in gutters:
        mid = (g_start + g_end) // 2
        cols.append((prev, mid))
        prev = mid
    cols.append((prev, page_width))

    # Merge columns that are too narrow to contain real content
    min_col_width = page_width * 0.12
    merged = []
    for col in cols:
        if merged and (col[1] - col[0]) < min_col_width:
            merged[-1] = (merged[-1][0], col[1])
        else:
            merged.append(col)

    return merged


# -- Step 3: Assign lines to columns --------------------------------------

def assign_lines_to_columns(
    lines: List[List[Dict]],
    col_ranges: List[Tuple[int, int]],
) -> List[List[List[Dict]]]:
    """
    Returns one list of lines per column, in top-to-bottom order.
    A line is assigned to the column that contains its horizontal midpoint.
    """
    buckets: List[List[List[Dict]]] = [[] for _ in col_ranges]
    for line in lines:
        if not line:
            continue
        x1 = line[0]["bbox"][0]
        x2 = line[-1]["bbox"][2]
        mid_x = (x1 + x2) / 2.0
        best = 0
        for i, (cs, ce) in enumerate(col_ranges):
            if cs <= mid_x < ce:
                best = i
                break
        buckets[best].append(line)
    return buckets


# -- Step 4: Block classification -----------------------------------------

def _line_avg_conf(line: List[Dict]) -> float:
    if not line:
        return 0.0
    return sum(float(w.get("conf", 0)) for w in line) / len(line)


def _line_text(line: List[Dict]) -> str:
    return " ".join((w.get("text") or "").strip() for w in line).strip()


def _is_label_value(text: str) -> bool:
    """Detect 'Label: Value' pattern - common in forms and claim dashboards."""
    """
    Detect 'Label: Value' pattern - common in forms and claim dashboards.

    Explicitly excluded:
    - URLs (start with http)
    - Table summary rows (start with Total/Totals/Subtotal/Grand Total)
    - Long lines > 200 chars (these are paragraphs that happen to have a colon)
    - Lines with no alphabetic label before the colon
    """
    if not text or len(text) > 200:
        return False
    if text.startswith("http"):
        return False
    # Exclude table summary rows
    if re.match(r'^\s*(totals?|subtotals?|grand\s+total)\b', text, re.IGNORECASE):
        return False
    # Must have at least one colon
    if ":" not in text:
        return False
    # The part before the first colon must contain letters (a real label)
    before_colon = text.split(":")[0].strip()
    if not any(c.isalpha() for c in before_colon):
        return False
    return True


def _is_table_row(line: List[Dict], page_width: int, prev_gaps: List[float]) -> bool:
    """
    Detect if a line is a table row (header or data).

    Triggers on:
    1. Any line with 3+ large gaps (>3% page width) AND 6+ words
       - catches header rows even with no prev_gaps reference.
    2. Any line with 2+ large gaps - standard data row.
    3. Lines whose gap pattern matches the previous row's pattern.
    """
    if len(line) < 2:
        return False

    gaps = [
        line[i]["bbox"][0] - line[i - 1]["bbox"][2]
        for i in range(1, len(line))
    ]
    large_gaps = sum(1 for g in gaps if g > page_width * 0.03)

    # Rule 1: header row - many words, many gaps, no prev_gaps needed
    if large_gaps >= 3 and len(line) >= 6:
        return True

    # Rule 2: standard data row - at least 2 large gaps
    if large_gaps >= 2:
        return True

    # Rule 3: gap pattern matches previous row (same table)
    if prev_gaps and len(gaps) == len(prev_gaps):
        similarity = sum(
            1 for a, b in zip(gaps, prev_gaps)
            if abs(a - b) < page_width * 0.03
        )
        if similarity >= len(gaps) - 1:
            return True

    return False


def _is_noise(line: List[Dict], med_h: float) -> bool:
    """
    A line is noise ONLY when it is clearly an OCR artifact:
    - Average confidence below 0.30 (very low - not just uncertain)
    - Entirely composed of single non-alphanumeric characters (icon glyphs)
    - Empty after stripping

    Deliberately NOT noise:
    - Short lines with high confidence (page titles, labels, KPI values)
    - Lines with mixed case or known words
    - Any line with confidence >= 0.50
    """
    if not line:
        return True
    texts = [(w.get("text") or "").strip() for w in line if (w.get("text") or "").strip()]
    if not texts:
        return True

    avg_conf = _line_avg_conf(line)

    # Hard floor: very low confidence across all words
    if avg_conf < 0.30:
        return True

    # All tokens are single non-alphanumeric characters (pure icon garbage)
    if all(len(t) == 1 and not t.isalnum() for t in texts):
        return True

    # Everything else is real content - preserve it
    return False


def _merge_orphan_header_words(
    typed: List[Tuple[str, List[Dict]]],
) -> List[Tuple[str, List[Dict]]]:
    """
    If a paragraph line sits at exactly the same y-level as a table_row line
    (within 8px), merge its words into that table_row line.
    This fixes detached left-column header words like "Claim ID" that get
    split off by the column-gap detector.
    """
    result = list(typed)
    for i, (btype, line) in enumerate(result):
        if btype != "paragraph" or not line:
            continue
        line_yc = sum((w["bbox"][1] + w["bbox"][3]) / 2 for w in line) / len(line)
        # Look for an adjacent table_row at the same y
        for j, (other_type, other_line) in enumerate(result):
            if i == j or other_type != "table_row" or not other_line:
                continue
            other_yc = sum(
                (w["bbox"][1] + w["bbox"][3]) / 2 for w in other_line
            ) / len(other_line)
            if abs(line_yc - other_yc) <= 8:
                # Merge this paragraph's words into the table_row, sorted by x
                merged = sorted(other_line + line, key=lambda w: w["bbox"][0])
                result[j] = ("table_row", merged)
                result[i] = ("table_row", [])   # empty - will be skipped
                break
    return [(bt, ln) for bt, ln in result if ln]


def _is_heading(line: List[Dict], med_h: float) -> bool:
    """
    Detect heading lines.

    Two detection paths:
    1. Height-based: line is significantly taller than body text (bold/large font).
    2. Content-based: single short line (2-6 words), all title-case, high confidence
       - catches section headers that are bold but same height as body text.
    """
    if not line:
        return False

    text = _line_text(line)
    words = text.split()

    if not words:
        return False

    # Path 1 - height-based (large font headings)
    avg_line_h = sum(_h(w) for w in line) / len(line)
    if avg_line_h >= med_h * 1.3:
        title_words = sum(1 for w in words if w and (w[0].isupper() or w.isupper()))
        if title_words >= max(1, len(words) * 0.6):
            return True

    # Path 2 - content-based (bold same-size section headers)
    # Conditions: 2-6 words, all title-case or uppercase, high confidence,
    #             no colon (colons = label:value, not headings),
    #             no digits (digits = data, not headings)
    if (2 <= len(words) <= 6
            and ":" not in text
            and not any(c.isdigit() for c in text)
            and all(w[0].isupper() for w in words if w and w[0].isalpha())
            and _line_avg_conf(line) >= 0.92):
        return True

    return False


def classify_blocks(
    col_lines: List[List[Dict]],
    page_width: int,
    med_h: float,
) -> List[Block]:
    """
    Convert a column's lines into typed Blocks.
    Groups consecutive lines of the same type together.
    """
    if not col_lines:
        return []

    typed: List[Tuple[str, List[Dict]]] = []
    prev_gaps: List[float] = []

    for line in col_lines:
        if _is_noise(line, med_h):
            typed.append(("noise", line))
            prev_gaps = []
            continue

        text = _line_text(line)
        if not text:
            continue

        if _is_table_row(line, page_width, prev_gaps):
            typed.append(("table_row", line))
            prev_gaps = [
                line[i]["bbox"][0] - line[i - 1]["bbox"][2]
                for i in range(1, len(line))
            ]
        elif _is_heading(line, med_h):
            typed.append(("heading", line))
            prev_gaps = []
        elif _is_label_value(text):
            typed.append(("label_value", line))
            prev_gaps = []
        else:
            typed.append(("paragraph", line))
            prev_gaps = []

    typed = _merge_orphan_header_words(typed)

    # Merge consecutive same-type lines into Blocks
    blocks: List[Block] = []
    reading_order = 0

    i = 0
    while i < len(typed):
        btype, first_line = typed[i]
        group = [first_line]
        j = i + 1

        # Headings and noise don't merge
        if btype not in ("heading", "noise"):
            while j < len(typed) and typed[j][0] == btype:
                group.append(typed[j][1])
                j += 1

        all_words = [w for line in group for w in line]
        if not all_words:
            i = j
            continue

        bbox = _merge_bbox([w["bbox"] for w in all_words])
        schema_lines = []
        for li, raw_line in enumerate(group):
            raw_line_sorted = sorted(raw_line, key=lambda w: w["bbox"][0])
            schema_lines.append(Line(
                text=_line_text(raw_line_sorted),
                conf=_line_avg_conf(raw_line_sorted),
                bbox=_merge_bbox([w["bbox"] for w in raw_line_sorted]),
                words=[Word(
                    text=w.get("text", ""),
                    conf=float(w.get("conf", 0.0)),
                    bbox=w["bbox"],
                ) for w in raw_line_sorted],
                line_index=li,
            ))

        blocks.append(Block(
            block_type=btype,
            lines=schema_lines,
            bbox=bbox,
            reading_order=reading_order,
        ))
        reading_order += 1
        i = j

    return blocks


# -- Step 5: Table reconstruction -----------------------------------------

def reconstruct_tables(blocks: List[Block], page_width: int) -> List[Dict[str, Any]]:
    """
    Reconstruct tables from table_row blocks.

    Key improvement over the baseline:
    - Does NOT require table_row blocks to be consecutive.
    - Finds all table_row blocks, groups them by shared column structure.
    - Collects paragraph/noise blocks whose words fall inside the table's
      y-range and x-align with table columns - these are wrapped cell values
      and are merged back into the correct row.
    - Returns structured table dicts with headers[], rows[], and markdown text.
    """
    if not blocks:
        return []

    # Collect all table_row blocks sorted by vertical position
    table_row_blocks = sorted(
        [b for b in blocks if b.block_type == "table_row"],
        key=lambda b: b.bbox[1],
    )

    if len(table_row_blocks) < 1:
        return []

    # -- Detect column x-anchors from the header row (first table_row) -----
    # Use word x-centers from the first row as column anchors.
    header_block = table_row_blocks[0]
    header_words = [w for ln in header_block.lines for w in ln.words]
    header_words_sorted = sorted(header_words, key=lambda w: w.bbox[0])

    # Build column anchor list: one per header word, using its x-center
    col_anchors = [(w.bbox[0] + w.bbox[2]) // 2 for w in header_words_sorted]
    n_cols = len(col_anchors)

    if n_cols < 2:
        return []

    def word_to_col(word_bbox: List[int]) -> int:
        """Map a word's x-center to the nearest column index."""
        xc = (word_bbox[0] + word_bbox[2]) // 2
        return min(range(n_cols), key=lambda i: abs(col_anchors[i] - xc))

    # -- Build one cell-grid per data row ----------------------------------
    # Start with the content from each table_row block (skip header row)
    table_y_top = header_block.bbox[1]
    table_y_bot = table_row_blocks[-1].bbox[3]

    # Build rows: each row is a list of n_cols cell-lists
    row_grid: List[List[List[str]]] = []  # row_grid[row][col] = [word_texts]

    # Header row
    header_cells: List[List[str]] = [[] for _ in range(n_cols)]
    for w in header_words_sorted:
        ci = word_to_col(w.bbox)
        header_cells[ci].append(w.text)

    # Data rows from table_row blocks (skip header)
    data_row_blocks = table_row_blocks[1:]
    for b in data_row_blocks:
        cells: List[List[str]] = [[] for _ in range(n_cols)]
        for ln in b.lines:
            for w in sorted(ln.words, key=lambda x: x.bbox[0]):
                ci = word_to_col(w.bbox)
                cells[ci].append(w.text)
        row_grid.append(cells)

    # -- Merge continuation lines from paragraph blocks ---------------------
    # Find paragraph/noise blocks whose y-center falls within the table y-range
    # and whose words x-align with table columns. These are wrapped cell values.
    continuation_blocks = [
        b for b in blocks
        if b.block_type in ("paragraph", "noise")
        and b.bbox[1] >= table_y_top
        and b.bbox[3] <= table_y_bot + 60   # small slack below last row
    ]

    for cb in continuation_blocks:
        # Find which data row this continuation belongs to (nearest row above it)
        cb_y_top = cb.bbox[1]
        best_row_idx = None
        best_dist = float("inf")
        for ri, rb in enumerate(data_row_blocks):
            # Continuation should be below or overlap the row, not above it
            if rb.bbox[1] <= cb_y_top + 10:
                dist = abs(cb_y_top - rb.bbox[1])
                if dist < best_dist:
                    best_dist = dist
                    best_row_idx = ri

        if best_row_idx is None:
            continue

        # Merge each word in the continuation into the correct column cell
        for ln in cb.lines:
            for w in sorted(ln.words, key=lambda x: x.bbox[0]):
                xc = (w.bbox[0] + w.bbox[2]) // 2
                # Only merge if x-center is reasonably close to a column anchor
                nearest_col = word_to_col(w.bbox)
                if abs(xc - col_anchors[nearest_col]) < page_width * 0.05:
                    row_grid[best_row_idx][nearest_col].append(w.text)

        # Mark this block as absorbed so it's not rendered separately
        cb.block_type = "table"

    # -- Build output -------------------------------------------------------
    def cells_to_str(cells: List[List[str]]) -> List[str]:
        return [" ".join(c).strip() for c in cells]

    header_row = cells_to_str(header_cells)
    data_rows = [cells_to_str(r) for r in row_grid]

    def md_row(cells: List[str]) -> str:
        return "| " + " | ".join(c or " " for c in cells) + " |"

    separator = "| " + " | ".join("---" for _ in header_row) + " |"
    md_lines = [md_row(header_row), separator] + [md_row(r) for r in data_rows]

    all_bboxes = [b.bbox for b in table_row_blocks]
    table_bbox = _merge_bbox(all_bboxes)

    # Mark all source table_row blocks as absorbed
    for b in table_row_blocks:
        b.block_type = "table"

    return [{
        "bbox": table_bbox,
        "headers": header_row,
        "rows": data_rows,
        "text": "\n".join(md_lines),
    }]


# -- Step 6: Zone detection (header / footer / sidebar) -------------------

def detect_zones(
    lines: List[List[Dict]],
    page_width: int,
    page_height: int,
) -> Dict[str, List[List[Dict]]]:
    """
    Separate lines into page zones based on position.

    Zone boundaries are adaptive:
    - HEADER: top N% where N is derived from the first large vertical gap
      from the top, capped at 15%.
    - FOOTER: bottom N% derived from the last large vertical gap from the
      bottom, capped at 10%.
    - SIDEBAR: left strip if a vertical text-free gap exists in the
      10-25% x-range.
    - BODY: everything else.

    Falls back to fixed 12% / 8% if no clear gap is found.
    """
    zones: Dict[str, List[List[Dict]]] = {
        "HEADER": [], "FOOTER": [], "SIDEBAR": [], "BODY": []
    }
    if not lines:
        return zones

    all_yc = sorted(_yc(w) for line in lines for w in line)
    _ = all_yc

    # Detect header boundary: first gap > 1.5x median line spacing near the top
    line_ycs = sorted(
        statistics.median([_yc(w) for w in line])
        for line in lines if line
    )
    if len(line_ycs) >= 2:
        spacings = [line_ycs[i + 1] - line_ycs[i] for i in range(len(line_ycs) - 1)]
        med_spacing = statistics.median(spacings) if spacings else 20
        # Find the first large gap in the top 30% of the page
        header_boundary = page_height * 0.12  # fallback
        for i, sp in enumerate(spacings):
            if line_ycs[i] > page_height * 0.30:
                break
            if sp > med_spacing * 2.5:
                header_boundary = min(line_ycs[i + 1] - med_spacing * 0.5,
                                      page_height * 0.13)
                break

        footer_boundary = page_height * 0.92  # fallback
        for i in range(len(spacings) - 1, -1, -1):
            if line_ycs[i] < page_height * 0.70:
                break
            if spacings[i] > med_spacing * 1.8:
                footer_boundary = max(line_ycs[i] + med_spacing * 0.5,
                                      page_height * 0.88)
                break
    else:
        header_boundary = page_height * 0.10
        footer_boundary = page_height * 0.93

    # Detect sidebar: vertical text-free band in the left 10-25% of page
    sidebar_boundary = -1
    if page_width > 0:
        profile = [0] * page_width
        for line in lines:
            for w in line:
                for x in range(max(0, w["bbox"][0]),
                               min(page_width, w["bbox"][2])):
                    profile[x] += 1
        scan_start = int(page_width * 0.10)
        scan_end = int(page_width * 0.28)
        zero_run = 0
        best_run = 0
        best_run_end = -1
        for x in range(scan_start, scan_end):
            if x < len(profile) and profile[x] == 0:
                zero_run += 1
                if zero_run > best_run:
                    best_run = zero_run
                    best_run_end = x
            else:
                zero_run = 0
        if best_run > page_width * 0.025:
            sidebar_boundary = best_run_end - best_run // 2

    # Assign lines to zones
    for line in lines:
        if not line:
            continue
        yc = statistics.median(_yc(w) for w in line)
        xc_line = (line[0]["bbox"][0] + line[-1]["bbox"][2]) / 2.0

        if yc < header_boundary:
            zones["HEADER"].append(line)
        elif yc > footer_boundary:
            zones["FOOTER"].append(line)
        elif sidebar_boundary > 0 and xc_line < sidebar_boundary:
            zones["SIDEBAR"].append(line)
        else:
            zones["BODY"].append(line)

    return zones


# -- Step 7: Reading order -------------------------------------------------

def reading_order_sort(blocks: List[Block]) -> List[Block]:
    """
    Sort blocks in natural reading order:
    - Top to bottom (primary)
    - Left to right within the same horizontal band
    A 'band' is a group of blocks whose y-ranges overlap significantly.
    """
    if not blocks:
        return []

    def y_top(b: Block) -> int:
        return b.bbox[1]

    def y_bot(b: Block) -> int:
        return b.bbox[3]

    def x_left(b: Block) -> int:
        return b.bbox[0]

    blocks_sorted = sorted(blocks, key=lambda b: (y_top(b), x_left(b)))

    # Group into horizontal bands
    bands: List[List[Block]] = []
    for b in blocks_sorted:
        placed = False
        for band in bands:
            # Check if this block overlaps with any block in the band
            band_y1 = min(y_top(bb) for bb in band)
            band_y2 = max(y_bot(bb) for bb in band)
            overlap = min(y_bot(b), band_y2) - max(y_top(b), band_y1)
            height = y_bot(b) - y_top(b)
            if height > 0 and overlap / height > 0.4:
                band.append(b)
                placed = True
                break
        if not placed:
            bands.append([b])

    # Within each band, sort left to right
    ordered: List[Block] = []
    for band in bands:
        ordered.extend(sorted(band, key=x_left))

    # Assign reading_order indices
    for i, b in enumerate(ordered):
        b.reading_order = i

    return ordered


# -- Main entry point ------------------------------------------------------

def analyse_layout(
    words: List[Dict],
    page_width: int,
    page_height: int,
) -> Tuple[List[Block], List[Column], List[Dict[str, Any]]]:
    """
    Full layout analysis pipeline.

    Input:  flat list of word dicts with 'text', 'conf', 'bbox' keys.
    Output: (blocks, columns, tables)
        blocks  - reading-order list of Block objects (all zones merged)
        columns - Column objects (one per detected column)
        tables  - list of reconstructed table dicts

    Called from pipeline.py after PaddleOCR produces words.
    """
    if not words or page_width <= 0 or page_height <= 0:
        return [], [], []

    med_h = _median_height(words)

    # 1. Group words into raw lines
    raw_lines = group_into_lines(words, page_width, page_height)

    # 2. Detect zones (header / footer / sidebar / body)
    zones = detect_zones(raw_lines, page_width, page_height)

    # 3. Detect column structure within BODY
    body_lines = zones["BODY"]
    col_ranges = detect_columns(body_lines, page_width)

    # 4. Assign body lines to columns
    col_buckets = assign_lines_to_columns(body_lines, col_ranges)

    # 5. Classify blocks per column
    columns: List[Column] = []
    all_body_blocks: List[Block] = []
    for ci, (col_lines, col_range) in enumerate(zip(col_buckets, col_ranges)):
        col_blocks = classify_blocks(col_lines, page_width, med_h)
        columns.append(Column(
            col_index=ci,
            x_range=list(col_range),
            blocks=col_blocks,
        ))
        all_body_blocks.extend(col_blocks)

    # 6. Classify header / footer / sidebar zones as single-column blocks
    zone_blocks: List[Block] = []
    for zone_name in ("HEADER", "SIDEBAR", "FOOTER"):
        zone_lines = zones[zone_name]
        if not zone_lines:
            continue
        zblocks = classify_blocks(zone_lines, page_width, med_h)
        for b in zblocks:
            b.block_type = zone_name.lower()
        zone_blocks.extend(zblocks)

    # 7. Reconstruct tables from table_row blocks
    tables = reconstruct_tables(all_body_blocks, page_width)

    # After reconstruction, also mark absorbed continuation blocks in columns
    for col in columns:
        col.blocks = [b for b in col.blocks if b.block_type != "table"]

    # 8. Merge all blocks and sort into reading order
    all_blocks = zone_blocks + all_body_blocks
    all_blocks = reading_order_sort(all_blocks)

    # 9. Filter noise blocks from output (keep in blocks list but mark)
    visible_blocks = [b for b in all_blocks if b.block_type != "noise"]

    return visible_blocks, columns, tables
