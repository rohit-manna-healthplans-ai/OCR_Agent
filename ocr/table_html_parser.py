
from __future__ import annotations

from html.parser import HTMLParser
from typing import Any, Dict, List, Optional, Tuple


class _TableHTMLParser(HTMLParser):
    """
    Minimal HTML table parser that supports <table><tr><td>/<th> with rowspan/colspan.

    Returns a list of rows, each row is a list of cell dicts:
      {"text": "...", "rowspan": int, "colspan": int}
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.in_table = False
        self.in_tr = False
        self.in_cell = False

        self._current_cell_text: List[str] = []
        self._current_cell: Optional[Dict[str, Any]] = None
        self._current_row: List[Dict[str, Any]] = []
        self.rows: List[List[Dict[str, Any]]] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        tag = tag.lower()
        attrs_dict = {k.lower(): (v if v is not None else "") for k, v in attrs}

        if tag == "table":
            self.in_table = True
            return

        if not self.in_table:
            return

        if tag == "tr":
            self.in_tr = True
            self._current_row = []
            return

        if self.in_tr and tag in ("td", "th"):
            self.in_cell = True
            self._current_cell_text = []
            rowspan = int(attrs_dict.get("rowspan") or "1") if (attrs_dict.get("rowspan") or "").isdigit() else 1
            colspan = int(attrs_dict.get("colspan") or "1") if (attrs_dict.get("colspan") or "").isdigit() else 1
            self._current_cell = {"text": "", "rowspan": max(1, rowspan), "colspan": max(1, colspan)}
            return

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()

        if tag == "table":
            self.in_table = False
            self.in_tr = False
            self.in_cell = False
            return

        if not self.in_table:
            return

        if tag == "tr":
            self.in_tr = False
            if self._current_row:
                self.rows.append(self._current_row)
            self._current_row = []
            return

        if self.in_tr and tag in ("td", "th"):
            self.in_cell = False
            if self._current_cell is not None:
                txt = "".join(self._current_cell_text)
                txt = " ".join(txt.replace("\xa0", " ").split())
                self._current_cell["text"] = txt
                self._current_row.append(self._current_cell)
            self._current_cell = None
            self._current_cell_text = []
            return

    def handle_data(self, data: str) -> None:
        if self.in_table and self.in_tr and self.in_cell:
            self._current_cell_text.append(data)


def _expand_to_grid(rows: List[List[Dict[str, Any]]]) -> List[List[str]]:
    """
    Convert row/cell objects (with rowspan/colspan) into a rectangular grid of strings.
    """
    grid: List[List[str]] = []
    # Tracks pending rowspans: list of tuples (col_index, remaining_rows, text, colspan)
    spans: List[Tuple[int, int, str, int]] = []

    for r in rows:
        # Apply active spans into the current row baseline
        current: List[Optional[str]] = []
        # First, place span placeholders as needed
        if spans:
            # We'll ensure current is large enough as we go
            spans_sorted = sorted(spans, key=lambda x: x[0])
            for col_idx, rem, text, colspan in spans_sorted:
                while len(current) < col_idx:
                    current.append(None)
                # fill colspan
                for j in range(colspan):
                    if len(current) == col_idx + j:
                        current.append(text if j == 0 else "")
                    else:
                        current[col_idx + j] = text if j == 0 else ""
        # Now fill cells in order, skipping occupied slots
        cpos = 0
        for cell in r:
            # advance to next empty
            while cpos < len(current) and current[cpos] is not None:
                cpos += 1
            while len(current) < cpos:
                current.append(None)

            text = str(cell.get("text") or "")
            rowspan = int(cell.get("rowspan") or 1)
            colspan = int(cell.get("colspan") or 1)

            # ensure length
            while len(current) < cpos:
                current.append(None)

            # place
            for j in range(colspan):
                if len(current) == cpos + j:
                    current.append(text if j == 0 else "")
                else:
                    current[cpos + j] = text if j == 0 else ""
            # register span for subsequent rows
            if rowspan > 1:
                spans.append((cpos, rowspan - 1, text, colspan))
            cpos += colspan

        # Normalize None -> ""
        row_out = [x if x is not None else "" for x in current]
        grid.append(row_out)

        # Decrement spans
        new_spans: List[Tuple[int, int, str, int]] = []
        for col_idx, rem, text, colspan in spans:
            if rem - 1 > 0:
                new_spans.append((col_idx, rem - 1, text, colspan))
        spans = new_spans

    # Make rectangular
    max_cols = max((len(r) for r in grid), default=0)
    grid = [r + [""] * (max_cols - len(r)) for r in grid]

    # Trim fully-empty trailing columns
    if max_cols > 0:
        keep_cols = [j for j in range(max_cols) if any((row[j] or "").strip() for row in grid)]
        if keep_cols and len(keep_cols) != max_cols:
            grid = [[row[j] for j in keep_cols] for row in grid]

    return grid


def html_table_to_cells(html: str) -> List[List[str]]:
    """
    Best-effort conversion from PPStructure table HTML to a 2D list of cell strings.
    Returns [] if parsing fails or no rows found.
    """
    if not html or "<table" not in html.lower():
        return []
    try:
        parser = _TableHTMLParser()
        parser.feed(html)
        rows = parser.rows
        if not rows:
            return []
        grid = _expand_to_grid(rows)
        # Drop empty rows
        grid = [r for r in grid if any((c or "").strip() for c in r)]
        # Ensure >= 2 cols to call it a table
        if not grid:
            return []
        if len(grid[0]) < 2:
            return []
        return grid
    except Exception:
        return []
