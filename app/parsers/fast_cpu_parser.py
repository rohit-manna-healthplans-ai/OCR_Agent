from __future__ import annotations

"""
FastLocalCPUParser - converts structured OCR output (blocks + layout) into
clean, human-readable formatted text.

Output format per page:
    <PAGE N>
    <HEADER>
      ... header lines ...
    <BODY>
      ... body content with layout-aware formatting ...
    <SIDEBAR>
      ... sidebar content ...
    <FOOTER>
      ... footer lines ...

    Tables are rendered as markdown tables.
    Label-value pairs are formatted as aligned key: value lines.
    Multi-column layouts are rendered column by column with a divider.
"""

import re
import statistics
from typing import List, Dict, Any, Optional

import pkg_resources
import spacy
from symspellpy import SymSpell, Verbosity


class FastLocalCPUParser:

    def __init__(self):
        self.sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        dictionary_path = pkg_resources.resource_filename(
            "symspellpy", "frequency_dictionary_en_82_765.txt"
        )
        self.sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
        self.nlp = spacy.load("en_core_web_sm")

    # -- Text correction ---------------------------------------------------

    def fix_text(self, text: str) -> str:
        """
        Conservative spell-correction using SymSpell + spaCy NER.
        Skips: named entities, proper nouns, numbers, IDs (mixed-case /
        contains digits), short tokens (<=2 chars), non-alpha tokens.
        """
        if not text or len(text.strip()) < 2:
            return text

        # Protect numeric adjacency
        t = re.sub(r"([a-zA-Z])([0-9])", r"\1 \2", text)
        t = re.sub(r"([0-9])([a-zA-Z])", r"\1 \2", t)

        doc = self.nlp(t)
        out = []
        for token in doc:
            word = token.text
            # Skip correction for special tokens
            if (
                token.ent_type_
                or token.pos_ in ("PROPN", "NUM", "SYM", "PUNCT", "X")
                or not word.isalpha()
                or not word.islower()
                or len(word) <= 2
                or any(c.isdigit() for c in word)
            ):
                out.append(token.text_with_ws)
                continue

            if len(word) > 3:
                suggestions = self.sym_spell.lookup(
                    word, Verbosity.CLOSEST, max_edit_distance=2
                )
                if suggestions and suggestions[0].term != word:
                    out.append(suggestions[0].term + token.whitespace_)
                else:
                    out.append(token.text_with_ws)
            else:
                out.append(token.text_with_ws)

        return "".join(out).strip()

    # -- Block renderers ---------------------------------------------------

    def _render_heading(self, block: Dict[str, Any]) -> List[str]:
        lines = block.get("lines", [])
        out = []
        for ln in lines:
            text = self.fix_text(ln.get("text", "").strip())
            if text:
                out.append(f"## {text}")
        return out

    def _render_paragraph(self, block: Dict[str, Any]) -> List[str]:
        lines = block.get("lines", [])
        out = []
        for ln in lines:
            text = self.fix_text(ln.get("text", "").strip())
            if text:
                out.append(text)
        return out

    def _render_label_value(self, block: Dict[str, Any], page_width: int) -> List[str]:
        """
        Render label-value lines with consistent alignment.
        Detects the colon separator and aligns values.
        Handles multi-value lines (multiple label: value pairs on one line).
        """
        lines = block.get("lines", [])
        out = []
        for ln in lines:
            text = ln.get("text", "").strip()
            words = ln.get("words", [])
            if not text:
                continue

            # If we have word-level bboxes, use gaps to detect multi-column pairs
            if words and page_width > 0:
                # Split at large horizontal gaps into separate label:value segments
                segments = []
                current_seg = [words[0]]
                gap_threshold = page_width * 0.06

                for i in range(1, len(words)):
                    gap = words[i]["bbox"][0] - words[i - 1]["bbox"][2]
                    if gap > gap_threshold:
                        segments.append(current_seg)
                        current_seg = [words[i]]
                    else:
                        current_seg.append(words[i])
                segments.append(current_seg)

                seg_texts = []
                for seg in segments:
                    seg_text = " ".join(
                        (w.get("text") or "").strip() for w in seg
                    ).strip()
                    if seg_text:
                        seg_texts.append(self.fix_text(seg_text))

                if len(seg_texts) > 1:
                    out.append("  |  ".join(seg_texts))
                elif seg_texts:
                    out.append(seg_texts[0])
            else:
                out.append(self.fix_text(text))

        return out

    def _render_table(self, block: Dict[str, Any]) -> List[str]:
        """Render a table block as a markdown table."""
        # Use pre-reconstructed table if available in block metadata
        table_text = block.get("text", "")
        if "|" in table_text:
            return table_text.split("\n")
        # Fallback: render lines as pipe-separated
        lines = block.get("lines", [])
        if not lines:
            return []
        rows = []
        for ln in lines:
            words = ln.get("words", [])
            if words:
                cells = [self.fix_text((w.get("text") or "").strip()) for w in words]
                rows.append("| " + " | ".join(cells) + " |")
            elif ln.get("text"):
                rows.append(ln["text"])
        return rows

    def _render_list_item(self, block: Dict[str, Any]) -> List[str]:
        lines = block.get("lines", [])
        out = []
        for ln in lines:
            text = self.fix_text(ln.get("text", "").strip())
            if text:
                out.append(f"• {text}")
        return out

    def _render_block(self, block: Dict[str, Any], page_width: int) -> List[str]:
        btype = block.get("block_type", "paragraph")
        if btype == "heading":
            return self._render_heading(block)
        if btype in ("table_row", "table"):
            # table blocks are rendered separately via _render_tables()
            # individual table_row remnants should not appear here
            return []
        if btype == "label_value":
            return self._render_label_value(block, page_width)
        if btype == "list_item":
            return self._render_list_item(block)
        if btype == "noise":
            return []
        # paragraph, caption, and anything else
        return self._render_paragraph(block)

    # -- Zone assembly -----------------------------------------------------

    def _blocks_for_zone(
        self, blocks: List[Dict], zone: str
    ) -> List[Dict]:
        return [b for b in blocks if b.get("block_type") == zone]

    def _body_blocks(self, blocks: List[Dict]) -> List[Dict]:
        body_types = {
            "paragraph", "heading", "table_row", "table",
            "label_value", "list_item", "caption"
        }
        return [b for b in blocks if b.get("block_type") in body_types]

    def _render_zone(
        self,
        zone_blocks: List[Dict],
        page_width: int,
        zone_tag: str,
        indent: str = "  ",
    ) -> List[str]:
        if not zone_blocks:
            return []
        out = [f"<{zone_tag}>"]
        for block in sorted(zone_blocks, key=lambda b: b.get("reading_order", 0)):
            rendered = self._render_block(block, page_width)
            for line in rendered:
                if line.strip():
                    out.append(f"{indent}{line}")
        return out

    # -- Column layout rendering -------------------------------------------

    def _render_columns(
        self,
        columns: List[Dict],
        page_width: int,
    ) -> List[str]:
        """
        Render multi-column body content.
        For single-column pages: render blocks in reading order directly.
        For multi-column pages: render each column separately with a divider.
        """
        if not columns:
            return []

        if len(columns) == 1:
            col = columns[0]
            out = []
            for block in sorted(
                col.get("blocks", []), key=lambda b: b.get("reading_order", 0)
            ):
                rendered = self._render_block(block, page_width)
                for line in rendered:
                    if line.strip():
                        out.append(f"  {line}")
            return out

        # Multi-column: render side by side with dividers
        out = []
        for ci, col in enumerate(columns):
            if ci > 0:
                out.append("  " + "─" * 40)
            col_blocks = sorted(
                col.get("blocks", []), key=lambda b: b.get("reading_order", 0)
            )
            for block in col_blocks:
                rendered = self._render_block(block, page_width)
                for line in rendered:
                    if line.strip():
                        out.append(f"  {line}")
        return out

    # -- Standalone table rendering ----------------------------------------

    def _render_tables(self, tables: List[Dict]) -> List[str]:
        if not tables:
            return []
        out = []
        for t in tables:
            out.append("")
            out.append("  <TABLE>")
            for row in t.get("text", "").split("\n"):
                if row.strip():
                    out.append(f"  {row}")
            out.append("  </TABLE>")
        return out

    # -- Main entry --------------------------------------------------------

    def process_data(self, data: Dict[str, Any]) -> str:
        """
        Convert OCR JSON output to structured formatted text.

        Uses layout blocks (if present) for intelligent rendering.
        Falls back to flat text for simple/legacy responses.
        """
        output_lines: List[str] = []
        pages = data.get("pages", [])

        if not pages:
            # Fallback: no pages, render flat text
            text = data.get("text", "")
            output_lines.append("<BODY>")
            for line in text.split("\n"):
                fixed = self.fix_text(line.strip())
                if fixed:
                    output_lines.append(f"  {fixed}")
            return "\n".join(output_lines)

        for idx, page in enumerate(pages):
            output_lines.append(f"\n<PAGE {idx + 1}>")

            W = page.get("width", 1920) or 1920
            H = page.get("height", 1080) or 1080
            _ = H
            blocks: List[Dict] = page.get("blocks", [])
            columns: List[Dict] = page.get("columns", [])
            tables: List[Dict] = page.get("tables", [])

            # -- If no layout blocks, fall back to word-based layout -------
            if not blocks:
                lines_raw = page.get("lines", [])
                all_words = []
                for line in lines_raw:
                    for w in line.get("words", []) or [
                        {"text": line["text"], "bbox": line["bbox"], "conf": line.get("conf", 0.9)}
                    ]:
                        text = self.fix_text((w.get("text") or "").strip())
                        if text:
                            all_words.append({"text": text, "bbox": w["bbox"]})

                if not all_words:
                    continue

                # Simple zone split for fallback
                header_y = H * 0.12
                footer_y = H * 0.92

                def simple_zone(w):
                    yc = (w["bbox"][1] + w["bbox"][3]) / 2
                    if yc < header_y:
                        return "HEADER"
                    if yc > footer_y:
                        return "FOOTER"
                    return "BODY"

                zone_words: Dict[str, List] = {
                    "HEADER": [], "BODY": [], "FOOTER": []
                }
                for w in all_words:
                    zone_words[simple_zone(w)].append(w)

                def build_rows(wlist):
                    if not wlist:
                        return []
                    heights = [max(1, w["bbox"][3] - w["bbox"][1]) for w in wlist]
                    med = statistics.median(heights)
                    wlist.sort(key=lambda x: (x["bbox"][1] + x["bbox"][3]) / 2)
                    rows, cur = [], [wlist[0]]
                    for i in range(1, len(wlist)):
                        y1 = (cur[0]["bbox"][1] + cur[0]["bbox"][3]) / 2
                        y2 = (wlist[i]["bbox"][1] + wlist[i]["bbox"][3]) / 2
                        if abs(y2 - y1) < med * 0.65:
                            cur.append(wlist[i])
                        else:
                            rows.append(sorted(cur, key=lambda x: x["bbox"][0]))
                            cur = [wlist[i]]
                    rows.append(sorted(cur, key=lambda x: x["bbox"][0]))
                    return rows

                for zone_tag in ("HEADER", "BODY", "FOOTER"):
                    rows = build_rows(zone_words[zone_tag])
                    if not rows:
                        continue
                    output_lines.append(f"<{zone_tag}>")
                    for row in rows:
                        parts = []
                        for j, w in enumerate(row):
                            if j > 0:
                                gap = row[j]["bbox"][0] - row[j - 1]["bbox"][2]
                                parts.append("  |  " if gap > W * 0.04 else " ")
                            parts.append(w["text"])
                        line_str = "".join(parts).strip()
                        if line_str:
                            output_lines.append(f"  {line_str}")
                continue

            # -- Full layout rendering using blocks ------------------------

            # Render HEADER zone
            header_blocks = self._blocks_for_zone(blocks, "header")
            output_lines.extend(self._render_zone(header_blocks, W, "HEADER"))

            # Render SIDEBAR zone
            sidebar_blocks = self._blocks_for_zone(blocks, "sidebar")
            output_lines.extend(self._render_zone(sidebar_blocks, W, "SIDEBAR"))

            # Render BODY: non-table blocks first, then tables inline
            body_blocks = self._body_blocks(blocks)
            has_content = bool(body_blocks) or bool(tables)

            if has_content:
                output_lines.append("<BODY>")

                # Render non-table body blocks
                non_table_blocks = [
                    b for b in body_blocks
                    if b.get("block_type") not in ("table", "table_row")
                ]
                for block in sorted(non_table_blocks, key=lambda b: b.get("reading_order", 0)):
                    rendered = self._render_block(block, W)
                    for line in rendered:
                        if line.strip():
                            output_lines.append(f"  {line}")

                # Render tables with clear delimiters
                if tables:
                    for t in tables:
                        output_lines.append("")
                        output_lines.append("  <TABLE>")
                        table_text = t.get("text", "")
                        for row_line in table_text.split("\n"):
                            if row_line.strip():
                                output_lines.append(f"  {row_line}")
                        output_lines.append("  </TABLE>")
                        output_lines.append("")

            # Render FOOTER zone
            footer_blocks = self._blocks_for_zone(blocks, "footer")
            output_lines.extend(self._render_zone(footer_blocks, W, "FOOTER"))

        return "\n".join(output_lines)
