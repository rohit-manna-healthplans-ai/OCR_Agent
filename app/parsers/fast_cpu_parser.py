from __future__ import annotations

import logging
import warnings
from typing import List, Dict, Any

import pkg_resources
import spacy
from symspellpy import SymSpell, Verbosity

from ocr.postprocess import fix_common_paddle_errors

logger = logging.getLogger(__name__)


class FastLocalCPUParser:

    def __init__(self):
        self.sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        dictionary_path = pkg_resources.resource_filename(
            "symspellpy", "frequency_dictionary_en_82_765.txt"
        )
        self.sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            warnings.warn(
                "en_core_web_sm not found; run: python -m spacy download en_core_web_sm "
                "or install the wheel from requirements.txt. "
                "Using blank English pipeline.",
                stacklevel=1,
            )
            logger.warning("spaCy model en_core_web_sm missing; using spacy.blank('en')")
            self.nlp = spacy.blank("en")

    def fix_text(self, text: str) -> str:
        if not text or len(text.strip()) < 2:
            return text

        t = fix_common_paddle_errors(text)

        doc = self.nlp(t)
        out = []
        for token in doc:
            word = token.text
            skip = (
                token.ent_type_
                or token.pos_ in ("PROPN", "NUM", "SYM", "PUNCT", "X")
                or not word.isalpha()
                or not word.islower()
                or len(word) <= 2
                or any(c.isdigit() for c in word)
            )
            if skip:
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

    def _render_heading(self, block: Dict) -> List[str]:
        out = []
        for ln in block.get("lines", []):
            t = self.fix_text(ln.get("text", "").strip())
            if t:
                out.append(f"## {t}")
        return out

    def _render_paragraph(self, block: Dict) -> List[str]:
        out = []
        for ln in block.get("lines", []):
            t = self.fix_text(ln.get("text", "").strip())
            if t:
                out.append(t)
        return out

    def _render_label_value(self, block: Dict, page_width: int) -> List[str]:
        out = []
        for ln in block.get("lines", []):
            text = ln.get("text", "").strip()
            words = ln.get("words", [])
            if not text:
                continue
            if words and page_width > 0:
                gap_thr = page_width * 0.05
                segments, current = [], [words[0]]
                for i in range(1, len(words)):
                    gap = words[i]["bbox"][0] - words[i - 1]["bbox"][2]
                    if gap > gap_thr:
                        segments.append(current)
                        current = [words[i]]
                    else:
                        current.append(words[i])
                segments.append(current)
                seg_texts = []
                for seg in segments:
                    st = self.fix_text(
                        " ".join((w.get("text") or "").strip() for w in seg).strip()
                    )
                    if st:
                        seg_texts.append(st)
                if len(seg_texts) > 1:
                    out.append("  |  ".join(seg_texts))
                elif seg_texts:
                    out.append(seg_texts[0])
            else:
                out.append(self.fix_text(text))
        return [x for x in out if x.strip()]

    def _render_noise(self, block: Dict) -> List[str]:
        out = []
        for ln in block.get("lines", []):
            t = self.fix_text(ln.get("text", "").strip())
            if t:
                out.append(t)
        return out

    def _render_block(self, block: Dict, page_width: int) -> List[str]:
        btype = block.get("block_type", "paragraph")
        if btype == "heading":
            return self._render_heading(block)
        if btype in ("table", "table_row"):
            return []
        if btype == "label_value":
            return self._render_label_value(block, page_width)
        if btype == "noise":
            return self._render_noise(block)
        return self._render_paragraph(block)

    def _render_tables(self, tables: List[Dict]) -> List[str]:
        if not tables:
            return []
        out = []
        for t in tables:
            out.append("")
            out.append("<TABLE>")
            for row_line in t.get("text", "").split("\n"):
                if row_line.strip():
                    out.append(f"  {row_line}")
            out.append("</TABLE>")
            out.append("")
        return out

    def _render_zone(
        self, zone_blocks: List[Dict], page_width: int, tag: str
    ) -> List[str]:
        if not zone_blocks:
            return []
        rendered = []
        for b in sorted(zone_blocks, key=lambda b: b.get("reading_order", 0)):
            for line in self._render_block(b, page_width):
                if line.strip():
                    rendered.append(f"  {line}")
        if not rendered:
            return []
        return [f"<{tag}>"] + rendered + [""]

    def _fallback_render(self, page: Dict) -> List[str]:
        import statistics

        W = page.get("width", 1920) or 1920
        H = page.get("height", 1080) or 1080
        all_words = []
        for line in page.get("lines", []):
            for w in line.get("words", []) or [
                {"text": line["text"], "bbox": line["bbox"],
                 "conf": line.get("conf", 0.9)}
            ]:
                t = self.fix_text((w.get("text") or "").strip())
                if t:
                    all_words.append({"text": t, "bbox": w["bbox"]})
        if not all_words:
            return []

        heights = [max(1, w["bbox"][3] - w["bbox"][1]) for w in all_words]
        med_h = statistics.median(heights) if heights else 15

        def zone(w):
            yc = (w["bbox"][1] + w["bbox"][3]) / 2
            return "HEADER" if yc < H * 0.10 else ("FOOTER" if yc > H * 0.93 else "BODY")

        buckets: Dict[str, List] = {"HEADER": [], "BODY": [], "FOOTER": []}
        for w in all_words:
            buckets[zone(w)].append(w)

        def build_rows(wlist):
            if not wlist:
                return []
            wlist.sort(key=lambda x: (x["bbox"][1] + x["bbox"][3]) / 2)
            rows, cur = [], [wlist[0]]
            for i in range(1, len(wlist)):
                y1 = (cur[0]["bbox"][1] + cur[0]["bbox"][3]) / 2
                y2 = (wlist[i]["bbox"][1] + wlist[i]["bbox"][3]) / 2
                if abs(y2 - y1) < med_h * 0.65:
                    cur.append(wlist[i])
                else:
                    rows.append(sorted(cur, key=lambda x: x["bbox"][0]))
                    cur = [wlist[i]]
            rows.append(sorted(cur, key=lambda x: x["bbox"][0]))
            return rows

        out = []
        for tag in ("HEADER", "BODY", "FOOTER"):
            rows = build_rows(buckets[tag])
            if not rows:
                continue
            out.append(f"<{tag}>")
            for row in rows:
                parts = []
                for j, w in enumerate(row):
                    if j > 0:
                        gap = row[j]["bbox"][0] - row[j - 1]["bbox"][2]
                        parts.append("  |  " if gap > W * 0.05 else " ")
                    parts.append(w["text"])
                line_str = "".join(parts).strip()
                if line_str:
                    out.append(f"  {line_str}")
            out.append("")
        return out

    def process_data(self, data: Dict[str, Any]) -> str:
        output_lines: List[str] = []
        pages = data.get("pages", [])

        if not pages:
            text = data.get("text", "")
            output_lines.append("<BODY>")
            for line in text.split("\n"):
                t = self.fix_text(line.strip())
                if t:
                    output_lines.append(f"  {t}")
            return "\n".join(output_lines)

        for idx, page in enumerate(pages):
            output_lines.append(f"\n<PAGE {idx + 1}>")

            W = page.get("width", 1920) or 1920
            blocks = page.get("blocks", [])
            tables = page.get("tables", [])

            if not blocks:
                output_lines.extend(self._fallback_render(page))
                continue

            def by_zone(z):
                return [b for b in blocks if b.get("block_type") == z]

            body_types = {"paragraph", "heading", "label_value",
                          "list_item", "caption", "noise"}
            body_blocks = [b for b in blocks if b.get("block_type") in body_types]

            output_lines.extend(self._render_zone(by_zone("header"), W, "HEADER"))
            output_lines.extend(self._render_zone(by_zone("sidebar"), W, "SIDEBAR"))

            has_content = bool(body_blocks) or bool(tables)
            if has_content:
                output_lines.append("<BODY>")

                queue = []
                for b in body_blocks:
                    queue.append((b.get("reading_order", 999), "block", b))
                if tables:
                    table_blocks = [b for b in blocks if b.get("block_type") == "table"]
                    table_ro = min(
                        (b.get("reading_order", 999) for b in table_blocks),
                        default=500,
                    )
                    queue.append((table_ro, "tables", tables))

                queue.sort(key=lambda x: x[0])

                for _, rtype, item in queue:
                    if rtype == "block":
                        for line in self._render_block(item, W):
                            if line.strip():
                                output_lines.append(f"  {line}")
                    else:
                        for tline in self._render_tables(item):
                            output_lines.append(
                                f"  {tline}" if tline.strip() else ""
                            )

                output_lines.append("")

            output_lines.extend(self._render_zone(by_zone("footer"), W, "FOOTER"))

        return "\n".join(output_lines)
