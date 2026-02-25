import re
import statistics
import numpy as np
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

    def smart_cpu_text_fix(self, text):
        if not text:
            return ""

        t = re.sub(r'([a-zA-Z])([0-9])', r'\1 \2', text)
        t = re.sub(r'([0-9])([a-zA-Z])', r'\1 \2', t)

        doc = self.nlp(t)
        final_words = []

        for token in doc:
            clean_w = token.text
            if token.ent_type_ or token.pos_ in ['PROPN', 'NUM'] or not clean_w.islower() or not clean_w.isalpha():
                final_words.append(token.text_with_ws)
                continue

            if len(clean_w) > 3:
                suggestions = self.sym_spell.lookup(clean_w, Verbosity.CLOSEST, max_edit_distance=2)
                if suggestions:
                    final_words.append(suggestions[0].term + token.whitespace_)
                else:
                    final_words.append(token.text_with_ws)
            else:
                final_words.append(token.text_with_ws)

        res = "".join(final_words).strip()
        return res.replace("Firewall", "View all").replace("Viewall", "View all")

    def calculate_geometry(self, words, W, H):
        if not words:
            return 15, -1

        heights = [(w['bbox'][3] - w['bbox'][1]) for w in words]
        median_h = statistics.median(heights) if heights else 15
        sidebar_x = -1

        if W > H * 1.1:
            middle_words = [w for w in words if (H * 0.15) < ((w['bbox'][1] + w['bbox'][3]) / 2) < (H * 0.85)]
            if middle_words:
                x_profile = np.zeros(int(W))
                for w in middle_words:
                    x_profile[int(w['bbox'][0]):int(w['bbox'][2])] = 1

                start, end = int(W * 0.10), int(W * 0.25)
                zeros = np.where(x_profile[start:end] == 0)[0]

                if len(zeros) > 0:
                    gaps = np.split(zeros, np.where(np.diff(zeros) != 1)[0] + 1)
                    widest = max(gaps, key=len)
                    if len(widest) > (W * 0.02):
                        calculated_boundary = start + widest[0] + (len(widest) / 2)
                        sidebar_x = min(calculated_boundary, W * 0.22)

        return median_h, sidebar_x

    def process_data(self, data: dict) -> str:
        output_lines = []
        pages = data.get("pages", [])

        if not pages:
            text = data.get("text", "")
            output_lines.append("<BODY>")
            for line in text.split("\n"):
                if line.strip():
                    output_lines.append(f"  {self.smart_cpu_text_fix(line)}")
            return "\n".join(output_lines)

        for idx, page in enumerate(pages):
            output_lines.append(f"\n<PAGE {idx+1}>")

            lines = page.get("lines", [])
            W = page.get("width", 1920) or 1920
            H = page.get("height", 1080) or 1080

            all_words = []
            for line in lines:
                words = line.get("words", []) or [{'text': line['text'], 'bbox': line['bbox']}]
                for w in words:
                    cleaned = self.smart_cpu_text_fix(w['text'])
                    if cleaned:
                        all_words.append({'text': cleaned, 'bbox': w['bbox']})

            median_h, sidebar_boundary = self.calculate_geometry(all_words, W, H)
            zones = {'HEADER': [], 'SIDEBAR': [], 'BODY': [], 'FOOTER': []}

            for w in all_words:
                y = (w['bbox'][1] + w['bbox'][3]) / 2
                x = (w['bbox'][0] + w['bbox'][2]) / 2

                if y < (H * 0.12):
                    zones['HEADER'].append(w)
                elif y > (H * 0.92):
                    zones['FOOTER'].append(w)
                elif sidebar_boundary != -1 and x < sidebar_boundary:
                    zones['SIDEBAR'].append(w)
                else:
                    zones['BODY'].append(w)

            def build_zone(word_list, is_body=False):
                if not word_list:
                    return []

                word_list.sort(key=lambda item: (item['bbox'][1] + item['bbox'][3]) / 2)
                rows, curr_row = [], [word_list[0]]

                for i in range(1, len(word_list)):
                    y1 = (curr_row[0]['bbox'][1] + curr_row[0]['bbox'][3]) / 2
                    y2 = (word_list[i]['bbox'][1] + word_list[i]['bbox'][3]) / 2

                    if abs(y2 - y1) < (median_h * 0.65):
                        curr_row.append(word_list[i])
                    else:
                        rows.append(sorted(curr_row, key=lambda item: item['bbox'][0]))
                        curr_row = [word_list[i]]

                rows.append(sorted(curr_row, key=lambda item: item['bbox'][0]))

                out = []
                for r in rows:
                    line_str = ""
                    for j in range(len(r)):
                        if j > 0:
                            gap = r[j]['bbox'][0] - r[j-1]['bbox'][2]
                            if gap > (W * 0.035) and is_body:
                                line_str += "   |   "
                            else:
                                line_str += " "
                        line_str += r[j]['text']
                    out.append(line_str.strip())
                return out

            for zone_name in ["HEADER", "SIDEBAR", "BODY", "FOOTER"]:
                zone_content = build_zone(zones[zone_name], is_body=(zone_name == "BODY"))
                if zone_content:
                    output_lines.append(f"<{zone_name}>")
                    for line in zone_content:
                        output_lines.append(f"  {line}")

        return "\n".join(output_lines)
