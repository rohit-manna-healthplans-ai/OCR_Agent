from __future__ import annotations

"""
Fast + stable pipeline (Windows/Python 3.11):
- PaddleOCR primary
- Tesseract fallback (auto routing, less frequent for speed)
- preset=auto uses only clean_doc (fast)
- layout via PP-Structure (page 0 only, optional)
- analysis always returned

PHASE-1 UPGRADE (BACKWARD COMPATIBLE):
- Memory-safe PDF rendering via render_pdf_pages() (generator)
- Optional multiprocessing for scanned PDFs (parallel_pages=True)

PHASE-UI UPGRADE (BACKWARD COMPATIBLE):
- Adds page["text_markdown"] with heading/bold heuristics derived from OCR bbox sizes.
  (No schema changes; extra keys are added post to_dict.)
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
from PIL import Image

os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("FLAGS_enable_mkldnn", "0")
os.environ.setdefault("FLAGS_use_dnnl", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

if TYPE_CHECKING:
    from paddleocr import PaddleOCR

from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402

from ocr.pdf_utils import extract_text_if_digital, render_pdf_pages, render_pdf_page  # noqa: E402
from ocr.preprocess import preprocess  # noqa: E402
from ocr.postprocess import normalize_linebreaks, validate_fields  # noqa: E402
from ocr.schemas import Word, Line, PageResult, to_dict  # noqa: E402
from ocr.table_detect import detect_tables_from_page  # noqa: E402
from ocr.tesseract_engine import run_tesseract, run_tesseract_single_char  # noqa: E402
from ocr.field_extract import extract_fields_from_page  # noqa: E402
from ocr.analyze import analyze_result  # noqa: E402
from ocr.layout import analyze_layout  # noqa: E402


def poly_to_xyxy(poly: Any) -> List[int]:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]


def merge_line_bbox(words: List[dict]) -> List[int]:
    return [
        int(min(w["bbox"][0] for w in words)),
        int(min(w["bbox"][1] for w in words)),
        int(max(w["bbox"][2] for w in words)),
        int(max(w["bbox"][3] for w in words)),
    ]


def compute_line_conf(words: List[dict]) -> float:
    return float(sum(float(w.get("conf", 0.0)) for w in words) / max(1, len(words))) if words else 0.0


def line_text(words: List[dict]) -> str:
    ws = sorted(words, key=lambda w: (w["bbox"][0], w["bbox"][1]))
    return " ".join((w.get("text") or "").strip() for w in ws).strip()


def group_into_lines(words: List[dict]) -> List[List[dict]]:
    if not words:
        return []

    def y_center(w: dict) -> float:
        return (w["bbox"][1] + w["bbox"][3]) / 2.0

    ws = sorted(words, key=lambda w: (y_center(w), w["bbox"][0]))
    lines: List[List[dict]] = []
    current: List[dict] = []
    current_y: Optional[float] = None

    for w in ws:
        yc = y_center(w)
        if current_y is None:
            current = [w]
            current_y = yc
            continue
        if abs(yc - current_y) <= 12:
            current.append(w)
        else:
            lines.append(sorted(current, key=lambda x: x["bbox"][0]))
            current = [w]
            current_y = yc

    if current:
        lines.append(sorted(current, key=lambda x: x["bbox"][0]))
    return lines


_OCR_INSTANCE: Optional[Any] = None


def get_paddle() -> Any:
    global _OCR_INSTANCE
    if _OCR_INSTANCE is not None:
        return _OCR_INSTANCE

    from paddleocr import PaddleOCR

    common_kwargs = dict(use_angle_cls=True, lang="en", show_log=False)

    try:
        _OCR_INSTANCE = PaddleOCR(**common_kwargs, use_mkldnn=False)
        return _OCR_INSTANCE
    except TypeError:
        pass
    try:
        _OCR_INSTANCE = PaddleOCR(**common_kwargs)
        return _OCR_INSTANCE
    except Exception:
        _OCR_INSTANCE = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
        return _OCR_INSTANCE


def _read_image(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.array(img)


def _paddle_words_on_image(img_rgb: np.ndarray) -> List[dict]:
    ocr = get_paddle()
    result = ocr.ocr(img_rgb, cls=True)

    words: List[dict] = []
    if result is None:
        return words

    items = result
    if isinstance(result, list) and len(result) == 1 and isinstance(result[0], list):
        if len(result[0]) == 0:
            return words
        first = result[0][0]
        if isinstance(first, (list, tuple)) and len(first) >= 2:
            items = result[0]

    for item in items:
        if not item:
            continue
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            poly = item[0]
            tc = item[1]
            if isinstance(tc, (list, tuple)) and len(tc) >= 2:
                text = str(tc[0])
                conf = float(tc[1])
            else:
                continue
            try:
                bbox = poly_to_xyxy(poly)
            except Exception:
                continue
            words.append({"text": text, "conf": conf, "bbox": bbox})
    return words


def _score_words(words: List[dict]) -> float:
    if not words:
        return 0.0
    avg_conf = sum(float(w.get("conf", 0.0)) for w in words) / max(1, len(words))
    return float(avg_conf * min(1.0, len(words) / 120.0))


def _ocr_best_of_presets(img_rgb: np.ndarray, preset: str) -> Tuple[List[dict], str, np.ndarray]:
    presets = ["clean_doc"] if preset == "auto" else [preset]
    candidates = []
    for pr in presets:
        proc, used = preprocess(img_rgb, preset=pr)
        words = _paddle_words_on_image(proc)
        score = _score_words(words)
        candidates.append((score, used, words, proc))
    candidates.sort(key=lambda x: x[0], reverse=True)
    best = candidates[0]
    return best[2], best[1], best[3]


def _route_engine(img_rgb: np.ndarray, paddle_words: List[dict], force_engine: str = "auto") -> Tuple[str, List[dict]]:
    if force_engine == "paddle":
        return "paddle", paddle_words
    if force_engine == "tesseract":
        return "tesseract", run_tesseract(img_rgb)

    score = _score_words(paddle_words)
    if score >= 0.35 and len(paddle_words) >= 15:
        return "paddle", paddle_words

    return "tesseract", run_tesseract(img_rgb)


def _uppercase_ratio(text: str) -> float:
    t = (text or "").strip()
    letters = [c for c in t if c.isalpha()]
    if not letters:
        return 0.0
    upp = sum(1 for c in letters if c.isupper())
    return upp / max(1, len(letters))


def _line_height(line_dict: Dict[str, Any]) -> int:
    bbox = line_dict.get("bbox") or [0, 0, 0, 0]
    try:
        return int(bbox[3] - bbox[1])
    except Exception:
        return 0


def _classify_line_style(line_text: str, line_h: int) -> str:
    # Quick heuristic tuned for DPI ~200
    if line_h >= 40:
        return "heading"
    if line_h >= 28:
        return "subheading"
    # Treat short, mostly uppercase as bold-ish label
    if len((line_text or "").strip()) <= 40 and _uppercase_ratio(line_text) >= 0.75:
        return "bold"
    return "normal"


def _build_markdown_for_page(page_dict: Dict[str, Any]) -> str:
    parts: List[str] = []
    for ln in page_dict.get("lines", []):
        txt = (ln.get("text") or "").strip()
        if not txt:
            continue
        h = _line_height(ln)
        style = _classify_line_style(txt, h)
        ln["style"] = style  # extra metadata, backward compatible

        if style == "heading":
            parts.append(f"# {txt}")
        elif style == "subheading":
            parts.append(f"## {txt}")
        elif style == "bold":
            parts.append(f"**{txt}**")
        else:
            parts.append(txt)
    return "\n".join(parts).strip()


def _process_scanned_pdf_page_worker(args: Tuple[str, int, int, str, str, bool]) -> Tuple[int, PageResult, Optional[dict], Optional[np.ndarray], str, str, int, int]:
    pdf_path, page_idx, dpi, preset, engine, return_debug = args

    img_rgb = render_pdf_page(pdf_path, page_idx, dpi=dpi)

    paddle_words, used_preset, used_img = _ocr_best_of_presets(img_rgb, preset=preset)
    engine_used, final_words = _route_engine(used_img, paddle_words, force_engine=engine)

    h, w = used_img.shape[:2]
    line_groups = group_into_lines(final_words)

    lines: List[Line] = []
    page_text_lines: List[str] = []
    for grp in line_groups:
        txt = line_text(grp)
        if not txt:
            continue
        bbox = merge_line_bbox(grp)
        conf = compute_line_conf(grp)
        lines.append(Line(
            text=txt,
            conf=conf,
            bbox=bbox,
            words=[Word(text=x["text"], conf=float(x["conf"]), bbox=x["bbox"]) for x in grp],
        ))
        page_text_lines.append(txt)

    page_text = "\n".join(page_text_lines).strip()
    page_result = PageResult(page_index=page_idx, width=w, height=h, text=page_text, lines=lines)

    dbg = None
    if return_debug:
        dbg = {
            "page_index": page_idx,
            "preset_used": used_preset,
            "engine_used": engine_used,
            "words_count": len(final_words),
            "lines_count": len(lines),
            "score": _score_words(final_words),
        }

    first_img = used_img if page_idx == 0 else None
    return page_idx, page_result, dbg, first_img, used_preset, engine_used, w, h


def run_ocr(
    input_path: str,
    preset: str = "auto",
    dpi: int = 200,
    max_pages: int = 5,
    return_debug: bool = True,
    engine: str = "auto",
    return_layout: bool = True,
    parallel_pages: bool = False,
    max_workers: Optional[int] = None,
) -> Dict[str, Any]:
    p = Path(input_path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {input_path}")

    suffix = p.suffix.lower()
    meta: Dict[str, Any] = {
        "engine": "auto(paddle+tesseract)",
        "lang": "en",
        "preset_requested": preset,
        "dpi": dpi,
        "max_pages": max_pages,
        "input_type": "pdf" if suffix == ".pdf" else "image",
        "mkldnn_disabled": True,
        "engine_requested": engine,
        "return_layout": bool(return_layout),
        "parallel_pages": bool(parallel_pages),
        "max_workers": int(max_workers) if max_workers else None,
        "has_markdown": True,
    }

    if suffix == ".pdf":
        digital = extract_text_if_digital(str(p), max_pages=max_pages)
        if digital:
            meta["digital_pdf"] = True
            pages: List[PageResult] = []
            for i, t in enumerate(digital.get("pages_text") or []):
                pages.append(PageResult(page_index=i, width=0, height=0, text=(t or ""), lines=[]))
            out = to_dict(pages, meta)
            out["fields"] = {}
            for pg in out.get("pages", []):
                pg["tables"] = []
                pg["layout"] = {"available": False, "blocks": []}
                pg["text_markdown"] = normalize_linebreaks(pg.get("text") or "")
            out["text"] = normalize_linebreaks(out.get("text") or "")
            out["analysis"] = analyze_result(out)
            return out

        meta["digital_pdf"] = False

        page_results: List[PageResult] = []
        debug_info: List[dict] = []

        first_page_img: Optional[np.ndarray] = None
        engine_used_first: Optional[str] = None
        preset_used_any: Optional[str] = None

        if parallel_pages:
            worker_count = int(max_workers) if max_workers else max(1, (os.cpu_count() or 2) - 1)
            tasks = [(str(p), i, dpi, preset, engine, return_debug) for i in range(max_pages)]
            with ProcessPoolExecutor(max_workers=worker_count) as ex:
                futures = [ex.submit(_process_scanned_pdf_page_worker, t) for t in tasks]
                for f in as_completed(futures):
                    page_idx, pr, dbg, first_img, used_preset, engine_used, w, h = f.result()
                    page_results.append(pr)
                    if dbg is not None:
                        debug_info.append(dbg)
                    if page_idx == 0 and first_img is not None:
                        first_page_img = first_img
                        engine_used_first = engine_used
                        preset_used_any = used_preset
        else:
            for page_idx, img_rgb in render_pdf_pages(str(p), dpi=dpi, max_pages=max_pages):
                paddle_words, used_preset, used_img = _ocr_best_of_presets(img_rgb, preset=preset)
                meta["preset_used"] = used_preset
                if page_idx == 0:
                    first_page_img = used_img

                engine_used, final_words = _route_engine(used_img, paddle_words, force_engine=engine)
                if page_idx == 0:
                    engine_used_first = engine_used
                meta["engine_used"] = engine_used_first or engine_used

                h, w = used_img.shape[:2]
                line_groups = group_into_lines(final_words)

                lines: List[Line] = []
                page_text_lines: List[str] = []
                for grp in line_groups:
                    txt = line_text(grp)
                    if not txt:
                        continue
                    bbox = merge_line_bbox(grp)
                    conf = compute_line_conf(grp)
                    lines.append(Line(
                        text=txt,
                        conf=conf,
                        bbox=bbox,
                        words=[Word(text=x["text"], conf=float(x["conf"]), bbox=x["bbox"]) for x in grp],
                    ))
                    page_text_lines.append(txt)

                page_text = "\n".join(page_text_lines).strip()
                page_results.append(PageResult(page_index=page_idx, width=w, height=h, text=page_text, lines=lines))

                if return_debug:
                    debug_info.append({
                        "page_index": page_idx,
                        "preset_used": used_preset,
                        "engine_used": engine_used,
                        "words_count": len(final_words),
                        "lines_count": len(lines),
                        "score": _score_words(final_words),
                    })

        page_results.sort(key=lambda x: x.page_index)
        debug_info.sort(key=lambda x: x.get("page_index", 0))

        if engine_used_first:
            meta["engine_used"] = engine_used_first
        if preset_used_any:
            meta["preset_used"] = preset_used_any

        out = to_dict(page_results, meta)

        for pg in out.get("pages", []):
            pg["text"] = normalize_linebreaks(pg.get("text") or "")
            pg["text_markdown"] = _build_markdown_for_page(pg)

        for pg in out.get("pages", []):
            pg["tables"] = detect_tables_from_page(pg)

        if return_layout:
            for i, pg in enumerate(out.get("pages", [])):
                if i == 0 and first_page_img is not None:
                    pg["layout"] = analyze_layout(first_page_img)
                else:
                    pg["layout"] = {"available": False, "blocks": []}
        else:
            for pg in out.get("pages", []):
                pg["layout"] = {"available": False, "blocks": []}

        if out.get("pages") and first_page_img is not None:
            def _char_fn(crop_rgb: np.ndarray) -> Dict[str, Any]:
                return run_tesseract_single_char(crop_rgb, whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
            out["fields"] = extract_fields_from_page(out["pages"][0], first_page_img, recognize_char_fn=_char_fn)
            out["fields"] = validate_fields(out.get("fields") or {})
        else:
            out["fields"] = {}

        if return_debug:
            out["debug"] = debug_info

        out["analysis"] = analyze_result(out)
        return out

    img = _read_image(str(p))
    paddle_words, used_preset, used_img = _ocr_best_of_presets(img, preset=preset)
    meta["preset_used"] = used_preset

    engine_used, final_words = _route_engine(used_img, paddle_words, force_engine=engine)
    meta["engine_used"] = engine_used

    h, w = used_img.shape[:2]
    line_groups = group_into_lines(final_words)

    lines: List[Line] = []
    page_text_lines: List[str] = []
    for grp in line_groups:
        txt = line_text(grp)
        if not txt:
            continue
        bbox = merge_line_bbox(grp)
        conf = compute_line_conf(grp)
        lines.append(Line(
            text=txt,
            conf=conf,
            bbox=bbox,
            words=[Word(text=x["text"], conf=float(x["conf"]), bbox=x["bbox"]) for x in grp],
        ))
        page_text_lines.append(txt)

    page_text = "\n".join(page_text_lines).strip()
    page_results = [PageResult(page_index=0, width=w, height=h, text=page_text, lines=lines)]

    debug_info: List[dict] = []
    if return_debug:
        debug_info.append({
            "page_index": 0,
            "preset_used": used_preset,
            "engine_used": engine_used,
            "words_count": len(final_words),
            "lines_count": len(lines),
            "score": _score_words(final_words),
        })

    out = to_dict(page_results, meta)
    out["text"] = normalize_linebreaks(out.get("text") or "")

    for pg in out.get("pages", []):
        pg["text"] = normalize_linebreaks(pg.get("text") or "")
        pg["text_markdown"] = _build_markdown_for_page(pg)
        pg["tables"] = detect_tables_from_page(pg)

    if return_layout:
        out["pages"][0]["layout"] = analyze_layout(used_img)
    else:
        out["pages"][0]["layout"] = {"available": False, "blocks": []}

    def _char_fn(crop_rgb: np.ndarray) -> Dict[str, Any]:
        return run_tesseract_single_char(crop_rgb, whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    out["fields"] = extract_fields_from_page(out["pages"][0], used_img, recognize_char_fn=_char_fn)
    out["fields"] = validate_fields(out.get("fields") or {})

    if return_debug:
        out["debug"] = debug_info

    out["analysis"] = analyze_result(out)
    return out
