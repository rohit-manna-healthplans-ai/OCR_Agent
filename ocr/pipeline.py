from __future__ import annotations

"""
FINAL production pipeline (Windows/Python 3.11) with:
- Primary OCR: PaddleOCR
- Fallback OCR: Tesseract (pytesseract)
- Routing: scoring + confidence gate
- Tables: bbox-based table detection (best-effort)
- PDFs: digital text extraction first; else render -> OCR
- Block letters (form boxes): detect box sequences + per-box single-char OCR (Tesseract PSM10)
- Analysis: returns analyzed document block with quality + warnings + per-page stats
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

# ---- Windows Stability Fix (PaddleOCR / PaddlePaddle) ----
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("FLAGS_enable_mkldnn", "0")
os.environ.setdefault("FLAGS_use_dnnl", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from paddleocr import PaddleOCR  # noqa: E402

from ocr.pdf_utils import extract_text_if_digital, render_pdf_to_images  # noqa: E402
from ocr.preprocess import preprocess  # noqa: E402
from ocr.postprocess import normalize_linebreaks, validate_fields  # noqa: E402
from ocr.schemas import Word, Line, PageResult, to_dict  # noqa: E402
from ocr.table_detect import detect_tables_from_page  # noqa: E402
from ocr.tesseract_engine import run_tesseract, run_tesseract_single_char  # noqa: E402
from ocr.field_extract import extract_fields_from_page  # noqa: E402
from ocr.analyze import analyze_result  # noqa: E402


# -----------------------------
# Geometry / line helpers
# -----------------------------
def poly_to_xyxy(poly: Any) -> List[int]:
    pts = poly
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
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
    heights = sorted(max(1, w["bbox"][3] - w["bbox"][1]) for w in ws)
    med_h = heights[len(heights) // 2] if heights else 12
    thr = max(8.0, med_h * 0.65)

    lines: List[List[dict]] = []
    current: List[dict] = []
    current_y: Optional[float] = None

    for w in ws:
        yc = y_center(w)
        if current_y is None:
            current = [w]
            current_y = yc
            continue
        if abs(yc - current_y) <= thr:
            current.append(w)
            current_y = (current_y * (len(current) - 1) + yc) / len(current)
        else:
            lines.append(sorted(current, key=lambda x: x["bbox"][0]))
            current = [w]
            current_y = yc

    if current:
        lines.append(sorted(current, key=lambda x: x["bbox"][0]))
    return lines


# -----------------------------
# PaddleOCR singleton
# -----------------------------
_OCR_INSTANCE: Optional[PaddleOCR] = None


def get_paddle() -> PaddleOCR:
    global _OCR_INSTANCE
    if _OCR_INSTANCE is not None:
        return _OCR_INSTANCE

    common_kwargs = dict(use_angle_cls=True, lang="en", show_log=False)

    try:
        _OCR_INSTANCE = PaddleOCR(**common_kwargs, use_mkldnn=False)
        return _OCR_INSTANCE
    except TypeError:
        pass

    try:
        _OCR_INSTANCE = PaddleOCR(**common_kwargs, enable_mkldnn=False)
        return _OCR_INSTANCE
    except TypeError:
        pass

    _OCR_INSTANCE = PaddleOCR(**common_kwargs)
    return _OCR_INSTANCE


# -----------------------------
# Routing score
# -----------------------------
def _score_words(words: List[dict]) -> float:
    if not words:
        return 0.0
    total_chars = sum(len((w.get("text") or "").strip()) for w in words)
    avg_conf = sum(float(w.get("conf") or 0.0) for w in words) / max(1, len(words))
    text = " ".join((w.get("text") or "") for w in words).lower()
    bonus = 0.0
    for kw in ["policy", "certificate", "company", "tpa", "name", "address", "city", "pincode", "pan", "ifsc"]:
        if kw in text:
            bonus += 0.25
    return (avg_conf * (1.0 + min(6.0, total_chars / 200.0))) + bonus


def _route_engine(img_rgb: np.ndarray, paddle_words: List[dict], force_engine: str = "auto") -> Tuple[str, List[dict]]:
    force_engine = (force_engine or "auto").lower().strip()
    if force_engine == "paddle":
        return "paddleocr", paddle_words
    if force_engine == "tesseract":
        return "tesseract", run_tesseract(img_rgb)

    paddle_score = _score_words(paddle_words)
    if paddle_score >= 0.65 and sum(len((w.get("text") or "")) for w in paddle_words) >= 60:
        return "paddleocr", paddle_words

    tess_words = run_tesseract(img_rgb)
    tess_score = _score_words(tess_words)
    if tess_score > paddle_score:
        return "tesseract", tess_words
    return "paddleocr", paddle_words


# -----------------------------
# OCR runners
# -----------------------------
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


# -----------------------------
# Preset selection (preprocess)
# -----------------------------
def _ocr_best_of_presets(img_rgb: np.ndarray, preset: str) -> Tuple[List[dict], str, np.ndarray]:
    candidates = []
    presets = ["clean_doc", "photo", "low_light"] if preset == "auto" else [preset]
    for pr in presets:
        proc, used = preprocess(img_rgb, preset=pr)
        words = _paddle_words_on_image(proc)
        score = _score_words(words)
        candidates.append((score, used, words, proc))
    candidates.sort(key=lambda x: x[0], reverse=True)
    best = candidates[0]
    return best[2], best[1], best[3]


def run_ocr(
    input_path: str,
    preset: str = "auto",
    dpi: int = 300,
    max_pages: int = 10,
    return_debug: bool = False,
    engine: str = "auto",
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
    }

    # Digital PDF shortcut (text-only)
    if suffix == ".pdf":
        digital = extract_text_if_digital(str(p))
        if digital is not None:
            meta["digital_pdf"] = True
            pages: List[PageResult] = []
            for i, t in enumerate(digital["pages_text"][:max_pages]):
                pages.append(PageResult(page_index=i, width=0, height=0, text=(t or ""), lines=[]))
            out = to_dict(pages, meta)
            out["fields"] = {}
            for pg in out.get("pages", []):
                pg["tables"] = []
            out["text"] = normalize_linebreaks(out.get("text") or "")
            out["analysis"] = analyze_result(out)
            return out

        meta["digital_pdf"] = False
        imgs = render_pdf_to_images(str(p), dpi=dpi, max_pages=max_pages)
    else:
        imgs = [_read_image(str(p))]

    page_results: List[PageResult] = []
    debug_info: List[dict] = []

    first_page_img: Optional[np.ndarray] = None
    engine_used_first: Optional[str] = None

    for page_idx, img_rgb in enumerate(imgs):
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
            lines.append(
                Line(
                    text=txt,
                    conf=conf,
                    bbox=bbox,
                    words=[Word(text=x["text"], conf=float(x["conf"]), bbox=x["bbox"]) for x in grp],
                )
            )
            page_text_lines.append(txt)

        page_text = "\n".join(page_text_lines).strip()
        page_results.append(PageResult(page_index=page_idx, width=w, height=h, text=page_text, lines=lines))

        if return_debug:
            debug_info.append(
                {
                    "page_index": page_idx,
                    "preset_used": used_preset,
                    "engine_used": engine_used,
                    "words_count": len(final_words),
                    "lines_count": len(lines),
                    "score": _score_words(final_words),
                }
            )

    out = to_dict(page_results, meta)

    # normalize text
    out["text"] = normalize_linebreaks(out.get("text") or "")
    for pg in out.get("pages", []):
        pg["text"] = normalize_linebreaks(pg.get("text") or "")

    # tables
    for pg in out.get("pages", []):
        pg["tables"] = detect_tables_from_page(pg)

    # fields + boxed sequences (block letters)
    if out.get("pages") and first_page_img is not None:
        def _char_fn(crop_rgb: np.ndarray) -> Dict[str, Any]:
            # choose whitelist suitable for forms (alnum)
            return run_tesseract_single_char(crop_rgb, whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")

        out["fields"] = extract_fields_from_page(out["pages"][0], first_page_img, recognize_char_fn=_char_fn)
        out["fields"] = validate_fields(out.get("fields") or {})
    else:
        out["fields"] = {}

    if return_debug:
        out["debug"] = debug_info

    out["analysis"] = analyze_result(out)
    return out
