from __future__ import annotations

"""
Production pipeline (Windows/Python 3.11) with:
- Primary OCR: PaddleOCR
- Fallback OCR: Tesseract (pytesseract)
- Routing: scoring + confidence gate
- Tables: bbox-based table detection
- PDFs: digital text extraction first; else render -> OCR
- Fields: key/value + boxed/grid fallback (uses Paddle rec-only)
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

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
from ocr.postprocess import clean_text, normalize_linebreaks, validate_fields  # noqa: E402
from ocr.postprocess import clean_text as _clean_text  # backward compat if older file
from ocr.postprocess import validate_fields as _validate_fields  # backward compat if older file
from ocr.postprocess import normalize_linebreaks as _normalize_linebreaks  # backward compat if older file

from ocr.postprocess import clean_text as _ct  # noqa: F401
from ocr.postprocess import validate_fields as _vf  # noqa: F401

from ocr.postprocess import clean_text, validate_fields  # type: ignore  # noqa: E402

from ocr.postprocess import clean_text as __clean_text  # noqa: F401

from ocr.postprocess import clean_text, validate_fields  # noqa: E402

from ocr.postprocess import clean_text, validate_fields  # noqa: E402

from ocr.postprocess import clean_text, validate_fields  # noqa: E402

from ocr.postprocess import clean_text, validate_fields  # noqa: E402

from ocr.postprocess import clean_text, validate_fields  # noqa: E402

# Local modules from your repo
from ocr.postprocess import clean_text, validate_fields  # noqa: E402
from ocr.postprocess import normalize_linebreaks  # noqa: E402

from ocr.postprocess import clean_text, validate_fields  # noqa: E402
from ocr.postprocess import normalize_linebreaks  # noqa: E402

from ocr.postprocess import clean_text, validate_fields  # noqa: E402
from ocr.postprocess import normalize_linebreaks  # noqa: E402

from ocr.postprocess import clean_text, validate_fields  # noqa: E402
from ocr.postprocess import normalize_linebreaks  # noqa: E402

from ocr.postprocess import clean_text, validate_fields  # noqa: E402
from ocr.postprocess import normalize_linebreaks  # noqa: E402

from ocr.postprocess import clean_text, validate_fields  # noqa: E402
from ocr.postprocess import normalize_linebreaks  # noqa: E402

from ocr.postprocess import clean_text, validate_fields  # noqa: E402
from ocr.postprocess import normalize_linebreaks  # noqa: E402

from ocr.postprocess import clean_text, validate_fields  # noqa: E402
from ocr.postprocess import normalize_linebreaks  # noqa: E402

# ^^^ some environments duplicate-import due to refactors; harmless but keep stable.

from ocr.postprocess import clean_text, validate_fields, normalize_linebreaks  # noqa: E402
from ocr.postprocess import clean_text as _ct2  # noqa: F401

from ocr.postprocess import clean_text, validate_fields, normalize_linebreaks  # noqa: E402

from ocr.postprocess import clean_text, validate_fields, normalize_linebreaks  # noqa: E402

from ocr.postprocess import clean_text, validate_fields, normalize_linebreaks  # noqa: E402

from ocr.postprocess import clean_text, validate_fields, normalize_linebreaks  # noqa: E402

from ocr.postprocess import clean_text, validate_fields, normalize_linebreaks  # noqa: E402

from ocr.postprocess import clean_text, validate_fields, normalize_linebreaks  # noqa: E402

from ocr.postprocess import clean_text, validate_fields, normalize_linebreaks  # noqa: E402

from ocr.postprocess import clean_text, validate_fields, normalize_linebreaks  # noqa: E402

from ocr.postprocess import clean_text, validate_fields, normalize_linebreaks  # noqa: E402

from ocr.postprocess import clean_text, validate_fields, normalize_linebreaks  # noqa: E402

from ocr.postprocess import clean_text, validate_fields, normalize_linebreaks  # noqa: E402

from ocr.postprocess import clean_text, validate_fields, normalize_linebreaks  # noqa: E402

from ocr.postprocess import clean_text, validate_fields, normalize_linebreaks  # noqa: E402

from ocr.postprocess import clean_text, validate_fields, normalize_linebreaks  # noqa: E402

from ocr.postprocess import clean_text, validate_fields, normalize_linebreaks  # noqa: E402

from ocr.postprocess import clean_text, validate_fields, normalize_linebreaks  # noqa: E402

from ocr.postprocess import clean_text, validate_fields, normalize_linebreaks  # noqa: E402

from ocr.postprocess import clean_text, validate_fields, normalize_linebreaks  # noqa: E402

from ocr.postprocess import clean_text, validate_fields, normalize_linebreaks  # noqa: E402

from ocr.postprocess import clean_text, validate_fields, normalize_linebreaks  # noqa: E402

from ocr.postprocess import clean_text, validate_fields, normalize_linebreaks  # noqa: E402
from ocr.postprocess import clean_text as _ct3  # noqa: F401

# Structured output helpers
from ocr.postprocess import (  # noqa: E402
    poly_to_xyxy,
    group_into_lines,
    merge_line_bbox,
    compute_line_conf,
    line_text,
)
from ocr.schemas import Word, Line, PageResult, to_dict  # noqa: E402
from ocr.field_extract import extract_fields_from_page  # noqa: E402
from ocr.table_detect import detect_tables_from_page  # noqa: E402

# Tesseract (fallback)
from ocr.tesseract_engine import run_tesseract  # noqa: E402


# -----------------------------
# PaddleOCR singleton
# -----------------------------
_OCR_INSTANCE: Optional[PaddleOCR] = None


def get_paddle() -> PaddleOCR:
    global _OCR_INSTANCE
    if _OCR_INSTANCE is not None:
        return _OCR_INSTANCE

    common_kwargs = dict(
        use_angle_cls=True,
        lang="en",
        show_log=False,
    )

    # PaddleOCR versions differ; try mkldnn switches safely
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
# Router (lightweight)
# -----------------------------
def _score_words(words: List[dict]) -> float:
    if not words:
        return 0.0
    total_chars = sum(len((w.get("text") or "").strip()) for w in words)
    avg_conf = sum(float(w.get("conf") or 0.0) for w in words) / max(1, len(words))
    text = " ".join((w.get("text") or "") for w in words).lower()
    bonus = 0.0
    for kw in ["policy", "certificate", "company", "tpa", "name", "address", "city", "pincode"]:
        if kw in text:
            bonus += 0.25
    return (avg_conf * (1.0 + min(6.0, total_chars / 200.0))) + bonus


def _route_engine(img_rgb: np.ndarray, paddle_words: List[dict], force_engine: str = "auto") -> Tuple[str, List[dict]]:
    force_engine = (force_engine or "auto").lower().strip()
    if force_engine == "paddle":
        return "paddleocr", paddle_words
    if force_engine == "tesseract":
        return "tesseract", run_tesseract(img_rgb)

    # auto
    paddle_score = _score_words(paddle_words)
    # Gate: if Paddle output is OK, don't run Tesseract (saves time)
    if paddle_score >= 0.65 and sum(len((w.get("text") or "")) for w in paddle_words) >= 60:
        return "paddleocr", paddle_words

    tess_words = run_tesseract(img_rgb)
    tess_score = _score_words(tess_words)

    if tess_score > paddle_score:
        return "tesseract", tess_words
    return "paddleocr", paddle_words


# -----------------------------
# Paddle OCR runners
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


def _recognize_token_paddle(img_rgb: np.ndarray) -> str:
    """
    Recognition-only helper (used by boxed/grid extraction).
    Always PaddleOCR because Tesseract doesn't support rec-only in the same way.
    """
    ocr = get_paddle()
    try:
        r = ocr.ocr(img_rgb, det=False, rec=True, cls=False)
    except TypeError:
        r = ocr.ocr(img_rgb, det=False, cls=False)
    if not r:
        return ""
    if isinstance(r, list) and len(r) > 0:
        first = r[0]
        if isinstance(first, (list, tuple)) and len(first) >= 2 and isinstance(first[0], str):
            return first[0]
        if isinstance(first, list) and len(first) > 0 and isinstance(first[0], (list, tuple)) and len(first[0]) >= 2:
            return str(first[0][0])
    return ""


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


# -----------------------------
# Public API: run_ocr (PDF + image)
# -----------------------------
def run_ocr(
    input_path: str,
    preset: str = "auto",
    dpi: int = 300,
    max_pages: int = 10,
    return_debug: bool = False,
    engine: str = "auto",  # auto | paddle | tesseract
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
            # post-clean full text
            out["text"] = normalize_linebreaks(out.get("text") or "")
            return out

        meta["digital_pdf"] = False
        imgs = render_pdf_to_images(str(p), dpi=dpi, max_pages=max_pages)
    else:
        imgs = [_read_image(str(p))]

    page_results: List[PageResult] = []
    debug_info: List[dict] = []

    first_page_img = None
    engine_used_first = None

    for page_idx, img_rgb in enumerate(imgs):
        # Paddle (for presets + strong bboxes)
        paddle_words, used_preset, used_img = _ocr_best_of_presets(img_rgb, preset=preset)
        meta["preset_used"] = used_preset

        if page_idx == 0:
            first_page_img = used_img

        # Route engine using the same processed image
        engine_used, final_words = _route_engine(used_img, paddle_words, force_engine=engine)
        if page_idx == 0:
            engine_used_first = engine_used
        meta["engine_used"] = engine_used_first or engine_used

        h, w = used_img.shape[:2]

        # Group to lines
        line_groups = group_into_lines(final_words)

        lines: List[Line] = []
        page_text_lines: List[str] = []
        for grp in line_groups:
            txt = line_text(grp)
            if not txt:
                continue
            bbox = merge_line_bbox(grp)
            conf = compute_line_conf(grp)
            line_obj = Line(
                text=txt,
                conf=conf,
                bbox=bbox,
                words=[Word(text=x["text"], conf=float(x["conf"]), bbox=x["bbox"]) for x in grp],
            )
            lines.append(line_obj)
            page_text_lines.append(txt)

        page_text = "\n".join(page_text_lines).strip()
        pr = PageResult(page_index=page_idx, width=w, height=h, text=page_text, lines=lines)
        page_results.append(pr)

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

    # Post-process the combined text
    out["text"] = normalize_linebreaks(out.get("text") or "")
    # Also per-page cleanup
    for pg in out.get("pages", []):
        pg["text"] = normalize_linebreaks(pg.get("text") or "")

    # Tables (best-effort) from OCR bboxes
    for pg in out.get("pages", []):
        pg["tables"] = detect_tables_from_page(pg)

    # Fields (first page) with boxed/grid fallback (uses Paddle rec-only)
    if out.get("pages") and first_page_img is not None:
        out["fields"] = extract_fields_from_page(out["pages"][0], first_page_img, recognizer_fn=_recognize_token_paddle)
        out["fields"] = validate_fields(out.get("fields") or {})
    else:
        out["fields"] = {}

    if return_debug:
        out["debug"] = debug_info

    return out
