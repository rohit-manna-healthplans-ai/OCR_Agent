from __future__ import annotations

# Windows Stability Fix (PaddleOCR/PaddlePaddle):
import os
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("FLAGS_enable_mkldnn", "0")
os.environ.setdefault("FLAGS_use_dnnl", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import numpy as np
from PIL import Image
from paddleocr import PaddleOCR

from ocr.pdf_utils import extract_text_if_digital, render_pdf_to_images
from ocr.preprocess import preprocess
from ocr.postprocess import (
    poly_to_xyxy,
    group_into_lines,
    merge_line_bbox,
    compute_line_conf,
    line_text,
)
from ocr.schemas import Word, Line, PageResult, to_dict
from ocr.field_extract import extract_fields_from_page

_OCR_INSTANCE: Optional[PaddleOCR] = None


def get_ocr() -> PaddleOCR:
    global _OCR_INSTANCE
    if _OCR_INSTANCE is not None:
        return _OCR_INSTANCE

    common_kwargs = dict(
        use_angle_cls=True,
        lang="en",
        show_log=False,
    )

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


def _recognize_token(img_rgb: np.ndarray) -> str:
    """
    Recognition-only call. Works better for single-character crops from boxed forms.
    """
    ocr = get_ocr()
    try:
        r = ocr.ocr(img_rgb, det=False, rec=True, cls=False)
    except TypeError:
        # older versions: det=False is enough
        r = ocr.ocr(img_rgb, det=False, cls=False)
    if not r:
        return ""
    # formats vary: could be [(text, conf)] or [[(text, conf)]]
    if isinstance(r, list) and len(r) > 0:
        first = r[0]
        if isinstance(first, (list, tuple)) and len(first) >= 2 and isinstance(first[0], str):
            return first[0]
        if isinstance(first, list) and len(first) > 0 and isinstance(first[0], (list, tuple)) and len(first[0]) >= 2:
            return str(first[0][0])
    return ""


def _read_image(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.array(img)


def _ocr_on_image(img_rgb: np.ndarray) -> List[dict]:
    ocr = get_ocr()
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
    total_chars = sum(len((w.get("text") or "").strip()) for w in words)
    avg_conf = sum(float(w.get("conf") or 0.0) for w in words) / max(1, len(words))
    text = " ".join((w.get("text") or "") for w in words).lower()
    bonus = 0.0
    for kw in ["policy", "certificate", "company", "tpa", "name", "address", "city", "insured", "claim"]:
        if kw in text:
            bonus += 0.25
    return (avg_conf * (1.0 + min(6.0, total_chars / 200.0))) + bonus


def _ocr_best_of_presets(img_rgb: np.ndarray, preset: str) -> Tuple[List[dict], str, np.ndarray]:
    candidates = []
    presets = ["clean_doc", "photo", "low_light"] if preset == "auto" else [preset]
    for pr in presets:
        proc, used = preprocess(img_rgb, preset=pr)
        words = _ocr_on_image(proc)
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
) -> Dict[str, Any]:
    p = Path(input_path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {input_path}")

    suffix = p.suffix.lower()

    meta: Dict[str, Any] = {
        "engine": "paddleocr",
        "lang": "en",
        "preset_requested": preset,
        "dpi": dpi,
        "max_pages": max_pages,
        "input_type": "pdf" if suffix == ".pdf" else "image",
        "mkldnn_disabled": True,
    }

    if suffix == ".pdf":
        digital = extract_text_if_digital(str(p))
        if digital is not None:
            meta["digital_pdf"] = True
            pages: List[PageResult] = []
            for i, t in enumerate(digital["pages_text"][:max_pages]):
                pages.append(PageResult(page_index=i, width=0, height=0, text=(t or ""), lines=[]))
            out = to_dict(pages, meta)
            out["fields"] = {}
            return out
        meta["digital_pdf"] = False
        imgs = render_pdf_to_images(str(p), dpi=dpi, max_pages=max_pages)
    else:
        imgs = [_read_image(str(p))]

    page_results: List[PageResult] = []
    debug_info: List[dict] = []

    first_page_img = None

    for page_idx, img_rgb in enumerate(imgs):
        words, used_preset, used_img = _ocr_best_of_presets(img_rgb, preset=preset)
        meta["preset_used"] = used_preset

        if page_idx == 0:
            first_page_img = used_img

        h, w = used_img.shape[:2]
        line_groups = group_into_lines(words)

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
        page_results.append(PageResult(page_index=page_idx, width=w, height=h, text=page_text, lines=lines))

        if return_debug:
            debug_info.append(
                {
                    "page_index": page_idx,
                    "preset_used": used_preset,
                    "words_count": len(words),
                    "lines_count": len(lines),
                    "score": _score_words(words),
                }
            )

    out = to_dict(page_results, meta)

    # Field extraction with boxed fallback (uses page image + recognition-only function)
    if out.get("pages") and first_page_img is not None:
        out["fields"] = extract_fields_from_page(out["pages"][0], first_page_img, recognizer_fn=_recognize_token)
    else:
        out["fields"] = {}

    if return_debug:
        out["debug"] = debug_info
    return out
