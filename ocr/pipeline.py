from __future__ import annotations

"""
Enterprise OCR Pipeline (BACKWARD COMPATIBLE) + Screen Intelligence

What's new:
- SCREEN MODE (auto-detected for screenshots):
  - Segments regions: topbar / sidebar / main / footer
  - OCR per region (parallel-ready; currently sequential for stability)
  - Table detection only on main/sidebar (not on chrome/footer)
  - Reconstructs a copyable "page" that matches screen order (Markdown)
  - Exposes out["regions"] with per-region OCR, tables, markdown
- TABLE IMPROVEMENTS:
  - kind="form_grid" vs kind="table" (from table_detect)
  - form_grid auto-converted to fields pairs
- FIELD IMPROVEMENTS:
  - UI-safe key/value extraction (no URL/taskbar junk) from main region

All existing API endpoints keep working.
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
from ocr.preprocess import preprocess, is_probable_screenshot, segment_screen_regions, crop_region  # noqa: E402
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
    """Group word boxes into reading-order lines (production-safe).

    Fixes segmentation issues caused by fixed pixel thresholds by using
    an adaptive y-threshold based on median word height.
    """
    if not words:
        return []

    def y_center(w: dict) -> float:
        return (w["bbox"][1] + w["bbox"][3]) / 2.0

    def height(w: dict) -> float:
        return float(max(1, w["bbox"][3] - w["bbox"][1]))

    ws = [w for w in words if w.get("bbox") and len(w["bbox"]) == 4 and (w.get("text") or "").strip()]
    if not ws:
        return []

    # median height -> adaptive line tolerance
    hs = sorted(height(w) for w in ws)
    med_h = hs[len(hs) // 2]
    y_tol = max(12.0, min(0.75 * med_h, 28.0))  # clamp for stability

    ws = sorted(ws, key=lambda w: (y_center(w), w["bbox"][0]))
    lines: List[List[dict]] = []
    current: List[dict] = []
    current_y: Optional[float] = None

    for w in ws:
        yc = y_center(w)
        if current_y is None:
            current = [w]
            current_y = yc
            continue

        # Update current_y slowly to follow drift (scanned docs)
        if abs(yc - current_y) <= y_tol:
            current.append(w)
            current_y = (0.7 * current_y) + (0.3 * yc)
        else:
            lines.append(sorted(current, key=lambda x: x["bbox"][0]))
            current = [w]
            current_y = yc

    if current:
        lines.append(sorted(current, key=lambda x: x["bbox"][0]))

    # Optional: merge very small "orphan" lines into nearest (reduces over-segmentation)
    if len(lines) >= 2:
        merged: List[List[dict]] = []
        for ln in lines:
            if not merged:
                merged.append(ln)
                continue
            if len(ln) == 1:
                prev = merged[-1]
                prev_h = float(max(1, prev[-1]["bbox"][3] - prev[-1]["bbox"][1]))
                ln_h = float(max(1, ln[0]["bbox"][3] - ln[0]["bbox"][1]))
                # if heights similar and y-gap small, merge
                gap = abs(y_center(ln[0]) - y_center(prev[-1]))
                if gap <= max(y_tol, 0.9 * max(prev_h, ln_h)):
                    merged[-1] = prev + ln
                    continue
            merged.append(ln)
        lines = [sorted(l, key=lambda x: x["bbox"][0]) for l in merged]

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
    """Run Paddle on multiple preprocessing presets and keep the best.

    Production fix:
    - When preset=="auto", try several presets instead of only "clean_doc".
      UI screenshots often lose thin/anti-aliased text with aggressive binarization.
      Trying a "screen" preset (no binarization) improves recall.
    """
    presets = ["clean_doc", "screen", "photo", "low_light"] if preset == "auto" else [preset]
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

    # Default to Tesseract for short/low-score outputs, but protect against
    # the "empty region" failure mode common in UI screenshots.
    tess_words = run_tesseract(img_rgb)
    if tess_words:
        return "tesseract", tess_words
    # If Tesseract returns nothing but Paddle found something, keep Paddle.
    if paddle_words:
        return "paddle", paddle_words
    return "tesseract", tess_words


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
    if line_h >= 40:
        return "heading"
    if line_h >= 28:
        return "subheading"
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


def _page_from_words(page_index: int, img_rgb: np.ndarray, words: List[dict]) -> PageResult:
    h, w = img_rgb.shape[:2]
    line_groups = group_into_lines(words)

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
    return PageResult(page_index=page_index, width=w, height=h, text=page_text, lines=lines)


def _convert_form_grids_to_fields(tables: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    pairs: List[Dict[str, Any]] = []
    for t in tables or []:
        if t.get("kind") != "form_grid":
            continue
        rows = t.get("rows") or []
        if len(rows) != 2:
            continue
        keys = rows[0]
        vals = rows[1]
        for k, v in zip(keys, vals):
            kk = (k or "").strip()
            vv = (v or "").strip()
            if len(kk) < 2 or len(vv) < 1:
                continue
            pairs.append({"key": kk, "value": vv, "confidence": 0.95, "method": "form_grid"})
    return pairs


def _process_scanned_pdf_page_worker(args: Tuple[str, int, int, str, str, bool]) -> Tuple[int, PageResult, Optional[dict], Optional[np.ndarray], str, str, int, int]:
    pdf_path, page_idx, dpi, preset, engine, return_debug = args

    img_rgb = render_pdf_page(pdf_path, page_idx, dpi=dpi)

    paddle_words, used_preset, used_img = _ocr_best_of_presets(img_rgb, preset=preset)
    engine_used, final_words = _route_engine(used_img, paddle_words, force_engine=engine)

    page_result = _page_from_words(page_idx, used_img, final_words)

    dbg = None
    if return_debug:
        dbg = {
            "page_index": page_idx,
            "preset_used": used_preset,
            "engine_used": engine_used,
            "words_count": len(final_words),
            "lines_count": len(page_result.lines),
            "score": _score_words(final_words),
        }

    first_img = used_img if page_idx == 0 else None
    return page_idx, page_result, dbg, first_img, used_preset, engine_used, page_result.width, page_result.height


def _run_screen_ocr(img_rgb: np.ndarray, preset: str, engine: str, return_layout: bool, return_debug: bool) -> Dict[str, Any]:
    H, W = img_rgb.shape[:2]
    meta: Dict[str, Any] = {
        "input_type": "image",
        "screen_mode": True,
        "dpi": None,
        "preset_requested": preset,
        "engine_requested": engine,
        "return_layout": bool(return_layout),
    }

    # Regions
    regions = segment_screen_regions(img_rgb)
    meta["regions_detected"] = [r["name"] for r in regions]

    region_outputs: List[Dict[str, Any]] = []
    combined_parts: List[str] = []
    debug_info: List[dict] = []

    def _has_meaningful_text(s: str) -> bool:
        s = (s or "").strip()
        if not s:
            return False
        # must contain at least one alphanumeric character
        return any(ch.isalnum() for ch in s)

    for idx, r in enumerate(regions):
        name = r["name"]
        bbox = r["bbox"]
        crop = crop_region(img_rgb, bbox)

        paddle_words, used_preset, used_img = _ocr_best_of_presets(crop, preset=preset)
        engine_used, final_words = _route_engine(used_img, paddle_words, force_engine=engine)

        pr = _page_from_words(0, used_img, final_words)
        rd = to_dict([pr], {"_region": name})
        region_page = rd["pages"][0]
        region_page["text"] = normalize_linebreaks(region_page.get("text") or "")
        region_page["text_markdown"] = _build_markdown_for_page(region_page)

        # region tables
        # IMPORTANT: use the same processed image that OCR ran on.
        # This keeps table detection aligned and avoids NameError bugs.
        region_tables = detect_tables_from_page(region_page, region_name=name, img_rgb=used_img)
        region_page["tables"] = region_tables

        # layout only for main region (fast, useful)
        if return_layout and name == "main":
            try:
                region_page["layout"] = analyze_layout(used_img)
            except Exception:
                region_page["layout"] = {"available": False, "blocks": []}
        else:
            region_page["layout"] = {"available": False, "blocks": []}

        region_out = {
            "name": name,
            "bbox": bbox,  # in full image coords
            "engine_used": engine_used,
            "preset_used": used_preset,
            "page": region_page,
        }
        region_outputs.append(region_out)

        if return_debug:
            debug_info.append({
                "region": name,
                "preset_used": used_preset,
                "engine_used": engine_used,
                "words_count": len(final_words),
                "lines_count": len(region_page.get("lines") or []),
                "score": _score_words(final_words),
                "bbox": bbox,
            })

        # Build combined markdown in screen order
        # Only show a region section when it actually contains meaningful text.
        # This prevents confusing empty [TOPBAR]/[FOOTER]/[SIDEBAR] blocks.
        if _has_meaningful_text(region_page.get("text") or ""):
            combined_parts.append(f"## [{name.upper()}]")
            combined_parts.append(region_page.get("text_markdown") or "")

    # Build a synthetic "page 0" that represents the whole screen in text form
    combined_md = "\n".join([p for p in combined_parts if p]).strip()
    combined_text = normalize_linebreaks("\n".join([normalize_linebreaks(r["page"].get("text") or "") for r in region_outputs]).strip())

    page0 = PageResult(page_index=0, width=W, height=H, text=combined_text, lines=[])
    out = to_dict([page0], meta)
    out["text"] = combined_text
    out["pages"][0]["text_markdown"] = combined_md
    out["pages"][0]["tables"] = []  # tables are stored in regions
    out["pages"][0]["layout"] = {"available": False, "blocks": []}

    # Fields: from MAIN region only (business fields), plus form_grid conversion
    fields_pairs: List[Dict[str, Any]] = []
    main_region = next((r for r in region_outputs if r["name"] == "main"), None)
    if main_region:
        main_page = main_region["page"]
        # Convert form grids to fields
        fields_pairs.extend(_convert_form_grids_to_fields(main_page.get("tables") or []))

        # Extract additional KV from main (UI-safe extractor)
        try:
            kv = extract_fields_from_page(main_page, crop_region(img_rgb, main_region["bbox"]))
            for p in (kv.get("pairs") or []):
                fields_pairs.append(p)
        except Exception:
            pass

    # Dedup fields
    dedup = []
    seen = set()
    for p in fields_pairs:
        k = (p.get("key") or "").strip()
        v = (p.get("value") or "").strip()
        sig = (k.lower(), v.lower())
        if not k or not v or sig in seen:
            continue
        seen.add(sig)
        dedup.append(p)

    out["fields"] = validate_fields({"pairs": dedup, "total_pairs": len(dedup), "engine": "screen_kv_v1"})
    out["regions"] = region_outputs

    if return_debug:
        out["debug"] = debug_info

    out["analysis"] = analyze_result(out)
    return out


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

    if suffix != ".pdf":
        img = _read_image(str(p))
        # AUTO screen mode for screenshots
        if is_probable_screenshot(img):
            return _run_screen_ocr(img, preset=preset, engine=engine, return_layout=return_layout, return_debug=return_debug)

    # ----------------------------
    # PDF / Standard image pipeline
    # ----------------------------
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

                pr = _page_from_words(page_idx, used_img, final_words)
                page_results.append(pr)

                if return_debug:
                    debug_info.append({
                        "page_index": page_idx,
                        "preset_used": used_preset,
                        "engine_used": engine_used,
                        "words_count": len(final_words),
                        "lines_count": len(pr.lines),
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
            pg["tables"] = detect_tables_from_page(pg, region_name="main", img_rgb=None)

        if return_layout:
            for i, pg in enumerate(out.get("pages", [])):
                if i == 0 and first_page_img is not None:
                    pg["layout"] = analyze_layout(first_page_img)
                else:
                    pg["layout"] = {"available": False, "blocks": []}
        else:
            for pg in out.get("pages", []):
                pg["layout"] = {"available": False, "blocks": []}

        # Fields from page 0 (document-style)
        if out.get("pages") and first_page_img is not None:
            def _char_fn(crop_rgb: np.ndarray) -> Dict[str, Any]:
                return run_tesseract_single_char(crop_rgb, whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
            out["fields"] = extract_fields_from_page(out["pages"][0], first_page_img, recognize_char_fn=_char_fn)
            # Also auto-convert any form_grids to fields
            fg_pairs = _convert_form_grids_to_fields(out["pages"][0].get("tables") or [])
            if fg_pairs:
                merged = (out.get("fields") or {}).get("pairs") or []
                merged.extend(fg_pairs)
                out["fields"]["pairs"] = merged
                out["fields"]["total_pairs"] = len(merged)
            out["fields"] = validate_fields(out.get("fields") or {})
        else:
            out["fields"] = {}

        if return_debug:
            out["debug"] = debug_info

        out["analysis"] = analyze_result(out)
        return out

    # Standard image pipeline
    img = _read_image(str(p))
    paddle_words, used_preset, used_img = _ocr_best_of_presets(img, preset=preset)
    meta["preset_used"] = used_preset

    engine_used, final_words = _route_engine(used_img, paddle_words, force_engine=engine)
    meta["engine_used"] = engine_used

    pr = _page_from_words(0, used_img, final_words)
    page_results = [pr]

    debug_info: List[dict] = []
    if return_debug:
        debug_info.append({
            "page_index": 0,
            "preset_used": used_preset,
            "engine_used": engine_used,
            "words_count": len(final_words),
            "lines_count": len(pr.lines),
            "score": _score_words(final_words),
        })

    out = to_dict(page_results, meta)
    out["text"] = normalize_linebreaks(out.get("text") or "")

    # Use the actual processed image for table detection if available.
    # Avoid relying on variables that may not exist in this scope.
    used_img_local = locals().get("used_img", None)
    img_rgb_local = locals().get("img_rgb", None)
    image_for_tables = used_img_local if used_img_local is not None else img_rgb_local

    for pg in out.get("pages", []):
        pg["text"] = normalize_linebreaks(pg.get("text") or "")
        pg["text_markdown"] = _build_markdown_for_page(pg)
        pg["tables"] = detect_tables_from_page(pg, region_name="main", img_rgb=image_for_tables)

    if return_layout:
        out["pages"][0]["layout"] = analyze_layout(used_img)
    else:
        out["pages"][0]["layout"] = {"available": False, "blocks": []}

    def _char_fn(crop_rgb: np.ndarray) -> Dict[str, Any]:
        return run_tesseract_single_char(crop_rgb, whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    out["fields"] = extract_fields_from_page(out["pages"][0], used_img, recognize_char_fn=_char_fn)

    # Convert form_grids to fields
    fg_pairs = _convert_form_grids_to_fields(out["pages"][0].get("tables") or [])
    if fg_pairs:
        merged = (out.get("fields") or {}).get("pairs") or []
        merged.extend(fg_pairs)
        out["fields"]["pairs"] = merged
        out["fields"]["total_pairs"] = len(merged)

    out["fields"] = validate_fields(out.get("fields") or {})

    if return_debug:
        out["debug"] = debug_info

    out["analysis"] = analyze_result(out)
    return out
