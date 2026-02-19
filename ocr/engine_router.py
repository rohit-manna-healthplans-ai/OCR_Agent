from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .tesseract_engine import run_tesseract


def score_words(words: List[Dict]) -> float:
    # Scores OCR word list (higher is better). Expects items like:
    #   {"text": str, "conf": float, "bbox": [x1,y1,x2,y2]}
    if not words:
        return 0.0
    total_chars = sum(len((w.get("text") or "").strip()) for w in words)
    avg_conf = sum(float(w.get("conf") or 0.0) for w in words) / max(1, len(words))
    return float(avg_conf * (1.0 + min(total_chars / 200.0, 5.0)))


def _likely_handwritten(img_rgb: np.ndarray, paddle_words: List[Dict]) -> bool:
    # No-OpenCV heuristic for handwriting.
    # Kept conservative: only marks handwritten when quality looks non-printed.
    if img_rgb is None:
        return False

    total_chars = sum(len((w.get("text") or "").strip()) for w in paddle_words)
    avg_conf = (sum(float(w.get("conf") or 0.0) for w in paddle_words) / max(1, len(paddle_words))) if paddle_words else 0.0

    gray = (
        img_rgb[..., 0].astype(np.float32) * 0.2989
        + img_rgb[..., 1].astype(np.float32) * 0.5870
        + img_rgb[..., 2].astype(np.float32) * 0.1140
    )
    gx = np.abs(gray[:, 1:] - gray[:, :-1]).mean() if gray.shape[1] > 1 else 0.0
    gy = np.abs(gray[1:, :] - gray[:-1, :]).mean() if gray.shape[0] > 1 else 0.0
    edge_level = float(gx + gy)

    if total_chars >= 120 and avg_conf >= 0.75:
        return False

    if total_chars >= 40 and avg_conf < 0.72 and edge_level >= 22.0:
        return True

    if total_chars >= 20 and avg_conf < 0.62 and edge_level >= 25.0:
        return True

    return False


def route_engine(
    img_rgb: np.ndarray,
    paddle_words: List[Dict],
    printed_engine: str = "tesseract",
) -> Tuple[str, List[Dict]]:
    # Routing policy:
    #   - Handwritten => PaddleOCR words
    #   - Printed => Tesseract words (default), fallback to Paddle if Tesseract is extremely weak
    printed_engine = (printed_engine or "tesseract").lower().strip()
    if printed_engine not in {"tesseract", "paddle"}:
        printed_engine = "tesseract"

    if _likely_handwritten(img_rgb, paddle_words):
        return "paddleocr(handwritten)", paddle_words

    if printed_engine == "paddle":
        return "paddleocr(printed)", paddle_words

    tess_words = run_tesseract(img_rgb)
    tess_score = score_words(tess_words)
    paddle_score = score_words(paddle_words)

    if tess_score < 0.25 and paddle_score > (tess_score + 0.20):
        return "paddleocr(fallback)", paddle_words

    return "tesseract(printed)", tess_words
