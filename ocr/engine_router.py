from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .tesseract_engine import run_tesseract


def score_words(words: List[Dict]) -> float:
    """Score OCR word list (higher is better). Favors both confidence and text volume."""
    if not words:
        return 0.0
    total_chars = sum(len((w.get("text") or "").strip()) for w in words)
    avg_conf = sum(float(w.get("conf") or 0.0) for w in words) / max(1, len(words))
    return float(avg_conf * (1.0 + min(total_chars / 200.0, 5.0)))


def _avg_conf(words: List[Dict]) -> float:
    if not words:
        return 0.0
    return float(sum(float(w.get("conf") or 0.0) for w in words) / max(1, len(words)))


def _conf_above_ratio(words: List[Dict], threshold: float) -> float:
    """Fraction of words with confidence >= threshold (0..1)."""
    if not words:
        return 0.0
    n = sum(1 for w in words if float(w.get("conf") or 0.0) >= threshold)
    return n / len(words)


def _word_heights(words: List[Dict]) -> List[float]:
    out: List[float] = []
    for w in words:
        b = w.get("bbox")
        if b and len(b) >= 4:
            out.append(float(b[3] - b[1]))
    return out


def _coefficient_of_variation(values: List[float]) -> float:
    """Std/mean; 0 if mean is 0 or empty."""
    if not values or len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    if mean <= 0:
        return 0.0
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return (variance ** 0.5) / mean


def _likely_handwritten(img_rgb: np.ndarray, paddle_words: List[Dict]) -> bool:
    """
    Smart handwritten vs printed detection using:
    - Text volume + average confidence (printed: high chars, high conf)
    - Edge complexity (handwritten: noisier edges)
    - Word height regularity (printed: more uniform heights; handwritten: more variable)
    - Confidence variance (handwritten: more variable per-word conf)
    """
    if img_rgb is None or not paddle_words:
        return False

    total_chars = sum(len((w.get("text") or "").strip()) for w in paddle_words)
    confs = [float(w.get("conf") or 0.0) for w in paddle_words]
    avg_conf = sum(confs) / len(confs)

    # Strong printed signal: lots of text + high confidence
    if total_chars >= 100 and avg_conf >= 0.78:
        return False
    if total_chars >= 150 and avg_conf >= 0.72:
        return False

    # Edge complexity (handwritten / noisy scans)
    gray = (
        img_rgb[..., 0].astype(np.float32) * 0.2989
        + img_rgb[..., 1].astype(np.float32) * 0.5870
        + img_rgb[..., 2].astype(np.float32) * 0.1140
    )
    gx = np.abs(gray[:, 1:] - gray[:, :-1]).mean() if gray.shape[1] > 1 else 0.0
    gy = np.abs(gray[1:, :] - gray[:-1, :]).mean() if gray.shape[0] > 1 else 0.0
    edge_level = float(gx + gy)

    # Word height regularity: printed text has more uniform heights
    heights = _word_heights(paddle_words)
    height_cv = _coefficient_of_variation(heights) if len(heights) >= 5 else 0.0
    # High height variation => likely handwritten
    if len(heights) >= 8 and height_cv > 0.35 and avg_conf < 0.70:
        return True

    # Confidence variance: handwritten often has mixed confidence per word
    conf_cv = _coefficient_of_variation(confs) if len(confs) >= 5 else 0.0
    if len(confs) >= 10 and conf_cv > 0.45 and avg_conf < 0.68:
        return True

    # Original heuristic: lower conf + high edge => handwriting
    if total_chars >= 30 and avg_conf < 0.72 and edge_level >= 22.0:
        return True
    if total_chars >= 15 and avg_conf < 0.60 and edge_level >= 25.0:
        return True

    return False


def route_engine(
    img_rgb: np.ndarray,
    paddle_words: List[Dict],
    printed_engine: str = "tesseract",
) -> Tuple[str, List[Dict]]:
    """
    Smart routing: handwritten => Paddle; printed => Tesseract with confidence checks.
    Confidence is in 0..1; fallback to Paddle when Tesseract is weak.
    """
    printed_engine = (printed_engine or "tesseract").lower().strip()
    if printed_engine not in {"tesseract", "paddle"}:
        printed_engine = "tesseract"

    # Handwritten: use Paddle only (skip Tesseract for speed + accuracy)
    if _likely_handwritten(img_rgb, paddle_words):
        return "paddleocr(handwritten)", paddle_words

    if printed_engine == "paddle":
        return "paddleocr(printed)", paddle_words

    # Printed path: run Tesseract and decide by quality
    tess_words = run_tesseract(img_rgb)
    tess_avg_conf = _avg_conf(tess_words)  # 0..1
    tess_high_conf_ratio = _conf_above_ratio(tess_words, 0.50)

    # Fallback: Tesseract confidence too low (use 0.20 = 20%, not 20)
    if tess_avg_conf < 0.20:
        return "paddleocr(conf_low_fallback)", paddle_words
    # Fallback: most words have low conf => Tesseract struggling
    if tess_high_conf_ratio < 0.40 and tess_avg_conf < 0.45:
        return "paddleocr(conf_weak_fallback)", paddle_words

    # Compare overall scores; prefer Tesseract when it's clearly better or comparable
    tess_score = score_words(tess_words)
    paddle_score = score_words(paddle_words)
    if tess_score < 0.25 and paddle_score > (tess_score + 0.20):
        return "paddleocr(score_fallback)", paddle_words
    # When close, prefer Tesseract for printed (usually faster and accurate for clean print)
    if tess_score >= paddle_score * 0.85 and tess_avg_conf >= 0.45:
        return "tesseract(printed)", tess_words
    if paddle_score > tess_score + 0.25:
        return "paddleocr(score_fallback)", paddle_words

    return "tesseract(printed)", tess_words
