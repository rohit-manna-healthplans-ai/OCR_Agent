from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np


def score_words(words: List[Dict]) -> float:
    """Score OCR word list (higher is better). Favors confidence and text volume."""
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
    if not values or len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    if mean <= 0:
        return 0.0
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return (variance ** 0.5) / mean


def _likely_handwritten(img_rgb: np.ndarray, paddle_words: List[Dict]) -> bool:
    """
    Heuristic: is this image likely handwritten vs. printed?
    Used for logging/debug only - PaddleOCR handles both cases now.
    """
    if img_rgb is None or not paddle_words:
        return False

    total_chars = sum(len((w.get("text") or "").strip()) for w in paddle_words)
    confs = [float(w.get("conf") or 0.0) for w in paddle_words]
    avg_conf = sum(confs) / len(confs)

    if total_chars >= 100 and avg_conf >= 0.78:
        return False
    if total_chars >= 150 and avg_conf >= 0.72:
        return False

    gray = (
        img_rgb[..., 0].astype(np.float32) * 0.2989
        + img_rgb[..., 1].astype(np.float32) * 0.5870
        + img_rgb[..., 2].astype(np.float32) * 0.1140
    )
    gx = np.abs(gray[:, 1:] - gray[:, :-1]).mean() if gray.shape[1] > 1 else 0.0
    gy = np.abs(gray[1:, :] - gray[:-1, :]).mean() if gray.shape[0] > 1 else 0.0
    edge_level = float(gx + gy)

    heights = _word_heights(paddle_words)
    height_cv = _coefficient_of_variation(heights) if len(heights) >= 5 else 0.0
    if len(heights) >= 8 and height_cv > 0.35 and avg_conf < 0.70:
        return True

    conf_cv = _coefficient_of_variation(confs) if len(confs) >= 5 else 0.0
    if len(confs) >= 10 and conf_cv > 0.45 and avg_conf < 0.68:
        return True

    if total_chars >= 30 and avg_conf < 0.72 and edge_level >= 22.0:
        return True
    if total_chars >= 15 and avg_conf < 0.60 and edge_level >= 25.0:
        return True

    return False


def select_best_paddle_result(
    candidates: List[Tuple[List[Dict], str, np.ndarray]]
) -> Tuple[List[Dict], str, np.ndarray]:
    """
    Given multiple (words, preset_name, processed_img) candidates from different
    preprocessing passes, return the one with the highest score_words().
    Tie-break: prefer higher avg_conf, then more words.
    """
    if not candidates:
        return [], "none", np.zeros((1, 1, 3), dtype=np.uint8)

    def sort_key(c: Tuple) -> Tuple[float, float, int]:
        words = c[0]
        return (score_words(words), _avg_conf(words), len(words))

    return max(candidates, key=sort_key)
