
from typing import List, Dict
from .tesseract_engine import run_tesseract

def score_words(words: List[Dict]) -> float:
    if not words:
        return 0.0
    total_chars = sum(len(w["text"]) for w in words)
    avg_conf = sum(w["conf"] for w in words) / max(1, len(words))
    return avg_conf * (1 + min(total_chars / 200.0, 5))

def route_engine(img_rgb, paddle_words):
    paddle_score = score_words(paddle_words)
    if paddle_score > 0.6:
        return "paddle", paddle_words

    tess_words = run_tesseract(img_rgb)
    tess_score = score_words(tess_words)

    if tess_score > paddle_score:
        return "tesseract", tess_words
    return "paddle", paddle_words
