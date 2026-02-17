from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import pytesseract


def _configure_tesseract_cmd() -> None:
    """
    Make Tesseract discoverable on Windows even if PATH is not refreshed.
    Priority:
      1) env TESSERACT_CMD (full path to tesseract.exe)
      2) common Windows install paths
      3) system PATH (default pytesseract behavior)
    """
    env_cmd = os.environ.get("TESSERACT_CMD")
    if env_cmd and Path(env_cmd).exists():
        pytesseract.pytesseract.tesseract_cmd = env_cmd
        return

    if os.name == "nt":
        candidates = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        ]
        for c in candidates:
            if Path(c).exists():
                pytesseract.pytesseract.tesseract_cmd = c
                return


def run_tesseract(img_rgb: np.ndarray) -> List[Dict[str, Any]]:
    """
    Returns words with bbox in [x1,y1,x2,y2] and conf in 0..1.
    """
    _configure_tesseract_cmd()

    try:
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
    except Exception as e:
        raise RuntimeError(
            "Tesseract is not available. Install Tesseract OCR and ensure it's in PATH "
            "or set env var TESSERACT_CMD to full path of tesseract.exe."
        ) from e

    words: List[Dict[str, Any]] = []
    n = len(data.get("text", []))
    for i in range(n):
        text = (data["text"][i] or "").strip()
        conf_raw = data.get("conf", ["-1"])[i]
        try:
            conf = float(conf_raw)
        except Exception:
            conf = -1.0

        if not text:
            continue

        x, y, w, h = int(data["left"][i]), int(data["top"][i]), int(data["width"][i]), int(data["height"][i])
        words.append(
            {
                "text": text,
                "conf": max(0.0, min(1.0, (conf / 100.0) if conf >= 0 else 0.0)),
                "bbox": [x, y, x + w, y + h],
            }
        )

    return words


def run_tesseract_single_char(
    img_rgb: np.ndarray,
    whitelist: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
) -> Dict[str, Any]:
    """
    Single character OCR for boxed/block-letter forms.
    Uses PSM 10 (single char). Returns {"text": <char>, "conf": <0..1>}
    """
    _configure_tesseract_cmd()
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # Preprocess: resize + threshold + padding for better single-char accuracy
    h, w = gray.shape[:2]
    scale = 3 if max(h, w) < 64 else 2
    gray = cv2.resize(gray, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 31, 10)

    thr = cv2.copyMakeBorder(thr, 8, 8, 8, 8, borderType=cv2.BORDER_CONSTANT, value=255)

    config = f'--oem 1 --psm 10 -c tessedit_char_whitelist={whitelist}'
    txt = pytesseract.image_to_string(thr, config=config) or ""
    txt = txt.strip()

    # Confidence from image_to_data (best-effort for single word)
    data = pytesseract.image_to_data(thr, config=config, output_type=pytesseract.Output.DICT)
    confs = []
    for c in data.get("conf", []):
        try:
            v = float(c)
            if v >= 0:
                confs.append(v)
        except Exception:
            pass
    conf = (sum(confs) / len(confs)) / 100.0 if confs else 0.0

    # Keep only 1 char
    if len(txt) > 1:
        txt = txt[0]
    return {"text": txt, "conf": max(0.0, min(1.0, conf))}
