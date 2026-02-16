from __future__ import annotations

from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path

import fitz  # PyMuPDF
import pdfplumber
import numpy as np


def extract_text_if_digital(pdf_path: str, min_chars_per_page: int = 200) -> Optional[Dict[str, Any]]:
    """
    If PDF has selectable text, return it (best accuracy) and skip OCR.
    """
    texts = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            t = (page.extract_text() or "").strip()
            texts.append(t)

    good_pages = sum(1 for t in texts if len(t) >= min_chars_per_page)
    if good_pages == 0:
        return None

    full_text = "\n\n".join([t for t in texts if t]).strip()
    return {
        "is_digital_pdf": True,
        "pages_text": texts,
        "text": full_text,
    }


def render_pdf_to_images(pdf_path: str, dpi: int = 300, max_pages: int = 10) -> List[np.ndarray]:
    """
    Render PDF pages to RGB images (numpy arrays).
    """
    doc = fitz.open(pdf_path)
    n = min(len(doc), max_pages)

    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    images = []
    for i in range(n):
        page = doc.load_page(i)
        pix = page.get_pixmap(matrix=mat, alpha=False)  # RGB
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        images.append(img)
    doc.close()
    return images
