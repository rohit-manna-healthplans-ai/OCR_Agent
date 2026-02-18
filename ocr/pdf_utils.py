from __future__ import annotations

from typing import List, Optional, Dict, Any, Iterator, Tuple
from pathlib import Path

import fitz  # PyMuPDF
import pdfplumber
import numpy as np


def extract_text_if_digital(pdf_path: str, min_chars_per_page: int = 200, max_pages: int = 10) -> Optional[Dict[str, Any]]:
    """
    If PDF has selectable text, return it (best accuracy) and skip OCR.
    """
    texts = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages[:max_pages]):
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
    Render PDF pages to RGB images (numpy arrays). (Legacy helper - loads pages into memory)
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


def render_pdf_pages(pdf_path: str, dpi: int = 300, max_pages: int = 10) -> Iterator[Tuple[int, np.ndarray]]:
    """
    Memory-safe PDF renderer.
    Yields (page_index, img_rgb) one page at a time.
    """
    doc = fitz.open(pdf_path)
    n = min(len(doc), max_pages)

    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    try:
        for i in range(n):
            page = doc.load_page(i)
            pix = page.get_pixmap(matrix=mat, alpha=False)  # RGB
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
            yield i, img
    finally:
        doc.close()


def render_pdf_page(pdf_path: str, page_index: int, dpi: int = 300) -> np.ndarray:
    """
    Render a single PDF page to an RGB numpy array.
    Useful for multiprocessing where you don't want to ship big arrays across processes.
    """
    doc = fitz.open(pdf_path)
    try:
        page = doc.load_page(page_index)
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)  # RGB
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        return img
    finally:
        doc.close()
