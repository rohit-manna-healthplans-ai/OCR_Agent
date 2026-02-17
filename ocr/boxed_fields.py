
"""
ULTRA FIXED VERSION - Fully compatible with field_extract.py

field_extract.py expects:
- Box object with attributes: x1, y1, x2, y2
- detect_char_boxes(image)
- group_boxes_into_rows(boxes)
- boxes_to_sequences(rows)

This version restores correct structure.
"""

import cv2
from dataclasses import dataclass


@dataclass
class Box:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def w(self):
        return self.x2 - self.x1

    @property
    def h(self):
        return self.y2 - self.y1


def _preprocess(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2
    )
    return thresh


def detect_char_boxes(image):
    thresh = _preprocess(image)

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        aspect_ratio = w / float(h) if h > 0 else 0

        # square-like filtering
        if 0.75 < aspect_ratio < 1.25 and 800 < area < 25000:
            boxes.append(Box(x, y, x + w, y + h))

    # sort top-to-bottom then left-to-right
    boxes = sorted(boxes, key=lambda b: (b.y1, b.x1))

    return boxes


def group_boxes_into_rows(boxes, row_threshold=20):
    rows = []
    current_row = []
    current_y = None

    for box in boxes:
        if current_y is None:
            current_y = box.y1

        if abs(box.y1 - current_y) > row_threshold:
            if current_row:
                rows.append(sorted(current_row, key=lambda b: b.x1))
            current_row = []
            current_y = box.y1

        current_row.append(box)

    if current_row:
        rows.append(sorted(current_row, key=lambda b: b.x1))

    return rows


def boxes_to_sequences(rows):
    """
    field_extract.py handles recognition itself.
    So we simply return grouped rows as-is.
    """
    return rows
