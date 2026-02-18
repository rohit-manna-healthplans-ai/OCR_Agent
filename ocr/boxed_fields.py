
"""
ENTERPRISE STABLE VERSION

Fixes:
✔ Adds Box.pad() method (required by field_extract.py)
✔ Fully compatible with pipeline
✔ Reduces contour noise (speed improvement)
✔ No recognition override

field_extract.py expects:
- Box.x1, y1, x2, y2
- Box.pad(pad, W, H)
- detect_char_boxes()
- group_boxes_into_rows()
- boxes_to_sequences()
"""

import cv2
from dataclasses import dataclass


# ---------------- BOX STRUCTURE ----------------

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

    def pad(self, pad, W, H):
        """
        Expands box safely within image boundaries.
        """
        nx1 = max(0, self.x1 - pad)
        ny1 = max(0, self.y1 - pad)
        nx2 = min(W, self.x2 + pad)
        ny2 = min(H, self.y2 + pad)
        return Box(nx1, ny1, nx2, ny2)


# ---------------- PREPROCESS ----------------

def _preprocess(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Slight blur reduces noise (speed + stability)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        15,
        3
    )

    return thresh


# ---------------- DETECT BOXES ----------------

def detect_char_boxes(image):
    H, W = image.shape[:2]
    thresh = _preprocess(image)

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # Faster filtering (avoid tiny noise)
        if w < 15 or h < 15:
            continue

        area = w * h
        aspect_ratio = w / float(h) if h > 0 else 0

        # Tight square-like filter
        if 0.75 < aspect_ratio < 1.25 and 500 < area < 20000:
            boxes.append(Box(x, y, x + w, y + h))

    # Sort efficiently
    boxes.sort(key=lambda b: (b.y1, b.x1))

    return boxes


# ---------------- GROUP ROWS ----------------

def group_boxes_into_rows(boxes, row_threshold=20):
    rows = []
    current_row = []
    current_y = None

    for box in boxes:
        if current_y is None:
            current_y = box.y1

        if abs(box.y1 - current_y) > row_threshold:
            if current_row:
                current_row.sort(key=lambda b: b.x1)
                rows.append(current_row)
            current_row = []
            current_y = box.y1

        current_row.append(box)

    if current_row:
        current_row.sort(key=lambda b: b.x1)
        rows.append(current_row)

    return rows


# ---------------- PASS THROUGH ----------------

def boxes_to_sequences(rows):
    return rows
