from __future__ import annotations

from typing import List, Tuple
import math


def poly_to_xyxy(poly) -> List[int]:
    # poly: [[x,y],[x,y],[x,y],[x,y]]
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]


def bbox_iou(a: List[int], b: List[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0, inter_x2 - inter_x1)
    ih = max(0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    denom = area_a + area_b - inter
    return (inter / denom) if denom > 0 else 0.0


def group_into_lines(words: List[dict]) -> List[List[dict]]:
    """
    words: [{text, conf, bbox=[x1,y1,x2,y2]}]
    Basic line clustering by y-center proximity.
    """
    if not words:
        return []

    # sort by y then x
    words = sorted(words, key=lambda w: ((w["bbox"][1] + w["bbox"][3]) / 2.0, w["bbox"][0]))

    lines: List[List[dict]] = []
    current: List[dict] = []
    cur_y = None

    for w in words:
        x1, y1, x2, y2 = w["bbox"]
        y_center = (y1 + y2) / 2.0
        height = max(1, y2 - y1)

        if cur_y is None:
            current = [w]
            cur_y = y_center
            continue

        # if y close enough => same line
        if abs(y_center - cur_y) <= max(10, 0.6 * height):
            current.append(w)
            # update running y
            cur_y = (cur_y * (len(current) - 1) + y_center) / len(current)
        else:
            lines.append(sorted(current, key=lambda ww: ww["bbox"][0]))
            current = [w]
            cur_y = y_center

    if current:
        lines.append(sorted(current, key=lambda ww: ww["bbox"][0]))

    return lines


def merge_line_bbox(words: List[dict]) -> List[int]:
    xs1 = [w["bbox"][0] for w in words]
    ys1 = [w["bbox"][1] for w in words]
    xs2 = [w["bbox"][2] for w in words]
    ys2 = [w["bbox"][3] for w in words]
    return [int(min(xs1)), int(min(ys1)), int(max(xs2)), int(max(ys2))]


def compute_line_conf(words: List[dict]) -> float:
    if not words:
        return 0.0
    # length-weighted average (simple)
    total = 0.0
    weight = 0.0
    for w in words:
        t = w["text"].strip()
        wgt = max(1, len(t))
        total += float(w["conf"]) * wgt
        weight += wgt
    return float(total / max(1.0, weight))


def line_text(words: List[dict]) -> str:
    # join with spaces; keep punctuation tight
    parts = []
    for w in words:
        t = w["text"].strip()
        if not t:
            continue
        parts.append(t)
    return " ".join(parts).strip()
