from __future__ import annotations

from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import cv2


# Cached objects (creation is relatively expensive in OpenCV)
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
_SHARPEN_KERNEL = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]], dtype=np.float32)


def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def _clahe(gray: np.ndarray) -> np.ndarray:
    return _CLAHE.apply(gray)


def _adaptive_thresh(gray: np.ndarray) -> np.ndarray:
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
    )


def _denoise(gray: np.ndarray) -> np.ndarray:
    return cv2.medianBlur(gray, 3)


def _sharpen(gray: np.ndarray) -> np.ndarray:
    return cv2.filter2D(gray, -1, _SHARPEN_KERNEL)


def _estimate_skew_angle_small(bin_img: np.ndarray) -> float:
    """
    Estimate small skew angle only.
    On forms (many boxes/lines), minAreaRect can return ~90deg and ruin the image.
    We only accept small angles (e.g., within +/- 12 degrees).
    """
    coords = np.column_stack(np.where(bin_img < 128))
    if coords.shape[0] < 1500:
        return 0.0

    rect = cv2.minAreaRect(coords)
    angle = float(rect[-1])  # [-90, 0)
    if angle < -45:
        angle = 90 + angle  # to [0, 45)
    if abs(angle) > 12:
        return 0.0
    return angle


def _rotate(img: np.ndarray, angle: float) -> np.ndarray:
    if abs(angle) < 0.05:
        return img
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    return rotated


def _upscale_rgb(img_rgb: np.ndarray, scale: float) -> np.ndarray:
    h, w = img_rgb.shape[:2]
    return cv2.resize(img_rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)


def preprocess(img_rgb: np.ndarray, preset: str = "auto") -> Tuple[np.ndarray, str]:
    """
    Returns processed RGB image (still RGB), and the preset used.

    Key fix:
    - deskew is clamped to small angles to avoid 90-degree rotations on forms
    - mild upscale added for boxed characters
    """
    preset = (preset or "auto").lower().strip()
    if preset not in {"auto", "clean_doc", "photo", "low_light"}:
        preset = "auto"

    gray = _to_gray(img_rgb)
    used = preset

    if preset == "auto":
        mean = float(gray.mean())
        std = float(gray.std())
        if mean < 110:
            used = "low_light"
        elif std > 75:
            used = "photo"
        else:
            used = "clean_doc"

    if used == "clean_doc":
        g = _denoise(gray)
        g = _clahe(g)
        b = _adaptive_thresh(g)
        angle = _estimate_skew_angle_small(b)
        out = _rotate(img_rgb, angle)

        h, w = out.shape[:2]
        if max(h, w) < 2200:
            out = _upscale_rgb(out, 1.5)

        return out, used

    if used == "photo":
        g = _denoise(gray)
        g = _clahe(g)
        g = _sharpen(g)
        h, w = img_rgb.shape[:2]
        scale = 1.6 if max(h, w) < 1800 else 1.25
        resized = _upscale_rgb(img_rgb, scale)

        g2 = _to_gray(resized)
        b = _adaptive_thresh(_clahe(_denoise(g2)))
        angle = _estimate_skew_angle_small(b)
        out = _rotate(resized, angle)
        return out, used

    g = _clahe(_denoise(gray))
    b = _adaptive_thresh(g)
    angle = _estimate_skew_angle_small(b)
    out = _rotate(img_rgb, angle)
    h, w = out.shape[:2]
    scale = 1.7 if max(h, w) < 1800 else 1.3
    out = _upscale_rgb(out, scale)
    return out, used


# ------------------------------------------------------------
# ENTERPRISE SCREEN OCR: Region Segmentation (no cropping away)
# ------------------------------------------------------------

def _downscale_for_scan(img_rgb: np.ndarray, target_w: int = 900) -> Tuple[np.ndarray, float]:
    h, w = img_rgb.shape[:2]
    if w <= target_w:
        return img_rgb, 1.0
    scale = target_w / float(w)
    small = cv2.resize(img_rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return small, scale


def is_probable_screenshot(img_rgb: np.ndarray) -> bool:
    """
    Heuristic: screenshot-like images typically have wide resolution and UI chrome bands.
    We DO NOT rely on this exclusively; we also validate segmentation quality.
    """
    h, w = img_rgb.shape[:2]
    if w < 900 or h < 600:
        return False
    aspect = w / float(h)
    return 1.15 <= aspect <= 2.2


def segment_screen_regions(img_rgb: np.ndarray) -> List[Dict[str, Any]]:
    """
    Returns regions with labels and bbox in ORIGINAL image coords.
    Labels: topbar, sidebar, main, footer
    This does NOT remove anything; it enables per-region OCR + structured output.
    """
    h, w = img_rgb.shape[:2]
    small, scale = _downscale_for_scan(img_rgb, target_w=900)
    hs, ws = small.shape[:2]

    # Basic ratio guesses (work well for webapp screenshots)
    top_h = int(hs * 0.12)
    bot_h = int(hs * 0.10)
    left_w = int(ws * 0.18)

    # Refine sidebar detection: if left strip is not distinct, reduce to 0
    left_strip = small[:, :left_w]
    center_strip = small[:, left_w:int(ws * 0.85)]
    try:
        diff = float(np.abs(left_strip.mean(axis=(0, 1)) - center_strip.mean(axis=(0, 1))).mean())
    except Exception:
        diff = 0.0
    if diff < 8.0:
        # Probably no strong sidebar
        left_w = int(ws * 0.12)

    # Validate top/footer bands presence (dark/colored uniformity)
    def band_score(band: np.ndarray) -> float:
        # Lower std -> more uniform band (typical for toolbars)
        g = cv2.cvtColor(band, cv2.COLOR_RGB2GRAY)
        return float(g.std())

    top_band = small[:top_h, :]
    bot_band = small[hs - bot_h:, :]
    top_std = band_score(top_band)
    bot_std = band_score(bot_band)

    # If band is very noisy, shrink band a bit
    if top_std > 55:
        top_h = int(hs * 0.09)
    if bot_std > 55:
        bot_h = int(hs * 0.08)

    # Regions in small coords
    regions_small = [
        {"name": "topbar", "bbox": [0, 0, ws, max(1, top_h)]},
        {"name": "footer", "bbox": [0, max(0, hs - bot_h), ws, hs]},
        {"name": "sidebar", "bbox": [0, top_h, max(1, left_w), max(top_h + 1, hs - bot_h)]},
        {"name": "main", "bbox": [left_w, top_h, ws, max(top_h + 1, hs - bot_h)]},
    ]

    # Scale back to original coords
    inv = 1.0 / float(scale)
    out_regions: List[Dict[str, Any]] = []
    for r in regions_small:
        x1, y1, x2, y2 = r["bbox"]
        bbox = [int(x1 * inv), int(y1 * inv), int(x2 * inv), int(y2 * inv)]
        bbox = [max(0, bbox[0]), max(0, bbox[1]), min(w, bbox[2]), min(h, bbox[3])]
        if bbox[2] - bbox[0] < 10 or bbox[3] - bbox[1] < 10:
            continue
        out_regions.append({"name": r["name"], "bbox": bbox})

    return out_regions


def crop_region(img_rgb: np.ndarray, bbox: List[int]) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    return img_rgb[y1:y2, x1:x2].copy()
