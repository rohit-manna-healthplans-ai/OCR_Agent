from __future__ import annotations

from typing import Tuple
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
    # Convert to a conventional deskew angle: negative means rotate clockwise
    # minAreaRect gives the angle of the rectangle; for text we keep it as-is.
    # Accept only small tilts
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
        # Forms often have lots of lines/boxes -> treat as clean_doc
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

        # upscale for small boxed handwriting
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

    # low_light
    g = _clahe(_denoise(gray))
    b = _adaptive_thresh(g)
    angle = _estimate_skew_angle_small(b)
    out = _rotate(img_rgb, angle)
    h, w = out.shape[:2]
    scale = 1.7 if max(h, w) < 1800 else 1.3
    out = _upscale_rgb(out, scale)
    return out, used
