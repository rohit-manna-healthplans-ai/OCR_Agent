
import pytesseract
import cv2
import numpy as np

def run_tesseract(img_rgb: np.ndarray):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)

    words = []
    n = len(data["text"])
    for i in range(n):
        text = data["text"][i].strip()
        conf = float(data["conf"][i]) if data["conf"][i] != "-1" else 0.0
        if text:
            x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            words.append({
                "text": text,
                "conf": conf / 100.0,
                "bbox": [x, y, x + w, y + h]
            })
    return words
