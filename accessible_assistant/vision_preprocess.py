"""Image-quality assessment, contrast enhancement and OCR preparation with OpenCV. All processing is local and independent of model inference."""
from __future__ import annotations

import math

import numpy as np

# Thresholds are starting points for 1-12 MP phone photos, not calibrated probabilities.
BLUR_VAR_MIN = 60.0       # variance of Laplacian on a 1024px-long-side gray image
DARK_MEAN_MAX = 60.0      # mean of HSV value channel (0-255)
BRIGHT_MEAN_MIN = 215.0
LOW_CONTRAST_STD = 25.0


def _cv2():
    import cv2
    return cv2


def to_bgr(image) -> np.ndarray:
    """PIL RGB image or ndarray -> BGR uint8 ndarray."""
    array = np.asarray(image.convert("RGB")) if hasattr(image, "convert") else np.asarray(image)
    if array.ndim == 2:
        return _cv2().cvtColor(array, _cv2().COLOR_GRAY2BGR)
    return np.ascontiguousarray(array[:, :, ::-1])


def to_pil(bgr: np.ndarray):
    from PIL import Image
    return Image.fromarray(np.ascontiguousarray(bgr[:, :, ::-1]))


def _resize_long_side(img: np.ndarray, size: int = 1024) -> np.ndarray:
    cv2 = _cv2()
    h, w = img.shape[:2]
    scale = size / max(h, w)
    return img if scale >= 1 else cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def assess_quality(bgr: np.ndarray) -> dict:
    """Blur (Laplacian variance), brightness (HSV V mean), contrast (gray std), edge density (Canny)."""
    cv2 = _cv2()
    small = _resize_long_side(bgr)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(cv2.cvtColor(small, cv2.COLOR_BGR2HSV)[:, :, 2].mean())
    contrast = float(gray.std())
    clipped = float((gray >= 250).mean())  # share of blown-out pixels
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float((edges > 0).mean())
    issues = []
    if blur_var < BLUR_VAR_MIN:
        issues.append("blurry")
    if brightness < DARK_MEAN_MAX:
        issues.append("dark")
    elif brightness > BRIGHT_MEAN_MIN and clipped > 0.5 and contrast < 45:
        # A bright page with dark text is normal; only flag washed-out, low-detail photos.
        issues.append("overexposed")
    if contrast < LOW_CONTRAST_STD:
        issues.append("low_contrast")
    advice = {
        "blurry": "عکس تار است؛ گوشی را چند لحظه ثابت نگه دارید و دوباره عکس بگیرید.",
        "dark": "عکس تاریک است؛ اگر ممکن است نور را بیشتر کنید یا چراغ گوشی را روشن کنید.",
        "overexposed": "عکس بیش از حد روشن است؛ زاویهٔ گوشی را کمی از منبع نور دور کنید.",
        "low_contrast": "تفاوت رنگ‌ها در عکس کم است؛ ممکن است جزئیات دقیق تشخیص داده نشوند.",
    }
    return {"blur_variance": round(blur_var, 1), "brightness": round(brightness, 1),
            "contrast": round(contrast, 1), "edge_density": round(edge_density, 4),
            "issues": issues, "advice_fa": [advice[i] for i in issues],
            "usable": "blurry" not in issues}


def gamma_correct(bgr: np.ndarray, gamma: float) -> np.ndarray:
    table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)], dtype=np.uint8)
    return _cv2().LUT(bgr, table)


def clahe_color(bgr: np.ndarray, clip: float = 2.0) -> np.ndarray:
    """Local contrast on the L channel of LAB, colours kept."""
    cv2 = _cv2()
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def enhance(bgr: np.ndarray, quality: dict | None = None) -> tuple[np.ndarray, list[str]]:
    """Camera clean-up chosen from the measured issues; returns image and applied steps."""
    cv2 = _cv2()
    quality = quality or assess_quality(bgr)
    steps, out = [], bgr
    out = cv2.fastNlMeansDenoisingColored(out, None, 5, 5, 7, 21)
    steps.append("denoise_nlmeans")
    if "dark" in quality["issues"]:
        out = gamma_correct(out, 1.8)
        steps.append("gamma_1.8")
    elif "overexposed" in quality["issues"]:
        out = gamma_correct(out, 0.7)
        steps.append("gamma_0.7")
    if {"dark", "low_contrast"} & set(quality["issues"]) or quality["contrast"] < 45:
        out = clahe_color(out)
        steps.append("clahe_lab")
    return out, steps


def remove_shadow(gray: np.ndarray) -> np.ndarray:
    """Estimate background by dilation + median blur, divide it out (uneven light/shadow)."""
    cv2 = _cv2()
    background = cv2.medianBlur(cv2.dilate(gray, np.ones((7, 7), np.uint8)), 21)
    diff = 255 - cv2.absdiff(gray, background)
    return cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)


def deskew_angle(gray: np.ndarray) -> float:
    """Dominant text angle from dark-pixel cloud (minAreaRect); 0 when unclear."""
    cv2 = _cv2()
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    points = cv2.findNonZero(binary)
    if points is None or len(points) < 50:
        return 0.0
    angle = cv2.minAreaRect(points)[-1]
    angle = angle - 90 if angle > 45 else angle
    return float(angle) if abs(angle) <= 15 else 0.0


def rotate(img: np.ndarray, angle: float) -> np.ndarray:
    cv2 = _cv2()
    h, w = img.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def prepare_for_ocr(bgr: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Gray -> shadow removal -> CLAHE -> deskew -> upscale small text -> light morphology."""
    cv2 = _cv2()
    steps = []
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = remove_shadow(gray)
    steps.append("shadow_removal")
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    steps.append("clahe")
    angle = deskew_angle(gray)
    if abs(angle) >= 0.5:
        gray = rotate(gray, angle)
        steps.append(f"deskew_{angle:.1f}")
    if max(gray.shape) < 1200:
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        steps.append("upscale_2x")
    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
    steps.append("morph_close_2")
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), steps


def text_regions(bgr: np.ndarray, max_regions: int = 40) -> list[dict]:
    """Classic text-region proposal: morphological gradient -> Otsu -> horizontal closing -> contours.
    Returns normalized boxes; a proposal, not proof that text exists."""
    cv2 = _cv2()
    small = _resize_long_side(bgr)
    h, w = small.shape[:2]
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    _, binary = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    joined = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3)))
    contours, _ = cv2.findContours(joined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions = []
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        if bw < 20 or bh < 8 or bw / max(bh, 1) < 1.5 or bw * bh > 0.5 * w * h:
            continue
        fill = cv2.countNonZero(binary[y:y + bh, x:x + bw]) / float(bw * bh)
        if fill < 0.25:
            continue
        regions.append({"box_xyxy_normalized": [x / w, y / h, (x + bw) / w, (y + bh) / h],
                        "fill_ratio": round(fill, 3)})
    regions.sort(key=lambda r: (r["box_xyxy_normalized"][1], -r["box_xyxy_normalized"][2]))
    return regions[:max_regions]
