"""Object foreground masks and text-component segmentation with OpenCV."""
from __future__ import annotations

import numpy as np


def object_mask(bgr: np.ndarray, box_normalized, iterations: int = 3) -> dict:
    """GrabCut initialised with the detector box; returns the foreground share of the box and its outline."""
    import cv2
    h, w = bgr.shape[:2]
    x1, y1, x2, y2 = (int(round(v * s)) for v, s in zip(box_normalized, (w, h, w, h)))
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return {"ok": False}
    mask = np.zeros((h, w), np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(bgr, mask, (x1, y1, x2 - x1, y2 - y1), bgd, fgd, iterations, cv2.GC_INIT_WITH_RECT)
    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    box_area = float((x2 - x1) * (y2 - y1))
    largest = max(contours, key=cv2.contourArea) if contours else None
    polygon = []
    if largest is not None:
        approx = cv2.approxPolyDP(largest, 0.01 * cv2.arcLength(largest, True), True)
        polygon = [[round(float(p[0][0]) / w, 4), round(float(p[0][1]) / h, 4)] for p in approx]
    return {"ok": True, "foreground_ratio": round(float((fg[y1:y2, x1:x2] > 0).sum()) / box_area, 3),
            "polygon_normalized": polygon}


def segment_objects(bgr: np.ndarray, objects, limit: int = 2, max_side: int = 320) -> list[dict]:
    """GrabCut outlines for the largest detections on a downscaled copy (keeps latency small)."""
    import cv2
    h, w = bgr.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    small = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else bgr
    def area(o):
        x1, y1, x2, y2 = o["bbox_xyxy_normalized"]
        return (x2 - x1) * (y2 - y1)
    out = []
    for obj in sorted(objects, key=area, reverse=True)[:limit]:
        result = object_mask(small, obj["bbox_xyxy_normalized"], iterations=2)
        if result.get("ok"):
            out.append({"observation_id": obj.get("observation_id"), "target_class": obj.get("target_class"),
                        "foreground_ratio": result["foreground_ratio"], "polygon_normalized": result["polygon_normalized"]})
    return out


def split_characters(gray_region: np.ndarray, min_area: int = 12) -> dict:
    """Binarise (Otsu, dark text) and label connected components; merge close blobs into words."""
    import cv2
    _, binary = cv2.threshold(gray_region, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    chars = [tuple(int(v) for v in stats[i, :4]) for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    chars.sort(key=lambda b: b[0])
    # Word break = a gap clearly wider than the typical gap between letters of the same word.
    gaps = [b[0] - (a[0] + a[2]) for a, b in zip(chars, chars[1:])]
    gap = max(3, int(np.median(gaps) * 2.5) + 1) if gaps else 4
    words = []
    for x, y, bw, bh in chars:
        if words and x - (words[-1][0] + words[-1][2]) <= gap:
            wx, wy, ww, wh = words[-1]
            nx, ny = min(wx, x), min(wy, y)
            words[-1] = (nx, ny, max(wx + ww, x + bw) - nx, max(wy + wh, y + bh) - ny)
        else:
            words.append((x, y, bw, bh))
    return {"characters": len(chars), "words": len(words), "word_boxes": words}
