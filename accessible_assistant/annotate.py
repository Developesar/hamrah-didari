"""Draw detector boxes, OCR polygons and text-region proposals on the request image (OpenCV).

Labels on the image are short Latin class names with the uncalibrated score (e.g. "chair 0.96",
as in the project's design image); the accessible Persian list is rendered as text by the UI.
"""
from __future__ import annotations

import numpy as np

COLORS = {"person": (60, 180, 75), "car": (0, 130, 200), "chair": (230, 25, 75), "door": (245, 130, 48),
          "stairs": (145, 30, 180), "signboard": (240, 50, 230)}
OCR_COLOR = (0, 200, 255)
REGION_COLOR = (180, 180, 180)


def _px(value, size):
    return int(round(min(1.0, max(0.0, float(value))) * size))


def draw(bgr: np.ndarray, objects=(), ocr_regions=(), text_regions=(), segments=()) -> np.ndarray:
    import cv2
    out = bgr.copy()
    h, w = out.shape[:2]
    thickness = max(2, round(max(h, w) / 400))
    font_scale = max(0.5, max(h, w) / 1200)
    for segment in segments:  # GrabCut outline, drawn under the boxes
        points = np.array([[_px(x, w), _px(y, h)] for x, y in segment.get("polygon_normalized", [])], np.int32)
        if len(points) >= 3:
            overlay = out.copy()
            cv2.fillPoly(overlay, [points], COLORS.get(segment.get("target_class"), (255, 255, 255)))
            out = cv2.addWeighted(overlay, 0.25, out, 0.75, 0)
    for region in text_regions:
        x1, y1, x2, y2 = region["box_xyxy_normalized"]
        cv2.rectangle(out, (_px(x1, w), _px(y1, h)), (_px(x2, w), _px(y2, h)), REGION_COLOR, 1)
    for region in ocr_regions:
        points = np.array([[_px(x, w), _px(y, h)] for x, y in region.get("polygon_normalized", [])], np.int32)
        if len(points) >= 3:
            cv2.polylines(out, [points], True, OCR_COLOR, thickness)
    for obj in objects:
        box = obj.get("bbox_xyxy_normalized")
        if not box or len(box) != 4:
            continue
        x1, y1, x2, y2 = (_px(box[0], w), _px(box[1], h), _px(box[2], w), _px(box[3], h))
        color = COLORS.get(obj.get("target_class"), (255, 255, 255))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        score = obj.get("score")
        if obj.get("source") in {"vision_language_grounding", "open_vocabulary_detector"}:
            label = "#" + str(obj.get("display_index", "?"))  # Persian names are listed in the UI table
        else:
            label = obj.get("target_class", "object") + (f" {score:.2f}" if isinstance(score, (int, float)) else "")
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        top = max(0, y1 - th - base - 4)
        cv2.rectangle(out, (x1, top), (x1 + tw + 6, top + th + base + 4), color, -1)
        cv2.putText(out, label, (x1 + 3, top + th + 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (255, 255, 255), thickness, cv2.LINE_AA)
    return out


def from_report(image, report: dict):
    """PIL image + pipeline report -> annotated PIL image, or None when there is nothing to draw."""
    from accessible_assistant.vision_preprocess import to_bgr, to_pil
    components = report.get("components", {}) if isinstance(report, dict) else {}
    objects = list(components.get("detector", {}).get("data", {}).get("objects", []))
    for index, obj in enumerate(report.get("open_vocabulary_objects", []) if isinstance(report, dict) else [], 1):
        objects.append({**obj, "display_index": index})
    ocr = components.get("ocr", {}).get("data", {}).get("regions", [])
    regions = report.get("text_regions", []) if isinstance(report, dict) else []
    segments = report.get("segments", []) if isinstance(report, dict) else []
    if not (objects or ocr or regions or segments):
        return None
    return to_pil(draw(to_bgr(image), objects, ocr, regions, segments))
