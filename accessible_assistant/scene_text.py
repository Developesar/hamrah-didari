"""Convert detections into deterministic Persian descriptions and compact model evidence."""
from __future__ import annotations

from collections import OrderedDict

NUMBERS = {1: "یک", 2: "دو", 3: "سه", 4: "چهار", 5: "پنج", 6: "شش", 7: "هفت", 8: "هشت", 9: "نه",
           10: "ده", 11: "یازده", 12: "دوازده"}
LABELS = {"person": "انسان", "car": "خودرو", "chair": "صندلی", "door": "در", "stairs": "پله", "signboard": "تابلو"}
POSITIONS_FA = {"left": "سمت چپ تصویر", "center": "بخش میانی تصویر", "right": "سمت راست تصویر", "unknown": "تصویر"}


def position(box) -> str:
    """Horizontal third of the frame: left / center / right (frame-relative, not the user's body)."""
    try:
        x1, _, x2, _ = (float(v) for v in box)
    except (TypeError, ValueError):
        return "unknown"
    if not (0 <= x1 <= x2 <= 1):
        return "unknown"
    center = (x1 + x2) / 2
    return "left" if center < 1 / 3 else "right" if center > 2 / 3 else "center"


def size_hint(box) -> float:
    try:
        x1, y1, x2, y2 = (float(v) for v in box)
        return max(0.0, (x2 - x1) * (y2 - y1))
    except (TypeError, ValueError):
        return 0.0


# Things that matter for moving around are named first in the spoken sentence.
PRIORITY = {"در", "پله", "انسان", "شخص", "خودرو", "اتوبوس", "کامیون", "دوچرخه", "موتورسیکلت", "صندلی", "میز",
            "نیمکت", "تابلو", "تابلوی ایست", "چراغ راهنمایی", "تیر", "نرده", "سگ", "گربه"}


def describe_objects(objects, limit: int = 30, max_groups: int = 6) -> str:
    """«دو صندلی در سمت راست تصویر و یک انسان در بخش میانی تصویر دیده می‌شود.»
    Mobility-relevant things first, then the biggest; at most max_groups phrases, the rest counted."""
    groups: "OrderedDict[tuple, int]" = OrderedDict()
    def name_of(obj):
        return LABELS.get(obj.get("target_class")) or obj.get("label") or "شیء"
    ordered = sorted(objects, key=lambda o: (name_of(o) not in PRIORITY, -size_hint(o.get("bbox_xyxy_normalized"))))[:limit]
    for obj in ordered:
        key = (name_of(obj), position(obj.get("bbox_xyxy_normalized")))
        groups[key] = groups.get(key, 0) + 1
    items = list(groups.items())
    shown, rest = items[:max_groups], sum(n for _, n in items[max_groups:])
    parts = [f"{NUMBERS.get(n, str(n))} {c} در {POSITIONS_FA[p]}" for (c, p), n in shown]
    if not parts:
        return ""
    sentence = parts[0] if len(parts) == 1 else "، ".join(parts[:-1]) + " و " + parts[-1]
    sentence += " دیده می‌شود."
    if rest:
        sentence += f" {NUMBERS.get(rest, str(rest))} مورد دیگر هم در فهرست روی صفحه آمده است."
    return sentence


def vlm_evidence(objects, limit: int = 10) -> list[dict]:
    """Compact, English-keyed facts for the VLM prompt (still marked as untrusted data there)."""
    return [{"object": o.get("target_class"), "position_in_frame": position(o.get("bbox_xyxy_normalized")),
             "relative_size": round(size_hint(o.get("bbox_xyxy_normalized")), 3),
             "detector_score": round(float(o.get("score", 0)), 2)}
            for o in sorted(objects, key=lambda o: -float(o.get("score", 0)))[:limit]]


# --- Approximate distance -------------------------------------------
# Pinhole camera: distance ~= real_height * focal_px / box_height_px. Typical heights are
# assumptions, phones differ, and a single photo has no depth, so only a coarse range for the
# nearest object is ever spoken, always marked as an estimate (frame-relative estimates).
TYPICAL_HEIGHT_M = {"person": 1.7, "car": 1.5, "chair": 0.9, "door": 2.0}
DEFAULT_FOCAL_35MM = 26.0  # common main phone camera when EXIF lacks it
RANGES = [(1.5, "حدود یک تا دو متر"), (3.0, "حدود دو تا سه متر"), (6.0, "حدود سه تا شش متر"),
          (float("inf"), "بیش از شش متر")]


def estimate_distance_m(obj, image_size, focal_35mm=None):
    """Metres or None (unknown class, cut-off box, bad geometry)."""
    height_m = TYPICAL_HEIGHT_M.get(obj.get("target_class"))
    try:
        x1, y1, x2, y2 = (float(v) for v in obj.get("bbox_xyxy_normalized"))
        width_px, height_px = (int(v) for v in image_size)
    except (TypeError, ValueError):
        return None
    if not height_m or y2 <= y1 or width_px <= 0 or height_px <= 0:
        return None
    if y1 <= 0.01 or y2 >= 0.99:  # cut off at the frame edge: true height unknown
        return None
    focal_px = (focal_35mm or DEFAULT_FOCAL_35MM) * max(width_px, height_px) / 36.0
    return height_m * focal_px / ((y2 - y1) * height_px)


def nearest_distance_sentence(objects, image_size, focal_35mm=None) -> str:
    estimates = [(d, o) for o in objects for d in [estimate_distance_m(o, image_size, focal_35mm)] if d]
    if not estimates:
        return ""
    distance, obj = min(estimates, key=lambda item: item[0])
    phrase = next(text for limit, text in RANGES if distance < limit)
    return (f"نزدیک‌ترین مورد، {LABELS.get(obj.get('target_class'), 'شیء')}، تخمینی {phrase} از دوربین است؛ "
            "این عدد از روی اندازهٔ تصویر حدس زده شده و دقیق نیست.")
