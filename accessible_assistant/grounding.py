"""Strict parser for Qwen3-VL 2D-grounding text. Model output is untrusted data.

Qwen3-VL returns boxes as JSON objects with "bbox_2d": [x1, y1, x2, y2] in
relative coordinates 0..1000 and a free-text "label". Anything that is not a
well-formed box with a known label is rejected, never guessed. A generative
model gives no score, so results are not calibrated and absence proves nothing.
"""
from __future__ import annotations

import json
import math
import re

from accessible_assistant.contracts import TARGET_CLASSES

# Fixed English vocabulary the evaluation prompt asks for -> project classes.
LABEL_MAP = {
    "person": "person", "people": "person", "man": "person", "woman": "person",
    "car": "car", "chair": "chair",
    "door": "door", "doors": "door", "doorway": "door",
    "stairs": "stairs", "stair": "stairs", "staircase": "stairs", "steps": "stairs",
    "signboard": "signboard", "sign": "signboard", "traffic sign": "signboard",
    "road sign": "signboard", "street sign": "signboard",
}
MAX_TEXT_CHARS = 20_000
MAX_OBJECTS = 50
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _reject_constant(value):
    raise ValueError("non-finite JSON constant: " + value)


def _box(raw):
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in raw):
        return None
    x1, y1, x2, y2 = (float(v) for v in raw)
    if not all(math.isfinite(v) and 0 <= v <= 1000 for v in (x1, y1, x2, y2)):
        return None
    if x1 >= x2 or y1 >= y2:
        return None
    return [x1 / 1000, y1 / 1000, x2 / 1000, y2 / 1000]


def parse_qwen_grounding(text, allowed_classes=TARGET_CLASSES) -> dict:
    """Return {"objects": [...], "rejected": int, "error": str | None}.

    allowed_classes=None means open vocabulary: any non-empty label (kept as written) is accepted.

    error is one of None, "invalid_text", "no_json", "invalid_json", "not_list",
    "too_many". On any error, objects is empty.
    """
    out = {"objects": [], "rejected": 0, "error": None}
    if not isinstance(text, str) or not text.strip() or len(text) > MAX_TEXT_CHARS:
        out["error"] = "invalid_text"
        return out
    fenced = _FENCE.search(text)
    body = fenced.group(1) if fenced else text
    start, end = body.find("["), body.rfind("]")
    if start < 0 or end <= start:
        out["error"] = "no_json"
        return out
    try:
        items = json.loads(body[start:end + 1], parse_constant=_reject_constant)
    except (ValueError, RecursionError):
        out["error"] = "invalid_json"
        return out
    if not isinstance(items, list):
        out["error"] = "not_list"
        return out
    if len(items) > MAX_OBJECTS:
        out["error"] = "too_many"
        return out
    open_vocabulary = allowed_classes is None
    allowed = set(allowed_classes or ())
    for item in items:
        label = item.get("label") if isinstance(item, dict) else None
        clean = label.strip()[:60] if isinstance(label, str) else ""
        target = LABEL_MAP.get(clean.lower()) if clean else None
        box = _box(item.get("bbox_2d")) if isinstance(item, dict) else None
        if box is None or not clean or (not open_vocabulary and (target is None or target not in allowed)):
            out["rejected"] += 1
            continue
        out["objects"].append({
            "observation_id": f"vlm-object-{len(out['objects']) + 1}",
            "target_class": target or "open_vocabulary", "label": clean, "source_label": clean,
            "bbox_xyxy_normalized": box, "score": None,
            "score_kind": "none_generative_model_unverified", "source": "vision_language_grounding",
        })
    return out
