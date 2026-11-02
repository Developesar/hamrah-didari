"""Frame-relative IoU tracking and box-growth trends. A visual approach trend is not a measured speed or a safety judgement."""
from __future__ import annotations


def iou(a, b) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class IoUTracker:
    def __init__(self, iou_threshold: float = 0.3, max_missed: int = 2):
        self.iou_threshold, self.max_missed = iou_threshold, max_missed
        self.tracks, self._next = {}, 1

    def update(self, frame_index: int, detections) -> dict:
        """detections: [{"target_class", "bbox_xyxy_normalized", "score"}]; returns {track_id: detection}."""
        assigned, used = {}, set()
        pairs = sorted(((iou(t["box"], d["bbox_xyxy_normalized"]), tid, i)
                        for tid, t in self.tracks.items() for i, d in enumerate(detections)
                        if t["class"] == d.get("target_class")), reverse=True)
        for overlap, tid, i in pairs:
            if overlap < self.iou_threshold or tid in assigned or i in used:
                continue
            assigned[tid], used = detections[i], used | {i}
        for i, d in enumerate(detections):
            if i not in used:
                tid, self._next = self._next, self._next + 1
                self.tracks[tid] = {"class": d.get("target_class"), "history": [], "missed": 0}
                assigned[tid] = d
        for tid in list(self.tracks):
            track = self.tracks[tid]
            if tid in assigned:
                box = assigned[tid]["bbox_xyxy_normalized"]
                track.update(box=box, missed=0)
                track["history"].append((frame_index, box))
            else:
                track["missed"] += 1
                if track["missed"] > self.max_missed:
                    del self.tracks[tid]
        return assigned

    @staticmethod
    def trend(history) -> str:
        """'approaching' / 'moving_away' / 'steady' from first vs last box height (needs >= 3 points)."""
        if len(history) < 3:
            return "unknown"
        first, last = history[0][1], history[-1][1]
        h0, h1 = first[3] - first[1], last[3] - last[1]
        if h0 <= 0:
            return "unknown"
        ratio = h1 / h0
        return "approaching" if ratio > 1.25 else "moving_away" if ratio < 0.8 else "steady"


def summarize(all_tracks) -> list[dict]:
    from accessible_assistant.scene_text import position
    rows = []
    for tid, track in all_tracks.items():
        if len(track["history"]) < 2:
            continue
        from accessible_assistant.sequence_model import predict
        learned = predict([box for _, box in track["history"]])
        rows.append({"track_id": tid, "class": track["class"], "frames": len(track["history"]),
                     "trend": learned or IoUTracker.trend(track["history"]),
                     "trend_source": "gru" if learned else "height_rule",
                     "position": position(track["history"][-1][1])})
    return rows


TREND_FA = {"approaching": "در حال نزدیک شدن به دوربین است", "moving_away": "در حال دور شدن از دوربین است",
            "steady": "تقریباً در همان فاصله مانده است", "unknown": "دیده شد"}


def describe_tracks(rows) -> str:
    from accessible_assistant.scene_text import LABELS, POSITIONS_FA
    if not rows:
        return "در این ویدئو، شیء پشتیبانی‌شده‌ای به‌طور پیوسته دیده نشد؛ این نتیجه نبود مانع را ثابت نمی‌کند."
    order = {"approaching": 0, "moving_away": 1, "steady": 2, "unknown": 3}
    parts = [f"یک {LABELS.get(r['class'], 'شیء')} در {POSITIONS_FA.get(r['position'], 'تصویر')} {TREND_FA[r['trend']]}"
             for r in sorted(rows, key=lambda r: order[r["trend"]])[:5]]
    return "؛ ".join(parts) + "."
