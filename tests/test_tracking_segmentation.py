"""Tracking and segmentation checks on synthetic inputs."""
import unittest
import numpy as np
from accessible_assistant.tracking import IoUTracker, describe_tracks, summarize
try:
    import cv2
except ImportError:
    cv2 = None

class TrackingTests(unittest.TestCase):
    def test_growing_box_is_approaching_and_ids_are_stable(self):
        tracker = IoUTracker()
        for frame in range(6):
            h = 0.2 + 0.08 * frame
            tracker.update(frame, [
                {"target_class": "person", "bbox_xyxy_normalized": [0.6, 0.5 - h / 2, 0.7, 0.5 + h / 2]},
                {"target_class": "car", "bbox_xyxy_normalized": [0.1, 0.4, 0.3, 0.6]}])
        rows = {r["class"]: r for r in summarize(tracker.tracks)}
        self.assertEqual(rows["person"]["trend"], "approaching")
        self.assertEqual(rows["car"]["trend"], "steady")
        self.assertEqual(len(tracker.tracks), 2)
        self.assertIn("نزدیک شدن", describe_tracks(list(rows.values())))

    def test_missed_tracks_expire(self):
        tracker = IoUTracker(max_missed=1)
        tracker.update(0, [{"target_class": "car", "bbox_xyxy_normalized": [0.1, 0.1, 0.2, 0.2]}])
        tracker.update(1, [])
        tracker.update(2, [])
        self.assertEqual(tracker.tracks, {})


@unittest.skipIf(cv2 is None, "OpenCV not installed")
class SegmentationTests(unittest.TestCase):
    def test_grabcut_finds_object_inside_box(self):
        from accessible_assistant.segmentation import object_mask
        img = np.full((200, 200, 3), (200, 220, 240), np.uint8)
        cv2.circle(img, (100, 100), 45, (20, 40, 160), -1)
        result = object_mask(img, [0.15, 0.15, 0.85, 0.85])
        self.assertTrue(result["ok"])
        self.assertTrue(0.25 < result["foreground_ratio"] < 0.85)

    def test_characters_and_words_are_split(self):
        from accessible_assistant.segmentation import split_characters
        gray = np.full((60, 400), 255, np.uint8)
        cv2.putText(gray, "AB CD EF", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 0, 3)
        result = split_characters(gray)
        self.assertGreaterEqual(result["characters"], 6)
        self.assertEqual(result["words"], 3)

