"""Deterministic descriptions and coarse geometry checks."""
import unittest
from accessible_assistant.scene_text import describe_objects, position, vlm_evidence

CHAIR_R = {"target_class": "chair", "bbox_xyxy_normalized": [0.7, 0.3, 0.95, 0.9], "score": 0.9}
CHAIR_R2 = {"target_class": "chair", "bbox_xyxy_normalized": [0.75, 0.4, 0.9, 0.8], "score": 0.8}
PERSON_C = {"target_class": "person", "bbox_xyxy_normalized": [0.4, 0.1, 0.6, 0.95], "score": 0.97}


class SceneTextTests(unittest.TestCase):
    def test_sentence_groups_counts_and_positions(self):
        self.assertEqual(describe_objects([CHAIR_R, PERSON_C, CHAIR_R2]),
                         "یک انسان در بخش میانی تصویر و دو صندلی در سمت راست تصویر دیده می‌شود.")
        self.assertEqual(describe_objects([]), "")
        self.assertEqual(position([0.0, 0, 0.2, 1]), "left")
        self.assertEqual(position(None), "unknown")

    def test_mobility_things_first_and_long_lists_are_cut(self):
        rug = {"label": "فرش", "bbox_xyxy_normalized": [0.0, 0.5, 1.0, 1.0]}
        door = {"label": "در", "bbox_xyxy_normalized": [0.45, 0.1, 0.55, 0.9]}
        small = [{"label": f"شیء{i}", "bbox_xyxy_normalized": [0.0, 0.0, 0.1, 0.1]} for i in range(7)]
        text = describe_objects([rug, door] + small)
        self.assertTrue(text.startswith("یک در در بخش میانی تصویر، یک فرش"))
        self.assertIn("سه مورد دیگر", text)

    def test_evidence_sorted_by_score(self):
        evidence = vlm_evidence([CHAIR_R, PERSON_C])
        self.assertEqual(evidence[0]["object"], "person")
        self.assertEqual(evidence[1]["position_in_frame"], "right")


class DistanceTests(unittest.TestCase):
    def test_pinhole_estimate_scales_and_skips_cut_off_boxes(self):
        from accessible_assistant.scene_text import estimate_distance_m, nearest_distance_sentence
        near = {"target_class": "person", "bbox_xyxy_normalized": [0.4, 0.2, 0.6, 0.8]}
        far = {"target_class": "person", "bbox_xyxy_normalized": [0.4, 0.2, 0.6, 0.35]}
        self.assertAlmostEqual(estimate_distance_m(near, (3000, 4000)), 2.05, places=1)
        self.assertGreater(estimate_distance_m(far, (3000, 4000)), estimate_distance_m(near, (3000, 4000)))
        self.assertIsNone(estimate_distance_m({"target_class": "person", "bbox_xyxy_normalized": [0, 0, 1, 1]}, (100, 100)))
        self.assertIsNone(estimate_distance_m({"target_class": "stairs", "bbox_xyxy_normalized": [0.1, 0.2, 0.3, 0.5]}, (100, 100)))
        sentence = nearest_distance_sentence([near, far], (3000, 4000))
        self.assertIn("تخمینی", sentence)
        self.assertIn("دقیق نیست", sentence)

