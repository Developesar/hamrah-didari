"""Parser checks with synthetic text only; no model runs and no accuracy claim."""
import unittest

from accessible_assistant.grounding import parse_qwen_grounding


class GroundingParseTests(unittest.TestCase):
    def test_fenced_json_is_normalized_and_mapped(self):
        text = 'Here:\n```json\n[{"bbox_2d": [100, 200, 300, 900], "label": "Door"},\n' \
               ' {"bbox_2d": [0, 500, 1000, 1000], "label": "staircase"}]\n```'
        result = parse_qwen_grounding(text)
        self.assertIsNone(result["error"])
        self.assertEqual([o["target_class"] for o in result["objects"]], ["door", "stairs"])
        self.assertEqual(result["objects"][0]["bbox_xyxy_normalized"], [0.1, 0.2, 0.3, 0.9])
        self.assertIsNone(result["objects"][0]["score"])

    def test_bad_boxes_and_unknown_labels_are_rejected_not_guessed(self):
        text = '[{"bbox_2d": [10, 10, 5, 50], "label": "door"},' \
               ' {"bbox_2d": [0, 0, 1200, 50], "label": "door"},' \
               ' {"bbox_2d": [0, 0, true, 50], "label": "door"},' \
               ' {"bbox_2d": [0, 0, 10], "label": "door"},' \
               ' {"bbox_2d": [0, 0, 10, 10], "label": "window"},' \
               ' {"bbox_2d": [0, 0, 10, 10]}, "door",' \
               ' {"bbox_2d": [0, 0, 10, 10], "label": "sign"}]'
        result = parse_qwen_grounding(text)
        self.assertIsNone(result["error"])
        self.assertEqual(result["rejected"], 7)
        self.assertEqual([o["target_class"] for o in result["objects"]], ["signboard"])

    def test_allowed_classes_filter(self):
        text = '[{"bbox_2d": [0, 0, 10, 10], "label": "door"}, {"bbox_2d": [0, 0, 10, 10], "label": "car"}]'
        result = parse_qwen_grounding(text, allowed_classes=("car",))
        self.assertEqual([o["target_class"] for o in result["objects"]], ["car"])
        self.assertEqual(result["rejected"], 1)

    def test_malformed_outputs_give_explicit_errors_and_no_objects(self):
        cases = {"": "invalid_text", None: "invalid_text", "no boxes here": "no_json",
                 "[{bbox_2d: 1}]": "invalid_json", '[{"bbox_2d": [NaN, 0, 1, 1], "label": "door"}]': "invalid_json",
                 "x" * 20_001: "invalid_text"}
        for text, error in cases.items():
            with self.subTest(error=error):
                result = parse_qwen_grounding(text)
                self.assertEqual(result["error"], error)
                self.assertEqual(result["objects"], [])

    def test_object_count_is_bounded(self):
        text = "[" + ",".join(['{"bbox_2d": [0, 0, 10, 10], "label": "door"}'] * 51) + "]"
        self.assertEqual(parse_qwen_grounding(text)["error"], "too_many")


if __name__ == "__main__":
    unittest.main()
