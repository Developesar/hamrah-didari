"""OpenCV preparation and drawing on synthetic images; behaviour checks, not quality claims."""
import io
import unittest

import numpy as np
from PIL import Image

try:
    import cv2
except ImportError:  # the UI-only environment may lack OpenCV
    cv2 = None


def page(dark=False, blur=False, angle=0.0):
    img = np.full((600, 900, 3), 40 if dark else 235, np.uint8)
    ink = (200, 200, 200) if dark else (20, 20, 20)
    for row in range(5):
        cv2.putText(img, "TEXT LINE %d SAMPLE" % row, (60, 110 + row * 90), cv2.FONT_HERSHEY_SIMPLEX, 1.6, ink, 4)
    if angle:
        m = cv2.getRotationMatrix2D((450, 300), angle, 1.0)
        img = cv2.warpAffine(img, m, (900, 600), borderValue=(235, 235, 235))
    if blur:
        img = cv2.GaussianBlur(img, (31, 31), 12)
    return img


@unittest.skipIf(cv2 is None, "OpenCV not installed")
class PreprocessTests(unittest.TestCase):
    def test_quality_flags_blur_and_darkness_with_persian_advice(self):
        from accessible_assistant.vision_preprocess import assess_quality
        sharp = assess_quality(page())
        self.assertNotIn("blurry", sharp["issues"])
        blurry = assess_quality(page(blur=True))
        self.assertIn("blurry", blurry["issues"])
        self.assertFalse(blurry["usable"])
        dark = assess_quality(page(dark=True))
        self.assertIn("dark", dark["issues"])
        self.assertTrue(all(isinstance(a, str) and a for a in dark["advice_fa"]))

    def test_enhance_brightens_dark_image(self):
        from accessible_assistant.vision_preprocess import assess_quality, enhance
        dark = page(dark=True)
        out, steps = enhance(dark, assess_quality(dark))
        self.assertIn("gamma_1.8", steps)
        self.assertGreater(out.mean(), dark.mean())

    def test_text_regions_and_deskew(self):
        from accessible_assistant.vision_preprocess import deskew_angle, prepare_for_ocr, text_regions
        regions = text_regions(page())
        self.assertGreaterEqual(len(regions), 3)
        for r in regions:
            x1, y1, x2, y2 = r["box_xyxy_normalized"]
            self.assertTrue(0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1)
        gray = cv2.cvtColor(page(angle=6), cv2.COLOR_BGR2GRAY)
        self.assertGreater(abs(deskew_angle(gray)), 2)
        prepared, steps = prepare_for_ocr(page(angle=6))
        self.assertTrue(any(s.startswith("deskew") for s in steps))
        self.assertEqual(prepared.ndim, 3)

    def test_draw_boxes_from_report(self):
        from accessible_assistant.annotate import from_report
        image = Image.new("RGB", (200, 100), "white")
        report = {"components": {"detector": {"data": {"objects": [
            {"target_class": "chair", "score": 0.96, "bbox_xyxy_normalized": [0.1, 0.2, 0.5, 0.9]}]}}}}
        drawn = np.asarray(from_report(image, report))
        self.assertLess(drawn.mean(), 255)
        self.assertIsNone(from_report(image, {"components": {}}))




if __name__ == "__main__":
    unittest.main()
