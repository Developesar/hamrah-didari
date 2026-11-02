"""Run selected components on synthetic data; no camera, network or model downloads."""
from pathlib import Path
import json
import numpy as np
import cv2
from accessible_assistant.vision_preprocess import assess_quality
from accessible_assistant.annotate import draw
from accessible_assistant.grounding import parse_qwen_grounding
from accessible_assistant.persian_text import normalize
from accessible_assistant.audio_preprocess import mfcc

def main():
    output = Path("output/demo")
    output.mkdir(parents=True, exist_ok=True)
    frame = np.full((360, 640, 3), 245, dtype=np.uint8)
    cv2.rectangle(frame, (220, 110), (420, 290), (130, 80, 25), -1)
    cv2.putText(frame, "Synthetic example", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)
    objects = [{"target_class": "chair", "bbox_xyxy_normalized": [0.34, 0.30, 0.66, 0.82], "score": 0.9}]
    cv2.imwrite(str(output / "annotated.png"), draw(frame, objects=objects))
    waveform = 0.2 * np.sin(2 * np.pi * 220 * np.arange(16000) / 16000)
    report = {
        "input": "synthetic image and tone; no inference",
        "quality": assess_quality(frame),
        "persian_text": normalize("كتاب 123"),
        "mfcc_shape": list(mfcc(waveform, 16000).shape),
    }
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf8")
    print("Created output/demo/annotated.png and output/demo/report.json")

if __name__ == "__main__":
    main()
