"""Optional compact GRU for box-sequence trends; falls back when weights are absent.
Training uses synthetic trajectories, not real-world approach labels.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "models/trained/approach_gru.pt"
CLASSES = ("approaching", "moving_away", "steady")


def features(boxes):
    import numpy as np
    arr = np.asarray(boxes, dtype=np.float32)
    h, w = arr[:, 3] - arr[:, 1], arr[:, 2] - arr[:, 0]
    cx, cy = (arr[:, 0] + arr[:, 2]) / 2, (arr[:, 1] + arr[:, 3]) / 2
    feats = np.stack([h / max(h[0], 1e-3) - 1, w / max(w[0], 1e-3) - 1, cx - cx[0], cy - cy[0]], axis=1)
    return feats.astype(np.float32)


def build(hidden: int = 16):
    import torch
    from torch import nn

    class ApproachGRU(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(4, hidden, batch_first=True)
            self.head = nn.Linear(hidden, len(CLASSES))

        def forward(self, x):
            _, last = self.gru(x)
            return self.head(last[-1])

    return ApproachGRU()


_MODEL = None


def predict(boxes) -> str | None:
    """Trend from the GRU, or None when torch/weights are unavailable or the track is too short."""
    global _MODEL
    if len(boxes) < 3 or not WEIGHTS.is_file():
        return None
    try:
        import torch
        if _MODEL is None:
            model = build()
            model.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=True))
            _MODEL = model.eval()
        with torch.inference_mode():
            logits = _MODEL(torch.from_numpy(features(boxes))[None])
        return CLASSES[int(logits.argmax(-1))]
    except Exception:
        return None
