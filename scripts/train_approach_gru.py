"""Train the approach GRU on synthetic box trajectories and compare it with the height rule.

  .venv-local\\Scripts\\python.exe -m scripts.train_approach_gru
Writes models/trained/approach_gru.pt and reports/sequence_model/train_report.json. Synthetic
data only (no user media); separate seeds for train/validation/test.
"""
import json
from pathlib import Path

import numpy as np

from accessible_assistant.sequence_model import CLASSES, WEIGHTS, build, features
from accessible_assistant.tracking import IoUTracker

ROOT = Path(__file__).resolve().parents[1]


def trajectory(rng, label):
    frames = int(rng.integers(4, 21))
    h0, cx, cy = rng.uniform(0.1, 0.5), rng.uniform(0.2, 0.8), rng.uniform(0.3, 0.7)
    growth = {"approaching": rng.uniform(1.3, 2.5), "moving_away": rng.uniform(0.35, 0.78),
              "steady": rng.uniform(0.9, 1.1)}[label]
    t = np.linspace(0, 1, frames)
    h = h0 * growth ** t * (1 + rng.normal(0, 0.06, frames))       # detector jitter
    w = h * rng.uniform(0.3, 1.2) * (1 + rng.normal(0, 0.06, frames))
    cx = cx + np.cumsum(rng.normal(0, 0.01, frames))                 # camera/body sway
    cy = cy + np.cumsum(rng.normal(0, 0.01, frames))
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    return np.clip(boxes, 0, 1)


def dataset(seed, n):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, len(CLASSES), n)
    return [trajectory(rng, CLASSES[i]) for i in labels], labels


def rule_accuracy(trajs, labels):
    hits = [IoUTracker.trend([(i, tuple(b)) for i, b in enumerate(t)]) == CLASSES[y] for t, y in zip(trajs, labels)]
    return float(np.mean(hits))


def main():
    import torch
    from torch import nn
    torch.manual_seed(0)
    train, val, test = dataset(1, 3000), dataset(2, 600), dataset(3, 1000)
    model = build()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    loss_fn = nn.CrossEntropyLoss()

    def batches(data, size=64):
        trajs, labels = data
        order = np.random.default_rng(0).permutation(len(trajs))
        for start in range(0, len(order), size):
            idx = order[start:start + size]
            seqs = [torch.from_numpy(features(trajs[i])) for i in idx]
            padded = nn.utils.rnn.pad_sequence(seqs, batch_first=True)
            lengths = torch.tensor([len(s) for s in seqs])
            packed = nn.utils.rnn.pack_padded_sequence(padded, lengths, batch_first=True, enforce_sorted=False)
            yield packed, torch.tensor(labels[idx])

    def forward(packed):
        _, last = model.gru(packed)
        return model.head(last[-1])

    def accuracy(data):
        model.eval()
        hits = total = 0
        with torch.no_grad():
            for packed, y in batches(data, 256):
                hits += int((forward(packed).argmax(-1) == y).sum())
                total += len(y)
        model.train()
        return hits / total

    history = []
    for epoch in range(15):
        for packed, y in batches(train):
            optimizer.zero_grad()
            loss = loss_fn(forward(packed), y)
            loss.backward()
            optimizer.step()
        history.append({"epoch": epoch + 1, "loss": round(float(loss), 4), "val_accuracy": round(accuracy(val), 4)})
        print(history[-1], flush=True)
    report = {"data": "synthetic trajectories; train/val/test seeds 1/2/3; 3000/600/1000",
              "gru_test_accuracy": round(accuracy(test), 4), "rule_test_accuracy": round(rule_accuracy(*test), 4),
              "history": history}
    WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), WEIGHTS)
    out = ROOT / "reports/sequence_model"
    out.mkdir(parents=True, exist_ok=True)
    (out / "train_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "history"}))


if __name__ == "__main__":
    main()
