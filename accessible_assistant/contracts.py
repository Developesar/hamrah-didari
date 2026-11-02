"""Shared result contracts for visual and speech components. Model scores are not calibrated probabilities."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

STATUSES = frozenset({
    "ok", "empty", "partial", "unavailable", "invalid_input", "unsupported",
    "error", "timeout", "cancelled", "stale",
})
TARGET_CLASSES = ("person", "car", "chair", "door", "stairs", "signboard")
PERSIAN_LABELS = {
    "person": "انسان", "car": "خودرو", "chair": "صندلی",
    "door": "در", "stairs": "پله", "signboard": "تابلو",
}


@dataclass
class ComponentResult:
    status: str
    data: dict[str, Any] = field(default_factory=dict)
    model_id: str | None = None
    elapsed_ms: float = 0.0
    error_code: str | None = None
    user_message: str = ""

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"Unknown component status: {self.status}")
        if self.elapsed_ms < 0:
            raise ValueError("elapsed_ms must be nonnegative")

    @property
    def usable(self) -> bool:
        return self.status in {"ok", "partial", "empty"}

    def metadata(self) -> dict[str, Any]:
        """Bounded report representation; never serialize recorded speech bytes."""
        result = asdict(self)
        if "audio_bytes" in result["data"]:
            payload = result["data"].pop("audio_bytes")
            result["data"]["audio_size_bytes"] = len(payload)
        return result
