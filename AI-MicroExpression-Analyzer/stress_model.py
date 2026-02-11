from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

OUTPUT_LABELS: Dict[str, Tuple[str, str]] = {
    "calm": ("🟢", "Calm"),
    "mild": ("🟡", "Slight Stress"),
    "high": ("🔴", "High Stress / Possible Deception Indicators"),
}


@dataclass
class StressScore:
    score: float
    label: str
    icon: str
    level: str  # "calm" | "mild" | "high"

    def formatted(self) -> str:
        return f"{self.icon} {self.label} ({self.score:.2f})"


class StressEstimator:
    def __init__(self) -> None:
        # Weights reflect heuristic importance of each feature
        self.weights = {
            "eyebrow_raise": 0.3,
            "lip_tension": 0.25,
            "head_nod_intensity": 0.2,
            "symmetry_delta": 0.15,
            "blink_rate": 0.1,
        }
        self.scalers = {
            "eyebrow_raise": 0.08,   # typical range 0.02–0.08
            "lip_tension": 1.0,      # normalized 0–1
            "head_nod_intensity": 1.5,  # smoothed delta; calm ≈ 0–0.3
            "symmetry_delta": 0.05,  # typical range 0–0.05
            "blink_rate": 30.0,      # blinks per minute; >20 elevated
        }
        self.thresholds = {
            "calm": 0.35,
            "mild": 0.65,
        }

    def predict(self, features: Dict[str, float]) -> StressScore:
        weighted_sum = 0.0
        for key, value in features.items():
            weight = self.weights.get(key, 0.0)
            scale = self.scalers.get(key, 1.0)
            weighted_sum += weight * min(value / scale, 1.5)
        score = float(np.clip(weighted_sum, 0.0, 1.5))
        if score < self.thresholds["calm"]:
            label_key = "calm"
        elif score < self.thresholds["mild"]:
            label_key = "mild"
        else:
            label_key = "high"
        icon, label = OUTPUT_LABELS[label_key]
        return StressScore(score=score, icon=icon, label=label, level=label_key)
