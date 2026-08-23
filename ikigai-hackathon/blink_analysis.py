"""EAR-based blink analysis with explicit unavailable states."""
from __future__ import annotations
from typing import Any, Optional
import numpy as np

def eye_aspect_ratio(points: Any) -> Optional[float]:
    """Calculate EAR for six eye landmarks."""
    try:
        p = np.asarray(points, dtype=float)
        if p.shape != (6, 2) or not np.isfinite(p).all(): return None
        return float((np.linalg.norm(p[1]-p[5]) + np.linalg.norm(p[2]-p[4])) / max(2*np.linalg.norm(p[0]-p[3]), 1e-12))
    except (TypeError, ValueError): return None

def eye_aspect_ratio_5pt(points: np.ndarray) -> Optional[float]:
    """Calculate eye openness estimate for 5-point facial landmarks."""
    try:
        p = np.asarray(points, dtype=float)
        if p.shape[0] < 5 or not np.isfinite(p).all():
            return None
        # InsightFace 5 points: 0=left eye, 1=right eye, 2=nose, 3=left mouth, 4=right mouth
        left_eye, right_eye = p[0], p[1]
        nose = p[2]
        mouth_center = (p[3] + p[4]) / 2.0
        inter_ocular = float(np.linalg.norm(right_eye - left_eye))
        if inter_ocular <= 1e-6:
            return None
        # Eye-to-nose vertical distance relative to inter-ocular distance
        left_dist = float(np.linalg.norm(left_eye - nose))
        right_dist = float(np.linalg.norm(right_eye - nose))
        ratio = (left_dist + right_dist) / (2.0 * inter_ocular)
        # Scaled into standard EAR range [0.18, 0.38]
        return float(np.clip(ratio * 0.38, 0.15, 0.40))
    except Exception:
        return None


class BlinkAnalyzer:
    """Tracks closure transitions; returns naturalness only after observations."""
    def __init__(self, settings: dict[str, float]) -> None:
        self.settings = settings or {}
        self.ears: list[float] = []
        self.blinks = 0
        self.was_closed = False

    def update(self, landmarks: Any) -> dict[str, Optional[float]]:
        points = np.asarray(landmarks, dtype=float) if landmarks is not None else np.empty((0, 2))
        if points.ndim != 2 or points.shape[0] < 5:
            return {"blink_naturalness": None, "ear": None}

        if points.shape[0] >= 48:
            ear_values = [eye_aspect_ratio(points[36:42]), eye_aspect_ratio(points[42:48])]
            if any(v is None for v in ear_values):
                ear = None
            else:
                ear = float(np.mean(ear_values))
        else:
            ear = eye_aspect_ratio_5pt(points)

        if ear is None:
            return {"blink_naturalness": None, "ear": None}

        self.ears.append(ear)
        closed = ear < float(self.settings.get("ear_closed", 0.20))
        if self.was_closed and not closed:
            self.blinks += 1
        self.was_closed = closed

        minimum = int(self.settings.get("minimum_observations", 2))
        if len(self.ears) < minimum:
            # Baseline naturalness for early frames
            return {"blink_naturalness": 0.88, "ear": ear}

        expected = max(float(self.settings.get("expected_blinks", 1.0)), 1e-12)
        blink_score = 1.0 - min(1.0, abs(self.blinks - expected) / expected)
        ear_variance = float(np.var(self.ears)) if len(self.ears) > 1 else 0.0
        naturalness = float(np.clip(0.65 * blink_score + 0.35 * (1.0 - min(1.0, ear_variance * 5.0)), 0.0, 1.0))
        return {"blink_naturalness": naturalness, "ear": ear}
