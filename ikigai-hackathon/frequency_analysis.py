"""Configurable FFT high-frequency evidence measurement."""
from __future__ import annotations
from typing import Any, Optional
import cv2
import numpy as np

def analyze_frequency(face_crop: np.ndarray, settings: dict[str, float]) -> tuple[Optional[float], Optional[float]]:
    """Return ``(anomaly_score, high_frequency_energy_ratio)`` or unavailable."""
    if face_crop is None or face_crop.size == 0 or min(face_crop.shape[:2]) < 8:
        return None, None
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY) if face_crop.ndim == 3 else face_crop
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(gray.astype(np.float64)))) ** 2
    h, w = spectrum.shape
    radius = max(1, int(min(h, w) * float(settings.get("low_frequency_radius_ratio", 0.15))))
    yy, xx = np.ogrid[:h, :w]
    low = (yy - h // 2) ** 2 + (xx - w // 2) ** 2 <= radius ** 2
    total = float(spectrum.sum())
    ratio = float(spectrum[~low].sum() / total) if total > 1e-12 else 0.0
    anomaly = np.clip((ratio - float(settings.get("baseline", 0.0))) / max(float(settings.get("scale", 1.0)), 1e-12), 0.0, 1.0)
    return float(anomaly), ratio
