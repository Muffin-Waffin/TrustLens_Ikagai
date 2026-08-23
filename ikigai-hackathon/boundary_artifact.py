"""Face-boundary discontinuity measurements for forensic evidence."""
from __future__ import annotations

from typing import Any, Optional
import cv2
import numpy as np


def analyze_boundary_artifact(frame: np.ndarray, bbox: Any, settings: dict[str, float]) -> Optional[float]:
    """Return normalized edge/color discontinuity around a valid face boundary.

    The caller supplies normalization settings (``baseline`` and ``scale``), so
    this function does not embed a detection threshold.
    """
    if frame is None or frame.ndim < 2 or bbox is None:
        return None
    try:
        x1, y1, x2, y2 = map(int, np.asarray(bbox).reshape(-1)[:4])
    except (TypeError, ValueError, IndexError):
        return None
    height, width = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    band = max(1, int(settings.get("band_width", 3)))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    gradient = cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))

    inside_parts = [
        gradient[y1:y2, x1:min(x1 + band, x2)].reshape(-1),
        gradient[y1:y2, max(x2 - band, x1):x2].reshape(-1),
        gradient[y1:min(y1 + band, y2), x1:x2].reshape(-1),
        gradient[max(y2 - band, y1):y2, x1:x2].reshape(-1),
    ]
    inside_pts = np.concatenate([p for p in inside_parts if p.size > 0]) if any(p.size > 0 for p in inside_parts) else np.array([0.0])
    inside = float(inside_pts.mean()) if inside_pts.size > 0 else 0.0

    outside_parts = [
        gradient[y1:y2, max(0, x1 - band):x1].reshape(-1),
        gradient[y1:y2, x2:min(width, x2 + band)].reshape(-1),
        gradient[max(0, y1 - band):y1, x1:x2].reshape(-1),
        gradient[y2:min(height, y2 + band), x1:x2].reshape(-1),
    ]
    outside_pts = np.concatenate([p for p in outside_parts if p.size > 0]) if any(p.size > 0 for p in outside_parts) else np.array([0.0])
    outside = float(outside_pts.mean()) if outside_pts.size > 0 else 0.0

    color = frame.astype(np.float32)
    face_crop = color[y1:y2, x1:x2]
    inner = face_crop.mean(axis=(0, 1)) if face_crop.size > 0 else np.zeros(color.shape[2] if color.ndim == 3 else 1)

    outer_rings = []
    top_ring = color[max(0, y1 - band):y1, x1:x2]
    bottom_ring = color[y2:min(height, y2 + band), x1:x2]
    left_ring = color[y1:y2, max(0, x1 - band):x1]
    right_ring = color[y1:y2, x2:min(width, x2 + band)]
    for r in [top_ring, bottom_ring, left_ring, right_ring]:
        if r.size > 0 and r.ndim >= 2:
            outer_rings.append(r.reshape(-1, color.shape[2] if color.ndim == 3 else 1))

    outer_ring = np.concatenate(outer_rings) if outer_rings else np.empty((0, color.shape[2] if color.ndim == 3 else 1))
    color_gap = float(np.linalg.norm(inner - outer_ring.mean(axis=0))) if outer_ring.size else 0.0
    raw = abs(float(inside - outside)) + float(settings.get("color_weight", 1.0)) * color_gap
    return float(np.clip((raw - float(settings.get("baseline", 0.0))) / max(float(settings.get("scale", 1.0)), 1e-12), 0.0, 1.0))
