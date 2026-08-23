"""Consecutive InsightFace embedding identity-stability analysis."""
from __future__ import annotations
from typing import Any, Optional
import numpy as np

def identity_similarity_and_stability(previous: Any, current: Any) -> tuple[Optional[float], Optional[float]]:
    """Return cosine similarity and stable-identity score (1 stable, 0 drift)."""
    if current is None:
        return None, None
    try:
        b = np.asarray(current, dtype=float).reshape(-1)
        if b.size == 0 or not np.isfinite(b).all():
            return None, None
        if previous is None:
            # Baseline stability for the first detected face
            return 1.0, 1.0
        a = np.asarray(previous, dtype=float).reshape(-1)
        if a.size == 0 or a.size != b.size or not np.isfinite(a).all():
            return None, None
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom <= 1e-12:
            return None, None
        similarity = float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))
        return similarity, float(np.clip((similarity + 1.0) / 2.0, 0.0, 1.0))
    except (TypeError, ValueError):
        return None, None
