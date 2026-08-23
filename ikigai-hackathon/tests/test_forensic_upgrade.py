"""Focused regression coverage for additive forensic evidence modules."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from forensic_engine import compute_consistency, compute_weighted_statistics, weighted_median
from boundary_artifact import analyze_boundary_artifact
from frequency_analysis import analyze_frequency
from blink_analysis import BlinkAnalyzer
from identity_drift import identity_similarity_and_stability

def test_weighted_median_cases():
    assert weighted_median([1, 2, 3], [1, 1, 1]) == 2
    assert weighted_median([1, 2, 3], [1, 5, 1]) == 2
    assert weighted_median([1, 2, 3], [10, 1, 1]) == 1
    assert weighted_median([1, 2], [0, 0]) == 0
    assert weighted_median([], []) == 0
    assert weighted_median([np.nan, 2], [1, 1]) == 2

def test_weighted_statistics_and_consistency():
    stats = compute_weighted_statistics([0, 1], [1, 3])
    assert stats["weighted_mean"] == 0.75
    assert stats["weighted_median"] == 1
    assert np.isclose(stats["weighted_std"], np.sqrt(0.1875))
    assert compute_consistency(0) == 1
    assert compute_consistency(0.5) == 0
    assert compute_consistency(0.7) == 0

def test_signal_unavailable_and_identical_identity():
    assert analyze_boundary_artifact(np.zeros((10, 10, 3), np.uint8), [9, 1, 2, 3], {}) is None
    assert analyze_frequency(np.zeros((20, 20, 3), np.uint8), {})[0] == 0
    checker = np.indices((32, 32)).sum(axis=0) % 2 * 255
    assert analyze_frequency(np.dstack([checker] * 3).astype(np.uint8), {"baseline": 0, "scale": 1})[1] > 0
    assert BlinkAnalyzer({}).update(None)["blink_naturalness"] is None
    assert identity_similarity_and_stability(None, [1, 0]) == (None, None)
    similarity, stable = identity_similarity_and_stability([1, 0], [1, 0])
    assert similarity == 1 and stable == 1
