"""
Tests for Phase 3 Forensic Engine.
"""

import pytest
import numpy as np

from forensic_engine import (
    compute_statistics,
    compute_consistency,
    compute_frame_coverage,
    compute_reliability,
    compute_evidence_confidence,
    determine_verdict,
    find_suspicious_frames,
    group_suspicious_segments,
    analyze_frame_predictions,
    SuspiciousFrame,
    SuspiciousSegment,
    ForensicResult,
)


def test_stable_high_scores():
    """High stable scores -> high manipulation + high consistency."""
    scores = np.array([0.87, 0.88, 0.89, 0.90, 0.88])
    stats = compute_statistics(scores)
    assert stats["median"] > 0.85
    assert stats["std"] < 0.02

    consistency = compute_consistency(stats["std"], 0.10)
    assert consistency > 0.8


def test_stable_low_scores():
    """Low stable scores -> low manipulation."""
    scores = np.array([0.10, 0.12, 0.08, 0.11, 0.09])
    stats = compute_statistics(scores)
    assert stats["median"] < 0.15


def test_highly_varying_scores():
    """Varying scores -> lower consistency."""
    scores = np.array([0.10, 0.92, 0.35, 0.89, 0.14])
    stats = compute_statistics(scores)
    assert stats["std"] > 0.3

    consistency = compute_consistency(stats["std"], 0.10)
    assert consistency < 0.5


def test_frame_coverage():
    """Frame coverage calculation."""
    assert compute_frame_coverage(54, 60) == 0.9
    assert compute_frame_coverage(0, 60) == 0.0
    assert compute_frame_coverage(10, 0) == 0.0
    assert compute_frame_coverage(100, 100) == 1.0


def test_reliability_calculation():
    """Reliability combines coverage, quality, consistency."""
    weights = {"coverage_weight": 0.35, "quality_weight": 0.35, "consistency_weight": 0.30}
    rel = compute_reliability(0.9, 0.9, 0.9, weights)
    assert 0.8 < rel <= 1.0

    rel_low = compute_reliability(0.3, 0.3, 0.3, weights)
    assert rel_low < 0.4


def test_evidence_confidence():
    """Evidence confidence = manipulation * reliability."""
    assert compute_evidence_confidence(0.92, 0.90) == pytest.approx(0.828, rel=0.01)
    assert compute_evidence_confidence(0.50, 0.50) == 0.25
    assert compute_evidence_confidence(1.0, 0.0) == 0.0


def test_verdict_real():
    """Low evidence confidence -> REAL."""
    thresholds = {"real_max": 0.35, "deepfake_min": 0.70, "strong_reliability_min": 0.60}
    verdict = determine_verdict(0.20, 0.80, thresholds, usable_frames=10)
    assert verdict == "REAL"


def test_verdict_inconclusive_low_confidence():
    """Medium evidence confidence -> INCONCLUSIVE."""
    thresholds = {"real_max": 0.35, "deepfake_min": 0.70, "strong_reliability_min": 0.60}
    verdict = determine_verdict(0.50, 0.80, thresholds)
    assert verdict == "INCONCLUSIVE"


def test_verdict_inconclusive_high_manipulation_low_reliability():
    """High manipulation but low reliability -> INCONCLUSIVE."""
    thresholds = {"real_max": 0.35, "deepfake_min": 0.70, "strong_reliability_min": 0.60}
    verdict = determine_verdict(0.85, 0.30, thresholds)
    assert verdict == "INCONCLUSIVE"


def test_verdict_likely_deepfake():
    """High manipulation + high reliability -> LIKELY_DEEPFAKE."""
    thresholds = {"real_max": 0.35, "deepfake_min": 0.70, "strong_reliability_min": 0.60}
    verdict = determine_verdict(0.85, 0.80, thresholds, usable_frames=10)
    assert verdict == "LIKELY_DEEPFAKE"


def test_midrange_model_score_is_not_promoted_to_deepfake():
    """Reliability must not turn a sub-threshold model score into a deepfake."""
    thresholds = {"real_max": 0.35, "deepfake_min": 0.70, "strong_reliability_min": 0.60}
    verdict = determine_verdict(
        evidence_confidence=0.50,
        reliability=0.93,
        thresholds=thresholds,
        usable_frames=10,
        manipulation_score=0.538,
    )
    assert verdict == "INCONCLUSIVE"


def test_no_usable_frames():
    """No usable frames -> INCONCLUSIVE."""
    config = {
        "forensic": {
            "consistency_scale": 0.10,
            "suspicious_frame_threshold": 0.70,
            "max_gap_seconds": 1.5,
            "top_k_frames": 5,
            "reliability": {"coverage_weight": 0.35, "quality_weight": 0.35, "consistency_weight": 0.30},
            "thresholds": {"real_max": 0.35, "deepfake_min": 0.70, "strong_reliability_min": 0.60},
        }
    }
    result = analyze_frame_predictions(
        [{"frame_index": 0, "timestamp_seconds": 0.0, "usable": False}],
        "test_video",
        config,
    )
    assert result.verdict == "INCONCLUSIVE"
    assert result.usable_frames == 0
    assert result.reliability == 0.0


def test_suspicious_frame_detection():
    """Frames above threshold are flagged as suspicious."""
    frame_data = [
        {"frame_index": 10, "timestamp_seconds": 1.0, "score": 0.85, "usable": True},
        {"frame_index": 20, "timestamp_seconds": 2.0, "score": 0.65, "usable": True},
        {"frame_index": 30, "timestamp_seconds": 3.0, "score": 0.92, "usable": True},
        {"frame_index": 40, "timestamp_seconds": 4.0, "score": 0.50, "usable": False},
    ]
    suspicious = find_suspicious_frames(frame_data, 0.70, 5)
    assert len(suspicious) == 2
    assert suspicious[0].score == 0.92
    assert suspicious[1].score == 0.85


def test_segment_grouping():
    """Nearby suspicious frames grouped into segments."""
    frames = [
        SuspiciousFrame(10, 1.0, 0.85),
        SuspiciousFrame(15, 1.5, 0.88),
        SuspiciousFrame(20, 2.0, 0.92),
        SuspiciousFrame(50, 5.0, 0.80),  # gap > 1.5
    ]
    segments = group_suspicious_segments(frames, 1.5)
    assert len(segments) == 2
    assert segments[0].start == 1.0
    assert segments[0].end == 2.0
    assert segments[1].start == 5.0
    assert segments[1].end == 5.0


def test_single_usable_frame():
    """Single usable frame: std=0 but low coverage -> low reliability."""
    config = {
        "forensic": {
            "consistency_scale": 0.10,
            "suspicious_frame_threshold": 0.70,
            "max_gap_seconds": 1.5,
            "top_k_frames": 5,
            "reliability": {"coverage_weight": 0.35, "quality_weight": 0.35, "consistency_weight": 0.30},
            "thresholds": {"real_max": 0.35, "deepfake_min": 0.70, "strong_reliability_min": 0.60},
        }
    }
    frame_data = [
        {"frame_index": 10, "timestamp_seconds": 1.0, "score": 0.90, "face_quality": 0.9, "usable": True},
        {"frame_index": 20, "timestamp_seconds": 2.0, "usable": False},
        {"frame_index": 30, "timestamp_seconds": 3.0, "usable": False},
    ]
    result = analyze_frame_predictions(frame_data, "test", config)
    assert result.std_score == 0.0
    assert result.consistency == 1.0
    assert result.frame_coverage < 0.5
    # With coverage=0.33, quality=0.9, consistency=1.0, weights 0.35/0.35/0.3:
    # reliability = 0.35*0.33 + 0.35*0.9 + 0.30*1.0 = 0.116 + 0.315 + 0.30 = 0.731
    assert result.reliability > 0.7


def test_output_schema():
    """Verify ForensicResult has all required fields."""
    config = {
        "forensic": {
            "consistency_scale": 0.10,
            "suspicious_frame_threshold": 0.70,
            "max_gap_seconds": 1.5,
            "top_k_frames": 5,
            "reliability": {"coverage_weight": 0.35, "quality_weight": 0.35, "consistency_weight": 0.30},
            "thresholds": {"real_max": 0.35, "deepfake_min": 0.70, "strong_reliability_min": 0.60},
        }
    }
    frame_data = [
        {"frame_index": i, "timestamp_seconds": float(i), "score": 0.8, "face_quality": 0.8, "usable": True}
        for i in range(10)
    ]
    result = analyze_frame_predictions(frame_data, "test_video", config)

    required_fields = [
        "video_id", "verdict", "manipulation_score", "mean_score", "median_score",
        "max_score", "std_score", "consistency", "frame_coverage", "average_face_quality",
        "reliability", "evidence_confidence", "sampled_frames", "usable_frames",
        "suspicious_frames", "suspicious_segments", "explanations"
    ]
    for field in required_fields:
        assert hasattr(result, field), f"Missing field: {field}"

    d = result.to_dict()
    for field in required_fields:
        assert field in d, f"Missing field in dict: {field}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
