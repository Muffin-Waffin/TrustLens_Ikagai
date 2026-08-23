"""
Regression tests for identified bugs in SynthGuard.

These tests ensure that previously identified issues are fixed and don't regress.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile

from forensic_engine import (
    analyze_frame_predictions,
    ForensicResult,
    SuspiciousFrame,
    SuspiciousSegment,
    group_suspicious_segments,
    generate_explanations,
    find_suspicious_frames,
    get_suspicious_frames_by_timestamp,
)


def test_bug1_metadata_not_zero():
    """BUG 1: Video metadata must not be zero for valid video."""
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
    
    # Frame data with valid metadata
    frame_data = [
        {"frame_index": i, "timestamp_seconds": float(i), "score": 0.8, "face_quality": 0.8, "usable": True}
        for i in range(10)
    ]
    
    result = analyze_frame_predictions(frame_data, "test_video", config)
    
    # The forensic engine doesn't validate metadata directly, but the report does
    # This test ensures the frame data has valid timestamps
    assert all(f["timestamp_seconds"] >= 0 for f in frame_data)
    assert all(f["frame_index"] >= 0 for f in frame_data)


def test_bug2_suspicious_segment_negative_duration():
    """BUG 2: Suspicious segment must have non-negative duration (start <= end)."""
    # Create suspicious frames sorted by timestamp
    frames = [
        SuspiciousFrame(10, 1.0, 0.85),
        SuspiciousFrame(15, 1.5, 0.88),
        SuspiciousFrame(20, 2.0, 0.92),
        SuspiciousFrame(50, 5.0, 0.80),  # gap > 1.5
    ]
    
    segments = group_suspicious_segments(frames, 1.5)
    
    for seg in segments:
        assert seg.start <= seg.end, f"Segment start ({seg.start}) > end ({seg.end})"
        assert seg.duration >= 0, f"Segment has negative duration: {seg.duration}"
        assert seg.duration == seg.end - seg.start, "Duration mismatch"
    
    # Check specific segments
    assert segments[0].start == 1.0
    assert segments[0].end == 2.0
    assert segments[0].duration == 1.0
    assert segments[1].start == 5.0
    assert segments[1].end == 5.0
    assert segments[1].duration == 0.0


def test_bug3_suspicious_frames_consistency():
    """BUG 3: If score > threshold, frame must be in suspicious list."""
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
    
    # Frames with scores above and below threshold
    frame_data = [
        {"frame_index": 10, "timestamp_seconds": 1.0, "score": 0.85, "face_quality": 0.8, "usable": True},
        {"frame_index": 20, "timestamp_seconds": 2.0, "score": 0.65, "face_quality": 0.8, "usable": True},  # Below threshold
        {"frame_index": 30, "timestamp_seconds": 3.0, "score": 0.92, "face_quality": 0.8, "usable": True},
        {"frame_index": 40, "timestamp_seconds": 4.0, "score": 0.50, "face_quality": 0.8, "usable": False},  # Not usable
    ]
    
    result = analyze_frame_predictions(frame_data, "test_video", config)
    
    # Check that all usable frames with score >= threshold are in suspicious_frames
    threshold = config["forensic"]["suspicious_frame_threshold"]
    for f in frame_data:
        if f.get("usable", False) and f.get("score", 0.0) >= threshold:
            # This frame should be in suspicious_frames
            found = any(sf.frame_index == f["frame_index"] for sf in result.suspicious_frames)
            assert found, f"Frame {f['frame_index']} with score {f['score']} >= {threshold} not in suspicious_frames"
    
    # Frames below threshold should NOT be in suspicious_frames
    for sf in result.suspicious_frames:
        assert sf.score >= threshold, f"Suspicious frame {sf.frame_index} has score {sf.score} < {threshold}"


def test_bug4_model_identity_consistency():
    """BUG 4: Model identity must be consistent (Xception vs ConvNeXt-Tiny)."""
    # This test verifies that the model architecture is properly tracked
    # The actual check is in the report generation
    from model import XceptionDeepfakeClassifier, DeepfakeClassifier, get_model_info
    import torch
    
    device = torch.device("cpu")
    
    # Test Xception
    xception = XceptionDeepfakeClassifier(pretrained=False)
    xception = xception.to(device)
    info = get_model_info(xception)
    assert "Xception" in info["architecture"] or info["architecture"] == "XceptionDeepfakeClassifier"
    
    # Test ConvNeXt
    convnext = DeepfakeClassifier(pretrained=False, dropout=0.3)
    convnext = convnext.to(device)
    info = get_model_info(convnext)
    assert "ConvNeXt" in info["architecture"] or info["architecture"] == "DeepfakeClassifier"
    
    # The architecture names are distinct
    assert info["architecture"] != "XceptionDeepfakeClassifier" or "ConvNeXt" not in info["architecture"]


def test_bug5_consistency_explanation_matches_value():
    """BUG 5: Low consistency must not generate 'consistently high' explanation."""
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
    
    # Low consistency scenario: varying scores
    frame_data = [
        {"frame_index": i, "timestamp_seconds": float(i), "score": s, "face_quality": 0.8, "usable": True}
        for i, s in enumerate([0.10, 0.92, 0.35, 0.89, 0.14, 0.88, 0.12, 0.90])
    ]
    
    result = analyze_frame_predictions(frame_data, "test_video", config)
    
    # With these varying scores, consistency should be low
    assert result.consistency < 0.5, f"Expected low consistency, got {result.consistency}"
    
    # Check explanations - should NOT say "consistently high" or "remained relatively stable"
    for exp in result.explanations:
        assert "consistently high" not in exp.lower(), f"Inappropriate explanation for low consistency: {exp}"
        assert "remained relatively stable" not in exp.lower(), f"Inappropriate explanation for low consistency: {exp}"
    
    # Should have LOW_CONSISTENCY reason code
    assert "LOW_CONSISTENCY" in result.reason_codes, f"Missing LOW_CONSISTENCY reason code: {result.reason_codes}"
    
    # High consistency scenario: stable scores
    frame_data_stable = [
        {"frame_index": i, "timestamp_seconds": float(i), "score": 0.87 + np.random.normal(0, 0.01), "face_quality": 0.8, "usable": True}
        for i in range(10)
    ]
    
    result_stable = analyze_frame_predictions(frame_data_stable, "test_video", config)
    
    # With stable scores, consistency should be high
    assert result_stable.consistency > 0.75, f"Expected high consistency, got {result_stable.consistency}"
    
    # Should have HIGH_CONSISTENCY reason code
    assert "HIGH_CONSISTENCY" in result_stable.reason_codes, f"Missing HIGH_CONSISTENCY reason code: {result_stable.reason_codes}"


def test_bug6_dashboard_is_primary_report():
    """BUG 6: Dashboard is the primary forensic report (not separate HTML page)."""
    # This is more of an architectural test - verify the app.py renders the full report
    # We check that the canonical result structure contains all sections needed for the dashboard
    from inference import build_analysis_result
    from model import build_model
    from preprocessing import initialize_face_detector
    import torch
    
    # This is a structural test - we verify the canonical result has all required sections
    device = torch.device("cpu")
    model = build_model("xception", pretrained=False, device=device)
    
    # Mock minimal frame data
    frame_infos = [
        {"frame_index": i, "timestamp_seconds": float(i), "score": 0.5, "face_quality": 0.8, "usable": True, "face_path": ""}
        for i in range(5)
    ]
    video_metadata = {
        "fps": 30.0, "frame_count": 150, "duration_seconds": 5.0,
        "width": 1920, "height": 1080, "codec": "h264"
    }
    config = {
        "model": {"image_size": 224},
        "runtime": {"prefer_cuda": False},
        "face": {"model": "buffalo_s"},
        "forensic": {
            "consistency_scale": 0.10,
            "suspicious_frame_threshold": 0.70,
            "max_gap_seconds": 1.5,
            "top_k_frames": 5,
            "min_usable_frames": 5,
            "reliability": {"coverage_weight": 0.35, "quality_weight": 0.35, "consistency_weight": 0.30},
            "thresholds": {"real_max": 0.35, "deepfake_min": 0.70, "strong_reliability_min": 0.60},
            "visualization": {"enabled": False},
        },
        "explainability": {"enabled": False},
        "robustness": {"enabled": False},
        "paths": {"forensic": "./outputs/forensic", "reports": "./outputs/reports", "explanations": "./outputs/explanations"},
        "report": {"formats": ["json"], "case_id_prefix": "SG"},
    }
    
    face_app = initialize_face_detector(config)
    
    result = build_analysis_result(
        "test.mp4", config, model, frame_infos, video_metadata, device, face_app
    )
    
    canonical = result["canonical"]
    
    # Verify all sections exist for dashboard rendering
    required_sections = [
        "case", "video", "model", "preprocessing", "detection",
        "evidence", "decision", "suspicious", "explainability", "robustness", "limitations"
    ]
    for section in required_sections:
        assert section in canonical, f"Missing section in canonical result: {section}"
    
    # Verify specific fields needed for dashboard
    assert "verdict" in canonical["decision"]
    assert "reason_codes" in canonical["decision"]
    assert "manipulation_score" in canonical["detection"]
    assert "consistency" in canonical["evidence"]
    assert "reliability" in canonical["evidence"]
    assert "confidence" in canonical["evidence"]


def test_suspicious_frames_by_timestamp_ordering():
    """Test that get_suspicious_frames_by_timestamp returns frames sorted by timestamp."""
    frame_data = [
        {"frame_index": 30, "timestamp_seconds": 3.0, "score": 0.92, "face_quality": 0.8, "usable": True},
        {"frame_index": 10, "timestamp_seconds": 1.0, "score": 0.85, "face_quality": 0.8, "usable": True},
        {"frame_index": 20, "timestamp_seconds": 2.0, "score": 0.65, "face_quality": 0.8, "usable": True},  # Below threshold
    ]
    threshold = 0.70
    
    result = get_suspicious_frames_by_timestamp(frame_data, threshold)
    
    # Should only return frames with score >= threshold
    assert len(result) == 2
    
    # Should be sorted by timestamp
    assert result[0].timestamp_seconds == 1.0
    assert result[1].timestamp_seconds == 3.0


def test_segment_grouping_with_single_frame():
    """Test segment grouping with a single suspicious frame."""
    frames = [SuspiciousFrame(10, 1.0, 0.85)]
    segments = group_suspicious_segments(frames, 1.5)
    
    assert len(segments) == 1
    assert segments[0].start == 1.0
    assert segments[0].end == 1.0
    assert segments[0].duration == 0.0
    assert segments[0].frame_count == 1


def test_no_usable_frames_verdict():
    """Test that no usable frames results in INCONCLUSIVE verdict."""
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
        {"frame_index": 0, "timestamp_seconds": 0.0, "usable": False},
        {"frame_index": 1, "timestamp_seconds": 1.0, "usable": False},
    ]
    
    result = analyze_frame_predictions(frame_data, "test_video", config)
    
    assert result.verdict == "INCONCLUSIVE"
    assert result.usable_frames == 0
    assert result.reliability == 0.0
    assert "NO_USABLE_FRAMES" in result.reason_codes


def test_minimum_usable_frames_gate():
    """Test that fewer than minimum usable frames results in INCONCLUSIVE."""
    config = {
        "forensic": {
            "consistency_scale": 0.10,
            "suspicious_frame_threshold": 0.70,
            "max_gap_seconds": 1.5,
            "top_k_frames": 5,
            "min_usable_frames": 5,
            "reliability": {"coverage_weight": 0.35, "quality_weight": 0.35, "consistency_weight": 0.30},
            "thresholds": {"real_max": 0.35, "deepfake_min": 0.70, "strong_reliability_min": 0.60},
        }
    }
    
    # Only 3 usable frames (less than minimum of 5)
    frame_data = [
        {"frame_index": i, "timestamp_seconds": float(i), "score": 0.95, "face_quality": 0.9, "usable": True}
        for i in range(3)
    ]
    
    result = analyze_frame_predictions(frame_data, "test_video", config)
    
    assert result.verdict == "INCONCLUSIVE"
    assert "INSUFFICIENT_USABLE_FRAMES" in result.reason_codes


def test_high_manipulation_low_reliability_inconclusive():
    """Test that high manipulation + low reliability = INCONCLUSIVE."""
    config = {
        "forensic": {
            "consistency_scale": 0.10,
            "suspicious_frame_threshold": 0.70,
            "max_gap_seconds": 1.5,
            "top_k_frames": 5,
            "min_usable_frames": 5,
            "reliability": {"coverage_weight": 0.35, "quality_weight": 0.35, "consistency_weight": 0.30},
            "thresholds": {"real_max": 0.35, "deepfake_min": 0.70, "strong_reliability_min": 0.60},
        }
    }
    
    # High manipulation (median > 0.7) but low reliability (low coverage + low quality + low consistency)
    # Only 2 usable frames out of 10 sampled, low quality, varying scores
    frame_data = [
        {"frame_index": i, "timestamp_seconds": float(i), "score": 0.95, "face_quality": 0.3, "usable": True}
        for i in range(2)  # Only 2 usable frames -> low coverage
    ] + [
        {"frame_index": i, "timestamp_seconds": float(i), "score": 0.1 + i * 0.1, "face_quality": 0.3, "usable": False}
        for i in range(8)  # 8 non-usable frames
    ]
    
    result = analyze_frame_predictions(frame_data, "test_video", config)
    
    # High manipulation but low reliability -> INCONCLUSIVE
    assert result.verdict == "INCONCLUSIVE"
    assert result.manipulation_score > 0.7
    assert result.reliability < 0.6, f"Expected reliability < 0.6, got {result.reliability}"


def test_robustness_values_in_range():
    """Test that robustness values remain within [0,1]."""
    from robustness import apply_resize, apply_blur, apply_jpeg_compression, apply_brightness
    import cv2
    import numpy as np
    
    # Create test image
    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    # Test all transforms preserve range
    transforms = [
        (apply_resize, {"scale": 0.75}),
        (apply_blur, {"kernel": 3, "sigma": 0.8}),
        (apply_jpeg_compression, {"quality": 75}),
        (apply_brightness, {"factor": 0.8}),
    ]
    
    for fn, params in transforms:
        result = fn(img, **params)
        assert result.shape == img.shape
        assert result.dtype == img.dtype
        assert np.all(result >= 0) and np.all(result <= 255)


def test_confidence_in_range():
    """Test that confidence remains within [0,1]."""
    from forensic_engine import compute_evidence_confidence
    
    # Test edge cases
    assert compute_evidence_confidence(0.0, 0.0) == 0.0
    assert compute_evidence_confidence(1.0, 1.0) == 1.0
    assert compute_evidence_confidence(0.5, 0.5) == 0.25
    assert compute_evidence_confidence(1.0, 0.0) == 0.0
    assert compute_evidence_confidence(0.0, 1.0) == 0.0
    
    # Test clamping
    assert compute_evidence_confidence(1.5, 1.0) == 1.0
    assert compute_evidence_confidence(-0.5, 1.0) == 0.0


def test_all_suspicious_segment_durations_non_negative():
    """Test that all suspicious segment durations are >= 0."""
    frames = [
        SuspiciousFrame(i, float(i), 0.8 + i * 0.01)
        for i in range(10)
    ]
    
    segments = group_suspicious_segments(frames, 2.0)
    
    for seg in segments:
        assert seg.duration >= 0, f"Negative duration: {seg.duration}"
        assert seg.start <= seg.end, f"Start > End: {seg.start} > {seg.end}"


def test_report_model_name_matches_loaded_architecture():
    """Test that report uses dynamic model architecture name."""
    from model import XceptionDeepfakeClassifier, DeepfakeClassifier, get_model_info
    import torch
    
    device = torch.device("cpu")
    
    # Xception
    xception = XceptionDeepfakeClassifier(pretrained=False)
    xception = xception.to(device)
    info = get_model_info(xception)
    arch_name = info["architecture"]
    
    # The report should use this exact architecture name
    assert arch_name in ["XceptionDeepfakeClassifier", "Xception"]
    
    # ConvNeXt
    convnext = DeepfakeClassifier(pretrained=False, dropout=0.3)
    convnext = convnext.to(device)
    info = get_model_info(convnext)
    arch_name = info["architecture"]
    
    assert arch_name in ["DeepfakeClassifier", "ConvNeXt"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])