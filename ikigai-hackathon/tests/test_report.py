"""
Tests for report module.
"""

import pytest
import tempfile
from pathlib import Path
import json

from report import (
    generate_html_report,
    save_json_report,
    generate_case_id,
    format_file_size,
)


def test_generate_case_id():
    """Test case ID generation."""
    case_id = generate_case_id("SG")
    assert case_id.startswith("SG-")
    assert len(case_id) > 10


def test_format_file_size():
    """Test file size formatting."""
    assert format_file_size(500) == "500.0 B"
    assert format_file_size(1024) == "1.0 KB"
    assert format_file_size(1024 * 1024) == "1.0 MB"
    assert format_file_size(1024 * 1024 * 1024) == "1.0 GB"


def test_generate_html_report():
    """Test HTML report generation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test video
        video_path = Path(tmpdir) / "test_video.mp4"
        video_path.write_bytes(b"test video content")
        
        forensic_result = {
            "verdict": "LIKELY_DEEPFAKE",
            "manipulation_score": 0.85,
            "evidence_confidence": 0.75,
            "reliability": 0.88,
            "consistency": 0.82,
            "frame_coverage": 0.90,
            "average_face_quality": 0.85,
            "mean_score": 0.82,
            "median_score": 0.85,
            "max_score": 0.92,
            "std_score": 0.05,
            "sampled_frames": 60,
            "usable_frames": 54,
            "video_metadata": {
                "width": 1920,
                "height": 1080,
                "fps": 30.0,
                "duration_seconds": 30.0,
                "frame_count": 900,
                "codec": "avc1",
            },
            "explanations": [
                "High manipulation scores across multiple frames.",
                "Consistent scores in suspicious interval.",
            ],
            "suspicious_segments": [
                {"start": 10.0, "end": 15.0, "duration": 5.0, "frame_count": 8, "peak_score": 0.92, "mean_score": 0.88},
            ],
            "suspicious_frames": [
                {"frame_index": 300, "timestamp_seconds": 10.0, "score": 0.92, "frame_path": ""},
            ],
        }
        
        config = {
            "forensic": {
                "thresholds": {
                    "real_max": 0.35,
                    "deepfake_min": 0.70,
                    "strong_reliability_min": 0.60,
                }
            },
            "model": {
                "name": "convnext_tiny",
                "image_size": 224,
            },
            "inference": {
                "checkpoint": "checkpoints/best_convnext_tiny.pt",
            },
            "report": {
                "case_id_prefix": "SG",
                "include_sha256": True,
            },
        }
        
        output_path = Path(tmpdir) / "report.html"
        html = generate_html_report(
            video_path,
            forensic_result,
            config,
            output_path=output_path,
        )
        
        assert output_path.exists()
        assert len(html) > 1000
        assert "SynthGuard Forensic Report" in html
        assert "LIKELY_DEEPFAKE" in html
        assert "0.85" in html  # manipulation score


def test_save_json_report():
    """Test JSON report generation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = Path(tmpdir) / "test_video.mp4"
        video_path.write_bytes(b"test video content")
        
        forensic_result = {
            "verdict": "REAL",
            "manipulation_score": 0.15,
            "sampled_frames": 60,
            "usable_frames": 54,
            "frame_coverage": 0.90,
            "average_face_quality": 0.85,
            "evidence_confidence": 0.75,
            "reliability": 0.88,
            "consistency": 0.82,
        }
        
        output_path = Path(tmpdir) / "report.json"
        report = save_json_report(
            video_path,
            forensic_result,
            output_path=output_path,
        )
        
        assert output_path.exists()
        
        with open(output_path) as f:
            loaded = json.load(f)
        
        # New canonical structure
        assert loaded["decision"]["verdict"] == "REAL"
        assert "case" in loaded
        assert "video" in loaded
        assert "model" in loaded
        assert "preprocessing" in loaded
        assert "detection" in loaded
        assert "evidence" in loaded
        assert "decision" in loaded
        assert "suspicious" in loaded
        assert "explainability" in loaded
        assert "robustness" in loaded
        assert "limitations" in loaded
        assert loaded["case"]["sha256"] != "N/A"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])