"""
Tests for Streamlit app module.
"""

import pytest
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_app_imports():
    """Test that app.py can be imported without errors."""
    # This just verifies no syntax/import errors
    import app
    assert app is not None


def test_config_loads():
    """Test config can be loaded."""
    from config import load_config
    config = load_config("config.yaml")
    assert "model" in config
    assert "forensic" in config
    assert "explainability" in config
    assert "robustness" in config
    assert "report" in config


def test_model_creation():
    """Test model can be created."""
    from model import DeepfakeClassifier, XceptionDeepfakeClassifier, build_model
    import torch
    
    model = build_model(pretrained=False, dropout=0.3, device=torch.device("cpu"))
    assert isinstance(model, (DeepfakeClassifier, XceptionDeepfakeClassifier))


def test_forensic_engine_imports():
    """Test forensic engine imports work."""
    from forensic_engine import analyze_frame_predictions
    assert callable(analyze_frame_predictions)


def test_explainability_imports():
    """Test explainability imports work."""
    from explainability import GradCAM, generate_explanations_for_video
    assert callable(GradCAM)
    assert callable(generate_explanations_for_video)


def test_robustness_imports():
    """Test robustness imports work."""
    from robustness import run_robustness_tests
    assert callable(run_robustness_tests)


def test_report_imports():
    """Test report imports work."""
    from report import generate_html_report, save_json_report
    assert callable(generate_html_report)
    assert callable(save_json_report)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])