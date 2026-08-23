"""
Tests for explainability module.
"""

import pytest
import torch
import numpy as np

from model import DeepfakeClassifier
from explainability import GradCAM, apply_colormap, create_overlay


def test_gradcam_initialization():
    """Test GradCAM can be initialized on ConvNeXt."""
    model = DeepfakeClassifier(pretrained=False, dropout=0.3)
    gradcam = GradCAM(model, target_layer_name="features.7.2")
    assert gradcam is not None


def test_gradcam_generate_cam():
    """Test GradCAM generates heatmap of correct shape."""
    model = DeepfakeClassifier(pretrained=False, dropout=0.3)
    model.eval()
    
    gradcam = GradCAM(model, target_layer_name="features.7.2")
    
    input_tensor = torch.randn(1, 3, 224, 224)
    cam = gradcam.generate_cam(input_tensor)
    
    assert isinstance(cam, np.ndarray)
    assert cam.shape == (224, 224)
    assert cam.min() >= 0.0
    assert cam.max() <= 1.0


def test_apply_colormap():
    """Test colormap application."""
    heatmap = np.random.rand(224, 224).astype(np.float32)
    colored = apply_colormap(heatmap, "jet")
    
    assert colored.shape == (224, 224, 3)
    assert colored.dtype == np.uint8


def test_create_overlay():
    """Test overlay creation."""
    original = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    heatmap = np.random.rand(224, 224).astype(np.float32)
    
    overlay = create_overlay(original, heatmap, alpha=0.5)
    
    assert overlay.shape == (224, 224, 3)
    assert overlay.dtype == np.uint8


def test_gradcam_different_layers():
    """Test GradCAM works with different target layers."""
    model = DeepfakeClassifier(pretrained=False, dropout=0.3)
    
    for layer in ["features.7.2", "features.7.1", "features.7.0"]:
        try:
            gradcam = GradCAM(model, target_layer_name=layer)
            input_tensor = torch.randn(1, 3, 224, 224)
            cam = gradcam.generate_cam(input_tensor)
            assert cam.shape == (224, 224)
        except Exception as e:
            pytest.fail(f"GradCAM failed for layer {layer}: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])