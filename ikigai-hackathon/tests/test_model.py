"""
Tests for model module.
"""

import pytest
import torch

from model import DeepfakeClassifier, build_model


def test_model_initializes():
    model = DeepfakeClassifier(pretrained=False, dropout=0.3)
    assert isinstance(model, DeepfakeClassifier)


def test_model_output_shape():
    model = DeepfakeClassifier(pretrained=False, dropout=0.3)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2,)


def test_model_forward_pass():
    model = DeepfakeClassifier(pretrained=False, dropout=0.3)
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert isinstance(out, torch.Tensor)
    assert out.numel() == 1


def test_build_model_cpu():
    model = build_model(pretrained=False, dropout=0.3, device=torch.device("cpu"))
    assert model is not None
    assert next(model.parameters()).device.type == "cpu"


def test_build_model_pretrained_false():
    model = DeepfakeClassifier(pretrained=False, dropout=0.3)
    assert model is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])