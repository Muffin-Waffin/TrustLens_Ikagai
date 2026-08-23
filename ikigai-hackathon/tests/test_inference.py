"""
Tests for inference module.
"""

import pytest
import torch
import numpy as np

from model import DeepfakeClassifier, build_model


def test_batch_inference():
    model = DeepfakeClassifier(pretrained=False, dropout=0.3)
    model.eval()
    device = torch.device("cpu")
    model.to(device)

    batch = torch.randn(4, 3, 224, 224).to(device)
    with torch.no_grad():
        logits = model(batch)
        probs = torch.sigmoid(logits).cpu().numpy()

    assert probs.shape == (4,)
    assert np.all(probs >= 0) and np.all(probs <= 1)


def test_single_inference():
    model = DeepfakeClassifier(pretrained=False, dropout=0.3)
    model.eval()
    device = torch.device("cpu")
    model.to(device)

    x = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        logits = model(x)
        prob = torch.sigmoid(logits).item()

    assert 0 <= prob <= 1


def test_model_device_assignment():
    model = build_model(pretrained=False, dropout=0.3, device=torch.device("cpu"))
    assert next(model.parameters()).device.type == "cpu"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])