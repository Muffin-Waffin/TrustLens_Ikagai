"""
Tests for data_loader module.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from data_loader import (
    DeepfakeFrameDataset,
    get_transforms,
    compute_class_weights,
    validate_split_integrity,
    set_seed,
)


def test_get_transforms_train():
    transform = get_transforms(224, is_train=True)
    assert transform is not None


def test_get_transforms_eval():
    transform = get_transforms(224, is_train=False)
    assert transform is not None


def test_compute_class_weights():
    with tempfile.TemporaryDirectory() as tmpdir:
        faces_root = Path(tmpdir) / "faces"
        metadata_root = Path(tmpdir) / "metadata"
        faces_root.mkdir()
        metadata_root.mkdir()

        # Create mock metadata
        import json
        meta = {
            "frames": [
                {"face_path": "faces/v1/frame_000000.jpg", "usable": True, "frame_index": 0, "timestamp_seconds": 0.0},
                {"face_path": "faces/v1/frame_000015.jpg", "usable": True, "frame_index": 15, "timestamp_seconds": 0.5},
            ]
        }
        (metadata_root / "v1.json").write_text(json.dumps(meta))

        # Create dummy face images
        import cv2
        import numpy as np
        (faces_root / "v1").mkdir()
        for f in ["frame_000000.jpg", "frame_000015.jpg"]:
            img = np.zeros((224, 224, 3), dtype=np.uint8)
            cv2.imwrite(str(faces_root / "v1" / f), img)

        df = pd.DataFrame([{"video_id": "v1", "label": 1, "manipulation_type": "DeepFakes", "split": "train"}])

        dataset = DeepfakeFrameDataset(df, faces_root, metadata_root)
        weights = compute_class_weights(dataset)
        assert "pos_weight" in weights


def test_validate_split_integrity_ok():
    train = pd.DataFrame({"video_id": ["v1", "v2"], "label": [0, 1]})
    val = pd.DataFrame({"video_id": ["v3"], "label": [0]})
    test = pd.DataFrame({"video_id": ["v4"], "label": [1]})
    validate_split_integrity(train, val, test)


def test_validate_split_integrity_fails():
    train = pd.DataFrame({"video_id": ["v1", "v2"], "label": [0, 1]})
    val = pd.DataFrame({"video_id": ["v2"], "label": [0]})  # v2 in both
    test = pd.DataFrame({"video_id": ["v3"], "label": [1]})
    with pytest.raises(ValueError, match="Split leakage"):
        validate_split_integrity(train, val, test)


def test_set_seed():
    set_seed(42)
    # Should not raise


if __name__ == "__main__":
    pytest.main([__file__, "-v"])