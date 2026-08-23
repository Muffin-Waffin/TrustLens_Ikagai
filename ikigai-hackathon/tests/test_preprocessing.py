"""
Tests for preprocessing module.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pandas as pd
import pytest
import yaml

import preprocessing
import dataset_prepare
import dataset_download


class TestSampleFrameIndices:
    def test_basic_sampling(self):
        metadata = {"frame_count": 300, "fps": 30.0}
        indices = preprocessing.sample_frame_indices(metadata, sample_fps=2.0, max_frames=120)
        assert len(indices) == 20
        assert indices[0] == 0
        assert indices[-1] == 285
        assert all(indices[i] < indices[i + 1] for i in range(len(indices) - 1))

    def test_max_frames_limit(self):
        metadata = {"frame_count": 10000, "fps": 30.0}
        indices = preprocessing.sample_frame_indices(metadata, sample_fps=2.0, max_frames=10)
        assert len(indices) == 10

    def test_no_duplicates(self):
        metadata = {"frame_count": 100, "fps": 30.0}
        indices = preprocessing.sample_frame_indices(metadata, sample_fps=2.0, max_frames=120)
        assert len(indices) == len(set(indices))

    def test_within_valid_range(self):
        metadata = {"frame_count": 100, "fps": 30.0}
        indices = preprocessing.sample_frame_indices(metadata, sample_fps=2.0, max_frames=120)
        assert all(0 <= idx < 100 for idx in indices)

    def test_zero_fps(self):
        metadata = {"frame_count": 100, "fps": 0.0}
        indices = preprocessing.sample_frame_indices(metadata, sample_fps=2.0, max_frames=120)
        assert indices == []


class TestFaceQuality:
    def test_quality_in_range(self):
        quality = preprocessing.compute_face_quality(0.9, 0.15, 200.0)
        assert 0.0 <= quality <= 1.0

    def test_quality_clamping(self):
        quality = preprocessing.compute_face_quality(1.5, 2.0, 500.0)
        assert 0.0 <= quality <= 1.0

    def test_quality_components(self):
        quality = preprocessing.compute_face_quality(0.0, 0.0, 0.0)
        assert quality == 0.0

    def test_quality_high_values(self):
        quality = preprocessing.compute_face_quality(1.0, 0.20, 200.0)
        assert quality == 1.0


class TestExpandBBox:
    def test_expansion_and_clamping(self):
        bbox = np.array([100, 100, 200, 200], dtype=np.float32)
        x1, y1, x2, y2 = preprocessing.expand_bbox(bbox, 0.2, 300, 300)
        assert x1 == 80
        assert y1 == 80
        assert x2 == 220
        assert y2 == 220

    def test_clamp_at_boundaries(self):
        bbox = np.array([10, 10, 50, 50], dtype=np.float32)
        x1, y1, x2, y2 = preprocessing.expand_bbox(bbox, 0.5, 100, 100)
        assert x1 == 0
        assert y1 == 0
        assert x2 == 70
        assert y2 == 70


class TestBlurScore:
    def test_blur_score_returns_float(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        score = preprocessing.compute_blur_score(img)
        assert isinstance(score, float)
        assert score >= 0.0

    def test_sharp_image_higher_score(self):
        sharp = np.zeros((100, 100, 3), dtype=np.uint8)
        sharp[::2, ::2] = 255
        blur = cv2.GaussianBlur(sharp, (15, 15), 0)

        sharp_score = preprocessing.compute_blur_score(sharp)
        blur_score = preprocessing.compute_blur_score(blur)

        assert sharp_score > blur_score


class TestGetVideoMetadata:
    def test_invalid_path_raises(self):
        with pytest.raises(FileNotFoundError):
            preprocessing.get_video_metadata("nonexistent.mp4")

    def test_creates_synthetic_video_for_testing(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(tmp_path, fourcc, 30.0, (640, 480))
            for _ in range(90):
                frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                out.write(frame)
            out.release()

            metadata = preprocessing.get_video_metadata(tmp_path)
            assert metadata["filename"] == Path(tmp_path).name
            assert metadata["fps"] == 30.0
            assert metadata["frame_count"] == 90
            assert metadata["width"] == 640
            assert metadata["height"] == 480
            assert metadata["duration_seconds"] == pytest.approx(3.0, rel=0.1)
        finally:
            Path(tmp_path).unlink(missing_ok=True)


class TestCreateSplits:
    def test_split_ratios(self):
        df = pd.DataFrame({"video_id": [f"v{i}" for i in range(100)], "label": [0]*50 + [1]*50})
        splits = dataset_prepare.create_splits(df, 0.7, 0.15, 0.15, 42)
        assert len(splits["train"]) == 70
        assert len(splits["val"]) == 15
        assert len(splits["test"]) == 15
        assert len(splits["all"]) == 100

    def test_no_overlap(self):
        df = pd.DataFrame({"video_id": [f"v{i}" for i in range(20)], "label": [0]*10 + [1]*10})
        splits = dataset_prepare.create_splits(df, 0.7, 0.15, 0.15, 42)
        train_ids = set(splits["train"]["video_id"])
        val_ids = set(splits["val"]["video_id"])
        test_ids = set(splits["test"]["video_id"])
        assert train_ids.isdisjoint(val_ids)
        assert train_ids.isdisjoint(test_ids)
        assert val_ids.isdisjoint(test_ids)

    def test_deterministic(self):
        df = pd.DataFrame({"video_id": [f"v{i}" for i in range(20)], "label": [0]*10 + [1]*10})
        splits1 = dataset_prepare.create_splits(df, 0.7, 0.15, 0.15, 42)
        splits2 = dataset_prepare.create_splits(df, 0.7, 0.15, 0.15, 42)
        assert list(splits1["train"]["video_id"]) == list(splits2["train"]["video_id"])


class TestSelectPrototypeVideos:
    def test_selection_counts(self):
        structure = {
            "real": [f"real/v{i}.mp4" for i in range(100)],
            "deepfakes": [f"deepfakes/v{i}.mp4" for i in range(100)],
            "face2face": [f"face2face/v{i}.mp4" for i in range(100)],
        }
        prototype = {"real": 10, "deepfakes": 20, "face2face": 5, "faceswap": 0, "neuraltextures": 0}
        selected = dataset_download.select_prototype_videos(structure, prototype, 42)
        assert len(selected) == 35
        real_count = sum(1 for _, label, _ in selected if label == 0)
        fake_count = sum(1 for _, label, _ in selected if label == 1)
        assert real_count == 10
        assert fake_count == 25


class TestConfigLoading:
    def test_load_config(self):
        config_data = {
            "dataset": {"source": "huggingface", "name": "test", "prototype": {"real": 10}, "random_seed": 42},
            "sampling": {"fps": 2.0, "max_frames": 120},
            "face": {"model": "buffalo_s", "min_confidence": 0.5, "min_face_area_ratio": 0.01, "expansion_ratio": 0.2},
            "image": {"size": 224},
            "quality": {"minimum_score": 0.35},
            "runtime": {"prefer_cuda": True},
            "paths": {"raw": "./data/raw", "subset": "./data/subset", "splits": "./data/splits", "processed": "./data/processed", "frames": "./outputs/frames", "faces": "./outputs/faces", "metadata": "./outputs/metadata"},
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            yaml.dump(config_data, tmp)
            tmp_path = tmp.name
        try:
            loaded = preprocessing.load_config(tmp_path)
            assert loaded == config_data
        finally:
            Path(tmp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])