"""
Test that all required dependencies can be imported.
"""

import sys

def test_opencv_import():
    import cv2
    assert hasattr(cv2, "__version__")
    print(f"OpenCV version: {cv2.__version__}")

def test_numpy_import():
    import numpy as np
    assert hasattr(np, "__version__")
    print(f"NumPy version: {np.__version__}")

def test_pandas_import():
    import pandas as pd
    assert hasattr(pd, "__version__")
    print(f"Pandas version: {pd.__version__}")

def test_insightface_import():
    import insightface
    from insightface.app import FaceAnalysis
    assert FaceAnalysis is not None
    print("InsightFace imported successfully")

def test_onnxruntime_import():
    import onnxruntime as ort
    providers = ort.get_available_providers()
    assert isinstance(providers, list)
    print(f"ONNX Runtime providers: {providers}")

def test_yaml_import():
    import yaml
    assert yaml is not None
    print("PyYAML imported successfully")

def test_tqdm_import():
    from tqdm import tqdm
    assert tqdm is not None
    print("tqdm imported successfully")

def test_huggingface_hub_import():
    import huggingface_hub
    assert huggingface_hub is not None
    print(f"HuggingFace Hub version: {huggingface_hub.__version__}")

def test_datasets_import():
    import datasets
    assert datasets is not None
    print(f"Datasets version: {datasets.__version__}")

def test_preprocessing_import():
    import preprocessing
    assert hasattr(preprocessing, "process_video_cli")
    assert hasattr(preprocessing, "get_video_metadata")
    assert hasattr(preprocessing, "sample_frame_indices")
    assert hasattr(preprocessing, "compute_face_quality")
    print("preprocessing module imported successfully")

def test_dataset_download_import():
    import dataset_download
    assert hasattr(dataset_download, "download_videos")
    assert hasattr(dataset_download, "select_prototype_videos")
    print("dataset_download module imported successfully")

def test_dataset_prepare_import():
    import dataset_prepare
    assert hasattr(dataset_prepare, "create_splits")
    assert hasattr(dataset_prepare, "validate_videos")
    print("dataset_prepare module imported successfully")

if __name__ == "__main__":
    test_opencv_import()
    test_numpy_import()
    test_pandas_import()
    test_insightface_import()
    test_onnxruntime_import()
    test_yaml_import()
    test_tqdm_import()
    test_huggingface_hub_import()
    test_datasets_import()
    test_preprocessing_import()
    test_dataset_download_import()
    test_dataset_prepare_import()
    print("\nAll imports successful!")