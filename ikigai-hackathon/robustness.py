"""
SynthGuard Phase 4: Robustness Testing Module

Tests model stability under common video transformations.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from model import DeepfakeClassifier
from preprocessing import (
    get_video_metadata,
    sample_frame_indices,
    initialize_face_detector,
    expand_bbox,
    compute_blur_score,
    compute_face_quality,
)
from forensic_engine import analyze_frame_predictions, ForensicResult


def compute_sha256(file_path: str | Path) -> str:
    """Compute SHA-256 hash of file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def apply_resize(frame: np.ndarray, scale: float = 0.75) -> np.ndarray:
    """Resize frame down and back up."""
    h, w = frame.shape[:2]
    new_w, new_h = int(w * scale), int(h * scale)
    down = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    up = cv2.resize(down, (w, h), interpolation=cv2.INTER_LINEAR)
    return up


def apply_blur(frame: np.ndarray, kernel: int = 3, sigma: float = 0.8) -> np.ndarray:
    """Apply mild Gaussian blur."""
    return cv2.GaussianBlur(frame, (kernel, kernel), sigma)


def apply_jpeg_compression(frame: np.ndarray, quality: int = 75) -> np.ndarray:
    """Apply JPEG compression artifact simulation."""
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, enc = cv2.imencode(".jpg", frame, encode_param)
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def apply_brightness(frame: np.ndarray, factor: float = 0.8) -> np.ndarray:
    """Adjust brightness."""
    return cv2.convertScaleAbs(frame, alpha=factor, beta=0)


TRANSFORM_FUNCTIONS = {
    "resize": apply_resize,
    "blur": apply_blur,
    "jpeg_compression": apply_jpeg_compression,
    "brightness": apply_brightness,
}


def transform_video(
    input_path: str | Path,
    output_path: str | Path,
    transform_name: str,
    params: dict[str, Any],
) -> None:
    """
    Apply transformation to entire video.

    Args:
        input_path: Source video
        output_path: Destination video
        transform_name: Name of transformation
        params: Transform parameters
    """
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Use mp4v codec (more compatible)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not out.isOpened():
        # Try alternative codec
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        if not out.isOpened():
            # Try MJPG
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    transform_fn = TRANSFORM_FUNCTIONS[transform_name]

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            transformed = transform_fn(frame, **params)
            out.write(transformed)
    finally:
        cap.release()
        out.release()


def process_video_for_robustness(
    video_path: str | Path,
    config: dict[str, Any],
    model: DeepfakeClassifier,
    face_app,
    device: torch.device,
) -> ForensicResult:
    """Run full pipeline on a video and return forensic result."""
    from inference import preprocess_video_for_inference, run_inference
    
    video_path = Path(video_path)
    face_crops, frame_infos, _ = preprocess_video_for_inference(video_path, config, face_app)
    scores = run_inference(face_crops, model, config, device)
    
    score_idx = 0
    for fi in frame_infos:
        if fi.get("usable", False):
            fi["score"] = float(scores[score_idx])
            score_idx += 1
    
    return analyze_frame_predictions(frame_infos, video_path.stem, config)


def run_robustness_tests(
    video_path: str | Path,
    config: dict[str, Any],
    model: DeepfakeClassifier,
    face_app,
    device: torch.device,
) -> dict[str, Any]:
    """
    Run robustness tests on a video.
    
    Returns dict with original and transformed results.
    """
    if not config.get("robustness", {}).get("enabled", True):
        return {"enabled": False}
    
    video_path = Path(video_path)
    transforms_config = config["robustness"]["transforms"]
    
    print(f"[INFO] Running robustness tests on {video_path.name}")
    
    # Process original
    print("[INFO] Processing original video...")
    original_result = process_video_for_robustness(video_path, config, model, face_app, device)
    original_score = original_result.manipulation_score
    
    results = {
        "video_id": video_path.stem,
        "sha256": compute_sha256(video_path),
        "original_score": original_score,
        "tests": [],
    }
    
    # Test each transformation
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        for transform_config in transforms_config:
            transform_name = transform_config["name"]
            params = transform_config.get("params", {})
            
            print(f"[INFO] Testing transformation: {transform_name}")
            
            transformed_path = tmpdir_path / f"{video_path.stem}_{transform_name}.mp4"
            
            try:
                transform_video(video_path, transformed_path, transform_name, params)
                transformed_result = process_video_for_robustness(
                    transformed_path, config, model, face_app, device
                )
                transformed_score = transformed_result.manipulation_score
                
                difference = abs(original_score - transformed_score)
                stability = 1.0 - difference
                stability = max(0.0, min(1.0, stability))
                
                test_result = {
                    "transform": transform_name,
                    "params": params,
                    "score": transformed_score,
                    "difference": difference,
                    "stability": stability,
                }
                
                results["tests"].append(test_result)
                print(f"  Score: {transformed_score:.4f}, Diff: {difference:.4f}, Stability: {stability:.4f}")
                
            except Exception as e:
                print(f"[WARN] Transformation {transform_name} failed: {e}")
                results["tests"].append({
                    "transform": transform_name,
                    "params": params,
                    "error": str(e),
                })
    
    # Compute overall stability
    valid_tests = [t for t in results["tests"] if "stability" in t]
    if valid_tests:
        overall = float(np.mean([t["stability"] for t in valid_tests]))
        results["overall_stability"] = overall
        
        # Interpretation
        thresholds = config["robustness"]["thresholds"]
        if overall >= thresholds["high"]:
            results["interpretation"] = "Evidence appears stable under tested transformations."
        elif overall >= thresholds["medium"]:
            results["interpretation"] = "Evidence shows moderate stability."
        else:
            results["interpretation"] = "Evidence is sensitive to media transformations."
    else:
        results["overall_stability"] = 0.0
        results["interpretation"] = "No valid transformations tested."
    
    return results


def save_robustness_report(results: dict[str, Any], output_dir: Path) -> None:
    """Save robustness results to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    video_id = results.get("video_id", "unknown")
    with open(output_dir / f"{video_id}_robustness.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"[INFO] Robustness report saved: {output_dir / f'{video_id}_robustness.json'}")


if __name__ == "__main__":
    # Quick test of transforms
    import sys
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1])
        if img is not None:
            print("Testing transforms...")
            resized = apply_resize(img, 0.75)
            blurred = apply_blur(img)
            jpeg = apply_jpeg_compression(img)
            bright = apply_brightness(img, 0.8)
            print("All transforms work!")