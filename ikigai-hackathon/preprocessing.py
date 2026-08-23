"""
SynthGuard Phase 1: Video Preprocessing Pipeline

Implements:
- Video metadata extraction
- Smart frame sampling
- Face detection (InsightFace buffalo_s)
- Face cropping and alignment
- Face quality estimation
- Metadata JSON generation
- Batch processing of dataset splits
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import yaml
from insightface.app import FaceAnalysis
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)


def load_config(config_path: str | Path) -> dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_video_metadata(video_path: str | Path) -> dict[str, Any]:
    """Extract video metadata using OpenCV."""
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if frame_count <= 0:
            raise ValueError("Video has zero or negative frame count")
        if width <= 0 or height <= 0:
            raise ValueError("Video has invalid dimensions")

        duration = frame_count / fps if fps > 0 else 0.0

        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "unknown"
        if fourcc_int != 0:
            codec = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)]).strip()

        return {
            "filename": path.name,
            "fps": float(fps),
            "frame_count": frame_count,
            "duration_seconds": float(duration),
            "width": width,
            "height": height,
            "codec": codec if codec else "unknown",
        }
    finally:
        cap.release()


def sample_frame_indices(
    metadata: dict[str, Any], sample_fps: float, max_frames: int
) -> list[int]:
    """Sample frame indices at target FPS."""
    total_frames = metadata["frame_count"]
    fps = metadata["fps"]

    if fps <= 0:
        return []

    interval = max(1, int(round(fps / sample_fps)))
    indices = list(range(0, total_frames, interval))

    if len(indices) > max_frames:
        step = len(indices) / max_frames
        indices = [indices[int(i * step)] for i in range(max_frames)]

    return indices


def initialize_face_detector(config: dict[str, Any]) -> FaceAnalysis:
    """Initialize InsightFace detector with appropriate ONNX provider."""
    providers = ["CPUExecutionProvider"]
    if config["runtime"]["prefer_cuda"]:
        import onnxruntime as ort
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    print(f"[INFO] ONNX providers: {providers}")

    # buffalo models include recognition; request it explicitly to retain bbox,
    # landmarks (when available), and normalized identity embeddings.
    app = FaceAnalysis(name=config["face"]["model"], allowed_modules=["detection", "recognition"], providers=providers)
    app.prepare(ctx_id=0 if "CUDAExecutionProvider" in providers else -1, det_size=(640, 640))

    provider_name = "CUDAExecutionProvider" if "CUDAExecutionProvider" in providers else "CPUExecutionProvider"
    print(f"[INFO] Face detector provider: {provider_name}")

    return app


def expand_bbox(
    bbox: np.ndarray, expansion_ratio: float, frame_width: int, frame_height: int
) -> tuple[int, int, int, int]:
    """Expand bounding box by ratio and clamp to frame boundaries."""
    x1, y1, x2, y2 = bbox.astype(float)
    w = x2 - x1
    h = y2 - y1

    exp_w = w * expansion_ratio
    exp_h = h * expansion_ratio

    x1 = max(0, int(x1 - exp_w))
    y1 = max(0, int(y1 - exp_h))
    x2 = min(frame_width, int(x2 + exp_w))
    y2 = min(frame_height, int(y2 + exp_h))

    return x1, y1, x2, y2


def compute_blur_score(image: np.ndarray) -> float:
    """Compute blur score using variance of Laplacian."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_face_quality(
    face_confidence: float,
    face_area_ratio: float,
    blur_score: float,
) -> float:
    """
    Compute heuristic face quality score in [0, 1].
    
    THIS IS A HEURISTIC SCORE, NOT A CALIBRATED PROBABILITY.
    
    Components:
    - confidence_component: clip(face_confidence, 0, 1)
    - area_component: clip(face_area_ratio / 0.10, 0, 1)
    - blur_component: clip(blur_score / 100.0, 0, 1)
    
    Weighted: 0.45 * confidence + 0.30 * area + 0.25 * blur
    """
    confidence_component = float(np.clip(face_confidence, 0.0, 1.0))
    area_component = float(np.clip(face_area_ratio / 0.10, 0.0, 1.0))
    blur_component = float(np.clip(blur_score / 100.0, 0.0, 1.0))

    quality = (
        0.45 * confidence_component
        + 0.30 * area_component
        + 0.25 * blur_component
    )
    return float(np.clip(quality, 0.0, 1.0))


def process_single_video(
    video_info: dict[str, Any],
    config: dict[str, Any],
    face_app: FaceAnalysis,
    paths: dict[str, Path]
) -> dict[str, Any]:
    """Process a single video through the full pipeline."""
    video_path = paths["raw"] / video_info["video_path"]
    video_id = video_info["video_id"]
    
    metadata = get_video_metadata(video_path)
    
    sample_fps = config["sampling"]["fps"]
    max_frames = config["sampling"]["max_frames"]
    frame_indices = sample_frame_indices(metadata, sample_fps, max_frames)

    frames_dir = paths["frames"] / video_id
    faces_dir = paths["faces"] / video_id
    metadata_dir = paths["metadata"]
    
    frames_dir.mkdir(parents=True, exist_ok=True)
    faces_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    project_root = Path(".")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video for processing: {video_path}")

    frames_data = []
    frames_with_faces = 0
    usable_face_frames = 0
    quality_sum = 0.0

    min_confidence = config["face"]["min_confidence"]
    min_area_ratio = config["face"]["min_face_area_ratio"]
    expansion_ratio = config["face"]["expansion_ratio"]
    target_size = config["image"]["size"]
    quality_threshold = config["quality"]["minimum_score"]

    try:
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if not ret or frame is None:
                frames_data.append({
                    "frame_index": frame_idx,
                    "timestamp_seconds": frame_idx / metadata["fps"] if metadata["fps"] > 0 else 0.0,
                    "face_found": False,
                    "face_confidence": 0.0,
                    "face_area_ratio": 0.0,
                    "blur_score": 0.0,
                    "face_quality": 0.0,
                    "usable": False,
                    "bbox": [0, 0, 0, 0],
                    "frame_path": "",
                    "face_path": "",
                })
                continue

            frame_filename = f"frame_{frame_idx:06d}.jpg"
            frame_path = frames_dir / frame_filename
            cv2.imwrite(str(frame_path), frame)

            faces = face_app.get(frame)

            if faces:
                primary_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

                if primary_face.det_score < min_confidence:
                    face_found = False
                else:
                    x1, y1, x2, y2 = expand_bbox(
                        primary_face.bbox, expansion_ratio, metadata["width"], metadata["height"]
                    )

                    face_w = x2 - x1
                    face_h = y2 - y1
                    face_area = face_w * face_h
                    frame_area = metadata["width"] * metadata["height"]
                    face_area_ratio = face_area / frame_area if frame_area > 0 else 0.0

                    if face_area_ratio < min_area_ratio:
                        face_found = False
                    else:
                        face_found = True
                        frames_with_faces += 1

                        face_crop = frame[y1:y2, x1:x2]
                        if face_crop.size > 0:
                            face_crop_resized = cv2.resize(face_crop, (target_size, target_size), interpolation=cv2.INTER_AREA)
                            face_filename = f"frame_{frame_idx:06d}.jpg"
                            face_path = faces_dir / face_filename
                            cv2.imwrite(str(face_path), face_crop_resized)

                            blur_score = compute_blur_score(face_crop_resized)
                            face_quality = compute_face_quality(
                                primary_face.det_score, face_area_ratio, blur_score
                            )

                            usable = face_quality >= quality_threshold
                            if usable:
                                usable_face_frames += 1
                                quality_sum += face_quality

                            frames_data.append({
                                "frame_index": frame_idx,
                                "timestamp_seconds": frame_idx / metadata["fps"] if metadata["fps"] > 0 else 0.0,
                                "face_found": True,
                                "face_confidence": float(primary_face.det_score),
                                "face_area_ratio": float(face_area_ratio),
                                "blur_score": float(blur_score),
                                "face_quality": float(face_quality),
                                "usable": usable,
                                "bbox": [x1, y1, x2, y2],
                                "landmarks": primary_face.kps.tolist() if getattr(primary_face, "kps", None) is not None else None,
                                "embedding": primary_face.normed_embedding.tolist() if getattr(primary_face, "normed_embedding", None) is not None else None,
                                "frame_path": str(frame_path.relative_to(project_root)),
                                "face_path": str(face_path.relative_to(project_root)),
                            })
                            continue

            frames_data.append({
                "frame_index": frame_idx,
                "timestamp_seconds": frame_idx / metadata["fps"] if metadata["fps"] > 0 else 0.0,
                "face_found": False,
                "face_confidence": 0.0,
                "face_area_ratio": 0.0,
                "blur_score": 0.0,
                "face_quality": 0.0,
                "usable": False,
                "bbox": [0, 0, 0, 0],
                "frame_path": str(frame_path.relative_to(project_root)),
                "face_path": "",
            })
    finally:
        cap.release()

    sampled_count = len(frame_indices)
    coverage = frames_with_faces / sampled_count if sampled_count > 0 else 0.0
    avg_quality = quality_sum / usable_face_frames if usable_face_frames > 0 else 0.0

    if frames_with_faces == 0:
        print(f"[WARN] {video_id}: No usable faces detected.")

    provider_name = face_app.det_model.session.get_providers()[0] if hasattr(face_app, 'det_model') and hasattr(face_app.det_model, 'session') else "CPUExecutionProvider"

    result = {
        "dataset": {
            "name": "FaceForensicsC23",
            "source": "bitmind/FaceForensicsC23"
        },
        "label": video_info["label"],
        "manipulation_type": video_info["manipulation_type"],
        "video": metadata,
        "sampling": {
            "requested_fps": sample_fps,
            "max_frames": max_frames,
            "sampled_frames": sampled_count,
        },
        "face_processing": {
            "frames_with_faces": frames_with_faces,
            "usable_face_frames": usable_face_frames,
            "coverage": float(coverage),
            "average_face_quality": float(avg_quality),
        },
        "frames": frames_data,
        "runtime": {
            "device": "CUDA" if "CUDA" in provider_name else "CPU",
            "face_provider": provider_name,
        },
    }

    metadata_path = metadata_dir / f"{video_id}.json"
    with open(metadata_path, "w") as f:
        json.dump(result, f, indent=2)

    return {
        "video_id": video_id,
        "frames_processed": sampled_count,
        "faces_found": frames_with_faces,
        "usable_faces": usable_face_frames,
        "coverage": coverage,
        "avg_quality": avg_quality,
        "metadata_path": str(metadata_path),
    }


def process_dataset_split(
    split_name: str,
    config: dict[str, Any],
    face_app: FaceAnalysis,
    paths: dict[str, Path]
) -> list[dict[str, Any]]:
    """Process all videos in a dataset split."""
    splits_dir = paths["splits"]
    split_path = splits_dir / f"{split_name}.csv"
    
    if not split_path.exists():
        print(f"[WARN] Split not found: {split_path}")
        return []
    
    df = pd.read_csv(split_path)
    print(f"[INFO] Processing {split_name} split: {len(df)} videos")
    
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {split_name}", unit="video"):
        try:
            result = process_single_video(row.to_dict(), config, face_app, paths)
            results.append(result)
        except Exception as e:
            print(f"[ERROR] Failed to process {row['video_id']}: {e}")
            results.append({
                "video_id": row["video_id"],
                "error": str(e),
            })
    
    return results


def process_video_cli(
    video_path: str | Path,
    config_path: str | Path = "config.yaml"
) -> dict[str, Any]:
    """Process a single video from CLI."""
    config = load_config(config_path)
    
    print(f"[INFO] Video: {Path(video_path).name}")
    
    metadata = get_video_metadata(video_path)
    print(f"[INFO] Resolution: {metadata['width']}x{metadata['height']}")
    print(f"[INFO] FPS: {metadata['fps']:.2f}")
    print(f"[INFO] Duration: {metadata['duration_seconds']:.2f} sec")
    print(f"[INFO] Total frames: {metadata['frame_count']}")

    sample_fps = config["sampling"]["fps"]
    max_frames = config["sampling"]["max_frames"]
    frame_indices = sample_frame_indices(metadata, sample_fps, max_frames)

    print(f"[INFO] Sampling: {sample_fps} FPS")
    print(f"[INFO] Candidate frames: {len(frame_indices)}")

    face_app = initialize_face_detector(config)

    video_id = Path(video_path).stem
    paths = {
        "raw": Path("."),
        "frames": Path(config["paths"]["frames"]),
        "faces": Path(config["paths"]["faces"]),
        "metadata": Path(config["paths"]["metadata"]),
        "splits": Path(config["paths"]["splits"]),
    }

    video_info = {
        "video_id": video_id,
        "video_path": str(video_path),
        "label": -1,
        "manipulation_type": "Unknown",
    }

    return process_single_video(video_info, config, face_app, paths)


def main():
    parser = argparse.ArgumentParser(description="SynthGuard Phase 1: Video Preprocessing")
    parser.add_argument("--input", help="Path to input video file")
    parser.add_argument("--dataset", action="store_true", help="Process entire dataset splits")
    parser.add_argument("--split", default="all", choices=["train", "val", "test", "all"], help="Split to process")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    paths = {k: Path(v) for k, v in config["paths"].items()}

    face_app = initialize_face_detector(config)

    if args.input:
        # Single video mode
        try:
            result = process_video_cli(args.input, args.config)
            print(f"\n[INFO] Frames processed: {result['face_processing']['frames_with_faces'] + sum(1 for f in result['frames'] if not f['face_found'])}")
            print(f"[INFO] Faces found: {result['face_processing']['frames_with_faces']}")
            print(f"[INFO] Usable face frames: {result['face_processing']['usable_face_frames']}")
            print(f"[INFO] Face coverage: {result['face_processing']['coverage'] * 100:.1f}%")
            print(f"[INFO] Average face quality: {result['face_processing']['average_face_quality']:.2f}")
            print(f"[INFO] Metadata saved to: {result.get('metadata_path', 'outputs/metadata/')}")
            print("[INFO] PHASE 1 COMPLETE")
        except Exception as e:
            print(f"[ERROR] {e}", file=sys.stderr)
            sys.exit(1)

    elif args.dataset:
        # Dataset mode
        splits_to_process = ["train", "val", "test"] if args.split == "all" else [args.split]
        all_results = []
        
        for split in splits_to_process:
            results = process_dataset_split(split, config, face_app, paths)
            all_results.extend(results)
        
        successful = [r for r in all_results if "error" not in r]
        failed = [r for r in all_results if "error" in r]
        
        print(f"\n[INFO] Dataset processing complete:")
        print(f"[INFO]   Successful: {len(successful)}")
        print(f"[INFO]   Failed: {len(failed)}")
        
        if successful:
            total_frames = sum(r["frames_processed"] for r in successful)
            total_faces = sum(r["faces_found"] for r in successful)
            total_usable = sum(r["usable_faces"] for r in successful)
            avg_coverage = sum(r["coverage"] for r in successful) / len(successful)
            avg_quality = sum(r["avg_quality"] for r in successful) / len(successful)
            
            print(f"[INFO]   Total frames processed: {total_frames}")
            print(f"[INFO]   Total faces found: {total_faces}")
            print(f"[INFO]   Total usable faces: {total_usable}")
            print(f"[INFO]   Average coverage: {avg_coverage * 100:.1f}%")
            print(f"[INFO]   Average quality: {avg_quality:.2f}")
        
        print("[INFO] PHASE 1 COMPLETE")

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
