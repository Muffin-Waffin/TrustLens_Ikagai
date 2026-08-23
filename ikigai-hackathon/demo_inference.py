"""
SynthGuard Phase 4: Demo Inference Script

Runs full pipeline on a single video using pretrained ImageNet weights (no fine-tuning).
This is for demonstration purposes only - results will be random without fine-tuning.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from model import DeepfakeClassifier, build_model
from preprocessing import (
    get_video_metadata,
    sample_frame_indices,
    initialize_face_detector,
    expand_bbox,
    compute_blur_score,
    compute_face_quality,
    load_config,
)
from forensic_engine import (
    analyze_frame_predictions,
    save_forensic_result,
    create_timeline_plot,
    create_contact_sheet,
    ForensicResult,
)
from explainability import (
    generate_explanations_for_video,
    create_enhanced_timeline,
)
from robustness import run_robustness_tests, save_robustness_report
from report import generate_html_report, save_json_report


def load_config(config_path: str | Path) -> dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def preprocess_video_for_inference(
    video_path: str | Path,
    config: dict[str, Any],
    face_app,
) -> tuple[list[np.ndarray], list[dict], dict]:
    """
    Run Phase 1 preprocessing on a single video.
    Returns face crops and metadata.
    """
    video_path = Path(video_path)
    metadata = get_video_metadata(video_path)

    sample_fps = config["sampling"]["fps"]
    max_frames = config["sampling"]["max_frames"]
    frame_indices = sample_frame_indices(metadata, sample_fps, max_frames)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    min_confidence = config["face"]["min_confidence"]
    min_area_ratio = config["face"]["min_face_area_ratio"]
    expansion_ratio = config["face"]["expansion_ratio"]
    target_size = config["image"]["size"]
    quality_threshold = config["quality"]["minimum_score"]

    face_crops = []
    frame_infos = []

    frames_dir = Path(config["paths"]["frames"]) / video_path.stem
    faces_dir = Path(config["paths"]["faces"]) / video_path.stem
    frames_dir.mkdir(parents=True, exist_ok=True)
    faces_dir.mkdir(parents=True, exist_ok=True)

    try:
        for frame_idx in tqdm(frame_indices, desc="Preprocessing", unit="frame"):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if not ret or frame is None:
                frame_infos.append({
                    "frame_index": frame_idx,
                    "timestamp_seconds": frame_idx / metadata["fps"] if metadata["fps"] > 0 else 0.0,
                    "face_found": False,
                    "usable": False,
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

                if primary_face.det_score >= min_confidence:
                    x1, y1, x2, y2 = expand_bbox(
                        primary_face.bbox, expansion_ratio, metadata["width"], metadata["height"]
                    )

                    face_w = x2 - x1
                    face_h = y2 - y1
                    face_area = face_w * face_h
                    frame_area = metadata["width"] * metadata["height"]
                    face_area_ratio = face_area / frame_area if frame_area > 0 else 0.0

                    if face_area_ratio >= min_area_ratio:
                        face_crop = frame[y1:y2, x1:x2]
                        if face_crop.size > 0:
                            face_crop_resized = cv2.resize(face_crop, (target_size, target_size), interpolation=cv2.INTER_AREA)

                            blur_score = compute_blur_score(face_crop_resized)
                            face_quality = compute_face_quality(
                                primary_face.det_score, face_area_ratio, blur_score
                            )

                            usable = face_quality >= quality_threshold

                            face_filename = f"frame_{frame_idx:06d}.jpg"
                            face_path = faces_dir / face_filename
                            if usable:
                                cv2.imwrite(str(face_path), face_crop_resized)

                            face_crops.append(face_crop_resized)
                            frame_infos.append({
                                "frame_index": frame_idx,
                                "timestamp_seconds": frame_idx / metadata["fps"] if metadata["fps"] > 0 else 0.0,
                                "face_found": True,
                                "face_confidence": float(primary_face.det_score),
                                "face_area_ratio": float(face_area_ratio),
                                "blur_score": float(blur_score),
                                "face_quality": float(face_quality),
                                "usable": usable,
                                "bbox": [x1, y1, x2, y2],
                                "frame_path": str(frame_path),
                                "face_path": str(face_path) if usable else "",
                            })
                            continue

            frame_infos.append({
                "frame_index": frame_idx,
                "timestamp_seconds": frame_idx / metadata["fps"] if metadata["fps"] > 0 else 0.0,
                "face_found": False,
                "usable": False,
                "frame_path": str(frame_path),
                "face_path": "",
            })
    finally:
        cap.release()

    return face_crops, frame_infos, metadata


@torch.no_grad()
def run_inference(
    face_crops: list[np.ndarray],
    model: DeepfakeClassifier,
    config: dict[str, Any],
    device: torch.device,
) -> np.ndarray:
    """Run model inference on face crops."""
    if not face_crops:
        return np.array([])

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    batch_size = config["inference"]["batch_size"]
    all_probs = []

    for i in range(0, len(face_crops), batch_size):
        batch_crops = face_crops[i:i + batch_size]
        batch_tensors = torch.stack([transform(Image.fromarray(c)) for c in batch_crops]).to(device)

        logits = model(batch_tensors)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.extend(probs)

    return np.array(all_probs)


def analyze_video(
    video_path: str | Path,
    config: dict[str, Any],
    model: DeepfakeClassifier,
    face_app,
    device: torch.device,
) -> dict[str, Any]:
    """
    Full pipeline: preprocess video -> run inference -> forensic engine -> explainability -> robustness -> report.
    Returns complete Phase 4 result dictionary.
    """
    video_path = Path(video_path)
    video_id = video_path.stem

    # Phase 1 + 2
    face_crops, frame_infos, video_metadata = preprocess_video_for_inference(
        video_path, config, face_app
    )

    scores = run_inference(face_crops, model, config, device)

    score_idx = 0
    for fi in frame_infos:
        if fi.get("usable", False):
            fi["score"] = float(scores[score_idx])
            score_idx += 1

    # Phase 3: Forensic engine
    result = analyze_frame_predictions(frame_infos, video_id, config)

    # Save forensic result
    forensic_dir = Path(config["paths"].get("forensic", "./outputs/forensic"))
    save_forensic_result(result, forensic_dir)

    # Phase 4: Explainability
    explanations = []
    timeline_path = None
    if config.get("explainability", {}).get("enabled", True):
        explanations = generate_explanations_for_video(
            frame_infos, video_id, config, model, device
        )
        timeline_path = create_enhanced_timeline(
            frame_infos, video_id, config, result.to_dict()
        )

        # Save enhanced visualizations
        if config["forensic"]["visualization"]["enabled"]:
            create_timeline_plot(
                frame_infos,
                video_id,
                forensic_dir,
                dpi=config["forensic"]["visualization"]["timeline_dpi"],
            )
            create_contact_sheet(
                frame_infos,
                video_id,
                forensic_dir,
                cols=config["forensic"]["visualization"]["contact_sheet_cols"],
            )

    # Phase 4: Robustness
    robustness_results = {}
    if config.get("robustness", {}).get("enabled", True):
        robustness_results = run_robustness_tests(
            video_path, config, model, face_app, device
        )
        robustness_dir = Path(config["paths"].get("robustness", "./outputs/robustness"))
        save_robustness_report(robustness_results, robustness_dir)

    # Phase 4: Report generation
    reports_dir = Path(config["paths"].get("reports", "./outputs/reports"))
    report_formats = config.get("report", {}).get("formats", ["html", "json"])

    if "html" in report_formats:
        generate_html_report(
            video_path,
            result.to_dict(),
            config,
            explanations,
            robustness_results,
            timeline_path,
            reports_dir / f"{video_id}_report.html",
            frame_predictions=frame_infos,
        )

    if "json" in report_formats:
        save_json_report(
            video_path,
            result.to_dict(),
            explanations,
            robustness_results,
            reports_dir / f"{video_id}_report.json",
        )

    # Return complete result
    return {
        "video_id": video_id,
        "video_metadata": video_metadata,
        "frame_infos": frame_infos,
        "forensic_result": result.to_dict(),
        "explanations": explanations,
        "timeline_path": timeline_path,
        "robustness_results": robustness_results,
    }


def print_forensic_result(result: dict) -> None:
    """Print formatted forensic result from dict."""
    print(f"\n[INFO] Video: {result.get('video_id', 'unknown')}")
    print(f"[INFO] Sampled frames: {result.get('sampled_frames', 0)}")
    print(f"[INFO] Usable face frames: {result.get('usable_frames', 0)}")

    print(f"\n[INFO] Manipulation score: {result.get('manipulation_score', 0):.4f}")
    print(f"[INFO] Mean score: {result.get('mean_score', 0):.4f}")
    print(f"[INFO] Median score: {result.get('median_score', 0):.4f}")
    print(f"[INFO] Standard deviation: {result.get('std_score', 0):.4f}")
    print(f"[INFO] Consistency: {result.get('consistency', 0):.4f}")
    print(f"[INFO] Face coverage: {result.get('frame_coverage', 0):.4f}")
    print(f"[INFO] Reliability: {result.get('reliability', 0):.4f}")
    print(f"[INFO] Evidence confidence: {result.get('evidence_confidence', 0):.4f}")

    print(f"\n[RESULT] {result.get('verdict', 'UNKNOWN')}")

    if result.get('suspicious_segments'):
        print("\n[INFO] Suspicious segments:")
        for seg in result['suspicious_segments']:
            print(f"  {seg['start']:.1f}-{seg['end']:.1f}s ({seg['frame_count']} frames, peak={seg['peak_score']:.2f})")

    print("\n[INFO] Explanations:")
    for exp in result.get('explanations', []):
        print(f"  - {exp}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="SynthGuard Phase 4: Demo Video Inference (no checkpoint required)")
    parser.add_argument("--input", required=True, help="Path to input video file")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--output", help="Optional output JSON path")
    args = parser.parse_args()

    config = load_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"[INFO] Device: CUDA")
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("[WARN] CUDA unavailable. Running on CPU.")

    print("[INFO] Building model with ImageNet pretrained weights (no fine-tuning)...")
    model = build_model(
        pretrained=config["model"]["pretrained"],
        dropout=config["model"]["dropout"],
        device=device,
    )
    print("[INFO] Model ready (using pretrained ImageNet weights)")

    face_app = initialize_face_detector(config)

    result = analyze_video(args.input, config, model, face_app, device)

    print_forensic_result(result["forensic_result"])

    # Print robustness summary
    robustness = result.get("robustness_results", {})
    if robustness and robustness.get("tests"):
        print("\n[INFO] Robustness results:")
        for test in robustness["tests"]:
            if "stability" in test:
                print(f"  {test['transform']}: score={test['score']:.4f}, stability={test['stability']:.4f}")
        print(f"  Overall stability: {robustness.get('overall_stability', 0):.4f}")

    # Print report paths
    video_id = result["video_id"]
    reports_dir = Path(config["paths"].get("reports", "./outputs/reports"))
    print(f"\n[INFO] Reports generated:")
    print(f"  HTML: {reports_dir / f'{video_id}_report.html'}")
    print(f"  JSON: {reports_dir / f'{video_id}_report.json'}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[INFO] Result saved: {args.output}")


if __name__ == "__main__":
    main()