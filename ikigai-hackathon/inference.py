"""
SynthGuard Phase 4: Inference Script

Runs full pipeline on a single video:
- Phase 1 preprocessing (sample frames, detect faces, crop)
- Phase 2 model inference
- Phase 3 forensic decision engine
- Phase 4 explainability + robustness + reporting
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Union

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from model import DeepfakeClassifier, XceptionDeepfakeClassifier, build_model, load_checkpoint, get_model_info
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
    generate_explanations,
)
from explainability import (
    generate_explanations_for_video,
    create_enhanced_timeline,
)
from robustness import run_robustness_tests, save_robustness_report
from report import generate_html_report, save_json_report
from boundary_artifact import analyze_boundary_artifact
from frequency_analysis import analyze_frequency
from blink_analysis import BlinkAnalyzer
from identity_drift import identity_similarity_and_stability


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
    blink_analyzer = BlinkAnalyzer(config["forensic"]["thresholds"].get("blink", {}))
    previous_embedding = None

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

                            if usable:
                                face_crops.append(face_crop_resized)
                            landmarks = primary_face.kps.tolist() if getattr(primary_face, "kps", None) is not None else None
                            embedding = None
                            if getattr(primary_face, "normed_embedding", None) is not None:
                                embedding = primary_face.normed_embedding.tolist()
                            elif getattr(primary_face, "embedding", None) is not None:
                                emb = np.asarray(primary_face.embedding, dtype=float)
                                norm = float(np.linalg.norm(emb))
                                if norm > 1e-12:
                                    embedding = (emb / norm).tolist()
                            boundary_score = analyze_boundary_artifact(frame, [x1, y1, x2, y2], config["forensic"]["thresholds"].get("boundary", {}))
                            frequency_anomaly, frequency_ratio = analyze_frequency(face_crop_resized, config["forensic"]["thresholds"].get("frequency", {}))
                            blink = blink_analyzer.update(landmarks)
                            identity_similarity, identity_drift = identity_similarity_and_stability(previous_embedding, embedding)
                            if embedding is not None:
                                previous_embedding = embedding
                            frame_infos.append({
                                "frame_index": frame_idx,
                                "timestamp_seconds": frame_idx / metadata["fps"] if metadata["fps"] > 0 else 0.0,
                                "face_found": True,
                                "face_confidence": float(primary_face.det_score),
                                "face_area_ratio": float(face_area_ratio),
                                "blur_score": float(blur_score),
                                "face_quality": float(face_quality),
                                "weight": float(face_quality),
                                "usable": usable,
                                "bbox": [x1, y1, x2, y2],
                                "landmarks": landmarks,
                                "embedding": embedding,
                                "boundary_score": boundary_score,
                                "frequency_anomaly": frequency_anomaly,
                                "frequency_energy_ratio": frequency_ratio,
                                "blink_naturalness": blink["blink_naturalness"],
                                "eye_aspect_ratio": blink["ear"],
                                "identity_similarity": identity_similarity,
                                "identity_drift": identity_drift,
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
    model: Union[DeepfakeClassifier, XceptionDeepfakeClassifier],
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

    is_xception = isinstance(model, XceptionDeepfakeClassifier)

    for i in range(0, len(face_crops), batch_size):
        batch_crops = face_crops[i:i + batch_size]
        batch_tensors = torch.stack([transform(Image.fromarray(c)) for c in batch_crops]).to(device)

        logits = model(batch_tensors)

        if is_xception:
            # Xception outputs 2 logits [real, fake] -> use softmax
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        else:
            # ConvNeXt outputs single logit -> use sigmoid
            probs = torch.sigmoid(logits).cpu().numpy()

        all_probs.extend(probs)

    return np.array(all_probs)


def build_analysis_result(
    video_path: str | Path,
    config: dict[str, Any],
    model: Union[DeepfakeClassifier, XceptionDeepfakeClassifier],
    frame_infos: list[dict],
    video_metadata: dict,
    device: torch.device,
    face_app,
) -> dict[str, Any]:
    """Build canonical AnalysisResult from all pipeline components."""
    video_path = Path(video_path)
    video_id = video_path.stem

    # Get model info
    model_info = get_model_info(model)
    model_info["architecture"] = model.__class__.__name__.replace("DeepfakeClassifier", "").replace("Xception", "Xception").strip()
    if not model_info["architecture"]:
        model_info["architecture"] = "Xception" if isinstance(model, XceptionDeepfakeClassifier) else "ConvNeXt-Tiny"
    model_info["checkpoint"] = config.get("inference", {}).get("checkpoint", "models/xception_best.pth")
    model_info["device"] = str(device)
    model_info["input_size"] = config["model"]["image_size"]

    # Run forensic engine
    forensic_result = analyze_frame_predictions(frame_infos, video_id, config)

    # Save forensic result
    forensic_dir = Path(config["paths"].get("forensic", "./outputs/forensic"))
    save_forensic_result(forensic_result, forensic_dir)

    # Phase 4: Explainability
    explanations = []
    timeline_path = None
    if config.get("explainability", {}).get("enabled", True):
        explanations = generate_explanations_for_video(
            frame_infos, video_id, config, model, device
        )
        timeline_path = create_enhanced_timeline(
            frame_infos, video_id, config, forensic_result.to_dict()
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
        forensic_result.robustness_stability_score = robustness_results.get("overall_stability")
        # Keep the plain-language evidence in sync with the newly available
        # robustness measurement.
        forensic_result.explanations, forensic_result.reason_codes = generate_explanations(
            forensic_result, config
        )

    # Build canonical result
    canonical_result = {
        "case": {
            "case_id": generate_case_id(config.get("report", {}).get("case_id_prefix", "SG")),
            "filename": video_path.name,
            "sha256": "",  # Will be filled by report
            "size_bytes": 0,  # Will be filled by report
            "analysis_timestamp": "",  # Will be filled by report
        },
        "video": {
            "fps": video_metadata.get("fps", 0),
            "frame_count": video_metadata.get("frame_count", 0),
            "duration_seconds": video_metadata.get("duration_seconds", 0),
            "width": video_metadata.get("width", 0),
            "height": video_metadata.get("height", 0),
            "codec": video_metadata.get("codec", "unknown"),
        },
        "model": {
            "architecture": model_info["architecture"],
            "checkpoint": Path(model_info["checkpoint"]).name,
            "device": model_info["device"],
            "input_size": model_info["input_size"],
            "parameters": model_info["parameters"],
        },
        "preprocessing": {
            "sampled_frames": forensic_result.sampled_frames,
            "faces_detected": forensic_result.sampled_frames,
            "usable_face_frames": forensic_result.usable_frames,
            "face_coverage": forensic_result.frame_coverage,
            "average_face_quality": forensic_result.average_face_quality,
        },
        "detection": {
            "manipulation_score": forensic_result.manipulation_score,
            "mean_score": forensic_result.mean_score,
            "median_score": forensic_result.median_score,
            "max_score": forensic_result.max_score,
            "std_score": forensic_result.std_score,
            "frame_predictions": [
                {
                    "frame_index": f.get("frame_index"),
                    "timestamp_seconds": f.get("timestamp_seconds"),
                    "score": f.get("score"),
                    "face_quality": f.get("face_quality"),
                    "usable": f.get("usable", False),
                    "face_path": f.get("face_path", ""),
                }
                for f in frame_infos if f.get("usable", False)
            ],
        },
        "evidence": {
            "consistency": forensic_result.consistency,
            "reliability": forensic_result.reliability,
            "confidence": forensic_result.evidence_confidence,
        },
        "decision": {
            "verdict": forensic_result.verdict,
            "reason_codes": forensic_result.reason_codes,
        },
        "suspicious": {
            "frames": forensic_result.suspicious_frames,
            "segments": forensic_result.suspicious_segments,
        },
        "explainability": {
            "attributions": explanations,
        },
        "robustness": {
            "tests": robustness_results.get("tests", []),
            "overall_stability": robustness_results.get("overall_stability", 0),
        },
        "forensic_result": forensic_result.to_dict(),
        "limitations": [
            "Analysis uses a single frame-level neural network detector.",
            "No explicit frequency-domain analysis was performed.",
            "No boundary artifact detection was performed.",
            "No temporal consistency modeling (TCN) was applied.",
            "No identity consistency verification was performed.",
            "Evidence confidence is a transparent heuristic, not a calibrated probability.",
            "Cross-dataset generalization has not been established.",
            "Robustness testing covers only a limited set of common transformations.",
        ],
    }

    # Also return components needed for report generation
    return {
        "canonical": canonical_result,
        "video_id": video_id,
        "video_metadata": video_metadata,
        "frame_infos": frame_infos,
        "forensic_result": forensic_result,
        "explanations": explanations,
        "timeline_path": timeline_path,
        "robustness_results": robustness_results,
        "model_info": model_info,
    }


def analyze_video(
    video_path: str | Path,
    config: dict[str, Any],
    model: Union[DeepfakeClassifier, XceptionDeepfakeClassifier],
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

    # Build canonical result
    result = build_analysis_result(
        video_path, config, model, frame_infos, video_metadata, device, face_app
    )

    # Phase 4: Report generation
    reports_dir = Path(config["paths"].get("reports", "./outputs/reports"))
    report_formats = config.get("report", {}).get("formats", ["html", "json"])

    if "html" in report_formats:
        generate_html_report(
            video_path,
            result["forensic_result"].to_dict(),
            config,
            result["explanations"],
            result["robustness_results"],
            result["timeline_path"],
            reports_dir / f"{video_id}_report.html",
            model_info=result["model_info"],
            frame_predictions=frame_infos,
        )

    if "json" in report_formats:
        save_json_report(
            video_path,
            result["forensic_result"].to_dict(),
            result["explanations"],
            result["robustness_results"],
            reports_dir / f"{video_id}_report.json",
            model_info=result["model_info"],
            video_metadata=video_metadata,
        )

    return result


def generate_case_id(prefix: str = "SG") -> str:
    """Generate a case ID."""
    from datetime import datetime
    date_str = datetime.now().strftime("%Y%m%d")
    time_str = datetime.now().strftime("%H%M%S")
    return f"{prefix}-{date_str}-{time_str[-4:]}"


def print_forensic_result(result: dict) -> None:
    """Print formatted forensic result."""
    forensic = result["forensic_result"]
    print(f"\n[INFO] Video: {forensic.video_id}")
    print(f"[INFO] Sampled frames: {forensic.sampled_frames}")
    print(f"[INFO] Usable face frames: {forensic.usable_frames}")

    print(f"\n[INFO] Manipulation score: {forensic.manipulation_score:.4f}")
    print(f"[INFO] Mean score: {forensic.mean_score:.4f}")
    print(f"[INFO] Median score: {forensic.median_score:.4f}")
    print(f"[INFO] Standard deviation: {forensic.std_score:.4f}")
    print(f"[INFO] Consistency: {forensic.consistency:.4f}")
    print(f"[INFO] Face coverage: {forensic.frame_coverage:.4f}")
    print(f"[INFO] Reliability: {forensic.reliability:.4f}")
    print(f"[INFO] Evidence confidence: {forensic.evidence_confidence:.4f}")

    print(f"\n[RESULT] {forensic.verdict}")

    if forensic.suspicious_segments:
        print("\n[INFO] Suspicious segments:")
        for seg in forensic.suspicious_segments:
            print(f"  {seg.start:.1f}-{seg.end:.1f}s ({seg.frame_count} frames, peak={seg.peak_score:.2f})")

    print("\n[INFO] Explanations:")
    for exp in forensic.explanations:
        print(f"  - {exp}")

    print("\n[INFO] Reason Codes:")
    for code in forensic.reason_codes:
        print(f"  - {code}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="SynthGuard Phase 4: Video Inference")
    parser.add_argument("--input", required=True, help="Path to input video file")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--checkpoint", default="models/xception_best.pth", help="Path to model checkpoint")
    parser.add_argument("--output", help="Optional output JSON path")
    args = parser.parse_args()

    config = load_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"[INFO] Device: CUDA")
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("[WARN] CUDA unavailable. Running on CPU.")

    model = build_model(
        model_name=config["model"]["name"],
        pretrained=config["model"]["pretrained"],
        dropout=config["model"]["dropout"],
        num_classes=config["model"].get("num_classes", 2),
        device=device,
    )

    load_checkpoint(args.checkpoint, model, device=device)
    print(f"[INFO] Loaded checkpoint: {args.checkpoint}")

    face_app = initialize_face_detector(config)

    result = analyze_video(args.input, config, model, face_app, device)

    print_forensic_result(result)

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
            json.dump(result["canonical"], f, indent=2)
        print(f"[INFO] Result saved: {args.output}")


if __name__ == "__main__":
    main()
