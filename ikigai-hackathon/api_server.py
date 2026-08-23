"""
SynthGuard API Server — Flask REST API wrapping the forensic pipeline.
"""

import os
import sys
import json
import tempfile
import hashlib
from datetime import datetime
from pathlib import Path

from flask import Flask, request, jsonify
from flask_cors import CORS

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

app = Flask(__name__)
CORS(app, origins=["http://localhost:5173", "http://localhost:5174", "http://localhost:5175"])

# ─── Globals ──────────────────────────────────────────────────────────────────

model = None
config = None
face_app = None
device = None
model_loaded = False
analysis_history: list[dict] = []

# ─── Model Loading ────────────────────────────────────────────────────────────

try:
    import torch
    from config import load_config
    from model import build_model, load_checkpoint
    from preprocessing import initialize_face_detector
    from inference import preprocess_video_for_inference, run_inference, build_analysis_result, analyze_video
    from forensic_engine import analyze_frame_predictions
    from explainability import generate_explanations_for_video
    from robustness import run_robustness_tests

    config = load_config("config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        model_name=config["model"]["name"],
        pretrained=config["model"]["pretrained"],
        dropout=config["model"]["dropout"],
        num_classes=config["model"].get("num_classes", 2),
        device=device,
    )
    checkpoint_path = config.get("inference", {}).get(
        "checkpoint", "models/xception_best.pth"
    )
    if Path(checkpoint_path).exists():
        load_checkpoint(checkpoint_path, model, device=device)
        model_loaded = True
    else:
        print(f"[WARN] Checkpoint not found: {checkpoint_path}. Using random weights.")

    face_app = initialize_face_detector(config)
    print("[INFO] Pipeline loaded successfully.")

except Exception as exc:
    print(f"[WARN] Could not load full pipeline: {exc}")
    print("[INFO] Only /api/demo and /api/health will work.")


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _device_name() -> str:
    try:
        import torch as _t
        if _t.cuda.is_available():
            return f"CUDA ({_t.cuda.get_device_name(0)})"
        return "CPU"
    except Exception:
        return "unknown"


# ─── Routes ───────────────────────────────────────────────────────────────────


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "ok",
            "model_loaded": model_loaded,
            "device": _device_name(),
            "timestamp": datetime.now().isoformat(),
        }
    )


@app.route("/api/analyze", methods=["POST"])
def analyze():
    if model is None or config is None:
        return jsonify({"error": "Model not loaded. Use /api/demo instead."}), 503

    if "video" not in request.files:
        return jsonify({"error": "No video file provided."}), 400

    video_file = request.files["video"]
    if video_file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    # Save to temp
    suffix = Path(video_file.filename).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        video_file.save(tmp)
        video_path = tmp.name

    try:
        # Run full analysis pipeline
        result = analyze_video(
            video_path, config, model, face_app, device
        )
        
        # Return canonical result
        canonical = result["canonical"]
        canonical["video_metadata"] = result["video_metadata"]
        canonical["frame_infos"] = result["frame_infos"]
        canonical["forensic_result"] = result["forensic_result"].to_dict()
        canonical["explanations"] = result["explanations"]
        canonical["robustness_results"] = result["robustness_results"]
        canonical["model_info"] = result["model_info"]
        
        analysis_history.append(canonical)
        return jsonify(canonical)

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    finally:
        try:
            os.unlink(video_path)
        except OSError:
            pass


@app.route("/api/history", methods=["GET"])
def history():
    return jsonify(analysis_history)


@app.route("/api/demo", methods=["GET"])
def demo():
    """Return realistic mock data for frontend development."""
    import random
    import math
    random.seed(42)

    # Build 60 frame_infos with realistic score progression
    frame_infos = []
    for i in range(60):
        ts = i * (10.0 / 60)
        # Low scores 0-3s, high 3-7s, low 7-10s
        if ts < 3.0:
            base = 0.28 + 0.05 * (ts / 3.0)
            noise = random.gauss(0, 0.04)
        elif ts < 7.0:
            progress = (ts - 3.0) / 4.0
            base = 0.55 + 0.35 * math.sin(progress * math.pi)
            noise = random.gauss(0, 0.05)
        else:
            base = 0.35 - 0.08 * ((ts - 7.0) / 3.0)
            noise = random.gauss(0, 0.04)

        score = max(0.05, min(0.98, base + noise))
        quality = 0.6 + random.random() * 0.25

        frame_infos.append({
            "frame_index": i * 5,
            "timestamp_seconds": round(ts, 3),
            "score": round(score, 4),
            "face_quality": round(quality, 3),
            "usable": True,
            "face_found": True,
        })

    # Pick top 5 suspicious frames
    sorted_frames = sorted(frame_infos, key=lambda f: f["score"], reverse=True)
    suspicious_frames = [
        {
            "frame_index": f["frame_index"],
            "timestamp_seconds": f["timestamp_seconds"],
            "score": f["score"],
        }
        for f in sorted_frames[:5]
    ]

    result = {
        "case": {
            "case_id": "SG-20260822-1208",
            "filename": "demo_sample_video.mp4",
            "sha256": "a" * 64,
            "size_bytes": 10485760,
            "analysis_timestamp": datetime.now().isoformat(),
        },
        "video": {
            "width": 1920,
            "height": 1080,
            "fps": 30.0,
            "frame_count": 300,
            "duration_seconds": 10.0,
            "codec": "h264",
        },
        "model": {
            "architecture": "Xception",
            "checkpoint": "xception_best.pth",
            "device": "CPU",
            "input_size": 224,
            "parameters": 20811050,
        },
        "preprocessing": {
            "sampled_frames": 60,
            "faces_detected": 60,
            "usable_face_frames": 51,
            "face_coverage": 0.85,
            "average_face_quality": 0.738,
        },
        "detection": {
            "mean_score": 0.583,
            "median_score": 0.551,
            "max_score": 0.921,
            "std_score": 0.218,
            "frame_predictions": frame_infos,
        },
        "evidence": {
            "consistency": 0.687,
            "reliability": 0.762,
            "confidence": 0.549,
        },
        "decision": {
            "verdict": "LIKELY_DEEPFAKE",
            "reason_codes": ["HIGH_MANIPULATION_SIGNAL", "HIGH_CONSISTENCY", "LOCALIZED_EVIDENCE"],
        },
        "suspicious": {
            "frames": suspicious_frames,
            "segments": [
                {
                    "start": 3.0,
                    "end": 5.5,
                    "duration": 2.5,
                    "frame_count": 15,
                    "peak_score": 0.921,
                    "mean_score": 0.812,
                },
                {
                    "start": 5.8,
                    "end": 7.0,
                    "duration": 1.2,
                    "frame_count": 7,
                    "peak_score": 0.867,
                    "mean_score": 0.764,
                },
            ],
        },
        "explainability": {
            "attributions": [],
        },
        "robustness": {
            "tests": [
                {"transform": "resize", "score": 0.681, "difference": 0.039, "stability": 0.946},
                {"transform": "blur", "score": 0.702, "difference": 0.018, "stability": 0.975},
                {"transform": "jpeg_compression", "score": 0.738, "difference": 0.018, "stability": 0.975},
                {"transform": "brightness", "score": 0.694, "difference": 0.026, "stability": 0.964},
            ],
            "overall_stability": 0.965,
        },
        "forensic_result": {
            "video_id": "demo_sample_video",
            "verdict": "LIKELY_DEEPFAKE",
            "manipulation_score": 0.812,
            "mean_score": 0.583,
            "median_score": 0.551,
            "max_score": 0.921,
            "std_score": 0.218,
            "raw_median_score": 0.551,
            "weighted_mean_score": 0.612,
            "weighted_median_score": 0.812,
            "weighted_std_score": 0.145,
            "min_frame_weight": 0.60,
            "max_frame_weight": 0.85,
            "mean_frame_weight": 0.738,
            "average_boundary_score": 0.684,
            "average_frequency_anomaly": 0.621,
            "blink_naturalness_score": 0.412,
            "identity_drift_score": 0.528,
            "robustness_stability_score": 0.965,
            "consistency": 0.687,
            "frame_coverage": 0.85,
            "average_face_quality": 0.738,
            "reliability": 0.762,
            "evidence_confidence": 0.619,
            "sampled_frames": 60,
            "usable_frames": 51,
            "suspicious_frames": suspicious_frames,
            "suspicious_segments": [
                {
                    "start": 3.0,
                    "end": 5.5,
                    "duration": 2.5,
                    "frame_count": 15,
                    "peak_score": 0.921,
                    "mean_score": 0.812,
                },
                {
                    "start": 5.8,
                    "end": 7.0,
                    "duration": 1.2,
                    "frame_count": 7,
                    "peak_score": 0.867,
                    "mean_score": 0.764,
                },
            ],
            "explanations": [
                "Multiple sampled face frames produced elevated manipulation scores.",
                "Boundary artifact evidence was measured at 0.684.",
                "Frequency anomaly evidence was measured at 0.621.",
                "The suspicious evidence is concentrated within a localized time segment (3.0-5.5 seconds).",
            ],
            "reason_codes": ["HIGH_MANIPULATION_SIGNAL", "HIGH_CONSISTENCY", "LOCALIZED_EVIDENCE", "BOUNDARY_ARTIFACT_EVIDENCE", "FREQUENCY_ANOMALY_EVIDENCE"],
        },
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

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, port=5000)