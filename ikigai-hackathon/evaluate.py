"""
SynthGuard Phase 2: Evaluation Script

Evaluates trained model on test set:
- Frame-level predictions
- Video-level aggregation
- Manipulation-type breakdown
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from tqdm import tqdm

from model import DeepfakeClassifier, build_model, load_checkpoint
from data_loader import DeepfakeFrameDataset, get_transforms, validate_split_integrity


def load_config(config_path: str | Path) -> dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def evaluate_frame_level(
    model: DeepfakeClassifier,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list]:
    """Get frame-level predictions."""
    model.eval()
    all_probs = []
    all_labels = []
    all_metadata = []

    for batch in tqdm(test_loader, desc="Frame evaluation"):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        logits = model(images)
        probs = torch.sigmoid(logits).cpu().numpy()

        all_probs.extend(probs)
        all_labels.extend(labels.cpu().numpy())

        for i in range(len(labels)):
            all_metadata.append({
                "video_id": batch["video_id"][i],
                "frame_index": batch["frame_index"][i].item() if torch.is_tensor(batch["frame_index"][i]) else batch["frame_index"][i],
                "timestamp_seconds": batch["timestamp_seconds"][i].item() if torch.is_tensor(batch["timestamp_seconds"][i]) else batch["timestamp_seconds"][i],
                "manipulation_type": batch["manipulation_type"][i],
            })

    return np.array(all_probs), np.array(all_labels), all_metadata


def aggregate_video_level(
    frame_probs: np.ndarray,
    frame_labels: np.ndarray,
    metadata: list[dict],
) -> pd.DataFrame:
    """Aggregate frame scores to video level using median."""
    df = pd.DataFrame(metadata)
    df["score"] = frame_probs
    df["label"] = frame_labels

    video_stats = df.groupby("video_id").agg(
        label=("label", "first"),
        manipulation_type=("manipulation_type", "first"),
        frame_count=("score", "count"),
        mean_score=("score", "mean"),
        median_score=("score", "median"),
        max_score=("score", "max"),
    ).reset_index()

    video_stats["video_score"] = video_stats["median_score"]
    return video_stats


def compute_video_metrics(video_df: pd.DataFrame) -> dict[str, Any]:
    """Compute video-level metrics."""
    labels = video_df["label"].values
    scores = video_df["video_score"].values
    preds = (scores > 0.5).astype(int)

    cm = confusion_matrix(labels, preds)

    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "roc_auc": roc_auc_score(labels, scores) if len(set(labels)) > 1 else 0.0,
        "confusion_matrix": cm.tolist(),
        "n_videos": len(video_df),
        "n_real": int((labels == 0).sum()),
        "n_fake": int((labels == 1).sum()),
    }


def compute_manipulation_breakdown(video_df: pd.DataFrame) -> dict[str, dict]:
    """Compute metrics per manipulation type."""
    breakdown = {}

    for manip_type in video_df["manipulation_type"].unique():
        subset = video_df[video_df["manipulation_type"] == manip_type]
        labels = subset["label"].values
        scores = subset["video_score"].values
        preds = (scores > 0.5).astype(int)

        n = len(subset)
        if n == 0:
            continue

        breakdown[manip_type] = {
            "count": int(n),
            "mean_score": float(scores.mean()),
            "median_score": float(np.median(scores)),
            "accuracy": float(accuracy_score(labels, preds)) if len(set(labels)) > 1 else float((labels == preds).mean()),
        }

    return breakdown


def save_frame_predictions(
    frame_probs: np.ndarray,
    frame_labels: np.ndarray,
    metadata: list[dict],
    output_path: Path,
) -> None:
    """Save frame-level predictions to CSV."""
    df = pd.DataFrame(metadata)
    df["score"] = frame_probs
    df["label"] = frame_labels
    df.to_csv(output_path, index=False)
    print(f"[INFO] Frame predictions saved: {output_path}")


def save_video_predictions(video_df: pd.DataFrame, output_path: Path) -> None:
    """Save video-level predictions to CSV."""
    video_df.to_csv(output_path, index=False)
    print(f"[INFO] Video predictions saved: {output_path}")


def save_metrics(metrics: dict, output_path: Path) -> None:
    """Save metrics to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[INFO] Metrics saved: {output_path}")


def print_results(metrics: dict, breakdown: dict) -> None:
    """Print evaluation results."""
    print("\n" + "=" * 60)
    print("[INFO] VIDEO-LEVEL TEST RESULTS")
    print("=" * 60)
    print(f"Total videos: {metrics['n_videos']}")
    print(f"Real: {metrics['n_real']}, Fake: {metrics['n_fake']}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1:        {metrics['f1']:.4f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"Confusion Matrix:")
    print(f"  TN={metrics['confusion_matrix'][0][0]}  FP={metrics['confusion_matrix'][0][1]}")
    print(f"  FN={metrics['confusion_matrix'][1][0]}  TP={metrics['confusion_matrix'][1][1]}")
    print("-" * 60)
    print("[INFO] MANIPULATION TYPE BREAKDOWN")
    print("-" * 60)
    for manip_type, stats in breakdown.items():
        print(f"{manip_type}:")
        print(f"  Count: {stats['count']}")
        print(f"  Mean score: {stats['mean_score']:.4f}")
        print(f"  Median score: {stats['median_score']:.4f}")
        print(f"  Accuracy: {stats['accuracy']:.4f}")
    print("=" * 60)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="SynthGuard Phase 2: Evaluation")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path (defaults to config inference.checkpoint)")
    args = parser.parse_args()

    config = load_config(args.config)
    checkpoint = args.checkpoint or config.get("inference", {}).get("checkpoint", "models/xception_best.pth")
    splits_dir = Path(config["paths"]["splits"])
    test_split = splits_dir / "test.csv"
    if not test_split.exists():
        raise FileNotFoundError(f"Held-out test split is missing: {test_split}")
    if not Path(checkpoint).exists():
        raise FileNotFoundError(f"Evaluation checkpoint is missing: {checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"[INFO] Device: CUDA")
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("[WARN] CUDA unavailable. Running on CPU.")

    model = build_model(
        pretrained=config["model"]["pretrained"],
        dropout=config["model"]["dropout"],
        device=device,
    )

    load_checkpoint(checkpoint, model, device=device)
    print(f"[INFO] Loaded checkpoint: {checkpoint}")

    faces_root = Path(config["paths"]["faces"])
    metadata_root = Path(config["paths"]["metadata"])

    test_df = pd.read_csv(test_split)

    test_dataset = DeepfakeFrameDataset(
        test_df, faces_root, metadata_root,
        get_transforms(config["model"]["image_size"], is_train=False),
        config["model"]["image_size"]
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config["inference"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )

    frame_probs, frame_labels, metadata = evaluate_frame_level(model, test_loader, device)

    frame_pred_path = Path("outputs/predictions/test_frame_predictions.csv")
    save_frame_predictions(frame_probs, frame_labels, metadata, frame_pred_path)

    video_df = aggregate_video_level(frame_probs, frame_labels, metadata)

    video_pred_path = Path("outputs/predictions/test_video_predictions.csv")
    save_video_predictions(video_df, video_pred_path)

    video_metrics = compute_video_metrics(video_df)
    breakdown = compute_manipulation_breakdown(video_df)

    print_results(video_metrics, breakdown)

    save_metrics({
        "video_metrics": video_metrics,
        "manipulation_breakdown": breakdown,
    }, Path("outputs/evaluation/video_metrics.json"))

    print("[INFO] Evaluation complete")


if __name__ == "__main__":
    main()
