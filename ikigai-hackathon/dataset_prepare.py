"""
SynthGuard Phase 1: Dataset Preparation Module

Creates train/val/test splits, sanity checks, and optional preview.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import cv2
import pandas as pd
import yaml
from tqdm import tqdm


def load_config(config_path: str | Path) -> dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def print_env_info(config: dict[str, Any]) -> None:
    print("=" * 60)
    print("[INFO] Environment Check")
    print("=" * 60)
    print(f"[INFO] Python: {sys.version.split()[0]}")
    
    try:
        import torch
        print(f"[INFO] PyTorch: {torch.__version__}")
        print(f"[INFO] CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("[WARN] CUDA not available - will use CPU")
    except ImportError:
        print("[INFO] PyTorch: not installed")
    
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        print(f"[INFO] ONNX providers: {providers}")
    except ImportError:
        pass
    
    print("=" * 60)


def load_manifest(manifest_path: Path) -> pd.DataFrame:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    return pd.read_csv(manifest_path)


def validate_videos(df: pd.DataFrame, raw_dir: Path) -> tuple[pd.DataFrame, list[dict]]:
    """Validate that video files exist and can be opened."""
    valid_rows = []
    broken = []
    
    print("[INFO] Validating video files...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Validating", unit="video"):
        video_path = raw_dir / row["video_path"]
        
        if not video_path.exists():
            broken.append({
                "video_id": row["video_id"],
                "video_path": row["video_path"],
                "error": "File not found"
            })
            continue
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            broken.append({
                "video_id": row["video_id"],
                "video_path": row["video_path"],
                "error": "Cannot open video"
            })
            continue
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        if frame_count <= 0 or width <= 0 or height <= 0:
            broken.append({
                "video_id": row["video_id"],
                "video_path": row["video_path"],
                "error": f"Invalid video properties: frames={frame_count}, {width}x{height}"
            })
            continue
        
        valid_rows.append(row)
    
    valid_df = pd.DataFrame(valid_rows) if valid_rows else pd.DataFrame(columns=df.columns)
    
    if broken:
        print(f"[WARN] {len(broken)} videos failed validation")
    
    return valid_df, broken


def create_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
) -> dict[str, pd.DataFrame]:
    """Create train/val/test splits at video level."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    
    random.seed(random_seed)
    
    shuffled = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    train_df = shuffled.iloc[:n_train].copy()
    val_df = shuffled.iloc[n_train:n_train + n_val].copy()
    test_df = shuffled.iloc[n_train + n_val:].copy()
    
    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"
    
    return {
        "train": train_df,
        "val": val_df,
        "test": test_df,
        "all": pd.concat([train_df, val_df, test_df], ignore_index=True)
    }


def save_splits(splits: dict[str, pd.DataFrame], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for name, df in splits.items():
        path = output_dir / f"{name}.csv"
        df.to_csv(path, index=False)
        print(f"[INFO] Saved {name}: {len(df)} videos -> {path}")


def print_dataset_summary(splits: dict[str, pd.DataFrame], broken_count: int) -> None:
    all_df = splits["all"]
    
    print("\n" + "=" * 60)
    print("[INFO] Dataset Summary")
    print("=" * 60)
    
    real_count = len(all_df[all_df["label"] == 0])
    fake_count = len(all_df[all_df["label"] == 1])
    
    print(f"Real: {real_count}")
    
    for manip_type in ["DeepFakes", "Face2Face", "FaceSwap", "NeuralTextures"]:
        count = len(all_df[(all_df["label"] == 1) & (all_df["manipulation_type"] == manip_type)])
        if count > 0:
            print(f"  {manip_type}: {count}")
    
    print(f"Total Fake: {fake_count}")
    print(f"Total: {len(all_df)}")
    print(f"Train: {len(splits['train'])}")
    print(f"Validation: {len(splits['val'])}")
    print(f"Test: {len(splits['test'])}")
    print(f"Broken: {broken_count}")
    print("=" * 60)


def save_dataset_summary(
    splits: dict[str, pd.DataFrame],
    broken_count: int,
    output_path: Path
) -> None:
    all_df = splits["all"]
    
    summary = {
        "total_videos": int(len(all_df)),
        "total_real": int(len(all_df[all_df["label"] == 0])),
        "total_fake": int(len(all_df[all_df["label"] == 1])),
        "total_deepfakes": int(len(all_df[(all_df["label"] == 1) & (all_df["manipulation_type"] == "DeepFakes")])),
        "total_face2face": int(len(all_df[(all_df["label"] == 1) & (all_df["manipulation_type"] == "Face2Face")])),
        "total_faceswap": int(len(all_df[(all_df["label"] == 1) & (all_df["manipulation_type"] == "FaceSwap")])),
        "total_neuraltextures": int(len(all_df[(all_df["label"] == 1) & (all_df["manipulation_type"] == "NeuralTextures")])),
        "train_count": int(len(splits["train"])),
        "val_count": int(len(splits["val"])),
        "test_count": int(len(splits["test"])),
        "broken_count": broken_count,
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[INFO] Dataset summary saved: {output_path}")


def create_preview_contact_sheet(
    splits: dict[str, pd.DataFrame],
    raw_dir: Path,
    output_path: Path,
    num_samples: int = 8
) -> None:
    """Create a simple text-based preview (contact sheet would need more deps)."""
    all_df = splits["all"]
    
    real_samples = all_df[all_df["label"] == 0].sample(min(num_samples // 2, len(all_df[all_df["label"] == 0])), random_state=42)
    fake_samples = all_df[all_df["label"] == 1].sample(min(num_samples // 2, len(all_df[all_df["label"] == 1])), random_state=42)
    
    print("\n[INFO] Dataset Preview:")
    print("-" * 80)
    print(f"{'Video ID':<30} {'Label':<6} {'Type':<15} {'Split':<8}")
    print("-" * 80)
    
    for _, row in pd.concat([real_samples, fake_samples]).iterrows():
        label_str = "REAL" if row["label"] == 0 else "FAKE"
        print(f"{row['video_id']:<30} {label_str:<6} {row['manipulation_type']:<15} {row['split']:<8}")
    
    print("-" * 80)


def main():
    parser = argparse.ArgumentParser(description="SynthGuard Phase 1: Dataset Preparation")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--preview", action="store_true", help="Show dataset preview")
    args = parser.parse_args()
    
    config = load_config(args.config)
    print_env_info(config)
    
    raw_dir = Path(config["paths"]["raw"])
    subset_dir = Path(config["paths"]["subset"])
    splits_dir = Path(config["paths"]["splits"])
    metadata_dir = Path(config["paths"]["metadata"])
    random_seed = config["dataset"]["random_seed"]
    
    manifest_path = subset_dir / "all.csv"
    
    if not manifest_path.exists():
        print(f"[ERROR] Manifest not found: {manifest_path}")
        print("[INFO] Run dataset_download.py first")
        sys.exit(1)
    
    df = load_manifest(manifest_path)
    print(f"[INFO] Loaded manifest: {len(df)} videos")
    
    valid_df, broken = validate_videos(df, raw_dir)
    print(f"[INFO] Valid videos: {len(valid_df)}, Broken: {len(broken)}")
    
    if len(valid_df) == 0:
        print("[ERROR] No valid videos found")
        sys.exit(1)
    
    broken_df = pd.DataFrame(broken)
    if not broken_df.empty:
        broken_path = subset_dir / "broken.csv"
        broken_df.to_csv(broken_path, index=False)
    
    splits = create_splits(valid_df, random_seed=random_seed)
    save_splits(splits, splits_dir)
    print_dataset_summary(splits, len(broken))
    save_dataset_summary(splits, len(broken), metadata_dir / "dataset_summary.json")
    
    if args.preview:
        create_preview_contact_sheet(splits, raw_dir, metadata_dir / "preview.txt")
    
    print("\n[INFO] Dataset preparation complete")


if __name__ == "__main__":
    main()