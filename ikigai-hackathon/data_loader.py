"""
SynthGuard Phase 2: Dataset and DataLoader

Video-level dataset with frame-level samples from Phase 1 face crops.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image


class DeepfakeFrameDataset(Dataset):
    """
    Frame-level dataset for deepfake detection.

    Each sample is a face crop from Phase 1 preprocessing.
    Labels are at video level (0=real, 1=fake).
    """

    def __init__(
        self,
        split_df: pd.DataFrame,
        faces_root: Path,
        metadata_root: Path,
        transform: transforms.Compose | None = None,
        image_size: int = 224,
    ):
        """
        Args:
            split_df: DataFrame with columns [video_id, label, manipulation_type, split]
            faces_root: Root directory of face crops (outputs/faces/)
            metadata_root: Root directory of metadata JSON (outputs/metadata/)
            transform: Optional transforms
            image_size: Target image size
        """
        self.transform = transform
        self.image_size = image_size
        self.samples = []

        print(f"[INFO] Building dataset from {len(split_df)} videos...")

        for _, row in split_df.iterrows():
            video_id = row["video_id"]
            label = int(row["label"])
            manipulation_type = row["manipulation_type"]

            meta_path = metadata_root / f"{video_id}.json"
            if not meta_path.exists():
                continue

            with open(meta_path, "r") as f:
                meta = json.load(f)

            for frame_info in meta.get("frames", []):
                if not frame_info.get("usable", False):
                    continue

                face_path = Path(frame_info["face_path"])
                if not face_path.exists():
                    face_path = faces_root / video_id / f"frame_{frame_info['frame_index']:06d}.jpg"

                if not face_path.exists():
                    continue

                self.samples.append({
                    "face_path": str(face_path),
                    "label": label,
                    "video_id": video_id,
                    "frame_index": frame_info["frame_index"],
                    "timestamp_seconds": frame_info["timestamp_seconds"],
                    "manipulation_type": manipulation_type,
                })

        print(f"[INFO] Dataset built: {len(self.samples)} usable face frames")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]

        image = Image.open(sample["face_path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        return {
            "image": image,
            "label": torch.tensor(sample["label"], dtype=torch.float32),
            "video_id": sample["video_id"],
            "frame_index": sample["frame_index"],
            "timestamp_seconds": sample["timestamp_seconds"],
            "manipulation_type": sample["manipulation_type"],
        }


def get_transforms(
    image_size: int = 224,
    is_train: bool = True,
) -> transforms.Compose:
    """Get data transforms for train/val."""
    if is_train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


def create_dataloaders(
    config: dict[str, Any],
) -> tuple[DataLoader, DataLoader, DataLoader, dict]:
    """
    Create train/val/test dataloaders from Phase 1 splits.

    Returns:
        train_loader, val_loader, test_loader, class_info
    """
    splits_dir = Path(config["paths"]["splits"])
    faces_root = Path(config["paths"]["faces"])
    metadata_root = Path(config["paths"]["metadata"])

    train_df = pd.read_csv(splits_dir / "train.csv")
    val_df = pd.read_csv(splits_dir / "val.csv")
    test_df = pd.read_csv(splits_dir / "test.csv")

    validate_split_integrity(train_df, val_df, test_df)

    train_transform = get_transforms(config["model"]["image_size"], is_train=True)
    eval_transform = get_transforms(config["model"]["image_size"], is_train=False)

    train_dataset = DeepfakeFrameDataset(
        train_df, faces_root, metadata_root, train_transform, config["model"]["image_size"]
    )
    val_dataset = DeepfakeFrameDataset(
        val_df, faces_root, metadata_root, eval_transform, config["model"]["image_size"]
    )
    test_dataset = DeepfakeFrameDataset(
        test_df, faces_root, metadata_root, eval_transform, config["model"]["image_size"]
    )

    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty. Check Phase 1 face extraction output.")

    class_info = compute_class_weights(train_dataset)

    train_loader = create_loader(train_dataset, config, class_info, shuffle=True)
    val_loader = create_loader(val_dataset, config, shuffle=False)
    test_loader = create_loader(test_dataset, config, shuffle=False)

    return train_loader, val_loader, test_loader, class_info


def create_loader(
    dataset: DeepfakeFrameDataset,
    config: dict[str, Any],
    class_info: dict | None = None,
    shuffle: bool = False,
) -> DataLoader:
    """Create DataLoader with optional weighted sampling for class imbalance."""
    batch_size = config["training"]["batch_size"]
    num_workers = config["training"]["num_workers"]

    if shuffle and class_info and "weights" in class_info:
        sampler = WeightedRandomSampler(
            weights=class_info["weights"],
            num_samples=len(dataset),
            replacement=True,
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def compute_class_weights(dataset: DeepfakeFrameDataset) -> dict:
    """Compute class weights for imbalanced data."""
    labels = [s["label"] for s in dataset.samples]
    n_real = labels.count(0)
    n_fake = labels.count(1)
    total = len(labels)

    print(f"[INFO] Real samples: {n_real}")
    print(f"[INFO] Fake samples: {n_fake}")

    if total == 0:
        print("[WARNING] No samples in dataset, using default weights")
        return {"pos_weight": 1.0}

    print(f"[INFO] Class balance: {n_real/total:.3f} / {n_fake/total:.3f}")

    if n_real == 0 or n_fake == 0:
        return {"pos_weight": 1.0}

    pos_weight = n_real / n_fake
    weights = [pos_weight if l == 1 else 1.0 for l in labels]

    return {
        "n_real": n_real,
        "n_fake": n_fake,
        "pos_weight": pos_weight,
        "weights": weights,
    }


def validate_split_integrity(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """Ensure no video_id appears in multiple splits."""
    train_ids = set(train_df["video_id"])
    val_ids = set(val_df["video_id"])
    test_ids = set(test_df["video_id"])

    train_val = train_ids & val_ids
    train_test = train_ids & test_ids
    val_test = val_ids & test_ids

    if train_val:
        raise ValueError(f"Split leakage: {len(train_val)} videos in both train and val")
    if train_test:
        raise ValueError(f"Split leakage: {len(train_test)} videos in both train and test")
    if val_test:
        raise ValueError(f"Split leakage: {len(val_test)} videos in both val and test")

    print("[INFO] Split integrity verified: no video_id overlap")


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False