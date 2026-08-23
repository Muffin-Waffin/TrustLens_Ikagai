"""
SynthGuard Phase 2: Training Script

Trains ConvNeXt-Tiny on frame-level face crops with video-level splits.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from model import DeepfakeClassifier, build_model, save_checkpoint
from data_loader import create_dataloaders, set_seed


def load_config(config_path: str | Path) -> dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train_one_epoch(
    model: DeepfakeClassifier,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler | None,
    device: torch.device,
    max_grad_norm: float = 1.0,
) -> tuple[float, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    use_amp = scaler is not None and device.type == "cuda"

    for batch in tqdm(loader, desc="Training", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad()

        if use_amp:
            with autocast(device_type="cuda"):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        total_loss += loss.item() * images.size(0)

        probs = torch.sigmoid(logits).detach().cpu().numpy()
        preds = (probs > 0.5).astype(int)
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy


@torch.no_grad()
def validate(
    model: DeepfakeClassifier,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Validate model."""
    model.eval()
    total_loss = 0.0
    all_probs = []
    all_labels = []

    use_amp = device.type == "cuda"

    for batch in tqdm(loader, desc="Validation", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        if use_amp:
            with autocast(device_type="cuda"):
                logits = model(images)
                loss = criterion(logits, labels)
        else:
            logits = model(images)
            loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)

        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)

    probs_np = np.array(all_probs)
    labels_np = np.array(all_labels)
    preds = (probs_np > 0.5).astype(int)

    metrics = {
        "loss": avg_loss,
        "accuracy": accuracy_score(labels_np, preds),
        "precision": precision_score(labels_np, preds, zero_division=0),
        "recall": recall_score(labels_np, preds, zero_division=0),
        "f1": f1_score(labels_np, preds, zero_division=0),
        "roc_auc": roc_auc_score(labels_np, probs_np) if len(set(labels_np)) > 1 else 0.0,
    }

    return metrics


def print_epoch_metrics(epoch: int, epochs: int, train_loss: float, train_acc: float, val_metrics: dict) -> None:
    """Print formatted epoch metrics."""
    print(f"\nEpoch {epoch}/{epochs}")
    print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"  Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f}")
    print(f"  Val Precision: {val_metrics['precision']:.4f} | Val Recall: {val_metrics['recall']:.4f}")
    print(f"  Val F1: {val_metrics['f1']:.4f} | Val AUC: {val_metrics['roc_auc']:.4f}")


def save_experiment_log(
    config: dict,
    best_auc: float,
    training_time: float,
    checkpoint_path: Path,
    history: list[dict],
) -> None:
    """Save experiment log."""
    log = {
        "timestamp": datetime.now().isoformat(),
        "model": config["model"]["name"],
        "dataset": "FaceForensicsC23",
        "batch_size": config["training"]["batch_size"],
        "learning_rate": config["training"]["learning_rate"],
        "epochs": config["training"]["epochs"],
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "best_validation_auc": best_auc,
        "training_time_seconds": training_time,
        "checkpoint": str(checkpoint_path),
        "history": history,
    }

    log_path = Path("outputs/evaluation/experiment.json")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"[INFO] Experiment log saved: {log_path}")


def main():
    parser = argparse.ArgumentParser(description="SynthGuard Phase 2: Training")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--resume", help="Resume from checkpoint")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["training"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"[INFO] Device: CUDA")
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("[WARN] CUDA unavailable. Running on CPU.")

    print(f"[INFO] Model: {config['model']['name']}")
    print(f"[INFO] Batch size: {config['training']['batch_size']}")
    print(f"[INFO] Learning rate: {config['training']['learning_rate']}")
    print(f"[INFO] Epochs: {config['training']['epochs']}")
    print(f"[INFO] Early stopping patience: {config['training']['early_stopping_patience']}")

    train_loader, val_loader, _, class_info = create_dataloaders(config)

    model = build_model(
        pretrained=config["model"]["pretrained"],
        dropout=config["model"]["dropout"],
        device=device,
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([class_info["pos_weight"]], device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    )

    scaler = GradScaler() if device.type == "cuda" else None

    start_epoch = 0
    best_auc = 0.0
    patience_counter = 0
    history = []

    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    best_checkpoint = checkpoint_dir / "best_convnext_tiny.pt"

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"]
        best_auc = checkpoint["metrics"].get("roc_auc", 0.0)
        print(f"[INFO] Resumed from epoch {start_epoch}, best AUC: {best_auc:.4f}")

    print("\n[INFO] Starting training...")
    start_time = time.time()

    for epoch in range(start_epoch + 1, config["training"]["epochs"] + 1):
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )
        val_metrics = validate(model, val_loader, criterion, device)

        scheduler.step(val_metrics["roc_auc"])

        epoch_time = time.time() - epoch_start
        print_epoch_metrics(epoch, config["training"]["epochs"], train_loss, train_acc, val_metrics)
        print(f"  Epoch time: {epoch_time:.1f}s | LR: {optimizer.param_groups[0]['lr']:.2e}")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "val_auc": val_metrics["roc_auc"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        })

        current_auc = val_metrics["roc_auc"]
        if current_auc > best_auc:
            best_auc = current_auc
            patience_counter = 0
            save_checkpoint(
                best_checkpoint,
                model,
                optimizer,
                epoch,
                val_metrics,
                config,
            )
            print(f"[INFO] New best checkpoint saved (AUC: {best_auc:.4f})")
        else:
            patience_counter += 1
            print(f"[INFO] No improvement. Patience: {patience_counter}/{config['training']['early_stopping_patience']}")

        if patience_counter >= config["training"]["early_stopping_patience"]:
            print(f"[INFO] Early stopping triggered after {epoch} epochs")
            break

    total_time = time.time() - start_time
    print(f"\n[INFO] Training completed in {total_time/60:.1f} minutes")
    print(f"[INFO] Best validation AUC: {best_auc:.4f}")
    print(f"[INFO] Best checkpoint: {best_checkpoint}")

    save_experiment_log(config, best_auc, total_time, best_checkpoint, history)


if __name__ == "__main__":
    main()