"""
SynthGuard: Deepfake Classifier Models

Supports:
- ConvNeXt-Tiny (legacy)
- Xception (legacy_xception from timm) - PRIMARY
"""

from __future__ import annotations

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Union

import timm


class DeepfakeClassifier(nn.Module):
    """ConvNeXt-Tiny based binary deepfake classifier (legacy)."""

    def __init__(
        self,
        pretrained: bool = True,
        dropout: float = 0.30,
    ):
        super().__init__()

        from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = convnext_tiny(weights=weights)

        in_features = self.backbone.classifier[-1].in_features
        self.backbone.classifier[-1] = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x).squeeze(-1)


class XceptionDeepfakeClassifier(nn.Module):
    """Xception (legacy_xception) based deepfake classifier with 2-class output."""

    def __init__(
        self,
        pretrained: bool = False,
        num_classes: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.backbone = timm.create_model(
            "legacy_xception",
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before final classifier."""
        return self.backbone.forward_features(x)


def build_model(
    model_name: str = "xception",
    pretrained: bool = False,
    dropout: float = 0.0,
    num_classes: int = 2,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """
    Build and return the model on the specified device.

    Args:
        model_name: "xception" or "convnext_tiny"
        pretrained: Use ImageNet pretrained weights (only for convnext_tiny)
        dropout: Dropout rate
        num_classes: Number of output classes (2 for Xception, 1 for ConvNeXt)
        device: Target device (auto-detected if None)

    Returns:
        Model on device
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"[INFO] Device: CUDA")
            print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            print("[WARN] CUDA unavailable. Running on CPU.")

    if model_name == "xception":
        model = XceptionDeepfakeClassifier(
            pretrained=pretrained,
            num_classes=num_classes,
            dropout=dropout,
        )
    elif model_name == "convnext_tiny":
        model = DeepfakeClassifier(pretrained=pretrained, dropout=dropout)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model = model.to(device)
    return model


def load_xception_checkpoint(
    checkpoint_path: Union[str, Path],
    model: XceptionDeepfakeClassifier,
    device: Optional[torch.device] = None,
) -> dict:
    """
    Load Xception checkpoint from file.

    Handles:
    - 'backbone.' prefix in keys
    - 'last_linear' -> 'fc' rename
    - Extra 'adjust_channel' keys (ignored)

    Args:
        checkpoint_path: Path to checkpoint file
        model: XceptionDeepfakeClassifier instance
        device: Target device

    Returns:
        Checkpoint metadata dict
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[MODEL] Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle raw state_dict vs full checkpoint
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        metadata = {k: v for k, v in checkpoint.items() if k != "model_state_dict"}
    else:
        state_dict = checkpoint
        metadata = {}

    # Remap keys: keep 'backbone.' prefix (model has self.backbone), rename 'last_linear' to 'fc', skip 'adjust_channel'
    new_state_dict = {}
    for k, v in state_dict.items():
        nk = k  # keep as-is since model has self.backbone

        if nk == "backbone.last_linear.weight":
            nk = "backbone.fc.weight"
        elif nk == "backbone.last_linear.bias":
            nk = "backbone.fc.bias"
        elif nk.startswith("backbone.adjust_channel"):
            continue  # skip auxiliary projection head

        new_state_dict[nk] = v

    # Load with strict=True to verify compatibility
    missing, unexpected = model.load_state_dict(new_state_dict, strict=True)

    if missing:
        print(f"[WARN] Missing keys: {missing}")
    if unexpected:
        print(f"[WARN] Unexpected keys: {unexpected}")

    print(f"[MODEL] Xception loaded successfully")
    print(f"[MODEL] Checkpoint: {checkpoint_path}")
    print(f"[MODEL] Device: {device}")
    print(f"[MODEL] Parameters: {sum(p.numel() for p in model.parameters()):,}")

    model.eval()
    return metadata


def load_convnext_checkpoint(
    checkpoint_path: Union[str, Path],
    model: DeepfakeClassifier,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> dict:
    """Load ConvNeXt checkpoint (legacy format)."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint


def load_checkpoint(
    checkpoint_path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> dict:
    """
    Universal checkpoint loader. Detects model type and loads appropriately.
    """
    if isinstance(model, XceptionDeepfakeClassifier):
        return load_xception_checkpoint(checkpoint_path, model, device)
    else:
        return load_convnext_checkpoint(checkpoint_path, model, optimizer, device)


def save_checkpoint(
    checkpoint_path: Union[str, Path],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    config: dict,
) -> None:
    """Save model checkpoint."""
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
        "config": config,
    }, checkpoint_path)


def get_model_info(model: nn.Module) -> dict:
    """Get model information."""
    return {
        "architecture": model.__class__.__name__,
        "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }


if __name__ == "__main__":
    # Quick test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model("xception", pretrained=False, device=device)
    print(get_model_info(model))

    x = torch.randn(2, 3, 224, 224).to(device)
    with torch.no_grad():
        out = model(x)
    print(f"Output shape: {out.shape}")
    probs = torch.softmax(out, dim=1)
    print(f"Probabilities: {probs}")