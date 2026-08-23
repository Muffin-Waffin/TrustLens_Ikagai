"""
SynthGuard Phase 4: Explainability Module

Implements Grad-CAM for ConvNeXt and Xception to visualize model attention on face crops.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Any, Union

from model import DeepfakeClassifier, XceptionDeepfakeClassifier


class GradCAM:
    """
    Grad-CAM implementation for ConvNeXt and Xception models.

    Generates class activation maps showing which regions of the input
    influenced the model's prediction.
    """

    def __init__(
        self,
        model: Union[DeepfakeClassifier, XceptionDeepfakeClassifier],
        target_layer_name: str = "features.7.2",
    ):
        self.model = model
        self.target_layer_name = target_layer_name
        self.gradients = None
        self.activations = None
        self._is_xception = isinstance(model, XceptionDeepfakeClassifier)
        self._register_hooks()

    def _register_hooks(self) -> None:
        """Register forward and backward hooks on target layer."""
        target_layer = self._get_target_layer()

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def _get_target_layer(self) -> nn.Module:
        """Get target layer by name."""
        parts = self.target_layer_name.split(".")
        if self._is_xception:
            module = self.model.backbone
        else:
            module = self.model.backbone
        for part in parts:
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part)
        return module

    def generate_cam(self, input_tensor: torch.Tensor, class_idx: int = 1) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for the input.

        Args:
            input_tensor: Input tensor (1, 3, H, W)
            class_idx: Target class index (1 for fake class in Xception, 0 for single logit)
            
        Returns:
            CAM heatmap as numpy array (H, W) in [0, 1]
        """
        self.model.eval()

        # Forward pass
        logits = self.model(input_tensor)

        # For Xception (2-class output), use class_idx (1 = fake)
        # For ConvNeXt (single logit), use the logit directly
        if self._is_xception:
            if logits.dim() == 2:
                score = logits[0, class_idx]
            else:
                score = logits[class_idx]
        else:
            # ConvNeXt single logit
            if logits.dim() == 1:
                score = logits[0]
            else:
                score = logits[0, 0]

        # Backward pass
        self.model.zero_grad()
        score.backward(retain_graph=True)

        # Get gradients and activations
        gradients = self.gradients
        activations = self.activations

        if gradients is None or activations is None:
            raise RuntimeError("Gradients or activations not captured. Check target layer.")

        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)

        # Weighted combination of activations
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = F.relu(cam)

        # Resize to input size
        cam = F.interpolate(
            cam,
            size=(input_tensor.shape[2], input_tensor.shape[3]),
            mode="bilinear",
            align_corners=False
        )

        # Normalize to [0, 1]
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam.squeeze().cpu().numpy()


def apply_colormap(heatmap: np.ndarray, colormap: str = "jet") -> np.ndarray:
    """Apply OpenCV colormap to heatmap."""
    heatmap_uint8 = np.uint8(255 * heatmap)

    colormap_map = {
        "jet": cv2.COLORMAP_JET,
        "hot": cv2.COLORMAP_HOT,
        "viridis": cv2.COLORMAP_VIRIDIS,
        "plasma": cv2.COLORMAP_PLASMA,
        "inferno": cv2.COLORMAP_INFERNO,
        "magma": cv2.COLORMAP_MAGMA,
    }

    cv_colormap = colormap_map.get(colormap, cv2.COLORMAP_JET)
    colored = cv2.applyColorMap(heatmap_uint8, cv_colormap)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def create_overlay(
    original_image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Create overlay of heatmap on original image.

    Args:
        original_image: RGB image (H, W, 3) in [0, 255]
        heatmap: Normalized heatmap (H, W) in [0, 1]
        alpha: Overlay transparency

    Returns:
        Overlay image (H, W, 3) in [0, 255]
    """
    heatmap_colored = apply_colormap(heatmap)
    heatmap_resized = cv2.resize(heatmap_colored, (original_image.shape[1], original_image.shape[0]))

    overlay = cv2.addWeighted(original_image, 1 - alpha, heatmap_resized, alpha, 0)
    return overlay.astype(np.uint8)


def generate_explanations_for_video(
    frame_predictions: list[dict],
    video_id: str,
    config: dict[str, Any],
    model: Union[DeepfakeClassifier, XceptionDeepfakeClassifier],
    device: torch.device,
) -> list[dict]:
    """
    Generate Grad-CAM explanations for suspicious frames.

    Args:
        frame_predictions: Frame data with scores
        video_id: Video identifier
        config: Configuration dict
        model: Trained model
        device: Torch device

    Returns:
        List of explanation dicts with paths
    """
    explain_config = config.get("explainability", {})
    if not explain_config.get("enabled", True):
        return []

    is_xception = isinstance(model, XceptionDeepfakeClassifier)
    target_layer = explain_config.get(
        "target_layer",
        "conv4.pointwise" if is_xception else "features.7.2"
    )
    max_frames = explain_config.get("max_frames", 6)
    colormap = explain_config.get("colormap", "jet")

    # Get top suspicious frames
    usable_frames = [f for f in frame_predictions if f.get("usable", False) and f.get("score", 0) > 0.5]
    usable_frames.sort(key=lambda f: f.get("score", 0), reverse=True)
    top_frames = usable_frames[:max_frames]

    if not top_frames:
        return []

    gradcam = GradCAM(model, target_layer)
    transform = None

    from torchvision import transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    output_dir = Path(config["paths"].get("explanations", "./outputs/explanations")) / video_id
    output_dir.mkdir(parents=True, exist_ok=True)

    explanations = []

    for frame_info in top_frames:
        frame_path = frame_info.get("face_path", "")
        if not frame_path or not Path(frame_path).exists():
            continue

        # Load original face crop
        original = cv2.imread(frame_path)
        if original is None:
            continue
        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

        # Prepare tensor
        input_tensor = transform(original_rgb).unsqueeze(0).to(device)

        try:
            # Generate CAM - use class_idx=1 for fake class in Xception
            class_idx = 1 if is_xception else 0
            cam = gradcam.generate_cam(input_tensor, class_idx=class_idx)

            # Create heatmap and overlay
            heatmap_colored = apply_colormap(cam, colormap)
            overlay = create_overlay(original_rgb, cam, alpha=0.5)

            # Save files
            base_name = f"frame_{frame_info['frame_index']:06d}"
            heatmap_path = output_dir / f"{base_name}_heatmap.jpg"
            overlay_path = output_dir / f"{base_name}_overlay.jpg"
            original_path = output_dir / f"{base_name}_original.jpg"

            cv2.imwrite(str(heatmap_path), cv2.cvtColor(heatmap_colored, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(original_path), cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR))

            explanations.append({
                "frame_index": frame_info["frame_index"],
                "timestamp_seconds": frame_info["timestamp_seconds"],
                "score": frame_info["score"],
                "heatmap_path": str(heatmap_path),
                "overlay_path": str(overlay_path),
                "original_path": str(original_path),
            })

        except Exception as e:
            print(f"[WARN] Failed to generate explanation for frame {frame_info['frame_index']}: {e}")
            continue

    # Save explanations JSON
    import json
    with open(output_dir / "explanations.json", "w") as f:
        json.dump({
            "video_id": video_id,
            "explanations": explanations,
            "method": "Grad-CAM",
            "target_layer": target_layer,
        }, f, indent=2)

    return explanations


def create_enhanced_timeline(
    frame_predictions: list[dict],
    video_id: str,
    config: dict[str, Any],
    forensic_result: dict[str, Any] | None = None,
) -> str | None:
    """
    Create enhanced timeline visualization with suspicious segments.

    Returns:
        Path to saved timeline image
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        return None

    output_dir = Path(config["paths"].get("explanations", "./outputs/explanations")) / video_id
    output_dir.mkdir(parents=True, exist_ok=True)

    usable = [f for f in frame_predictions if f.get("usable", False)]
    if not usable:
        return None

    timestamps = [f["timestamp_seconds"] for f in usable]
    scores = [f["score"] for f in usable]

    fig, ax = plt.subplots(figsize=(12, 5))

    # Plot all scores
    ax.plot(timestamps, scores, "b-", alpha=0.6, linewidth=1, label="Frame Score")
    ax.scatter(timestamps, scores, c=scores, cmap="Reds", s=25, alpha=0.8, edgecolors="none")

    # Threshold line
    threshold = config["forensic"]["suspicious_frame_threshold"]
    ax.axhline(y=threshold, color="red", linestyle="--", alpha=0.7, linewidth=1.5, label=f"Suspicious Threshold ({threshold})")

    # Mean line
    mean_score = np.mean(scores)
    ax.axhline(y=mean_score, color="blue", linestyle=":", alpha=0.7, linewidth=1.5, label=f"Mean Score ({mean_score:.2f})")

    # Shade suspicious segments
    if forensic_result and forensic_result.get("suspicious_segments"):
        for seg in forensic_result["suspicious_segments"]:
            ax.axvspan(seg["start"], seg["end"], alpha=0.15, color="red", label="Suspicious Segment" if seg == forensic_result["suspicious_segments"][0] else "")

    ax.set_xlabel("Time (seconds)", fontsize=12)
    ax.set_ylabel("Manipulation Score", fontsize=12)
    ax.set_title(f"Frame-Level Manipulation Scores: {video_id}", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=10)

    plt.tight_layout()
    timeline_path = output_dir / f"{video_id}_timeline.png"
    plt.savefig(timeline_path, dpi=config["forensic"]["visualization"]["timeline_dpi"])
    plt.close()

    return str(timeline_path)