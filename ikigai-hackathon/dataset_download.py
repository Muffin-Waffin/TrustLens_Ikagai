"""
SynthGuard Phase 1: Dataset Download Module

Downloads a prototype subset from Hugging Face FaceForensicsC23 dataset.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml
from huggingface_hub import hf_hub_download, list_repo_files
from tqdm import tqdm


category_map = {
    "real": "Real",
    "deepfakes": "DeepFakes",
    "face2face": "Face2Face",
    "faceswap": "FaceSwap",
    "neuraltextures": "NeuralTextures",
}


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
        version = getattr(torch, '__version__', 'unknown')
        print(f"[INFO] PyTorch: {version}")
        print(f"[INFO] CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("[WARN] CUDA not available - will use CPU")
    except ImportError:
        print("[INFO] PyTorch: not installed")
    except Exception as e:
        print(f"[INFO] PyTorch: error - {e}")
    
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        print(f"[INFO] ONNX providers: {providers}")
    except ImportError:
        print("[INFO] ONNX Runtime: not installed")
    
    try:
        import insightface
        print("[INFO] InsightFace: available")
    except ImportError:
        print("[INFO] InsightFace: not installed")
    
    try:
        import huggingface_hub
        print(f"[INFO] HuggingFace Hub: {huggingface_hub.__version__}")
    except ImportError:
        print("[INFO] HuggingFace Hub: not installed")
    
    try:
        import datasets
        print(f"[INFO] Datasets: {datasets.__version__}")
    except ImportError:
        print("[INFO] Datasets: not installed")
    
    print("=" * 60)


def get_dataset_structure(repo_id: str) -> dict[str, list[str]]:
    """Inspect the dataset repository structure."""
    print(f"[INFO] Inspecting dataset structure: {repo_id}")
    files = list_repo_files(repo_id, repo_type="dataset")
    
    structure = {}
    for f in files:
        parts = f.split("/")
        if len(parts) >= 2:
            category = parts[0]
            if category not in structure:
                structure[category] = []
            structure[category].append(f)
    
    # Handle zip file structure - check if it's a single zip
    if "FaceForensics++_C23.zip" in files:
        print("[INFO] Found FaceForensics++_C23.zip - this dataset needs to be downloaded and extracted")
        structure["zip"] = ["FaceForensics++_C23.zip"]
    
    for cat, fs in structure.items():
        print(f"[INFO]   {cat}: {len(fs)} files")
    
    return structure


def select_prototype_videos(
    structure: dict[str, list[str]],
    prototype_counts: dict[str, int],
    random_seed: int
) -> list[tuple[str, str, str]]:
    """
    Select videos for the prototype subset.
    
    Returns list of (video_path, label, manipulation_type)
    """
    random.seed(random_seed)
    
    category_map = {
        "real": "Real",
        "deepfakes": "DeepFakes",
        "face2face": "Face2Face",
        "faceswap": "FaceSwap",
        "neuraltextures": "NeuralTextures",
    }
    
    selected = []
    
    # Handle zip file case
    if "zip" in structure:
        print("[INFO] Dataset is a zip file - downloading and extracting...")
        # Return special marker for zip handling
        return [("ZIP_DOWNLOAD_REQUIRED", 0, "zip")]
    
    for config_key, count in prototype_counts.items():
        if count <= 0:
            continue
        
        hf_category = config_key.lower()
        if hf_category not in structure:
            print(f"[WARN] Category '{hf_category}' not found in dataset")
            continue
        
        available = structure[hf_category]
        video_files = [f for f in available if f.endswith((".mp4", ".avi", ".mov"))]
        
        if len(video_files) < count:
            print(f"[WARN] Only {len(video_files)} videos available for {hf_category}, requested {count}")
            count = len(video_files)
        
        selected_videos = random.sample(video_files, count)
        label = 0 if config_key == "real" else 1
        manipulation = category_map.get(config_key, config_key)
        
        for v in selected_videos:
            selected.append((v, label, manipulation))
    
    return selected


def download_videos(
    repo_id: str,
    selected_videos: list[tuple[str, str, str]],
    output_dir: Path,
    token: str | None = None
) -> list[dict[str, Any]]:
    """Download selected videos from Hugging Face."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    downloaded = []
    failed = []
    
    # Handle zip download case
    if selected_videos and selected_videos[0][0] == "ZIP_DOWNLOAD_REQUIRED":
        zip_path = output_dir / "FaceForensics++_C23.zip"
        
        if not zip_path.exists():
            print(f"[INFO] Downloading FaceForensics++_C23.zip (this may take a while)...")
            try:
                local_path = hf_hub_download(
                    repo_id=repo_id,
                    filename="FaceForensics++_C23.zip",
                    repo_type="dataset",
                    token=token,
                    local_dir=str(output_dir),
                    local_dir_use_symlinks=False,
                )
                local_path = Path(local_path)
                if local_path != zip_path:
                    shutil.move(str(local_path), str(zip_path))
                print(f"[INFO] Zip downloaded: {zip_path}")
            except Exception as e:
                print(f"[ERROR] Failed to download zip: {e}")
                return []
        
        # Extract zip
        print(f"[INFO] Extracting zip file...")
        import zipfile
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Extract to a temporary location first
                extract_dir = output_dir / "extracted"
                extract_dir.mkdir(parents=True, exist_ok=True)
                zip_ref.extractall(extract_dir)
                print(f"[INFO] Extracted to {extract_dir}")
                
                # Find video files
                video_files = list(extract_dir.rglob("*.mp4")) + list(extract_dir.rglob("*.avi")) + list(extract_dir.rglob("*.mov"))
                print(f"[INFO] Found {len(video_files)} video files")
                
                # Organize by category (folder structure)
                for video_file in video_files:
                    rel_path = video_file.relative_to(extract_dir)
                    parts = rel_path.parts
                    if len(parts) >= 2:
                        category = parts[0]
                        manipulation = category_map.get(category.lower(), category)
                        label = 0 if category.lower() == "real" else 1
                        
                        video_id = video_file.stem
                        new_path = output_dir / f"{video_id}{video_file.suffix}"
                        
                        if not new_path.exists():
                            shutil.move(str(video_file), str(new_path))
                        
                        downloaded.append({
                            "video_id": video_id,
                            "video_path": str(new_path.relative_to(output_dir)),
                            "absolute_path": str(new_path),
                            "label": label,
                            "manipulation_type": manipulation,
                            "source_dataset": "FaceForensicsC23",
                        })
        except Exception as e:
            print(f"[ERROR] Failed to extract zip: {e}")
            import traceback
            traceback.print_exc()
        
        return downloaded
    
    print(f"[INFO] Downloading {len(selected_videos)} videos to {output_dir}")
    
    for video_rel_path, label, manipulation in tqdm(selected_videos, desc="Downloading", unit="video"):
        try:
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=video_rel_path,
                repo_type="dataset",
                token=token,
                local_dir=str(output_dir),
                local_dir_use_symlinks=False,
            )
            
            local_path = Path(local_path)
            if not local_path.exists():
                failed.append({
                    "video_rel_path": video_rel_path,
                    "error": "Download completed but file not found"
                })
                continue
            
            video_id = Path(video_rel_path).stem
            new_path = output_dir / f"{video_id}{local_path.suffix}"
            
            if local_path != new_path:
                shutil.move(str(local_path), str(new_path))
            
            downloaded.append({
                "video_id": video_id,
                "video_path": str(new_path.relative_to(output_dir)),
                "absolute_path": str(new_path),
                "label": label,
                "manipulation_type": manipulation,
                "source_dataset": "FaceForensicsC23",
            })
            
        except Exception as e:
            failed.append({
                "video_rel_path": video_rel_path,
                "error": str(e)
            })
    
    if failed:
        broken_path = output_dir / "broken.csv"
        import pandas as pd
        pd.DataFrame(failed).to_csv(broken_path, index=False)
        print(f"[WARN] {len(failed)} downloads failed. See {broken_path}")
    
    return downloaded


def save_manifest(videos: list[dict[str, Any]], output_path: Path) -> None:
    """Save the video manifest as CSV."""
    import pandas as pd
    df = pd.DataFrame(videos)
    original_len = len(df)
    df = df.drop_duplicates(subset=["video_id"], keep="first")
    print(f"[INFO] Removed {original_len - len(df)} duplicate video_ids")
    df.to_csv(output_path, index=False)
    print(f"[INFO] Manifest saved: {output_path} ({len(df)} unique videos)")


def main():
    parser = argparse.ArgumentParser(description="SynthGuard Phase 1: Dataset Download")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--token", default=None, help="Hugging Face token (optional for public datasets)")
    args = parser.parse_args()
    
    config = load_config(args.config)
    print_env_info(config)
    
    repo_id = config["dataset"]["name"]
    prototype_counts = config["dataset"]["prototype"]
    random_seed = config["dataset"]["random_seed"]
    raw_dir = Path(config["paths"]["raw"])
    subset_dir = Path(config["paths"]["subset"])
    
    print(f"\n[INFO] Dataset: {repo_id}")
    print(f"[INFO] Prototype counts: {prototype_counts}")
    print(f"[INFO] Random seed: {random_seed}")
    print(f"[INFO] Output directory: {raw_dir}")
    
    structure = get_dataset_structure(repo_id)
    
    selected = select_prototype_videos(structure, prototype_counts, random_seed)
    print(f"[INFO] Selected {len(selected)} videos total")
    
    if not selected:
        print("[ERROR] No videos selected. Check dataset structure and config.")
        sys.exit(1)
    
    downloaded = download_videos(repo_id, selected, raw_dir, args.token)
    
    if not downloaded:
        print("[ERROR] No videos downloaded successfully.")
        sys.exit(1)
    
    manifest_path = subset_dir / "all.csv"
    save_manifest(downloaded, manifest_path)
    
    print(f"\n[INFO] Download complete: {len(downloaded)} videos")
    print(f"[INFO] Manifest: {manifest_path}")


if __name__ == "__main__":
    main()