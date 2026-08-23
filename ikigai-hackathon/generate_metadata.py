"""
Generate metadata JSONs from existing face crops (fast path).
Skips face detection since crops already exist.
"""

import json
from pathlib import Path

import pandas as pd
import yaml

from preprocessing import get_video_metadata


def load_config(config_path: str | Path) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def generate_metadata_from_existing(
    split_name: str,
    config: dict,
    paths: dict[str, Path],
) -> int:
    """Generate metadata JSONs from existing face crops."""
    splits_dir = paths["splits"]
    split_path = splits_dir / f"{split_name}.csv"
    
    if not split_path.exists():
        print(f"[WARN] Split not found: {split_path}")
        return 0
    
    df = pd.read_csv(split_path)
    print(f"[INFO] Generating metadata for {split_name}: {len(df)} videos")
    
    faces_root = paths["faces"]
    frames_root = paths["frames"]
    metadata_dir = paths["metadata"]
    metadata_dir.mkdir(parents=True, exist_ok=True)
    
    raw_root = paths["raw"]
    project_root = Path(".")
    
    success_count = 0
    
    for _, row in df.iterrows():
        video_id = row["video_id"]
        label = int(row["label"])
        manipulation_type = row["manipulation_type"]
        
        video_path = raw_root / row["video_path"]
        
        face_dir = faces_root / video_id
        if not face_dir.exists():
            print(f"[WARN] No face crops for {video_id}")
            continue
        
        face_files = sorted(face_dir.glob("frame_*.jpg"))
        if not face_files:
            print(f"[WARN] No face frames in {face_dir}")
            continue
        
        # Get video metadata for timestamps
        try:
            video_meta = get_video_metadata(video_path)
        except Exception as e:
            print(f"[WARN] Failed to get video metadata for {video_id}: {e}")
            continue
        
        frames_data = []
        for face_file in face_files:
            frame_idx = int(face_file.stem.split("_")[1])
            timestamp = frame_idx / video_meta["fps"] if video_meta["fps"] > 0 else 0.0
            
            rel_frame_path = (frames_root / video_id / face_file.name).relative_to(project_root)
            rel_face_path = face_file.relative_to(project_root)
            
            frames_data.append({
                "frame_index": frame_idx,
                "timestamp_seconds": float(timestamp),
                "face_found": True,
                "face_confidence": 0.9,  # placeholder
                "face_area_ratio": 0.05,  # placeholder
                "blur_score": 100.0,  # placeholder
                "face_quality": 0.8,  # placeholder - above threshold
                "usable": True,
                "bbox": [0, 0, 0, 0],  # placeholder
                "frame_path": str(rel_frame_path),
                "face_path": str(rel_face_path),
            })
        
        usable_frames = len([f for f in frames_data if f["usable"]])
        
        result = {
            "dataset": {
                "name": "FaceForensicsC23",
                "source": "bitmind/FaceForensicsC23"
            },
            "label": label,
            "manipulation_type": manipulation_type,
            "video": video_meta,
            "sampling": {
                "requested_fps": config["sampling"]["fps"],
                "max_frames": config["sampling"]["max_frames"],
                "sampled_frames": len(frames_data),
            },
            "face_processing": {
                "frames_with_faces": len(frames_data),
                "usable_face_frames": usable_frames,
                "coverage": 1.0,
                "average_face_quality": 0.8,
            },
            "frames": frames_data,
            "runtime": {
                "device": "CPU",
                "face_provider": "CPUExecutionProvider",
            },
        }
        
        metadata_path = metadata_dir / f"{video_id}.json"
        with open(metadata_path, "w") as f:
            json.dump(result, f, indent=2)
        
        success_count += 1
        if success_count % 100 == 0:
            print(f"  Generated {success_count} metadata files...")
    
    print(f"[INFO] Generated {success_count} metadata files for {split_name}")
    return success_count


def main():
    config = load_config("config.yaml")
    paths = {k: Path(v) for k, v in config["paths"].items()}
    
    total = 0
    for split in ["train", "val", "test"]:
        total += generate_metadata_from_existing(split, config, paths)
    
    print(f"\n[INFO] Total metadata files generated: {total}")


if __name__ == "__main__":
    main()