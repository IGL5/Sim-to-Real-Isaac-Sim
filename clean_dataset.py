import os
import glob
import argparse
import cv2
import json
import numpy as np
from pathlib import Path

def clean_dataset(data_dir, dry_run=False, threshold_mean=10.0, threshold_std=5.0, remove_empty=False):
    """
    Analizes datasets of Isaac Sim with hierarchical structure:
    root/CameraName/rgb/*.png
    root/CameraName/object_detection/*.txt
    ...
    """
    
    # Search recursively for all 'rgb' folders inside the data directory
    search_pattern = os.path.join(data_dir, "**", "rgb")
    rgb_folders = glob.glob(search_pattern, recursive=True)
    
    if not rgb_folders:
        print(f"[ERROR] No rgb folders found inside: {data_dir}")
        print("Make sure you point to the root directory (e.g: _output_data)")
        return

    print(f"--- Starting cleaning in {len(rgb_folders)} camera folders ---")
    if remove_empty:
        print("🧹 Mode active: Empty frames (backgrounds without objects) WILL BE DELETED.")
    
    total_corrupted = 0
    total_empty = 0
    total_kept = 0

    for rgb_folder in rgb_folders:
        camera_root = os.path.dirname(rgb_folder) 
        camera_name = os.path.basename(camera_root)
        
        print(f"\nProcessing camera: {camera_name}")
        
        png_files = sorted(glob.glob(os.path.join(rgb_folder, "*.png")))
        
        if not png_files:
            print("  -> Empty folder.")
            continue
            
        print(f"  -> Analyzing {len(png_files)} images...")

        for rgb_path in png_files:
            frame_id = Path(rgb_path).stem
            img = cv2.imread(rgb_path)
            
            is_corrupted = False
            is_empty = False

            if img is None:
                print(f"  [WARN] Corrupted file: {rgb_path}")
                is_corrupted = True
            else:
                # Calculate metrics
                mean_val = np.mean(img)
                std_val = np.std(img)
                
                # Criteria: Too dark OR Flat
                if mean_val < threshold_mean or std_val < threshold_std:
                    is_corrupted = True

            if not is_corrupted and remove_empty:
                txt_path = os.path.join(camera_root, "object_detection", f"{frame_id}.txt")
                if os.path.exists(txt_path):
                    with open(txt_path, 'r') as f:
                        if not f.read().strip():
                            is_empty = True
                else:
                    is_empty = True

            if is_corrupted or is_empty:
                if not dry_run:
                    delete_hierarchical_frame(rgb_path, camera_root)
                else:
                    reason = "CORRUPTED" if is_corrupted else "EMPTY"
                    print(f"  [DRY-RUN] The frame ID: {frame_id} would be deleted ({reason}).")
                
                if is_corrupted:
                    total_corrupted += 1
                elif is_empty:
                    total_empty += 1
            else:
                total_kept += 1

    print("\n--- FINAL SUMMARY ---")
    print(f"✅ Valid frames kept: {total_kept}")
    print(f"🗑️ Corrupted frames deleted: {total_corrupted}")
    if remove_empty or total_empty > 0:
        print(f"🧹 Empty frames (backgrounds) deleted: {total_empty}")

    if dry_run:
        print("⚠️  DRY-RUN: No files were deleted.")
    else:
        metadata_path = os.path.join(data_dir, "generation_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            
            meta["cleaning"] = {
                "corrupted_deleted": total_corrupted,
                "empty_deleted": total_empty,
                "valid_kept": total_kept
            }
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(meta, f, indent=4)
            print(f"📝 Metadata updated with cleaning stats in: {metadata_path}")
        else:
            print("[WARN] Metadata file not found. Skipping metadata update.")


def delete_hierarchical_frame(rgb_path, camera_root):
    """
    Deletes the frame identified by 'rgb_path' in ALL subfolders of 'camera_root'
    (rgb, depth, object_detection, etc.).
    """
    frame_id = Path(rgb_path).stem
    
    subfolders = [f.path for f in os.scandir(camera_root) if f.is_dir()]
    
    for folder in subfolders:
        target_pattern = os.path.join(folder, f"{frame_id}.*")
        matching_files = glob.glob(target_pattern)
        
        for f_path in matching_files:
            try:
                os.remove(f_path)
            except OSError as e:
                print(f"  [ERROR] Could not delete {f_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset cleaner for Isaac Sim datasets")
    parser.add_argument("--dir", type=str, default="_output_data", help="Root directory of the dataset")
    parser.add_argument("--dry", action="store_true", help="Dry run, do not delete files")
    parser.add_argument("--thresh_mean", type=float, default=5.0, help="Darkness threshold (0-255)")
    parser.add_argument("--empty", action="store_true", help="Delete pure backgrounds")
    
    args = parser.parse_args()
    
    if os.path.exists(args.dir):
        clean_dataset(args.dir, dry_run=args.dry, threshold_mean=args.thresh_mean, remove_empty=args.empty)
    else:
        print(f"The directory {args.dir} does not exist.")