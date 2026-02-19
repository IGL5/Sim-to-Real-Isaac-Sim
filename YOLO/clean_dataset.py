import os
import glob
import argparse
import cv2
import numpy as np
from pathlib import Path

def clean_dataset(data_dir, dry_run=False, threshold_mean=10.0, threshold_std=5.0):
    """
    Analizes datasets of Isaac Sim with hierarchical structure:
    root/CameraName/rgb/*.png
    root/CameraName/object_detection/*.txt
    ...
    """
    
    # 1. Search recursively for all 'rgb' folders inside the data directory
    # This works even if you have multiple cameras (DroneCamera, Camera2, etc.)
    search_pattern = os.path.join(data_dir, "**", "rgb")
    rgb_folders = glob.glob(search_pattern, recursive=True)
    
    if not rgb_folders:
        print(f"[ERROR] No rgb folders found inside: {data_dir}")
        print("Make sure you point to the root directory (e.g: _output_data)")
        return

    print(f"--- Starting cleaning in {len(rgb_folders)} camera folders ---")
    
    total_deleted = 0
    total_kept = 0

    for rgb_folder in rgb_folders:
        # rgb_folder es ej: "_output_data/DroneCamera/rgb"
        camera_root = os.path.dirname(rgb_folder) # ej: "_output_data/DroneCamera"
        camera_name = os.path.basename(camera_root)
        
        print(f"\nProcessing camera: {camera_name}")
        
        # List images png in the rgb folder
        png_files = sorted(glob.glob(os.path.join(rgb_folder, "*.png")))
        
        if not png_files:
            print("  -> Empty folder.")
            continue
            
        print(f"  -> Analyzing {len(png_files)} images...")

        for rgb_path in png_files:
            # Read image
            img = cv2.imread(rgb_path)
            
            is_bad = False
            if img is None:
                print(f"  [WARN] Corrupted file: {rgb_path}")
                is_bad = True
            else:
                # Calculate metrics
                mean_val = np.mean(img)
                std_val = np.std(img)
                
                # Criteria: Too dark OR Flat
                if mean_val < threshold_mean or std_val < threshold_std:
                    is_bad = True
                    # print(f"    Detected bad frame: {os.path.basename(rgb_path)} (Mean:{mean_val:.1f})")

            if is_bad:
                if not dry_run:
                    delete_hierarchical_frame(rgb_path, camera_root)
                else:
                    print(f"  [DRY-RUN] The frame ID: {Path(rgb_path).stem} would be deleted.")
                
                total_deleted += 1
            else:
                total_kept += 1

    print("\n--- FINAL SUMMARY ---")
    print(f"✅ Valid frames kept: {total_kept}")
    print(f"🗑️ Corrupted frames deleted: {total_deleted}")
    if dry_run:
        print("⚠️  DRY-RUN: No files were deleted.")


def delete_hierarchical_frame(rgb_path, camera_root):
    """
    Deletes the frame identified by 'rgb_path' in ALL subfolders of 'camera_root'
    (rgb, depth, object_detection, etc.).
    
    Args:
        rgb_path: Full path to the bad image (e.g: .../rgb/0.png)
        camera_root: Root directory of the camera (e.g: .../DroneCamera)
    """
    # Get the frame ID (file name without extension)
    # ej: "0.png" -> stem es "0"
    frame_id = Path(rgb_path).stem
    
    # List all data subfolders (rgb, depth, labels, etc.)
    # Scan camera_root to see what folders exist
    subfolders = [f.path for f in os.scandir(camera_root) if f.is_dir()]
    
    for folder in subfolders:
        # In each folder, look for files named "frame_id.*"
        # This will cover 0.png, 0.txt, 0.npy, 0.json...
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
    
    args = parser.parse_args()
    
    if os.path.exists(args.dir):
        clean_dataset(args.dir, dry_run=args.dry, threshold_mean=args.thresh_mean)
    else:
        print(f"The directory {args.dir} does not exist.")