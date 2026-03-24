import os
import glob
import argparse
import cv2
import json
import numpy as np
import shutil
from pathlib import Path

# ==========================================
# 1. CORE LOGIC / ORCHESTRATOR
# ==========================================
def clean_dataset(args):
    """ Orchestrator function that manages the dataset iteration. """
    search_pattern = os.path.join(args.dir, "**", "rgb")
    rgb_folders = glob.glob(search_pattern, recursive=True)
    
    if not rgb_folders:
        print(f"[ERROR] No rgb folders found inside: {args.dir}")
        return

    print(f"--- Starting cleaning in {len(rgb_folders)} camera folders ---")
    
    stats = {"corrupted": 0, "empty": 0, "kept": 0, "reviewed": 0}
    remove_empty_logic = args.empty or args.move_empty

    for rgb_folder in rgb_folders:
        camera_root = os.path.dirname(rgb_folder) 
        camera_name = os.path.basename(camera_root)
        
        print(f"\nProcessing camera: {camera_name}")
        png_files = sorted(glob.glob(os.path.join(rgb_folder, "*.png")))
        
        if not png_files: continue
        print(f"  -> Analyzing {len(png_files)} images...")

        for rgb_path in png_files:
            process_frame(rgb_path, camera_root, camera_name, args, stats, remove_empty_logic)

    print_summary(stats, args)
    if not args.dry:
        update_metadata(args.dir, stats, args)

# ==========================================
# 2. FRAME PROCESSING FLOW
# ==========================================
def process_frame(rgb_path, camera_root, camera_name, args, stats, remove_empty_logic):
    """ Applies the pipeline of checks to a single frame. """
    frame_id = Path(rgb_path).stem
    txt_path = os.path.join(camera_root, "object_detection", f"{frame_id}.txt")
    
    # Read image once for all operations
    img = cv2.imread(rgb_path)

    # CHECK 1: Corrupted or Flat Image
    if is_image_corrupted(img, args.thresh_mean, args.thresh_std):
        handle_action(rgb_path, camera_root, "delete", args.dry, stats, "corrupted", frame_id)
        return # Stop processing this frame

    # CHECK 2: Empty Labels (Backgrounds)
    if remove_empty_logic and is_label_empty(txt_path):
        action = "move" if args.move_empty else "delete"
        target_dir = f"{args.dir}_empty" if args.move_empty else None
        handle_action(rgb_path, camera_root, action, args.dry, stats, "empty", frame_id, target_dir)
        return # Stop processing this frame

    # CHECK 3: Review Mode (Suspicious labels)
    if args.review:
        is_suspicious, annotated_img = analyze_kitti_labels(img, txt_path, args.area_thresh)
        if is_suspicious:
            save_review_image(annotated_img, args.dir, camera_name, frame_id)
            stats["reviewed"] += 1

    # If it survived all checks, we keep it
    stats["kept"] += 1

# ==========================================
# 3. ANALYSIS MODULES
# ==========================================
def is_image_corrupted(img, thresh_mean, thresh_std):
    """ Returns True if image is missing, too dark, or flat. """
    if img is None: return True
    if np.mean(img) < thresh_mean or np.std(img) < thresh_std: return True
    return False

def is_label_empty(txt_path):
    """ Returns True if label file doesn't exist or is empty. """
    if not os.path.exists(txt_path): return True
    with open(txt_path, 'r') as f:
        if not f.read().strip(): return True
    return False

def analyze_kitti_labels(img, txt_path, area_thresh):
    """ 
    Parses KITTI format. Flags high occlusion, truncation, or tiny areas. 
    Returns: (bool is_suspicious, numpy_array annotated_image)
    """
    if not os.path.exists(txt_path) or img is None:
        return False, None
        
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        
    img_review = None
    is_suspicious = False
    
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 8: # Basic KITTI length check
            try:
                truncated = float(parts[1])
                occluded = int(float(parts[2]))
                xmin, ymin, xmax, ymax = map(float, parts[4:8])
                
                area = (xmax - xmin) * (ymax - ymin)
                
                reasons = []
                if occluded in [2, 3]: reasons.append(f"Occ:{occluded}")
                if truncated > 0.6: reasons.append(f"Trunc:{truncated:.2f}")
                if area < area_thresh: reasons.append(f"Area:{area:.0f}")
                
                if reasons:
                    is_suspicious = True
                    if img_review is None:
                        img_review = img.copy()
                    
                    # Draw visual feedback
                    cv2.rectangle(img_review, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2)
                    reason_text = " | ".join(reasons)
                    cv2.putText(img_review, reason_text, (int(xmin), int(ymin) - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
            except ValueError:
                continue
                
    return is_suspicious, img_review

# ==========================================
# 4. ACTION & FILE SYSTEM MODULES
# ==========================================
def handle_action(rgb_path, camera_root, action, dry_run, stats, stat_key, frame_id, target_dir=None):
    """ Executes file deletion or moving, respecting the dry_run flag. """
    stats[stat_key] += 1
    
    if dry_run:
        print(f"  [DRY-RUN] Frame {frame_id} would be {action}d ({stat_key}).")
        return
        
    if action == "delete":
        delete_hierarchical_frame(rgb_path, camera_root)
    elif action == "move" and target_dir:
        move_hierarchical_frame(rgb_path, camera_root, target_dir)

def save_review_image(annotated_img, data_dir, camera_name, frame_id):
    """ Saves the flagged image into the review folder. """
    dest_folder = os.path.join(data_dir, "_review_occlusions", camera_name)
    os.makedirs(dest_folder, exist_ok=True)
    cv2.imwrite(os.path.join(dest_folder, f"{frame_id}.png"), annotated_img)

def delete_hierarchical_frame(rgb_path, camera_root):
    frame_id = Path(rgb_path).stem
    for folder in [f.path for f in os.scandir(camera_root) if f.is_dir()]:
        for f_path in glob.glob(os.path.join(folder, f"{frame_id}.*")):
            try: os.remove(f_path)
            except OSError as e: print(f"  [ERROR] Could not delete {f_path}: {e}")

def move_hierarchical_frame(rgb_path, camera_root, target_root_dir):
    frame_id = Path(rgb_path).stem
    camera_name = os.path.basename(camera_root)
    dest_camera_root = os.path.join(target_root_dir, camera_name)
    
    for folder in [f.path for f in os.scandir(camera_root) if f.is_dir()]:
        dest_folder = os.path.join(dest_camera_root, os.path.basename(folder))
        os.makedirs(dest_folder, exist_ok=True)
        
        for f_path in glob.glob(os.path.join(folder, f"{frame_id}.*")):
            try: shutil.move(f_path, os.path.join(dest_folder, os.path.basename(f_path)))
            except OSError as e: print(f"  [ERROR] Could not move {f_path}: {e}")

# ==========================================
# 5. METADATA & UI MODULES
# ==========================================
def print_summary(stats, args):
    print("\n--- FINAL SUMMARY ---")
    print(f"✅ Valid frames kept: {stats['kept']}")
    print(f"🗑️ Corrupted frames deleted: {stats['corrupted']}")
    if args.empty or args.move_empty or stats['empty'] > 0:
        print(f"📦 Empty frames handled: {stats['empty']}")
    if args.review:
        print(f"🔍 Frames flagged for manual review: {stats['reviewed']}")
    if args.dry:
        print("⚠️  DRY-RUN: No files were actually moved or deleted.")

def update_metadata(data_dir, stats, args):
    metadata_path = os.path.join(data_dir, "generation_metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        
        meta["cleaning"] = {
            "corrupted_deleted": stats["corrupted"],
            "empty_deleted": stats["empty"],
            "valid_kept": stats["kept"],
            "empty_were_moved": args.move_empty,
            "frames_flagged_for_review": stats["reviewed"] if args.review else 0
        }
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=4)
        print(f"📝 Metadata updated in: {metadata_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset cleaner for Isaac Sim datasets")
    parser.add_argument("--dir", type=str, default="_output_data", help="Root directory of the dataset")
    parser.add_argument("--dry", action="store_true", help="Dry run, do not delete files")
    parser.add_argument("--thresh_mean", type=float, default=5.0, help="Darkness threshold (0-255)")
    parser.add_argument("--empty", action="store_true", help="Delete pure backgrounds")
    parser.add_argument("--move_empty", action="store_true", help="Move empty backgrounds instead of deleting")
    parser.add_argument("--review", action="store_true", help="Generate a visual review of occluded/truncated bboxes")
    parser.add_argument("--area_thresh", type=float, default=1000.0, help="Min area (px) for bounding boxes")
    
    args = parser.parse_args()

    if os.path.exists(args.dir):
        clean_dataset(args)
    else:
        print(f"The directory {args.dir} does not exist.")