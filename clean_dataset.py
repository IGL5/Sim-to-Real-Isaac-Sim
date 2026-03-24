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

    # Load whitelist if cleaning labels
    whitelist_data = {}
    if args.clean_labels:
        whitelist_data = load_whitelist(args.whitelist)
        print(f"🛡️  Loaded whitelist with {len(whitelist_data)} frame rules.")

    print(f"--- Starting processing in {len(rgb_folders)} camera folders ---")
    
    stats = {"corrupted": 0, "empty": 0, "kept": 0, "reviewed": 0, "labels_removed": 0}
    remove_empty_logic = args.empty or args.move_empty

    for rgb_folder in rgb_folders:
        camera_root = os.path.dirname(rgb_folder) 
        camera_name = os.path.basename(camera_root)
        
        print(f"\nProcessing camera: {camera_name}")
        png_files = sorted(glob.glob(os.path.join(rgb_folder, "*.png")))
        
        if not png_files: continue
        print(f"  -> Analyzing {len(png_files)} images...")

        for rgb_path in png_files:
            process_frame(rgb_path, camera_root, camera_name, args, stats, remove_empty_logic, whitelist_data)

    print_summary(stats, args)
    if not args.dry:
        update_metadata(args.dir, stats, args)

# ==========================================
# 2. FRAME PROCESSING FLOW
# ==========================================
def process_frame(rgb_path, camera_root, camera_name, args, stats, remove_empty_logic, whitelist_data):
    """ Applies the pipeline of checks to a single frame. """
    frame_id = Path(rgb_path).stem
    txt_path = os.path.join(camera_root, "object_detection", f"{frame_id}.txt")
    
    img = cv2.imread(rgb_path)

    # CHECK 1: Corrupted or Flat Image
    if is_image_corrupted(img, args.thresh_mean, args.thresh_std):
        handle_action(rgb_path, camera_root, "delete", args.dry, stats, "corrupted", frame_id)
        return

    # CHECK 2: Clean Labels (Removes bad bboxes directly from the .txt)
    if args.clean_labels and os.path.exists(txt_path):
        removed_count = apply_label_cleaning(txt_path, frame_id, args.area_thresh, whitelist_data, args.dry)
        stats["labels_removed"] += removed_count

    # CHECK 3: Empty Labels (Backgrounds) - Evaluated AFTER cleaning labels!
    if remove_empty_logic and is_label_empty(txt_path):
        action = "move" if args.move_empty else "delete"
        target_dir = f"{args.dir}_empty" if args.move_empty else None
        handle_action(rgb_path, camera_root, action, args.dry, stats, "empty", frame_id, target_dir)
        return

    # CHECK 4: Review Mode (Suspicious labels visualization)
    if args.review:
        is_suspicious, annotated_img = analyze_kitti_labels(img, txt_path, args.area_thresh)
        if is_suspicious:
            save_review_image(annotated_img, args.dir, camera_name, frame_id)
            stats["reviewed"] += 1

    stats["kept"] += 1

# ==========================================
# 3. ANALYSIS & CLEANING MODULES
# ==========================================
def is_image_corrupted(img, thresh_mean, thresh_std):
    if img is None: return True
    if np.mean(img) < thresh_mean or np.std(img) < thresh_std: return True
    return False

def is_label_empty(txt_path):
    if not os.path.exists(txt_path): return True
    with open(txt_path, 'r') as f:
        if not f.read().strip(): return True
    return False

def analyze_kitti_labels(img, txt_path, area_thresh):
    """ Parses KITTI. Flags suspicious boxes and draws their LINE INDEX for whitelisting. """
    if not os.path.exists(txt_path) or img is None:
        return False, None
        
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        
    img_review = None
    is_suspicious = False
    
    for idx, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) >= 8:
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
                    if img_review is None: img_review = img.copy()
                    
                    cv2.rectangle(img_review, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2)
                    # Show the line index [#idx] so the user can whitelist it
                    reason_text = f"[#{idx}] " + " | ".join(reasons)
                    cv2.putText(img_review, reason_text, (int(xmin), int(ymin) - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
            except ValueError:
                continue
                
    return is_suspicious, img_review

def load_whitelist(whitelist_path):
    """ Loads whitelist file. Format: 'frame_id:line_index' or 'frame_id' to save all lines. """
    whitelist_data = {}
    if not os.path.exists(whitelist_path):
        return whitelist_data
        
    with open(whitelist_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): continue
            
            if ":" in line:
                frame_id, idx_str = line.split(":")
                whitelist_data.setdefault(frame_id, []).append(int(idx_str))
            else:
                whitelist_data.setdefault(line, []).append("ALL")
    return whitelist_data

def apply_label_cleaning(txt_path, frame_id, area_thresh, whitelist_data, dry_run):
    """ Reads the .txt and overwrites it omitting suspicious lines (unless whitelisted). """
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        
    keep_lines = []
    removed_count = 0
    
    frame_whitelist = whitelist_data.get(frame_id, [])
    
    for idx, line in enumerate(lines):
        parts = line.strip().split()
        is_suspicious = False
        
        if len(parts) >= 8:
            try:
                truncated, occluded = float(parts[1]), int(float(parts[2]))
                xmin, ymin, xmax, ymax = map(float, parts[4:8])
                area = (xmax - xmin) * (ymax - ymin)
                
                if occluded in [2, 3] or truncated > 0.6 or area < area_thresh:
                    is_suspicious = True
            except ValueError:
                pass
        
        # Check whitelist
        if is_suspicious:
            if "ALL" in frame_whitelist or idx in frame_whitelist:
                is_suspicious = False # Saved by the user!
                
        if is_suspicious:
            removed_count += 1
        else:
            keep_lines.append(line)
            
    if removed_count > 0 and not dry_run:
        with open(txt_path, 'w') as f:
            f.writelines(keep_lines)
            
    return removed_count

# ==========================================
# 4. ACTION & FILE SYSTEM MODULES
# ==========================================
def handle_action(rgb_path, camera_root, action, dry_run, stats, stat_key, frame_id, target_dir=None):
    stats[stat_key] += 1
    if dry_run: return
    if action == "delete": delete_hierarchical_frame(rgb_path, camera_root)
    elif action == "move" and target_dir: move_hierarchical_frame(rgb_path, camera_root, target_dir)

def save_review_image(annotated_img, data_dir, camera_name, frame_id):
    dest_folder = os.path.join(data_dir, "_review_occlusions", camera_name)
    os.makedirs(dest_folder, exist_ok=True)
    cv2.imwrite(os.path.join(dest_folder, f"{frame_id}.png"), annotated_img)

def delete_hierarchical_frame(rgb_path, camera_root):
    frame_id = Path(rgb_path).stem
    for folder in [f.path for f in os.scandir(camera_root) if f.is_dir()]:
        for f_path in glob.glob(os.path.join(folder, f"{frame_id}.*")):
            try: os.remove(f_path)
            except OSError: pass

def move_hierarchical_frame(rgb_path, camera_root, target_root_dir):
    frame_id = Path(rgb_path).stem
    camera_name = os.path.basename(camera_root)
    dest_camera_root = os.path.join(target_root_dir, camera_name)
    
    for folder in [f.path for f in os.scandir(camera_root) if f.is_dir()]:
        dest_folder = os.path.join(dest_camera_root, os.path.basename(folder))
        os.makedirs(dest_folder, exist_ok=True)
        for f_path in glob.glob(os.path.join(folder, f"{frame_id}.*")):
            try: shutil.move(f_path, os.path.join(dest_folder, os.path.basename(f_path)))
            except OSError: pass

# ==========================================
# 5. METADATA & UI MODULES
# ==========================================
def print_summary(stats, args):
    print("\n--- FINAL SUMMARY ---")
    print(f"✅ Valid frames kept: {stats['kept']}")
    print(f"🗑️ Corrupted frames deleted: {stats['corrupted']}")
    if args.empty or args.move_empty or stats['empty'] > 0:
        print(f"📦 Empty frames handled: {stats['empty']}")
    if args.clean_labels:
        print(f"✂️  Bad bounding boxes removed from .txt files: {stats['labels_removed']}")
    if args.review:
        print(f"🔍 Frames flagged for manual review: {stats['reviewed']}")

        if stats['reviewed'] > 0:
            print("\n💡 NEXT STEPS (Whitelist Guide):")
            print("  1. Open the '_review_occlusions' folder and check the images.")
            print("  2. If a bounding box is valid, note its frame ID and line number (e.g., [#1]).")
            print("  3. Add it to 'whitelist.txt'. Format: 'frame_id:1' (saves line 1) or 'frame_id' (saves all).")
            print("  4. Run the script again to apply changes: python clean_dataset.py --clean_labels")

    if args.dry:
        print("⚠️  DRY-RUN: No files were actually modified.")

def update_metadata(data_dir, stats, args):
    metadata_path = os.path.join(data_dir, "generation_metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        
        meta.setdefault("cleaning", {}).update({
            "corrupted_deleted": stats["corrupted"],
            "empty_deleted": stats["empty"],
            "valid_kept": stats["kept"],
            "bad_bboxes_removed": stats["labels_removed"]
        })
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset cleaner for Isaac Sim datasets")
    parser.add_argument("--dir", type=str, default="_output_data", help="Root directory")
    parser.add_argument("--dry", action="store_true", help="Dry run, no modifications")
    parser.add_argument("--thresh_mean", type=float, default=5.0, help="Darkness threshold")
    parser.add_argument("--thresh_std", type=float, default=5.0, help="Flatness/std threshold")
    parser.add_argument("--empty", action="store_true", help="Delete pure backgrounds")
    parser.add_argument("--move_empty", action="store_true", help="Move empty backgrounds")
    parser.add_argument("--review", action="store_true", help="Generate review images of bad bboxes")
    parser.add_argument("--clean_labels", action="store_true", help="Actually remove bad bboxes from .txt")
    parser.add_argument("--whitelist", type=str, default="whitelist.txt", help="Path to whitelist file")
    parser.add_argument("--area_thresh", type=float, default=1000.0, help="Min area for bboxes")
    
    args = parser.parse_args()

    if os.path.exists(args.dir):
        clean_dataset(args)
    else:
        print(f"The directory {args.dir} does not exist.")