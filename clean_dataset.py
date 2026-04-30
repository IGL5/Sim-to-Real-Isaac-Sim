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
    if args.clean_labels or args.review:
        whitelist_data = load_whitelist(args.whitelist)
        print(f"🛡️  Loaded whitelist with {len(whitelist_data)} frame rules.")

    if args.review:
        review_dir = os.path.join(args.dir, "_review_occlusions")
        if os.path.exists(review_dir):
            shutil.rmtree(review_dir)
        os.makedirs(review_dir, exist_ok=True)
        print("🧹 Cleared previous review folder.")

    print(f"--- Starting processing in {len(rgb_folders)} camera folders ---")
    
    stats = {
        "corrupted": 0, "empty": 0, "kept": 0, "reviewed": 0, 
        "labels_removed": 0, "labels_saved": 0, "total_labels": 0,
        "reason_occ": 0, "reason_trunc": 0, "reason_area": 0,
        "reason_edge": 0, "reason_giant": 0
    }
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
        img_h, img_w = img.shape[:2]
        res = apply_label_cleaning(txt_path, frame_id, args.area_thresh, args.max_area_ratio, whitelist_data, args.dry, img_w, img_h)
        stats["labels_removed"] += res["removed"]
        stats["labels_saved"] += res["saved"]
        stats["total_labels"] += res["total"]
        stats["reason_occ"] += res["occ"]
        stats["reason_trunc"] += res["trunc"]
        stats["reason_area"] += res["area"]
        stats["reason_edge"] += res["edge"]
        stats["reason_giant"] += res["giant"]

        if args.dry and res["removed"] > 0:
            print(f"  [DRY-RUN] Frame {frame_id}: {res['removed']} bad bboxes would be removed.")

    # CHECK 3: Empty Labels (Backgrounds) - Evaluated AFTER cleaning labels!
    if remove_empty_logic and is_label_empty(txt_path):
        action = "move" if args.move_empty else "delete"
        target_dir = f"{args.dir}_empty" if args.move_empty else None
        handle_action(rgb_path, camera_root, action, args.dry, stats, "empty", frame_id, target_dir)
        return

    # CHECK 4: Review Mode (Suspicious labels visualization)
    if args.review:
        is_suspicious, annotated_img = analyze_kitti_labels(img, txt_path, args.area_thresh, args.max_area_ratio, whitelist_data, frame_id)
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

def analyze_kitti_labels(img, txt_path, area_thresh, max_area_ratio, whitelist_data, frame_id):
    """ Parses KITTI. Flags suspicious boxes and draws their LINE INDEX for whitelisting. """
    if not os.path.exists(txt_path) or img is None:
        return False, None
        
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        
    img_review = None
    is_suspicious_overall = False
    img_h, img_w = img.shape[:2]
    
    # Extract whitelist for this specific frame
    frame_whitelist = whitelist_data.get(frame_id, [])
    
    for idx, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) >= 8:
            try:
                truncated = float(parts[1])
                occluded = int(float(parts[2]))
                xmin, ymin, xmax, ymax = map(float, parts[4:8])
                
                bbox_w = xmax - xmin
                bbox_h = ymax - ymin
                area = bbox_w * bbox_h
                img_area = img_w * img_h
                aspect_ratio = bbox_w / bbox_h if bbox_h > 0 else 0
                
                margin = 2
                touches_edge = (xmin <= margin) or (ymin <= margin) or (xmax >= img_w - margin) or (ymax >= img_h - margin)
                
                reasons = []
                if occluded in [2, 3]: reasons.append(f"Occ:{occluded}")
                if truncated > 0.6: reasons.append(f"Trunc:{truncated:.2f}")
                if area < area_thresh: reasons.append(f"Area:{area:.0f}")
                if touches_edge and (aspect_ratio > 3.0 or aspect_ratio < 0.33): 
                    reasons.append(f"EdgeAR:{aspect_ratio:.1f}")
                if (area / img_area) > max_area_ratio:  
                    reasons.append(f"Giant:{(area/img_area)*100:.0f}%")
                
                if reasons:
                    # If the user has already saved it, we ignore it visually
                    if "ALL" in frame_whitelist or idx in frame_whitelist:
                        continue 
                        
                    is_suspicious_overall = True
                    if img_review is None: img_review = img.copy()
                    
                    cv2.rectangle(img_review, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2)
                    reason_text = f"[#{idx}] " + " | ".join(reasons)
                    cv2.putText(img_review, reason_text, (int(xmin), int(ymin) - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
            except ValueError:
                continue
                
    return is_suspicious_overall, img_review

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

def apply_label_cleaning(txt_path, frame_id, area_thresh, max_area_ratio, whitelist_data, dry_run, img_w, img_h):
    """ Reads the .txt and overwrites it omitting suspicious lines (unless whitelisted). """
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        
    keep_lines = []
    removed_count, saved_count, total_labels = 0, 0, 0
    r_occ, r_trunc, r_area, r_edge, r_giant = 0, 0, 0, 0, 0
    
    frame_whitelist = whitelist_data.get(frame_id, [])
    
    for idx, line in enumerate(lines):
        parts = line.strip().split()
        is_suspicious = False
        primary_reason = None
        
        if len(parts) >= 8:
            total_labels += 1
            try:
                truncated, occluded = float(parts[1]), int(float(parts[2]))
                xmin, ymin, xmax, ymax = map(float, parts[4:8])
                
                bbox_w = xmax - xmin
                bbox_h = ymax - ymin
                area = bbox_w * bbox_h
                aspect_ratio = bbox_w / bbox_h if bbox_h > 0 else 0
                touches_edge = (xmin <= 2) or (ymin <= 2) or (xmax >= img_w - 2) or (ymax >= img_h - 2)
                
                # Priority of reasons
                if occluded in [2, 3]:
                    is_suspicious, primary_reason = True, "occ"
                elif (area / (img_w * img_h)) > max_area_ratio:
                    is_suspicious, primary_reason = True, "giant"
                elif touches_edge and (aspect_ratio > 3.0 or aspect_ratio < 0.33):
                    is_suspicious, primary_reason = True, "edge"
                elif truncated > 0.6:
                    is_suspicious, primary_reason = True, "trunc"
                elif area < area_thresh:
                    is_suspicious, primary_reason = True, "area"
            except ValueError:
                pass
        
        # Check whitelist
        if is_suspicious:
            if "ALL" in frame_whitelist or idx in frame_whitelist:
                is_suspicious = False
                saved_count += 1
                
        if is_suspicious:
            removed_count += 1
            if primary_reason == "occ": r_occ += 1
            elif primary_reason == "edge": r_edge += 1
            elif primary_reason == "trunc": r_trunc += 1
            elif primary_reason == "area": r_area += 1
        else:
            keep_lines.append(line)
            
    if removed_count > 0 and not dry_run:
        with open(txt_path, 'w') as f:
            f.writelines(keep_lines)
            
    return {
        "removed": removed_count, "saved": saved_count, "total": total_labels,
        "occ": r_occ, "trunc": r_trunc, "area": r_area, "edge": r_edge, "giant": r_giant
    }

# ==========================================
# 4. ACTION & FILE SYSTEM MODULES
# ==========================================
def handle_action(rgb_path, camera_root, action, dry_run, stats, stat_key, frame_id, target_dir=None):
    stats[stat_key] += 1
    if dry_run: 
        action_str = "moved" if action == "move" else "deleted"
        print(f"  [DRY-RUN] Frame {frame_id} would be {action_str} (Reason: {stat_key.upper()})")
        return
        
    if action == "delete": 
        delete_hierarchical_frame(rgb_path, camera_root)
    elif action == "move" and target_dir: 
        move_hierarchical_frame(rgb_path, camera_root, target_dir)

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
            print("  3. Add it to '_output_data/whitelist.txt'. Format: 'frame_id:1' (saves line 1) or 'frame_id' (saves all).")
            print("  4. Run the script again to apply changes: python clean_dataset.py --clean_labels")

    if args.dry:
        print("⚠️  DRY-RUN: No files were actually modified.")

def update_metadata(data_dir, stats, args):
    metadata_path = os.path.join(data_dir, "generation_metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        
        # Reference to the cleaning block
        cleaning_meta = meta.setdefault("cleaning", {})
        
        # 1. ACCUMULATE the deleted ones
        cleaning_meta["corrupted_deleted"] = cleaning_meta.get("corrupted_deleted", 0) + stats["corrupted"]
        cleaning_meta["empty_deleted"] = cleaning_meta.get("empty_deleted", 0) + stats["empty"]
        
        # 2. OVERWRITE valid_kept
        cleaning_meta["valid_kept"] = stats["kept"]
        
        # Save if move_empty has ever been used
        if args.move_empty:
            cleaning_meta["empty_were_moved"] = True
            
        # 3. ONLY update the label metrics if we are in --clean_labels mode
        if args.clean_labels:
            prev_labels = cleaning_meta.get("labels_stats", {})
            prev_reasons = prev_labels.get("removal_reasons", {})
            
            # A) Accumulate the deleted ones and the reasons (because they physically disappear from the .txt)
            accumulated_removed = prev_labels.get("labels_removed", 0) + stats["labels_removed"]
            
            # B) Calculate the TOTAL original historical (The ones read today + the ones already deleted yesterday)
            real_total_labels = stats["total_labels"] + prev_labels.get("labels_removed", 0)
            
            # C) Calculate the percentage on the real historical total
            perc_removed = 0.0
            if real_total_labels > 0:
                perc_removed = round((accumulated_removed / real_total_labels) * 100, 2)
            
            cleaning_meta["labels_stats"] = {
                "total_labels_found": real_total_labels,
                "labels_removed": accumulated_removed,
                # D) The saved ones are NOT accumulated
                "labels_saved_by_whitelist": stats["labels_saved"],
                "percentage_removed_percent": perc_removed,
                "removal_reasons": {
                    "occlusion": prev_reasons.get("occlusion", 0) + stats["reason_occ"],
                    "truncation": prev_reasons.get("truncation", 0) + stats["reason_trunc"],
                    "small_area": prev_reasons.get("small_area", 0) + stats["reason_area"],
                    "edge_cut": prev_reasons.get("edge_cut", 0) + stats["reason_edge"],
                    "giant_bbox": prev_reasons.get("giant_bbox", 0) + stats["reason_giant"]
                }
            }
        
        # Update the total number of valid frames generated in 'performance' if it exists
        if "performance" in meta:
            meta["performance"]["total_valid_frames_after_cleaning"] = stats["kept"]
        
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
    parser.add_argument("--whitelist", type=str, default=None, help="Path to whitelist file")
    parser.add_argument("--area_thresh", type=float, default=1000.0, help="Min area for bboxes")
    parser.add_argument("--max_area_ratio", type=float, default=0.50, help="Max percentage of the image a bbox can occupy (0.0 to 1.0)")

    args = parser.parse_args()

    if args.whitelist is None:
        args.whitelist = os.path.join(args.dir, "whitelist.txt")

    if os.path.exists(args.dir):
        clean_dataset(args)
    else:
        print(f"The directory {args.dir} does not exist.")