import os
import cv2
import shutil
import random
import argparse
import json
import numpy as np
from datetime import datetime

# --- CONFIG ---

# Input paths (where are your NEW data from Isaac Sim)
LABELS_KITTI = ""
IMAGES_DIR = ""

# Output path (where the dataset will be stored)
BASE_OUTPUT = os.path.join(os.getcwd(), "dataset_yolo_output")

# Classes to detect
with open("classes.txt", "r") as f:
    CLASES = [line.strip() for line in f.readlines() if line.strip()]

# Split ratios
TRAIN_RATIO = 0.7
VAL_RATIO   = 0.2
TEST_RATIO  = 0.1

def change_coordinates(size, box):
    """ Converts from (Xmin, Xmax, Ymin, Ymax) to YOLO (CenterX, CenterY, W, H) normalized """
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    return (x * dw, y * dh, w * dw, h * dh)

def get_dir_size(path):
    """ Calculates the total size of a directory in bytes """
    total = 0
    if os.path.exists(path):
        with os.scandir(path) as it:
            for entry in it:
                if entry.is_file():
                    total += entry.stat().st_size
                elif entry.is_dir():
                    total += get_dir_size(entry.path)
    return total

def create_dir_structure(append_mode):
    """ 
    Creates the folder structure.
    If append_mode is False, erases what was there to start from scratch.
    """
    if os.path.exists(BASE_OUTPUT) and not append_mode:
        print(f"🧹 Reset mode: Deleting previous dataset in {BASE_OUTPUT}...")
        shutil.rmtree(BASE_OUTPUT)
    
    subsets = ['train', 'val', 'test']
    for subset in subsets:
        os.makedirs(os.path.join(BASE_OUTPUT, 'images', subset), exist_ok=True)
        os.makedirs(os.path.join(BASE_OUTPUT, 'labels', subset), exist_ok=True)
    
    if not append_mode:
        print(f"📂 Structure created clean in: {BASE_OUTPUT}")
    else:
        print(f"📂 Structure verified (Append mode).")

def process_pair(filename_base, subset_name, unique_prefix, move_mode=False, is_yolo=False, override_class=-1):
    """
    Processes a pair of image/label, changes their name with a unique prefix
    and saves them in the corresponding subset.
    """
    
    # 1. Localize image
    img_path = None
    img_ext = None
    for ext in [".png", ".jpg", ".jpeg"]:
        temp_path = os.path.join(IMAGES_DIR, filename_base + ext)
        if os.path.exists(temp_path):
            img_path = temp_path
            img_ext = ext
            break
    
    if img_path is None:
        return False, 0, []

    # Read dimensions
    img = cv2.imread(img_path)
    if img is None:
        return False, 0, []
    height, width, _ = img.shape

    # 2. Process label
    kitti_path = os.path.join(LABELS_KITTI, filename_base + ".txt")
    yolo_lines = []
    bboxes_stats = []
    
    if not os.path.exists(kitti_path):
        return False, 0, []

    with open(kitti_path, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        parts = line.strip().split(' ')
        if len(parts) < 5: 
            continue
            
        if is_yolo:
            try:
                # YOLO format: class_id cx cy w h
                class_id = int(parts[0])
                cx, cy, w_box, h_box = map(float, parts[1:5])

                if override_class >= 0:
                    class_id = override_class
                
                # Check if the class_id is valid according to our classes.txt
                if class_id < 0 or class_id >= len(CLASES):
                    continue
                    
                yolo_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w_box:.6f} {h_box:.6f}")
                
                area = w_box * h_box
                aspect_ratio = w_box / h_box if h_box > 0 else 0
                
                bboxes_stats.append({
                    "area": area, "ar": aspect_ratio, "cx": cx, "cy": cy
                })
            except (ValueError, IndexError):
                continue
        else:
            # KITTI original format
            class_name_raw = parts[0]
            class_name = None
            
            # Clean multiple labels (Ej: "bicycle,pedal" -> "pedal")
            if ',' in class_name_raw:
                for c in reversed(class_name_raw.split(',')):
                    if c in CLASES:
                        class_name = c
                        break
            else:
                class_name = class_name_raw
                
            # If we don't find the class in classes.txt after cleaning, we ignore it
            if class_name is None or class_name not in CLASES:
                continue
                
            if override_class >= 0:
                class_id = override_class
            else:
                class_id = CLASES.index(class_name)
            try:
                xmin, ymin = float(parts[4]), float(parts[5])
                xmax, ymax = float(parts[6]), float(parts[7])
                
                bbox = change_coordinates((width, height), (xmin, xmax, ymin, ymax))
                yolo_lines.append(f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}")

                w_k, h_k = bbox[2], bbox[3]
                area = w_k * h_k
                aspect_ratio = w_k / h_k if h_k > 0 else 0
                
                bboxes_stats.append({
                    "area": area, "ar": aspect_ratio, "cx": bbox[0], "cy": bbox[1]
                })
            except (ValueError, IndexError):
                continue

    # 3. Save with NEW UNIQUE NAME
    new_filename = f"{unique_prefix}_{filename_base}"
    
    dest_img = os.path.join(BASE_OUTPUT, 'images', subset_name, new_filename + img_ext)
    dest_lbl = os.path.join(BASE_OUTPUT, 'labels', subset_name, new_filename + ".txt")
    
    if move_mode:
        shutil.move(img_path, dest_img)
        if os.path.exists(kitti_path):
            os.remove(kitti_path)
    else:
        shutil.copy2(img_path, dest_img)
    
    # Save new txt
    with open(dest_lbl, 'w') as f_out:
        if yolo_lines:
            f_out.write('\n'.join(yolo_lines))
            
    return True, len(yolo_lines), bboxes_stats

def process_subset(file_list, subset_name, batch_prefix, move_mode=False, is_yolo=False, override_class=-1):
    count_imgs = 0
    count_objs = 0
    count_bgs = 0
    
    # Lists for EDA
    all_areas, all_ars, all_cx, all_cy = [], [], [], []
    
    for fname in file_list:
        success, num_objects, bbox_stats = process_pair(fname, subset_name, batch_prefix, move_mode, is_yolo, override_class)
        if success:
            count_imgs += 1
            count_objs += num_objects
            if num_objects == 0:
                count_bgs += 1
            
            # Get metrics from each object
            for stat in bbox_stats:
                all_areas.append(stat["area"])
                all_ars.append(stat["ar"])
                all_cx.append(stat["cx"])
                all_cy.append(stat["cy"])
    
    def get_stats(arr):
        if not arr: return {"mean": 0.0, "std": 0.0}
        return {
            "mean": round(float(np.mean(arr)), 4),
            "std": round(float(np.std(arr)), 4)
        }
        
    eda_stats = {
        "bbox_area": get_stats(all_areas),
        "aspect_ratio": get_stats(all_ars),
        "center_x": get_stats(all_cx),
        "center_y": get_stats(all_cy)
    }
                
    return {
        "images": count_imgs, 
        "objects": count_objs, 
        "backgrounds": count_bgs,
        "eda": eda_stats
    }

def main():
    parser = argparse.ArgumentParser(description="Dataset manager from KITTI to YOLO")
    parser.add_argument('--append', action='store_true', help="Add new data to the existing dataset without deleting anything")
    parser.add_argument('--move', action='store_true', help="Move files instead of copying to save disk space (DELETES ORIGINALS)")
    parser.add_argument('--limit', type=int, default=0, help="Maximum number of images to process (0 = all)")
    parser.add_argument('--source', type=str, default="_output_data", help="Source folder name relative to parent directory")
    parser.add_argument('--is_yolo', action='store_true', help="Indicates that source labels are already in YOLO format")
    parser.add_argument('--override_class', type=int, default=-1, help="Override the class ID for all imported labels (Ej: --override_class 1)")
    args = parser.parse_args()

    global LABELS_KITTI, IMAGES_DIR
    LABELS_KITTI = os.path.join(os.getcwd(), "..", args.source, "DroneCamera", "object_detection")
    IMAGES_DIR = os.path.join(os.getcwd(), "..", args.source, "DroneCamera", "rgb")

    if not os.path.exists(LABELS_KITTI) or not os.path.exists(IMAGES_DIR):
        print(f"❌ Error: Verify the input paths ({LABELS_KITTI})")
        return

    # 1. Prepare structure
    create_dir_structure(args.append)

    # 2. Generate unique prefix for this batch of data
    # We use date and time until seconds to ensure uniqueness: "20231027_153022"
    batch_prefix = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"🆔 ID of Batch (Batch ID): {batch_prefix}")

    if args.move:
        print("⚠️  WARNING: Flag '--move' active. Original files in _output_data will be DELETED to save space.")

    # 3. List files
    all_files = [os.path.splitext(f)[0] for f in os.listdir(LABELS_KITTI) if f.endswith(".txt")]
    total_files = len(all_files)
    
    if total_files == 0:
        print("⚠️ No files found to process.")
        return

    # 4. Shuffle and divide
    random.shuffle(all_files)

    if args.limit > 0:
        limit = min(args.limit, total_files)
        all_files = all_files[:limit]
        total_files = len(all_files)
        print(f"✂️  LIMIT ACTIVE: Processing reduced to {total_files} random images.")

    train_end = int(total_files * TRAIN_RATIO)
    val_end = train_end + int(total_files * VAL_RATIO)
    
    train_files = all_files[:train_end]
    val_files = all_files[train_end:val_end]
    test_files = all_files[val_end:]

    print(f"📊 New files found: {total_files}")
    print(f"   Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(test_files)}")
    
    if args.append:
        print("   -> New data will be added to the existing dataset.")
    else:
        print("   -> A new dataset will be created (deleting the previous one).")
    
    print("-" * 40)

    # 5. Process passing the prefix
    print("🚀 Processing Train...")
    train_stats = process_subset(train_files, 'train', batch_prefix, args.move, args.is_yolo, args.override_class)
    
    print("🚀 Processing Val...")
    val_stats = process_subset(val_files, 'val', batch_prefix, args.move, args.is_yolo, args.override_class)
    
    print("🚀 Processing Test...")
    test_stats = process_subset(test_files, 'test', batch_prefix, args.move, args.is_yolo, args.override_class)

    print("-" * 40)
    print("✅ PROCESSING COMPLETED")
    total_added_imgs = train_stats["images"] + val_stats["images"] + test_stats["images"]
    print(f"New files added: {total_added_imgs}")
    print(f"Dataset located in: {BASE_OUTPUT}")

    # 6. Process JSON
    source_meta_path = os.path.join(os.getcwd(), "..", args.source, "generation_metadata.json")
    batch_meta = {"batch_id": batch_prefix} # Fallback if it doesn't exist
    
    if os.path.exists(source_meta_path):
        with open(source_meta_path, 'r', encoding='utf-8') as f:
            batch_meta.update(json.load(f))
            
    # Add the split data to the batch
    batch_meta["yolo_split"] = {
        "train": train_stats,
        "val": val_stats,
        "test": test_stats,
        "total_added": total_added_imgs
    }

    master_meta_path = os.path.join(BASE_OUTPUT, "dataset_metadata.json")
    master_meta = {
        "global_totals": {
            "train": {"images": 0, "objects": 0, "backgrounds": 0},
            "val": {"images": 0, "objects": 0, "backgrounds": 0},
            "test": {"images": 0, "objects": 0, "backgrounds": 0},
            "total_images": 0,
            "total_objects": 0,
            "total_backgrounds": 0
        },
        "sessions": []
    }

    # If append mode
    if args.append and os.path.exists(master_meta_path):
        with open(master_meta_path, 'r', encoding='utf-8') as f:
            loaded_meta = json.load(f)
            if isinstance(loaded_meta.get("global_totals", {}).get("train"), dict):
                master_meta = loaded_meta

    master_meta["sessions"].append(batch_meta)
    
    for split, stats in [("train", train_stats), ("val", val_stats), ("test", test_stats)]:
        master_meta["global_totals"][split]["images"] += stats["images"]
        master_meta["global_totals"][split]["objects"] += stats["objects"]
        master_meta["global_totals"][split]["backgrounds"] += stats["backgrounds"]
        
        master_meta["global_totals"]["total_images"] += stats["images"]
        master_meta["global_totals"]["total_objects"] += stats["objects"]
        master_meta["global_totals"]["total_backgrounds"] += stats["backgrounds"]

    images_dir = os.path.join(BASE_OUTPUT, 'images')
    total_size_bytes = get_dir_size(images_dir)
    total_size_mb = round(total_size_bytes / (1024 * 1024), 2)
    
    total_imgs = master_meta["global_totals"]["total_images"]
    avg_img_mb = round(total_size_mb / total_imgs, 2) if total_imgs > 0 else 0
    
    master_meta["global_totals"]["size_mb"] = total_size_mb
    master_meta["global_totals"]["avg_image_mb"] = avg_img_mb

    # Save Master Metadata
    with open(master_meta_path, 'w', encoding='utf-8') as f:
        json.dump(master_meta, f, indent=4)
        
    print(f"🧠 Master Metadata saved/updated in: {master_meta_path}")


if __name__ == "__main__":
    main()