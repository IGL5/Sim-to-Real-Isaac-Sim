from pathlib import Path
import numpy as np
import cv2
import shutil
import random
import argparse
import sys
from datetime import datetime

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass
import src.core.config as config
from src.core.utils import math_utils as mu
from src.core.utils import project_utils as pu
from src.core.metadata.dataset_builder import DatasetMetadata


# Classes to detect
CLASES = pu.get_project_classes()
if not CLASES:
    print("   Asegúrate de ejecutar el simulador al menos una vez para generarlo.")
    exit(1)

def create_dir_structure(append_mode):
    """ 
    Creates the folder structure.
    If append_mode is False, erases what was there to start from scratch.
    """
    processed_dir = Path(config.PROCESSED_DATA_DIR)
    if processed_dir.exists() and not append_mode:
        print(f"🧹 Reset mode: Deleting previous dataset in {config.PROCESSED_DATA_DIR}...")
        shutil.rmtree(processed_dir)
    
    subsets = ['train', 'val', 'test']
    for subset in subsets:
        (processed_dir / 'images' / subset).mkdir(parents=True, exist_ok=True)
        (processed_dir / 'labels' / subset).mkdir(parents=True, exist_ok=True)
    
    if not append_mode:
        print(f"📂 Structure created clean in: {config.PROCESSED_DATA_DIR}")
    else:
        print(f"📂 Structure verified (Append mode).")

def process_pair(filename_base, subset_name, unique_prefix, raw_labels_path, raw_images_path, move_mode=False, is_yolo=False, is_segmentation=False, raw_seg_path=None, override_all=-1, override_map=None):
    """
    Processes a pair of image/label, changes their name with a unique prefix
    and saves them in the corresponding subset. Supports detection and segmentation formats.
    """
    if override_map is None:
        override_map = {}
    
    # 1. Localize image
    img_path = None
    img_ext = None
    for ext in config.VALID_IMAGE_EXTENSIONS:
        temp_path = raw_images_path / (filename_base + ext)
        if temp_path.exists():
            img_path = temp_path
            img_ext = ext
            break
    
    if img_path is None:
        return False, 0, []

    # Read dimensions
    img = cv2.imread(str(img_path))
    if img is None:
        return False, 0, []
    height, width, _ = img.shape

    # 2. Process label
    kitti_path = raw_labels_path / f"{filename_base}.txt"
    yolo_lines = []
    bboxes_stats = []

    # Load segmentation mask image if available for raw mode
    seg_mask = None
    seg_img_file = None
    if is_segmentation and not is_yolo and raw_seg_path is not None:
        temp_seg_file = raw_seg_path / f"{filename_base}.png"
        if temp_seg_file.exists():
            seg_img_file = temp_seg_file
            seg_mask = cv2.imread(str(seg_img_file), cv2.IMREAD_UNCHANGED)

    if kitti_path.exists():
        with open(kitti_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split(' ')
            if len(parts) < 5: 
                continue
                
            if is_yolo:
                try:
                    class_id = int(parts[0])

                    if override_all >= 0:
                        class_id = override_all
                    elif str(class_id) in override_map:
                        class_id = override_map[str(class_id)]
                    
                    if class_id < 0 or class_id >= len(CLASES):
                        continue

                    coords = list(map(float, parts[1:]))

                    if is_segmentation or len(coords) > 4:
                        if len(coords) >= 6 and len(coords) % 2 == 0:
                            poly_str = " ".join(f"{c:.6f}" for c in coords)
                            yolo_lines.append(f"{class_id} {poly_str}")
                            bboxes_stats.append(mu.polygon_to_bbox(coords))
                    else:
                        cx, cy, w_box, h_box = coords[:4]
                        yolo_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w_box:.6f} {h_box:.6f}")
                        area = w_box * h_box
                        aspect_ratio = w_box / h_box if h_box > 0 else 0
                        bboxes_stats.append({
                            "area": area, "ar": aspect_ratio, "cx": cx, "cy": cy
                        })
                except (ValueError, IndexError) as e:
                    print(f"⚠️ Error parsing YOLO line in {filename_base}: {e} -> {line.strip()}")
                    continue
            else:
                # KITTI original format
                class_name_raw = parts[0]
                class_name = None
                
                if ',' in class_name_raw:
                    for c in reversed(class_name_raw.split(',')):
                        if c in CLASES or c in override_map or override_all >= 0:
                            class_name = c
                            break
                else:
                    class_name = class_name_raw
                    
                if class_name is None:
                    continue
                    
                if override_all >= 0:
                    class_id = override_all
                elif class_name in override_map:
                    class_id = override_map[class_name]
                elif class_name in CLASES:
                    class_id = CLASES.index(class_name)
                else:
                    continue  
                    
                if class_id < 0 or class_id >= len(CLASES):
                    continue

                try:
                    xmin, ymin = float(parts[4]), float(parts[5])
                    xmax, ymax = float(parts[6]), float(parts[7])

                    if is_segmentation:
                        polygons_found = []
                        if seg_mask is not None:
                            x1_i, y1_i = max(0, int(xmin)), max(0, int(ymin))
                            x2_i, y2_i = min(width, int(xmax)), min(height, int(ymax))

                            if seg_mask.ndim == 3:
                                crop = seg_mask[y1_i:y2_i, x1_i:x2_i, :3]
                                non_zero_mask = np.any(crop > 0, axis=2)
                            else:
                                crop = seg_mask[y1_i:y2_i, x1_i:x2_i]
                                non_zero_mask = crop > 0

                            if np.any(non_zero_mask):
                                if seg_mask.ndim == 3:
                                    binary_mask = np.any(seg_mask[:, :, :3] > 0, axis=2)
                                else:
                                    binary_mask = seg_mask > 0
                                polygons_found = mu.mask_to_polygons(binary_mask, width, height)

                        if polygons_found:
                            for poly in polygons_found:
                                poly_str = " ".join(f"{pt:.6f}" for pt in poly)
                                yolo_lines.append(f"{class_id} {poly_str}")
                                bboxes_stats.append(mu.polygon_to_bbox(poly))
                        else:
                            rect_poly = [
                                max(0.0, min(1.0, xmin / width)), max(0.0, min(1.0, ymin / height)),
                                max(0.0, min(1.0, xmax / width)), max(0.0, min(1.0, ymin / height)),
                                max(0.0, min(1.0, xmax / width)), max(0.0, min(1.0, ymax / height)),
                                max(0.0, min(1.0, xmin / width)), max(0.0, min(1.0, ymax / height))
                            ]
                            poly_str = " ".join(f"{pt:.6f}" for pt in rect_poly)
                            yolo_lines.append(f"{class_id} {poly_str}")
                            bboxes_stats.append(mu.polygon_to_bbox(rect_poly))
                    else:
                        bbox = mu.corners_to_yolo(xmin, xmax, ymin, ymax, width, height)
                        yolo_lines.append(f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}")

                        w_k, h_k = bbox[2], bbox[3]
                        area = w_k * h_k
                        aspect_ratio = w_k / h_k if h_k > 0 else 0
                        bboxes_stats.append({"area": area, "ar": aspect_ratio, "cx": bbox[0], "cy": bbox[1]})
                except (ValueError, IndexError) as e:
                    print(f"⚠️ Error parsing KITTI line in {filename_base}: {e} -> {line.strip()}")
                    continue
    elif is_segmentation and seg_mask is not None:
        # Extract polygons directly from mask image when KITTI txt is not present
        class_id = override_all if override_all >= 0 else 0
        if seg_mask.ndim == 3:
            binary_mask = (np.sum(seg_mask[:, :, :3], axis=2) > 0).astype(np.uint8)
            polygons_found = mu.mask_to_polygons(binary_mask, width, height)
            for poly in polygons_found:
                poly_str = " ".join(f"{pt:.6f}" for pt in poly)
                yolo_lines.append(f"{class_id} {poly_str}")
                bboxes_stats.append(mu.polygon_to_bbox(poly))
        else:
            unique_ids = np.unique(seg_mask)
            unique_ids = unique_ids[unique_ids > 0]
            for inst_id in unique_ids:
                instance_mask = (seg_mask == inst_id)
                polygons_found = mu.mask_to_polygons(instance_mask, width, height)
                for poly in polygons_found:
                    poly_str = " ".join(f"{pt:.6f}" for pt in poly)
                    yolo_lines.append(f"{class_id} {poly_str}")
                    bboxes_stats.append(mu.polygon_to_bbox(poly))

    # 3. Save with NEW UNIQUE NAME
    new_filename = f"{unique_prefix}_{filename_base}"
    
    dest_img = Path(config.PROCESSED_DATA_DIR) / 'images' / subset_name / f"{new_filename}{img_ext}"
    dest_lbl = Path(config.PROCESSED_DATA_DIR) / 'labels' / subset_name / f"{new_filename}.txt"
    
    if move_mode:
        shutil.move(str(img_path), str(dest_img))
        if kitti_path.exists():
            kitti_path.unlink()
        if seg_img_file is not None and seg_img_file.exists():
            seg_img_file.unlink()
    else:
        shutil.copy2(str(img_path), str(dest_img))
    
    # Save new txt
    with open(dest_lbl, 'w') as f_out:
        if yolo_lines:
            f_out.write('\n'.join(yolo_lines))
            
    return True, len(yolo_lines), bboxes_stats

def process_subset(file_list, subset_name, batch_prefix, raw_labels_path, raw_images_path, move_mode=False, is_yolo=False, is_segmentation=False, raw_seg_path=None, override_all=-1, override_map=None):
    count_imgs = 0
    count_objs = 0
    count_bgs = 0
    
    # Lists for EDA
    all_areas, all_ars, all_cx, all_cy = [], [], [], []
    
    for fname in file_list:
        success, num_objects, bbox_stats = process_pair(
            fname, subset_name, batch_prefix, raw_labels_path, raw_images_path,
            move_mode=move_mode, is_yolo=is_yolo, is_segmentation=is_segmentation,
            raw_seg_path=raw_seg_path, override_all=override_all, override_map=override_map
        )
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
        
    eda_stats = {
        "bbox_area": mu.calculate_1d_stats(all_areas),
        "aspect_ratio": mu.calculate_1d_stats(all_ars),
        "center_x": mu.calculate_1d_stats(all_cx),
        "center_y": mu.calculate_1d_stats(all_cy)
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
    parser.add_argument('--source', type=str, default=config.RAW_DATA_DIR, help="Path to the raw dataset folder")
    parser.add_argument('--is_yolo', action='store_true', help="Indicates that source labels are already in YOLO format")
    parser.add_argument('--segmentation', action='store_true', help="Conversion of pixel masks to YOLO polygons")
    parser.add_argument('--override_class', nargs='+', default=[], help="Override classes. Use a single number to override ALL" \
                        " (e.g., --override_class 0) or pairs to map specific classes " \
                        "(e.g., --override_class mountain_bike=0 road_bike=0)")
    args = parser.parse_args()

    raw_images_path = Path(args.source) / config.RAW_IMAGES_SUBPATH
    raw_labels_path = Path(args.source) / config.RAW_LABELS_SUBPATH

    def _has_valid_masks(folder, sample_limit=5):
        if not folder.exists():
            return False
        png_files = list(folder.glob("*.png"))
        if not png_files:
            return False
        for f in png_files[:sample_limit]:
            mask = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
            if mask is not None and np.any(mask > 0):
                return True
        return False

    raw_seg_path = None
    if args.segmentation:
        inst_path = Path(args.source) / config.RAW_INSTANCE_SEG_SUBPATH
        sem_path = Path(args.source) / config.RAW_SEMANTIC_SEG_SUBPATH
        if _has_valid_masks(inst_path):
            raw_seg_path = inst_path
            print(f" (Segmentation) Instance segmentation masks found in: {raw_seg_path}")
        elif _has_valid_masks(sem_path):
            raw_seg_path = sem_path
            print(f" (Segmentation) Semantic segmentation masks found in: {raw_seg_path}")
        elif inst_path.exists():
            raw_seg_path = inst_path
        elif sem_path.exists():
            raw_seg_path = sem_path
        else:
            print(f"[WARN] Flag '--segmentation' active, but no mask folder found. Fallback to bounding box polygons.")

    if not raw_images_path.exists():
        print(f"❌ Error: Didn't find raw image data in: {raw_images_path}")
        return

    override_map = {}
    override_all = -1
    
    if args.override_class:
        if len(args.override_class) == 1 and args.override_class[0].isdigit():
            override_all = int(args.override_class[0])
            print(f"⚠️  Hammer mode: Converting ALL found classes to ID: {override_all}")
        else:
            for mapping in args.override_class:
                if '=' in mapping:
                    src, dst = mapping.split('=')
                    override_map[src] = int(dst)
                else:
                    print(f"❌ Invalid format: {mapping}. Ignoring. Use origin=destination format (e.g., road_bike=0)")
            print(f"🔀 Scalpel mode. Active mapping: {override_map}")

    # 1. Prepare structure
    create_dir_structure(args.append)

    # 2. Generate unique prefix for this batch of data
    batch_prefix = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"🆔 ID of Batch (Batch ID): {batch_prefix}")

    if args.move:
        print(f"⚠️  WARNING: Flag '--move' active. Original files in {args.source} will be DELETED to save space.")

    # 3. List files
    if raw_labels_path.exists() and len(list(raw_labels_path.glob("*.txt"))) > 0:
        all_files = [f.stem for f in raw_labels_path.glob("*.txt")]
    elif raw_seg_path is not None and raw_seg_path.exists() and len(list(raw_seg_path.glob("*.png"))) > 0:
        all_files = [f.stem for f in raw_seg_path.glob("*.png")]
    else:
        all_files = [f.stem for f in raw_images_path.glob("*.*") if f.suffix.lower() in config.VALID_IMAGE_EXTENSIONS]

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

    train_end = int(total_files * config.TRAIN_RATIO)
    val_end = train_end + int(total_files * config.VAL_RATIO)
    
    train_files = all_files[:train_end]
    val_files = all_files[train_end:val_end]
    test_files = all_files[val_end:]

    print(f"📊 New files found: {total_files}")
    print(f"   Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(test_files)}")
    if args.segmentation:
        print("   Task: Segmentation (YOLO polygon labels)")
    else:
        print("   Task: Detection (YOLO bounding box labels)")
    
    if args.append:
        print("   -> New data will be added to the existing dataset.")
    else:
        print("   -> A new dataset will be created (deleting the previous one).")
    
    print("-" * 40)

    # 5. Process passing the prefix
    print("🚀 Processing Train...")
    train_stats = process_subset(train_files, 'train', batch_prefix, raw_labels_path, raw_images_path, args.move, args.is_yolo, args.segmentation, raw_seg_path, override_all, override_map)
    
    print("🚀 Processing Val...")
    val_stats = process_subset(val_files, 'val', batch_prefix, raw_labels_path, raw_images_path, args.move, args.is_yolo, args.segmentation, raw_seg_path, override_all, override_map)
    
    print("🚀 Processing Test...")
    test_stats = process_subset(test_files, 'test', batch_prefix, raw_labels_path, raw_images_path, args.move, args.is_yolo, args.segmentation, raw_seg_path, override_all, override_map)

    print("-" * 40)
    print("✅ PROCESSING COMPLETED")
    total_added_imgs = train_stats["images"] + val_stats["images"] + test_stats["images"]
    print(f"New files added: {total_added_imgs}")
    print(f"Dataset located in: {config.PROCESSED_DATA_DIR}")

    # 6. Update Dataset Metadata
    meta_manager = DatasetMetadata(config.DATASET_METADATA_PATH)
    
    source_meta_path = Path(args.source) / config.FILE_GEN_META

    meta_manager.record_session(
        batch_id=batch_prefix,
        source_meta_path=source_meta_path,
        train_stats=train_stats,
        val_stats=val_stats,
        test_stats=test_stats,
        total_added_imgs=total_added_imgs
    )
    
    meta_manager.set_timestamp(key_name=config.UPDATE_TIMESTAMP_KEY)
    meta_manager.commit()


if __name__ == "__main__":
    main()