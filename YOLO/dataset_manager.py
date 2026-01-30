import os
import cv2
import shutil
import random
import argparse
import time
from datetime import datetime

# --- CONFIG ---

# Input paths (where are your NEW data from Isaac Sim)
LABELS_KITTI = os.path.join(os.getcwd(), "..", "_output_data", "DroneCamera", "object_detection")
IMAGES_DIR = os.path.join(os.getcwd(), "..", "_output_data", "DroneCamera", "rgb")

# Output path (where the dataset will be stored)
BASE_OUTPUT = os.path.join(os.getcwd(), "dataset_yolo_output")

# Classes to detect
CLASES = ["cyclist"]

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

def process_pair(filename_base, subset_name, unique_prefix):
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
        return False 

    # Read dimensions
    img = cv2.imread(img_path)
    if img is None:
        return False
    height, width, _ = img.shape

    # 2. Process KITTI label
    kitti_path = os.path.join(LABELS_KITTI, filename_base + ".txt")
    yolo_lines = []
    
    # If the label file doesn't exist (sometimes Isaac generates the image but fails the txt), skip it
    if not os.path.exists(kitti_path):
        return False

    with open(kitti_path, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        parts = line.strip().split(' ')
        class_name = parts[0]
        
        if class_name not in CLASES:
            continue
            
        class_id = CLASES.index(class_name)
        
        try:
            xmin, ymin = float(parts[4]), float(parts[5])
            xmax, ymax = float(parts[6]), float(parts[7])
            
            bbox = change_coordinates((width, height), (xmin, xmax, ymin, ymax))
            yolo_lines.append(f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}")
        except (ValueError, IndexError):
            continue

    # 3. Save with NEW UNIQUE NAME
    # Format: batch_20231027_rgb_0001.png
    new_filename = f"{unique_prefix}_{filename_base}"
    
    dest_img = os.path.join(BASE_OUTPUT, 'images', subset_name, new_filename + img_ext)
    dest_lbl = os.path.join(BASE_OUTPUT, 'labels', subset_name, new_filename + ".txt")
    
    # Copy image
    shutil.copy2(img_path, dest_img)
    
    # Save new txt
    with open(dest_lbl, 'w') as f_out:
        if yolo_lines:
            f_out.write('\n'.join(yolo_lines))
            
    return True

def main():
    parser = argparse.ArgumentParser(description="Dataset manager from KITTI to YOLO")
    parser.add_argument('--append', action='store_true', help="Add new data to the existing dataset without deleting anything.")
    args = parser.parse_args()

    if not os.path.exists(LABELS_KITTI) or not os.path.exists(IMAGES_DIR):
        print(f"❌ Error: Verify the input paths ({LABELS_KITTI})")
        return

    # 1. Prepare structure
    create_dir_structure(args.append)

    # 2. Generate unique prefix for this batch of data
    # We use date and time until seconds to ensure uniqueness: "20231027_153022"
    batch_prefix = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"🆔 ID of Batch (Batch ID): {batch_prefix}")

    # 3. List files
    all_files = [os.path.splitext(f)[0] for f in os.listdir(LABELS_KITTI) if f.endswith(".txt")]
    total_files = len(all_files)
    
    if total_files == 0:
        print("⚠️ No files found to process.")
        return

    # 4. Shuffle and divide
    random.shuffle(all_files)

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
    def process_subset(file_list, subset_name):
        count = 0
        for fname in file_list:
            if process_pair(fname, subset_name, batch_prefix):
                count += 1
        return count

    print("🚀 Processing Train...")
    c_train = process_subset(train_files, 'train')
    
    print("🚀 Processing Val...")
    c_val = process_subset(val_files, 'val')
    
    print("🚀 Processing Test...")
    c_test = process_subset(test_files, 'test')

    print("-" * 40)
    print("✅ PROCESSING COMPLETED")
    print(f"New files added: {c_train + c_val + c_test}")
    print(f"Dataset located in: {BASE_OUTPUT}")


if __name__ == "__main__":
    main()