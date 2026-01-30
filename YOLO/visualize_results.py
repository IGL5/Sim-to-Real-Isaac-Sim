import os
import cv2
import shutil
import glob
import argparse
import sys
from ultralytics import YOLO

# Import the class from the other file
try:
    from report_utils import ReportGenerator
except ImportError:
    print("❌ CRITICAL ERROR: Not finding 'report_utils.py'.")
    print("   Make sure both files are in the same folder.")
    sys.exit(1)

# --- CONFIGURATION ---
# Base directory where experiments are saved
PROJECT_DIR = os.path.join(os.getcwd(), "cyclist_detector")
DEFAULT_EXP_NAME = "yolov8_s_default"

# Default paths (Test Dataset)
DEFAULT_TEST_IMAGES = os.path.join(os.getcwd(), "dataset_yolo_output", "images", "test")
DEFAULT_TEST_LABELS = os.path.join(os.getcwd(), "dataset_yolo_output", "labels", "test")

# Output for reports
OUTPUT_DIR = os.path.join(os.getcwd(), "audit_report")

# Parameters
CONF_THRESHOLD = 0.4
IOU_THRESHOLD = 0.5
LIMIT_IMAGES = 100 


# --- MODEL SELECTION ---

def select_model_path():
    """
    Interactively selects the model path.
    """
    print("\n--- 🤖 MODEL SELECTION ---")
    
    # Check if project dir exists
    if not os.path.exists(PROJECT_DIR):
        print(f"⚠️ Warning: Project directory '{PROJECT_DIR}' not found.")
        print("   (Maybe you haven't trained any model yet?)")
    
    user_input = input(f"Enter experiment name (default: '{DEFAULT_EXP_NAME}'): ").strip()
    
    exp_name = user_input if user_input else DEFAULT_EXP_NAME
    
    # Construct path: project/exp_name/weights/best.pt
    path = os.path.join(PROJECT_DIR, exp_name, "weights", "best.pt")
    
    print(f"-> Selected model: {path}\n")
    return path


# --- UTILITIES ---

def check_system_integrity(model_path, check_dataset=False):
    """
    Checks that everything necessary exists before starting.
    """
    # 1. Check Model
    if not os.path.exists(model_path):
        print(f"❌ ERROR: Not finding the model file in:")
        print(f"   -> {model_path}")
        print("   Did you run the training script (train_YOLO.py)?")
        return False

    # 2. Check Dataset (only if we are going to audit)
    if check_dataset:
        if not os.path.exists(DEFAULT_TEST_IMAGES):
            print(f"❌ ERROR: Not finding the test images folder:")
            print(f"   -> {DEFAULT_TEST_IMAGES}")
            return False
        
        # Check if not empty
        if not os.listdir(DEFAULT_TEST_IMAGES):
            print(f"⚠️ WARNING: The test folder is empty ({DEFAULT_TEST_IMAGES}).")
            return False

    return True

def parse_kitti_label(label_path, width, height):
    """ Converts txt to list of boxes [x1, y1, x2, y2] """
    boxes = []
    if not os.path.exists(label_path): return boxes
    
    try:
        with open(label_path, 'r') as f:
            for line in f:
                p = line.strip().split()
                # YOLO format: id xc yc w h
                xc, yc, w, h = float(p[1]), float(p[2]), float(p[3]), float(p[4])
                x1 = (xc - w/2) * width
                y1 = (yc - h/2) * height
                x2 = (xc + w/2) * width
                y2 = (yc + h/2) * height
                boxes.append([x1, y1, x2, y2])
    except Exception as e:
        print(f"⚠️ Error reading label {label_path}: {e}")
        
    return boxes

def draw_boxes(img, boxes, color=(0, 255, 0), label="", confidences=None):
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = map(int, b)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # If there is a specific confidence, we use it. If not, we use the generic label
        text_to_draw = label
        if confidences is not None and i < len(confidences):
            text_to_draw = f"{confidences[i]:.2f}"
            
        if text_to_draw: 
            cv2.putText(img, text_to_draw, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return img


# --- EXECUTION MODES ---

def run_audit_mode(model_path, draw_all=False):
    """ Audit mode (Dataset Test with Labels) """
    if not check_system_integrity(model_path, check_dataset=True):
        return

    print(f"--- 🕵️ STARTING AUDIT (Draw All: {draw_all}) ---")
    
    # 1. Prepare folders according to the mode
    if os.path.exists(OUTPUT_DIR): shutil.rmtree(OUTPUT_DIR)
    
    path_fn = os.path.join(OUTPUT_DIR, "audit_missed_FN")
    path_fp = os.path.join(OUTPUT_DIR, "audit_invented_FP")
    path_all = os.path.join(OUTPUT_DIR, "audit_all")

    if draw_all:
        os.makedirs(path_all)
        print(f"📂 Saving everything in: {path_all}")
    else:
        os.makedirs(path_fn)
        os.makedirs(path_fp)
        print(f"📂 Separating errors in: {path_fn} and {path_fp}")

    reporter = ReportGenerator(OUTPUT_DIR, IOU_THRESHOLD)
    model = YOLO(model_path)
    
    image_files = glob.glob(os.path.join(DEFAULT_TEST_IMAGES, "*.*"))
    print(f"📸 Processing {len(image_files)} images...")

    for i, img_path in enumerate(image_files):
        filename = os.path.basename(img_path)
        txt_path = os.path.join(DEFAULT_TEST_LABELS, os.path.splitext(filename)[0] + ".txt")

        # Read image
        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w, _ = img.shape
        gt_boxes = parse_kitti_label(txt_path, w, h)

        # Inference
        results = model.predict(source=img, conf=CONF_THRESHOLD, verbose=False)[0]
        
        pred_boxes = []
        confidences = []
        for box in results.boxes:
            coords = box.xyxy[0].cpu().numpy()
            pred_boxes.append(coords)
            confidences.append(float(box.conf))

        # Statistics
        reporter.update(pred_boxes, gt_boxes, confidences)

        # Save and Organization Logic
        if i < LIMIT_IMAGES:
            num_gt = len(gt_boxes)
            num_pred = len(pred_boxes)
            
            # Simple error classification for the folder
            status = "OK"
            if num_gt > num_pred: status = "FN"
            elif num_pred > num_gt: status = "FP"

            should_save = False
            save_path = ""

            if draw_all:
                should_save = True
                # In 'all' mode, we save everything with a prefix to identify
                save_path = os.path.join(path_all, f"{status}_{filename}")
            else:
                # In normal mode, we only save if there is an error
                if status == "FN":
                    should_save = True
                    save_path = os.path.join(path_fn, filename)
                elif status == "FP":
                    should_save = True
                    save_path = os.path.join(path_fp, filename)

            if should_save:
                img = draw_boxes(img, gt_boxes, color=(0, 255, 0), label="REAL")
                img = draw_boxes(img, pred_boxes, color=(255, 0, 0), confidences=confidences)
                
                cv2.imwrite(save_path, img)

    # Generate final report
    reporter.generate_plots()
    reporter.generate_html_report()


def run_inference_mode(model_path, source_folder):
    """ Inference Mode (New images without labels) """
    if not check_system_integrity(model_path, check_dataset=False):
        return

    if not os.path.exists(source_folder):
        print(f"❌ ERROR: Not finding the source folder: {source_folder}")
        return

    print(f"--- 🌍 REAL INFERENCE MODE ---")
    save_dir = os.path.join(OUTPUT_DIR, "inference_real")
    if os.path.exists(save_dir): shutil.rmtree(save_dir)
    os.makedirs(save_dir)
    
    model = YOLO(model_path)
    images = glob.glob(os.path.join(source_folder, "*.*"))
    
    if not images:
        print("⚠️ No images found in the specified folder.")    
        return

    print(f"Processing {len(images)} images...")
    for i, img_path in enumerate(images):
        if i >= LIMIT_IMAGES: break
        
        try:
            res = model.predict(img_path, conf=CONF_THRESHOLD, verbose=False)
            filename = "PRED_" + os.path.basename(img_path)
            res[0].save(os.path.join(save_dir, filename))
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    print(f"✅ Results saved in: {save_dir}")


def run_video_mode(model_path, video_path):
    """ Video Mode """
    if not check_system_integrity(model_path, check_dataset=False):
        return

    if not os.path.exists(video_path):
        print(f"❌ ERROR: Not finding the video file: {video_path}")
        return

    print(f"--- 🎥 VIDEO MODE ---")
    print(f"Processing: {video_path}")
    print("This may take a while depending on the duration...")

    model = YOLO(model_path)
    
    # save=True makes YOLO automatically save the video in runs/detect/predict...
    # stream=True saves memory on long videos
    results = model.predict(source=video_path, save=True, conf=CONF_THRESHOLD, stream=True)
    
    # We need to iterate over the generator to process the video
    for r in results:
        pass 

    print("\n✅ Video processed correctly.")
    print("📂 Look for it in the folder: 'runs/detect' (the most recent)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO Diagnostic Tool")
    
    # Options
    parser.add_argument('--draw_all', action='store_true', help="Audit: Save ALL images (hits and misses)")
    parser.add_argument('--source', type=str, default=None, help="Inference: Folder of new images (without labels)")
    parser.add_argument('--video', type=str, default=None, help="Video: Path to the MP4/AVI file")
    
    args = parser.parse_args()

    # Ask for Model Name (or use default)
    selected_model_path = select_model_path()

    # Mode Selector
    if args.video:
        run_video_mode(selected_model_path, args.video)
    elif args.source:
        run_inference_mode(selected_model_path, args.source)
    else:
        # Default: Audit the test dataset
        run_audit_mode(selected_model_path, draw_all=args.draw_all)