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
    from inference_utils import InferenceReportGenerator
except ImportError:
    print("❌ CRITICAL ERROR: Not finding 'report_utils.py' or 'inference_utils.py'.")
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


def draw_overlapping_pairs(img, pred_boxes, pairs_indices, confidences=None):
    """
    Dibuja SOLO los pares de cajas que se solapan para resaltarlos.
    pairs_indices: Lista de tuplas [(idx1, idx2), (idx3, idx4)...]
    """
    alert_color = (0, 165, 255) 
    thickness = 3
    
    boxes_to_draw_idx = set()
    for i, j in pairs_indices:
        boxes_to_draw_idx.add(i)
        boxes_to_draw_idx.add(j)
        
    for idx in boxes_to_draw_idx:
        box = pred_boxes[idx]
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), alert_color, thickness)
        
        if confidences:
            conf = confidences[idx]
            label = f"{conf:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), alert_color, -1)
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
    return img


# --- EXECUTION MODES ---

def run_audit_mode(model_path, draw_all=False):
    """ Audit mode (Dataset Test with Labels) """
    if not check_system_integrity(model_path, check_dataset=True):
        return

    print(f"--- 🕵️ STARTING AUDIT (Draw All: {draw_all}) ---")
    
    # 1. Prepare folders according to the mode
    if os.path.exists(OUTPUT_DIR): shutil.rmtree(OUTPUT_DIR)
    
    path_fn = os.path.join(OUTPUT_DIR, "audit", "audit_missed_FN")
    path_fp = os.path.join(OUTPUT_DIR, "audit", "audit_invented_FP")
    path_poor = os.path.join(OUTPUT_DIR, "audit", "audit_poor_bbox")
    path_all = os.path.join(OUTPUT_DIR, "audit", "audit_all")

    if draw_all:
        os.makedirs(path_all)
        print(f"📂 Saving everything in: {path_all}")
    else:
        os.makedirs(path_fn)
        os.makedirs(path_fp)
        os.makedirs(path_poor)
        print(f"📂 Separating errors in: {path_fn}, {path_fp} and {path_poor}")

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
        img_stats = reporter.update(pred_boxes, gt_boxes, confidences)

        # Save and Organization Logic
        if i < LIMIT_IMAGES:
            
            # A. Check if there is any error in the image
            has_errors = img_stats["FN"] > 0 or img_stats["poor_bbox"] > 0 or img_stats["FP"] > 0
            
            # B. Draw the boxes ONLY ONCE, only if it is necessary to save the image
            if draw_all or has_errors:
                img_drawn = draw_boxes(img.copy(), gt_boxes, color=(0, 255, 0), label="REAL")
                img_drawn = draw_boxes(img_drawn, pred_boxes, color=(255, 0, 0), confidences=confidences)
                
                # C. Save logic according to the mode
                if draw_all:
                    # Default status OK. The order of the if marks the priority of the error.
                    status = "OK"
                    if img_stats["FN"] > 0: status = "FN"
                    elif img_stats["poor_bbox"] > 0: status = "POOR"
                    elif img_stats["FP"] > 0: status = "FP"
                    
                    save_path = os.path.join(path_all, f"{status}_{filename}")
                    cv2.imwrite(save_path, img_drawn)
                    
                else:
                    # Save the same pre-drawn image in the error folders it has
                    if img_stats["FN"] > 0:
                        cv2.imwrite(os.path.join(path_fn, filename), img_drawn)
                    if img_stats["poor_bbox"] > 0:
                        cv2.imwrite(os.path.join(path_poor, filename), img_drawn)
                    if img_stats["FP"] > 0:
                        cv2.imwrite(os.path.join(path_fp, filename), img_drawn)

    # Generate final report
    reporter.generate_plots()
    reporter.generate_html_report()


def run_inference_mode(model_path, source_folder):
    """ Inference Mode (New images without labels) """
    OVERLAP_THRESHOLD_ANALYSIS = 0.45
    
    if not check_system_integrity(model_path, check_dataset=False):
        return

    if not os.path.exists(source_folder):
        print(f"❌ ERROR: Not finding the source folder: {source_folder}")
        return

    print(f"--- 🌍 REAL INFERENCE MODE ---")
    save_dir = os.path.join(OUTPUT_DIR, "inference_real")
    if os.path.exists(save_dir): shutil.rmtree(save_dir)
    os.makedirs(save_dir)

    reporter = InferenceReportGenerator(save_dir, overlap_threshold=OVERLAP_THRESHOLD_ANALYSIS)
    overlaps_dir_path = reporter.overlaps_dir
    
    model = YOLO(model_path)
    images = glob.glob(os.path.join(source_folder, "*.*"))
    
    if not images:
        print("⚠️ No images found in the specified folder.")    
        return

    print(f"Processing {len(images)} images...")
    for i, img_path in enumerate(images):
        if i >= LIMIT_IMAGES: break

        filename = os.path.basename(img_path)
        
        try:
            img_orig = cv2.imread(img_path)
            h, w = img_orig.shape[:2]

            res = model.predict(img_path, conf=CONF_THRESHOLD, verbose=False)[0]
            
            pred_boxes = []
            confidences = []
            for box in res.boxes:
                coords = box.xyxy[0].cpu().numpy()
                pred_boxes.append(coords)
                confidences.append(float(box.conf))
                
            problematic_pairs = reporter.update(pred_boxes, confidences, (h, w), filename)
            
            if problematic_pairs:
                img_overlap_evidence = img_orig.copy()
                img_overlap_evidence = draw_overlapping_pairs(
                    img_overlap_evidence, 
                    pred_boxes, 
                    problematic_pairs, 
                    confidences
                )
                evidence_path = os.path.join(overlaps_dir_path, f"OVERLAP_{filename}")
                cv2.imwrite(evidence_path, img_overlap_evidence)
            res_plotted = res.plot()
            cv2.imwrite(os.path.join(save_dir, f"PRED_{filename}"), res_plotted)
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            import traceback
            traceback.print_exc()
            
    reporter.generate_plots()
    reporter.generate_html_report()


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