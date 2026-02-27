import os
import cv2
import shutil
import glob
import argparse
import sys
from ultralytics import YOLO
import modules.core_visual_utils as cvu

# Import the class from the other file
try:
    from modules.audit_reporter import ReportGenerator
    from modules.inference_reporter import InferenceReportGenerator
except ImportError:
    print("❌ CRITICAL ERROR: Not finding 'audit_reporter.py' or 'inference_reporter.py'.")
    print("   Make sure both files are on 'modules' folder.")
    sys.exit(1)


def select_model_path():
    """
    Interactively selects the model path.
    """
    print("\n--- 🤖 MODEL SELECTION ---")
    
    # Check if project dir exists
    if not os.path.exists(cvu.PROJECT_DIR):
        print(f"⚠️ Warning: Project directory '{cvu.PROJECT_DIR}' not found.")
        print("   (Maybe you haven't trained any model yet?)")
    
    user_input = input(f"Enter experiment name (default: '{cvu.DEFAULT_EXP_NAME}'): ").strip()
    
    exp_name = user_input if user_input else cvu.DEFAULT_EXP_NAME
    
    # Construct path: project/exp_name/weights/best.pt
    path = os.path.join(cvu.PROJECT_DIR, exp_name, "weights", "best.pt")
    
    print(f"-> Selected model: {path}\n")
    return path


def run_audit_mode(model_path, draw_all=False):
    """ Audit mode (Dataset Test with Labels) """
    if not cvu.check_system_integrity(model_path, check_dataset=True):
        return

    print(f"--- 🕵️ STARTING AUDIT (Draw All: {draw_all}) ---")
    
    # 1. Prepare folders according to the mode
    if os.path.exists(cvu.OUTPUT_DIR): shutil.rmtree(cvu.OUTPUT_DIR)
    
    path_fn = os.path.join(cvu.OUTPUT_DIR, "audit", "audit_missed_FN")
    path_fp = os.path.join(cvu.OUTPUT_DIR, "audit", "audit_invented_FP")
    path_poor = os.path.join(cvu.OUTPUT_DIR, "audit", "audit_poor_bbox")
    path_all = os.path.join(cvu.OUTPUT_DIR, "audit", "audit_all")

    if draw_all:
        os.makedirs(path_all)
        short_all = path_all[path_all.find("YOLO"):] if "YOLO" in path_all else path_all
        print(f"📂 Saving everything in: {short_all}")
    else:
        os.makedirs(path_fn)
        os.makedirs(path_fp)
        os.makedirs(path_poor)
        shorten = lambda p: p[p.find("YOLO"):] if "YOLO" in p else p
        print(f"📂 Separating errors in: {shorten(path_fn)}, {shorten(path_fp)} and {shorten(path_poor)}")

    reporter = ReportGenerator(cvu.OUTPUT_DIR, cvu.IOU_THRESHOLD)
    model = YOLO(model_path)
    
    image_files = glob.glob(os.path.join(cvu.DEFAULT_TEST_IMAGES, "*.*"))
    print(f"📸 Processing {len(image_files)} images...")

    for i, img_path in enumerate(image_files):
        filename = os.path.basename(img_path)
        txt_path = os.path.join(cvu.DEFAULT_TEST_LABELS, os.path.splitext(filename)[0] + ".txt")

        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w, _ = img.shape
        gt_boxes = cvu.parse_kitti_label(txt_path, w, h)

        # Inference
        results = model.predict(source=img, conf=cvu.CONF_THRESHOLD, verbose=False)[0]
        
        pred_boxes = []
        confidences = []
        for box in results.boxes:
            coords = box.xyxy[0].cpu().numpy()
            pred_boxes.append(coords)
            confidences.append(float(box.conf))

        # Statistics
        img_stats = reporter.update(pred_boxes, gt_boxes, confidences, (h, w))

        # Save and Organization Logic
        if i < cvu.LIMIT_IMAGES:
            
            # A. Check if there is any error in the image
            has_errors = img_stats["FN"] > 0 or img_stats["poor_bbox"] > 0 or img_stats["FP"] > 0
            
            # B. Draw the boxes
            if draw_all or has_errors:
                img_drawn = cvu.draw_boxes(img.copy(), gt_boxes, color=(0, 255, 0), label="REAL")
                img_drawn = cvu.draw_boxes(img_drawn, pred_boxes, color=(255, 0, 0), confidences=confidences)
                
                # C. Save logic according to the mode
                if draw_all:
                    status = "OK"
                    if img_stats["FN"] > 0: status = "FN"
                    elif img_stats["poor_bbox"] > 0: status = "POOR"
                    elif img_stats["FP"] > 0: status = "FP"
                    
                    save_path = os.path.join(path_all, f"{status}_{filename}")
                    cv2.imwrite(save_path, img_drawn)
                    
                else:
                    if img_stats["FN"] > 0:
                        cv2.imwrite(os.path.join(path_fn, filename), img_drawn)
                    if img_stats["poor_bbox"] > 0:
                        cv2.imwrite(os.path.join(path_poor, filename), img_drawn)
                    if img_stats["FP"] > 0:
                        cv2.imwrite(os.path.join(path_fp, filename), img_drawn)

    # Generate final report
    exp_name = os.path.basename(os.path.dirname(os.path.dirname(model_path)))
    reporter.generate_plots()
    reporter.generate_html_report(exp_name)


def run_inference_mode(model_path, source_folder):
    """ Inference Mode (New images without labels) """
    
    if not cvu.check_system_integrity(model_path, check_dataset=False):
        return

    if not os.path.exists(source_folder):
        print(f"❌ ERROR: Not finding the source folder: {source_folder}")
        return

    print(f"--- 🌍 REAL INFERENCE MODE ---")
    
    save_dir = os.path.join(cvu.OUTPUT_DIR, "inference_real")
    if os.path.exists(save_dir): shutil.rmtree(save_dir)
    os.makedirs(save_dir)
    
    short_dir = save_dir[save_dir.find("YOLO"):] if "YOLO" in save_dir else save_dir
    print(f"📂 Saving inference results in: {short_dir}")

    reporter = InferenceReportGenerator(save_dir, overlap_threshold=cvu.OVERLAP_THRESHOLD_ANALYSIS)
    overlaps_dir_path = reporter.overlaps_dir
    
    model = YOLO(model_path)
    images = glob.glob(os.path.join(source_folder, "*.*"))
    
    if not images:
        print("⚠️ No images found in the specified folder.")    
        return

    print(f"Processing {len(images)} images...")
    
    # Process each image for detection and quality analysis
    for i, img_path in enumerate(images):
        if i >= cvu.LIMIT_IMAGES: break

        filename = os.path.basename(img_path)
        
        try:
            img_orig = cv2.imread(img_path)
            if img_orig is None:
                print(f"⚠️ Warning: Could not read image {filename}. Skipping...")
                continue

            h, w, _ = img_orig.shape

            # Run YOLO inference
            res = model.predict(img_path, conf=cvu.CONF_THRESHOLD, verbose=False)[0]
            
            pred_boxes = []
            confidences = []
            for box in res.boxes:
                coords = box.xyxy[0].cpu().numpy()
                pred_boxes.append(coords)
                confidences.append(float(box.conf))
                
            # Analyze predictions for overlapping boxes (potential double detections)
            problematic_pairs = reporter.update(pred_boxes, confidences, (h, w), filename)
            
            # Save visual evidence if overlaps are found
            if problematic_pairs:
                img_overlap_evidence = img_orig.copy()
                img_overlap_evidence = cvu.draw_overlapping_pairs(
                    img_overlap_evidence, 
                    pred_boxes, 
                    problematic_pairs, 
                    confidences
                )
                evidence_path = os.path.join(overlaps_dir_path, f"OVERLAP_{filename}")
                cv2.imwrite(evidence_path, img_overlap_evidence)
            
            # Save standard detection visualization
            res_plotted = res.plot()
            cv2.imwrite(os.path.join(save_dir, f"PRED_{filename}"), res_plotted)
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            import traceback
            traceback.print_exc()
            
    # Finalize and generate the summary report
    exp_name = os.path.basename(os.path.dirname(os.path.dirname(model_path)))
    reporter.generate_plots()
    reporter.generate_html_report(exp_name)


def run_video_mode(model_path, video_path):
    """ Video Mode """
    if not cvu.check_system_integrity(model_path, check_dataset=False):
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
    results = model.predict(source=video_path, save=True, conf=cvu.CONF_THRESHOLD, stream=True)
    
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