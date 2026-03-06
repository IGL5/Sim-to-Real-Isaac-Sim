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
    Interactively selects the model path checking for available models.
    """
    print("\n--- 🤖 MODEL SELECTION ---")
    
    # Check if project directory exists
    if not os.path.exists(cvu.PROJECT_DIR):
        print(f"❌ ERROR: Project directory '{cvu.PROJECT_DIR}' not found.")
        print("   (You need to train a model first using train_YOLO.py)")
        sys.exit(1)
        
    # Loop through the project directory to find models with weights
    available_models = []
    for d in os.listdir(cvu.PROJECT_DIR):
        model_dir = os.path.join(cvu.PROJECT_DIR, d)
        if os.path.isdir(model_dir):
            weights_path = os.path.join(model_dir, "weights", "best.pt")
            if os.path.exists(weights_path):
                available_models.append(d)
                
    if not available_models:
        print(f"❌ ERROR: No trained models found in '{cvu.PROJECT_DIR}'.")
        print("   (Folders exist, but none contain 'weights/best.pt')")
        sys.exit(1)
        
    print("📂 Available trained models:")
    for i, m in enumerate(available_models):
        print(f"  [{i+1}] {m}")
        
    while True:
        user_input = input(f"\nSelect a model [1-{len(available_models)}] (default: 1): ").strip()
        
        if not user_input:
            exp_name = available_models[0]
            break
        
        if user_input.isdigit():
            idx = int(user_input) - 1
            if 0 <= idx < len(available_models):
                exp_name = available_models[idx]
                break
            else:
                print(f"  ⚠️  Number out of range. Please choose between 1 and {len(available_models)}.")
        else:
            if user_input in available_models:
                exp_name = user_input
                break
            print("  ⚠️  Invalid input. Please enter a valid number.")
            
    path = os.path.join(cvu.PROJECT_DIR, exp_name, "weights", "best.pt")
    print(f"✅ Selected model: {exp_name}\n")
    return path


def save_evaluation_results(exp_name, mode):
    """ Copies the temporary audit_report into a persistent iteration folder """
    base_eval_dir = os.path.join(cvu.PROJECT_DIR, exp_name, "evaluations")
    os.makedirs(base_eval_dir, exist_ok=True)
    
    # Find the next iteration number
    existing_dirs = [d for d in os.listdir(base_eval_dir) if os.path.isdir(os.path.join(base_eval_dir, d))]
    iter_num = len(existing_dirs) + 1
    eval_dir = os.path.join(base_eval_dir, f"iter_{iter_num:03d}_{mode}")
    
    # Copy the entire workspace (HTML, Plots, JSON, Images)
    shutil.copytree(cvu.OUTPUT_DIR, eval_dir)
    print(f"\n📦 Persistent Evaluation saved at: {eval_dir}")


def run_audit_mode(model_path, draw_all=False, save_persistently=False, custom_img_dir=None, custom_lbl_dir=None, keep_dir=False):
    """ Audit mode (Dataset Test with Labels) - Supports both Sim and Real """
    if not cvu.check_system_integrity(model_path, check_dataset=(custom_img_dir is None)):
        return

    is_real = custom_img_dir is not None and custom_lbl_dir is not None
    prefix = "real_audit" if is_real else "audit"

    print(f"--- 🕵️ STARTING {'REAL ' if is_real else ''}AUDIT (Draw All: {draw_all}) ---")
    
    # 1. Prepare folders according to the mode
    if not keep_dir:
        if os.path.exists(cvu.OUTPUT_DIR): shutil.rmtree(cvu.OUTPUT_DIR)
    
    path_fn = os.path.join(cvu.OUTPUT_DIR, prefix, f"{prefix}_missed_FN")
    path_fp = os.path.join(cvu.OUTPUT_DIR, prefix, f"{prefix}_invented_FP")
    path_poor = os.path.join(cvu.OUTPUT_DIR, prefix, f"{prefix}_poor_bbox")
    path_all = os.path.join(cvu.OUTPUT_DIR, prefix, f"{prefix}_all")

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

    reporter = ReportGenerator(cvu.OUTPUT_DIR, cvu.IOU_THRESHOLD, prefix=prefix)
    model = YOLO(model_path)

    # Determine which folders to use
    img_dir = custom_img_dir if is_real else cvu.DEFAULT_TEST_IMAGES
    lbl_dir = custom_lbl_dir if is_real else cvu.DEFAULT_TEST_LABELS
    
    image_files = [f for f in glob.glob(os.path.join(img_dir, "*")) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"📸 Processing {len(image_files)} images...")

    for i, img_path in enumerate(image_files):
        filename = os.path.basename(img_path)
        txt_path = os.path.join(lbl_dir, os.path.splitext(filename)[0] + ".txt")

        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w, _ = img.shape
        gt_boxes = cvu.parse_kitti_label(txt_path, w, h)

        # Inference
        results = model.predict(source=img, conf=cvu.CONF_THRESHOLD, verbose=False)[0]
        speed_dict = results.speed
        
        pred_boxes = []
        confidences = []
        for box in results.boxes:
            coords = box.xyxy[0].cpu().numpy()
            pred_boxes.append(coords)
            confidences.append(float(box.conf))

        # Statistics
        img_stats = reporter.update(pred_boxes, gt_boxes, confidences, (h, w), speed_dict)

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
    reporter.generate_html_report(exp_name, is_real_audit=is_real)

    if save_persistently:
        save_evaluation_results(exp_name, prefix)


def run_inference_mode(model_path, source_folder, save_persistently=False):
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
    image_files = [f for f in glob.glob(os.path.join(source_folder, "*")) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print("⚠️ No images found in the specified folder.")    
        return

    print(f"Processing {len(image_files)} images...")
    
    # Process each image for detection and quality analysis
    for i, img_path in enumerate(image_files):
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
            speed_dict = res.speed
            
            pred_boxes = []
            confidences = []
            for box in res.boxes:
                coords = box.xyxy[0].cpu().numpy()
                pred_boxes.append(coords)
                confidences.append(float(box.conf))
                
            # Analyze predictions for overlapping boxes (potential double detections)
            problematic_pairs = reporter.update(pred_boxes, confidences, (h, w), filename, speed_dict)
            
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

    if save_persistently:
        save_evaluation_results(exp_name, "inference")


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
    parser.add_argument('--source', type=str, default=None, help="Inference: Folder of new images")
    parser.add_argument('--labels', type=str, default=None, help="Real Audit: Folder of labels for the new images")
    parser.add_argument('--video', type=str, default=None, help="Video: Path to the MP4/AVI file") 
    parser.add_argument('--save', action='store_true', help="Persistently save this evaluation in the model's folder")
    parser.add_argument('--keep', action='store_true', help="Do not delete the output directory before running (useful to combine reports)")
    
    args = parser.parse_args()

    # Ask for Model Name (or use default)
    selected_model_path = select_model_path()

    # Mode Selector
    if args.video:
        run_video_mode(selected_model_path, args.video)
    elif args.source and args.labels:
        # REAL AUDIT MODE (Has images and has labels)
        run_audit_mode(selected_model_path, draw_all=args.draw_all, save_persistently=args.save, 
                       custom_img_dir=args.source, custom_lbl_dir=args.labels, keep=args.keep)
    elif args.source:
        # INFERENCE MODE (Has images)
        run_inference_mode(selected_model_path, args.source, save_persistently=args.save)
    else:
        # SYNTHETIC AUDIT MODE (Default, uses Isaac Sim test set)
        if args.labels:
            print("⚠️ Warning: --labels ignored because --source was not provided.")
        run_audit_mode(selected_model_path, draw_all=args.draw_all, save_persistently=args.save, keep=args.keep)