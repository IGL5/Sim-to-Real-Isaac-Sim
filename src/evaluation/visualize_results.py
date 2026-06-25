from pathlib import Path
import sys
import cv2
import shutil
import argparse
from ultralytics import YOLO
from src.core import config
from src.evaluation.utils import data_utils as du
from src.evaluation.utils import visual_utils as vu
from src.core.utils import project_utils as pu
from src.core.metadata.train_builder import TrainMetadata

# Import the class from the other file
try:
    from src.evaluation.utils.audit_reporter import ReportGenerator
    from src.evaluation.utils.inference_reporter import InferenceReportGenerator
except ImportError:
    print("❌ CRITICAL ERROR: Not finding 'audit_reporter.py' or 'inference_reporter.py'.")
    print("   Make sure both files are on 'modules' folder.")
    sys.exit(1)



def get_model_img_size(model_path):
    """
    Attempts to read the training image size (input size) from training_metadata.json or args.yaml.
    Falls back to 640 if not found.
    """
    exp_dir = Path(model_path).parent.parent
    
    # Try 1: training_metadata.json
    json_path = exp_dir / "metadata" / "training_metadata.json"
    if json_path.exists():
        try:
            meta = TrainMetadata(json_path)
            img_size = meta.get_img_size()
            if img_size:
                return int(img_size)
        except Exception as e:
            print(f"⚠️ Warning reading training_metadata.json: {e}")

    # Try 2: args.yaml
    yaml_path = exp_dir / "args.yaml"
    if yaml_path.exists():
        try:
            import yaml
            with open(yaml_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                imgsz = data.get("imgsz")
                if imgsz:
                    print(f"🔍 Found imgsz {imgsz} in args.yaml")
                    return int(imgsz)
        except Exception as e:
            print(f"⚠️ Warning reading args.yaml: {e}")

    print("⚠️ Could not determine model input size. Using default: 640")
    return 640


def resize_image_to_imgsz(img, imgsz):
    """
    Resizes an image so that its maximum dimension matches imgsz, keeping aspect ratio.
    """
    h, w = img.shape[:2]
    scale = imgsz / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA), scale


def check_system_integrity(model_path, check_dataset=False):
    """
    Checks that everything necessary exists before starting.
    """
    # 1. Check Model
    if not Path(model_path).exists():
        print(f"❌ ERROR: Not finding the model file in:")
        print(f"   -> {model_path}")
        print("   Did you run the training script (train_YOLO.py)?")
        return False

    # 2. Check Dataset (only if we are going to audit)
    if check_dataset:
        test_images_dir = Path(config.DATASET_TEST_IMAGES)
        if not test_images_dir.exists():
            print(f"❌ ERROR: Not finding the test images folder:")
            print(f"   -> {test_images_dir}")
            return False
        
        # Check if not empty
        if not any(test_images_dir.iterdir()):
            print(f"⚠️ WARNING: The test folder is empty ({test_images_dir}).")
            return False

    return True


def select_model_path(preselected_model=None):
    """
    Interactively selects the model path checking for available models.
    """

    if preselected_model:
        path = str(Path(config.PROJECT_DIR) / preselected_model / config.BEST_MODEL_SUBPATH)
        if Path(path).exists():
            print(f"🤖 Auto-selected model: {preselected_model}")
            return path
        else:
            print(f"❌ ERROR: Preselected model '{preselected_model}' not found.")
            sys.exit(1)

    print("\n--- 🤖 MODEL SELECTION ---")
    
    # Check if project directory exists
    project_dir = Path(config.PROJECT_DIR)
    if not project_dir.exists():
        print(f"❌ ERROR: Project directory '{project_dir}' not found.")
        print("   (You need to train a model first using train_YOLO.py)")
        sys.exit(1)
        
    available_models = pu.get_available_models()
    
    print("📂 Available trained models:")
    for i, m in enumerate(available_models):
        print(f"  [{i+1}] {m}")
        
    while True:
        user_input = input(f"\nSelect a model [1-{len(available_models)}] (default: {len(available_models)}): ").strip()
        
        if not user_input:
            exp_name = available_models[-1]
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
            
    path = str(Path(config.PROJECT_DIR) / exp_name / config.BEST_MODEL_SUBPATH)
    print(f"✅ Selected model: {exp_name}\n")
    return path


def save_evaluation_results(exp_name, mode):
    """ Copies the temporary audit_report into a persistent iteration folder """
    base_eval_dir = Path(config.PROJECT_DIR) / exp_name / config.SAVED_EVAL_FOLDER_NAME
    base_eval_dir.mkdir(parents=True, exist_ok=True)
    
    # Find the next iteration number
    existing_dirs = [d for d in base_eval_dir.iterdir() if d.is_dir()]
    iter_num = len(existing_dirs) + 1
    eval_dir = base_eval_dir / f"iter_{iter_num:03d}_{mode}"
    
    # Copy the entire workspace (HTML, Plots, JSON, Images)
    shutil.copytree(str(config.EVALUATION_OUTPUT_DIR), str(eval_dir))
    print(f"\n📦 Persistent Evaluation saved at: {eval_dir}")


def run_audit_mode(model_path, draw_all=False, save_persistently=False, custom_img_dir=None, custom_lbl_dir=None, keep=False, manual_class_map=None, save_resized=False, imgsz=640):
    """ Audit mode (Dataset Test with Labels) - Supports both Sim and Real """
    if not check_system_integrity(model_path, check_dataset=(custom_img_dir is None)):
        return

    is_real = custom_img_dir is not None and custom_lbl_dir is not None
    prefix = "real_audit" if is_real else "audit"

    print(f"--- 🕵️ STARTING {'REAL ' if is_real else ''}AUDIT (Draw All: {draw_all}) ---")
    
    # 1. Prepare folders according to the mode
    if not keep:
        eval_out = Path(config.EVALUATION_OUTPUT_DIR)
        if eval_out.exists(): shutil.rmtree(str(eval_out))
    
    path_fn = Path(config.EVALUATION_OUTPUT_DIR) / prefix / f"{prefix}_missed_FN"
    path_fp = Path(config.EVALUATION_OUTPUT_DIR) / prefix / f"{prefix}_invented_FP"
    path_poor = Path(config.EVALUATION_OUTPUT_DIR) / prefix / f"{prefix}_poor_bbox"
    path_all = Path(config.EVALUATION_OUTPUT_DIR) / prefix / f"{prefix}_all"

    if draw_all:
        path_all.mkdir(parents=True, exist_ok=True)
        short_all = str(path_all)[str(path_all).find("YOLO"):] if "YOLO" in str(path_all) else str(path_all)
        print(f"📂 Saving everything in: {short_all}")
    else:
        path_fn.mkdir(parents=True, exist_ok=True)
        path_fp.mkdir(parents=True, exist_ok=True)
        path_poor.mkdir(parents=True, exist_ok=True)

    # Logic for class translation and filtering
    dataset_classes = pu.get_project_classes(lowercase=True)
    if not dataset_classes: dataset_classes = ['bicycle']
        
    dataset_class_names = {i: name.capitalize() for i, name in enumerate(dataset_classes)}
    
    model = YOLO(model_path)
    
    if manual_class_map:
        model_to_dataset_map = manual_class_map
    else:
        model_to_dataset_map = {}
        for mod_idx, mod_name in model.names.items():
            if mod_name.lower() in dataset_classes:
                model_to_dataset_map[mod_idx] = dataset_classes.index(mod_name.lower())
    
    print(f"🧠 Active classes in dataset: {dataset_class_names}")

    reporter = ReportGenerator(config.IOU_THRESHOLD, prefix=prefix, user_conf_threshold=config.CONF_THRESHOLD, class_names=dataset_class_names)

    img_dir = Path(custom_img_dir) if is_real else Path(config.DATASET_TEST_IMAGES)
    lbl_dir = Path(custom_lbl_dir) if is_real else Path(config.DATASET_TEST_LABELS)
    
    image_files = [f for f in img_dir.iterdir() if f.is_file() and f.name.lower().endswith(config.VALID_IMAGE_EXTENSIONS)]
    print(f"📸 Processing {len(image_files)} images...")

    for i, img_path in enumerate(image_files):
        filename = img_path.name
        txt_path = lbl_dir / f"{img_path.stem}.txt"

        img = cv2.imread(str(img_path))
        if img is None: continue

        h, w, _ = img.shape
        if txt_path.exists():
            gt_boxes = du.parse_kitti_label(str(txt_path), w, h)
            # Only keep ground truth boxes whose class ID is in our active dataset classes
            gt_boxes = [box for box in gt_boxes if box[0] in dataset_class_names]
        else:
            gt_boxes = []

        # Inference (always get all detections for PR curve)
        results = model.predict(
            source=img, 
            conf=0.001, 
            verbose=False,
            project=str(config.EVALUATION_OUTPUT_DIR),
            name="yolo_temp",
            exist_ok=True
        )[0]
        speed_dict = results.speed
        
        pred_boxes = []
        confidences = []
        pred_classes = []
        
        for box in results.boxes:
            mod_cls = int(box.cls[0])
            if mod_cls in model_to_dataset_map:
                dataset_cls = model_to_dataset_map[mod_cls]
                coords = box.xyxy[0].cpu().numpy()
                pred_boxes.append(coords)
                confidences.append(float(box.conf))
                pred_classes.append(dataset_cls)

        img_stats = reporter.update(pred_boxes, pred_classes, gt_boxes, confidences, (h, w), speed_dict)

        valid_pred_boxes = []
        valid_confidences = []
        valid_classes = [] 
        for j, c in enumerate(confidences):
            if c >= config.CONF_THRESHOLD:
                valid_pred_boxes.append(pred_boxes[j])
                valid_confidences.append(c)
                valid_classes.append(pred_classes[j]) 

        if i < config.LIMIT_IMAGES_PER_VIS:
            has_errors = img_stats["FN"] > 0 or img_stats["poor_bbox"] > 0 or img_stats["FP"] > 0
            
            if draw_all or has_errors:
                if save_resized:
                    img_resized, scale = resize_image_to_imgsz(img, imgsz)
                    gt_boxes_drawn = [[box[0], box[1]*scale, box[2]*scale, box[3]*scale, box[4]*scale] for box in gt_boxes]
                    valid_pred_boxes_drawn = [box * scale for box in valid_pred_boxes]
                    img_to_draw = img_resized.copy()
                else:
                    gt_boxes_drawn = gt_boxes
                    valid_pred_boxes_drawn = valid_pred_boxes
                    img_to_draw = img.copy()

                # Draw using translated class names from our dataset
                img_drawn = vu.draw_boxes(img_to_draw, gt_boxes_drawn, color=(0, 255, 0), class_names=dataset_class_names)
                img_drawn = vu.draw_boxes(img_drawn, valid_pred_boxes_drawn, color=(255, 0, 0), confidences=valid_confidences, classes=valid_classes, class_names=dataset_class_names)
                
                if draw_all:
                    status = "OK"
                    if img_stats["FN"] > 0: status = "FN"
                    elif img_stats["poor_bbox"] > 0: status = "POOR"
                    elif img_stats["FP"] > 0: status = "FP"
                    
                    save_path = path_all / f"{status}_{filename}"
                    cv2.imwrite(str(save_path), img_drawn)
                    
                else:
                    if img_stats["FN"] > 0:
                        cv2.imwrite(str(path_fn / filename), img_drawn)
                    if img_stats["poor_bbox"] > 0:
                        cv2.imwrite(str(path_poor / filename), img_drawn)
                    if img_stats["FP"] > 0:
                        cv2.imwrite(str(path_fp / filename), img_drawn)

    exp_name = Path(model_path).parent.parent.name
    reporter.generate_plots()
    reporter.generate_html_report(exp_name)

    if save_persistently:
        save_evaluation_results(exp_name, prefix)


def run_inference_mode(model_path, source_folder, save_persistently=False, keep=False, manual_class_map=None, save_resized=False, imgsz=640):
    """ Inference Mode (New images without labels) """
    if not check_system_integrity(model_path, check_dataset=False): return
    if not Path(source_folder).exists(): return

    print(f"--- 🌍 REAL INFERENCE MODE ---")
    if not keep:
        eval_out = Path(config.EVALUATION_OUTPUT_DIR)
        if eval_out.exists(): shutil.rmtree(str(eval_out))
    
    images_output_dir = Path(config.EVALUATION_OUTPUT_DIR) / "inference_real"
    images_output_dir.mkdir(parents=True, exist_ok=True)

    # Logic for class translation and filtering
    dataset_classes = pu.get_project_classes(lowercase=True)
    if not dataset_classes: dataset_classes = ['bicycle']
        
    dataset_class_names = {i: name.capitalize() for i, name in enumerate(dataset_classes)}
    
    model = YOLO(model_path)

    if manual_class_map:
        model_to_dataset_map = manual_class_map
    else:
        model_to_dataset_map = {}
        for mod_idx, mod_name in model.names.items():
            if mod_name.lower() in dataset_classes:
                model_to_dataset_map[mod_idx] = dataset_classes.index(mod_name.lower())
    
    reporter = InferenceReportGenerator(conf_threshold=config.CONF_THRESHOLD, overlap_threshold=config.OVERLAP_THRESHOLD_ANALYSIS, class_names=dataset_class_names)
    overlaps_dir_path = Path(reporter.plots_dir) / "suspicious_overlaps"
    overlaps_dir_path.mkdir(parents=True, exist_ok=True)
    
    source_dir = Path(source_folder)
    image_files = [f for f in source_dir.iterdir() if f.is_file() and f.name.lower().endswith(config.VALID_IMAGE_EXTENSIONS)]
    print(f"📸 Processing {len(image_files)} images...")
    
    for i, img_path in enumerate(image_files):
        filename = img_path.name
         
        try:
            img_orig = cv2.imread(str(img_path))
            if img_orig is None: continue
            h, w, _ = img_orig.shape
 
            # Inference (low threshold to capture below-threshold detections for the plot)
            res = model.predict(
                source=img_path, 
                conf=min(0.01, config.CONF_THRESHOLD), 
                verbose=False,
                project=str(config.EVALUATION_OUTPUT_DIR),
                name="yolo_temp",
                exist_ok=True
            )[0]
            speed_dict = res.speed
            
            pred_boxes, confidences, pred_classes = [], [], []
            for box in res.boxes:
                mod_cls = int(box.cls[0])
                if mod_cls in model_to_dataset_map:
                    dataset_cls = model_to_dataset_map[mod_cls]
                    pred_boxes.append(box.xyxy[0].cpu().numpy())
                    confidences.append(float(box.conf))
                    pred_classes.append(dataset_cls)
                
            # Pass the classes to the reporter so it analyzes INTRA-CLASS overlaps
            problematic_pairs = reporter.update(pred_boxes, pred_classes, confidences, (h, w), filename, speed_dict)
            
            if i < config.LIMIT_IMAGES_PER_VIS: 
                # Filter valid predictions above threshold for drawing
                valid_pred_boxes = []
                valid_confidences = []
                valid_pred_classes = []
                for idx, conf in enumerate(confidences):
                    if conf >= config.CONF_THRESHOLD:
                        valid_pred_boxes.append(pred_boxes[idx])
                        valid_confidences.append(conf)
                        valid_pred_classes.append(pred_classes[idx])
                
                if save_resized:
                    img_resized, scale = resize_image_to_imgsz(img_orig, imgsz)
                    valid_pred_boxes_drawn = [box * scale for box in valid_pred_boxes]
                    img_to_draw = img_resized.copy()
                else:
                    valid_pred_boxes_drawn = valid_pred_boxes
                    img_to_draw = img_orig.copy()

                if problematic_pairs:
                    img_overlap = vu.draw_overlapping_pairs(img_to_draw.copy(), valid_pred_boxes_drawn, problematic_pairs, valid_confidences)
                    cv2.imwrite(str(overlaps_dir_path / f"OVERLAP_{filename}"), img_overlap)
                
                # Draw with our own colors and translated labels
                img_drawn = vu.draw_boxes(img_to_draw.copy(), valid_pred_boxes_drawn, color=(255, 0, 0), confidences=valid_confidences, classes=valid_pred_classes, class_names=dataset_class_names)
                cv2.imwrite(str(images_output_dir / f"PRED_{filename}"), img_drawn)
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            
    exp_name = Path(model_path).parent.parent.name
    reporter.generate_plots()
    reporter.generate_html_report(exp_name)

    if save_persistently: save_evaluation_results(exp_name, "inference")


def run_video_mode(model_path, video_path):
    """ Video Mode """
    if not check_system_integrity(model_path, check_dataset=False):
        return

    if not Path(video_path).exists():
        print(f"❌ ERROR: Not finding the video file: {video_path}")
        return

    print(f"--- 🎥 VIDEO MODE ---")
    print(f"Processing: {video_path}")
    print("This may take a while depending on the duration...")

    model = YOLO(model_path)
    
    # save=True makes YOLO automatically save the video in runs/detect/predict...
    # stream=True saves memory on long videos
    video_out_dir = Path(config.EVALUATION_OUTPUT_DIR) / "video_output"
    
    if "coco" in model_path.lower():
        results = model.predict(
            source=video_path, save=True, conf=config.CONF_THRESHOLD, stream=True, classes=[1],
            project=str(config.EVALUATION_OUTPUT_DIR), name="video_output", exist_ok=True
        )
    else:
        results = model.predict(
            source=video_path, save=True, conf=config.CONF_THRESHOLD, stream=True,
            project=str(config.EVALUATION_OUTPUT_DIR), name="video_output", exist_ok=True
        )
    
    # We need to iterate over the generator to process the video
    for r in results:
        pass 

    print("\n✅ Video processed correctly.")
    print(f"📂 Look for it in the folder: {video_out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO Diagnostic Tool")
    
    # Options
    parser.add_argument('--draw_all', action='store_true', help="Audit: Save ALL images (hits and misses)")
    parser.add_argument('--source', type=str, default=None, help="Inference: Folder of new images")
    parser.add_argument('--labels', type=str, default=None, help="Real Audit: Folder of labels for the new images")
    parser.add_argument('--video', type=str, default=None, help="Video: Path to the MP4/AVI file") 
    parser.add_argument('--save', action='store_true', help="Persistently save this evaluation in the model's folder")
    parser.add_argument('--keep', action='store_true', help="Do not delete the output directory before running (useful to combine reports)")
    parser.add_argument('--conf', type=float, default=None, help="Confidence threshold for detection")
    parser.add_argument('--model', type=str, default=None, help="Bypass interactive menu and specify model name directly")
    parser.add_argument('--class_map', type=str, default=None, help="Manual class mapping. Format: model_class_number:dataset_class_number")
    parser.add_argument('--save_resized', action='store_true', help="Save visual output images resized to model training input size")
    
    args = parser.parse_args()

    class_map = pu.parse_class_map(args.class_map)

    if args.conf is not None:
        config.CONF_THRESHOLD = args.conf
        print(f"\nConfidence threshold set to: {config.CONF_THRESHOLD}")

    # Ask for Model Name (or use default)
    selected_model_path = select_model_path(args.model)

    # Determine the model input size if we want to save resized images
    model_imgsz = 640
    if args.save_resized:
        model_imgsz = get_model_img_size(selected_model_path)

    # Mode Selector
    if args.video:
        run_video_mode(selected_model_path, args.video)
    elif args.source and args.labels:
        # REAL AUDIT MODE (Has images and has labels)
        run_audit_mode(selected_model_path, draw_all=args.draw_all, save_persistently=args.save, 
                       custom_img_dir=args.source, custom_lbl_dir=args.labels, keep=args.keep, manual_class_map=class_map,
                       save_resized=args.save_resized, imgsz=model_imgsz)
    elif args.source:
        # INFERENCE MODE (Has images)
        run_inference_mode(selected_model_path, args.source, save_persistently=args.save, keep=args.keep, manual_class_map=class_map,
                           save_resized=args.save_resized, imgsz=model_imgsz)
    else:
        # SYNTHETIC AUDIT MODE (Default, uses Isaac Sim test set)
        if args.labels:
            print("⚠️ Warning: --labels ignored because --source was not provided.")
        run_audit_mode(selected_model_path, draw_all=args.draw_all, save_persistently=args.save, keep=args.keep, manual_class_map=class_map,
                       save_resized=args.save_resized, imgsz=model_imgsz)