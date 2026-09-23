import time
import shutil
import argparse
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

from src.core import config
from src.core.utils import math_utils as mu
from src.core.utils import project_utils as pu
from src.evaluation.utils.tiled_reporter import TiledReporter

# Color palette for classes (BGR)
PALETTE = [
    (0, 215, 255),   # Gold / Yellow
    (0, 165, 255),   # Orange
    (50, 205, 50),   # Lime Green
    (255, 105, 180), # Hot Pink
    (0, 255, 255),   # Cyan
    (238, 130, 238), # Violet
    (30, 144, 255),  # Dodger Blue
]


def select_model_path(preselected_model=None):
    """Interactively selects a trained model or validates the preselected one."""
    if preselected_model:
        # Check if direct file path or experiment name
        if Path(preselected_model).exists():
            print(f"🤖 Model loaded directly: {preselected_model}")
            return str(Path(preselected_model))
        
        path = Path(config.PROJECT_DIR) / preselected_model / config.BEST_MODEL_SUBPATH
        if path.exists():
            print(f"🤖 Auto-selected model: {preselected_model}")
            return str(path)
        else:
            print(f"❌ ERROR: Preselected model '{preselected_model}' not found in {config.PROJECT_DIR}.")
            exit(1)

    print("\n--- 🤖 MODEL SELECTION FOR TILED INFERENCE ---")
    project_dir = Path(config.PROJECT_DIR)
    if not project_dir.exists():
        print(f"❌ ERROR: Project directory '{project_dir}' not found.")
        exit(1)

    available_models = pu.get_available_models()
    if not available_models:
        print(f"❌ ERROR: No trained models with '{config.BEST_MODEL_SUBPATH}' found in {project_dir}.")
        exit(1)

    print("📂 Available models:")
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
        elif user_input in available_models:
            exp_name = user_input
            break
        print("  ⚠️  Invalid input. Please enter a valid number from the list.")

    path = Path(config.PROJECT_DIR) / exp_name / config.BEST_MODEL_SUBPATH
    print(f"✅ Selected model: {exp_name}\n")
    return str(path)


def draw_high_res_annotations(img, boxes, scores, classes, class_names):
    """
    Renders boxes and labels on high-resolution images with adaptive line thickness and font scale.
    """
    annotated = img.copy()
    h, w = img.shape[:2]
    
    # Adaptive scaling based on image resolution (baseline: 1000px)
    scale_factor = max(1.0, max(w, h) / 1000.0)
    line_thick = max(2, int(scale_factor * 1.5))
    font_scale = max(0.5, scale_factor * 0.45)
    font_thick = max(1, int(scale_factor * 0.9))

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        c_id = int(classes[i])
        conf = float(scores[i])
        c_name = class_names.get(c_id, f"Class {c_id}")
        color = PALETTE[c_id % len(PALETTE)]

        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, line_thick)

        # Label text with background box
        label_text = f"{c_name} {conf:.2f}"
        (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)

        bg_y1 = max(0, y1 - th - baseline - 6)
        bg_y2 = y1
        bg_x2 = min(w, x1 + tw + 8)

        if y1 < th + baseline + 10:
            bg_y1 = y1
            bg_y2 = y1 + th + baseline + 6
            text_pos = (x1 + 4, y1 + th + 2)
        else:
            text_pos = (x1 + 4, y1 - baseline - 4)

        cv2.rectangle(annotated, (x1, bg_y1), (bg_x2, bg_y2), color, -1)
        cv2.putText(annotated, label_text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thick, cv2.LINE_AA)

    return annotated


def draw_grid_preview(img, tile_coords):
    """Draws the grid slicing lines over a thumbnail/copy of the image for visual verification."""
    grid_img = img.copy()
    h, w = img.shape[:2]
    scale_factor = max(1.0, max(w, h) / 1000.0)
    line_thick = max(1, int(scale_factor * 1.0))
    font_scale = max(0.4, scale_factor * 0.35)

    for idx, (x1, y1, x2, y2) in enumerate(tile_coords):
        cv2.rectangle(grid_img, (x1, y1), (x2, y2), (0, 255, 0), line_thick)
        cv2.putText(grid_img, f"#{idx+1}", (x1 + 10, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), line_thick, cv2.LINE_AA)

    return grid_img


def save_crops(img, boxes, classes, class_names, filename, crops_dir):
    """Saves individual crops of detected objects."""
    crops_dir.mkdir(parents=True, exist_ok=True)
    h, w = img.shape[:2]
    stem = Path(filename).stem

    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = map(int, b)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 > x1 and y2 > y1:
            crop = img[y1:y2, x1:x2]
            c_name = class_names.get(int(classes[i]), f"class_{classes[i]}").replace(" ", "_")
            crop_path = crops_dir / f"{stem}_{i:03d}_{c_name}.jpg"
            cv2.imwrite(str(crop_path), crop)


def run_tiled_inference(source, model_path, tile_size=640, overlap=0.2,
                        conf_thresh=0.4, iou_thresh=0.5, fusion_method='wbf',
                        batch_size=8, class_map=None, save_crops_flag=False,
                        save_grid_flag=False, save_persistently=False):
    """
    Main execution loop for Tiled / Sliced Inference.
    """
    source_path = Path(source)
    if not source_path.exists():
        print(f"❌ ERROR: Source path does not exist: {source_path}")
        return

    # Collect images
    if source_path.is_file():
        image_files = [source_path]
    else:
        image_files = [f for f in source_path.iterdir() if f.is_file() and f.name.lower().endswith(config.VALID_IMAGE_EXTENSIONS)]

    if not image_files:
        print(f"⚠️ No valid images found in {source_path}")
        return

    print(f"\n🚀 STARTING TILED INFERENCE PIPELINE")
    print(f" • Model:       {model_path}")
    print(f" • Images:      {len(image_files)} from {source_path}")
    print(f" • Tile Size:   {tile_size}x{tile_size}")
    print(f" • Overlap:     {int(overlap * 100)}%")
    print(f" • Fusion:      {fusion_method.upper()} (IoU: {iou_thresh})")
    print(f" • Confidence:  {conf_thresh}")
    print(f" • Batch Size:  {batch_size}")
    print("=" * 79)

    # Clean temporary output directory
    eval_out = Path(config.TILED_OUTPUT_DIR)
    if eval_out.exists():
        shutil.rmtree(str(eval_out), ignore_errors=True)
    eval_out.mkdir(parents=True, exist_ok=True)

    # Class mappings
    dataset_classes = pu.get_project_classes(lowercase=True)
    if not dataset_classes:
        dataset_classes = ['bicycle']
    dataset_class_names = {i: name.capitalize() for i, name in enumerate(dataset_classes)}

    model = YOLO(model_path)

    if class_map:
        model_to_dataset = class_map
    else:
        model_to_dataset = {}
        for mod_idx, mod_name in model.names.items():
            if mod_name.lower() in dataset_classes:
                model_to_dataset[mod_idx] = dataset_classes.index(mod_name.lower())
        
        # If no classes matched classes.txt, fallback to the model's native classes
        if not model_to_dataset:
            print(f"ℹ️ Model classes {model.names} did not match classes.txt. Using model's own classes directly.")
            model_to_dataset = {int(k): int(k) for k in model.names.keys()}
            dataset_class_names = {int(k): str(v).capitalize() for k, v in model.names.items()}

    reporter = TiledReporter(
        output_dir=config.TILED_OUTPUT_DIR,
        conf_threshold=conf_thresh,
        iou_threshold=iou_thresh,
        tile_size=tile_size,
        overlap_ratio=overlap,
        fusion_method=fusion_method,
        class_names=dataset_class_names
    )

    for img_idx, img_file in enumerate(image_files, 1):
        filename = img_file.name
        img = cv2.imread(str(img_file))
        if img is None:
            print(f"⚠️ Could not read image: {img_file}")
            continue

        img_h, img_w = img.shape[:2]

        # 1. Slicing & Grid Calculation
        t_start = time.perf_counter()
        t_slice_start = time.perf_counter()
        tile_coords = mu.calculate_tile_grid(img_w, img_h, tile_size=tile_size, overlap_ratio=overlap)
        num_tiles = len(tile_coords)
        t_slice_ms = (time.perf_counter() - t_slice_start) * 1000.0

        # Save grid debug if requested
        if save_grid_flag:
            grid_img = draw_grid_preview(img, tile_coords)
            reporter.grid_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(reporter.grid_dir / f"GRID_{filename}"), grid_img)

        # 2. Batch Tile Inference
        t_inf_start = time.perf_counter()
        raw_boxes = []
        raw_scores = []
        raw_classes = []

        # Process in batches
        for b_start in range(0, num_tiles, batch_size):
            b_coords = tile_coords[b_start:b_start + batch_size]
            b_slices = [img[y1:y2, x1:x2] for (x1, y1, x2, y2) in b_coords]

            batch_results = model.predict(
                source=b_slices,
                conf=conf_thresh,
                verbose=False,
                project=str(config.TILED_OUTPUT_DIR),
                name="yolo_temp",
                exist_ok=True
            )

            for tile_i, res in enumerate(batch_results):
                off_x1, off_y1, _, _ = b_coords[tile_i]

                for box in res.boxes:
                    mod_cls = int(box.cls[0])
                    if mod_cls in model_to_dataset:
                        ds_cls = model_to_dataset[mod_cls]
                        c_score = float(box.conf[0])
                        # Local tile coords [x1, y1, x2, y2]
                        lx1, ly1, lx2, ly2 = box.xyxy[0].cpu().numpy()
                        # Map to global image coordinates
                        gx1 = lx1 + off_x1
                        gy1 = ly1 + off_y1
                        gx2 = lx2 + off_x1
                        gy2 = ly2 + off_y1

                        raw_boxes.append([gx1, gy1, gx2, gy2])
                        raw_scores.append(c_score)
                        raw_classes.append(ds_cls)

        t_inf_ms = (time.perf_counter() - t_inf_start) * 1000.0

        # 3. Fusion & Reconciliation of Overlaps
        t_fus_start = time.perf_counter()
        if len(raw_boxes) > 0:
            if fusion_method == 'nms':
                fused_boxes, fused_scores, fused_classes = mu.nms_boxes(
                    raw_boxes, raw_scores, raw_classes, iou_thresh=iou_thresh
                )
            else:
                fused_boxes, fused_scores, fused_classes = mu.weighted_boxes_fusion(
                    raw_boxes, raw_scores, raw_classes, iou_thresh=iou_thresh, conf_mode="mean"
                )
        else:
            fused_boxes = np.empty((0, 4), dtype=np.float32)
            fused_scores = np.empty((0,), dtype=np.float32)
            fused_classes = np.empty((0,), dtype=int)

        t_fus_ms = (time.perf_counter() - t_fus_start) * 1000.0
        t_total_ms = (time.perf_counter() - t_start) * 1000.0

        timings = {
            "slicing": t_slice_ms,
            "inference": t_inf_ms,
            "fusion": t_fus_ms,
            "total": t_total_ms
        }

        # 4. Save High-Resolution Annotated Visual Output
        annotated_img = draw_high_res_annotations(img, fused_boxes, fused_scores, fused_classes, dataset_class_names)
        evidence_name = f"PRED_{filename}"
        cv2.imwrite(str(reporter.images_dir / evidence_name), annotated_img)

        if save_crops_flag and len(fused_boxes) > 0:
            save_crops(img, fused_boxes, fused_classes, dataset_class_names, filename, reporter.crops_dir)

        # 5. Record Stats & Print Terminal Summary
        reporter.record_image(
            filename=filename,
            img_w=img_w,
            img_h=img_h,
            num_tiles=num_tiles,
            boxes=fused_boxes,
            scores=fused_scores,
            classes=fused_classes,
            timings=timings,
            evidence_filename=evidence_name
        )

        reporter.print_image_summary(
            filename=filename,
            img_w=img_w,
            img_h=img_h,
            num_tiles=num_tiles,
            boxes=fused_boxes,
            scores=fused_scores,
            classes=fused_classes,
            timings=timings
        )

    # Final Global Summary in Terminal
    reporter.print_global_summary()

    # Generate HTML report (always generated as standard)
    exp_name = Path(model_path).parent.parent.name
    report_path = reporter.generate_html_report(model_path=model_path, experiment_name=exp_name)
    print(f"✨ Interactive HTML Report available at:\n   -> {report_path}")

    # Save persistently if requested
    if save_persistently:
        base_eval_dir = Path(config.PROJECT_DIR) / exp_name / config.SAVED_EVAL_FOLDER_NAME
        base_eval_dir.mkdir(parents=True, exist_ok=True)
        existing_dirs = [d for d in base_eval_dir.iterdir() if d.is_dir()]
        iter_num = len(existing_dirs) + 1
        persistent_dir = base_eval_dir / f"iter_{iter_num:03d}_tiled_inference"
        shutil.copytree(str(config.TILED_OUTPUT_DIR), str(persistent_dir))
        print(f"\n📦 Persistent Evaluation saved at:\n   -> {persistent_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tiled / Sliced Inference for High-Resolution Real Images")
    parser.add_argument('--source', type=str, default="data/grapas/images",
                        help="Path to image folder or single image file (default: data/grapas/images)")
    parser.add_argument('--model', type=str, default=None,
                        help="Model experiment name or path to .pt weights (if omitted, opens interactive menu)")
    parser.add_argument('--tile_size', type=int, default=640,
                        help="Size of square tiles in pixels (default: 640)")
    parser.add_argument('--overlap', type=float, default=0.2,
                        help="Overlap ratio between adjacent tiles [0.0 - 0.9] (default: 0.2 = 20%%)")
    parser.add_argument('--conf', type=float, default=None,
                        help=f"Confidence threshold (default: {config.CONF_THRESHOLD})")
    parser.add_argument('--iou', type=float, default=None,
                        help=f"IoU threshold for box fusion/NMS (default: {config.IOU_THRESHOLD})")
    parser.add_argument('--fusion', type=str, choices=['wbf', 'nms'], default='wbf',
                        help="Box reconciliation algorithm: 'wbf' (Weighted Boxes Fusion) or 'nms' (default: wbf)")
    parser.add_argument('--batch_size', type=int, default=8,
                        help="Batch size for concurrent tile inference on GPU (default: 8)")
    parser.add_argument('--class_map', type=str, default=None,
                        help="Manual class mapping string, e.g. '0:1,1:0'")
    parser.add_argument('--save_grid', action='store_true',
                        help="Save debug images showing the tile grid slicing lines")
    parser.add_argument('--save_crops', action='store_true',
                        help="Save cropped images of each detected object")
    parser.add_argument('--save', action='store_true',
                        help="Persistently save results into model's evaluations folder")

    args = parser.parse_args()

    # Resolve model path
    selected_model = select_model_path(args.model)

    # Resolve thresholds
    conf_threshold = args.conf if args.conf is not None else config.CONF_THRESHOLD
    iou_threshold = args.iou if args.iou is not None else config.IOU_THRESHOLD

    parsed_class_map = pu.parse_class_map(args.class_map)

    run_tiled_inference(
        source=args.source,
        model_path=selected_model,
        tile_size=args.tile_size,
        overlap=args.overlap,
        conf_thresh=conf_threshold,
        iou_thresh=iou_threshold,
        fusion_method=args.fusion,
        batch_size=args.batch_size,
        class_map=parsed_class_map,
        save_crops_flag=args.save_crops,
        save_grid_flag=args.save_grid,
        save_persistently=args.save
    )

