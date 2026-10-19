import argparse
import shutil
import sys
from pathlib import Path
import cv2

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
if hasattr(sys.stderr, 'reconfigure'):
    try:
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

from src.core import config
from src.core.metadata.clean_builder import CleaningMetadata
from src.data_prep.utils.clean_fs import (
    is_image_corrupted,
    is_label_empty,
    discover_image_folders,
    resolve_label_path,
    resolve_valid_raw_mask_folder,
    handle_action,
    save_review_image
)
from src.data_prep.utils.clean_analyzer import (
    detect_dataset_format,
    load_filter_file,
    analyze_labels,
    apply_label_cleaning
)


# ==========================================
# 1. CORE LOGIC / ORCHESTRATOR
# ==========================================
def clean_dataset(args):
    """ Orchestrator function that manages the dataset iteration. """
    dir_path = Path(args.dir)
    image_folders = discover_image_folders(dir_path)
    
    if not image_folders:
        print(f"[ERROR] No image folders (rgb/ or images/) found inside: {args.dir}")
        return

    # Infer or lock format globally for the dataset once before the loop
    dataset_format = detect_dataset_format(image_folders, forced_format=args.format)
    mode_desc = "forced via CLI" if args.format != 'auto' else "auto-detected at startup"
    print(f"📋 Dataset Label Format: {dataset_format.upper()} ({mode_desc})")

    # Resolve filter lists (whitelist and blacklist)
    whitelist_data = {}
    blacklist_data = {}
    filtering_mode = 'whitelist' # default policy
    filter_data = {}

    bl_explicit = getattr(args, 'blacklist', None) is not None
    wl_explicit = getattr(args, 'whitelist', None) is not None

    bl_path = Path(args.blacklist) if bl_explicit else (dir_path / "blacklist.txt")
    wl_path = Path(args.whitelist) if wl_explicit else (dir_path / "whitelist.txt")

    if bl_path.exists():
        blacklist_data = load_filter_file(bl_path, list_type='blacklist')
    if wl_path.exists():
        whitelist_data = load_filter_file(wl_path, list_type='whitelist')

    if bl_explicit and not wl_explicit:
        filtering_mode = 'blacklist'
        filter_data = blacklist_data
        print(f"🚫 Active Filtering Mode: BLACKLIST ({len(blacklist_data)} frame rules loaded from {bl_path.name}) [CLI priority]")
        print("   -> ONLY annotations listed in blacklist will be deleted. All others will be kept.")
    elif wl_explicit and not bl_explicit:
        filtering_mode = 'whitelist'
        filter_data = whitelist_data
        print(f"🛡️  Active Filtering Mode: WHITELIST ({len(whitelist_data)} frame rules loaded from {wl_path.name}) [CLI priority]")
        print("   -> Suspicious annotations will be deleted UNLESS protected by whitelist.")
    elif bl_path.exists():
        filtering_mode = 'blacklist'
        filter_data = blacklist_data
        print(f"🚫 Active Filtering Mode: BLACKLIST ({len(blacklist_data)} frame rules loaded from {bl_path.name})")
        print("   -> ONLY annotations listed in blacklist will be deleted. All others will be kept.")
        if wl_path.exists() and len(whitelist_data) > 0:
            print(f"⚠️  Both '{bl_path.name}' and '{wl_path.name}' detected in {dir_path.name}.")
            print("   Blacklist takes precedence: whitelist rules are IGNORED.")
    elif wl_path.exists() and len(whitelist_data) > 0:
        filtering_mode = 'whitelist'
        filter_data = whitelist_data
        print(f"🛡️  Active Filtering Mode: WHITELIST ({len(whitelist_data)} frame rules loaded from {wl_path.name})")
        print("   -> Suspicious annotations will be deleted UNLESS protected by whitelist.")
    else:
        filtering_mode = 'whitelist'
        filter_data = {}
        if args.clean_labels or args.review:
            print("🛡️  Active Filtering Mode: DEFAULT WHITELIST (No whitelist/blacklist file found).")
            if args.clean_labels:
                print("   -> All suspicious annotations will be cleaned.")

    args.filtering_mode = filtering_mode

    if args.review:
        review_dir = dir_path / "_review_occlusions"
        if review_dir.exists():
            shutil.rmtree(review_dir)
        review_dir.mkdir(parents=True, exist_ok=True)
        print("🧹 Cleared previous review folder.")

    print(f"--- Starting processing in {len(image_folders)} folder(s) ---")
    
    stats = {
        "corrupted": 0, "empty": 0, "kept": 0, "reviewed": 0, 
        "labels_removed": 0, "labels_saved": 0, "total_labels": 0,
        "reason_occ": 0, "reason_trunc": 0, "reason_area": 0,
        "reason_edge": 0, "reason_giant": 0, "reason_degen": 0,
        "reason_manual": 0, "filtering_mode": filtering_mode
    }
    remove_empty_logic = args.empty or args.move_empty

    for img_folder in image_folders:
        camera_root = img_folder.parent 
        folder_display_name = f"{camera_root.name}/{img_folder.name}" if camera_root != dir_path else img_folder.name
        
        print(f"\nProcessing folder: {folder_display_name}")
        image_files = []
        for ext in config.VALID_IMAGE_EXTENSIONS:
            image_files.extend(img_folder.glob(f"*{ext}"))
        image_files = sorted(image_files)
        
        if not image_files:
            continue
        print(f"  -> Analyzing {len(image_files)} images...")

        # Resolve valid raw segmentation mask folder if segmentation mode requested
        raw_mask_folder = None
        if getattr(args, 'segmentation', False):
            raw_mask_folder = resolve_valid_raw_mask_folder(camera_root)
            if raw_mask_folder:
                print(f"  🎨 Using valid masks from: {raw_mask_folder.name}")
            else:
                print(f"  ⚠️  Flag --seg active, but no valid non-zero mask folder found in '{camera_root.name}'. Falling back to bounding box area.")

        for img_path in image_files:
            process_frame(
                img_path, camera_root, folder_display_name, args, stats,
                remove_empty_logic, filter_data, dataset_format,
                raw_mask_folder=raw_mask_folder, filtering_mode=filtering_mode
            )

    print_summary(stats, args)
    if not args.dry:
        meta_path = dir_path / config.FILE_GEN_META
        # Only record metadata if this directory has a generation_metadata.json (typically raw synthetic data)
        if meta_path.exists() or dir_path == config.RAW_DATA_DIR:
            meta_manager = CleaningMetadata(str(meta_path))
            meta_manager.record_cleaning_stats(
                stats=stats, 
                move_empty_flag=args.move_empty, 
                clean_labels_flag=args.clean_labels
            )
            meta_manager.set_timestamp(key_name=config.UPDATE_TIMESTAMP_KEY)
            meta_manager.commit()


# ==========================================
# 2. FRAME PROCESSING FLOW
# ==========================================
def process_frame(img_path, camera_root, folder_display_name, args, stats, remove_empty_logic, filter_data,
                  dataset_format, raw_mask_folder=None, filtering_mode='whitelist'):
    """ Applies the pipeline of checks to a single frame. """
    frame_id = img_path.stem
    txt_path = resolve_label_path(img_path, camera_root)
    
    img = cv2.imread(str(img_path))

    # CHECK 1: Corrupted or Flat Image
    if is_image_corrupted(img, args.thresh_mean, args.thresh_std):
        handle_action(img_path, camera_root, "delete", args.dry, stats, "corrupted", frame_id, txt_path=txt_path)
        return

    edge_ar_thresh = getattr(args, 'edge_ar_thresh', 3.0)
    area_metric = getattr(args, 'area_metric', 'polygon')

    # Load segmentation mask image if segmentation mode is requested and valid mask folder exists
    seg_mask = None
    if raw_mask_folder is not None:
        mask_file = raw_mask_folder / f"{frame_id}.png"
        if mask_file.exists():
            seg_mask = cv2.imread(str(mask_file), cv2.IMREAD_UNCHANGED)

    # CHECK 2: Clean Labels (Removes bad annotations directly from the .txt)
    if args.clean_labels and txt_path.exists():
        img_h, img_w = img.shape[:2]
        res = apply_label_cleaning(
            txt_path, frame_id, args.area_thresh, args.max_area_ratio,
            filter_data, args.dry, img_w, img_h, edge_ar_thresh,
            format_mode=dataset_format, area_metric=area_metric, seg_mask=seg_mask,
            filtering_mode=filtering_mode
        )
        stats["labels_removed"] += res["removed"]
        stats["labels_saved"] += res["saved"]
        stats["total_labels"] += res["total"]
        stats["reason_occ"] += res["occ"]
        stats["reason_trunc"] += res["trunc"]
        stats["reason_area"] += res["area"]
        stats["reason_edge"] += res["edge"]
        stats["reason_giant"] += res["giant"]
        stats["reason_degen"] += res["degen"]
        stats["reason_manual"] += res.get("manual", 0)

        if args.dry and res["removed"] > 0:
            print(f"  [DRY-RUN] Frame {frame_id}: {res['removed']} bad annotations would be removed.")

    # CHECK 3: Empty Labels (Backgrounds) - Evaluated AFTER cleaning labels!
    if remove_empty_logic and is_label_empty(txt_path):
        action = "move" if args.move_empty else "delete"
        target_dir = f"{args.dir}_empty" if args.move_empty else None
        handle_action(img_path, camera_root, action, args.dry, stats, "empty", frame_id, target_dir, txt_path=txt_path)
        return

    # CHECK 4: Review Mode (Suspicious labels visualization)
    if args.review:
        is_suspicious, annotated_img, frame_reasons = analyze_labels(
            img, txt_path, args.area_thresh, args.max_area_ratio,
            filter_data, frame_id, edge_ar_thresh,
            format_mode=dataset_format, area_metric=area_metric, seg_mask=seg_mask,
            return_reasons=True, filtering_mode=filtering_mode
        )
        if is_suspicious:
            safe_cam_name = folder_display_name.replace("/", "_").replace("\\", "_")
            save_review_image(annotated_img, args.dir, safe_cam_name, frame_id)
            stats["reviewed"] += 1
            if not args.clean_labels:
                for r in frame_reasons:
                    k = f"reason_{r}"
                    if k in stats:
                        stats[k] += 1

    stats["kept"] += 1


# ==========================================
# 3. METADATA & UI MODULES
# ==========================================
def print_summary(stats, args):
    print("\n--- FINAL SUMMARY ---")
    print(f"✅ Valid frames kept: {stats['kept']}")
    print(f"🗑️ Corrupted frames deleted: {stats['corrupted']}")
    if args.empty or args.move_empty or stats['empty'] > 0:
        print(f"📦 Empty frames handled: {stats['empty']}")

    filtering_mode = getattr(args, 'filtering_mode', 'whitelist')

    if args.clean_labels:
        mode_label = "BLACKLIST" if filtering_mode == 'blacklist' else "WHITELIST"
        print(f"✂️  Bad annotations removed from label files ({mode_label} mode): {stats['labels_removed']}")
        if stats['labels_removed'] > 0:
            details = []
            if stats['reason_area'] > 0: details.append(f"Small Area: {stats['reason_area']}")
            if stats['reason_edge'] > 0: details.append(f"Edge Slivers: {stats['reason_edge']}")
            if stats['reason_giant'] > 0: details.append(f"Giant: {stats['reason_giant']}")
            if stats['reason_occ'] > 0: details.append(f"Occluded: {stats['reason_occ']}")
            if stats['reason_trunc'] > 0: details.append(f"Truncated: {stats['reason_trunc']}")
            if stats['reason_degen'] > 0: details.append(f"Degenerate: {stats['reason_degen']}")
            if stats.get('reason_manual', 0) > 0: details.append(f"Manual/Blacklist: {stats['reason_manual']}")
            print(f"   Breakdown: {', '.join(details)}")
        if filtering_mode == 'blacklist':
            print(f"🛡️  Annotations preserved by blacklist policy: {stats['labels_saved']}")
        elif stats['labels_saved'] > 0:
            print(f"🛡️  Annotations preserved by whitelist: {stats['labels_saved']}")

    if args.review:
        print(f"🔍 Frames flagged for manual review: {stats['reviewed']}")
        if stats['reviewed'] > 0 and not args.clean_labels:
            details = []
            if stats['reason_area'] > 0: details.append(f"Small Area: {stats['reason_area']}")
            if stats['reason_edge'] > 0: details.append(f"Edge Slivers: {stats['reason_edge']}")
            if stats['reason_giant'] > 0: details.append(f"Giant: {stats['reason_giant']}")
            if stats['reason_occ'] > 0: details.append(f"Occluded: {stats['reason_occ']}")
            if stats['reason_trunc'] > 0: details.append(f"Truncated: {stats['reason_trunc']}")
            if stats['reason_degen'] > 0: details.append(f"Degenerate: {stats['reason_degen']}")
            if stats.get('reason_manual', 0) > 0: details.append(f"Manual/Blacklist: {stats['reason_manual']}")
            if details:
                print(f"   Review Breakdown: {', '.join(details)}")

        if stats['reviewed'] > 0:
            target_wl = Path(args.dir) / "whitelist.txt"
            target_bl = Path(args.dir) / "blacklist.txt"
            print("\n💡 NEXT STEPS (Cleaning Strategies):")
            print("  1. Open the '_review_occlusions' folder and inspect the images.")
            print("     - Note: An annotation's line index is shown at the start of the label (e.g. [#0], [#1]).")
            print("  2. Choose the workflow that best fits your preference:")
            print("     • OPTION A (Whitelist): If most flagged annotations are bad and you want to KEEP only a few:")
            print(f"       -> Create '{target_wl}' and list the ones to SAVE.")
            print("          Format: 'frame_id:line_index' (e.g. '000001:2') or 'frame_id' to save all lines.")
            print("          (All suspicious annotations not in whitelist will be deleted)")
            print("     • OPTION B (Blacklist): If most flagged annotations are valid and you want to DELETE only a few:")
            print(f"       -> Create '{target_bl}' and list the ones to DELETE.")
            print("          Format: 'frame_id:line_index' (e.g. '000041:1') to delete a specific annotation,")
            print("                  'frame_id' (e.g. '000041') to delete ONLY the flagged/suspicious annotations in that frame,")
            print("                  'frame_id:all' (e.g. '000041:all') to purge the entire frame unconditionally.")
            print("          (ONLY targeted annotations in blacklist will be deleted; everything else is kept)")
            print("          [Note: If blacklist.txt exists, it automatically takes precedence over whitelist.txt]")
            print("  3. Run the script again to apply your changes:")
            print(f"     python -m src.data_prep.clean_dataset --dir {args.dir} --clean_labels")

    if getattr(args, 'edge_ar_thresh', 3.0) == 0.0 or getattr(args, 'no_edge_filter', False):
        print("ℹ️  Edge aspect ratio filter was DISABLED (narrow elongated objects touching edge were preserved).")

    if args.dry:
        print("⚠️  DRY-RUN: No files were actually modified.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset cleaner for Isaac Sim and YOLO datasets (Detection & Segmentation)")
    parser.add_argument("--dir", type=str, default=config.RAW_DATA_DIR, help="Root directory")
    parser.add_argument("--dry", action="store_true", help="Dry run, no modifications")
    parser.add_argument("--thresh_mean", type=float, default=10.0, help="Darkness threshold")
    parser.add_argument("--thresh_std", type=float, default=10.0, help="Flatness/std threshold")
    parser.add_argument("--empty", action="store_true", help="Delete pure backgrounds")
    parser.add_argument("--move_empty", action="store_true", help="Move empty backgrounds")
    parser.add_argument("--review", action="store_true", help="Generate review images of bad annotations")
    parser.add_argument("--clean_labels", action="store_true", help="Actually remove bad annotations from .txt")
    parser.add_argument("--whitelist", type=str, default=None, help="Path to whitelist file (default: <dir>/whitelist.txt)")
    parser.add_argument("--blacklist", type=str, default=None, help="Path to blacklist file (default: <dir>/blacklist.txt, takes priority)")
    parser.add_argument("--area_thresh", type=float, default=1000.0, help="Min area in pixels for annotations")
    parser.add_argument("--max_area_ratio", type=float, default=0.50, help="Max percentage of the image an annotation can occupy (0.0 to 1.0)")
    parser.add_argument("--edge_ar_thresh", type=float, default=3.0, help="Aspect ratio threshold for annotations touching the edge (0 to disable)")
    parser.add_argument("--no_edge_filter", "--disable_edge_ar", action="store_true",
                        help="Disable edge aspect-ratio filter (recommended for elongated objects like poles, towers, wires)")
    parser.add_argument("--format", type=str, choices=['auto', 'kitti', 'yolo_bbox', 'yolo_seg'], default='auto',
                        help="Label format: 'auto' (infers dataset format from sample files at startup), or force format")
    parser.add_argument("--segmentation", "--seg", "--draw_seg", action="store_true", dest="segmentation",
                        help="Enable segmentation-aware cleaning and soft mask drawing (inspects mask pixels in raw datasets)")
    parser.add_argument("--area_metric", type=str, choices=['polygon', 'bbox'], default='polygon',
                        help="Metric for area thresholding in segmentation: 'polygon' (contourArea/mask pixels) or 'bbox' (bounding box area)")

    args = parser.parse_args()

    if getattr(args, 'no_edge_filter', False):
        args.edge_ar_thresh = 0.0

    if Path(args.dir).exists():
        clean_dataset(args)
    else:
        print(f"The directory {args.dir} does not exist.")