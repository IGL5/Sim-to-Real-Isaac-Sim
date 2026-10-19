import shutil
from pathlib import Path
import cv2
import numpy as np

from src.core import config


def is_image_corrupted(img, thresh_mean, thresh_std):
    """ Checks if an image is unreadable, completely flat, or corrupted. """
    if img is None:
        return True
    if np.mean(img) < thresh_mean or np.std(img) < thresh_std:
        return True
    return False


def is_label_empty(txt_path):
    """ Checks if a label file is missing or has only whitespace. """
    if not txt_path.exists():
        return True
    with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
        if not f.read().strip():
            return True
    return False


def discover_image_folders(dir_path):
    """ Finds folders containing images, supporting both raw ('rgb') and processed ('images'). """
    # 1. Search for 'rgb' folders (Isaac Sim raw layout)
    rgb_folders = sorted([d for d in dir_path.rglob("rgb") if d.is_dir()])
    if rgb_folders:
        return rgb_folders
        
    # 2. Search for 'images' folders (YOLO processed / dataset layout)
    images_folders = []
    for d in dir_path.rglob("images"):
        if d.is_dir():
            sub_dirs = [s for s in d.iterdir() if s.is_dir()]
            if sub_dirs:
                images_folders.extend(sorted(sub_dirs))
            else:
                images_folders.append(d)
    if images_folders:
        return sorted(list(set(images_folders)))
        
    # 3. Fallback: check if dir_path itself directly contains valid images
    for ext in config.VALID_IMAGE_EXTENSIONS:
        if list(dir_path.glob(f"*{ext}")):
            return [dir_path]
            
    return []


def resolve_label_path(img_path, camera_root):
    """
    Locates the matching label file for an image:
    1. camera_root / 'object_detection' / f'{frame_id}.txt' (Isaac Sim raw)
    2. camera_root / 'labels' / f'{frame_id}.txt' (standard YOLO)
    3. Matching subdirectory in 'labels' (e.g. images/train -> labels/train)
    4. Sibling / same folder
    """
    frame_id = img_path.stem
    candidates = [
        camera_root / "object_detection" / f"{frame_id}.txt",
        camera_root / "labels" / f"{frame_id}.txt"
    ]
    
    # 1. Standard YOLO dataset layout: replace 'images' with 'labels' in path
    if "images" in img_path.parts:
        parts = list(img_path.parts)
        idx = len(parts) - 1 - parts[::-1].index("images")
        parts[idx] = "labels"
        candidates.append(Path(*parts).with_suffix(".txt"))

    # 2. Raw Isaac Sim layout: replace 'rgb' with 'object_detection' in path
    if "rgb" in img_path.parts:
        parts = list(img_path.parts)
        idx = len(parts) - 1 - parts[::-1].index("rgb")
        parts[idx] = "object_detection"
        candidates.append(Path(*parts).with_suffix(".txt"))

    for c in candidates:
        if c.exists():
            return c
            
    # Default fallback
    if (camera_root / "object_detection").exists():
        return camera_root / "object_detection" / f"{frame_id}.txt"
    if (camera_root / "labels").exists():
        return camera_root / "labels" / f"{frame_id}.txt"
    return candidates[0]


def resolve_valid_raw_mask_folder(camera_root, sample_limit=5):
    """
    Finds the segmentation mask folder for raw datasets (e.g., instance_segmentation or semantic_segmentation).
    Verifies that sample masks actually contain non-zero pixels, avoiding all-black masks (such as uint16 zeros).
    """
    candidates = [
        camera_root / "instance_segmentation",
        camera_root / "semantic_segmentation",
    ]
    for folder in candidates:
        if not folder.exists() or not folder.is_dir():
            continue
        pngs = list(folder.glob("*.png"))
        if not pngs:
            continue
        has_non_zero = False
        for p in pngs[:sample_limit]:
            m = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if m is not None:
                if m.ndim == 3:
                    if np.any(m[:, :, :3] > 0):
                        has_non_zero = True
                        break
                elif np.any(m > 0):
                    has_non_zero = True
                    break
        if has_non_zero:
            return folder
    return None


def handle_action(rgb_path, camera_root, action, dry_run, stats, stat_key, frame_id, target_dir=None, txt_path=None):
    """ Records stat counters and triggers physical delete/move operations on disk. """
    stats[stat_key] += 1
    if dry_run: 
        action_str = "moved" if action == "move" else "deleted"
        print(f"  [DRY-RUN] Frame {frame_id} would be {action_str} (Reason: {stat_key.upper()})")
        return
        
    if action == "delete": 
        delete_hierarchical_frame(rgb_path, camera_root, txt_path=txt_path)
    elif action == "move" and target_dir: 
        move_hierarchical_frame(rgb_path, camera_root, target_dir, txt_path=txt_path)


def save_review_image(annotated_img, data_dir, camera_name, frame_id):
    """ Writes annotated review image with rendered boxes to _review_occlusions folder. """
    dest_folder = Path(data_dir) / "_review_occlusions" / camera_name
    dest_folder.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dest_folder / f"{frame_id}.png"), annotated_img)


def delete_hierarchical_frame(rgb_path, camera_root, txt_path=None):
    """ Deletes RGB image and corresponding multimodal files (depth, normals, segmentation, labels). """
    frame_id = rgb_path.stem
    if camera_root.exists() and camera_root.is_dir():
        for folder in [f for f in camera_root.iterdir() if f.is_dir()]:
            for f_path in folder.glob(f"{frame_id}.*"):
                try:
                    f_path.unlink()
                except OSError as e:
                    print(f"⚠️ Error deleting {f_path}: {e}")
    if rgb_path.exists():
        try:
            rgb_path.unlink()
        except OSError as e:
            print(f"⚠️ Error deleting {rgb_path}: {e}")
    if txt_path and txt_path.exists():
        try:
            txt_path.unlink()
        except OSError as e:
            print(f"⚠️ Error deleting {txt_path}: {e}")


def move_hierarchical_frame(rgb_path, camera_root, target_root_dir, txt_path=None):
    """ Moves RGB image and its multimodal sibling files to an archive/empty directory. """
    frame_id = rgb_path.stem
    camera_name = camera_root.name
    dest_camera_root = Path(target_root_dir) / camera_name
    
    if camera_root.exists() and camera_root.is_dir():
        for folder in [f for f in camera_root.iterdir() if f.is_dir()]:
            dest_folder = dest_camera_root / folder.name
            dest_folder.mkdir(parents=True, exist_ok=True)
            for f_path in folder.glob(f"{frame_id}.*"):
                try:
                    shutil.move(str(f_path), str(dest_folder / f_path.name))
                except OSError as e:
                    print(f"⚠️ Error moving {f_path}: {e}")
                
    if rgb_path.exists():
        dest_img_folder = dest_camera_root / rgb_path.parent.name
        dest_img_folder.mkdir(parents=True, exist_ok=True)
        try:
            shutil.move(str(rgb_path), str(dest_img_folder / rgb_path.name))
        except OSError as e:
            print(f"⚠️ Error moving {rgb_path}: {e}")
        
    if txt_path and txt_path.exists():
        dest_lbl_folder = dest_camera_root / txt_path.parent.name
        dest_lbl_folder.mkdir(parents=True, exist_ok=True)
        try:
            shutil.move(str(txt_path), str(dest_lbl_folder / txt_path.name))
        except OSError as e:
            print(f"⚠️ Error moving {txt_path}: {e}")
