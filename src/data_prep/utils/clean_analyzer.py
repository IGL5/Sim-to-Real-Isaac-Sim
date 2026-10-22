from pathlib import Path
import cv2
import numpy as np

from src.core import config

# Universal set of supported label filters
ALL_LABEL_FILTERS = {"degen", "occ", "giant", "edge", "trunc", "area"}

# Default semantic color palette for synthetic dataset classes (RGB in Isaac Sim masks)
SEMANTIC_CLASS_COLORS = {
    "electric_pole_wood": (220, 179, 62),
    "electric_pole_concrete": (27, 186, 239),
    "electric_pole_synth": (239, 186, 27),
    "electric_pole_manesman": (240, 4, 111),
    "car": (142, 0, 0),
    "bicycle": (32, 11, 119),
    "bicycle_street": (45, 120, 200),
    "bicycle_racing": (200, 45, 120),
    "bicycle_mountain": (45, 200, 120),
    "pedal": (255, 255, 0),
    "grapas": (0, 255, 255),
    "tornilleria": (255, 0, 255)
}

def extract_class_mask_from_crop(crop, class_name=None):
    """
    Extracts the binary mask for a specific class from an RGB semantic crop.
    Matches known palette color for the class if present, otherwise selects the
    dominant non-zero color in the crop (avoiding cross-class contour merging).
    """
    if crop is None or crop.size == 0:
        return np.zeros((0, 0), dtype=bool)

    if class_name and class_name in SEMANTIC_CLASS_COLORS:
        r, g, b = SEMANTIC_CLASS_COLORS[class_name]
        matches_rgb = np.all(crop == (r, g, b), axis=-1)
        if np.any(matches_rgb):
            return matches_rgb
        matches_bgr = np.all(crop == (b, g, r), axis=-1)
        if np.any(matches_bgr):
            return matches_bgr

    # Fallback: find non-zero pixels and pick dominant non-zero color in this bounding box
    flat = crop.reshape(-1, 3)
    non_black = flat[np.any(flat > 0, axis=-1)]
    if len(non_black) == 0:
        return np.zeros(crop.shape[:2], dtype=bool)

    unq, counts = np.unique(non_black, axis=0, return_counts=True)
    dominant_color = unq[np.argmax(counts)]
    return np.all(crop == dominant_color, axis=-1)


def bbox_iou(boxA, boxB):
    """ Computes Intersection over Union between two (x1, y1, x2, y2) bounding boxes. """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0.0, xB - xA + 1.0) * max(0.0, yB - yA + 1.0)
    areaA = max(0.0, boxA[2] - boxA[0] + 1.0) * max(0.0, boxA[3] - boxA[1] + 1.0)
    areaB = max(0.0, boxB[2] - boxB[0] + 1.0) * max(0.0, boxB[3] - boxB[1] + 1.0)
    denom = areaA + areaB - inter
    return inter / denom if denom > 0 else 0.0


def extract_instance_mask_from_crop(seg_mask, xmin, ymin, xmax, ymax, class_name=None, mask_cache=None):
    """
    Extracts the binary mask for an object inside its bounding box from a segmentation image.
    Supports:
    - 2D uint16/uint8 instance masks (e.g. Isaac Sim instance_segmentation).
      When multiple instances overlap inside the crop, accurately identifies the target
      instance by evaluating bounding box IoU and containment across the full frame,
      avoiding foreground interloper mask corruption.
    - 3D RGB/RGBA instance or semantic masks (e.g. instance_rgb or semantic_segmentation).
    """
    if seg_mask is None or seg_mask.size == 0:
        return np.zeros((0, 0), dtype=bool)

    h, w = seg_mask.shape[:2]
    x1_i, y1_i = max(0, int(xmin)), max(0, int(ymin))
    x2_i, y2_i = min(w, int(xmax)), min(h, int(ymax))

    if x2_i <= x1_i or y2_i <= y1_i:
        return np.zeros((0, 0), dtype=bool)

    # 1. Handle 3D / RGB / RGBA masks
    if seg_mask.ndim == 3:
        crop_3d = seg_mask[y1_i:y2_i, x1_i:x2_i, :3]
        if class_name and class_name in SEMANTIC_CLASS_COLORS:
            r, g, b = SEMANTIC_CLASS_COLORS[class_name]
            matches = np.all(crop_3d == (r, g, b), axis=-1) | np.all(crop_3d == (b, g, r), axis=-1)
            if np.any(matches):
                return matches

        full_mask = (
            seg_mask[:, :, 0].astype(np.int64) |
            (seg_mask[:, :, 1].astype(np.int64) << 8) |
            (seg_mask[:, :, 2].astype(np.int64) << 16)
        )
        crop = full_mask[y1_i:y2_i, x1_i:x2_i]
    else:
        crop = seg_mask[y1_i:y2_i, x1_i:x2_i]
        full_mask = seg_mask

    pos_pixels = crop[crop > 0]
    if len(pos_pixels) == 0:
        return np.zeros_like(crop, dtype=bool)

    vals, counts = np.unique(pos_pixels, return_counts=True)
    if len(vals) == 1:
        return (crop == vals[0])

    target_box = (float(x1_i), float(y1_i), float(x2_i), float(y2_i))
    best_uid = None
    best_score = -1.0

    cache = mask_cache if mask_cache is not None else {}
    for uid, cnt in zip(vals, counts):
        if uid in cache:
            cand_box, total_pixels = cache[uid]
        else:
            ys, xs = np.where(full_mask == uid)
            if len(xs) == 0:
                cand_box, total_pixels = None, 0
            else:
                cand_box = (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
                total_pixels = len(xs)
            cache[uid] = (cand_box, total_pixels)

        if cand_box is None or total_pixels == 0:
            continue

        iou = bbox_iou(target_box, cand_box)
        containment = cnt / float(total_pixels)
        score = iou * 0.7 + containment * 0.3
        if score > best_score:
            best_score = score
            best_uid = uid

    if best_uid is not None:
        return (crop == best_uid)

    dom_id = vals[np.argmax(counts)]
    return (crop == dom_id)


def classify_single_line(parts):
    """ Classifies a single line's tokens for format voting. """
    if len(parts) >= 8 and (not parts[0].isdigit() or ',' in parts[0]):
        return 'kitti'
    if len(parts) == 5 and parts[0].isdigit():
        return 'yolo_bbox'
    if len(parts) >= 7 and (len(parts) - 1) % 2 == 0 and parts[0].isdigit():
        return 'yolo_seg'
    if len(parts) >= 8:
        try:
            if any(float(x) > 1.0 for x in parts[4:8]):
                return 'kitti'
        except ValueError:
            pass
    return 'unknown'


def detect_dataset_format(image_folders, forced_format='auto', sample_size=15):
    """
    Samples non-empty label files at startup to infer the format for the entire dataset
    once, avoiding per-line detection overhead and edge-case classification errors.
    """
    if forced_format != 'auto':
        return forced_format

    votes = {'kitti': 0, 'yolo_bbox': 0, 'yolo_seg': 0}
    sampled_files = 0

    from src.data_prep.utils.clean_fs import resolve_label_path

    for img_folder in image_folders:
        camera_root = img_folder.parent
        sample_imgs = []
        for ext in config.VALID_IMAGE_EXTENSIONS:
            sample_imgs.extend(img_folder.glob(f"*{ext}"))
            if len(sample_imgs) >= sample_size * 2:
                break
        
        for img_path in sample_imgs:
            txt_path = resolve_label_path(img_path, camera_root)
            if txt_path and txt_path.exists() and txt_path.stat().st_size > 0:
                try:
                    with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
                        for line in f:
                            parts = line.strip().split()
                            if not parts:
                                continue
                            fmt = classify_single_line(parts)
                            if fmt in votes:
                                votes[fmt] += 1
                    sampled_files += 1
                    if sampled_files >= sample_size:
                        break
                except Exception:
                    continue
        if sampled_files >= sample_size:
            break

    total_votes = sum(votes.values())
    if total_votes == 0:
        return 'kitti'

    return max(votes, key=votes.get)


def parse_label_line(line, img_w, img_h, line_format='auto', area_metric='polygon', seg_mask=None, mask_cache=None):
    """
    Parses a single label line using line_format ('kitti', 'yolo_bbox', or 'yolo_seg').
    If line_format is 'auto', falls back to classifying this individual line.
    If seg_mask is provided for KITTI, counts real mask pixels inside the bbox.
    """
    parts = line.strip().split()
    if not parts:
        return None

    if line_format == 'auto':
        line_format = classify_single_line(parts)

    res = {
        "format": line_format,
        "is_valid": True,
        "is_degenerate": False,
        "degen_reason": None,
        "class_id_or_name": parts[0],
        "polygon": None,
        "contours": None,
        "bbox": (0.0, 0.0, 0.0, 0.0), # xmin, ymin, xmax, ymax in pixels
        "area": 0.0,
        "bbox_area": 0.0,
        "aspect_ratio": 0.0,
        "touches_edge": False,
        "occluded": None,
        "truncated": None
    }

    margin = 2

    if line_format == 'kitti':
        if len(parts) < 8:
            res["is_valid"] = False
            return res
        try:
            truncated = float(parts[1])
            occluded = int(float(parts[2]))
            xmin, ymin, xmax, ymax = map(float, parts[4:8])
            bbox_w = max(0.0, xmax - xmin)
            bbox_h = max(0.0, ymax - ymin)
            bbox_area = bbox_w * bbox_h
            area = bbox_area
            ar = bbox_w / bbox_h if bbox_h > 0 else 0.0
            touches = (xmin <= margin) or (ymin <= margin) or (xmax >= img_w - margin) or (ymax >= img_h - margin)
            
            # Mask-assisted checks if seg_mask is available
            contours_found = None
            if seg_mask is not None:
                x1_i, y1_i = max(0, int(xmin)), max(0, int(ymin))
                x2_i, y2_i = min(img_w, int(xmax)), min(img_h, int(ymax))
                if x2_i > x1_i and y2_i > y1_i:
                    class_name = parts[0]
                    non_zero = extract_instance_mask_from_crop(
                        seg_mask, x1_i, y1_i, x2_i, y2_i, class_name=class_name, mask_cache=mask_cache
                    )
                    mask_pixels = int(np.count_nonzero(non_zero))
                    if mask_pixels == 0:
                        res["is_degenerate"] = True
                        res["degen_reason"] = "EmptyMask"
                        area = 0.0
                    else:
                        area = float(mask_pixels) if area_metric == 'polygon' else bbox_area
                        cnts, _ = cv2.findContours(non_zero.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if cnts:
                            contours_found = [c + np.array([[[x1_i, y1_i]]]) for c in cnts]
                else:
                    res["is_degenerate"] = True
                    res["degen_reason"] = "CollapsedBbox"
                    area = 0.0

            res.update({
                "truncated": truncated, "occluded": occluded,
                "bbox": (xmin, ymin, xmax, ymax),
                "area": area, "bbox_area": bbox_area, "aspect_ratio": ar,
                "touches_edge": touches,
                "contours": contours_found
            })
        except ValueError:
            res["is_valid"] = False

    elif line_format == 'yolo_bbox':
        if len(parts) != 5:
            res["is_valid"] = False
            return res
        try:
            xc, yc, w_box, h_box = map(float, parts[1:5])
            xc_px, yc_px = xc * img_w, yc * img_h
            w_px, h_px = w_box * img_w, h_box * img_h
            xmin = max(0.0, xc_px - w_px / 2.0)
            ymin = max(0.0, yc_px - h_px / 2.0)
            xmax = min(float(img_w), xc_px + w_px / 2.0)
            ymax = min(float(img_h), yc_px + h_px / 2.0)
            bbox_w = max(0.0, xmax - xmin)
            bbox_h = max(0.0, ymax - ymin)
            area = bbox_w * bbox_h
            ar = bbox_w / bbox_h if bbox_h > 0 else 0.0
            touches = (xmin <= margin) or (ymin <= margin) or (xmax >= img_w - margin) or (ymax >= img_h - margin)

            if bbox_w < 1.0 or bbox_h < 1.0:
                res["is_degenerate"] = True
                res["degen_reason"] = "TinyBbox"

            res.update({
                "bbox": (xmin, ymin, xmax, ymax),
                "area": area, "bbox_area": area, "aspect_ratio": ar,
                "touches_edge": touches
            })
        except ValueError:
            res["is_valid"] = False

    elif line_format == 'yolo_seg':
        try:
            coords = [float(x) for x in parts[1:]]
            if len(coords) < 6 or len(coords) % 2 != 0:
                res["is_degenerate"] = True
                res["degen_reason"] = "TooFewPoints"
                return res

            xs_norm = coords[0::2]
            ys_norm = coords[1::2]
            pts_px = [[x * img_w, y * img_h] for x, y in zip(xs_norm, ys_norm)]
            pts_arr = np.array(pts_px, dtype=np.float32)

            xmin = float(np.min(pts_arr[:, 0]))
            xmax = float(np.max(pts_arr[:, 0]))
            ymin = float(np.min(pts_arr[:, 1]))
            ymax = float(np.max(pts_arr[:, 1]))

            bbox_w = max(0.0, xmax - xmin)
            bbox_h = max(0.0, ymax - ymin)
            bbox_area = bbox_w * bbox_h
            ar = bbox_w / bbox_h if bbox_h > 0 else 0.0
            touches = (xmin <= margin) or (ymin <= margin) or (xmax >= img_w - margin) or (ymax >= img_h - margin)

            poly_area = float(cv2.contourArea(pts_arr))

            if len(pts_px) < 3 or poly_area < 1.0 or bbox_w < 1.0 or bbox_h < 1.0:
                res["is_degenerate"] = True
                res["degen_reason"] = "CollapsedPoly"

            effective_area = poly_area if area_metric == 'polygon' else bbox_area

            res.update({
                "polygon": pts_px,
                "bbox": (xmin, ymin, xmax, ymax),
                "area": effective_area,
                "bbox_area": bbox_area,
                "aspect_ratio": ar,
                "touches_edge": touches
            })
        except (ValueError, IndexError):
            res["is_valid"] = False
    else:
        res["is_valid"] = False

    return res


def evaluate_suspicious(obj, area_thresh, max_area_ratio, edge_ar_thresh, img_w, img_h, active_filters=None):
    """
    Evaluates whether a parsed object is suspicious based on active filters.
    Returns: (is_suspicious, reasons_list, primary_reason)
    """
    if active_filters is None:
        active_filters = ALL_LABEL_FILTERS

    img_area = img_w * img_h

    reasons = []
    primary_reason = None

    # Priority 1: Degenerate annotation (unusable for training)
    if "degen" in active_filters and obj["is_degenerate"]:
        reasons.append(f"Degen:{obj['degen_reason']}")
        if not primary_reason:
            primary_reason = "degen"

    # Priority 2: Occlusion (KITTI)
    if "occ" in active_filters and obj["occluded"] in [2, 3]:
        reasons.append(f"Occ:{obj['occluded']}")
        if not primary_reason:
            primary_reason = "occ"

    # Priority 3: Giant area (covers more than max_area_ratio)
    if "giant" in active_filters:
        area_ratio = obj["area"] / img_area if img_area > 0 else 0.0
        if area_ratio > max_area_ratio:
            reasons.append(f"Giant:{area_ratio * 100:.0f}%")
            if not primary_reason:
                primary_reason = "giant"

    # Priority 4: Edge sliver (touches image edge and has extreme aspect ratio)
    if "edge" in active_filters and edge_ar_thresh and edge_ar_thresh > 0:
        upper_limit = max(edge_ar_thresh, 1.0 / edge_ar_thresh)
        lower_limit = 1.0 / upper_limit
        if obj["touches_edge"] and (obj["aspect_ratio"] > upper_limit or obj["aspect_ratio"] < lower_limit):
            reasons.append(f"EdgeAR:{obj['aspect_ratio']:.1f}")
            if not primary_reason:
                primary_reason = "edge"

    # Priority 5: Truncation (KITTI)
    if "trunc" in active_filters and obj["truncated"] is not None and obj["truncated"] > 0.6:
        reasons.append(f"Trunc:{obj['truncated']:.2f}")
        if not primary_reason:
            primary_reason = "trunc"

    # Priority 6: Small area (< area_thresh)
    if "area" in active_filters and obj["area"] < area_thresh:
        reasons.append(f"Area:{obj['area']:.0f}")
        if not primary_reason:
            primary_reason = "area"

    is_suspicious = len(reasons) > 0
    return is_suspicious, reasons, primary_reason


def load_filter_file(file_path, list_type='whitelist'):
    """
    Loads filter file (whitelist or blacklist).
    Supported formats:
    - 'frame_id:line_index' -> Target specific annotation index (e.g. '000041:1').
    - 'frame_id' -> In whitelist: saves all annotations.
                    In blacklist: deletes ONLY flagged/suspicious annotations in that frame (safe default!).
    - 'frame_id:all' or 'frame_id:*' -> Purges/protects all annotations unconditionally.
    """
    filter_data = {}
    if not file_path:
        return filter_data
    path = Path(file_path)
    if not path.exists():
        return filter_data
        
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            
            if ":" in line:
                frame_id, idx_str = line.split(":", 1)
                frame_id = frame_id.strip()
                idx_str = idx_str.strip()
                if idx_str.lower() in ["all", "*"]:
                    filter_data.setdefault(frame_id, []).append("ALL")
                else:
                    try:
                        filter_data.setdefault(frame_id, []).append(int(idx_str))
                    except ValueError:
                        pass
            else:
                frame_id = line.strip()
                if list_type == 'blacklist':
                    # Safe default for blacklist: only purge flagged/suspicious annotations in this frame!
                    filter_data.setdefault(frame_id, []).append("FLAGGED")
                else:
                    # Whitelist default: protect all annotations in this frame
                    filter_data.setdefault(frame_id, []).append("ALL")
    return filter_data


# Aliases for backward compatibility
load_whitelist = lambda p: load_filter_file(p, list_type='whitelist')
load_blacklist = lambda p: load_filter_file(p, list_type='blacklist')


def analyze_labels(img, txt_path, area_thresh, max_area_ratio, filter_data=None, frame_id=None,
                   edge_ar_thresh=3.0, format_mode='auto', area_metric='polygon', seg_mask=None,
                   return_reasons=False, filtering_mode='whitelist', whitelist_data=None,
                   active_filters=None):
    """
    Parses annotations (KITTI or YOLO detection/segmentation). Flags suspicious entries and renders review visuals.
    - Whitelist Mode: Whitelisted annotations are ignored/skipped visually. Suspicious entries are drawn in red.
    - Blacklist Mode: Blacklisted annotations are drawn in red ([BLACKLIST]). Other suspicious annotations are drawn in yellow ([KEPT]).
    """
    if filter_data is None:
        filter_data = whitelist_data if whitelist_data is not None else {}

    if not txt_path.exists() or img is None:
        if return_reasons:
            return False, None, []
        return False, None
        
    with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
        
    img_review = None
    is_suspicious_overall = False
    img_h, img_w = img.shape[:2]
    frame_reasons = []
    
    frame_rules = filter_data.get(frame_id, [])
    mask_cache = {} if seg_mask is not None else None
    
    for idx, line in enumerate(lines):
        obj = parse_label_line(line, img_w, img_h, line_format=format_mode, area_metric=area_metric, seg_mask=seg_mask, mask_cache=mask_cache)
        if not obj or not obj["is_valid"]:
            continue

        is_suspicious, reasons, primary_reason = evaluate_suspicious(
            obj, area_thresh, max_area_ratio, edge_ar_thresh, img_w, img_h, active_filters=active_filters
        )

        if filtering_mode == 'blacklist':
            is_blacklisted = (idx in frame_rules) or ("ALL" in frame_rules) or (("FLAGGED" in frame_rules) and is_suspicious)
            
            # If the annotation is neither blacklisted nor suspicious, skip it completely
            if not is_blacklisted and not is_suspicious:
                continue

            is_suspicious_overall = True
            
            if is_blacklisted:
                tag = "[BLACKLIST -> DELETE]"
                color = (0, 0, 255) # RED: Target of removal
                alpha = 0.35
                thickness = 2
                if primary_reason:
                    frame_reasons.append(primary_reason)
                else:
                    frame_reasons.append("manual")
            else:
                tag = "[KEPT]"
                color = (0, 215, 255) # YELLOW: Candidate kept by policy
                alpha = 0.15
                thickness = 1
                if primary_reason:
                    frame_reasons.append(primary_reason)

            if img_review is None: 
                img_review = img.copy()

            xmin, ymin, xmax, ymax = obj["bbox"]
            reason_str = (" | " + " | ".join(reasons)) if reasons else ""
            reason_text = f"[#{idx}] {tag}{reason_str}"

            if obj.get("contours"):
                overlay = img_review.copy()
                cv2.fillPoly(overlay, obj["contours"], color)
                cv2.addWeighted(overlay, alpha, img_review, 1 - alpha, 0, img_review)
                cv2.polylines(img_review, obj["contours"], isClosed=True, color=color, thickness=thickness)
                anchor_x = int(xmin)
                anchor_y = max(15, int(ymin) - 5)
                cv2.putText(img_review, reason_text, (anchor_x, anchor_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
            elif obj["polygon"] is not None:
                pts_int = np.array(obj["polygon"], dtype=np.int32).reshape((-1, 1, 2))
                overlay = img_review.copy()
                cv2.fillPoly(overlay, [pts_int], color)
                cv2.addWeighted(overlay, alpha, img_review, 1 - alpha, 0, img_review)
                cv2.polylines(img_review, [pts_int], isClosed=True, color=color, thickness=thickness)
                anchor_x = int(pts_int[0][0][0])
                anchor_y = max(15, int(ymin) - 5)
                cv2.putText(img_review, reason_text, (anchor_x, anchor_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
            else:
                cv2.rectangle(img_review, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, thickness)
                text_y = int(ymin) - 5
                if text_y < 10:
                    text_y = int(ymin) + 15
                cv2.putText(img_review, reason_text, (int(xmin), text_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

        else:
            # WHITELIST MODE:
            if is_suspicious:
                if "ALL" in frame_rules or idx in frame_rules:
                    # Whitelisted: safe, don't flag as suspicious for deletion
                    continue 
                    
                is_suspicious_overall = True
                if primary_reason:
                    frame_reasons.append(primary_reason)
                if img_review is None: 
                    img_review = img.copy()
                
                xmin, ymin, xmax, ymax = obj["bbox"]
                reason_text = f"[#{idx}] " + " | ".join(reasons)

                if obj.get("contours"):
                    overlay = img_review.copy()
                    cv2.fillPoly(overlay, obj["contours"], (0, 0, 255))
                    cv2.addWeighted(overlay, 0.25, img_review, 0.75, 0, img_review)
                    cv2.polylines(img_review, obj["contours"], isClosed=True, color=(0, 0, 255), thickness=2)
                    anchor_x = int(xmin)
                    anchor_y = max(15, int(ymin) - 5)
                    cv2.putText(img_review, reason_text, (anchor_x, anchor_y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
                elif obj["polygon"] is not None:
                    pts_int = np.array(obj["polygon"], dtype=np.int32).reshape((-1, 1, 2))
                    overlay = img_review.copy()
                    cv2.fillPoly(overlay, [pts_int], (0, 0, 255))
                    cv2.addWeighted(overlay, 0.25, img_review, 0.75, 0, img_review)
                    cv2.polylines(img_review, [pts_int], isClosed=True, color=(0, 0, 255), thickness=2)
                    anchor_x = int(pts_int[0][0][0])
                    anchor_y = max(15, int(ymin) - 5)
                    cv2.putText(img_review, reason_text, (anchor_x, anchor_y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
                else:
                    cv2.rectangle(img_review, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2)
                    text_y = int(ymin) - 5
                    if text_y < 10:
                        text_y = int(ymin) + 15
                    cv2.putText(img_review, reason_text, (int(xmin), text_y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
                
    if return_reasons:
        return is_suspicious_overall, img_review, frame_reasons
    return is_suspicious_overall, img_review


# Backward compatibility alias
analyze_kitti_labels = analyze_labels


def apply_label_cleaning(txt_path, frame_id, area_thresh, max_area_ratio, filter_data=None, dry_run=False,
                         img_w=None, img_h=None, edge_ar_thresh=3.0, format_mode='auto', area_metric='polygon',
                         seg_mask=None, filtering_mode='whitelist', whitelist_data=None,
                         active_filters=None):
    """ Reads the .txt and overwrites it omitting suspicious lines (unless whitelisted) or deleting blacklisted lines. """
    if filter_data is None:
        filter_data = whitelist_data if whitelist_data is not None else {}

    with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
        
    keep_lines = []
    removed_count, saved_count, total_labels = 0, 0, 0
    r_occ, r_trunc, r_area, r_edge, r_giant, r_degen, r_manual = 0, 0, 0, 0, 0, 0, 0

    frame_rules = filter_data.get(frame_id, [])
    mask_cache = {} if seg_mask is not None else None
    
    for idx, line in enumerate(lines):
        obj = parse_label_line(line, img_w, img_h, line_format=format_mode, area_metric=area_metric, seg_mask=seg_mask, mask_cache=mask_cache)
        if not obj or not obj["is_valid"]:
            # Preserve unparseable lines to avoid data corruption
            keep_lines.append(line)
            continue
            
        total_labels += 1
        is_suspicious, _, primary_reason = evaluate_suspicious(
            obj, area_thresh, max_area_ratio, edge_ar_thresh, img_w, img_h, active_filters=active_filters
        )
        
        remove_this_label = False

        if filtering_mode == 'blacklist':
            # In Blacklist mode:
            # - idx in frame_rules: user specifically targeted this line index -> DELETE
            # - "ALL" in frame_rules: user explicitly requested full frame purge (:all) -> DELETE
            # - "FLAGGED" in frame_rules: user listed just frame_id -> DELETE ONLY IF SUSPICIOUS! (Protects valid objects!)
            is_explicit_idx = idx in frame_rules
            is_all_purge = "ALL" in frame_rules
            is_flagged_purge = ("FLAGGED" in frame_rules) and is_suspicious

            if is_explicit_idx or is_all_purge or is_flagged_purge:
                remove_this_label = True
            else:
                saved_count += 1
        else:
            # In Whitelist mode: Delete suspicious lines unless whitelisted!
            if is_suspicious:
                is_whitelisted = ("ALL" in frame_rules) or (idx in frame_rules)
                if is_whitelisted:
                    saved_count += 1
                else:
                    remove_this_label = True

        if remove_this_label:
            removed_count += 1
            if primary_reason == "degen":
                r_degen += 1
            elif primary_reason == "occ":
                r_occ += 1
            elif primary_reason == "giant":
                r_giant += 1
            elif primary_reason == "edge":
                r_edge += 1
            elif primary_reason == "trunc":
                r_trunc += 1
            elif primary_reason == "area":
                r_area += 1
            else:
                r_manual += 1
        else:
            keep_lines.append(line)
            
    if removed_count > 0 and not dry_run:
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.writelines(keep_lines)
            
    return {
        "removed": removed_count, "saved": saved_count, "total": total_labels,
        "occ": r_occ, "trunc": r_trunc, "area": r_area, "edge": r_edge, "giant": r_giant,
        "degen": r_degen, "manual": r_manual
    }
