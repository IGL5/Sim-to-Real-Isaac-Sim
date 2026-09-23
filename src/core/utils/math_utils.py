import numpy as np
import cv2


def calculate_iou_matrix(boxesA, boxesB):
        """ 
        Calculates the Intersection over Union (IoU) matrix between two sets of boxes.
        boxesA: List or array of N boxes [x1, y1, x2, y2]
        boxesB: List or array of M boxes [x1, y1, x2, y2]
        Returns: Numpy matrix of shape (N, M) with the IoUs.
        """
        if len(boxesA) == 0 or len(boxesB) == 0:
            return np.zeros((len(boxesA), len(boxesB)))

        bA = np.array(boxesA)
        bB = np.array(boxesB)

        A = bA[:, np.newaxis, :]
        B = bB[np.newaxis, :, :]
        xA = np.maximum(A[..., 0], B[..., 0])
        yA = np.maximum(A[..., 1], B[..., 1])
        xB = np.minimum(A[..., 2], B[..., 2])
        yB = np.minimum(A[..., 3], B[..., 3])

        # Intersection area
        interArea = np.maximum(0, xB - xA) * np.maximum(0, yB - yA)

        # Individual areas
        boxAArea = (A[..., 2] - A[..., 0]) * (A[..., 3] - A[..., 1])
        boxBArea = (B[..., 2] - B[..., 0]) * (B[..., 3] - B[..., 1])

        iou = interArea / (boxAArea + boxBArea - interArea + 1e-6)

        return iou

# --- Statistics ---

def calculate_1d_stats(arr):
    """Calculates mean, median, and std for a 1D array/list."""
    if not arr:
        return {"mean": 0.0, "median": 0.0, "std": 0.0}
    return {
        "mean": round(float(np.mean(arr)), 4),
        "median": round(float(np.median(arr)), 4),
        "std": round(float(np.std(arr)), 4)
    }

def calculate_spatial_stats(centers):
    """Calculates center of mass and dispersion for a list of 2D points [(x, y), ...]."""
    if not centers:
        return {"center_of_mass_x": 0.0, "center_of_mass_y": 0.0, "dispersion_x": 0.0, "dispersion_y": 0.0}
    
    c_arr = np.array(centers)
    return {
        "center_of_mass_x": round(float(np.mean(c_arr[:, 0])), 4),
        "center_of_mass_y": round(float(np.mean(c_arr[:, 1])), 4),
        "dispersion_x": round(float(np.std(c_arr[:, 0])), 4),
        "dispersion_y": round(float(np.std(c_arr[:, 1])), 4)
    }

def yolo_to_corners(xc, yc, w, h, img_w, img_h):
    """ Converts from YOLO normalized to absolute pixel coordinates (x1, y1, x2, y2) """
    x1 = int((xc - w / 2) * img_w)
    y1 = int((yc - h / 2) * img_h)
    x2 = int((xc + w / 2) * img_w)
    y2 = int((yc + h / 2) * img_h)
    return x1, y1, x2, y2

def corners_to_yolo(xmin, xmax, ymin, ymax, img_w, img_h):
    """ Converts from absolute corners to YOLO normalized format (xc, yc, w, h) """
    dw, dh = 1.0 / img_w, 1.0 / img_h
    xc = (xmin + xmax) / 2.0
    yc = (ymin + ymax) / 2.0
    w = xmax - xmin
    h = ymax - ymin
    return xc * dw, yc * dh, w * dw, h * dh

def calculate_speed_stats(speeds_dict):
    """ Calculates average and FPS stats from raw timings """
    import numpy as np
    avg_pre = np.mean(speeds_dict["preprocess"]) if speeds_dict.get("preprocess") else 0
    avg_inf = np.mean(speeds_dict["inference"]) if speeds_dict.get("inference") else 0
    avg_post = np.mean(speeds_dict["postprocess"]) if speeds_dict.get("postprocess") else 0
    total_ms = avg_pre + avg_inf + avg_post

    return {
        "preprocess_ms": round(float(avg_pre), 2),
        "inference_ms": round(float(avg_inf), 2),
        "postprocess_ms": round(float(avg_post), 2),
        "total_ms": round(float(total_ms), 2),
        "fps": round(1000.0 / total_ms, 2) if total_ms > 0 else 0
    }


def mask_to_polygons(mask, img_w, img_h, min_points=3, epsilon_factor=0.003):
    """
    Converts a binary pixel mask (numpy 2D array, non-zero for target instance)
    into a list of normalized polygons. Each polygon is a list of floats: [x1, y1, x2, y2, ...].
    """
    if mask is None or not np.any(mask):
        return []
    
    binary_mask = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    polygons = []
    for contour in contours:
        if cv2.contourArea(contour) < 5:
            continue
        
        epsilon = epsilon_factor * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        pts = approx.reshape(-1, 2)
        
        if len(pts) < min_points:
            continue
            
        norm_pts = []
        for x, y in pts:
            nx = float(np.clip(x / img_w, 0.0, 1.0))
            ny = float(np.clip(y / img_h, 0.0, 1.0))
            norm_pts.extend([nx, ny])
            
        polygons.append(norm_pts)
        
    return polygons


def polygon_to_bbox(polygon_coords):
    """
    Calculates bounding box stats (cx, cy, w, h, area, aspect_ratio)
    from a list of normalized polygon coordinates [x1, y1, x2, y2, ...].
    """
    if not polygon_coords or len(polygon_coords) < 6:
        return {"area": 0.0, "ar": 0.0, "cx": 0.0, "cy": 0.0}
    
    xs = polygon_coords[0::2]
    ys = polygon_coords[1::2]
    
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    
    w_box = max(0.0, xmax - xmin)
    h_box = max(0.0, ymax - ymin)
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    area = w_box * h_box
    aspect_ratio = w_box / h_box if h_box > 0 else 0.0
    
    return {
        "area": area,
        "ar": aspect_ratio,
        "cx": cx,
        "cy": cy
    }


def polygon_to_mask(polygon_pts, img_w, img_h):
    """
    Renders polygon points (either normalized [0..1] or pixel coordinates) into a binary mask of shape (img_h, img_w).
    polygon_pts: List or numpy array of floats [x1, y1, ...] or shape (N, 2)
    """
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    if polygon_pts is None or len(polygon_pts) == 0:
        return mask

    pts = np.array(polygon_pts, dtype=np.float32)
    if pts.ndim == 1:
        if len(pts) < 6:
            return mask
        pts = pts.reshape(-1, 2)
    elif pts.ndim == 2 and pts.shape[1] == 2:
        if len(pts) < 3:
            return mask
    else:
        return mask

    # Check if normalized [0, 1]
    if np.max(pts) <= 1.05:
        pts = pts.copy()
        pts[:, 0] *= img_w
        pts[:, 1] *= img_h

    pts_int = pts.astype(np.int32)
    cv2.fillPoly(mask, [pts_int], 1)
    return mask


def calculate_mask_iou_matrix(masksA, masksB, img_w=640, img_h=640):
    """
    Calculates Intersection over Union matrix between two sets of masks or polygons.
    masksA: List of masks (2D numpy arrays) or polygons [x1, y1, ...] or shape (N, 2)
    masksB: List of masks (2D numpy arrays) or polygons [x1, y1, ...] or shape (N, 2)
    Returns: Numpy matrix of shape (N, M) with the mask IoUs.
    """
    if len(masksA) == 0 or len(masksB) == 0:
        return np.zeros((len(masksA), len(masksB)))

    renderedA = []
    for mA in masksA:
        if isinstance(mA, np.ndarray) and mA.ndim == 2 and mA.shape[0] == img_h and mA.shape[1] == img_w:
            renderedA.append((mA > 0).astype(np.uint8))
        else:
            renderedA.append(polygon_to_mask(mA, img_w, img_h))

    renderedB = []
    for mB in masksB:
        if isinstance(mB, np.ndarray) and mB.ndim == 2 and mB.shape[0] == img_h and mB.shape[1] == img_w:
            renderedB.append((mB > 0).astype(np.uint8))
        else:
            renderedB.append(polygon_to_mask(mB, img_w, img_h))

    iou_mat = np.zeros((len(renderedA), len(renderedB)), dtype=np.float32)
    for i, mA in enumerate(renderedA):
        for j, mB in enumerate(renderedB):
            intersection = np.logical_and(mA, mB).sum()
            union = np.logical_or(mA, mB).sum()
            iou_mat[i, j] = intersection / (union + 1e-6)

    return iou_mat


def calculate_tile_grid(img_w, img_h, tile_size=640, overlap_ratio=0.2):
    """
    Generates a grid of tile bounding boxes (x1, y1, x2, y2) with overlap.
    Guarantees complete coverage of the image without going out of bounds.
    """
    if tile_size >= img_w and tile_size >= img_h:
        return [(0, 0, img_w, img_h)]

    step_x = max(1, int(tile_size * (1.0 - overlap_ratio)))
    step_y = max(1, int(tile_size * (1.0 - overlap_ratio)))

    x_coords = []
    curr_x = 0
    while curr_x + tile_size < img_w:
        x_coords.append(curr_x)
        curr_x += step_x
    x_coords.append(max(0, img_w - tile_size))
    x_coords = sorted(list(set(x_coords)))

    y_coords = []
    curr_y = 0
    while curr_y + tile_size < img_h:
        y_coords.append(curr_y)
        curr_y += step_y
    y_coords.append(max(0, img_h - tile_size))
    y_coords = sorted(list(set(y_coords)))

    tiles = []
    for y in y_coords:
        for x in x_coords:
            x2 = min(img_w, x + tile_size)
            y2 = min(img_h, y + tile_size)
            tiles.append((x, y, x2, y2))

    return tiles


def weighted_boxes_fusion(boxes, scores, classes, iou_thresh=0.5, conf_mode="mean"):
    """
    Performs Weighted Boxes Fusion (WBF) across detected bounding boxes per class.
    boxes: array-like of shape (N, 4) with [x1, y1, x2, y2]
    scores: array-like of shape (N,) with confidences
    classes: array-like of shape (N,) with class IDs
    iou_thresh: IoU threshold for clustering boxes
    conf_mode: 'mean' or 'max' for fused confidence score
    """
    if len(boxes) == 0:
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=int)

    boxes = np.array(boxes, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)
    classes = np.array(classes, dtype=int)

    unique_classes = np.unique(classes)
    fused_boxes = []
    fused_scores = []
    fused_classes = []

    for c in unique_classes:
        c_mask = (classes == c)
        c_boxes = boxes[c_mask]
        c_scores = scores[c_mask]

        order = np.argsort(c_scores)[::-1]
        c_boxes = c_boxes[order]
        c_scores = c_scores[order]

        clusters = []
        for b, s in zip(c_boxes, c_scores):
            matched = False
            for cluster in clusters:
                cl_boxes = np.array(cluster['boxes'])
                cl_scores = np.array(cluster['scores'])
                w_box = np.average(cl_boxes, axis=0, weights=cl_scores)

                iou = calculate_iou_matrix([b], [w_box])[0, 0]
                if iou >= iou_thresh:
                    cluster['boxes'].append(b)
                    cluster['scores'].append(s)
                    matched = True
                    break

            if not matched:
                clusters.append({'boxes': [b], 'scores': [s]})

        for cluster in clusters:
            cl_b = np.array(cluster['boxes'])
            cl_s = np.array(cluster['scores'])
            f_box = np.average(cl_b, axis=0, weights=cl_s)
            
            if conf_mode == "max":
                f_score = float(np.max(cl_s))
            else:
                f_score = float(np.mean(cl_s))

            fused_boxes.append(f_box)
            fused_scores.append(f_score)
            fused_classes.append(c)

    if not fused_boxes:
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=int)

    return np.array(fused_boxes, dtype=np.float32), np.array(fused_scores, dtype=np.float32), np.array(fused_classes, dtype=int)


def nms_boxes(boxes, scores, classes, iou_thresh=0.5):
    """
    Performs Non-Maximum Suppression (NMS) per class.
    """
    if len(boxes) == 0:
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=int)

    try:
        import torch
        import torchvision.ops as ops
        t_boxes = torch.as_tensor(boxes, dtype=torch.float32)
        t_scores = torch.as_tensor(scores, dtype=torch.float32)
        t_classes = torch.as_tensor(classes, dtype=torch.int64)
        
        keep_indices = ops.batched_nms(t_boxes, t_scores, t_classes, iou_threshold=iou_thresh).cpu().numpy()
        return np.array(boxes, dtype=np.float32)[keep_indices], np.array(scores, dtype=np.float32)[keep_indices], np.array(classes, dtype=int)[keep_indices]
    except Exception:
        boxes = np.array(boxes, dtype=np.float32)
        scores = np.array(scores, dtype=np.float32)
        classes = np.array(classes, dtype=int)

        keep = []
        for c in np.unique(classes):
            c_indices = np.where(classes == c)[0]
            c_boxes = boxes[c_indices]
            c_scores = scores[c_indices]
            order = np.argsort(c_scores)[::-1]

            while len(order) > 0:
                idx = order[0]
                keep.append(c_indices[idx])
                if len(order) == 1:
                    break
                ious = calculate_iou_matrix([c_boxes[idx]], c_boxes[order[1:]])[0]
                order = order[1:][ious < iou_thresh]

        keep = np.array(keep, dtype=int)
        return boxes[keep], scores[keep], classes[keep]


def calculate_pairwise_dispersion(centers, img_w=None, img_h=None):
    """
    Calculates pairwise spatial dispersion and nearest-neighbor clustering distance.
    centers: list or array of 2D points [(x, y), ...]
    img_w, img_h: if provided, points are normalized by image dimensions.
    """
    if not centers or len(centers) < 2:
        return {
            "mean_pairwise_dist": 0.0,
            "median_pairwise_dist": 0.0,
            "std_pairwise_dist": 0.0,
            "mean_nearest_neighbor_dist": 0.0,
            "category": "N/A (<= 1 objeto)"
        }

    c_arr = np.array(centers, dtype=np.float32)
    if img_w is not None and img_h is not None and (img_w > 1 or img_h > 1):
        c_arr[:, 0] /= img_w
        c_arr[:, 1] /= img_h

    diag = np.sqrt(2.0)
    diffs = c_arr[:, np.newaxis, :] - c_arr[np.newaxis, :, :]
    dist_matrix = np.linalg.norm(diffs, axis=-1) / diag

    N = len(centers)
    triu_indices = np.triu_indices(N, k=1)
    pairwise_dists = dist_matrix[triu_indices]

    np.fill_diagonal(dist_matrix, np.inf)
    nn_dists = np.min(dist_matrix, axis=1)

    mean_pw = float(np.mean(pairwise_dists))
    median_pw = float(np.median(pairwise_dists))
    std_pw = float(np.std(pairwise_dists))
    mean_nn = float(np.mean(nn_dists))

    if mean_pw < 0.15:
        category = "Muy Agrupado (Cluster Local)"
    elif mean_pw < 0.35:
        category = "Moderadamente Agrupado"
    else:
        category = "Disperso (Distribuido)"

    return {
        "mean_pairwise_dist": round(mean_pw, 4),
        "median_pairwise_dist": round(median_pw, 4),
        "std_pairwise_dist": round(std_pw, 4),
        "mean_nearest_neighbor_dist": round(mean_nn, 4),
        "category": category
    }

