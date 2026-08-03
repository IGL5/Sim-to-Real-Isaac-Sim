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

