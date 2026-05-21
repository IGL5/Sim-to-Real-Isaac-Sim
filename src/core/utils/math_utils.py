import numpy as np


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