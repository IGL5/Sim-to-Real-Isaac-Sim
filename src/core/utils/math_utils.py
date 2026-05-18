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