import numpy as np
import os
import cv2

# --- CONFIGURATION ---
# Base directory where experiments are saved
PROJECT_DIR = os.path.join(os.getcwd(), "cyclist_detector")
DEFAULT_EXP_NAME = "yolov8_s_default"

# Default paths (Test Dataset)
DEFAULT_TEST_IMAGES = os.path.join(os.getcwd(), "dataset_yolo_output", "images", "test")
DEFAULT_TEST_LABELS = os.path.join(os.getcwd(), "dataset_yolo_output", "labels", "test")

# Output for reports
OUTPUT_DIR = os.path.join(os.getcwd(), "audit_report")

# Parameters
CONF_THRESHOLD = 0.4
IOU_THRESHOLD = 0.5
LIMIT_IMAGES = 100
OVERLAP_THRESHOLD_ANALYSIS = 0.5

# --- UTILITIES ---

def check_system_integrity(model_path, check_dataset=False):
    """
    Checks that everything necessary exists before starting.
    """
    # 1. Check Model
    if not os.path.exists(model_path):
        print(f"❌ ERROR: Not finding the model file in:")
        print(f"   -> {model_path}")
        print("   Did you run the training script (train_YOLO.py)?")
        return False

    # 2. Check Dataset (only if we are going to audit)
    if check_dataset:
        if not os.path.exists(DEFAULT_TEST_IMAGES):
            print(f"❌ ERROR: Not finding the test images folder:")
            print(f"   -> {DEFAULT_TEST_IMAGES}")
            return False
        
        # Check if not empty
        if not os.listdir(DEFAULT_TEST_IMAGES):
            print(f"⚠️ WARNING: The test folder is empty ({DEFAULT_TEST_IMAGES}).")
            return False

    return True

def parse_kitti_label(label_path, width, height):
    """ Converts txt to list of boxes [x1, y1, x2, y2] """
    boxes = []
    if not os.path.exists(label_path): return boxes
    
    try:
        with open(label_path, 'r') as f:
            for line in f:
                p = line.strip().split()
                # YOLO format: id xc yc w h
                xc, yc, w, h = float(p[1]), float(p[2]), float(p[3]), float(p[4])
                x1 = (xc - w/2) * width
                y1 = (yc - h/2) * height
                x2 = (xc + w/2) * width
                y2 = (yc + h/2) * height
                boxes.append([x1, y1, x2, y2])
    except Exception as e:
        print(f"⚠️ Error reading label {label_path}: {e}")
        
    return boxes

# --- Drawing ---

def draw_boxes(img, boxes, color=(0, 255, 0), label="", confidences=None):
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = map(int, b)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # If there is a specific confidence, we use it. If not, we use the generic label
        text_to_draw = label
        if confidences is not None and i < len(confidences):
            text_to_draw = f"{confidences[i]:.2f}"
            
        if text_to_draw: 
            cv2.putText(img, text_to_draw, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return img


def draw_overlapping_pairs(img, pred_boxes, pairs_indices, confidences=None):
    """
    Dibuja SOLO los pares de cajas que se solapan para resaltarlos.
    pairs_indices: Lista de tuplas [(idx1, idx2), (idx3, idx4)...]
    """
    alert_color = (0, 165, 255) 
    thickness = 3
    
    boxes_to_draw_idx = set()
    for i, j in pairs_indices:
        boxes_to_draw_idx.add(i)
        boxes_to_draw_idx.add(j)
        
    for idx in boxes_to_draw_idx:
        box = pred_boxes[idx]
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), alert_color, thickness)
        
        if confidences:
            conf = confidences[idx]
            label = f"{conf:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), alert_color, -1)
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
    return img

# --- IoU ---

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

        # Área de intersección
        interArea = np.maximum(0, xB - xA) * np.maximum(0, yB - yA)

        # Áreas individuales
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