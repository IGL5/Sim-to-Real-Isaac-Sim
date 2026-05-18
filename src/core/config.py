import os


# --- CONFIGURATION ---
DEFAULT_EXP_NAME = "yolov8_s_default"

# Output directories
RAW_DATA_DIR = "data/01_raw"
PROCESSED_DATA_DIR = "data/02_processed"
METRICS_DIR = "data/05_metrics"

# Dataset directories
DATASET_TEST_IMAGES = os.path.join(PROCESSED_DATA_DIR, "images", "test")
DATASET_TEST_LABELS = os.path.join(PROCESSED_DATA_DIR, "labels", "test")

# Class file
CLASSES_PATH = os.path.join("src", "core", "classes.txt")

# Thresholds
CONF_THRESHOLD = 0.4
IOU_THRESHOLD = 0.5
LIMIT_IMAGES = 100
OVERLAP_THRESHOLD_ANALYSIS = 0.5

def get_dynamic_project_name():
    """Reads classes.txt to automatically name the folder (e.g: '04_bicycle_detector')"""
    try:
        with open(CLASSES_PATH, "r", encoding='utf-8') as f:
            first_class = f.readline().strip().replace(" ", "_")
            if first_class:
                return f"04_{first_class}_detector"
    except Exception:
        pass
    # Backup name in case the file is accidentally deleted
    return "04_bicycle_detector"

PROJECT_NAME = get_dynamic_project_name()
PROJECT_DIR = os.path.join("data", PROJECT_NAME)