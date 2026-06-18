from pathlib import Path

# --- CONFIGURATION ---
DEFAULT_EXP_NAME = "yolov8_s_default"

# Output directories
RAW_DATA_DIR = Path("data/01_raw")
PROCESSED_DATA_DIR = Path("data/02_processed")
METRICS_DIR = Path("data/05_metrics")
BASE_MODELS_DIR = Path("data/base_models")
TEMPLATES_DIR = Path("templates")

# Temporary directories
XAI_OUTPUT_DIR = METRICS_DIR / "xai_report"
COMPARISON_OUTPUT_DIR = METRICS_DIR / "comparison_report"
EVALUATION_OUTPUT_DIR = METRICS_DIR / "temp_eval"
PLOTS_EVAL_DIR = EVALUATION_OUTPUT_DIR / "plots"

# Simulation Camera Name
CAMERA_NAME = "DroneCamera"

# Data Split Ratios
TRAIN_RATIO = 0.7
VAL_RATIO   = 0.2
TEST_RATIO  = 0.1

# Dataset paths
DATASET_IMAGES = PROCESSED_DATA_DIR / "images"
DATASET_LABELS = PROCESSED_DATA_DIR / "labels"
DATASET_TEST_IMAGES = DATASET_IMAGES / "test"
DATASET_TEST_LABELS = DATASET_LABELS / "test"
DATASET_VAL_IMAGES = DATASET_IMAGES / "val"
RAW_IMAGES_SUBPATH = Path(CAMERA_NAME) / "rgb"
RAW_LABELS_SUBPATH = Path(CAMERA_NAME) / "object_detection"

# Model paths
BEST_MODEL_SUBPATH = "weights/best.pt"

# Metadata timestamp keys
GENERATION_TIMESTAMP_KEY = "generation_date"
TRAIN_TIMESTAMP_KEY = "train_date"
UPDATE_TIMESTAMP_KEY = "last_updated"

# Metadata names
METADATA_FOLDER_NAME = "metadata"
SAVED_EVAL_FOLDER_NAME = "evaluations"

FILE_GEN_META = "generation_metadata.json"
FILE_DATASET_META = "dataset_metadata.json"
FILE_TRAIN_META = "training_metadata.json"
FILE_AUDIT_META = "audit_metadata.json"
FILE_REAL_AUDIT_META = "real_audit_metadata.json"
FILE_INFERENCE_META = "inference_metadata.json"

# Metadata paths
GENERATION_METADATA_PATH = RAW_DATA_DIR / FILE_GEN_META
DATASET_METADATA_PATH = PROCESSED_DATA_DIR / FILE_DATASET_META

# Class file
CLASSES_PATH = Path("src") / "core" / "classes.txt"

# Evaluation defaults
LIMIT_IMAGES_PER_VIS = 250

# Supported Extensions
VALID_IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg')

# Plots filenames
CONFUSION_MATRIX_FILENAME = "confusion_matrix.png"
CONFIDENCE_DIST_FILENAME = "confidence_distribution.png"
HEATMAP_FILENAME = "heatmap.png"
PR_CURVE_FILENAME = "pr_curve.png"
F1_CURVE_FILENAME = "f1_curve.png"

# XAI file names
ORIGINAL_XAI_PATH = XAI_OUTPUT_DIR / "01_Original_Image.jpg"
PREDICTIONS_XAI_PATH = XAI_OUTPUT_DIR / "00_Predictions_vs_GroundTruth.jpg"
STRUCTURAL_XAI_PATH = XAI_OUTPUT_DIR / "02_Structural_Analysis.png"
HEATMAPS_XAI_PATH = XAI_OUTPUT_DIR / "heatmaps"

# Thresholds
CONF_THRESHOLD = 0.4
IOU_THRESHOLD = 0.5
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
PROJECT_DIR = (Path("data") / PROJECT_NAME).resolve()