import os
import re
from src.core import config

def get_available_models():
    """
    Scans the project directory and returns a list of valid models
    that contain the best.pt file. Sorts them by YOLO version.
    """
    if not os.path.exists(config.PROJECT_DIR):
        return []
        
    available_models = []
    for d in os.listdir(config.PROJECT_DIR):
        model_dir = os.path.join(config.PROJECT_DIR, d)
        weights_path = os.path.join(model_dir, config.BEST_MODEL_SUBPATH)
        
        if os.path.isdir(model_dir) and os.path.exists(weights_path):
            available_models.append(d)

    # Inner function to extract version and sort logically (YOLOv8, YOLOv9, etc)
    def get_yolo_version(model_name):
        match = re.match(r'^yolov?(\d+)', model_name, re.IGNORECASE)
        return int(match.group(1)) if match else 0 
        
    available_models.sort(key=get_yolo_version)
    return available_models


def get_project_classes(lowercase=False):
    """
    Reads the centralized class contract safely.
    """
    if not os.path.exists(config.CLASSES_PATH):
        print(f"❌ ERROR: Class file not found at {config.CLASSES_PATH}")
        return []
        
    try:
        with open(config.CLASSES_PATH, "r", encoding="utf-8") as f:
            classes = [line.strip() for line in f if line.strip()]
            
        if lowercase:
            return [c.lower() for c in classes]
        return classes
        
    except Exception as e:
        print(f"⚠️ Error reading classes file: {e}")
        return []