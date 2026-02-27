import os
import yaml
from ultralytics import YOLO
import torch
import argparse
import time
import json
from datetime import datetime

# DEFAULT CONFIGURATION
DATASET_ROOT = os.path.join(os.getcwd(), "dataset_yolo_output")

TRAIN_IMGS = "images/train" 
VAL_IMGS = "images/val"
TEST_REL  = "images/test"

TRAIN_LABELS = "labels"

PROJECT_NAME = "cyclist_detector"
DEFAULT_EXP_NAME = "yolov8_s_default"

# Model type
# 'yolov8n.pt' -> Nano (Very fast, ideal for drones/Jetson Nano)
# 'yolov8s.pt' -> Small (Balanced)
# 'yolov8m.pt' -> Medium (More precise, requires good GPU)
DEFAULT_MODEL = 'yolov8s.pt'

# Training params
DEFAULT_EPOCHS = 50
IMG_SIZE = 640
BATCH_SIZE = 16
WORKERS = 4



def check_gpu():
    """
    Checks if a GPU is available and returns the device.
    """
    if torch.cuda.is_available():
        print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
        return 0 # Uses GPU 0
    else:
        print("⚠️ GPU not detected. CPU will be used (it will be slower).")
        return 'cpu'


def create_yaml_config():
    """
    Creates a YAML config file for YOLO.
    """
    abs_path = os.path.abspath(DATASET_ROOT)

    # Check if the dataset exists
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Dataset not found in: {abs_path}. Has the script 1 been executed?")

    config = {
        'path': abs_path,
        'train': TRAIN_IMGS,
        'val': VAL_IMGS,
        'test': TEST_REL,

        'names': {
            0: 'bicycle'  # Class name
        }
    }

    yaml_path = os.path.join(abs_path, 'dataset_config.yaml')

    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"📄 Config file created at: {yaml_path}")
    return yaml_path


def interactive_selection():
    """
    Interactive flow to select version, size and name.
    """
    print("\n--- 🎛️ INTERACTIVE CONFIGURATION ---")
    
    # 1. Select YOLO version
    version_input = input("Use YOLOv8 or YOLO11? [8/11] (default 8): ").strip()
    if version_input == "11":
        ver_prefix = "yolo11"
        print("   -> Selected: YOLO11")
    else:
        ver_prefix = "yolov8"
        print("   -> Selected: YOLOv8 (Default)")

    # 2. Select model size
    size_input = input("Model Size? [n/s/m] (default s): ").strip().lower()
    if size_input == 'n':
        size_suffix = 'n'
        print("   -> Selected: Nano")
    elif size_input == 'm':
        size_suffix = 'm'
        print("   -> Selected: Medium")
    else:
        size_suffix = 's'
        print("   -> Selected: Small (Default)")

    model_to_use = f"{ver_prefix}{size_suffix}.pt"

    # 3. Select experiment name
    default_name = f"{ver_prefix}_{size_suffix}_custom"
    name_input = input(f"Experiment Name? (default '{default_name}'): ").strip()
    
    if name_input:
        exp_name = name_input
        print(f"   -> Name: {exp_name}")
    else:
        exp_name = default_name
        print(f"   -> Name: {default_name} (Default)")

    print("--------------------------------------\n")
    return model_to_use, exp_name


def main():
    parser = argparse.ArgumentParser(description="YOLO Training Tool")
    
    # Options
    parser.add_argument('--epochs', type=int, default=None, help="Override number of epochs")
    parser.add_argument('--patience', type=int, default=15, help="Override patience")
    parser.add_argument('--select', action='store_true', help="Interactive mode to choose model version and size")
    args = parser.parse_args()

    start_time = time.time()
    start_date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    device = check_gpu()

    # Model selection logic
    model_type = None
    experiment_name = None
    if args.select:
        model_type, experiment_name = interactive_selection()
    else:
        model_type = DEFAULT_MODEL
        experiment_name = DEFAULT_EXP_NAME

    # Define epochs
    epochs_to_run = args.epochs if args.epochs else DEFAULT_EPOCHS

    print(f"🚀 Starting training: {model_type} | Epochs: {epochs_to_run} | Exp: {experiment_name}")

    # 1. Create the treasure map (YAML)
    yaml_file = create_yaml_config()

    # 2. Load the model
    try:
        model = YOLO(model_type)
    except Exception as e:
        print(f"❌ Error loading model {model_type}. Make sure ultralytics is updated.")
        print(e)
        return

    # 3. Train
    model.train(
        data=yaml_file,
        epochs=epochs_to_run, 
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        workers=WORKERS,
        project=PROJECT_NAME,
        name=experiment_name,
        device=device,
        patience=args.patience,     # If it doesn't improve in x epochs, stop (0 = disabled).
        save=True,                  # Save the best model
        exist_ok=True,              # If the experiment already exists, it will be overwritten.
        verbose=True,               # Show training progress
        freeze=10                   # Freeze the first 10 layers
    )

    print("\n--- Training completed ---")
    best_weight = os.path.join(PROJECT_NAME, experiment_name, 'weights', 'best.pt')
    print(f"💾 Best model saved at: {best_weight}")

    # 4. Validation (TEST SET)
    print("\n📊 Evaluating final precision in the TEST SET...")
    metrics = model.val(split='test')
    print(f"🎯 mAP50-95 (Test Set): {metrics.box.map:.4f}")
    print(f"🎯 mAP50 (Test Set):    {metrics.box.map50:.4f}")

    # 5. Export to ONNX (Ideal para Isaac Sim / ROS / TensorRT)
    try:
        print("\n📦 Exporting to ONNX...")
        onnx_path = model.export(format="onnx", dynamic=True)
        print(f"✅ Model exported for deployment: {onnx_path}")
    except Exception as e:
        print(f"⚠️ Export to ONNX failed (non-critical): {e}")

    # 6. Compile metadata
    print("\n📝 Compiling training metadata...")
    end_time = time.time()
    duration_secs = end_time - start_time
    duration_formatted = time.strftime("%H:%M:%S", time.gmtime(duration_secs))

    training_metadata = {
        "experiment_info": {
            "project_name": PROJECT_NAME,
            "experiment_name": experiment_name,
            "start_date": start_date_str,
            "duration_seconds": round(duration_secs, 2),
            "duration_formatted": duration_formatted
        },
        "hardware": {
            "device": "GPU" if device == 0 else "CPU"
        },
        "hyperparameters": {
            "model_base": model_type,
            "epochs_requested": epochs_to_run,
            "patience": args.patience,
            "img_size": IMG_SIZE,
            "batch_size": BATCH_SIZE,
            "workers": WORKERS
        },
        "metrics_test_set": {
            "mAP50_95": round(float(metrics.box.map), 4),
            "mAP50": round(float(metrics.box.map50), 4)
        },
        "artifacts": {
            "best_weights": best_weight,
            "onnx_model": onnx_path
        }
    }

    # Save metadata
    metadata_dir = os.path.join(PROJECT_NAME, experiment_name)
    os.makedirs(metadata_dir, exist_ok=True)
    metadata_path = os.path.join(metadata_dir, 'training_metadata.json')

    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(training_metadata, f, indent=4)

    print(f"💾 Training metadata saved at: {metadata_path}")


if __name__ == '__main__':
    main()