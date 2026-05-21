from pathlib import Path
import yaml
from ultralytics import YOLO
import torch
import argparse
import time
import json
import re
import sys
import platform
from datetime import datetime
import pandas as pd
import shutil
import src.core.config as config
from src.core.utils import project_utils as pu


TRAIN_IMGS = "images/train"
VAL_IMGS = "images/val"
TEST_REL  = "images/test"

TRAIN_LABELS = "labels"

DEFAULT_EXP_NAME = "yolov8_s_default"

# Model type
# 'yolov8n.pt' -> Nano (Very fast, ideal for drones/Jetson Nano)
# 'yolov8s.pt' -> Small (Balanced)
# 'yolov8m.pt' -> Medium (More precise, requires good GPU)
DEFAULT_MODEL = 'yolov8s.pt'

# Training params
DEFAULT_EPOCHS = 50
DEFAULT_PATIENCE = 15
IMG_SIZE = 640      # Trains on SD resolution
BATCH_SIZE = 16
WORKERS = 4
FREEZE_LAYERS = 10


def check_gpu():
    """
    Checks if a GPU is available and returns the device.
    """
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
        return 0, gpu_name # Uses GPU 0
    else:
        cpu_name = platform.processor() or "CPU"
        print("⚠️ GPU not detected. CPU will be used (it will be slower).")
        return 'cpu', cpu_name


def create_yaml_config():
    """
    Creates a YAML config file for YOLO.
    """

    # Check if the dataset exists
    if not Path(config.PROCESSED_DATA_DIR).exists():
        raise FileNotFoundError(f"Dataset not found in: {config.PROCESSED_DATA_DIR}. Has the script 1 been executed?")

    # Read classes from file
    class_names = pu.get_project_classes()

    yaml_config = {
        'path': config.PROCESSED_DATA_DIR,
        'train': TRAIN_IMGS,
        'val': VAL_IMGS,
        'test': TEST_REL,
        'names': {i: name for i, name in enumerate(class_names)}
    }

    yaml_path = Path(config.PROCESSED_DATA_DIR) / 'dataset_config.yaml'

    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_config, f, default_flow_style=False)

    print(f"📄 Config file created at: {yaml_path}")
    return str(yaml_path)


def recover_misplaced_runs(exp_name):
    """
    Looks for the last training saved in the default folder 'runs/' 
    and moves it to the correct folder of our project.
    """
    expected_dir = Path(config.PROJECT_DIR) / exp_name
    expected_weights = expected_dir / config.BEST_MODEL_SUBPATH
    
    if expected_weights.exists():
        return
    
    search_path = Path("runs").rglob("best.pt")
    all_weights = list(search_path)
    
    if not all_weights:
        return
        
    latest_weight = max(all_weights, key=lambda p: p.stat().st_ctime)
    latest_train_dir = latest_weight.parent.parent
    
    expected_dir.mkdir(parents=True, exist_ok=True)
    
    # We move the content of the runs folder to our project folder
    for item in latest_train_dir.iterdir():
        d = expected_dir / item.name
        if item.is_dir():
            shutil.copytree(str(item), str(d), dirs_exist_ok=True)
        else:
            shutil.copy2(str(item), str(d))
            
    print(f"✅ Archivos rescatados con éxito a: {expected_dir}")


def get_next_experiment_prefix(ver_prefix, size_suffix, base_model_idx=None):
    """
    Scans the project directory to find the next available experiment number
    for a specific architecture, enforcing a strict naming convention.
    """
    project_dir = Path(config.PROJECT_DIR)
    if not project_dir.exists():
        project_dir.mkdir(parents=True, exist_ok=True)

    max_n = 0
    
    # Regular expression to search for the main number (e.g: yolov8_s_01_)
    pattern_n = re.compile(rf"^{ver_prefix}_{size_suffix}_(\d+)_")

    for d in project_dir.iterdir():
        if not d.is_dir(): 
            continue
        
        # Extract the global experiment number
        match_n = pattern_n.search(d.name)
        if match_n:
            n = int(match_n.group(1))
            if n > max_n: 
                max_n = n

    next_n = max_n + 1
    
    # Force it to have at least 2 digits
    next_n_str = f"{next_n:02d}" 
    
    # If we pass the index of a base model, it builds the inheritance tag
    if base_model_idx is not None:
        try:
            base_str = f"{int(base_model_idx):02d}"
        except ValueError:
            base_str = str(base_model_idx)
            
        return f"{ver_prefix}_{size_suffix}_{next_n_str}_finetuned{base_str}_"
    else:
        return f"{ver_prefix}_{size_suffix}_{next_n_str}_"


def interactive_selection():
    """
    Interactive flow to select version, size and name.
    """
    print("\n--- 🎛️ INTERACTIVE CONFIGURATION ---")
    
    # 1. Select YOLO version
    print("Available YOLO Architectures:")
    print("  [8]  YOLOv8  (Classic baseline)")
    print("  [9]  YOLOv9  (High precision / PGI)")
    print("  [10] YOLOv10 (NMS-free, ultra-low latency)")
    print("  [11] YOLO11  (Stable SOTA, highest efficiency)")
    print("  [26] YOLO26  (Bleeding-edge 2026 release)")
    version_input = input("Select version [8/9/10/11/26] (default 8): ").strip()

    if version_input == "26":
        model_prefix = "yolo26"
        name_prefix = "yolov26"
    elif version_input == "11":
        model_prefix = "yolo11"
        name_prefix = "yolov11"
    elif version_input == "10":
        model_prefix = "yolov10"
        name_prefix = "yolov10"
    elif version_input == "9":
        model_prefix = "yolov9"
        name_prefix = "yolov9"
    else:
        model_prefix = "yolov8"
        name_prefix = "yolov8"
        version_input = "8"

    print(f"   -> Selected: {name_prefix.upper()}")

    # 2. Select model size
    print("\nModel Sizes:")
    print("  [n] Nano   (Fastest, lowest accuracy)")
    print("  [s] Small  (Balanced - RECOMMENDED)")
    print("  [m] Medium (Slower, higher accuracy)")
    size_input = input("Model Size? [n/s/m] (default s): ").strip().lower()
    
    if size_input == 'n':
        if version_input == "9":
            size_suffix = 't'
            print("   -> Selected: Tiny")
        else:
            size_suffix = 'n'
            print("   -> Selected: Nano")
    elif size_input == 'm':
        size_suffix = 'm'
        print("   -> Selected: Medium")
    else:
        size_suffix = 's'
        print("   -> Selected: Small")

    model_to_use = f"{model_prefix}{size_suffix}.pt"

    # 3. Select experiment name
    prefix = get_next_experiment_prefix(name_prefix, size_suffix)
    print("\nExperiment Naming (Strict Policy):")
    print(f"   -> Mandatory Prefix: {prefix}")
    name_input = input("Custom description? (default 'custom'): ").strip()
    
    desc = name_input if name_input else "custom"
    exp_name = f"{prefix}{desc}"
    
    print(f"   -> Final Name: {exp_name}")
    print("--------------------------------------\n")
    return model_to_use, exp_name

def select_existing_model():
    """
    Interactive flow to select an already trained model for fine-tuning.
    """
    print("\n--- 🤖 EXISTING MODEL SELECTION (FINE-TUNING) ---")
    
    project_dir = Path(config.PROJECT_DIR)
    if not project_dir.exists():
        print(f"❌ ERROR: Project directory '{project_dir}' not found.")
        sys.exit(1)
        
    available_models = pu.get_available_models()
        
    print("📂 Available trained models to fine-tune:")
    for i, m in enumerate(available_models):
        print(f"  [{i+1}] {m}")
        
    while True:
        user_input = input(f"\nSelect a base model [1-{len(available_models)}] (default: 1): ").strip()
        
        if not user_input:
            base_exp_name = available_models[0]
            break
        
        if user_input.isdigit():
            idx = int(user_input) - 1
            if 0 <= idx < len(available_models):
                base_exp_name = available_models[idx]
                break
            else:
                print(f"  ⚠️  Number out of range. Please choose between 1 and {len(available_models)}.")
        else:
            if user_input in available_models:
                base_exp_name = user_input
                break
            print("  ⚠️  Invalid input. Please enter a valid number.")
            
    path_to_weights = str(Path(config.PROJECT_DIR) / base_exp_name / config.BEST_MODEL_SUBPATH)
    
    # Extract version, size and number of the base model (e.g: 'yolov8_s_12_custom')
    match = re.search(r"^(yolo\w+)_([nsmxl])_(\d+)_", base_exp_name)
    if match:
        ver_prefix = match.group(1)
        size_suffix = match.group(2)
        base_idx = match.group(3)
    else:
        # Fallback security if the base model did not follow the regulations (e.g: manually created)
        ver_prefix = "yolov8"
        size_suffix = "s"
        base_idx = "X" # Indicator of unknown origin model
        
    prefix = get_next_experiment_prefix(ver_prefix, size_suffix, base_model_idx=base_idx)
    
    print("\nExperiment Naming (Strict Policy):")
    print(f"   -> Mandatory Prefix: {prefix}")
    name_input = input("New Custom description? (default 'custom'): ").strip()
    
    desc = name_input if name_input else "custom"
    exp_name = f"{prefix}{desc}"
    
    print(f"✅ Selected base model: {base_exp_name}")
    print(f"✅ New experiment name: {exp_name}")
    print("--------------------------------------\n")
    
    return path_to_weights, exp_name


def main():
    parser = argparse.ArgumentParser(description="YOLO Training Tool")
    
    # Options
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS, help="Override number of epochs")
    parser.add_argument('--patience', type=int, default=DEFAULT_PATIENCE, help="Override patience (0 = disabled)")
    parser.add_argument('--freeze', type=int, default=FREEZE_LAYERS, help="Override freeze layers")
    parser.add_argument('--img_size', type=int, default=IMG_SIZE, help="Override image size [640 (SD default), 960 (1/2), 1280 (HD)]")
    parser.add_argument('--finetune', action='store_true', help="Fine-tune an existing trained model instead of using a base COCO model")
    parser.add_argument('--lr0', type=float, default=0.00, help="Initial learning rate (use 0.0001 for fine-tuning)")
    args = parser.parse_args()

    start_time = time.time()
    start_date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    device, device_name = check_gpu()

    # Model selection logic
    if args.finetune:
        model_type, experiment_name = select_existing_model()
    else:
        model_type, experiment_name = interactive_selection()

    print(f"🚀 Starting training: {model_type} | Epochs: {args.epochs} | Exp: {experiment_name}| LR: {args.lr0}")

    # 1. Create the treasure map (YAML)
    yaml_file = create_yaml_config()

    # 2. Load the model
    try:
        model = YOLO(model_type)
    except Exception as e:
        print(f"❌ Error loading model {model_type}. Make sure ultralytics is updated.")
        print(f"   Details: {e}")
        return

    train_kwargs = {
        'data': yaml_file,
        'epochs': args.epochs,
        'imgsz': args.img_size,
        'batch': BATCH_SIZE,
        'workers': WORKERS,
        'project': config.PROJECT_DIR,
        'name': experiment_name,
        'device': device,
        'patience': args.patience,
        'save': True,
        'exist_ok': True,
        'verbose': True,
        'freeze': args.freeze
    }

    # 3. Train
    if args.lr0 > 0.00:
        train_kwargs['lr0'] = args.lr0

    model.train(**train_kwargs)
    
    recover_misplaced_runs(experiment_name)

    print("\n--- Training completed ---")
    best_weight = Path(config.PROJECT_DIR) / experiment_name / config.BEST_MODEL_SUBPATH
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
        onnx_path = None

    # 6. Compile metadata
    print("\n📝 Compiling training metadata...")
    end_time = time.time()
    duration_secs = end_time - start_time
    duration_formatted = time.strftime("%H:%M:%S", time.gmtime(duration_secs))

    epochs_run = args.epochs
    best_epoch = -1

    results_csv_path = Path(config.PROJECT_DIR) / experiment_name / 'results.csv'
    if results_csv_path.exists():
        try:
            df = pd.read_csv(str(results_csv_path))
            epochs_run = len(df)
            
            df.columns = df.columns.str.strip()
            
            if 'metrics/mAP50(B)' in df.columns and 'metrics/mAP50-95(B)' in df.columns:
                fitness = (df['metrics/mAP50(B)'] * 0.1) + (df['metrics/mAP50-95(B)'] * 0.9)
                best_idx = fitness.idxmax()
                
                if 'epoch' in df.columns:
                    best_epoch = int(df.loc[best_idx, 'epoch'])
                else:
                    best_epoch = int(best_idx) + 1
        except Exception as e:
            print(f"⚠️ Could not read results.csv to find best epoch: {e}")

    if epochs_run < args.epochs and best_epoch == -1:
        best_epoch = epochs_run - args.patience

    args_yaml_path = Path(config.PROJECT_DIR) / experiment_name / 'args.yaml'
    aug_data = {}
    if args_yaml_path.exists():
        try:
            with open(args_yaml_path, 'r', encoding='utf-8') as f:
                yolo_args = yaml.safe_load(f)
                aug_data = {
                    "mosaic": yolo_args.get("mosaic", 1.0),
                    "mixup": yolo_args.get("mixup", 0.0),
                    "degrees": yolo_args.get("degrees", 0.0),
                    "translate": yolo_args.get("translate", 0.1),
                    "scale": yolo_args.get("scale", 0.5),
                    "fliplr": yolo_args.get("fliplr", 0.5),
                    "hsv_s": yolo_args.get("hsv_s", 0.7)
                }
        except Exception as e:
            print(f"⚠️ Could not parse args.yaml: {e}")

    
    weight_size_mb = 0.0
    if Path(best_weight).exists():
        weight_size_mb = round(Path(best_weight).stat().st_size / (1024 * 1024), 2)

    training_metadata = {
        "experiment_info": {
            "project_name": config.PROJECT_NAME,
            "experiment_name": experiment_name,
            "start_date": start_date_str,
            "duration_seconds": round(duration_secs, 2),
            "duration_formatted": duration_formatted
        },
        "hardware": {
            "device": "GPU" if device == 0 else "CPU",
            "device_name": device_name
        },
        "hyperparameters": {
            "model_base": model_type,
            "epochs_requested": args.epochs,
            "epochs_run": epochs_run,
            "best_epoch": best_epoch,
            "patience": args.patience,
            "freeze_layers": args.freeze,
            "learning_rate": args.lr0,
            "img_size": IMG_SIZE,
            "batch_size": BATCH_SIZE,
            "workers": WORKERS
        },
        "data_augmentation": aug_data,
        "metrics_test_set": {
            "mAP50_95": round(float(metrics.box.map), 4),
            "mAP50": round(float(metrics.box.map50), 4)
        },
        "artifacts": {
            "best_weights": best_weight,
            "best_weights_mb": weight_size_mb,
            "onnx_model": onnx_path
        }
    }

    # Save metadata
    metadata_dir = Path(config.PROJECT_DIR) / experiment_name / config.METADATA_FOLDER_NAME
    metadata_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = metadata_dir / config.FILE_TRAIN_META

    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(training_metadata, f, indent=4)

    dataset_metadata = Path(config.DATASET_METADATA_PATH)
    if dataset_metadata.exists():
        shutil.copy2(str(dataset_metadata), str(metadata_dir / config.FILE_DATASET_META))

    print(f"💾 Training and Dataset metadata saved at: {metadata_dir}")


if __name__ == '__main__':
    main()