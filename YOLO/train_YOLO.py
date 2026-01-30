import os
import yaml
from ultralytics import YOLO
import torch
import argparse

# CONFIG
DATASET_ROOT = os.path.join(os.getcwd(), "dataset_yolo_output")

TRAIN_IMGS = "images/train" 
VAL_IMGS = "images/val"
TEST_REL  = "images/test"

TRAIN_LABELS = "labels"

PROJECT_NAME = "cyclist_detector"
EXPERIMENT_NAME = "v1_yolov8_small"

# Model type
# 'yolov8n.pt' -> Nano (Very fast, ideal for drones/Jetson Nano)
# 'yolov8s.pt' -> Small (Balanced)
# 'yolov8m.pt' -> Medium (More precise, requires good GPU)
MODEL_TYPE = 'yolov8s.pt'

# Training params
EPOCHS = 50
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
            0: 'cyclist'  # Class name
        }
    }

    yaml_path = os.path.join(abs_path, 'dataset_config.yaml')

    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"📄 Config file created at: {yaml_path}")
    return yaml_path


def main():
    parser = argparse.ArgumentParser(description="YOLO Training Tool")
    
    # Options
    parser.add_argument('--epochs', type=int, default=None, help="Number of epochs")
    args = parser.parse_args()

    device = check_gpu()
    print(f"--- Starting training with {MODEL_TYPE} ---")

    # 1. Create the treasure map (YAML)
    yaml_file = create_yaml_config()

    # 2. Load the model
    model = YOLO(MODEL_TYPE)

    # 3. Train
    results = model.train(
        data=yaml_file,
        epochs=args.epochs if args.epochs else EPOCHS, 
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        workers=WORKERS,
        project=PROJECT_NAME,
        name=EXPERIMENT_NAME,
        device=device,
        patience=15,   # If it doesn't improve in 15 epochs, stop.
        save=True,     # Save the best model
        exist_ok=True, # If the experiment already exists, it will be overwritten.
        verbose=True   # Show training progress
    )

    print("\n--- Training completed ---")
    best_weight = os.path.join(PROJECT_NAME, EXPERIMENT_NAME, 'weights', 'best.pt')
    print(f"💾 Best model saved at: {best_weight}")

    # 4. Validation (TEST SET)
    print("\n📊 Evaluating final precision in the TEST SET...")
    metrics = model.val(split='test')
    print(f"🎯 mAP50-95 (Test Set): {metrics.box.map:.4f}")
    print(f"🎯 mAP50 (Test Set):    {metrics.box.map50:.4f}")

    # 5. Export to ONNX (Ideal para Isaac Sim / ROS / TensorRT)
    try:
        print("\n📦 Exporting to ONNX...")
        path = model.export(format="onnx", dynamic=True)
        print(f"✅ Model exported for deployment: {path}")
    except Exception as e:
        print(f"⚠️ Export to ONNX failed (non-critical): {e}")


if __name__ == '__main__':
    main()