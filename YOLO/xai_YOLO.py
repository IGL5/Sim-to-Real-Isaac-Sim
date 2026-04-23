import os
import sys
import glob
import re
import cv2
import torch
import random
import shutil
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO

try:
    import modules.core_visual_utils as cvu
    PROJECT_NAME = cvu.PROJECT_NAME
except ImportError:
    PROJECT_NAME = "yolo_project"

DATASET_ROOT = os.path.join(os.getcwd(), "dataset_yolo_output")
VAL_IMGS = os.path.join(DATASET_ROOT, "images", "val")
LOCAL_OUTPUT_DIR = os.path.join(os.getcwd(), "xai_output")

# ==========================================
# 1. INTERACTION TOOLS
# ==========================================

def select_trained_model(prompt_title="AVAILABLE MODELS"):
    """Allows selecting a model showing a personalized title."""
    if not os.path.exists(PROJECT_NAME):
        print(f"❌ ERROR: No trained models found in '{PROJECT_NAME}'.")
        sys.exit(1)
        
    available_models = [d for d in os.listdir(PROJECT_NAME) 
                        if os.path.isdir(os.path.join(PROJECT_NAME, d)) 
                        and os.path.exists(os.path.join(PROJECT_NAME, d, "weights", "best.pt"))]
                
    if not available_models:
        print(f"❌ ERROR: No trained models found in '{PROJECT_NAME}'.")
        sys.exit(1)

    def get_yolo_version(model_name):
        match = re.match(r'^yolov?(\d+)', model_name, re.IGNORECASE)
        return int(match.group(1)) if match else 0 
        
    available_models.sort(key=get_yolo_version)
        
    print(f"\n--- 🧠 {prompt_title} ---")
    for i, m in enumerate(available_models):
        print(f"  [{i+1}] {m}")
        
    while True:
        user_input = input(f"\nSelect {prompt_title} [1-{len(available_models)}] (default: 1): ").strip()
        if not user_input:
            exp_name = available_models[0]
            break
        if user_input.isdigit() and 1 <= int(user_input) <= len(available_models):
            exp_name = available_models[int(user_input) - 1]
            break
        print("  ⚠️ Invalid input.")
            
    weights_path = os.path.join(PROJECT_NAME, exp_name, "weights", "best.pt")
    print(f"✅ Model loaded: {exp_name}")
    return weights_path, exp_name

def select_image():
    """Selects an image by path or randomly."""
    print("\n--- 🖼️ IMAGE SELECTION ---")
    ruta = input("Image path (press Enter to take a random validation image): ").strip()
    
    if ruta and os.path.exists(ruta):
        print(f"✅ Image selected: {ruta}")
        return ruta
    elif ruta:
        print("❌ Path not found. Using random image...")
            
    extensiones = ('.jpg', '.jpeg', '.png')
    images = [
        os.path.join(VAL_IMGS, f) 
        for f in os.listdir(VAL_IMGS) 
        if f.lower().endswith(extensiones)
    ]

    if not images:
        print(f"❌ ERROR: No images found in {VAL_IMGS}")
        sys.exit(1)
        
    selected = random.choice(images)
    print(f"✅ Random image: {selected}")
    return selected

# ==========================================
# 2. SPATIAL XAI (WHERE THE NETWORK LOOKS)
# ==========================================

def analyze_spatial(model_path, img_path, save_dir):
    print("\n" + "="*40)
    print("🔭 STARTING SPATIAL ANALYSIS (GRAD-CAM / HEATMAPS)")
    print("="*40)
    
    model = YOLO(model_path)
    
    # Backup original
    shutil.copy2(img_path, os.path.join(save_dir, "01_Original_Image.jpg"))
    
    # 1. Predictions vs Ground Truth
    print("🎨 Generating comparison: Ground Truth vs Prediction...")
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    results = model(img_path, verbose=False)[0]
    img_plotted = results.plot()
    
    lbl_path = img_path.replace("images", "labels").replace(".jpg", ".txt").replace(".png", ".txt")
    if os.path.exists(lbl_path):
        with open(lbl_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    xc, yc, bw, bh = map(float, parts[1:5])
                    x1 = int((xc - bw / 2) * w)
                    y1 = int((yc - bh / 2) * h)
                    x2 = int((xc + bw / 2) * w)
                    y2 = int((yc + bh / 2) * h)
                    cv2.rectangle(img_plotted, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv2.putText(img_plotted, "GT Real", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    cv2.imwrite(os.path.join(save_dir, "00_Predictions_vs_GroundTruth.jpg"), img_plotted)

    # 2. Extract heatmaps from all layers
    print("📸 Extracting heatmaps by layer...")
    img_bgr = cv2.imread(img_path)
    orig_h, orig_w = img_bgr.shape[:2]
    img_resized = cv2.resize(img_bgr, (640, 640))
    img_tensor = torch.from_numpy(img_resized).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    device = next(model.model.parameters()).device
    img_tensor = img_tensor.to(device)

    heat_dir = os.path.join(save_dir, "heatmaps")
    os.makedirs(heat_dir, exist_ok=True)

    for layer_idx in range(len(model.model.model)):
        target_layer = model.model.model[layer_idx]
        layer_type = type(target_layer).__name__
        
        if layer_type in ['Concat', 'Detect', 'Segment', 'Pose']: continue

        activation = {}
        def get_activation(name):
            def hook(module, input, output):
                activation[name] = output[0].detach() if isinstance(output, tuple) else output.detach()
            return hook

        handle = target_layer.register_forward_hook(get_activation('layer'))
        try:
            _ = model(img_tensor, verbose=False)
            act = activation['layer'].squeeze().cpu().numpy()
            
            if len(act.shape) == 3:
                heatmap = np.maximum(np.mean(act, axis=0), 0)
                if np.max(heatmap) > 0: heatmap /= np.max(heatmap)
                
                heatmap_resized = cv2.resize(heatmap, (orig_w, orig_h))
                heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
                superimposed_img = cv2.addWeighted(img_bgr, 0.5, heatmap_color, 0.5, 0)

                cv2.putText(superimposed_img, f"L{layer_idx:02d} ({layer_type})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.imwrite(os.path.join(heat_dir, f"layer_{layer_idx:02d}_heat.jpg"), superimposed_img)
        except Exception:
            pass
        finally:
            handle.remove()
    print(f"✅ Spatial heatmaps saved to: {heat_dir}")

# ==========================================
# 3. STRUCTURAL XAI (WEIGHTS AND BIASES)
# ==========================================

def analyze_structural(base_path, fine_path, save_dir):
    print("\n" + "="*40)
    print("🔬 STARTING STRUCTURAL ANALYSIS (WEIGHTS, BIASES AND SPARSITY)")
    print("="*40)
    
    print("Loading tensors into memory...")
    ckpt_base = torch.load(base_path, map_location='cpu', weights_only=False)
    ckpt_fine = torch.load(fine_path, map_location='cpu', weights_only=False)
    
    w_base = ckpt_base['model'].state_dict() if hasattr(ckpt_base['model'], 'state_dict') else ckpt_base['model'].float().state_dict()
    w_fine = ckpt_fine['model'].state_dict() if hasattr(ckpt_fine['model'], 'state_dict') else ckpt_fine['model'].float().state_dict()

    layers, drift_mags, bias_shifts, dead_base, dead_fine = [], [], [], [], []

    print("Scanning layers...")
    layer_indices = sorted(list(set([int(k.split('.')[1]) for k in w_base.keys() if 'model.' in k and k.split('.')[1].isdigit()])))

    for idx in layer_indices:
        layer_weights_b = [w_base[k].numpy().flatten() for k in w_base.keys() if f"model.{idx}." in k and "weight" in k]
        layer_weights_f = [w_fine[k].numpy().flatten() for k in w_fine.keys() if f"model.{idx}." in k and "weight" in k]
        
        layer_biases_b = [w_base[k].numpy().flatten() for k in w_base.keys() if f"model.{idx}." in k and "bias" in k]
        layer_biases_f = [w_fine[k].numpy().flatten() for k in w_fine.keys() if f"model.{idx}." in k and "bias" in k]

        if layer_weights_b and layer_weights_f:
            b_flat = np.concatenate(layer_weights_b)
            f_flat = np.concatenate(layer_weights_f)
            
            # 1. Transfer Learning Drift (L1 Absoluto)
            if b_flat.shape == f_flat.shape:
                drift = float(np.mean(np.abs(f_flat - b_flat)))
            else:
                # If architecture changed (e.g. different number of classes in the final layer)
                drift = float(np.abs(np.mean(np.abs(f_flat)) - np.mean(np.abs(b_flat))))
                print(f"  ⚠️ Layer L{idx}: Architecture change detected (Base: {len(b_flat)} weights vs Fine: {len(f_flat)} weights).")
            
            # 2. Sparsity (% Dead < 0.01)
            db = float(np.sum(np.abs(b_flat) < 0.01) / len(b_flat) * 100)
            df = float(np.sum(np.abs(f_flat) < 0.01) / len(f_flat) * 100)
            
            # 3. Bias Shift
            b_shift = 0.0
            if layer_biases_b and layer_biases_f:
                bb_flat = np.concatenate(layer_biases_b)
                bf_flat = np.concatenate(layer_biases_f)
                if bb_flat.shape == bf_flat.shape:
                    b_shift = float(np.mean(np.abs(bf_flat - bb_flat)))
                else:
                    b_shift = float(np.abs(np.mean(np.abs(bf_flat)) - np.mean(np.abs(bb_flat))))

            layers.append(f"L{idx}")
            drift_mags.append(drift)
            bias_shifts.append(b_shift)
            dead_base.append(db)
            dead_fine.append(df)

    # --- PLOTTING RESULTS ---
    sns.set_theme(style="darkgrid", context="paper")
    fig, axes = plt.subplots(3, 1, figsize=(14, 15), sharex=True)
    
    # Graph 1: Transfer Learning Drift
    sns.barplot(x=layers, y=drift_mags, color="#e74c3c", ax=axes[0])
    axes[0].set_title("1. Transfer Learning Drift (Weight Drift)", fontweight='bold', fontsize=14)
    axes[0].set_ylabel("Absolute Change")
    axes[0].axvline(x=9.5, color='black', linestyle='--', alpha=0.5, label="End of Backbone")
    axes[0].legend()

    # Graph 2: Bias Shift
    sns.barplot(x=layers, y=bias_shifts, color="#9b59b6", ax=axes[1])
    axes[1].set_title("2. Illumination/Activation Shift (Bias Shift)", fontweight='bold', fontsize=14)
    axes[1].set_ylabel("Bias Shift")
    axes[1].axvline(x=9.5, color='black', linestyle='--', alpha=0.5)

    # Graph 3: Sparsity (Catastrophic Forgetting)
    axes[2].plot(layers, dead_base, marker='o', color='blue', label='Base (COCO)', linewidth=2)
    axes[2].plot(layers, dead_fine, marker='^', color='orange', label='Fine-Tuned', linewidth=2)
    axes[2].set_title("3. Sparsity Profile (% Dead Neurons - Catastrophic Forgetting)", fontweight='bold', fontsize=14)
    axes[2].set_ylabel("% Inactive Weights")
    axes[2].set_xlabel("Network Layers", fontweight='bold')
    axes[2].axvline(x=9.5, color='black', linestyle='--', alpha=0.5)
    axes[2].legend()

    plt.tight_layout()
    out_img = os.path.join(save_dir, "02_Structural_Analysis.png")
    plt.savefig(out_img, dpi=300)
    print(f"✅ Structural Analysis saved to: {out_img}")

# ==========================================
# 4. MAIN ORCHESTRATOR
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="XAI Suite for YOLO")
    args = parser.parse_args()

    print("========================================")
    print("🚀 YOLO XAI SUITE - COMPLETE EXPLAINABILITY")
    print("========================================")
    
    print("\nWhat type of XAI analysis do you want to perform?")
    print("  [1] SPATIAL XAI    (Heatmaps on 1 image. Requires: 1 Model, 1 Image)")
    print("  [2] STRUCTURAL XAI (Weight drift and forgetting. Requires: Base Model, Fine-Tuned Model)")
    print("  [3] FULL SUITE  (Both analyses. Requires: 2 Models, 1 Image)")
    
    modo = input("\n-> Select option [1/2/3] (default 3): ").strip()
    if not modo: modo = "3"
    
    # Prepare local directory
    if os.path.exists(LOCAL_OUTPUT_DIR):
        shutil.rmtree(LOCAL_OUTPUT_DIR)
    os.makedirs(LOCAL_OUTPUT_DIR)

    # Flow according to selection
    if modo == "1":
        path_mod, _ = select_trained_model("MODEL TO ANALYZE")
        path_img = select_image()
        analyze_spatial(path_mod, path_img, LOCAL_OUTPUT_DIR)

    elif modo == "2":
        path_base, _ = select_trained_model("BASE MODEL (e.g. COCO)")
        path_fine, _ = select_trained_model("FINE-TUNED MODEL")
        analyze_structural(path_base, path_fine, LOCAL_OUTPUT_DIR)

    elif modo == "3":
        path_base, _ = select_trained_model("BASE MODEL (e.g. COCO)")
        path_fine, exp_fine = select_trained_model("FINE-TUNED MODEL (We will analyze its heatmaps)")
        path_img = select_image()
        analyze_spatial(path_fine, path_img, LOCAL_OUTPUT_DIR)
        analyze_structural(path_base, path_fine, LOCAL_OUTPUT_DIR)
    else:
        print("❌ Invalid option.")
        sys.exit(1)

    print(f"\n🎉 All results have been saved to the folder: {LOCAL_OUTPUT_DIR} !")

if __name__ == '__main__':
    main()