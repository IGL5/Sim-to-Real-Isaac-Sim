import os
import subprocess
import argparse
import modules.core_visual_utils as cvu

def get_available_models():
    """ Returns the models that are ready to be audited """
    if not os.path.exists(cvu.PROJECT_DIR):
        return []
    models = []
    for d in os.listdir(cvu.PROJECT_DIR):
        model_dir = os.path.join(cvu.PROJECT_DIR, d)
        if os.path.isdir(model_dir) and os.path.exists(os.path.join(model_dir, "weights", "best.pt")):
            models.append(d)
    return models

def select_models(available_models):
    """ Uses the same range selection logic as in compare_models """
    print("\n📂 Available models:")
    for i, m in enumerate(available_models):
        print(f"  [{i+1}] {m}")
        
    print("\n✏️  Enter the NUMBERS of the models to process (e.g., 1, 3-5).")
    user_input = input("-> Selection: ").strip()
    
    selected_indices = []
    for part in user_input.split(','):
        part = part.strip()
        if not part: continue
            
        if '-' in part:
            range_parts = part.split('-')
            if len(range_parts) == 2 and range_parts[0].strip().isdigit() and range_parts[1].strip().isdigit():
                start_idx, end_idx = int(range_parts[0]), int(range_parts[1])
                if start_idx > end_idx: start_idx, end_idx = end_idx, start_idx
                for i in range(start_idx, end_idx + 1):
                    if 0 <= i-1 < len(available_models) and (i-1) not in selected_indices:
                        selected_indices.append(i-1)
        elif part.isdigit():
            idx = int(part) - 1
            if 0 <= idx < len(available_models) and idx not in selected_indices:
                selected_indices.append(idx)
                
    return [available_models[idx] for idx in selected_indices]

def main():
    parser = argparse.ArgumentParser(description="Batch Orchestrator for YOLO Audits")
    parser.add_argument('--complete', action='store_true', help="Show advanced options (draw_all, conf, video)")
    args = parser.parse_args()

    print("--- 🚂 YOLO BATCH AUDITOR PIPELINE ---")
    
    models = get_available_models()
    if not models:
        print("❌ No models found.")
        return
        
    selected_models = select_models(models)
    if not selected_models:
        print("🛑 Operation cancelled.")
        return

    # 1. Configure the flows to execute
    print("\n⚙️  What audits do you want to include in the Pipeline? (You can put several: e.g. 1,3)")
    print("  [1] Synthetic Audit (Isaac Sim)")
    print("  [2] Real Inference (Photos without labels)")
    print("  [3] Real Audit (Photos with labels)")
    
    # Show Flujo 4 dynamically only if we are in --complete mode
    valid_choices = ['1', '2', '3']
    if args.complete:
        print("  [4] Video Audit (Requires .mp4/.avi file)")
        valid_choices.append('4')
    
    flow_input = input("-> Selection: ").strip()
    flows = []
    for x in flow_input.split(','):
        x = x.strip()
        if x in valid_choices and int(x) not in flows:
            flows.append(int(x))
    
    if not flows:
        print("🛑 No valid flow selected.")
        return

    # Collect paths conditionally according to the chosen flows
    source_dir = None
    labels_dir = None
    video_path = None
    
    if 2 in flows or 3 in flows:
        source_dir = input("\n📁 Path to real images (--source): ").strip()
    if 3 in flows:
        labels_dir = input("📁 Path to real labels (--labels): ").strip()
    if 4 in flows:
        video_path = input("\n📁 Path to video file (--video): ").strip()

    # 2. Advanced Options (--complete)
    adv_flags = []
    conf_val = None
    if args.complete:
        print("\n🛠️  ADVANCED OPTIONS")
        if input("Activate --draw_all for image audits? (y/N): ").strip().lower() == 's':
            adv_flags.append("--draw_all")
            
        conf_val = input("Force a specific --conf? (Leave blank to skip): ").strip()
        if conf_val:
            adv_flags.extend(["--conf", conf_val])

    # Separate HTML flows (1, 2, 3) from video flow (4) to manage --keep and --save well
    report_flows = [f for f in flows if f != 4]

    # 3. BATCH EXECUTION
    print(f"\n🚀 STARTING BATCH FOR {len(selected_models)} MODELS...\n")
    
    for model_idx, model_name in enumerate(selected_models):
        print(f"===========================================================")
        print(f"🤖 PROCESSING MODEL [{model_idx+1}/{len(selected_models)}]: {model_name}")
        print(f"===========================================================")
        
        for flow in flows:
            # Special logic isolated for Video
            if flow == 4:
                cmd = ["python", "visualize_results.py", "--model", model_name, "--video", video_path]
                if conf_val:
                    cmd.extend(["--conf", conf_val])
                print(f"\n▶️  Executing Flow 4 (Video) for {model_name}...")
                subprocess.run(cmd)
                continue # Skip the rest of the loop (does not use --keep or --save)
            
            # Logic for image audits (Flows 1, 2, 3)
            cmd = ["python", "visualize_results.py", "--model", model_name]
            cmd.extend(adv_flags)
            
            if flow == 2:
                cmd.extend(["--source", source_dir])
            elif flow == 3:
                cmd.extend(["--source", source_dir, "--labels", labels_dir])
                
            # Smart KEEP and SAVE logic (calculated only on report_flows)
            idx_in_reports = report_flows.index(flow)
            if idx_in_reports > 0:
                cmd.append("--keep") 
            
            if idx_in_reports == len(report_flows) - 1:
                cmd.append("--save") 
                
            print(f"\n▶️  Executing Flow {flow} for {model_name}...")
            subprocess.run(cmd)

    print("\n🎉 BATCH COMPLETED SUCCESSFULLY.")

if __name__ == "__main__":
    main()