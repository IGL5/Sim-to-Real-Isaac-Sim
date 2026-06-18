from pathlib import Path
from src.core.utils.project_utils import get_available_models
from src.evaluation.utils.comparison_reporter import ComparisonReporter
from src.core import config
from src.core.metadata.audit_builder import AuditMetadata


def find_audits_for_model(model_name, target_json="audit_metadata.json"):
    """
    Searches for metadata files in the root of the model and in saved iterations.
    Returns ONLY valid audits for the selected environment.
    """
    model_dir = Path(config.PROJECT_DIR) / model_name
    
    # 1. Search in the root (temporary / last audit)
    base_audit = model_dir / target_json
    
    # 2. Search in the persistent evaluations (iter_xxx)
    eval_audits = list((model_dir / config.SAVED_EVAL_FOLDER_NAME).glob(f"*/{target_json}"))
    
    all_audits = []
    if base_audit.exists():
        all_audits.append(str(base_audit))
    all_audits.extend([str(p) for p in eval_audits])
    
    # 3. Read the JSONs to verify validity and extract the date
    audit_info = []
    for path in set(all_audits): # set to avoid accidental duplicates
        try:
            audit_dto = AuditMetadata(path).get_html_summary()
            date = audit_dto.get("date", "Fecha desconocida")
            parent_folder = Path(path).parent.name
            
            audit_info.append({
                "path": path,
                "date": date,
                "folder": parent_folder
            })
        except Exception as e:
            print(f"⚠️ Could not read JSON at {path}: {e}")
            
    # Sort from most recent to oldest
    audit_info.sort(key=lambda x: x["date"], reverse=True)
    return audit_info


# 1. GENERAL BENCHMARK
def run_benchmark_flow(available_models):
    print("\n🌍 ENVIRONMENT SELECTION (Benchmark)")
    print("  [1] Simulation (Isaac Sim -> audit_metadata.json)")
    print("  [2] Real World (Real Photos -> real_audit_metadata.json)")
    
    env_input = input("-> Selection [1/2] (default 1): ").strip() or "1"
    target_json = config.FILE_REAL_AUDIT_META if env_input == "2" else config.FILE_AUDIT_META
    env_name = "Real World" if env_input == "2" else "Simulation"
    
    print(f"\n✅ Selected mode: Benchmark in {env_name}.")
    print(f"\n📂 Available models for {env_name}:")
    for i, m in enumerate(available_models):
        print(f"  [{i+1}] {m}")
        
    print("\n✏️  Introduce the NUMBERS of the models you want to compare separated by commas or hyphens for ranges.")
    print("   (Example: 1, 3-5, 8)")
    
    user_input = input("-> Selections: ").strip()
    if not user_input:
        print("Operation cancelled.")
        return {}
        
    selected_indices = []
    for part in user_input.split(','):
        part = part.strip()
        if not part:
            continue
            
        # If it contains a hyphen, it's a range (ej: 1-3)
        if '-' in part:
            range_parts = part.split('-')
            if len(range_parts) == 2 and range_parts[0].strip().isdigit() and range_parts[1].strip().isdigit():
                start_idx = int(range_parts[0].strip())
                end_idx = int(range_parts[1].strip())
                
                # In case the user puts it backwards (ej: 3-1)
                if start_idx > end_idx:
                    start_idx, end_idx = end_idx, start_idx
                    
                for i in range(start_idx, end_idx + 1):
                    idx = i - 1
                    if 0 <= idx < len(available_models):
                        if idx not in selected_indices:
                            selected_indices.append(idx)
                    else:
                        print(f"  ⚠️  [IGNORED] The number {i} is out of range.")
            else:
                print(f"  ⚠️  [IGNORED] '{part}' is not a valid range.")
                
        # If it's a normal number (ej: 4)
        elif part.isdigit():
            idx = int(part) - 1
            if 0 <= idx < len(available_models):
                if idx not in selected_indices:
                    selected_indices.append(idx)
            else:
                print(f"  ⚠️  [IGNORED] The number {part} is out of range.")
        else:
            print(f"  ⚠️  [IGNORED] '{part}' is not a valid format.")
            
    # Reconstruct the final list of models from the unique indices
    selected_models = [available_models[idx] for idx in selected_indices]
            
    final_audits = {}
    print("\n🔍 Verifying audits...")
    
    for model_name in selected_models:
        audits = find_audits_for_model(model_name, target_json=target_json)
        
        if len(audits) == 0:
            print(f"  ❌ [DISCARDED] '{model_name}': No valid audits found.")
            continue
            
        if len(audits) == 1:
            print(f"  ✅ [ACCEPTED] '{model_name}': 1 audit found ({audits[0]['folder']}).")
            selected_audit = audits[0]
        else:
            print(f"  ⚠️  [ATTENTION] '{model_name}': ¡Multiple valid audits found!")
            for i, aud in enumerate(audits):
                print(f"      [{i+1}] Folder: {aud['folder']} | Date: {aud['date']}")
            
            choice = -1
            while choice < 1 or choice > len(audits):
                try:
                    choice = int(input(f"      Select an option [1-{len(audits)}]: "))
                except ValueError:
                    pass
            
            selected_audit = audits[choice-1]
            print(f"  ✅ [ACCEPTED] '{model_name}': Selected option {choice} ({selected_audit['folder']}).")

        # Generate a unique label for the plots
        if selected_audit['folder'] == model_name:
            label = f"{model_name} (Last)"
        else:
            label = f"{model_name} ({selected_audit['folder']})"
            
        # If the user selects the same audit twice, we number it
        base_label = label
        counter = 2
        while label in final_audits:
            label = f"{base_label} v{counter}"
            counter += 1
            
        # Save the audit path and the real folder name
        final_audits[label] = {
            "path": selected_audit["path"],
            "model_root_name": model_name
        }
        
    return final_audits


# 2. SIM-TO-REAL GAP
def run_sim_to_real_flow(available_models):
    print("\n🌉 SIM-TO-REAL GAP ANALYSIS")
    print("📂 Available models:")
    for i, m in enumerate(available_models):
        print(f"  [{i+1}] {m}")
        
    print("\n✏️  Introduce the NUMBER of the model you want to analyze (ej: 1).")
    user_input = input("-> Selection: ").strip()
    
    if not user_input.isdigit() or not (1 <= int(user_input) <= len(available_models)):
        print("❌ Invalid selection.")
        return {}
        
    model_name = available_models[int(user_input) - 1]
    
    print(f"\n🔍 Searching for reports for '{model_name}'...")
    sim_audits = find_audits_for_model(model_name, target_json=config.FILE_AUDIT_META)
    real_audits = find_audits_for_model(model_name, target_json=config.FILE_REAL_AUDIT_META)
    
    if not sim_audits:
        print(f"  ❌ No simulation audit found ({config.FILE_AUDIT_META}).")
    if not real_audits:
        print(f"  ❌ No reality audit found ({config.FILE_REAL_AUDIT_META}).")
        
    if not sim_audits or not real_audits:
        print("\n🛑 ERROR: The model needs to be evaluated in BOTH environments to measure the Gap.")
        return {}
        
    # Automatically take the most recent evaluation of each
    sim_audit = sim_audits[0]
    real_audit = real_audits[0]
    
    print(f"  ✅ Simulation found: {sim_audit['folder']} ({sim_audit['date']})")
    print(f"  ✅ Reality found:   {real_audit['folder']} ({real_audit['date']})")
    
    # We trick the system by registering them as if they were two different models
    return {
        f"{model_name} (Simulation)": {
            "path": sim_audit["path"],
            "model_root_name": model_name
        },
        f"{model_name} (Reality)": {
            "path": real_audit["path"],
            "model_root_name": model_name
        }
    }


# MAIN
def main():
    print("--- ⚖️  MODEL BENCHMARKING & SIM-TO-REAL ANALYSIS ---")
    
    # Load available models at the beginning
    available_models = get_available_models()
    if not available_models:
        print(f"❌ No models found in the '{config.PROJECT_DIR}' directory.")
        return

    # Show main menu
    print("\nWhat type of analysis do you want to perform?")
    print("  [1] General Benchmark (Compare several models in the SAME environment)")
    print("  [2] Sim-to-Real Gap Analysis (Compare Simulation vs Reality of ONE model)")
    
    flow_input = input("-> Selection [1/2] (default 1): ").strip() or "1"
    
    final_audits_to_compare = {}

    # Derive the logic to the corresponding function
    if flow_input == "2":
        final_audits_to_compare = run_sim_to_real_flow(available_models)
    else:
        final_audits_to_compare = run_benchmark_flow(available_models)

    # Final verification
    if not final_audits_to_compare or len(final_audits_to_compare) < 2:
        print("\n🛑 ERROR: Need at least 2 models with valid audits to compare.")
        print("Please run audits on your models or select others.")
        return
        
    print("\n🎉 Selection completed successfully!")
    print("Summary of files ready to compare:")
    shorten = lambda p: p[p.find("YOLO"):] if "YOLO" in p else p
    for m, path in final_audits_to_compare.items():
        print(f"  - {m} -> {shorten(path['path'])}")
        
    print("\n🚀 Passing data to comparison engine...")
    
    # Call the generation module
    try:
        reporter = ComparisonReporter(final_audits_to_compare)
        reporter.generate_comparison()
    except NameError as e:
        print(f"❌ ComparisonReporter not available: {e}")


if __name__ == "__main__":
    main()