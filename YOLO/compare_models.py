import os
import glob
import json

try:
    from modules.comparison_reporter import ComparisonReporter
except ImportError:
    print("\n[WARNING] The module modules/comparison_reporter.py was not found.")

PROJECT_DIR = "cyclist_detector"

def get_available_models():
    """Returns a list of model names in the project directory."""
    if not os.path.exists(PROJECT_DIR):
        return []
    return [d for d in os.listdir(PROJECT_DIR) if os.path.isdir(os.path.join(PROJECT_DIR, d))]

def find_audits_for_model(model_name):
    """
    Searches for audit_metadata.json files in the root of the model and in saved iterations.
    Returns ONLY valid audits.
    """
    model_dir = os.path.join(PROJECT_DIR, model_name)
    
    # 1. Search in the root (temporary / last audit)
    base_audit = os.path.join(model_dir, "audit_metadata.json")
    
    # 2. Search in the persistent evaluations (iter_xxx)
    eval_audits = glob.glob(os.path.join(model_dir, "evaluations", "*", "audit_metadata.json"))
    
    all_audits = []
    if os.path.exists(base_audit):
        all_audits.append(base_audit)
    all_audits.extend(eval_audits)
    
    # 3. Read the JSONs to verify validity and extract the date
    audit_info = []
    for path in set(all_audits): # set to avoid accidental duplicates
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                date = data.get("audit_date", "Fecha desconocida")
                parent_folder = os.path.basename(os.path.dirname(path))
                
                audit_info.append({
                    "path": path,
                    "date": date,
                    "folder": parent_folder
                })
        except Exception:
            pass # If the JSON is corrupt, we ignore it automatically
            
    # Sort from most recent to oldest
    audit_info.sort(key=lambda x: x["date"], reverse=True)
    return audit_info


def main():
    print("--- ⚖️  MODEL BENCHMARKING (COMPARATION) ---")
    available_models = get_available_models()
    
    if not available_models:
        print(f"❌ No models found in the '{PROJECT_DIR}' directory.")
        return
        
    # 1. Show available models NUMERATED
    print("\n📂 Available models:")
    for i, m in enumerate(available_models):
        print(f"  [{i+1}] {m}")
        
    print("\n✏️  Introduce the NUMBERS of the models you want to compare separated by commas.")
    print("   (Example: 1, 3)")
    
    # 2. Get Input numeric from user
    user_input = input("-> Selections: ").strip()
    if not user_input:
        print("Operation cancelled.")
        return
        
    # Process and validate numbers
    selected_models = []
    for part in user_input.split(','):
        part = part.strip()
        if part.isdigit():
            idx = int(part) - 1
            if 0 <= idx < len(available_models):
                selected_models.append(available_models[idx])
            else:
                print(f"  ⚠️  [IGNORED] The number {part} is out of range.")
        else:
            print(f"  ⚠️  [IGNORED] '{part}' is not a valid number.")
            
    final_audits_to_compare = {}
    
    print("\n🔍 Verifying audits...")
    
    # 3. Validation logic
    for model_name in selected_models:
        audits = find_audits_for_model(model_name)
        
        if len(audits) == 0:
            print(f"  ❌ [DISCARDED] '{model_name}': No valid audits found.")
            continue
            
        selected_audit = None
        
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
        while label in final_audits_to_compare:
            label = f"{base_label} v{counter}"
            counter += 1
            
        # Save the audit path and the real folder name
        final_audits_to_compare[label] = {
            "path": selected_audit["path"],
            "model_root_name": model_name
        }

    # 4. Final check
    if len(final_audits_to_compare) < 2:
        print("\n🛑 ERROR: Need at least 2 models with valid audits to compare.")
        print("Please run audits on your models or select others.")
        return
        
    print("\n🎉 Selection completed successfully!")
    print("Summary of files ready to compare:")
    for m, path in final_audits_to_compare.items():
        print(f"  - {m} -> {path}")
        
    print("\n🚀 Passing data to comparison engine...")
    
    # 5. Call to the generation module
    try:
        reporter = ComparisonReporter(final_audits_to_compare)
        reporter.generate_comparison()
    except NameError:
        pass

if __name__ == "__main__":
    main()