import os
import glob
import json
try:
    from modules.comparison_reporter import ComparisonReporter
except ImportError:
    print("\n[WARNING] El módulo modules/comparison_reporter.py no se ha encontrado.")

PROJECT_DIR = "cyclist_detector"

def get_available_models():
    """Devuelve una lista con los nombres de las carpetas dentro del proyecto."""
    if not os.path.exists(PROJECT_DIR):
        return []
    return [d for d in os.listdir(PROJECT_DIR) if os.path.isdir(os.path.join(PROJECT_DIR, d))]

def find_audits_for_model(model_name):
    """
    Busca archivos audit_metadata.json en la raíz del modelo y en iteraciones guardadas.
    Devuelve una lista de diccionarios con la ruta, la fecha y la carpeta contenedora.
    """
    model_dir = os.path.join(PROJECT_DIR, model_name)
    
    # 1. Buscar en la raíz (auditoría temporal / última)
    base_audit = os.path.join(model_dir, "audit_metadata.json")
    
    # 2. Buscar en las evaluaciones persistentes (iter_xxx)
    eval_audits = glob.glob(os.path.join(model_dir, "evaluations", "*", "audit_metadata.json"))
    
    all_audits = []
    if os.path.exists(base_audit):
        all_audits.append(base_audit)
    all_audits.extend(eval_audits)
    
    # 3. Leer los JSON para extraer la fecha exacta de la auditoría
    audit_info = []
    for path in set(all_audits): # set para evitar duplicados accidentales
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
            pass
            
    # Ordenar de más reciente a más antigua
    audit_info.sort(key=lambda x: x["date"], reverse=True)
    return audit_info

def main():
    print("--- ⚖️  MODEL BENCHMARKING (COMPARACIÓN) ---")
    available_models = get_available_models()
    
    if not available_models:
        print(f"❌ No se encontraron modelos en la carpeta '{PROJECT_DIR}'.")
        return
        
    # 1. Mostrar modelos disponibles
    print("\n📂 Modelos disponibles:")
    for m in available_models:
        print(f"  - {m}")
        
    print("\n✏️  Introduce los nombres de los modelos que deseas comparar separados por comas.")
    print("   (Ejemplo: yolov8_s_default, yolov8_n_custom)")
    
    # 2. Recoger Input del usuario
    user_input = input("-> Modelos: ").strip()
    if not user_input:
        print("Operación cancelada.")
        return
        
    # Limpiar espacios en blanco alrededor de las comas
    selected_models = [m.strip() for m in user_input.split(',')]
    
    final_audits_to_compare = {}
    
    print("\n🔍 Verificando auditorías...")
    # 3. Lógica de validación y resolución de conflictos
    for model_name in selected_models:
        if model_name not in available_models:
            print(f"  ⚠️  [IGNORADO] El modelo '{model_name}' no existe.")
            continue
            
        audits = find_audits_for_model(model_name)
        
        if len(audits) == 0:
            print(f"  ❌ [DESCARTADO] '{model_name}': No tiene ninguna auditoría (audit_metadata.json).")
        elif len(audits) == 1:
            print(f"  ✅ [ACEPTADO] '{model_name}': 1 auditoría encontrada ({audits[0]['folder']}).")
            final_audits_to_compare[model_name] = audits[0]["path"]
        else:
            print(f"  ⚠️  [ATENCIÓN] '{model_name}': ¡Múltiples auditorías encontradas!")
            for i, aud in enumerate(audits):
                print(f"      [{i+1}] Carpeta: {aud['folder']} | Fecha: {aud['date']}")
            
            choice = -1
            while choice < 1 or choice > len(audits):
                try:
                    choice = int(input(f"      ¿Cuál deseas usar para '{model_name}'? [1-{len(audits)}]: "))
                except ValueError:
                    pass
            
            selected_audit = audits[choice-1]
            print(f"  ✅ [ACEPTADO] '{model_name}': Seleccionada la opción {choice} ({selected_audit['folder']}).")
            final_audits_to_compare[model_name] = selected_audit["path"]

    # 4. Comprobación final
    if len(final_audits_to_compare) < 2:
        print("\n🛑 ERROR: Necesitas al menos 2 modelos con auditorías válidas para poder comparar.")
        print("Por favor, realiza auditorías en tus modelos o selecciona otros.")
        return
        
    print("\n🎉 ¡Selección completada con éxito!")
    print("Resumen de archivos listos para comparar:")
    for m, path in final_audits_to_compare.items():
        print(f"  - {m} -> {path}")
        
    print("\n🚀 Pasando datos al motor de comparación...")
    
    # 5. Llamada al módulo de generación
    reporter = ComparisonReporter(final_audits_to_compare)
    reporter.generate_comparison()

if __name__ == "__main__":
    main()