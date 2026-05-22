import yaml
import pandas as pd
import time
from datetime import datetime
from pathlib import Path
from src.core.metadata.base_manager import BaseMetadataManager

class TrainMetadata(BaseMetadataManager):
    def __init__(self, filepath):
        super().__init__(filepath)

    def record_experiment_info(self, project_name, experiment_name, start_time_secs):
        """
        Calcula la duración exacta del entrenamiento y formatea las fechas.
        El script solo le pasa el time.time() del momento en que arrancó.
        """
        # Cálculos de tiempo
        duration_secs = time.time() - start_time_secs
        duration_formatted = time.strftime("%H:%M:%S", time.gmtime(duration_secs))
        
        # Convertir el float de inicio a string legible
        start_date_str = datetime.fromtimestamp(start_time_secs).strftime("%Y-%m-%d %H:%M:%S")

        self.update_section("experiment_info", {
            "project_name": project_name,
            "experiment_name": experiment_name,
            "start_date": start_date_str,
            "duration_seconds": round(duration_secs, 2),
            "duration_formatted": duration_formatted
        })

    def record_hardware(self, device_type, device_name):
        self.update_section("hardware", {
            "device": device_type,
            "device_name": device_name
        })

    def extract_yolo_training_data(self, model_base, epochs_requested, experiment_dir):
        """
        El Builder actúa como Parser. Lee los archivos internos que YOLO 
        acaba de generar y extrae automáticamente los hiperparámetros, 
        la mejor época y las técnicas de Data Augmentation.
        """
        exp_path = Path(experiment_dir)
        args_path = exp_path / 'args.yaml'
        csv_path = exp_path / 'results.csv'

        # Valores por defecto por si falla la lectura
        epochs_run = 0
        best_epoch = 0
        yolo_args = {}

        # 1. Leer hiperparámetros y aumentación de args.yaml
        if args_path.exists():
            with open(args_path, 'r', encoding='utf-8') as f:
                yolo_args = yaml.safe_load(f)

        # 2. Leer las épocas del results.csv
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df.columns = df.columns.str.strip() # YOLO deja espacios en las columnas
            epochs_run = len(df)
            
            # YOLO suele guardar la métrica combinada, buscamos la fila con mayor mAP50-95
            if 'metrics/mAP50-95(B)' in df.columns:
                best_epoch = int(df['metrics/mAP50-95(B)'].idxmax() + 1)
            else:
                best_epoch = epochs_run # Fallback

        # --- GUARDAR HIPERPARÁMETROS ---
        self.update_section("hyperparameters", {
            "model_base": str(model_base),
            "epochs_requested": int(epochs_requested),
            "epochs_run": int(epochs_run),
            "best_epoch": int(best_epoch),
            "patience": int(yolo_args.get('patience', 0)),
            # YOLO a veces pone 'freeze' como None en lugar de 0
            "freeze_layers": int(yolo_args.get('freeze') or 0), 
            "learning_rate": float(yolo_args.get('lr0', 0.0)),
            "img_size": int(yolo_args.get('imgsz', 0)),
            "batch_size": int(yolo_args.get('batch', 0)),
            "workers": int(yolo_args.get('workers', 0))
        })

        # --- GUARDAR DATA AUGMENTATION ---
        aug_keys = ['hsv_h', 'hsv_s', 'hsv_v', 'degrees', 'translate', 'scale', 
                    'shear', 'perspective', 'flipud', 'fliplr', 'mosaic', 'mixup', 'copy_paste']
        
        aug_data = {k: yolo_args.get(k, 0.0) for k in aug_keys}
        self.update_section("data_augmentation", aug_data)

    def record_metrics(self, map50_95, map50):
        self.update_section("metrics_test_set", {
            "mAP50_95": round(float(map50_95), 4),
            "mAP50": round(float(map50), 4)
        })

    def record_artifacts(self, best_weights_path, onnx_model_path):
        weights_path = Path(best_weights_path)
        weight_size_mb = 0.0
        
        if weights_path.exists():
            weight_size_mb = round(weights_path.stat().st_size / (1024 * 1024), 2)

        self.update_section("artifacts", {
            "best_weights": str(weights_path),
            "best_weights_mb": weight_size_mb,
            "onnx_model": str(onnx_model_path) if onnx_model_path else "Not exported"
        })