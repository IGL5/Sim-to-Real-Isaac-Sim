import os
import json
import src.core.config as config
from src.evaluation.utils.html_generator import HTMLReportGenerator

class ComparisonReporter:
    def __init__(self, models_dict):
        self.models_dict = models_dict
        self.project_dir = config.PROJECT_DIR
        self.output_dir = config.COMPARISON_OUTPUT_DIR
        os.makedirs(self.output_dir, exist_ok=True)
        self.data = {}

    def _load_json(self, path):
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None

    def load_all_data(self):
        for display_label, info in self.models_dict.items():
            audit_path = info["path"]
            real_model_name = info["model_root_name"] 
            
            audit_meta = self._load_json(audit_path)
            
            model_root = os.path.join(self.project_dir, real_model_name)
            train_meta_path = os.path.join(model_root, config.METADATA_FOLDER_NAME, config.FILE_TRAIN_META)
            dataset_meta_path = os.path.join(model_root, config.METADATA_FOLDER_NAME, config.FILE_DATASET_META)
            
            train_meta = self._load_json(train_meta_path)
            dataset_meta = self._load_json(dataset_meta_path)

            self.data[display_label] = {
                "audit": audit_meta,
                "train": train_meta,
                "dataset": dataset_meta
            }

    def prepare_chart_data(self):
        """ Extracts the key metrics in clean arrays to inject them into Chart.js (Multi-Class) """
        labels = list(self.data.keys())
        aliases = [f"M{i+1}" for i in range(len(labels))]
        
        # 1. Encontrar TODAS las clases únicas entre los modelos comparados
        all_classes = set()
        for m in labels:
            audit = self.data[m].get("audit") or {}
            metrics = audit.get("metrics") or {}
            per_class = metrics.get("per_class") or {}
            for c_name in per_class.keys():
                all_classes.add(c_name)

        class_list = ["Global"] + sorted(list(all_classes))
        
        # 2. Inicializar la super-estructura de datos
        chart_data = {
            "labels": labels,
            "aliases": aliases,
            "classes": class_list,
            "global_metrics": {
                "fps": [], "inf_time": [], "nms_time": [], "pre_time": [], "gen_time": []
            },
            "metrics_by_class": {}
        }
        
        for c in class_list:
            chart_data["metrics_by_class"][c] = {
                "map50": [], "map50_95": [], "precision": [], "recall": [], "f1": [], "conf_tp": [], "conf_fp": []
            }
            
        # 3. Rellenar los datos
        for m in labels:
            audit = self.data[m].get("audit") or {}
            metrics = audit.get("metrics") or {}
            
            # Velocidad y Tiempo de Generación (Siempre son Globales)
            speed_stats = metrics.get("speed_stats") or {}
            chart_data["global_metrics"]["fps"].append(speed_stats.get("fps", 0))
            chart_data["global_metrics"]["inf_time"].append(speed_stats.get("inference_ms", 0))
            chart_data["global_metrics"]["nms_time"].append(speed_stats.get("postprocess_ms", 0))
            chart_data["global_metrics"]["pre_time"].append(speed_stats.get("preprocess_ms", 0))
            
            ds_meta = self.data[m].get("dataset") or {}
            t = 0.0
            if "sessions" in ds_meta and len(ds_meta["sessions"]) > 0:
                t = ds_meta["sessions"][-1].get("performance", {}).get("average_time_per_frame_seconds", 0.0)
            chart_data["global_metrics"]["gen_time"].append(t)
            
            # Métricas Globales / Macro
            chart_data["metrics_by_class"]["Global"]["map50"].append(metrics.get("map_50", 0))
            chart_data["metrics_by_class"]["Global"]["map50_95"].append(metrics.get("map_50_95", 0))
            chart_data["metrics_by_class"]["Global"]["precision"].append(metrics.get("precision", 0))
            chart_data["metrics_by_class"]["Global"]["recall"].append(metrics.get("recall", 0))
            chart_data["metrics_by_class"]["Global"]["f1"].append(metrics.get("f1", 0))
            
            conf_stats = metrics.get("confidence_stats") or {}
            chart_data["metrics_by_class"]["Global"]["conf_tp"].append(conf_stats.get("True_Positives", {}).get("mean", 0))
            chart_data["metrics_by_class"]["Global"]["conf_fp"].append(conf_stats.get("False_Positives", {}).get("mean", 0))
            
            # Métricas Per-Class
            per_class = metrics.get("per_class") or {}
            for c in class_list:
                if c == "Global": continue
                c_data = per_class.get(c) or {}
                
                chart_data["metrics_by_class"][c]["map50"].append(c_data.get("ap_50", 0))
                chart_data["metrics_by_class"][c]["map50_95"].append(c_data.get("ap_50_95", 0))
                chart_data["metrics_by_class"][c]["precision"].append(c_data.get("precision", 0))
                chart_data["metrics_by_class"][c]["recall"].append(c_data.get("recall", 0))
                chart_data["metrics_by_class"][c]["f1"].append(c_data.get("f1", 0))
                
                c_conf = c_data.get("confidence_stats") or {}
                chart_data["metrics_by_class"][c]["conf_tp"].append(c_conf.get("True_Positives", {}).get("mean", 0))
                chart_data["metrics_by_class"][c]["conf_fp"].append(c_conf.get("False_Positives", {}).get("mean", 0))

        return chart_data

    def generate_comparison(self):
        self.load_all_data()
        
        # Get the data ready for JS
        chart_data = self.prepare_chart_data()
        
        print("📝 Compiling interactive HTML dashboard...")
        
        generator = HTMLReportGenerator()
        # Pass both the raw data (for the table) and the graph data to Jinja2
        generator.generate_comparison_html(os.path.join(self.output_dir, "comparison_report.html"), self.data, chart_data)