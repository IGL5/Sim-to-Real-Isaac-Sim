from pathlib import Path
from datetime import datetime
import src.core.config as config
from src.evaluation.utils.html_generator import HTMLReportGenerator
from src.core.metadata.audit_builder import AuditMetadata
from src.core.metadata.train_builder import TrainMetadata
from src.core.metadata.dataset_builder import DatasetMetadata

class ComparisonReporter:
    def __init__(self, models_dict):
        self.models_dict = models_dict
        self.project_dir = config.PROJECT_DIR
        self.output_dir = Path(config.COMPARISON_OUTPUT_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data = {}


    def load_all_data(self):
        """Load the metrics from the audit, train and dataset JSONs for each model"""
        for display_label, info in self.models_dict.items():
            audit_path = info["path"]
            real_model_name = info["model_root_name"] 
            
            audit_meta = AuditMetadata(audit_path).get_html_summary()
            
            model_root = Path(self.project_dir) / real_model_name
            train_meta_path = model_root / config.METADATA_FOLDER_NAME / config.FILE_TRAIN_META
            dataset_meta_path = model_root / config.METADATA_FOLDER_NAME / config.FILE_DATASET_META
            
            train_meta = TrainMetadata(train_meta_path).get_html_summary()
            dataset_meta = DatasetMetadata(dataset_meta_path).get_html_summary()

            self.data[display_label] = {
                "audit": audit_meta,
                "train": train_meta,
                "dataset": dataset_meta
            }

    def prepare_chart_data(self):
        """ Extracts the key metrics in clean arrays to inject them into Chart.js (Multi-Class) """
        labels = list(self.data.keys())
        aliases = [f"M{i+1}" for i in range(len(labels))]
        
        # 1. Encontrar TODAS las clases únicas entre los modelos comparados usando get_html_summary() de AuditMetadata
        all_classes = set()
        for m in labels:
            audit = self.data[m].get("audit") or {}
            classes = audit.get("classes") or []
            for c in classes:
                all_classes.add(c.get("name"))

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
            
            # Velocidad y Tiempo de Generación (Siempre son Globales)
            speeds = audit.get("speeds") or {}
            chart_data["global_metrics"]["fps"].append(speeds.get("fps", 0))
            chart_data["global_metrics"]["inf_time"].append(speeds.get("inference_ms", 0))
            chart_data["global_metrics"]["nms_time"].append(speeds.get("postprocess_ms", 0))
            chart_data["global_metrics"]["pre_time"].append(speeds.get("preprocess_ms", 0))
            
            ds_meta = self.data[m].get("dataset") or {}
            latest_session = ds_meta.get("latest_session") or {}
            t = latest_session.get("performance", {}).get("average_time_per_frame_seconds", 0.0)
            chart_data["global_metrics"]["gen_time"].append(t)
            
            # Métricas Globales / Macro (Dividimos precisión y recall por 100 para rango 0..1 en los gráficos)
            global_metrics = audit.get("global") or {}
            chart_data["metrics_by_class"]["Global"]["map50"].append(global_metrics.get("map50", 0))
            chart_data["metrics_by_class"]["Global"]["map50_95"].append(global_metrics.get("map50_95", 0))
            chart_data["metrics_by_class"]["Global"]["precision"].append(global_metrics.get("precision", 0) / 100.0)
            chart_data["metrics_by_class"]["Global"]["recall"].append(global_metrics.get("recall", 0) / 100.0)
            chart_data["metrics_by_class"]["Global"]["f1"].append(global_metrics.get("f1", 0))
            
            conf_stats_global = audit.get("confidence_global") or {}
            chart_data["metrics_by_class"]["Global"]["conf_tp"].append(conf_stats_global.get("True_Positives", {}).get("mean", 0))
            chart_data["metrics_by_class"]["Global"]["conf_fp"].append(conf_stats_global.get("False_Positives", {}).get("mean", 0))
            
            # Métricas Per-Class
            classes = audit.get("classes") or []
            class_map = {c_item.get("name"): c_item for c_item in classes}
            for c in class_list:
                if c == "Global": continue
                c_data = class_map.get(c) or {}
                
                chart_data["metrics_by_class"][c]["map50"].append(c_data.get("ap50", 0))
                chart_data["metrics_by_class"][c]["map50_95"].append(c_data.get("ap50_95", 0))
                chart_data["metrics_by_class"][c]["precision"].append(c_data.get("precision", 0) / 100.0)
                chart_data["metrics_by_class"][c]["recall"].append(c_data.get("recall", 0) / 100.0)
                chart_data["metrics_by_class"][c]["f1"].append(c_data.get("f1", 0))
                
                chart_data["metrics_by_class"][c]["conf_tp"].append(c_data.get("conf_tp_mean", 0))
                chart_data["metrics_by_class"][c]["conf_fp"].append(c_data.get("conf_fp_mean", 0))

        return chart_data

    def generate_comparison(self):
        self.load_all_data()
        
        # Get the data ready for JS
        chart_data = self.prepare_chart_data()
        
        print("📝 Compiling interactive HTML dashboard...")
        
        generator = HTMLReportGenerator()
        html_context = {
            "report_title": "YOLO Model Comparison",
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "models": list(self.data.keys()),
            "data": self.data,
            "chart_data": chart_data
        }
        generator.generate_comparison_html(str(self.output_dir / "comparison_report.html"), html_context)