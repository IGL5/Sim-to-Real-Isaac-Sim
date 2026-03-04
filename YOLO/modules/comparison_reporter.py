import os
import json
from . import core_visual_utils as cvu
from .html_generator import HTMLReportGenerator

class ComparisonReporter:
    def __init__(self, models_dict):
        self.models_dict = models_dict
        self.project_dir = cvu.PROJECT_DIR
        self.output_dir = os.path.join(os.getcwd(), "comparison_report")
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
            train_meta_path = os.path.join(model_root, "metadata", "training_metadata.json")
            dataset_meta_path = os.path.join(model_root, "metadata", "dataset_metadata.json")
            
            train_meta = self._load_json(train_meta_path)
            dataset_meta = self._load_json(dataset_meta_path)

            self.data[display_label] = {
                "audit": audit_meta,
                "train": train_meta,
                "dataset": dataset_meta
            }

    def prepare_chart_data(self):
        """ Extracts the key metrics in clean arrays to inject them into Chart.js """
        labels = list(self.data.keys())
        
        chart_data = {
            "labels": labels,
            "map50": [],
            "precision": [],
            "recall": [],
            "conf_tp": [],
            "conf_fp": [],
            "gen_time": []
        }
        
        for m in labels:
            # Secure extraction with get() if a json was incomplete
            audit = self.data[m].get("audit") or {}
            metrics = audit.get("metrics") or {}
            
            chart_data["map50"].append(metrics.get("ap50", 0))
            chart_data["precision"].append(metrics.get("precision", 0))
            chart_data["recall"].append(metrics.get("recall", 0))
            
            conf_stats = metrics.get("confidence_stats") or {}
            chart_data["conf_tp"].append(conf_stats.get("True_Positives", {}).get("mean", 0))
            chart_data["conf_fp"].append(conf_stats.get("False_Positives", {}).get("mean", 0))
            
            ds_meta = self.data[m].get("dataset") or {}
            t = 0.0
            if "sessions" in ds_meta and len(ds_meta["sessions"]) > 0:
                t = ds_meta["sessions"][-1].get("performance", {}).get("average_time_per_frame_seconds", 0.0)
            chart_data["gen_time"].append(t)
            
        return chart_data

    def generate_comparison(self):
        self.load_all_data()
        
        # Get the data ready for JS
        chart_data = self.prepare_chart_data()
        
        print("📝 Compiling interactive HTML dashboard...")
        templates_dir = os.path.join(os.getcwd(), "modules", "templates")
        dataset_out_dir = os.path.join(os.getcwd(), "dataset_yolo_output")
        
        generator = HTMLReportGenerator(templates_dir, self.project_dir, dataset_out_dir)
        
        output_path = os.path.join(self.output_dir, "comparison_report.html")
        
        # Pass both the raw data (for the table) and the graph data to Jinja2
        generator.generate_comparison_html(output_path, self.data, chart_data)
        print(f"\n✅ Comparison completed! Report available at: {output_path}")