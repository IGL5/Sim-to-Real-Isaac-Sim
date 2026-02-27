import os
import json
from datetime import datetime
from jinja2 import Environment, FileSystemLoader

class HTMLReportGenerator:
    def __init__(self, templates_dir, project_dir, dataset_out_dir):
        self.env = Environment(loader=FileSystemLoader(templates_dir))
        self.project_dir = project_dir
        self.dataset_out_dir = dataset_out_dir

    def _load_json_safe(self, filepath):
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ Error leyendo {filepath}: {e}")
        return None
        
    def _get_common_context(self, experiment_name, report_title):
        """Carga los JSONs para inyectarlos en la plantilla base"""
        dataset_meta_path = os.path.join(self.dataset_out_dir, "dataset_metadata.json")
        train_meta_path = os.path.join(self.project_dir, experiment_name, "training_metadata.json")
        
        return {
            "report_title": report_title,
            "experiment_name": experiment_name,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "dataset_meta": self._load_json_safe(dataset_meta_path),
            "train_meta": self._load_json_safe(train_meta_path)
        }

    def generate_audit_html(self, output_path, experiment_name, metrics):
        template = self.env.get_template('audit_template.html')
        context = self._get_common_context(experiment_name, "🔎 YOLO Audit Report")
        context["metrics"] = metrics
        
        html_content = template.render(context)
        with open(output_path, "w", encoding='utf-8') as f:
            f.write(html_content)
        print(f"✅ Audit HTML Report generated at: {output_path}")

    def generate_inference_html(self, output_path, experiment_name, stats, overlap_thresh):
        template = self.env.get_template('inference_template.html')
        context = self._get_common_context(experiment_name, "🌍 Real Inference Report")
        context["stats"] = stats
        context["overlap_thresh"] = overlap_thresh
        
        html_content = template.render(context)
        with open(output_path, "w", encoding='utf-8') as f:
            f.write(html_content)
        print(f"✅ Inference HTML Report generated at: {output_path}")