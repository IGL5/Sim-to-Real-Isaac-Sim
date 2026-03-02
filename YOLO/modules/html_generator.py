import os
import math
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
        """Load JSONs for template base injection"""
        dataset_meta_path = os.path.join(self.dataset_out_dir, "dataset_metadata.json")
        train_meta_path = os.path.join(self.project_dir, experiment_name, "training_metadata.json")
        
        dataset_meta = self._load_json_safe(dataset_meta_path)
        train_meta = self._load_json_safe(train_meta_path)

        latest_eda = None
        visuals = {}
        
        if dataset_meta and "sessions" in dataset_meta and len(dataset_meta["sessions"]) > 0:
            last_session = dataset_meta["sessions"][-1]
            latest_eda = last_session.get("yolo_split", {}).get("train", {}).get("eda")
            
        if latest_eda:
            # A) Calculate Area
            area_mean = latest_eda.get("bbox_area", {}).get("mean", 0)
            area_std = latest_eda.get("bbox_area", {}).get("std", 0)
            val_max = min(1.0, area_mean + area_std)
            val_min = max(0.0, area_mean - area_std)
            visuals["area_max_side"] = math.sqrt(val_max) * 100 if val_max > 0 else 0
            visuals["area_min_side"] = math.sqrt(val_min) * 100 if val_min > 0 else 0
            
            # B) Aspect Ratio
            ar_mean = latest_eda.get("aspect_ratio", {}).get("mean", 1.0)
            if ar_mean >= 1.0:
                visuals["ar_w"] = 50
                visuals["ar_h"] = 50 / ar_mean
            else:
                visuals["ar_h"] = 50
                visuals["ar_w"] = 50 * ar_mean
                
            # C) Diana
            visuals["gt_cx"] = latest_eda.get("center_x", {}).get("mean", 0.5) * 100
            visuals["gt_cy"] = latest_eda.get("center_y", {}).get("mean", 0.5) * 100
            visuals["gt_rx"] = latest_eda.get("center_x", {}).get("std", 0) * 100
            visuals["gt_ry"] = latest_eda.get("center_y", {}).get("std", 0) * 100

        return {
            "report_title": report_title,
            "experiment_name": experiment_name,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "dataset_meta": dataset_meta,
            "train_meta": train_meta,
            "latest_eda": latest_eda,
            "visuals": visuals
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