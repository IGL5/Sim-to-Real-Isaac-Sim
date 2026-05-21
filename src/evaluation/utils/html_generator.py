import os
import math
import json
from datetime import datetime
from jinja2 import Environment, FileSystemLoader
from src.core import config

class HTMLReportGenerator:
    def __init__(self):
        """Initialize HTML Report Generator with Jinja2 environment"""
        self.env = Environment(loader=FileSystemLoader(config.TEMPLATES_DIR))
        self.project_dir = config.PROJECT_DIR

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
        dataset_meta_path = os.path.join(self.project_dir, experiment_name, config.METADATA_FOLDER_NAME, config.FILE_DATASET_META)
        train_meta_path = os.path.join(self.project_dir, experiment_name, config.METADATA_FOLDER_NAME, config.FILE_TRAIN_META)
        
        dataset_meta = self._load_json_safe(dataset_meta_path)
        train_meta = self._load_json_safe(train_meta_path)

        latest_eda = None
        latest_session = None
        visuals = {}
        
        if dataset_meta and "sessions" in dataset_meta and len(dataset_meta["sessions"]) > 0:
            latest_session = dataset_meta["sessions"][-1]
            latest_eda = latest_session.get("yolo_split", {}).get("train", {}).get("eda")

            coverage = latest_session.get("spatial_coverage", {})
            if coverage:
                cam_max = coverage.get("camera_distance_range", [0, 0])[1]
                dist_max = coverage.get("distractor_max_radius", 0)
                obj_max = coverage.get("objects_max_radius", 0)
                
                abs_max = max(cam_max, dist_max, obj_max, 1.0)
                
                visuals["cov_cam"] = (cam_max / abs_max) * 50
                visuals["cov_dist"] = (dist_max / abs_max) * 50
                visuals["cov_obj"] = (obj_max / abs_max) * 50
            
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
            "latest_session": latest_session,
            "visuals": visuals
        }

    def generate_audit_html(self, file_output_path, experiment_name, metrics):
        template = self.env.get_template('audit_template.html')
        context = self._get_common_context(experiment_name, "YOLO Audit Report")
        context["metrics"] = metrics
        
        html_content = template.render(context)
        with open(file_output_path, "w", encoding='utf-8') as f:
            f.write(html_content)
        print(f"✅ Audit HTML Report generated at: {file_output_path}")

    def generate_inference_html(self, file_output_path, experiment_name, stats, overlap_thresh):
        template = self.env.get_template('inference_template.html')
        context = self._get_common_context(experiment_name, "Real Inference Report")
        context["stats"] = stats
        context["overlap_thresh"] = overlap_thresh
        
        html_content = template.render(context)
        with open(file_output_path, "w", encoding='utf-8') as f:
            f.write(html_content)
        print(f"✅ Inference HTML Report generated at: {file_output_path}")

    def generate_comparison_html(self, file_output_path, comparison_data, chart_data):
        template = self.env.get_template('compare_template.html')
        context = {
            "report_title": "Benchmark: Model Comparison",
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "data": comparison_data,
            "models": list(comparison_data.keys()),
            "chart_data": chart_data
        }
        
        html_content = template.render(context)
        with open(file_output_path, "w", encoding='utf-8') as f:
            f.write(html_content)
        print(f"✅ Comparison HTML Report generated at: {file_output_path}")