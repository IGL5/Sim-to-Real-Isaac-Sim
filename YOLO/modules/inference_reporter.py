import os
import numpy as np
import json
from datetime import datetime
from . import core_visual_utils as cvu
from .html_generator import HTMLReportGenerator
from . import plot_generator


class InferenceReportGenerator:
    def __init__(self, output_dir, overlap_threshold=0.5):
        self.output_dir = output_dir
        self.plots_dir = os.path.join(output_dir, "..", "plots")
        self.overlaps_dir = os.path.join(self.plots_dir, "suspicious_overlaps_imgs")
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.overlaps_dir, exist_ok=True)
        self.overlap_threshold = overlap_threshold
        self.stats = {
            "total_images": 0,
            "total_detections": 0,
            "confidences": [],
            "bbox_centers_norm": [],
            "overlap_events": []
        }

    def update(self, pred_boxes, confidences, img_shape, filename):
        h, w = img_shape
        self.stats["total_images"] += 1
        self.stats["total_detections"] += len(pred_boxes)
        self.stats["confidences"].extend(confidences)
        
        # Centers for the Heatmap
        for box in pred_boxes:
            cx_abs = (box[0] + box[2]) / 2
            cy_abs = (box[1] + box[3]) / 2
            self.stats["bbox_centers_norm"].append((cx_abs/w, cy_abs/h))
            
        # Overlapping / concentric detections (IoU > 0.45 in the same image)
        iou_matrix = cvu.calculate_iou_matrix(pred_boxes, pred_boxes)
        overlapping_pairs = np.argwhere(np.triu(iou_matrix, k=1) > self.overlap_threshold)
        
        problematic_pairs_indices = []
        if len(overlapping_pairs) > 0:
            problematic_pairs_indices = overlapping_pairs.tolist()
            overlap_img_name = f"OVERLAP_{filename}"
            self.stats["overlap_events"].append({
                "orig_filename": filename,
                "evidence_filename": overlap_img_name,
                "count": len(overlapping_pairs)
            })
            
        return problematic_pairs_indices

    def generate_plots(self):
        print("📊 Generating inference plots...")
        
        # 1. Confidence Histogram
        if self.stats["confidences"]:
            plot_generator.plot_confidence_histogram(
                confs_primary=self.stats["confidences"], label_primary='Detections', color_primary='blue',
                output_path=os.path.join(self.plots_dir, "inference_conf_dist.png"),
                title="Real World Confidence Distribution"
            )

        # 2. Heatmap
        plot_generator.plot_normalized_heatmap(
            self.stats["bbox_centers_norm"],
            os.path.join(self.plots_dir, "inference_heatmap.png"),
            title="Normalized Detection Heatmap (All Resolutions)",
            cmap='magma'
        )

    def generate_html_report(self, experiment_name="yolov8_s_default"):
        """ Calculates final metrics and delegates HTML creation to Jinja2 """
        print("📝 Compiling inference statistics for the report...")
        
        # 1. Summary calculations
        avg_detections = self.stats["total_detections"] / max(1, self.stats["total_images"])
        total_overlaps = sum(event["count"] for event in self.stats["overlap_events"])

        # 2. Pack metrics for the template
        stats_dict = {
            "total_images": self.stats["total_images"],
            "total_detections": self.stats["total_detections"],
            "avg_detections": avg_detections,
            "total_overlaps": total_overlaps,
            "overlap_events": self.stats["overlap_events"]
        }

        stats_dict["confidence_stats"] = cvu.calculate_1d_stats(self.stats["confidences"])
        stats_dict["spatial_stats"] = cvu.calculate_spatial_stats(self.stats["bbox_centers_norm"])

        # 3. Save inference metadata to JSON
        project_dir = os.path.join(os.getcwd(), "cyclist_detector")
        
        inference_metadata = {
            "inference_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "stats": stats_dict,
            "evaluation_params": {
                "overlap_threshold": self.overlap_threshold
            }
        }
        
        inference_json_path = os.path.join(project_dir, experiment_name, "inference_metadata.json")
        try:
            with open(inference_json_path, "w", encoding='utf-8') as f:
                json.dump(inference_metadata, f, indent=4)
            print(f"💾 Inference stats saved at: {inference_json_path}")
        except Exception as e:
            print(f"⚠️ Could not save inference JSON: {e}")

        # 4. Instantiate the generator and create the HTML
        templates_dir = os.path.join(os.getcwd(), "modules", "templates")
        project_dir = os.path.join(os.getcwd(), "cyclist_detector")
        dataset_out_dir = os.path.join(os.getcwd(), "dataset_yolo_output")
        
        generator = HTMLReportGenerator(templates_dir, project_dir, dataset_out_dir)
        
        output_path = os.path.join(self.output_dir, "..", "inference_report.html")
        generator.generate_inference_html(output_path, experiment_name, stats_dict, self.overlap_threshold)