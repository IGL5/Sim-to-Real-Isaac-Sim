from pathlib import Path
import numpy as np
from datetime import datetime
from collections import defaultdict
from src.evaluation.utils.html_generator import HTMLReportGenerator
from src.core.metadata.inference_builder import InferenceMetadata
from src.core.metadata.dataset_builder import DatasetMetadata
from src.core.metadata.train_builder import TrainMetadata
from src.evaluation.utils import plot_generator
from src.core.utils import math_utils as mu
from src.core import config

class InferenceReportGenerator:
    def __init__(self, overlap_threshold=0.5, class_names=None):
        self.output_dir = config.EVALUATION_OUTPUT_DIR
        self.plots_dir = Path(config.PLOTS_EVAL_DIR) / "inference"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        self.overlap_threshold = overlap_threshold
        self.class_names = class_names if class_names else {}
        self.MAX_OVERLAPS_HTML = 50
        
        self.stats = {
            "total_images": 0,
            "total_detections": 0,
            "confidences": [],
            "bbox_centers_norm": [],
            "overlap_events": [],
            "speeds": {"preprocess": [], "inference": [], "postprocess": []}
        }
        
        self.class_stats = defaultdict(lambda: {
            "detections": 0,
            "confidences": [],
            "bbox_centers": []
        })

    def update(self, pred_boxes, pred_classes, confidences, img_shape, filename, speed_dict=None):
        h, w = img_shape
        self.stats["total_images"] += 1
        self.stats["total_detections"] += len(pred_boxes)
        self.stats["confidences"].extend(confidences)
        
        if speed_dict:
            self.stats["speeds"]["preprocess"].append(speed_dict.get('preprocess', 0))
            self.stats["speeds"]["inference"].append(speed_dict.get('inference', 0))
            self.stats["speeds"]["postprocess"].append(speed_dict.get('postprocess', 0))
        
        for i, box in enumerate(pred_boxes):
            c_id = pred_classes[i]
            cx_abs = (box[0] + box[2]) / 2
            cy_abs = (box[1] + box[3]) / 2
            
            # Global
            self.stats["bbox_centers_norm"].append((cx_abs/w, cy_abs/h))
            
            # Per-class
            self.class_stats[c_id]["detections"] += 1
            self.class_stats[c_id]["confidences"].append(confidences[i])
            self.class_stats[c_id]["bbox_centers"].append((cx_abs/w, cy_abs/h))
            
        # Overlaps INTRA-CLASS
        problematic_pairs_indices = []
        unique_classes = set(pred_classes)
        
        for c_id in unique_classes:
            idx_list = [i for i, c in enumerate(pred_classes) if c == c_id]
            if len(idx_list) > 1:
                class_boxes = [pred_boxes[i] for i in idx_list]
                iou_matrix = mu.calculate_iou_matrix(class_boxes, class_boxes)
                pairs = np.argwhere(np.triu(iou_matrix, k=1) > self.overlap_threshold)
                for p in pairs:
                    problematic_pairs_indices.append((idx_list[p[0]], idx_list[p[1]]))
        
        if len(problematic_pairs_indices) > 0:
            if len(self.stats["overlap_events"]) < self.MAX_OVERLAPS_HTML:
                overlap_img_name = f"OVERLAP_{filename}"
                self.stats["overlap_events"].append({
                    "orig_filename": filename,
                    "evidence_filename": overlap_img_name,
                    "count": len(problematic_pairs_indices)
                })
            
        return problematic_pairs_indices

    def generate_plots(self):
        print("📊 Generating inference plots (Multi-Class)...")
        
        # Global Histogram
        if self.stats["confidences"]:
            plot_generator.plot_confidence_histogram(
                self.stats["confidences"], [], [], [], 
                threshold=0.0, 
                output_path=str(self.plots_dir / f"inference_{config.CONFIDENCE_DIST_FILENAME}"),
                title="Real World Confidence Distribution (Global)"
            )

        # Global Heatmap
        plot_generator.plot_normalized_heatmap(
            self.stats["bbox_centers_norm"],
            str(self.plots_dir / f"inference_{config.HEATMAP_FILENAME}"),
            title="Normalized Detection Heatmap (Global)",
            cmap='magma'
        )
        
        # Per-Class Plots
        for c_id, s in self.class_stats.items():
            c_name = self.class_names.get(c_id, f"Class_{c_id}")
            safe_name = c_name.replace(" ", "_")
            
            if s["confidences"]:
                plot_generator.plot_confidence_histogram(
                    s["confidences"], [], [], [], 
                    threshold=0.0, 
                    output_path=str(self.plots_dir / f"inference_conf_dist_{safe_name}.png"),
                    title=f"Confidence ({c_name})"
                )
            if s["bbox_centers"]:
                plot_generator.plot_normalized_heatmap(
                    s["bbox_centers"],
                    str(self.plots_dir / f"inference_heatmap_{safe_name}.png"),
                    title=f"Heatmap ({c_name})", cmap='magma'
                )

    def generate_html_report(self, experiment_name="yolov8_s_default"):
        print("📝 Compiling inference statistics for the report...")
        
        avg_detections = self.stats["total_detections"] / max(1, self.stats["total_images"])
        
        inference_json_path = Path(self.output_dir) / config.FILE_INFERENCE_META
        meta_manager = InferenceMetadata(inference_json_path)
        meta_manager.set_timestamp(key_name="inference_date")
        
        meta_manager.record_global_stats(
            total_images=self.stats["total_images"],
            total_detections=self.stats["total_detections"],
            avg_detections=avg_detections,
            total_overlaps=sum(event["count"] for event in self.stats["overlap_events"]),
            overlap_events=self.stats["overlap_events"]
        )
        
        # Class Breakdown for HTML
        for c_id, s in self.class_stats.items():
            c_name = self.class_names.get(c_id, f"Class {c_id}")
            safe_name = c_name.replace(" ", "_")
            mean_conf = float(np.mean(s["confidences"])) if s["confidences"] else 0.0
            
            meta_manager.record_class_stats(
                class_name=c_name,
                safe_name=safe_name,
                detections=s["detections"],
                avg_confidence=mean_conf,
                conf_stats=mu.calculate_1d_stats(s["confidences"]),
                spatial_stats=mu.calculate_spatial_stats(s["bbox_centers"])
            )

        meta_manager.record_confidence_stats(mu.calculate_1d_stats(self.stats["confidences"]))
        meta_manager.record_spatial_stats(mu.calculate_spatial_stats(self.stats["bbox_centers_norm"]))
        meta_manager.record_speed_stats(mu.calculate_speed_stats(self.stats["speeds"]))

        # Save JSON
        meta_manager.commit()
        
        exp_dir = Path(config.PROJECT_DIR) / experiment_name
        dataset_meta_path = exp_dir / config.METADATA_FOLDER_NAME / config.FILE_DATASET_META
        train_meta_path = exp_dir / config.METADATA_FOLDER_NAME / config.FILE_TRAIN_META
        
        html_context = {
            "report_title": "Real Inference Report",
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "overlap_thresh": self.overlap_threshold,
            
            # Usamos 'stats' porque inference_template.html itera sobre {{ stats.per_class... }}
            "stats": meta_manager.get_html_summary(), 
            
            # Contextos comunes
            "dataset": DatasetMetadata(dataset_meta_path).get_html_summary() if dataset_meta_path.exists() else None,
            "train": TrainMetadata(train_meta_path).get_html_summary() if train_meta_path.exists() else None
        }
        
        generator = HTMLReportGenerator()
        report_html_path = Path(self.output_dir) / "inference_report.html"
        # Llamada simplificada
        generator.generate_inference_html(str(report_html_path), html_context)