import os
import numpy as np
import json
from datetime import datetime
from collections import defaultdict
from . import core_visual_utils as cvu
from .html_generator import HTMLReportGenerator
from . import plot_generator

class InferenceReportGenerator:
    def __init__(self, output_dir, overlap_threshold=0.5, class_names=None):
        self.output_dir = output_dir
        self.plots_dir = os.path.join(output_dir, "..", "plots", "inference")
        self.overlaps_dir = os.path.join(self.plots_dir, "suspicious_overlaps_imgs")
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.overlaps_dir, exist_ok=True)
        
        self.overlap_threshold = overlap_threshold
        self.class_names = class_names if class_names else {}
        self.MAX_OVERLAPS_HTML = 50  # Límite para no colapsar el navegador
        
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
            
        # Overlaps INTRA-CLASE (Solo nos preocupa si dos bicis se solapan)
        problematic_pairs_indices = []
        unique_classes = set(pred_classes)
        
        for c_id in unique_classes:
            idx_list = [i for i, c in enumerate(pred_classes) if c == c_id]
            if len(idx_list) > 1:
                class_boxes = [pred_boxes[i] for i in idx_list]
                iou_matrix = cvu.calculate_iou_matrix(class_boxes, class_boxes)
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
        
        # Histograma Global (Aprovechamos la nueva función)
        if self.stats["confidences"]:
            plot_generator.plot_confidence_histogram(
                self.stats["confidences"], [], [], [], 
                threshold=0.0, 
                output_path=os.path.join(self.plots_dir, "inference_conf_dist.png"),
                title="Real World Confidence Distribution (Global)"
            )

        # Heatmap Global
        plot_generator.plot_normalized_heatmap(
            self.stats["bbox_centers_norm"],
            os.path.join(self.plots_dir, "inference_heatmap.png"),
            title="Normalized Detection Heatmap (Global)",
            cmap='magma'
        )
        
        # Gráficos por clase
        for c_id, s in self.class_stats.items():
            c_name = self.class_names.get(c_id, f"Class_{c_id}")
            safe_name = c_name.replace(" ", "_")
            
            if s["confidences"]:
                plot_generator.plot_confidence_histogram(
                    s["confidences"], [], [], [], 
                    threshold=0.0, 
                    output_path=os.path.join(self.plots_dir, f"inference_conf_dist_{safe_name}.png"),
                    title=f"Confidence ({c_name})"
                )
            if s["bbox_centers"]:
                plot_generator.plot_normalized_heatmap(
                    s["bbox_centers"],
                    os.path.join(self.plots_dir, f"inference_heatmap_{safe_name}.png"),
                    title=f"Heatmap ({c_name})", cmap='magma'
                )

    def generate_html_report(self, experiment_name="yolov8_s_default"):
        print("📝 Compiling inference statistics for the report...")
        
        avg_detections = self.stats["total_detections"] / max(1, self.stats["total_images"])
        
        # Desglose de Clases para HTML
        class_summary = {}
        for c_id, s in self.class_stats.items():
            c_name = self.class_names.get(c_id, f"Class {c_id}")
            safe_name = c_name.replace(" ", "_")
            mean_conf = np.mean(s["confidences"]) if s["confidences"] else 0
            
            class_summary[c_name] = {
                "detections": s["detections"],
                "avg_confidence": mean_conf,
                "safe_name": safe_name,
                "confidence_stats": cvu.calculate_1d_stats(s["confidences"]),
                "spatial_stats": cvu.calculate_spatial_stats(s["bbox_centers"])
            }

        stats_dict = {
            "total_images": self.stats["total_images"],
            "total_detections": self.stats["total_detections"],
            "avg_detections": avg_detections,
            "total_overlaps": sum(event["count"] for event in self.stats["overlap_events"]),
            "overlap_events": self.stats["overlap_events"],
            "per_class": class_summary,
            "confidence_stats": cvu.calculate_1d_stats(self.stats["confidences"]),
            "spatial_stats": cvu.calculate_spatial_stats(self.stats["bbox_centers_norm"])
        }

        avg_pre = np.mean(self.stats["speeds"]["preprocess"]) if self.stats["speeds"]["preprocess"] else 0
        avg_inf = np.mean(self.stats["speeds"]["inference"]) if self.stats["speeds"]["inference"] else 0
        avg_post = np.mean(self.stats["speeds"]["postprocess"]) if self.stats["speeds"]["postprocess"] else 0
        total_ms = avg_pre + avg_inf + avg_post

        stats_dict["speed_stats"] = {
            "preprocess_ms": round(avg_pre, 2), "inference_ms": round(avg_inf, 2),
            "postprocess_ms": round(avg_post, 2), "total_ms": round(total_ms, 2),
            "fps": round(1000 / total_ms, 2) if total_ms > 0 else 0
        }

        # Guardar JSON y llamar a Jinja
        inference_json_path = os.path.join(self.output_dir, "..", "inference_metadata.json")
        try:
            with open(inference_json_path, "w", encoding='utf-8') as f:
                json.dump({"stats": stats_dict}, f, indent=4)
        except: pass

        templates_dir = os.path.join(os.getcwd(), "modules", "templates")
        generator = HTMLReportGenerator(templates_dir, cvu.PROJECT_DIR, os.path.join(os.getcwd(), "dataset_yolo_output"))
        generator.generate_inference_html(os.path.join(self.output_dir, "..", "inference_report.html"), experiment_name, stats_dict, self.overlap_threshold)