import os
import numpy as np
import json
from datetime import datetime
from . import core_visual_utils as cvu
from .html_generator import HTMLReportGenerator
from . import plot_generator


class ReportGenerator:
    def __init__(self, output_dir, iou_threshold=0.5, user_conf_threshold=0.5, prefix="audit"):
        self.output_dir = output_dir
        self.iou_threshold = iou_threshold
        self.user_conf_threshold = user_conf_threshold
        self.prefix = prefix
        self.stats = {
            "TP": 0, "FP": 0, "FN": 0,
            "confidences_TP": [],
            "confidences_FP": [],
            "bbox_centers": [],     # For the heatmap
            "total_gt": 0,          # Total ground truth objects
            "all_predictions": [],  # We save (confidence, best_iou) for PR curve
            "speeds": {"preprocess": [], "inference": [], "postprocess": []}
        }
        self.plots_dir = os.path.join(output_dir, "plots", self.prefix)
        os.makedirs(self.plots_dir, exist_ok=True)

    def update(self, pred_boxes, gt_boxes, confidences, img_shape, speed_dict=None):
        h, w = img_shape
        self.stats["total_gt"] += len(gt_boxes)
        img_stats = {"TP": 0, "FP": 0, "FN": 0, "poor_bbox": 0}

        if speed_dict:
            self.stats["speeds"]["preprocess"].append(speed_dict.get('preprocess', 0))
            self.stats["speeds"]["inference"].append(speed_dict.get('inference', 0))
            self.stats["speeds"]["postprocess"].append(speed_dict.get('postprocess', 0))

        iou_matrix = cvu.calculate_iou_matrix(pred_boxes, gt_boxes)
        
        matched_gt_all = set()    # Para el mAP (Todas las predicciones)
        matched_gt_thresh = set() # Para el Dashboard (Solo >= Umbral)

        for i, pred in enumerate(pred_boxes):
            conf = confidences[i]
            
            if len(gt_boxes) > 0:
                best_iou = np.max(iou_matrix[i])
                best_gt_idx = np.argmax(iou_matrix[i]) 
            else:
                best_iou = 0
                best_gt_idx = -1
                
            # 1. GUARDADO ABSOLUTO (Para curva PR y mAP correcto)
            self.stats["all_predictions"].append({
                "conf": conf,
                "iou": best_iou,
                "is_duplicate": best_gt_idx in matched_gt_all
            })
            if best_iou >= self.iou_threshold:
                matched_gt_all.add(best_gt_idx)
                
            # 2. GUARDADO FILTRADO (Para las métricas de negocio y visualización)
            if conf >= self.user_conf_threshold:
                cx_abs = (pred[0] + pred[2]) / 2
                cy_abs = (pred[1] + pred[3]) / 2
                self.stats["bbox_centers"].append((cx_abs/w, cy_abs/h))
                
                if best_iou >= self.iou_threshold and best_gt_idx not in matched_gt_thresh:
                    self.stats["TP"] += 1
                    self.stats["confidences_TP"].append(conf)
                    matched_gt_thresh.add(best_gt_idx)
                    img_stats["TP"] += 1
                else:
                    self.stats["FP"] += 1
                    self.stats["confidences_FP"].append(conf)
                    if 0.1 <= best_iou < self.iou_threshold:
                        img_stats["poor_bbox"] += 1
                    else:
                        img_stats["FP"] += 1

        img_fn = len(gt_boxes) - len(matched_gt_thresh)
        self.stats["FN"] += img_fn
        img_stats["FN"] = img_fn
        
        return img_stats

    def calculate_ap(self, iou_thresh):
        preds = sorted(self.stats["all_predictions"], key=lambda x: x["conf"], reverse=True)
        
        tps = np.zeros(len(preds))
        fps = np.zeros(len(preds))
        confs = np.zeros(len(preds))
        
        for i, p in enumerate(preds):
            confs[i] = p["conf"]
            if p["iou"] >= iou_thresh and not p["is_duplicate"]:
                tps[i] = 1
            else:
                fps[i] = 1
                
        tp_cumsum = np.cumsum(tps)
        fp_cumsum = np.cumsum(fps)
        
        total_gt = self.stats["total_gt"] + 1e-6 
        recalls = tp_cumsum / total_gt
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
        
        # Guardar F1 crudo antes de suavizar la curva
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-6)
        
        p_smooth = np.concatenate(([1.0], precisions, [0.0]))
        r_smooth = np.concatenate(([0.0], recalls, [1.0]))
        for i in range(len(p_smooth) - 1, 0, -1):
            p_smooth[i - 1] = np.maximum(p_smooth[i - 1], p_smooth[i])
            
        indices = np.where(r_smooth[1:] != r_smooth[:-1])[0]
        ap = np.sum((r_smooth[indices + 1] - r_smooth[indices]) * p_smooth[indices + 1])
        
        return ap, p_smooth, r_smooth, confs, f1_scores

    def generate_plots(self):
        print("📊 Generating statistical plots...")

        # 1. Confusion Matrix
        plot_generator.plot_confusion_matrix(
            self.stats["TP"], self.stats["FP"], self.stats["FN"],
            os.path.join(self.plots_dir, "confusion_matrix.png")
        )

        # 2. Confidence Histogram
        plot_generator.plot_confidence_histogram(
            confs_primary=self.stats["confidences_TP"], label_primary='Hits (TP)', color_primary='green',
            confs_secondary=self.stats["confidences_FP"], label_secondary='Errors (FP)', color_secondary='red',
            output_path=os.path.join(self.plots_dir, "confidence_dist.png"),
            title="Confidence Distribution"
        )

        # 3. Heatmap
        plot_generator.plot_normalized_heatmap(
            self.stats["bbox_centers"],
            os.path.join(self.plots_dir, "heatmap.png"),
            title="Normalized Detection Heatmap",
            cmap='inferno'
        )

        # 4. Precision-Recall Curve y F1 Curve
        ap50, precisions, recalls, confs, f1_scores = self.calculate_ap(0.5)
        plot_generator.plot_pr_curve(
            precisions, recalls, ap50,
            os.path.join(self.plots_dir, "pr_curve.png")
        )
        
        if len(f1_scores) > 0:
            best_idx = np.argmax(f1_scores)
            plot_generator.plot_f1_curve(
                confs, f1_scores, confs[best_idx], f1_scores[best_idx],
                os.path.join(self.plots_dir, "f1_curve.png")
            )

    def generate_html_report(self, experiment_name="yolov8_s_default"):
        """ Calculates final metrics and delegates HTML creation to Jinja2 """
        print("📝 Compiling numerical metrics for the report...")
        
        # 1. Mathematical calculations
        total_pred = self.stats["TP"] + self.stats["FP"]
        total_real = self.stats["TP"] + self.stats["FN"]
        
        precision = self.stats["TP"] / total_pred if total_pred > 0 else 0
        recall = self.stats["TP"] / total_real if total_real > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        map50, *_ = self.calculate_ap(0.5)
        
        ap_sum = 0
        thresholds = np.arange(0.5, 1.0, 0.05)
        for t in thresholds:
            ap_t, *_ = self.calculate_ap(t)
            ap_sum += ap_t
        map_50_95 = ap_sum / len(thresholds)

        # 2. Pack metrics for the template
        metrics_dict = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "map_50": map50,
            "map_50_95": map_50_95,
            "total_real": total_real,
            "total_pred": total_pred,
            "tp": self.stats["TP"],
            "fn": self.stats["FN"],
            "fp": self.stats["FP"]
        }

        metrics_dict["prefix"] = self.prefix

        metrics_dict["confidence_stats"] = {
            "True_Positives": cvu.calculate_1d_stats(self.stats["confidences_TP"]),
            "False_Positives": cvu.calculate_1d_stats(self.stats["confidences_FP"])
        }
        metrics_dict["spatial_stats"] = cvu.calculate_spatial_stats(self.stats["bbox_centers"])

        avg_pre = np.mean(self.stats["speeds"]["preprocess"]) if self.stats["speeds"]["preprocess"] else 0
        avg_inf = np.mean(self.stats["speeds"]["inference"]) if self.stats["speeds"]["inference"] else 0
        avg_post = np.mean(self.stats["speeds"]["postprocess"]) if self.stats["speeds"]["postprocess"] else 0
        total_ms = avg_pre + avg_inf + avg_post
        fps = 1000 / total_ms if total_ms > 0 else 0

        metrics_dict["speed_stats"] = {
            "preprocess_ms": round(avg_pre, 2),
            "inference_ms": round(avg_inf, 2),
            "postprocess_ms": round(avg_post, 2),
            "total_ms": round(total_ms, 2),
            "fps": round(fps, 2)
        }

        # 3. Save audit metadata to JSON
        audit_metadata = {
            "audit_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "metrics": metrics_dict,
            "evaluation_params": {
                "iou_threshold": self.iou_threshold
            }
        }
        
        audit_json_path = os.path.join(self.output_dir, f"{self.prefix}_metadata.json")
        try:
            with open(audit_json_path, "w", encoding='utf-8') as f:
                json.dump(audit_metadata, f, indent=4)
            print(f"💾 Audit metrics saved at: {audit_json_path}")
        except Exception as e:
            print(f"⚠️ Could not save audit JSON: {e}")

        # 4. Instantiate the generator and create the HTML
        templates_dir = os.path.join(os.getcwd(), "modules", "templates")
        project_dir = cvu.PROJECT_DIR
        dataset_out_dir = os.path.join(os.getcwd(), "dataset_yolo_output")
        
        generator = HTMLReportGenerator(templates_dir, project_dir, dataset_out_dir)
        
        output_path = os.path.join(self.output_dir, f"{self.prefix}_report.html")
        generator.generate_audit_html(output_path, experiment_name, metrics_dict)