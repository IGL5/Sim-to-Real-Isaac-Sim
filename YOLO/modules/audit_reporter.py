import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from .core_visual_utils import calculate_iou_matrix
from .html_generator import HTMLReportGenerator


class ReportGenerator:
    def __init__(self, output_dir, iou_threshold=0.5):
        self.output_dir = output_dir
        self.iou_threshold = iou_threshold
        self.stats = {
            "TP": 0, "FP": 0, "FN": 0,
            "confidences_TP": [],
            "confidences_FP": [],
            "bbox_centers": [],     # For the heatmap
            "total_gt": 0,          # Total ground truth objects
            "all_predictions": []   # We save (confidence, best_iou) for PR curve
        }
        self.plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(self.plots_dir, exist_ok=True)

    def update(self, pred_boxes, gt_boxes, confidences):
        """
        Receive the boxes of ONE image, calculate matchings and update statistics.
        """
        matched_gt = set()
        self.stats["total_gt"] += len(gt_boxes)
        img_stats = {"TP": 0, "FP": 0, "FN": 0, "poor_bbox": 0}
        
        # Save centers for Heatmap
        for box in pred_boxes:
            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2
            self.stats["bbox_centers"].append((cx, cy))

        iou_matrix = calculate_iou_matrix(pred_boxes, gt_boxes)

        # Comparamos predicciones con la realidad
        for i, pred in enumerate(pred_boxes):
            if len(gt_boxes) > 0:
                best_iou = np.max(iou_matrix[i])
                best_gt_idx = np.argmax(iou_matrix[i]) 
            else:
                best_iou = 0
                best_gt_idx = -1
                
            self.stats["all_predictions"].append({
                "conf": confidences[i],
                "iou": best_iou,
                "is_duplicate": best_gt_idx in matched_gt
            })
            
            # Hit Criterion (TP)
            if best_iou >= self.iou_threshold and best_gt_idx not in matched_gt:
                self.stats["TP"] += 1
                self.stats["confidences_TP"].append(confidences[i])
                matched_gt.add(best_gt_idx)
                img_stats["TP"] += 1
            else:
                self.stats["FP"] += 1 # False Positive global
                self.stats["confidences_FP"].append(confidences[i])
                
                # Classify if it is a pure FP (invention/duplicate) or a poor box
                if 0.1 <= best_iou < self.iou_threshold:
                    img_stats["poor_bbox"] += 1
                else:
                    img_stats["FP"] += 1

        # What was not matched is False Negative
        img_fn = len(gt_boxes) - len(matched_gt)
        self.stats["FN"] += img_fn
        img_stats["FN"] = img_fn
        
        return img_stats

    def calculate_ap(self, iou_thresh):
        """ Calculate the Average Precision for a specific IoU threshold """
        # Sort all predictions by confidence from highest to lowest
        preds = sorted(self.stats["all_predictions"], key=lambda x: x["conf"], reverse=True)
        
        tps = np.zeros(len(preds))
        fps = np.zeros(len(preds))
        
        for i, p in enumerate(preds):
            # It is a hit if it exceeds the threshold and is not a duplicate detection of the same object
            if p["iou"] >= iou_thresh and not p["is_duplicate"]:
                tps[i] = 1
            else:
                fps[i] = 1
                
        # Cumulative sums
        tp_cumsum = np.cumsum(tps)
        fp_cumsum = np.cumsum(fps)
        
        # Precision and Recall at each point
        total_gt = self.stats["total_gt"] + 1e-6 
        recalls = tp_cumsum / total_gt
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
        
        # Add extremes for the PR curve and smooth it (standard interpolation)
        precisions = np.concatenate(([1.0], precisions, [0.0]))
        recalls = np.concatenate(([0.0], recalls, [1.0]))
        for i in range(len(precisions) - 1, 0, -1):
            precisions[i - 1] = np.maximum(precisions[i - 1], precisions[i])
            
        # Calculate area under the curve
        indices = np.where(recalls[1:] != recalls[:-1])[0]
        ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
        
        return ap, precisions, recalls

    def generate_html_report(self, experiment_name="yolov8_s_default"):
        """ Calculates final metrics and delegates HTML creation to Jinja2 """
        print("📝 Compiling numerical metrics for the report...")
        
        # 1. Mathematical calculations
        total_pred = self.stats["TP"] + self.stats["FP"]
        total_real = self.stats["TP"] + self.stats["FN"]
        
        precision = self.stats["TP"] / total_pred if total_pred > 0 else 0
        recall = self.stats["TP"] / total_real if total_real > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        ap50, _, _ = self.calculate_ap(0.5)
        
        ap_sum = 0
        thresholds = np.arange(0.5, 1.0, 0.05)
        for t in thresholds:
            ap_t, _, _ = self.calculate_ap(t)
            ap_sum += ap_t
        map_50_95 = ap_sum / len(thresholds)

        # 2. Pack metrics for the template
        metrics_dict = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "ap50": ap50,
            "map_50_95": map_50_95,
            "total_real": total_real,
            "total_pred": total_pred,
            "tp": self.stats["TP"],
            "fn": self.stats["FN"],
            "fp": self.stats["FP"]
        }

        # 3. Pack plots paths (relative paths for the HTML)
        plots_dict = {
            "confusion_matrix": "plots/confusion_matrix.png",
            "pr_curve": "plots/pr_curve.png",
            "confidence_dist": "plots/confidence_dist.png",
            "heatmap": "plots/heatmap.png"
        }

        # 4. Instantiate the generator and create the HTML
        templates_dir = os.path.join(os.getcwd(), "modules", "templates")
        project_dir = os.path.join(os.getcwd(), "cyclist_detector")
        dataset_out_dir = os.path.join(os.getcwd(), "dataset_yolo_output")
        
        generator = HTMLReportGenerator(templates_dir, project_dir, dataset_out_dir)
        
        output_path = os.path.join(self.output_dir, "report.html")
        generator.generate_audit_html(output_path, experiment_name, metrics_dict, plots_dict)