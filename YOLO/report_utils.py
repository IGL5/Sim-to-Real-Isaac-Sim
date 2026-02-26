import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

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

    def calculate_iou_matrix(self, boxesA, boxesB):
        """ 
        Calculates the Intersection over Union (IoU) matrix between two sets of boxes.
        boxesA: List or array of N boxes [x1, y1, x2, y2]
        boxesB: List or array of M boxes [x1, y1, x2, y2]
        Returns: Numpy matrix of shape (N, M) with the IoUs.
        """
        if len(boxesA) == 0 or len(boxesB) == 0:
            return np.zeros((len(boxesA), len(boxesB)))

        bA = np.array(boxesA)
        bB = np.array(boxesB)

        A = bA[:, np.newaxis, :]
        B = bB[np.newaxis, :, :]
        xA = np.maximum(A[..., 0], B[..., 0])
        yA = np.maximum(A[..., 1], B[..., 1])
        xB = np.minimum(A[..., 2], B[..., 2])
        yB = np.minimum(A[..., 3], B[..., 3])

        # Área de intersección
        interArea = np.maximum(0, xB - xA) * np.maximum(0, yB - yA)

        # Áreas individuales
        boxAArea = (A[..., 2] - A[..., 0]) * (A[..., 3] - A[..., 1])
        boxBArea = (B[..., 2] - B[..., 0]) * (B[..., 3] - B[..., 1])

        iou = interArea / (boxAArea + boxBArea - interArea + 1e-6)

        return iou

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

        iou_matrix = self.calculate_iou_matrix(pred_boxes, gt_boxes)

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

    def generate_plots(self):
        print("📊 Generating statistical plots...")
        
        # 1. Confusion Matrix
        plt.figure(figsize=(6, 5))
        matrix = [[self.stats["TP"], self.stats["FP"]], [self.stats["FN"], 0]]
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, 
                    xticklabels=["Pred Pos", "Pred Neg"], yticklabels=["Real Pos", "Real Neg"])
        plt.title("Confusion Matrix")
        plt.savefig(os.path.join(self.plots_dir, "confusion_matrix.png"))
        plt.close()

        # 2. Confidence Histogram
        plt.figure(figsize=(8, 5))
        plt.hist(self.stats["confidences_TP"], bins=20, alpha=0.7, label='Hits (TP)', color='green')
        plt.hist(self.stats["confidences_FP"], bins=20, alpha=0.7, label='Errors (FP)', color='red')
        plt.title("Confidence Distribution")
        plt.legend()
        plt.savefig(os.path.join(self.plots_dir, "confidence_dist.png"))
        plt.close()

        # 3. Heatmap
        if self.stats["bbox_centers"]:
            centers = np.array(self.stats["bbox_centers"])
            plt.figure(figsize=(8, 6))
            plt.hexbin(centers[:, 0], centers[:, 1], gridsize=20, cmap='inferno', mincnt=1)
            plt.colorbar(label='Detections')
            plt.title("Detection Heatmap")
            plt.gca().invert_yaxis() 
            plt.savefig(os.path.join(self.plots_dir, "heatmap.png"))
            plt.close()

        # 4. Precision-Recall Curve
        ap50, precisions, recalls = self.calculate_ap(0.5)
        plt.figure(figsize=(8, 5))
        plt.plot(recalls, precisions, color='blue', lw=2, label=f'PR Curve (mAP@50 = {ap50:.4f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve (IoU=0.50)')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.legend(loc="lower left")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(self.plots_dir, "pr_curve.png"))
        plt.close()

    def generate_html_report(self):
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

        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: sans-serif; background: #f4f4f9; padding: 20px; }}
                h1 {{ color: #333; }}
                .container {{ display: flex; flex-wrap: wrap; gap: 20px; }}
                .card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); flex: 1; min-width: 200px; }}
                .metric {{ font-size: 2em; font-weight: bold; color: #2c3e50; }}
                .map-metric {{ color: #1f618d; }} /* NUEVO: Azul oscuro profesional */
                
                /* NUEVO: Cuadrícula de 2 columnas para que las gráficas se vean gigantes */
                .plot-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 20px; }}
                .plot-grid .card {{ min-width: 0; display: flex; justify-content: center; align-items: center; }}
                
                img {{ max-width: 100%; border-radius: 8px; margin-top: 10px; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
                td, th {{ padding: 10px; border-bottom: 1px solid #ddd; }}
            </style>
        </head>
        <body>
            <h1>📊 YOLO Audit Report</h1>
            <p>Date: {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>
            
            <div class="container">
                <div class="card">
                    <div>Precision</div>
                    <div class="metric">{precision:.2%}</div>
                </div>
                <div class="card">
                    <div>Recall</div>
                    <div class="metric">{recall:.2%}</div>
                </div>
                <div class="card">
                    <div>F1-Score</div>
                    <div class="metric">{f1:.2f}</div>
                </div>
                <div class="card">
                    <div>mAP@50</div>
                    <div class="metric map-metric">{ap50:.3f}</div>
                </div>
                <div class="card">
                    <div>mAP@50-95</div>
                    <div class="metric map-metric">{map_50_95:.3f}</div>
                </div>
            </div>

            <div class="plot-grid">
                <div class="card"><img src="plots/confusion_matrix.png"></div>
                <div class="card"><img src="plots/confidence_dist.png"></div>
                <div class="card"><img src="plots/heatmap.png"></div>
                <div class="card"><img src="plots/pr_curve.png"></div>
            </div>
            
            <div class="card" style="margin-top:20px;">
                <h3>Numerical Detail</h3>
                <table>
                    <tr><td>Total Real (GT)</td><td>{total_real}</td></tr>
                    <tr><td>Total Predicted</td><td>{total_pred}</td></tr>
                    <tr><td>TP (Hits)</td><td>{self.stats['TP']}</td></tr>
                    <tr><td>FN (Missed)</td><td>{self.stats['FN']}</td></tr>
                    <tr><td>FP (Invented)</td><td>{self.stats['FP']}</td></tr>
                </table>
            </div>
        </body>
        </html>
        """
        
        path = os.path.join(self.output_dir, "report.html")
        with open(path, "w", encoding='utf-8') as f:
            f.write(html)
        print(f"✅ HTML Report generated at: {path}")