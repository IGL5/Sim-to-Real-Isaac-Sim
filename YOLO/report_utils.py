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
            "bbox_centers": [] # For the heatmap
        }
        self.plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(self.plots_dir, exist_ok=True)

    def calculate_iou(self, boxA, boxB):
        """ Calculate Intersection over Union between two boxes [x1, y1, x2, y2] """
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
        return iou

    def update(self, pred_boxes, gt_boxes, confidences):
        """
        Receive the boxes of ONE image, calculate matchings and update statistics.
        """
        matched_gt = set()
        
        # Save centers for Heatmap
        for box in pred_boxes:
            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2
            self.stats["bbox_centers"].append((cx, cy))

        # Compare predictions with reality
        for i, pred in enumerate(pred_boxes):
            best_iou = 0
            best_gt_idx = -1
            
            for j, gt in enumerate(gt_boxes):
                iou = self.calculate_iou(pred, gt)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
            
            # Hit Criterion (TP)
            if best_iou >= self.iou_threshold and best_gt_idx not in matched_gt:
                self.stats["TP"] += 1
                self.stats["confidences_TP"].append(confidences[i])
                matched_gt.add(best_gt_idx)
            else:
                self.stats["FP"] += 1 # False Positive
                self.stats["confidences_FP"].append(confidences[i])

        # What was not matched is False Negative
        self.stats["FN"] += (len(gt_boxes) - len(matched_gt))

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

    def generate_html_report(self):
        total_pred = self.stats["TP"] + self.stats["FP"]
        total_real = self.stats["TP"] + self.stats["FN"]
        
        precision = self.stats["TP"] / total_pred if total_pred > 0 else 0
        recall = self.stats["TP"] / total_real if total_real > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: sans-serif; background: #f4f4f9; padding: 20px; }}
                h1 {{ color: #333; }}
                .container {{ display: flex; flex-wrap: wrap; gap: 20px; }}
                .card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); flex: 1; min-width: 300px; }}
                .metric {{ font-size: 2em; font-weight: bold; color: #2c3e50; }}
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
            </div>

            <div class="container" style="margin-top:20px;">
                <div class="card"><img src="plots/confusion_matrix.png"></div>
                <div class="card"><img src="plots/confidence_dist.png"></div>
                <div class="card"><img src="plots/heatmap.png"></div>
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