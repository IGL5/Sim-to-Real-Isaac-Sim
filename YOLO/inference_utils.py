import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


class InferenceReportGenerator:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.plots_dir = os.path.join(output_dir, "..", "plots")
        os.makedirs(self.plots_dir, exist_ok=True)
        self.stats = {
            "total_images": 0,
            "total_detections": 0,
            "confidences": [],
            "bbox_centers": [],
            "overlapping_detections": 0
        }

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

    def update(self, pred_boxes, confidences):
        self.stats["total_images"] += 1
        self.stats["total_detections"] += len(pred_boxes)
        self.stats["confidences"].extend(confidences)
        
        # Centers for the Heatmap
        for box in pred_boxes:
            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2
            self.stats["bbox_centers"].append((cx, cy))
            
        # Overlapping / concentric detections (IoU > 0.45 in the same image)
        if len(pred_boxes) > 1:
            iou_matrix = self.calculate_iou_matrix(pred_boxes, pred_boxes)
            upper_tri = np.triu(iou_matrix, k=1)
            overlaps = np.sum(upper_tri > 0.45)
            self.stats["overlapping_detections"] += overlaps

    def generate_plots(self):
        print("📊 Generating inference plots...")
        
        # 1. Confidence Histogram
        if self.stats["confidences"]:
            plt.figure(figsize=(8, 5))
            plt.hist(self.stats["confidences"], bins=20, alpha=0.7, color='blue')
            plt.title("Real World Confidence Distribution")
            plt.xlabel("Confidence")
            plt.ylabel("Frequency")
            plt.savefig(os.path.join(self.plots_dir, "inference_conf_dist.png"))
            plt.close()

        # 2. Heatmap
        if self.stats["bbox_centers"]:
            centers = np.array(self.stats["bbox_centers"])
            plt.figure(figsize=(8, 6))
            plt.hexbin(centers[:, 0], centers[:, 1], gridsize=20, cmap='magma', mincnt=1)
            plt.colorbar(label='Detections')
            plt.title("Real World Detection Heatmap")
            plt.gca().invert_yaxis()
            plt.savefig(os.path.join(self.plots_dir, "inference_heatmap.png"))
            plt.close()

    def generate_html_report(self):
        avg_detections = self.stats["total_detections"] / max(1, self.stats["total_images"])
        
        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: sans-serif; background: #f4f4f9; padding: 20px; }}
                h1 {{ color: #333; }}
                .container {{ display: flex; flex-wrap: wrap; gap: 20px; }}
                .card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); flex: 1; min-width: 250px; }}
                .metric {{ font-size: 2em; font-weight: bold; color: #8e44ad; }}
                img {{ max-width: 100%; border-radius: 8px; margin-top: 10px; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
                td, th {{ padding: 10px; border-bottom: 1px solid #ddd; }}
            </style>
        </head>
        <body>
            <h1>🌍 Real Inference Analytics Report</h1>
            <p>Date: {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>
            
            <div class="container">
                <div class="card">
                    <div>Images Processed</div>
                    <div class="metric">{self.stats['total_images']}</div>
                </div>
                <div class="card">
                    <div>Total Detections</div>
                    <div class="metric">{self.stats['total_detections']}</div>
                </div>
                <div class="card">
                    <div>Crowdness (Avg/Img)</div>
                    <div class="metric">{avg_detections:.2f}</div>
                </div>
                <div class="card">
                    <div>Suspicious Overlaps</div>
                    <div class="metric" style="color:#e74c3c;">{self.stats['overlapping_detections']}</div>
                </div>
            </div>

            <div class="container" style="margin-top:20px;">
                <div class="card"><img src="plots/inference_conf_dist.png"></div>
                <div class="card"><img src="plots/inference_heatmap.png"></div>
            </div>
        </body>
        </html>
        """
        
        path = os.path.join(self.output_dir, "..", "inference_report.html")
        with open(path, "w", encoding='utf-8') as f:
            f.write(html)
        print(f"✅ Real Inference HTML Report generated at: {path}")