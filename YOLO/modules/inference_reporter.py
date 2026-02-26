import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from .core_visual_utils import calculate_iou_matrix


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
        iou_matrix = calculate_iou_matrix(pred_boxes, pred_boxes)
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
            plt.figure(figsize=(8, 5))
            plt.hist(self.stats["confidences"], bins=20, alpha=0.7, color='blue')
            plt.title("Real World Confidence Distribution")
            plt.xlabel("Confidence")
            plt.ylabel("Frequency")
            plt.savefig(os.path.join(self.plots_dir, "inference_conf_dist.png"))
            plt.close()

        # 2. Heatmap
        if self.stats["bbox_centers_norm"]:
            centers = np.array(self.stats["bbox_centers_norm"])
            plt.figure(figsize=(8, 6))
            plt.hexbin(centers[:, 0], centers[:, 1], gridsize=20, cmap='magma', mincnt=1, extent=[0, 1, 0, 1])
            plt.colorbar(label='Detections')
            plt.title("Normalized Detection Heatmap (All Resolutions)")
            plt.gca().invert_yaxis()
            plt.xlabel("Normalized Width (0.0 - 1.0)")
            plt.ylabel("Normalized Height (0.0 - 1.0)")
            plt.savefig(os.path.join(self.plots_dir, "inference_heatmap.png"))
            plt.close()

    def generate_html_report(self):
        avg_detections = self.stats["total_detections"] / max(1, self.stats["total_images"])
        total_overlaps = sum(event["count"] for event in self.stats["overlap_events"])

        overlap_gallery_html = ""
        if not self.stats["overlap_events"]:
             overlap_gallery_html = "<p>✅ No suspicious overlaps detected above threshold.</p>"
        else:
            for event in self.stats["overlap_events"]:
                 img_rel_path = os.path.join("plots", "suspicious_overlaps_imgs", event["evidence_filename"])
                 overlap_gallery_html += f"""
                 <div class="card overlap-card">
                    <p><strong>Source:</strong> {event['orig_filename']}</p>
                    <p>Found <strong>{event['count']}</strong> risk pair(s)</p>
                    <img src="{img_rel_path}" alt="Overlap Evidence">
                 </div>
                 """
        
        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: sans-serif; background: #f4f4f9; padding: 20px; color: #333; }}
                h1 {{ color: #2c3e50; }}
                h2 {{ color: #e74c3c; margin-top: 40px; border-bottom: 2px solid #e74c3c; padding-bottom: 10px; }}
                .container {{ display: flex; flex-wrap: wrap; gap: 20px; }}
                .card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); flex: 1; min-width: 200px; }}
                .metric {{ font-size: 2em; font-weight: bold; color: #8e44ad; }}
                .alert-metric {{ color: #e74c3c; }}
                .plot-grid, .gallery-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; margin-top: 20px; }}
                .gallery-grid .overlap-card {{ min-width: 300px; border-left: 5px solid #e74c3c; }}

                img {{ max-width: 100%; border-radius: 8px; margin-top: 10px; }}
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
                <div class="card" style="background-color: #fdedec;">
                    <div>Total Suspicious Pairs (IoU > {self.overlap_threshold})</div>
                    <div class="metric alert-metric">{total_overlaps}</div>
                </div>
            </div>

            <div class="plot-grid">
                <div class="card">
                    <h3>Confidence Distribution</h3>
                    <img src="plots/inference_conf_dist.png">
                </div>
                <div class="card">
                    <h3>Spatial Distribution (Heatmap)</h3>
                    <p style="font-size:0.9em; color:#666;">Normalized to 1x1 frame regardless of resolution.</p>
                    <img src="plots/inference_heatmap.png">
                </div>
            </div>

            <h2>⚠️ Suspicious Overlap Analysis</h2>
            <div class="gallery-grid">
                {overlap_gallery_html}
            </div>
        </body>
        </html>
        """
        
        path = os.path.join(self.output_dir, "..", "inference_report.html")
        with open(path, "w", encoding='utf-8') as f:
            f.write(html)
        print(f"✅ Real Inference HTML Report generated at: {path}")