import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import numpy as np

try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

from src.core import config
from src.core.utils import math_utils as mu
from src.core.metadata.tiled_builder import TiledInferenceMetadata
from src.core.metadata.dataset_builder import DatasetMetadata
from src.core.metadata.train_builder import TrainMetadata
from src.evaluation.utils.html_generator import HTMLReportGenerator
from src.evaluation.utils import plot_generator

class TiledReporter:
    """
    Orchestrates metrics gathering, terminal formatting, plot generation,
    and metadata persistence for Tiled / Sliced Inference on high-res images.
    """
    def __init__(self, output_dir=None, conf_threshold=0.4, iou_threshold=0.5,
                 tile_size=640, overlap_ratio=0.2, fusion_method='wbf', class_names=None):
        self.output_dir = Path(output_dir) if output_dir else config.TILED_OUTPUT_DIR
        self.images_dir = self.output_dir / "images"
        self.plots_dir = self.output_dir / "plots"
        self.grid_dir = self.output_dir / "grid_debug"
        self.crops_dir = self.output_dir / "crops"

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.tile_size = tile_size
        self.overlap_ratio = overlap_ratio
        self.fusion_method = fusion_method.lower()
        self.class_names = class_names if class_names else {}

        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        self.stats = {
            "total_images": 0,
            "total_detections": 0,
            "total_tiles": 0,
            "confidences": [],
            "bbox_centers_norm": [],
            "bbox_areas_pct": [],
            "bbox_areas_px": [],
            "images": [],
            "speeds": {"slicing": [], "inference": [], "fusion": [], "total": []}
        }

        self.class_stats = defaultdict(lambda: {
            "detections": 0,
            "confidences": [],
            "bbox_centers": [],
            "areas_px": [],
            "areas_pct": []
        })

    def record_image(self, filename, img_w, img_h, num_tiles,
                     boxes, scores, classes, timings, evidence_filename=None):
        """
        Records the results of a single processed high-resolution image.
        boxes: (N, 4) in absolute pixels
        scores: (N,)
        classes: (N,)
        timings: dict with 'slicing', 'inference', 'fusion', 'total' in ms
        """
        self.stats["total_images"] += 1
        self.stats["total_tiles"] += num_tiles
        self.stats["total_detections"] += len(boxes)

        for k in ["slicing", "inference", "fusion", "total"]:
            if k in timings:
                self.stats["speeds"][k].append(timings[k])

        centers_norm = []
        image_area = float(img_w * img_h)

        for i, box in enumerate(boxes):
            c_id = int(classes[i])
            conf = float(scores[i])
            x1, y1, x2, y2 = box

            bw = max(0.0, x2 - x1)
            bh = max(0.0, y2 - y1)
            area_px = bw * bh
            area_pct = (area_px / image_area) * 100.0 if image_area > 0 else 0.0

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            cx_n = cx / img_w if img_w > 0 else 0.0
            cy_n = cy / img_h if img_h > 0 else 0.0

            centers_norm.append((cx_n, cy_n))
            self.stats["confidences"].append(conf)
            self.stats["bbox_centers_norm"].append((cx_n, cy_n))
            self.stats["bbox_areas_px"].append(area_px)
            self.stats["bbox_areas_pct"].append(area_pct)

            # Per-class accumulators
            self.class_stats[c_id]["detections"] += 1
            self.class_stats[c_id]["confidences"].append(conf)
            self.class_stats[c_id]["bbox_centers"].append((cx_n, cy_n))
            self.class_stats[c_id]["areas_px"].append(area_px)
            self.class_stats[c_id]["areas_pct"].append(area_pct)

        # Pairwise dispersion for this specific image
        dispersion_info = mu.calculate_pairwise_dispersion(centers_norm)
        mean_conf = float(np.mean(scores)) if len(scores) > 0 else 0.0

        image_info = {
            "filename": filename,
            "evidence_filename": evidence_filename or f"PRED_{filename}",
            "width": img_w,
            "height": img_h,
            "num_tiles": num_tiles,
            "detections": len(boxes),
            "mean_conf": round(mean_conf, 3),
            "dispersion": dispersion_info,
            "time_ms": round(timings.get("total", 0.0), 1)
        }
        self.stats["images"].append(image_info)
        return image_info

    def print_image_summary(self, filename, img_w, img_h, num_tiles,
                            boxes, scores, classes, timings):
        """Prints a clean, structured summary for a single image in terminal."""
        num_det = len(boxes)
        tot_time = timings.get('total', 0.0)
        fps = (1000.0 / tot_time) if tot_time > 0 else 0.0

        print(f"\n+-----------------------------------------------------------------------------+")
        print(f"| [IMAGE] {filename:<40} ({img_w}x{img_h}) |")
        print(f"| Tiles: {num_tiles:<4} ({self.tile_size}x{self.tile_size}, overlap {int(self.overlap_ratio*100)}%)  | Time: {tot_time:6.1f} ms ({fps:4.1f} FPS/img)   |")
        print(f"+-----------------------------------------------------------------------------+")

        if num_det == 0:
            print(f"| No objects detected with confidence >= {self.conf_threshold:<5}                            |")
        else:
            print(f"| {'CLASS':<22} {'DETECTIONS':<12} {'MEAN CONF':<15} {'AVG AREA':<12}       |")
            print(f"+-----------------------------------------------------------------------------+")
            
            # Group by class for this image
            u_classes = np.unique(classes)
            for c_id in u_classes:
                idx = np.where(classes == c_id)[0]
                c_name = self.class_names.get(int(c_id), f"Class_{c_id}")
                c_scores = scores[idx]
                c_mean_conf = np.mean(c_scores)
                c_std_conf = np.std(c_scores)
                
                # Area
                c_boxes = boxes[idx]
                c_areas_pct = [((b[2]-b[0]) * (b[3]-b[1]) / (img_w * img_h)) * 100.0 for b in c_boxes]
                avg_area = np.mean(c_areas_pct)

                conf_str = f"{c_mean_conf:.2f} (+/-{c_std_conf:.2f})"
                area_str = f"{avg_area:.3f}%"
                print(f"| {c_name:<22} {len(idx):<12} {conf_str:<15} {area_str:<12}       |")

            centers_norm = [((b[0]+b[2])/(2.0*img_w), (b[1]+b[3])/(2.0*img_h)) for b in boxes]
            disp = mu.calculate_pairwise_dispersion(centers_norm)
            print(f"+-----------------------------------------------------------------------------+")
            print(f"| Dispersion: {disp['category']:<35} (d_pairwise: {disp['mean_pairwise_dist']:.3f})   |")

        print(f"+-----------------------------------------------------------------------------+")

    def print_global_summary(self):
        """Prints global benchmark and detections table in terminal."""
        total_imgs = self.stats["total_images"]
        total_det = self.stats["total_detections"]
        total_tiles = self.stats["total_tiles"]
        avg_tiles = total_tiles / max(1, total_imgs)
        avg_det = total_det / max(1, total_imgs)

        avg_slicing = np.mean(self.stats["speeds"]["slicing"]) if self.stats["speeds"]["slicing"] else 0.0
        avg_inf = np.mean(self.stats["speeds"]["inference"]) if self.stats["speeds"]["inference"] else 0.0
        avg_fus = np.mean(self.stats["speeds"]["fusion"]) if self.stats["speeds"]["fusion"] else 0.0
        avg_tot = np.mean(self.stats["speeds"]["total"]) if self.stats["speeds"]["total"] else 0.0
        fps = (1000.0 / avg_tot) if avg_tot > 0 else 0.0

        print("\n" + "=" * 79)
        print(f" [GLOBAL RESUME] TILED INFERENCE SUMMARY")
        print("=" * 79)
        print(f" * Total images processed: {total_imgs}")
        print(f" * Total detections fused: {total_det} (Average per image: {avg_det:.2f})")
        print(f" * Total tiles evaluated: {total_tiles} (Average per image: {avg_tiles:.1f})")
        print(f" * Configuration: Tile {self.tile_size}x{self.tile_size} | Overlap {int(self.overlap_ratio*100)}% | Fusion: {self.fusion_method.upper()}")
        print(f" * Average Timings:")
        print(f"     - Slicing:    {avg_slicing:6.1f} ms")
        print(f"     - Inference:  {avg_inf:6.1f} ms (GPU batches)")
        print(f"     - Fusion:     {avg_fus:6.1f} ms")
        print(f"     - Total/Img:  {avg_tot:6.1f} ms ({fps:.1f} FPS)")
        print("-" * 79)

        if total_det > 0:
            print(f" {'CLASS':<22} {'TOTAL':<8} {'MEAN CONF (+/-STD)':<20} {'DISPERSION':<16} {'AVG AREA':<10}")
            print("-" * 79)
            for c_id, s in self.class_stats.items():
                c_name = self.class_names.get(c_id, f"Class_{c_id}")
                count = s["detections"]
                if count > 0:
                    c_conf = np.mean(s["confidences"])
                    c_std = np.std(s["confidences"])
                    c_area = np.mean(s["areas_pct"])
                    c_disp = mu.calculate_spatial_stats(s["bbox_centers"])
                    disp_str = f"+/-{c_disp['dispersion_x']:.2f}, +/-{c_disp['dispersion_y']:.2f}"
                    conf_str = f"{c_conf:.2f} (+/-{c_std:.2f})"
                    print(f" {c_name:<22} {count:<8} {conf_str:<20} {disp_str:<16} {c_area:.3f}%")
        else:
            print(" No valid detections recorded across the batch.")
        print("=" * 79 + "\n")


    def generate_plots(self):
        """Generates global diagnostic plots for the HTML report."""
        if not self.stats["confidences"]:
            return

        # Confidence distribution
        plot_generator.plot_confidence_histogram(
            tp_kept=self.stats["confidences"],
            fp_kept=[],
            tp_disc=[],
            fp_disc=[],
            threshold=self.conf_threshold,
            output_path=str(self.plots_dir / "tiled_confidence_distribution.png"),
            title=f"Tiled Inference Confidence Distribution ({self.fusion_method.upper()})",
            is_inference=True
        )

        # Spatial Heatmap
        if self.stats["bbox_centers_norm"]:
            plot_generator.plot_normalized_heatmap(
                centers=self.stats["bbox_centers_norm"],
                output_path=str(self.plots_dir / "tiled_heatmap.png"),
                title="Tiled Normalized Detection Heatmap",
                cmap='magma'
            )

    def generate_html_report(self, model_path=None, experiment_name="yolov8_s_default"):
        """Compiles metadata and renders the Jinja2 HTML report using pipeline builders."""
        print("📝 Generating Tiled Inference HTML Report...")
        self.generate_plots()

        tiled_json_path = self.output_dir / config.FILE_TILED_META
        meta_manager = TiledInferenceMetadata(tiled_json_path)
        meta_manager.set_timestamp(key_name="tiled_inference_date")

        total_imgs = max(1, self.stats["total_images"])
        avg_det = self.stats["total_detections"] / total_imgs
        avg_tiles = self.stats["total_tiles"] / total_imgs

        meta_manager.record_global_stats(
            total_images=self.stats["total_images"],
            total_detections=self.stats["total_detections"],
            avg_detections=avg_det,
            total_tiles=self.stats["total_tiles"],
            avg_tiles_per_img=avg_tiles,
            tile_size=self.tile_size,
            overlap_ratio=self.overlap_ratio,
            fusion_method=self.fusion_method,
            conf_threshold=self.conf_threshold,
            iou_threshold=self.iou_threshold
        )

        # Class breakdown
        for c_id, s in self.class_stats.items():
            c_name = self.class_names.get(c_id, f"Class {c_id}")
            safe_name = c_name.replace(" ", "_").lower()
            mean_conf = float(np.mean(s["confidences"])) if s["confidences"] else 0.0

            conf_stats = mu.calculate_1d_stats(s["confidences"])
            spatial_stats = mu.calculate_spatial_stats(s["bbox_centers"])
            size_stats = {
                "avg_area_pct": round(float(np.mean(s["areas_pct"])), 4) if s["areas_pct"] else 0.0,
                "min_area_px": round(float(np.min(s["areas_px"])), 1) if s["areas_px"] else 0.0,
                "max_area_px": round(float(np.max(s["areas_px"])), 1) if s["areas_px"] else 0.0
            }

            meta_manager.record_class_stats(
                class_name=c_name,
                safe_name=safe_name,
                detections=s["detections"],
                avg_confidence=mean_conf,
                conf_stats=conf_stats,
                spatial_stats=spatial_stats,
                size_stats=size_stats
            )

        # Speed stats
        avg_slicing = float(np.mean(self.stats["speeds"]["slicing"])) if self.stats["speeds"]["slicing"] else 0.0
        avg_inf = float(np.mean(self.stats["speeds"]["inference"])) if self.stats["speeds"]["inference"] else 0.0
        avg_fus = float(np.mean(self.stats["speeds"]["fusion"])) if self.stats["speeds"]["fusion"] else 0.0
        avg_tot = float(np.mean(self.stats["speeds"]["total"])) if self.stats["speeds"]["total"] else 0.0
        fps = round(1000.0 / avg_tot, 2) if avg_tot > 0 else 0.0

        meta_manager.record_speed_stats({
            "slicing_ms": round(avg_slicing, 1),
            "inference_ms": round(avg_inf, 1),
            "fusion_ms": round(avg_fus, 1),
            "total_ms": round(avg_tot, 1),
            "fps": fps
        })

        meta_manager.record_images_details(self.stats["images"])
        meta_manager.commit()

        if model_path:
            exp_dir = Path(model_path).parent.parent
        else:
            exp_dir = Path(config.PROJECT_DIR) / experiment_name

        dataset_meta_path = exp_dir / config.METADATA_FOLDER_NAME / config.FILE_DATASET_META
        if not dataset_meta_path.exists():
            dataset_meta_path = config.DATASET_METADATA_PATH

        train_meta_path = exp_dir / config.METADATA_FOLDER_NAME / config.FILE_TRAIN_META

        dataset_summary = DatasetMetadata(dataset_meta_path).get_html_summary() if dataset_meta_path.exists() else None
        train_summary = TrainMetadata(train_meta_path).get_html_summary() if train_meta_path.exists() else None

        html_context = {
            "report_title": "Tiled High-Res Inference Report",
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "stats": meta_manager.get_html_summary(),
            "tiled": meta_manager.get_html_summary(),
            "dataset": dataset_summary,
            "train": train_summary
        }


        generator = HTMLReportGenerator()
        report_html_path = self.output_dir / "tiled_inference_report.html"
        generator.generate_tiled_inference_html(str(report_html_path), html_context)
        return str(report_html_path)
