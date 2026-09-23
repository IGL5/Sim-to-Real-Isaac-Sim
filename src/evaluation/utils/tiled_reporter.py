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
        std_conf = float(np.std(scores)) if len(scores) > 0 else 0.0

        # Calculate per-image class breakdown
        class_breakdown = []
        if len(boxes) > 0:
            u_classes = np.unique(classes)
            for c_id in u_classes:
                c_id_int = int(c_id)
                idx = np.where(classes == c_id)[0]
                c_name = self.class_names.get(c_id_int, f"Class_{c_id_int}")
                c_scores = scores[idx]
                c_boxes = boxes[idx]
                c_areas = [((b[2] - b[0]) * (b[3] - b[1]) / image_area) * 100.0 for b in c_boxes] if image_area > 0 else []
                class_breakdown.append({
                    "class_id": c_id_int,
                    "name": c_name,
                    "count": int(len(idx)),
                    "mean_conf": round(float(np.mean(c_scores)), 3),
                    "std_conf": round(float(np.std(c_scores)), 3),
                    "avg_area_pct": round(float(np.mean(c_areas)), 4) if c_areas else 0.0
                })

        img_areas_pct = [((b[2] - b[0]) * (b[3] - b[1]) / image_area) * 100.0 for b in boxes] if image_area > 0 and len(boxes) > 0 else []
        avg_area_pct = float(np.mean(img_areas_pct)) if img_areas_pct else 0.0
        classes_summary_str = ", ".join([f"{c['name']} ({c['count']})" for c in class_breakdown]) if class_breakdown else "None"
        tot_time = timings.get("total", 0.0)
        fps = round(1000.0 / tot_time, 1) if tot_time > 0 else 0.0

        image_info = {
            "index": self.stats["total_images"],
            "filename": filename,
            "evidence_filename": evidence_filename or f"PRED_{filename}",
            "width": img_w,
            "height": img_h,
            "num_tiles": num_tiles,
            "detections": len(boxes),
            "mean_conf": round(mean_conf, 3),
            "std_conf": round(std_conf, 3),
            "avg_area_pct": round(avg_area_pct, 4),
            "dispersion": dispersion_info,
            "class_breakdown": class_breakdown,
            "classes_summary": classes_summary_str,
            "time_ms": round(tot_time, 1),
            "fps": fps,
            "timings": {k: round(v, 1) for k, v in timings.items()}
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

        # Condensed Dispersion Distribution across images
        clustered = 0
        moderate = 0
        dispersed = 0
        single_or_empty = 0
        valid_pairwise = []

        for img_data in self.stats["images"]:
            disp = img_data.get("dispersion", {})
            cat = disp.get("category", "")
            p_dist = disp.get("mean_pairwise_dist", 0.0)
            if p_dist > 0:
                valid_pairwise.append(p_dist)

            if "Muy Agrupado" in cat or "Cluster" in cat:
                clustered += 1
            elif "Modera" in cat or "Uniforme" in cat:
                moderate += 1
            elif "Disperso" in cat or "Distribuido" in cat:
                dispersed += 1
            else:
                single_or_empty += 1

        tot_imgs = max(1, len(self.stats["images"]))
        dispersion_summary = {
            "clustered_count": clustered,
            "clustered_pct": round(clustered / tot_imgs * 100.0, 1),
            "moderate_count": moderate,
            "moderate_pct": round(moderate / tot_imgs * 100.0, 1),
            "dispersed_count": dispersed,
            "dispersed_pct": round(dispersed / tot_imgs * 100.0, 1),
            "empty_count": single_or_empty,
            "empty_pct": round(single_or_empty / tot_imgs * 100.0, 1),
            "avg_pairwise_dist": round(float(np.mean(valid_pairwise)), 4) if valid_pairwise else 0.0
        }
        meta_manager.record_dispersion_summary(dispersion_summary)

        # Condensed Scale & Area Distribution across all detections
        all_areas = self.stats["bbox_areas_pct"]
        if all_areas:
            tot_boxes = max(1, len(all_areas))
            small_count = sum(1 for a in all_areas if a < 0.1)
            medium_count = sum(1 for a in all_areas if 0.1 <= a <= 1.0)
            large_count = sum(1 for a in all_areas if a > 1.0)

            area_summary = {
                "avg_area_pct": round(float(np.mean(all_areas)), 4),
                "median_area_pct": round(float(np.median(all_areas)), 4),
                "min_area_pct": round(float(np.min(all_areas)), 4),
                "max_area_pct": round(float(np.max(all_areas)), 4),
                "small_pct": round(small_count / tot_boxes * 100.0, 1),
                "medium_pct": round(medium_count / tot_boxes * 100.0, 1),
                "large_pct": round(large_count / tot_boxes * 100.0, 1),
                "small_count": small_count,
                "medium_count": medium_count,
                "large_count": large_count
            }
        else:
            area_summary = {
                "avg_area_pct": 0.0,
                "median_area_pct": 0.0,
                "min_area_pct": 0.0,
                "max_area_pct": 0.0,
                "small_pct": 0.0,
                "medium_pct": 0.0,
                "large_pct": 0.0,
                "small_count": 0,
                "medium_count": 0,
                "large_count": 0
            }
        meta_manager.record_area_summary(area_summary)

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
