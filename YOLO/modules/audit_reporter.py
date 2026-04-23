import os
import numpy as np
import json
from datetime import datetime
from collections import defaultdict
from . import core_visual_utils as cvu
from .html_generator import HTMLReportGenerator
from . import plot_generator

class ReportGenerator:
    def __init__(self, output_dir, iou_threshold=0.5, user_conf_threshold=0.5, prefix="audit", class_names=None):
        self.output_dir = output_dir
        self.iou_threshold = iou_threshold
        self.user_conf_threshold = user_conf_threshold
        self.prefix = prefix
        
        self.class_names = class_names if class_names else {}
        
        self.class_stats = defaultdict(lambda: {
            "TP": 0, "FP": 0, "FN": 0,
            "confidences_TP": [],
            "confidences_FP": [],
            "discarded_TP": [],
            "discarded_FP": [],
            "bbox_centers": [], # <--- AÑADIDO: Rastreo espacial por clase
            "total_gt": 0,
            "all_predictions": []
        })
        
        self.global_stats = {
            "bbox_centers": [],
            "speeds": {"preprocess": [], "inference": [], "postprocess": []}
        }
        
        self.confusion_pairs = []
        
        self.plots_dir = os.path.join(output_dir, "plots", self.prefix)
        os.makedirs(self.plots_dir, exist_ok=True)

    def update(self, pred_boxes, pred_classes, gt_boxes, confidences, img_shape, speed_dict=None):
        h, w = img_shape
        img_stats = {"TP": 0, "FP": 0, "FN": 0, "poor_bbox": 0}

        if speed_dict:
            self.global_stats["speeds"]["preprocess"].append(speed_dict.get('preprocess', 0))
            self.global_stats["speeds"]["inference"].append(speed_dict.get('inference', 0))
            self.global_stats["speeds"]["postprocess"].append(speed_dict.get('postprocess', 0))

        gt_classes = []
        gt_coords = []
        for gt in gt_boxes:
            c_id = int(gt[0])
            gt_classes.append(c_id)
            gt_coords.append(gt[1:])
            self.class_stats[c_id]["total_gt"] += 1
            
        iou_matrix = cvu.calculate_iou_matrix(pred_boxes, gt_coords)
        
        matched_gt_all = set()   
        matched_gt_thresh = set() 

        pred_indices = np.argsort(confidences)[::-1]

        for i in pred_indices:
            pred = pred_boxes[i]
            conf = confidences[i]
            p_cls = int(pred_classes[i])
            
            if len(gt_coords) > 0:
                best_iou = np.max(iou_matrix[i])
                best_gt_idx = np.argmax(iou_matrix[i]) 
                g_cls = gt_classes[best_gt_idx]
            else:
                best_iou = 0
                best_gt_idx = -1
                g_cls = -1
                
            is_correct_match = (best_iou >= self.iou_threshold) and (best_gt_idx not in matched_gt_all) and (p_cls == g_cls)
            
            self.class_stats[p_cls]["all_predictions"].append({
                "conf": conf, "iou": best_iou, "is_tp": is_correct_match
            })
            
            if best_iou >= self.iou_threshold and best_gt_idx not in matched_gt_all:
                matched_gt_all.add(best_gt_idx)
                
            if conf >= self.user_conf_threshold:
                cx_abs = (pred[0] + pred[2]) / 2
                cy_abs = (pred[1] + pred[3]) / 2
                # <--- AÑADIDO: Guardamos la coordenada globalmente y en su clase
                self.global_stats["bbox_centers"].append((cx_abs/w, cy_abs/h))
                self.class_stats[p_cls]["bbox_centers"].append((cx_abs/w, cy_abs/h))
                
                if best_iou >= self.iou_threshold and best_gt_idx not in matched_gt_thresh:
                    matched_gt_thresh.add(best_gt_idx)
                    
                    if p_cls == g_cls:
                        self.class_stats[p_cls]["TP"] += 1
                        self.class_stats[p_cls]["confidences_TP"].append(conf)
                        self.confusion_pairs.append((g_cls, p_cls))
                        img_stats["TP"] += 1
                    else:
                        self.class_stats[p_cls]["FP"] += 1
                        self.class_stats[p_cls]["confidences_FP"].append(conf)
                        self.confusion_pairs.append((g_cls, p_cls))
                        img_stats["FP"] += 1
                else:
                    self.class_stats[p_cls]["FP"] += 1
                    self.class_stats[p_cls]["confidences_FP"].append(conf)
                    self.confusion_pairs.append((-1, p_cls)) 
                    if 0.1 <= best_iou < self.iou_threshold:
                        img_stats["poor_bbox"] += 1
                    else:
                        img_stats["FP"] += 1
            elif conf >= 0.2:
                if best_iou >= self.iou_threshold and best_gt_idx not in matched_gt_thresh and p_cls == g_cls:
                    self.class_stats[p_cls]["discarded_TP"].append(conf)
                else:
                    self.class_stats[p_cls]["discarded_FP"].append(conf)

        for gt_idx, g_cls in enumerate(gt_classes):
            if gt_idx not in matched_gt_thresh:
                self.class_stats[g_cls]["FN"] += 1
                self.confusion_pairs.append((g_cls, -1)) 
                img_stats["FN"] += 1
        
        return img_stats

    def calculate_ap(self, class_id, iou_thresh):
        if class_id not in self.class_stats or self.class_stats[class_id]["total_gt"] == 0:
            return 0.0, [0.0], [0.0], [0.0], [0.0]
            
        preds = sorted(self.class_stats[class_id]["all_predictions"], key=lambda x: x["conf"], reverse=True)
        total_gt = self.class_stats[class_id]["total_gt"]
        
        tps = np.zeros(len(preds))
        fps = np.zeros(len(preds))
        confs = np.zeros(len(preds))
        
        for i, p in enumerate(preds):
            confs[i] = p["conf"]
            if p["is_tp"] and p["iou"] >= iou_thresh:
                tps[i] = 1
            else:
                fps[i] = 1
                
        tp_cumsum = np.cumsum(tps)
        fp_cumsum = np.cumsum(fps)
        
        recalls = tp_cumsum / (total_gt + 1e-6)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-6)
        
        p_smooth = np.concatenate(([1.0], precisions, [0.0]))
        r_smooth = np.concatenate(([0.0], recalls, [1.0]))
        for i in range(len(p_smooth) - 1, 0, -1):
            p_smooth[i - 1] = np.maximum(p_smooth[i - 1], p_smooth[i])
            
        indices = np.where(r_smooth[1:] != r_smooth[:-1])[0]
        ap = np.sum((r_smooth[indices + 1] - r_smooth[indices]) * p_smooth[indices + 1])
        
        return float(ap), p_smooth.tolist(), r_smooth.tolist(), confs.tolist(), f1_scores.tolist()

    def generate_plots(self):
        print("📊 Generating Multi-Class statistical plots (including Spatial Data)...")

        plot_generator.plot_confusion_matrix(
            self.confusion_pairs, self.class_names,
            os.path.join(self.plots_dir, "confusion_matrix.png")
        )

        all_tp_conf, all_fp_conf, all_tp_disc, all_fp_disc = [], [], [], []
        for s in self.class_stats.values():
            all_tp_conf.extend(s["confidences_TP"])
            all_fp_conf.extend(s["confidences_FP"])
            all_tp_disc.extend(s["discarded_TP"])
            all_fp_disc.extend(s["discarded_FP"])
            
        # Globales
        plot_generator.plot_confidence_histogram(
            all_tp_conf, all_fp_conf, all_tp_disc, all_fp_disc,
            self.user_conf_threshold,
            os.path.join(self.plots_dir, "confidence_dist.png")
        )

        plot_generator.plot_normalized_heatmap(
            self.global_stats["bbox_centers"],
            os.path.join(self.plots_dir, "heatmap.png"),
            title="Normalized Detection Heatmap (Global)", cmap='inferno'
        )

        pr_data = {}
        f1_data = {}
        
        for c_id, stats in self.class_stats.items():
            ap50, precisions, recalls, confs, f1_scores = self.calculate_ap(c_id, 0.5)
            c_name = self.class_names.get(c_id, f"Clase {c_id}")
            safe_name = c_name.replace(" ", "_")
            
            # --- AÑADIDO: Generar Gráficos Espaciales PER-CLASS ---
            plot_generator.plot_confidence_histogram(
                stats["confidences_TP"], stats["confidences_FP"], 
                stats["discarded_TP"], stats["discarded_FP"],
                self.user_conf_threshold,
                os.path.join(self.plots_dir, f"confidence_dist_{safe_name}.png"),
                title=f"Confidence Distribution ({c_name})"
            )
            plot_generator.plot_normalized_heatmap(
                stats["bbox_centers"],
                os.path.join(self.plots_dir, f"heatmap_{safe_name}.png"),
                title=f"Normalized Detection Heatmap ({c_name})", cmap='inferno'
            )
            # -----------------------------------------------------

            pr_data[c_id] = {
                'precisions': precisions, 'recalls': recalls, 
                'ap50': ap50, 'name': c_name
            }
            
            best_f1, best_conf = 0.0, 0.0
            if len(f1_scores) > 0:
                best_idx = np.argmax(f1_scores)
                best_f1, best_conf = f1_scores[best_idx], confs[best_idx]
                
            f1_data[c_id] = {
                'confs': confs, 'f1s': f1_scores, 
                'best_f1': best_f1, 'best_conf': best_conf, 'name': c_name
            }
            
        plot_generator.plot_pr_curve(pr_data, os.path.join(self.plots_dir, "pr_curve.png"))
        plot_generator.plot_f1_curve(f1_data, os.path.join(self.plots_dir, "f1_curve.png"))

    def generate_html_report(self, experiment_name="yolov8_s_default"):
        print("📝 Compiling Multiclass numerical metrics for the report...")
        
        global_tp = global_fp = global_fn = global_total_real = global_total_pred = 0
        class_metrics = {}
        ap50_list, ap50_95_list = [], []
        all_conf_tp, all_conf_fp = [], []
        
        for c_id, stats in self.class_stats.items():
            c_name = self.class_names.get(c_id, f"Class {c_id}")
            safe_name = c_name.replace(" ", "_")
            
            c_tp, c_fp, c_fn = stats["TP"], stats["FP"], stats["FN"]
            c_pred = c_tp + c_fp
            c_real = c_tp + c_fn
            
            global_tp += c_tp
            global_fp += c_fp
            global_fn += c_fn
            global_total_real += c_real
            global_total_pred += c_pred
            
            all_conf_tp.extend(stats["confidences_TP"])
            all_conf_fp.extend(stats["confidences_FP"])
            
            prec = c_tp / c_pred if c_pred > 0 else 0
            rec = c_tp / c_real if c_real > 0 else 0
            f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
            
            ap50, p_sm, r_sm, cfs, f1s = self.calculate_ap(c_id, 0.5)
            
            ap_sum = 0
            thresholds = np.arange(0.5, 1.0, 0.05)
            for t in thresholds:
                ap_t, *_ = self.calculate_ap(c_id, t)
                ap_sum += ap_t
            ap50_95 = ap_sum / len(thresholds)
            
            best_f1, best_conf = 0.0, 0.0
            if len(f1s) > 0:
                best_idx = np.argmax(f1s)
                best_f1, best_conf = float(f1s[best_idx]), float(cfs[best_idx])
                
            ap50_list.append(ap50)
            ap50_95_list.append(ap50_95)
            
            class_metrics[c_name] = {
                "precision": prec, "recall": rec, "f1": f1,
                "ap_50": ap50, "ap_50_95": ap50_95,
                "optimal_f1": best_f1, "optimal_conf": best_conf,
                "tp": c_tp, "fp": c_fp, "fn": c_fn,
                "safe_name": safe_name, # <--- AÑADIDO PARA EL HTML
                "confidence_stats": {
                    "True_Positives": cvu.calculate_1d_stats(stats["confidences_TP"]),
                    "False_Positives": cvu.calculate_1d_stats(stats["confidences_FP"]),
                },
                "spatial_stats": cvu.calculate_spatial_stats(stats.get("bbox_centers", []))
            }

        g_prec = global_tp / global_total_pred if global_total_pred > 0 else 0
        g_rec = global_tp / global_total_real if global_total_real > 0 else 0
        g_f1 = 2 * (g_prec * g_rec) / (g_prec + g_rec) if (g_prec + g_rec) > 0 else 0
        mAP50 = np.mean(ap50_list) if ap50_list else 0
        mAP50_95 = np.mean(ap50_95_list) if ap50_95_list else 0
        
        target_class = list(self.class_stats.keys())[0] if self.class_stats else 0
        
        metrics_dict = {
            "precision": g_prec, "recall": g_rec, "f1": g_f1,
            "map_50": mAP50, "map_50_95": mAP50_95,
            "optimal_f1": class_metrics.get(self.class_names.get(target_class, ""), {}).get("optimal_f1", 0),
            "total_real": global_total_real, "total_pred": global_total_pred,
            "tp": global_tp, "fp": global_fp, "fn": global_fn,
            "per_class": class_metrics,
            "prefix": self.prefix
        }

        metrics_dict["confidence_stats"] = {
            "True_Positives": cvu.calculate_1d_stats(all_conf_tp),
            "False_Positives": cvu.calculate_1d_stats(all_conf_fp),
        }
        metrics_dict["spatial_stats"] = cvu.calculate_spatial_stats(self.global_stats["bbox_centers"])

        avg_pre = np.mean(self.global_stats["speeds"]["preprocess"]) if self.global_stats["speeds"]["preprocess"] else 0
        avg_inf = np.mean(self.global_stats["speeds"]["inference"]) if self.global_stats["speeds"]["inference"] else 0
        avg_post = np.mean(self.global_stats["speeds"]["postprocess"]) if self.global_stats["speeds"]["postprocess"] else 0
        total_ms = avg_pre + avg_inf + avg_post

        metrics_dict["speed_stats"] = {
            "preprocess_ms": round(avg_pre, 2), "inference_ms": round(avg_inf, 2),
            "postprocess_ms": round(avg_post, 2), "total_ms": round(total_ms, 2),
            "fps": round(1000 / total_ms, 2) if total_ms > 0 else 0
        }

        audit_metadata = {
            "audit_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "metrics": metrics_dict,
            "evaluation_params": {"iou_threshold": self.iou_threshold}
        }
        
        audit_json_path = os.path.join(self.output_dir, f"{self.prefix}_metadata.json")
        try:
            with open(audit_json_path, "w", encoding='utf-8') as f:
                json.dump(audit_metadata, f, indent=4)
        except Exception as e:
            print(f"⚠️ Could not save audit JSON: {e}")

        templates_dir = os.path.join(os.getcwd(), "modules", "templates")
        generator = HTMLReportGenerator(templates_dir, cvu.PROJECT_DIR, os.path.join(os.getcwd(), "dataset_yolo_output"))
        generator.generate_audit_html(os.path.join(self.output_dir, f"{self.prefix}_report.html"), experiment_name, metrics_dict)