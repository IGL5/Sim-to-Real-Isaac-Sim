from src.core.metadata.base_manager import BaseMetadataManager

class AuditMetadata(BaseMetadataManager):
    def __init__(self, filepath):
        super().__init__(filepath)

    def record_evaluation_params(self, iou_threshold):
        self.update_section("evaluation_params", {"iou_threshold": iou_threshold})

    def record_class_metric(self, class_name, safe_name, precision, recall, f1, ap_50, ap_50_95, opt_f1, opt_conf, tp, fp, fn, conf_tp_stats, conf_fp_stats, spatial_stats):
        """The Builder takes care of structuring and nesting the data for each class."""
        if "metrics" not in self.data: 
            self.data["metrics"] = {}
        if "per_class" not in self.data["metrics"]: 
            self.data["metrics"]["per_class"] = {}
            
        self.data["metrics"]["per_class"][class_name] = {
            "precision": precision, "recall": recall, "f1": f1,
            "ap_50": ap_50, "ap_50_95": ap_50_95,
            "optimal_f1": opt_f1, "optimal_conf": opt_conf,
            "tp": tp, "fp": fp, "fn": fn,
            "safe_name": safe_name, 
            "confidence_stats": {
                "True_Positives": conf_tp_stats,
                "False_Positives": conf_fp_stats,
            },
            "spatial_stats": spatial_stats
        }

    def record_metrics(self, precision, recall, f1, map50, map50_95, optimal_f1, total_real, total_pred, tp, fp, fn, prefix):
        """Global metrics of the evaluation"""
        if "metrics" not in self.data: self.data["metrics"] = {}
        self.data["metrics"].update({
            "macro_precision": precision,
            "macro_recall": recall,
            "macro_f1": f1,
            "map_50": map50,
            "map_50_95": map50_95,
            "optimal_f1": optimal_f1,
            "total_real": total_real,
            "total_pred": total_pred,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "prefix": prefix
        })

    def record_confidence_stats(self, conf_tp_stats, conf_fp_stats):
        if "metrics" not in self.data: self.data["metrics"] = {}
        self.data["metrics"]["confidence_stats"] = {
            "True_Positives": conf_tp_stats,
            "False_Positives": conf_fp_stats
        }

    def record_spatial_stats(self, spatial_stats):
        if "metrics" not in self.data: self.data["metrics"] = {}
        self.data["metrics"]["spatial_stats"] = spatial_stats

    def record_speed_stats(self, speed_stats):
        if "metrics" not in self.data: self.data["metrics"] = {}
        self.data["metrics"]["speed_stats"] = speed_stats

    # --- THE GETTER FOR THE VIEW (HTML) ---
    def get_html_summary(self):
        """
        Transforma el JSON anidado en un DTO perfectamente adaptado 
        para que la plantilla HTML renderice sin esfuerzo ni peligro.
        """
        metrics_sec = self.data.get("metrics", {})
        per_class_raw = metrics_sec.get("per_class", {})
        
        # Procesamos y estandarizamos el desglose por clases para la vista
        html_classes = []
        for c_name, c_data in per_class_raw.items():
            spatial = c_data.get("spatial_stats", {})
            conf_stats = c_data.get("confidence_stats", {})
            
            html_classes.append({
                "name": c_name,
                "safe_name": c_data.get("safe_name", "class"),
                "precision": round(c_data.get("precision", 0.0) * 100, 2),
                "recall": round(c_data.get("recall", 0.0) * 100, 2),
                "f1": round(c_data.get("f1", 0.0), 2),
                "ap50": round(c_data.get("ap_50", 0.0), 3),
                "ap50_95": round(c_data.get("ap_50_95", 0.0), 3),
                "optimal_f1": round(c_data.get("optimal_f1", 0.0), 2),
                "optimal_conf": round(c_data.get("optimal_conf", 0.0), 2),
                "tp": c_data.get("tp", 0),
                "fp": c_data.get("fp", 0),
                "fn": c_data.get("fn", 0),
                
                # Aplanamos el segundo nivel de estadísticas espaciales para el HTML
                "com_x": spatial.get("center_of_mass_x", 0.5),
                "com_y": spatial.get("center_of_mass_y", 0.5),
                "disp_x": spatial.get("dispersion_x", 0.0),
                "disp_y": spatial.get("dispersion_y", 0.0),
                
                # Aplanamos las confianzas medias
                "conf_tp_mean": conf_stats.get("True_Positives", {}).get("mean", 0.0),
                "conf_fp_mean": conf_stats.get("False_Positives", {}).get("mean", 0.0)
            })

        # Retornamos la bandeja limpia con Namespaces claros
        return {
            "prefix": metrics_sec.get("prefix", "audit"),
            "global": {
                "precision": round(metrics_sec.get("macro_precision", 0.0) * 100, 2),
                "recall": round(metrics_sec.get("macro_recall", 0.0) * 100, 2),
                "f1": round(metrics_sec.get("macro_f1", 0.0), 2),
                "map50": round(metrics_sec.get("map_50", 0.0), 3),
                "map50_95": round(metrics_sec.get("map_50_95", 0.0), 3),
                "optimal_f1": round(metrics_sec.get("optimal_f1", 0.0), 2),
                "total_real": metrics_sec.get("total_real", 0),
                "total_pred": metrics_sec.get("total_pred", 0),
                "tp": metrics_sec.get("tp", 0),
                "fp": metrics_sec.get("fp", 0),
                "fn": metrics_sec.get("fn", 0),
            },
            "classes": html_classes,
            "speeds": metrics_sec.get("speed_stats", {}),
            "spatial_global": metrics_sec.get("spatial_stats", {}),
            "confidence_global": metrics_sec.get("confidence_stats", {})
        }