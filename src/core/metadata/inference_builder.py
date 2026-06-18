from src.core.metadata.base_manager import BaseMetadataManager

class InferenceMetadata(BaseMetadataManager):
    def __init__(self, filepath):
        super().__init__(filepath)

    def record_global_stats(self, total_images, total_detections, avg_detections, total_overlaps, overlap_events):
        if "stats" not in self.data: 
            self.data["stats"] = {}
        self.data["stats"].update({
            "total_images": total_images,
            "total_detections": total_detections,
            "avg_detections": avg_detections,
            "total_overlaps": total_overlaps,
            "overlap_events": overlap_events,
        })
        
    def record_class_stats(self, class_name, safe_name, detections, avg_confidence, conf_stats, spatial_stats):
        if "stats" not in self.data: 
            self.data["stats"] = {}
        if "per_class" not in self.data["stats"]: 
            self.data["stats"]["per_class"] = {}
            
        self.data["stats"]["per_class"][class_name] = {
            "safe_name": safe_name,
            "detections": detections,
            "avg_confidence": avg_confidence,
            "confidence_stats": conf_stats,
            "spatial_stats": spatial_stats
        }

    def record_confidence_stats(self, conf_stats):
        if "stats" not in self.data: 
            self.data["stats"] = {}
        self.data["stats"]["confidence_stats"] = conf_stats

    def record_spatial_stats(self, spatial_stats):
        if "stats" not in self.data: 
            self.data["stats"] = {}
        self.data["stats"]["spatial_stats"] = spatial_stats

    def record_speed_stats(self, speed_stats):
        if "stats" not in self.data: 
            self.data["stats"] = {}
        self.data["stats"]["speed_stats"] = speed_stats

    # --- THE GETTER FOR THE VIEW (HTML) ---
    def get_html_summary(self):
        """
        Transforma el JSON anidado en un DTO plano listo para el reporte de inferencia.
        """
        stats_sec = self.data.get("stats", {})
        per_class_raw = stats_sec.get("per_class", {})
        
        # Procesamos y estandarizamos el desglose por clases para la vista
        html_classes = []
        for c_name, c_data in per_class_raw.items():
            spatial = c_data.get("spatial_stats", {})
            conf_stats = c_data.get("confidence_stats", {})
            
            html_classes.append({
                "name": c_name,
                "safe_name": c_data.get("safe_name", "class"),
                "detections": c_data.get("detections", 0),
                "avg_confidence": round(c_data.get("avg_confidence", 0.0), 3),
                
                # Aplanamos el segundo nivel de estadísticas espaciales
                "com_x": spatial.get("center_of_mass_x", 0.5),
                "com_y": spatial.get("center_of_mass_y", 0.5),
                "disp_x": spatial.get("dispersion_x", 0.0),
                "disp_y": spatial.get("dispersion_y", 0.0),
                
                # Aplanamos las estadísticas de confianza (1D)
                "conf_mean": conf_stats.get("mean", 0.0),
                "conf_std": conf_stats.get("std", 0.0)
            })

        # Retornamos la bandeja limpia con Namespaces
        return {
            "total_images": stats_sec.get("total_images", 0),
            "total_detections": stats_sec.get("total_detections", 0),
            "avg_detections": round(stats_sec.get("avg_detections", 0.0), 2),
            "total_overlaps": stats_sec.get("total_overlaps", 0),
            "overlap_events": stats_sec.get("overlap_events", []),
            
            "classes": html_classes,
            
            "speed_stats": stats_sec.get("speed_stats", {}),
            "spatial_global": stats_sec.get("spatial_stats", {}),
            "confidence_global": stats_sec.get("confidence_stats", {})
        }
