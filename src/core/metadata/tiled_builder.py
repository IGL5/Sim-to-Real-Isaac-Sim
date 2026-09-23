from src.core.metadata.base_manager import BaseMetadataManager

class TiledInferenceMetadata(BaseMetadataManager):
    """
    Metadata builder/repository for Tiled/Sliced Inference on high-resolution images.
    Adheres strictly to the BaseMetadataManager pattern (no direct json imports).
    """
    def __init__(self, filepath):
        super().__init__(filepath)

    def record_global_stats(self, total_images, total_detections, avg_detections,
                            total_tiles, avg_tiles_per_img, tile_size, overlap_ratio,
                            fusion_method, conf_threshold, iou_threshold):
        if "stats" not in self.data:
            self.data["stats"] = {}
        self.data["stats"].update({
            "total_images": total_images,
            "total_detections": total_detections,
            "avg_detections": avg_detections,
            "total_tiles": total_tiles,
            "avg_tiles_per_img": avg_tiles_per_img,
            "tile_size": tile_size,
            "overlap_ratio": overlap_ratio,
            "fusion_method": fusion_method,
            "conf_threshold": conf_threshold,
            "iou_threshold": iou_threshold
        })

    def record_class_stats(self, class_name, safe_name, detections, avg_confidence, conf_stats, spatial_stats, size_stats):
        if "stats" not in self.data:
            self.data["stats"] = {}
        if "per_class" not in self.data["stats"]:
            self.data["stats"]["per_class"] = {}

        self.data["stats"]["per_class"][class_name] = {
            "safe_name": safe_name,
            "detections": detections,
            "avg_confidence": avg_confidence,
            "confidence_stats": conf_stats,
            "spatial_stats": spatial_stats,
            "size_stats": size_stats
        }

    def record_speed_stats(self, speed_stats):
        if "stats" not in self.data:
            self.data["stats"] = {}
        self.data["stats"]["speed_stats"] = speed_stats

    def record_dispersion_summary(self, dispersion_summary):
        if "stats" not in self.data:
            self.data["stats"] = {}
        self.data["stats"]["dispersion_summary"] = dispersion_summary

    def record_area_summary(self, area_summary):
        if "stats" not in self.data:
            self.data["stats"] = {}
        self.data["stats"]["area_summary"] = area_summary

    def record_images_details(self, images_details):
        if "stats" not in self.data:
            self.data["stats"] = {}
        self.data["stats"]["images"] = images_details

    def get_html_summary(self):
        """
        Translates raw nested metadata into a flattened, typed DTO optimal for Jinja2 templates.
        """
        stats_sec = self.data.get("stats", {})
        per_class_raw = stats_sec.get("per_class", {})

        html_classes = []
        for c_name, c_data in per_class_raw.items():
            conf_stats = c_data.get("confidence_stats", {})
            spatial = c_data.get("spatial_stats", {})
            size_st = c_data.get("size_stats", {})

            html_classes.append({
                "name": c_name,
                "safe_name": c_data.get("safe_name", "class"),
                "detections": c_data.get("detections", 0),
                "avg_confidence": round(c_data.get("avg_confidence", 0.0), 3),
                "conf_mean": conf_stats.get("mean", 0.0),
                "conf_std": conf_stats.get("std", 0.0),
                "com_x": spatial.get("center_of_mass_x", 0.5),
                "com_y": spatial.get("center_of_mass_y", 0.5),
                "disp_x": spatial.get("dispersion_x", 0.0),
                "disp_y": spatial.get("dispersion_y", 0.0),
                "avg_area_pct": size_st.get("avg_area_pct", 0.0),
                "min_area_px": size_st.get("min_area_px", 0),
                "max_area_px": size_st.get("max_area_px", 0)
            })

        return {
            "total_images": stats_sec.get("total_images", 0),
            "total_detections": stats_sec.get("total_detections", 0),
            "avg_detections": round(stats_sec.get("avg_detections", 0.0), 2),
            "total_tiles": stats_sec.get("total_tiles", 0),
            "avg_tiles_per_img": round(stats_sec.get("avg_tiles_per_img", 0.0), 1),
            "tile_size": stats_sec.get("tile_size", 640),
            "overlap_ratio": stats_sec.get("overlap_ratio", 0.2),
            "fusion_method": stats_sec.get("fusion_method", "wbf"),
            "conf_threshold": stats_sec.get("conf_threshold", 0.4),
            "iou_threshold": stats_sec.get("iou_threshold", 0.5),
            "classes": html_classes,
            "speed_stats": stats_sec.get("speed_stats", {}),
            "dispersion_summary": stats_sec.get("dispersion_summary", {}),
            "area_summary": stats_sec.get("area_summary", {}),
            "images": stats_sec.get("images", [])
        }
