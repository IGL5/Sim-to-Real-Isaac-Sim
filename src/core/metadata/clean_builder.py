from src.core.metadata.base_manager import BaseMetadataManager

class CleaningMetadata(BaseMetadataManager):
    def __init__(self, filepath):
        super().__init__(filepath)

    def record_cleaning_stats(self, stats, move_empty_flag, clean_labels_flag):
        """
        Accumulates cleaning metrics over the existing metadata.
        The cleaner passes the raw dictionaries and the Builder handles the accumulation logic.
        """
        # Get the previous section
        cleaning_meta = self.data.get("cleaning", {})
        
        # 1. Base accumulation
        payload = {
            "corrupted_deleted": cleaning_meta.get("corrupted_deleted", 0) + stats["corrupted"],
            "empty_deleted": cleaning_meta.get("empty_deleted", 0) + stats["empty"],
            "valid_kept": stats["kept"] # Overwritten with the final count of this pass
        }
        
        if move_empty_flag:
            payload["empty_were_moved"] = True
            
        # 2. Complex logic for labels
        if clean_labels_flag:
            prev_labels = cleaning_meta.get("labels_stats", {})
            prev_reasons = prev_labels.get("removal_reasons", {})
            
            accumulated_removed = prev_labels.get("labels_removed", 0) + stats["labels_removed"]
            real_total_labels = stats["total_labels"] + prev_labels.get("labels_removed", 0)
            
            perc_removed = 0.0
            if real_total_labels > 0:
                perc_removed = round((accumulated_removed / real_total_labels) * 100, 2)
            
            payload["labels_stats"] = {
                "total_labels_found": real_total_labels,
                "labels_removed": accumulated_removed,
                "labels_saved_by_whitelist": stats["labels_saved"], # Labels saved in this pass
                "percentage_removed_percent": perc_removed,
                "removal_reasons": {
                    "occlusion": prev_reasons.get("occlusion", 0) + stats["reason_occ"],
                    "truncation": prev_reasons.get("truncation", 0) + stats["reason_trunc"],
                    "small_area": prev_reasons.get("small_area", 0) + stats["reason_area"],
                    "edge_cut": prev_reasons.get("edge_cut", 0) + stats["reason_edge"],
                    "giant_bbox": prev_reasons.get("giant_bbox", 0) + stats["reason_giant"]
                }
            }

        # Update the section
        self.update_section("cleaning", payload)

        # Synchronize with the "performance" section of simulation
        if "performance" in self.data:
            self.data["performance"]["total_valid_frames_after_cleaning"] = stats["kept"]

    # --- THE GETTER FOR THE VIEW (HTML) ---
    @staticmethod
    def get_html_summary_from_session(session_data):
        """Devuelve los datos de limpieza aplanados para la vista."""
        cleaning = session_data.get("cleaning", {})
        if not cleaning:
            return None
            
        return {
            "corrupted": cleaning.get("corrupted_deleted", 0),
            "empty": cleaning.get("empty_deleted", 0),
            "kept": cleaning.get("valid_kept", 0),
        }