from pathlib import Path
from src.core.metadata.base_manager import BaseMetadataManager
from src.core.metadata.sim_builder import SimulationMetadata
from src.core.metadata.clean_builder import CleaningMetadata
from src.core import config


class DatasetMetadata(BaseMetadataManager):
    def __init__(self, filepath):
        super().__init__(filepath)
        self._ensure_structure()

    def _ensure_structure(self):
        """Asegura que el esquema base de un Master Dataset exista siempre."""
        if "global_totals" not in self.data:
            self.data["global_totals"] = {
                "train": {"images": 0, "objects": 0, "backgrounds": 0},
                "val": {"images": 0, "objects": 0, "backgrounds": 0},
                "test": {"images": 0, "objects": 0, "backgrounds": 0},
                "total_images": 0,
                "total_objects": 0,
                "total_backgrounds": 0,
                "size_mb": 0.0,
                "avg_image_mb": 0.0
            }
        if "sessions" not in self.data:
            self.data["sessions"] = []

    def _get_dir_size(self, dir_path):
        """Calcula el tamaño total de un directorio en bytes de forma recursiva."""
        total = 0
        path = Path(dir_path)
        if path.exists():
            for entry in path.rglob('*'):
                if entry.is_file():
                    total += entry.stat().st_size
        return total

    def record_session(self, batch_id, source_meta_path, train_stats, val_stats, test_stats, total_added_imgs):
        """
        Registra una nueva sesión (tanda de datos) y actualiza los totales históricos.
        Absorbe la información del JSON de Simulación/Limpieza (source_meta_path).
        """
        # 1. Preparar la sesión (intentando fusionar los datos previos del simulador)
        session_meta = {"batch_id": batch_id}
        session_meta.update(self.read_json(source_meta_path))

        # 2. Inyectar las estadísticas de división (Train/Val/Test) a esta sesión
        session_meta["yolo_split"] = {
            "train": train_stats,
            "val": val_stats,
            "test": test_stats,
            "total_added": total_added_imgs
        }

        # Guardar en el histórico de sesiones
        self.data["sessions"].append(session_meta)

        # 3. Actualizar Totales Globales del Dataset
        for split, stats in [("train", train_stats), ("val", val_stats), ("test", test_stats)]:
            self.data["global_totals"][split]["images"] += stats["images"]
            self.data["global_totals"][split]["objects"] += stats["objects"]
            self.data["global_totals"][split]["backgrounds"] += stats["backgrounds"]
            
            self.data["global_totals"]["total_images"] += stats["images"]
            self.data["global_totals"]["total_objects"] += stats["objects"]
            self.data["global_totals"]["total_backgrounds"] += stats["backgrounds"]

        # 4. Calcular el peso en disco (MB) del Dataset Procesado
        total_size_bytes = self._get_dir_size(config.DATASET_IMAGES)
        total_size_mb = round(total_size_bytes / (1024 * 1024), 2)
        
        total_imgs = self.data["global_totals"]["total_images"]
        avg_img_mb = round(total_size_mb / total_imgs, 2) if total_imgs > 0 else 0
        
        self.data["global_totals"]["size_mb"] = total_size_mb
        self.data["global_totals"]["avg_image_mb"] = avg_img_mb

    # --- THE GETTER FOR THE VIEW (HTML) ---
    def get_html_summary(self):
        """
        Actúa como un ViewModel. Extrae el EDA de la última sesión
        y calcula las variables relativas de dibujo (visuals) para el HTML.
        """
        import math
        
        summary = {
            "global_totals": self.data.get("global_totals", {}),
            "latest_session": None,
            "latest_eda": None,
            "visuals": {}
        }

        sessions = self.data.get("sessions", [])
        if not sessions: return summary

        # Recuperar la última sesión y su EDA
        latest_session = sessions[-1]
        summary["latest_session"] = latest_session

        summary["sim_visuals"] = SimulationMetadata.get_html_summary_from_session(latest_session)
        summary["clean_stats"] = CleaningMetadata.get_html_summary_from_session(latest_session)
        
        latest_eda = latest_session.get("yolo_split", {}).get("train", {}).get("eda")
        dataset_visuals = {}

        # 2. Cálculos de Dibujo del EDA (Si existen)
        if latest_eda:
            # A) Área
            area_mean = latest_eda.get("bbox_area", {}).get("mean", 0)
            area_std = latest_eda.get("bbox_area", {}).get("std", 0)
            val_max = min(1.0, area_mean + area_std)
            val_min = max(0.0, area_mean - area_std)
            dataset_visuals["area_max_side"] = math.sqrt(val_max) * 100 if val_max > 0 else 0
            dataset_visuals["area_min_side"] = math.sqrt(val_min) * 100 if val_min > 0 else 0
            
            # B) Aspect Ratio
            ar_mean = latest_eda.get("aspect_ratio", {}).get("mean", 1.0)
            if ar_mean >= 1.0:
                dataset_visuals["ar_w"] = 50
                dataset_visuals["ar_h"] = 50 / ar_mean
            else:
                dataset_visuals["ar_h"] = 50
                dataset_visuals["ar_w"] = 50 * ar_mean
                
            # C) Diana Espacial
            dataset_visuals["gt_cx"] = latest_eda.get("center_x", {}).get("mean", 0.5) * 100
            dataset_visuals["gt_cy"] = latest_eda.get("center_y", {}).get("mean", 0.5) * 100
            dataset_visuals["gt_rx"] = latest_eda.get("center_x", {}).get("std", 0) * 100
            dataset_visuals["gt_ry"] = latest_eda.get("center_y", {}).get("std", 0) * 100

        summary["dataset_visuals"] = dataset_visuals
        summary["latest_eda"] = latest_eda
        return summary