import os
import json
from src.core.metadata.base_manager import BaseMetadataManager
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

    def _get_dir_size(self, path):
        """Calcula el tamaño total de un directorio en bytes de forma recursiva."""
        total = 0
        if os.path.exists(path):
            with os.scandir(path) as it:
                for entry in it:
                    if entry.is_file():
                        total += entry.stat().st_size
                    elif entry.is_dir():
                        total += self._get_dir_size(entry.path)
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