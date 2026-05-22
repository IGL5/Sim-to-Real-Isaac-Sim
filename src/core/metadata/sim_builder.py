import time
from src.core.metadata.base_manager import BaseMetadataManager


class SimulationMetadata(BaseMetadataManager):
    def __init__(self, filepath):
        super().__init__(filepath)

    def record_config(self, width, height, renderer, world_limits, rt_subframes):
        self.update_section("config", {
            "width": width,
            "height": height,
            "renderer": renderer,
            "world_limits": world_limits,
            "rt_subframes": rt_subframes
        })

    def record_performance(self, requested_frames, generated_frames, attempts, start_time_secs):
        """ 
        The Builder calculates the elapsed time and efficiencies.
        The simulator only passes 'when it started' and 'how many frames it made'.
        """
        total_time_secs = time.time() - start_time_secs
        safe_frames = max(1, generated_frames)
        efficiency = round((generated_frames / attempts) * 100, 2) if attempts > 0 else 0
        avg_time = round(total_time_secs / safe_frames, 2)

        self.update_section("performance", {
            "total_frames_requested": requested_frames,
            "total_frames_generated": generated_frames,
            "total_attempts": attempts,
            "generation_efficiency_percent": efficiency,
            "total_time_seconds": round(total_time_secs, 2),
            "average_time_per_frame_seconds": avg_time
        })

    def record_content_density(self, total_detectables, total_distractors, empty_frames, generated_frames):
        safe_frames = max(1, generated_frames)
        
        self.update_section("content_density", {
            "average_detectables_per_frame": round(total_detectables / safe_frames, 2),
            "average_distractors_per_frame": round(total_distractors / safe_frames, 2),
            "empty_target_frames": empty_frames,
            "total_detectables_spawned": total_detectables
        })

    def record_domain_randomization(self, sky_active, terrain_active, hdr_range, hdrs_avail, mats_loaded, obj_configs, dist_configs, distinct_objs, distinct_dists):
        """
        The simulator passes the 'raw' lists (with duplicates). 
        The Builder takes care of cleaning them and extracting the unique values.
        """

        obj_mat_randomized = []
        for k, v in obj_configs.items():
            if v.get('active', True) and v.get('randomize_materials'):
                obj_mat_randomized.extend(v['randomize_materials'])

        dist_mat_randomized = []
        for k, v in dist_configs.items():
            if v.get('active', True) and v.get('randomize_materials'):
                dist_mat_randomized.extend(v['randomize_materials'])

        self.update_section("domain_randomization", {
            "sky_active": sky_active,
            "terrain_active": terrain_active,
            "hdr_intensity_range": hdr_range,
            "hdr_maps_available": hdrs_avail,
            "pbr_materials_loaded": mats_loaded,
            "object_materials_randomized": list(set(obj_mat_randomized)),
            "distractor_materials_randomized": list(set(dist_mat_randomized)),
            "distinct_assets_used": {
                "objects": distinct_objs,
                "distractors": distinct_dists
            }
        })

    def record_spatial_coverage(self, cam_dist, cam_height, obj_max_rad, dist_max_rad):
        self.update_section("spatial_coverage", {
            "camera_distance_range": cam_dist,
            "camera_height_range": cam_height,
            "objects_max_radius": obj_max_rad,
            "distractor_max_radius": dist_max_rad
        })

    def build_theoretical_distribution(self, sim_config):
        """
        Calculates the theoretical distribution based on the Expected Value of a weighted sampling process.
        """
        dist_dict = {}
        
        # --- 1. Detectables ---
        obj_budget_max = sim_config.OBJECTS_BUDGET_RANGE[1]
        active_objs = {k: v for k, v in sim_config.OBJECTS_CONFIG.items() if v.get('active', True)}
        total_obj_weight = sum(v.get('selection_weight', 1) for v in active_objs.values())
        
        if total_obj_weight > 0:
            # A) Expected average cost per draw
            expected_cost_per_draw = sum(
                (v.get('selection_weight', 1) / total_obj_weight) * v.get('cost_units', 1.0) 
                for v in active_objs.values()
            )
            # B) How many objects we expect to generate in total before the budget runs out
            expected_total_draws = obj_budget_max / expected_cost_per_draw if expected_cost_per_draw > 0 else 0
            
            # C) Distribute that total according to the pure probability of each one
            for k, v in active_objs.items():
                prob = v.get('selection_weight', 1) / total_obj_weight
                expected_count = expected_total_draws * prob
                dist_dict[k] = {"type": "detectable", "expected_max_count": expected_count}

        # --- 2. Distractors ---
        dist_budget_max = sim_config.DISTRACTOR_BUDGET_RANGE[1]
        active_dists = {k: v for k, v in sim_config.DISTRACTOR_CONFIG.items() if v.get('active', True)}
        total_dist_weight = sum(v.get('selection_weight', 1) for v in active_dists.values())
        
        if total_dist_weight > 0:
            expected_cost_per_draw_dist = sum(
                (v.get('selection_weight', 1) / total_dist_weight) * v.get('cost_units', 1.0) 
                for v in active_dists.values()
            )
            expected_total_draws_dist = dist_budget_max / expected_cost_per_draw_dist if expected_cost_per_draw_dist > 0 else 0
            
            for k, v in active_dists.items():
                prob = v.get('selection_weight', 1) / total_dist_weight
                expected_count = expected_total_draws_dist * prob
                dist_dict[k] = {"type": "distractor", "expected_max_count": expected_count}
                
        # --- 3. Normalize to percentages ---
        total_expected = sum(item["expected_max_count"] for item in dist_dict.values())
        for k in dist_dict.keys():
            dist_dict[k]["percentage"] = round((dist_dict[k]["expected_max_count"] / total_expected) * 100, 2) if total_expected > 0 else 0
            dist_dict[k]["expected_max_count"] = round(dist_dict[k]["expected_max_count"], 1)

        self.update_section("theoretical_distribution", dist_dict)

    # --- THE GETTER FOR THE VIEW (HTML) ---
    @staticmethod
    def get_html_summary_from_session(session_data):
        """
        Recibe el bloque en crudo de la sesión y devuelve los visuales 
        y datos aplanados específicos de la simulación 3D.
        """
        sim_summary = {}
        coverage = session_data.get("spatial_coverage", {})
        
        if coverage:
            cam_max = coverage.get("camera_distance_range", [0, 0])[1]
            dist_max = coverage.get("distractor_max_radius", 0)
            obj_max = coverage.get("objects_max_radius", 0)
            
            abs_max = max(cam_max, dist_max, obj_max, 1.0)
            
            sim_summary["cov_cam"] = (cam_max / abs_max) * 50
            sim_summary["cov_dist"] = (dist_max / abs_max) * 50
            sim_summary["cov_obj"] = (obj_max / abs_max) * 50
            
        return sim_summary