from pathlib import Path
import random
import math
import time
import carb
import traceback
import shutil

# --- MODULES ---
import src.core.config as config
from src.simulation.utils import sim_config
from src.core.metadata.sim_builder import SimulationMetadata

# --- ISAAC SIMULATION APP ---
from isaacsim.simulation_app import SimulationApp
simulation_app = SimulationApp(launch_config=sim_config.CONFIG)

# --- ISAAC / USD / REP IMPORTS ---
from isaacsim.core.utils.stage import get_current_stage, open_stage
from omni.timeline import get_timeline_interface
from pxr import UsdPhysics, Sdf, UsdGeom, Gf, UsdLux
import omni.replicator.core as rep

from src.simulation.utils import scene_utils
from src.simulation.utils import asset_manager

# Increase subframes if shadows/ghosting appears of moving objects
rep.settings.carb_settings("/omni/replicator/RTSubframes", sim_config.RT_SUBFRAMES)

def main():
    data_dir_path = Path(sim_config.args.data_dir)
    if data_dir_path.exists():
        shutil.rmtree(str(data_dir_path))
    data_dir_path.mkdir(parents=True, exist_ok=True)
    
    # Create or overwrite classes.txt
    asset_manager.update_yolo_classes_txt()

    # --- 1. LOAD MAP & SKY ---
    found_hdrs = asset_manager.discover_hdr_maps(sim_config.HDR_MAPS_DIR)
    
    if found_hdrs:
        sim_config.AVAILABLE_HDRS = found_hdrs
    else:
        print("[CRITICAL] No HDR files found!")

    # Open scene
    open_stage(sim_config.MAP_PATH)
    stage = get_current_stage()

    # Physics Scene
    if not stage.GetPrimAtPath("/PhysicsScene"):
        scene = UsdPhysics.Scene.Define(stage, Sdf.Path("/PhysicsScene"))
        scene.CreateGravityDirectionAttr().Set((0, 0, -1))
        scene.CreateGravityMagnitudeAttr().Set(9.81)

    # --- 2. LIGHTS SETUP ---
    print("--- Configuring Lights ---")

    # SkyDome
    dome_prim = stage.GetPrimAtPath(sim_config.SKY_PATH)
    
    if dome_prim.IsValid():
        print("   -> Found SkyDome. Configuring...")
        dome_light = UsdLux.DomeLight(dome_prim)
        dome_light.CreateTextureFormatAttr().Set(UsdLux.Tokens.latlong)
        
        # Apply initial HDR texture
        if sim_config.AVAILABLE_HDRS:
            light_path = Path(sim_config.HDR_MAPS_DIR) / sim_config.AVAILABLE_HDRS[0]
            dome_light.CreateTextureFileAttr().Set(Sdf.AssetPath(str(light_path)))
            # High intensity to compete with the sun
            dome_light.CreateIntensityAttr().Set(600.0)
    else:
        print("[WARN] SkyDome not found in USD. Background might be black.")

    # Sun
    sun_prim = stage.GetPrimAtPath(sim_config.SUN_PATH)
    
    if sun_prim.IsValid():
        print("   -> Found Sun_Light. Randomizing time of day...")
        sun_light = UsdLux.DistantLight(sun_prim)
        sun_light.CreateIntensityAttr().Set(1100.0)
        
        # Random rotation (Simulate time of day and direction)
        xform = UsdGeom.Xformable(sun_prim)
        xform.ClearXformOpOrder()
        
        elevation = random.uniform(30, 85)
        azimuth = random.uniform(0, 360)
        
        # Apply rotation
        xform.AddRotateXYZOp().Set(Gf.Vec3f(0, elevation, azimuth))
    else:
        print("[WARN] Sun_Light not found manually. Shadows might be weak.")

    timeline = get_timeline_interface()
    timeline.play()

    # --- 3. MATERIAL SETUP ---
    loaded_materials = asset_manager.load_pbr_materials(stage) 
    terrain_paths_map = scene_utils.find_prims_by_material_name(stage, sim_config.ENVIRONMENT_LOOKUP_KEYS)
    asset_manager.setup_scene_materials_initial(stage, terrain_paths_map, loaded_materials)

    # --- 4. LOAD ASSETS ---
    print("\n--- Loading Detectable Objects ---")
    detectable_pools, n_distinct_assets_obj = asset_manager.create_class_pool(
        stage, 
        sim_config.OBJECTS_CONFIG, 
        sim_config.ASSETS_ROOT_DIR, 
        apply_semantics=True
    )
    
    print("\n--- Loading Distractors ---")
    distractor_pools, n_distinct_assets_distractor = asset_manager.create_class_pool(
        stage, 
        sim_config.DISTRACTOR_CONFIG, 
        sim_config.ASSETS_ROOT_DIR, 
        apply_semantics=False
    )


    # --- 5. REPLICATOR CAMERA SETUP ---
    cam_rep = rep.create.camera(
        position=(0, 0, 0),
        rotation=(0, 0, 0),
        focal_length=18.0,
        name=config.CAMERA_NAME
    )

    camera_path = None
    for prim in stage.Traverse():
        if prim.GetName() == config.CAMERA_NAME:
            camera_path = str(prim.GetPath())
            break
    
    cam_prim = stage.GetPrimAtPath(camera_path)
    cam_xform = UsdGeom.Xformable(cam_prim)
    cam_xform.ClearXformOpOrder()
    cam_xform.AddTransformOp().Set(Gf.Matrix4d(1.0))
    
    # Writer
    writer = rep.WriterRegistry.get("KittiWriter")
    writer.initialize(output_dir=sim_config.args.data_dir, omit_semantic_type=True)
    
    render_product = rep.create.render_product(cam_rep, (sim_config.CONFIG["width"], sim_config.CONFIG["height"]))
    writer.attach(render_product)

    # Run physics warmup
    rep.orchestrator.stop()
    for i in range(60):
        simulation_app.update()

    # --- 6. VALIDATION CHECK ---
    print("\n--- Configuration Safety Check ---")
    
    # Check Objects
    level_obj, msg_obj = scene_utils.validate_placement_config(
        sim_config.OBJECTS_CONFIG, 
        sim_config.OBJECTS_BUDGET_RANGE[1], # Use max of range
        sim_config.OBJECTS_MAX_RADIUS, 
        "Detectables"
    )
    print(msg_obj)
    
    # Check Distractors
    level_dist, msg_dist = scene_utils.validate_placement_config(
        sim_config.DISTRACTOR_CONFIG, 
        sim_config.DISTRACTOR_BUDGET_RANGE[1], 
        sim_config.DISTRACTOR_MAX_RADIUS, 
        "Distractors"
    )
    print(msg_dist)
    
    if level_obj == "red" or level_dist == "red":
        print("\n🛑 CRITICAL ERROR: Impossible density detected. Increase MAX_RADIUS or decrease BUDGET.")
        input("ℹ️ Press Enter to continue anyway or Ctrl+C to cancel...")
    elif level_obj == "orange" or level_dist == "orange":
        print("\n⚠️  WARNING: High density detected! Consider increasing MAX_RADIUS or decreasing BUDGET.")
        print("⏳ Continuing in 10 seconds... (Press Ctrl+C to cancel)")
        time.sleep(10)

    # --- 7. MAIN LOOP ---
    print(f"\nStarting generation of {sim_config.CONFIG['num_frames']} frames...")

    frames_generated = 0
    max_attempts = sim_config.CONFIG["num_frames"] * 5 # Avoid infinite loops
    attempts = 0

    # Timer global
    total_start_time = time.time()
    frame_start_time = time.time()

    track_total_detectables = 0
    track_total_distractors = 0
    track_empty_target_frames = 0
    
    while frames_generated < sim_config.CONFIG["num_frames"] and attempts < max_attempts:

        # Timer frame
        frame_start_time = time.time()

        attempts += 1
        print(f"\n--- ATTEMPTING FRAME {frames_generated} (Attempt {attempts}) ---")

        # A. APPLY MATERIAL & SKY RANDOMIZATION
        asset_manager.randomize_and_assign_new_materials(stage, terrain_paths_map, loaded_materials)

        # Randomize sky
        if sim_config.AVAILABLE_HDRS and dome_prim.IsValid() and sim_config.RANDOMIZE_SKY:
            hdr_name = random.choice(sim_config.AVAILABLE_HDRS)
            light_path = Path(sim_config.HDR_MAPS_DIR) / hdr_name
            
            hdr_intensity = random.uniform(sim_config.HDR_INTENSITY_RANGE[0] * 1000, 
                                        sim_config.HDR_INTENSITY_RANGE[1] * 1000)
        
            asset_manager.setup_dome_light(stage, sim_config.SKY_PATH, light_path, hdr_intensity)

        # Randomize sun
        if sim_config.RANDOMIZE_SKY and sun_prim.IsValid():
            elevation = random.uniform(15, 80)
            azimuth = random.uniform(0, 360)
            
            xform = UsdGeom.Xformable(sun_prim)
            ops = xform.GetOrderedXformOps()
            if ops:
                ops[0].Set(Gf.Vec3f(0, elevation, azimuth))

        # B. CHOOSE TARGET
        tx = random.uniform(sim_config.WORLD_LIMITS[0], sim_config.WORLD_LIMITS[1])
        ty = random.uniform(sim_config.WORLD_LIMITS[2], sim_config.WORLD_LIMITS[3])
        tz = scene_utils.get_ground_height(tx, ty)
        
        # If it returns the error value (-9999) or is out of logical limits
        if tz == -9999.0 or tz < -200.0: 
            print(f"[RETRY] Invalid Ground at ({tx:.1f}, {ty:.1f}). Raycast failed: {tz}.")
            continue
            
        current_target = (tx, ty, tz)
        
        # C. POSITION CAMERA (Using Replicator LookAt)
        jitter_angle = random.uniform(0, 2 * math.pi)
        jitter_dist = random.uniform(0, sim_config.LOOKAT_JITTER_RADIUS)
        jx = tx + jitter_dist * math.cos(jitter_angle)
        jy = ty + jitter_dist * math.sin(jitter_angle)
        camera_look_at_target = (jx, jy, tz)

        cam_x, cam_y, cam_z = scene_utils.get_drone_camera_pose(current_target)
        
        scene_utils.update_camera_pose(stage, camera_path, (cam_x, cam_y, cam_z), camera_look_at_target)

        # D. POSITION DETECTABLE OBJECTS (Cyclists, Vehicles...)
        detectables_obstacles = scene_utils.place_objects_from_config(
            stage=stage,
            target_pos=current_target,
            config_map=sim_config.OBJECTS_CONFIG,
            pools_paths_map=detectable_pools,
            budget_range=sim_config.OBJECTS_BUDGET_RANGE,
            max_radius=sim_config.OBJECTS_MAX_RADIUS,
            previous_obstacles=[]
        )
        
        # E. POSITION DISTRACTORS (Rocks, Vegetation...)
        all_obstacles = detectables_obstacles 

        distractor_obstacles = scene_utils.place_objects_from_config(
            stage=stage,
            target_pos=current_target,
            config_map=sim_config.DISTRACTOR_CONFIG,
            pools_paths_map=distractor_pools,
            budget_range=sim_config.DISTRACTOR_BUDGET_RANGE,
            max_radius=sim_config.DISTRACTOR_MAX_RADIUS,
            previous_obstacles=all_obstacles
        )

        num_detectables = len(detectables_obstacles)
        track_total_detectables += num_detectables
        track_total_distractors += len(distractor_obstacles)
        
        if num_detectables == 0:
            track_empty_target_frames += 1

        # F. SHOOT
        rep.orchestrator.step(delta_time=0.0, rt_subframes=sim_config.RT_SUBFRAMES)
        rep.BackendDispatch.wait_until_done()

        # Calculate frame duration
        frame_duration = time.time() - frame_start_time
        
        # Intelligent logging (Frame 1 and every 50)
        if frames_generated == 1 or (frames_generated + 1) % 50 == 0:
            elapsed_total = time.time() - total_start_time
            avg_time = elapsed_total / (frames_generated + 1)
            print(f"⏱️  [Frame {frames_generated}] Duration: {frame_duration:.2f}s | Avg: {avg_time:.2f}s | Total: {elapsed_total/60:.1f}min")

        frames_generated += 1

    # Wait until writes are done
    print("Finalizing writes...")
    rep.BackendDispatch.wait_until_done()
    simulation_app.update()

    # --- 8. METADATA EXPORTATION ---
    meta_manager = SimulationMetadata(config.GENERATION_METADATA_PATH)
    
    meta_manager.set_timestamp(key_name=config.GENERATION_TIMESTAMP_KEY)

    meta_manager.record_performance(
        requested_frames=sim_config.CONFIG["num_frames"],
        generated_frames=frames_generated,
        attempts=attempts,
        start_time_secs=total_start_time
    )

    meta_manager.record_content_density(
        total_detectables=track_total_detectables,
        total_distractors=track_total_distractors,
        empty_frames=track_empty_target_frames,
        generated_frames=frames_generated
    )

    meta_manager.record_domain_randomization(
        sky_active=sim_config.RANDOMIZE_SKY,
        terrain_active=sim_config.RANDOMIZE_TERRAIN,
        hdr_range=sim_config.HDR_INTENSITY_RANGE,
        hdrs_avail=len(sim_config.AVAILABLE_HDRS),
        mats_loaded=len(loaded_materials),
        obj_configs=sim_config.OBJECTS_CONFIG,
        dist_configs=sim_config.DISTRACTOR_CONFIG,
        distinct_objs=n_distinct_assets_obj,
        distinct_dists=n_distinct_assets_distractor
    )

    meta_manager.record_spatial_coverage(
        cam_dist=sim_config.CAMERA_DISTANCE_RANGE,
        cam_height=sim_config.CAMERA_HEIGHT_RANGE,
        obj_max_rad=sim_config.OBJECTS_MAX_RADIUS,
        dist_max_rad=sim_config.DISTRACTOR_MAX_RADIUS
    )
    
    meta_manager.build_theoretical_distribution(sim_config)
    
    meta_manager.commit()
    print(f"📊 Metadata exported successfully to: {config.GENERATION_METADATA_PATH}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        carb.log_error(f"Exception: {e}")
        traceback.print_exc()
    finally:
        simulation_app.close()
