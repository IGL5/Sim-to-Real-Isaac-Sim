import os
import random
import math
import time
import carb
import traceback
import json
from datetime import datetime

# --- MODULES ---
from modules import config

# --- ISAAC SIMULATION APP ---
from isaacsim.simulation_app import SimulationApp
simulation_app = SimulationApp(launch_config=config.CONFIG)

# --- ISAAC / USD / REP IMPORTS ---
from isaacsim.core.utils.stage import get_current_stage, open_stage
from omni.timeline import get_timeline_interface
from pxr import UsdPhysics, Sdf, UsdGeom, Gf, UsdLux
import omni.replicator.core as rep

from modules import scene_utils
from modules import content

# Increase subframes if shadows/ghosting appears of moving objects
rep.settings.carb_settings("/omni/replicator/RTSubframes", 64)

def main():
    # --- 1. LOAD MAP & SKY ---
    found_hdrs = content.discover_hdr_maps(config.HDR_MAPS_DIR)
    
    if found_hdrs:
        config.AVAILABLE_HDRS = found_hdrs
    else:
        print("[CRITICAL] No HDR files found!")

    map_path = os.path.join(os.getcwd(), "assets", "map", config.MAP_NAME)
    open_stage(map_path)
    stage = get_current_stage()

    # Physics Scene
    if not stage.GetPrimAtPath("/PhysicsScene"):
        scene = UsdPhysics.Scene.Define(stage, Sdf.Path("/PhysicsScene"))
        scene.CreateGravityDirectionAttr().Set((0, 0, -1))
        scene.CreateGravityMagnitudeAttr().Set(9.81)

    # --- 2. LIGHTS SETUP ---
    print("--- Configuring Lights ---")

    # SkyDome
    dome_light_path = "/World/SkyDome"
    dome_prim = stage.GetPrimAtPath(dome_light_path)
    
    if dome_prim.IsValid():
        print("   -> Found SkyDome. Configuring...")
        dome_light = UsdLux.DomeLight(dome_prim)
        dome_light.CreateTextureFormatAttr().Set(UsdLux.Tokens.latlong)
        
        # Apply initial HDR texture
        if config.AVAILABLE_HDRS:
            light_path = os.path.join(config.HDR_MAPS_DIR, config.AVAILABLE_HDRS[0])
            dome_light.CreateTextureFileAttr().Set(Sdf.AssetPath(light_path))
            # High intensity to compete with the sun
            dome_light.CreateIntensityAttr().Set(2000.0)
    else:
        print("[WARN] SkyDome not found in USD. Background might be black.")

    # Sun
    sun_path = "/World/Sun_Light"
    sun_prim = stage.GetPrimAtPath(sun_path)
    
    if sun_prim.IsValid():
        print("   -> Found Sun_Light. Randomizing time of day...")
        sun_light = UsdLux.DistantLight(sun_prim)
        sun_light.CreateIntensityAttr().Set(2500.0)
        
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
    loaded_materials = content.load_pbr_materials(stage) 
    terrain_paths_map = scene_utils.find_prims_by_material_name(stage, config.ENVIRONMENT_LOOKUP_KEYS)
    content.setup_scene_materials_initial(stage, terrain_paths_map, loaded_materials)

    # --- 4. LOAD ASSETS ---
    print("\n--- Loading Detectable Objects ---")
    detectable_pools = content.create_class_pool(
        stage, 
        config.OBJECTS_CONFIG, 
        config.ASSETS_ROOT_DIR, 
        apply_semantics=True
    )
    
    print("\n--- Loading Distractors ---")
    distractor_pools = content.create_class_pool(
        stage, 
        config.DISTRACTOR_CONFIG, 
        config.ASSETS_ROOT_DIR, 
        apply_semantics=False
    )


    # --- 5. REPLICATOR CAMERA SETUP ---
    cam_rep = rep.create.camera(
        position=(0, 0, 0),
        rotation=(0, 0, 0),
        focal_length=18.0,
        name="DroneCamera"
    )

    camera_path = None
    for prim in stage.Traverse():
        if prim.GetName() == "DroneCamera":
            camera_path = str(prim.GetPath())
            break
    
    cam_prim = stage.GetPrimAtPath(camera_path)
    cam_xform = UsdGeom.Xformable(cam_prim)
    cam_xform.ClearXformOpOrder()
    cam_xform.AddTransformOp().Set(Gf.Matrix4d(1.0))
    
    # Writer
    writer = rep.WriterRegistry.get("KittiWriter")
    writer.initialize(output_dir=config.args.data_dir, omit_semantic_type=True)
    
    render_product = rep.create.render_product(cam_rep, (config.CONFIG["width"], config.CONFIG["height"]))
    writer.attach(render_product)

    # Run physics warmup
    rep.orchestrator.stop()
    for i in range(60):
        simulation_app.update()

    # --- 6. VALIDATION CHECK ---
    print("\n--- Configuration Safety Check ---")
    
    # Check Objects
    level_obj, msg_obj = scene_utils.validate_placement_config(
        config.OBJECTS_CONFIG, 
        config.OBJECTS_BUDGET_RANGE[1], # Use max of range
        config.OBJECTS_MAX_RADIUS, 
        "Detectables"
    )
    print(msg_obj)
    
    # Check Distractors
    level_dist, msg_dist = scene_utils.validate_placement_config(
        config.DISTRACTOR_CONFIG, 
        config.DISTRACTOR_BUDGET_RANGE[1], 
        config.DISTRACTOR_MAX_RADIUS, 
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
    print(f"\nStarting generation of {config.CONFIG['num_frames']} frames...")

    frames_generated = 0
    max_attempts = config.CONFIG["num_frames"] * 5 # Avoid infinite loops
    attempts = 0

    # Timer global
    total_start_time = time.time()
    frame_start_time = time.time()

    track_total_detectables = 0
    track_total_distractors = 0
    track_empty_target_frames = 0
    
    while frames_generated < config.CONFIG["num_frames"] and attempts < max_attempts:

        # Timer frame
        frame_start_time = time.time()

        attempts += 1
        print(f"\n--- ATTEMPTING FRAME {frames_generated} (Attempt {attempts}) ---")

        # A. APPLY MATERIAL & SKY RANDOMIZATION
        content.randomize_and_assign_new_materials(stage, terrain_paths_map, loaded_materials)

        # Randomize sky
        if config.AVAILABLE_HDRS and dome_prim.IsValid():
            hdr_name = random.choice(config.AVAILABLE_HDRS)
            light_path = os.path.join(config.HDR_MAPS_DIR, hdr_name)
            
            hdr_intensity = random.uniform(config.HDR_INTENSITY_RANGE[0] * 1000, 
                                         config.HDR_INTENSITY_RANGE[1] * 1000)
            
            content.setup_dome_light(stage, dome_light_path, light_path, hdr_intensity)

        # Randomize sun
        if sun_prim.IsValid():
            elevation = random.uniform(15, 80)
            azimuth = random.uniform(0, 360)
            
            xform = UsdGeom.Xformable(sun_prim)
            ops = xform.GetOrderedXformOps()
            if ops:
                ops[0].Set(Gf.Vec3f(0, elevation, azimuth))

        # B. CHOOSE TARGET
        tx = random.uniform(config.WORLD_LIMITS[0], config.WORLD_LIMITS[1])
        ty = random.uniform(config.WORLD_LIMITS[2], config.WORLD_LIMITS[3])
        tz = scene_utils.get_ground_height(tx, ty)
        
        # If it returns the error value (-9999) or is out of logical limits
        if tz == -9999.0 or tz < -200.0: 
            print(f"[RETRY] Invalid Ground at ({tx:.1f}, {ty:.1f}). Raycast failed: {tz}.")
            continue
            
        current_target = (tx, ty, tz)
        
        # C. POSITION CAMERA (Using Replicator LookAt)
        jitter_angle = random.uniform(0, 2 * math.pi)
        jitter_dist = random.uniform(0, config.LOOKAT_JITTER_RADIUS)
        jx = tx + jitter_dist * math.cos(jitter_angle)
        jy = ty + jitter_dist * math.sin(jitter_angle)
        camera_look_at_target = (jx, jy, tz)

        cam_x, cam_y, cam_z = scene_utils.get_drone_camera_pose(current_target)
        
        scene_utils.update_camera_pose(stage, camera_path, (cam_x, cam_y, cam_z), camera_look_at_target)

        # D. POSITION DETECTABLE OBJECTS (Cyclists, Vehicles...)
        detectables_obstacles = scene_utils.place_objects_from_config(
            stage=stage,
            target_pos=current_target,
            config_map=config.OBJECTS_CONFIG,
            pools_paths_map=detectable_pools,
            budget_range=config.OBJECTS_BUDGET_RANGE,
            max_radius=config.OBJECTS_MAX_RADIUS,
            previous_obstacles=[]
        )
        
        # E. POSITION DISTRACTORS (Rocks, Vegetation...)
        all_obstacles = detectables_obstacles 

        distractor_obstacles = scene_utils.place_objects_from_config(
            stage=stage,
            target_pos=current_target,
            config_map=config.DISTRACTOR_CONFIG,
            pools_paths_map=distractor_pools,
            budget_range=config.DISTRACTOR_BUDGET_RANGE,
            max_radius=config.DISTRACTOR_MAX_RADIUS,
            previous_obstacles=all_obstacles
        )

        num_detectables = len(detectables_obstacles)
        track_total_detectables += num_detectables
        track_total_distractors += len(distractor_obstacles)
        
        if num_detectables == 0:
            track_empty_target_frames += 1

        # F. SHOOT
        simulation_app.update()
        simulation_app.update()
        simulation_app.update()

        rep.orchestrator.step(delta_time=0.0, rt_subframes=64)
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
    elapsed_total_seconds = time.time() - total_start_time
    
    safe_frames = frames_generated if frames_generated > 0 else 1
    
    metadata = {
        "generation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "width": config.CONFIG["width"],
            "height": config.CONFIG["height"],
            "renderer": config.CONFIG["renderer"],
            "world_limits": config.WORLD_LIMITS
        },
        "performance": {
            "total_frames_requested": config.CONFIG["num_frames"],
            "total_frames_generated": frames_generated,
            "total_attempts": attempts,
            "generation_efficiency_percent": round((frames_generated / attempts) * 100, 2) if attempts > 0 else 0,
            "total_time_seconds": round(elapsed_total_seconds, 2),
            "average_time_per_frame_seconds": round(elapsed_total_seconds / safe_frames, 2)
        },
        "content_density": {
            "average_detectables_per_frame": round(track_total_detectables / safe_frames, 2),
            "average_distractors_per_frame": round(track_total_distractors / safe_frames, 2),
            "empty_target_frames": track_empty_target_frames,
            "total_detectables_spawned": track_total_detectables
        },
        "domain_randomization": {
            "hdr_maps_available": len(config.AVAILABLE_HDRS),
            "pbr_materials_loaded": len(loaded_materials)
        }
    }

    # Create the metadata file and save it
    metadata_path = os.path.join(config.args.data_dir, "generation_metadata.json")
    os.makedirs(config.args.data_dir, exist_ok=True)
    
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4)
        
    print(f"📊 Metadata exported successfully to: {metadata_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        carb.log_error(f"Exception: {e}")
        traceback.print_exc()
    finally:
        simulation_app.close()
