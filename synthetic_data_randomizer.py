import os
import random
import math
import carb
import traceback

# --- MODULES ---
from modules import config

# --- ISAAC SIMULATION APP ---
from isaacsim.simulation_app import SimulationApp
simulation_app = SimulationApp(launch_config=config.CONFIG)

# --- ISAAC / USD / REP IMPORTS ---
from isaacsim.core.utils.stage import get_current_stage, open_stage
from omni.timeline import get_timeline_interface
from pxr import UsdPhysics, Sdf
import omni.replicator.core as rep

from modules import scene_utils
from modules import content

# Increase subframes if shadows/ghosting appears of moving objects
rep.settings.carb_settings("/omni/replicator/RTSubframes", 64)

def main():
    # --- 1. LOAD MAP (Scaled and Centered) ---
    map_path = os.path.join(os.getcwd(), "map", "Environment_variable.usd")
    open_stage(map_path)
    stage = get_current_stage()

    # Physics Scene
    if not stage.GetPrimAtPath("/PhysicsScene"):
        scene = UsdPhysics.Scene.Define(stage, Sdf.Path("/PhysicsScene"))
        scene.CreateGravityDirectionAttr().Set((0, 0, -1))
        scene.CreateGravityMagnitudeAttr().Set(9.81)

    timeline = get_timeline_interface()
    timeline.play()

    # --- 2. MATERIAL SETUP ---
    loaded_materials = content.load_pbr_materials(stage) 
    terrain_paths_map = scene_utils.find_prims_by_material_name(stage, config.ENVIRONMENT_LOOKUP_KEYS)
    content.setup_scene_materials_initial(stage, terrain_paths_map, loaded_materials)

    # --- 3. LOAD ASSETS ---
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

    # Run physics warmup
    for i in range(100):
        simulation_app.update()
    
    # --- 4. LIGHTS AND CAMERA (SETUP) ---
    
    # Ambient Light (Fill)
    rep.create.light(
        light_type="Dome", 
        intensity=10, 
        texture=None
    )

    # Sun (Main)
    distant_light = rep.create.light(
        light_type="Distant", 
        intensity=50, 
        rotation=(300, 0, 0)
    )
    with distant_light:
        rep.modify.attribute("inputs:angle", 0.5)
    
    # REPLICATOR CAMERA
    cam_rep = rep.create.camera(focal_length=18.0, name="DroneCamera")
    
    # Writer
    writer = rep.WriterRegistry.get("KittiWriter")
    writer.initialize(output_dir=config.args.data_dir, omit_semantic_type=True)
    
    render_product = rep.create.render_product(cam_rep, (config.CONFIG["width"], config.CONFIG["height"]))
    writer.attach(render_product)

    # --- 5. VALIDATION CHECK ---
    print("\n--- Configuration Safety Check ---")
    
    # Check Objects
    ok_obj, msg_obj = scene_utils.validate_placement_config(
        config.OBJECTS_CONFIG, 
        config.OBJECTS_BUDGET_RANGE[1], # Use max of range
        config.OBJECTS_MAX_RADIUS, 
        "Detectables"
    )
    print(msg_obj)
    
    # Check Distractors
    ok_dist, msg_dist = scene_utils.validate_placement_config(
        config.DISTRACTOR_CONFIG, 
        config.DISTRACTOR_BUDGET_RANGE[1], 
        config.DISTRACTOR_MAX_RADIUS, 
        "Distractors"
    )
    print(msg_dist)
    
    if not ok_obj or not ok_dist:
        print("\n⚠️  WARNING: High density detected! Consider increasing MAX_RADIUS or decreasing BUDGET.")
        input("Press Enter to continue anyway...")

    # --- 6. MAIN LOOP ---
    print(f"Starting generation of {config.CONFIG['num_frames']} frames...")
    rep.orchestrator.stop()

    frames_generated = 0
    max_attempts = config.CONFIG["num_frames"] * 5 # Avoid infinite loops
    attempts = 0
    
    while frames_generated < config.CONFIG["num_frames"] and attempts < max_attempts:
        attempts += 1
        print(f"\n--- ATTEMPTING FRAME {frames_generated} (Attempt {attempts}) ---")

        # A. APPLY MATERIAL RANDOMIZATION
        content.randomize_and_assign_new_materials(stage, terrain_paths_map, loaded_materials)

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
        
        with cam_rep:
            rep.modify.pose(
                position=(cam_x, cam_y, cam_z),
                look_at=camera_look_at_target
            )

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

        scene_utils.place_objects_from_config(
            stage=stage,
            target_pos=current_target,
            config_map=config.DISTRACTOR_CONFIG,
            pools_paths_map=distractor_pools,
            budget_range=config.DISTRACTOR_BUDGET_RANGE,
            max_radius=config.DISTRACTOR_MAX_RADIUS,
            previous_obstacles=all_obstacles
        )

        # F. SHOOT
        simulation_app.update()
        simulation_app.update()
        simulation_app.update()

        rep.orchestrator.step(delta_time=0.0, rt_subframes=64)
        rep.BackendDispatch.wait_until_done()

        frames_generated += 1

    # Wait until writes are done
    print("Finalizing writes...")
    rep.BackendDispatch.wait_until_done()
    simulation_app.update()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        carb.log_error(f"Exception: {e}")
        traceback.print_exc()
    finally:
        simulation_app.close()
