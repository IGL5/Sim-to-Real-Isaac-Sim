import os
import glob
import argparse
import random
import math


parser = argparse.ArgumentParser("Dataset generator")
parser.add_argument("--headless", type=bool, default=False, help="Launch script headless, default is False")
parser.add_argument("--height", type=int, default=544, help="Height of image")
parser.add_argument("--width", type=int, default=960, help="Width of image")
parser.add_argument("--num_frames", type=int, default=1, help="Number of frames to record")
parser.add_argument("--distractors", type=str, default="None",
                    help="Options are ")
parser.add_argument("--data_dir", type=str, default=os.getcwd() + "/_output_data",
                    help="Location where data will be output")

args, unknown_args = parser.parse_known_args()

# "renderer": "RayTracedLighting" is another option to consider
CONFIG = {"renderer": "PathTracing", "headless": args.headless,
          "width": args.width, "height": args.height, "num_frames": args.num_frames}


from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp(launch_config=CONFIG)


import carb
from isaacsim.core.utils.nucleus import get_assets_root_path
from isaacsim.core.utils.stage import get_current_stage, open_stage
from omni.timeline import get_timeline_interface
from pxr import UsdPhysics, PhysxSchema, Semantics, UsdShade, Sdf, UsdGeom, Gf, Usd
import omni.replicator.core as rep
from omni.physx import get_physx_scene_query_interface


# Increase subframes if shadows/ghosting appears of moving objects
rep.settings.carb_settings("/omni/replicator/RTSubframes", 128)

# CONSTANTS
WORLD_LIMITS = (-1300, 1300, -1300, 1300)
TEXTURES_ROOT_DIR = os.path.join(os.getcwd(), "assets", "textures")

# ASSET POOLS
ASSETS_ROOT_DIR = os.path.join(os.getcwd(), "assets", "objects")
OBJECTS_CONFIG = {
    "cyclist": {
        "count": 2,           # How many objects we want
        "wheelbase": 0.6,     # Physics: For incline calculation (None if not applicable)
        "scale": 1.0
    },
}

# --- CONFIGURATION: ENVIRONMENT TARGETS ---
ENVIRONMENT_LOOKUP_KEYS = [
    "Terrain",
    "Terrain_flat",
    # "Road",
    # "Path",
    # "Lake"
]

# --- DEBUGGING ---
DEBUG_WHEEL_CONTACT = False


def find_prims_by_material_name(stage, material_names):
    """
    Finds prims matching material names. 
    Ensures a prim is assigned to ONLY ONE group (the most specific one)
    to avoid Replicator conflicts.
    """
    found_paths = {name: [] for name in material_names}
    
    sorted_keys = sorted(material_names, key=len, reverse=True)
    
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh): 
            continue
            
        binding_api = UsdShade.MaterialBindingAPI(prim)
        if binding_api:
            direct_binding = binding_api.GetDirectBinding()
            material = direct_binding.GetMaterial()
            if material:
                mat_path = material.GetPath()
                mat_name = mat_path.name
                
                for target_name in sorted_keys:
                    if target_name in mat_name:
                        found_paths[target_name].append(str(prim.GetPath()))
                        break 
    
    return found_paths


def update_semantics(stage, keep_semantics=[]):
    """ Remove semantics from the stage except for keep_semantic classes"""
    for prim in stage.Traverse():
        if prim.HasAPI(Semantics.SemanticsAPI):
            processed_instances = set()
            for property in prim.GetProperties():
                is_semantic = Semantics.SemanticsAPI.IsSemanticsAPIPath(property.GetPath())
                if is_semantic:
                    instance_name = property.SplitName()[1]
                    if instance_name in processed_instances:
                        continue

                    processed_instances.add(instance_name)
                    sem = Semantics.SemanticsAPI.Get(prim, instance_name)
                    type_attr = sem.GetSemanticTypeAttr()
                    data_attr = sem.GetSemanticDataAttr()

                    for semantic_class in keep_semantics:
                        # Check for our data classes needed for the model
                        if data_attr.Get() == semantic_class:
                            continue
                        else:
                            # remove semantics of all other prims
                            prim.RemoveProperty(type_attr.GetName())
                            prim.RemoveProperty(data_attr.GetName())
                            prim.RemoveAPI(Semantics.SemanticsAPI, instance_name)


def prefix_with_isaac_asset_server(relative_path):
    assets_root_path = get_assets_root_path()
    if assets_root_path is None:
        raise Exception("Nucleus server not found, could not access Isaac Sim assets folder")
    return assets_root_path + relative_path


def run_orchestrator():
    rep.orchestrator.run()

    while not rep.orchestrator.get_is_started():
        simulation_app.update()
    while rep.orchestrator.get_is_started():
        simulation_app.update()

    rep.BackendDispatch.wait_until_done()
    rep.orchestrator.stop()


def discover_objective_assets(base_dir, obj_dir):
    """
    Scans automatically the assets folder to find objective models.
    Expected structure: base_dir/obj_dir/[model_name]/[model_name].usd
    """
    objective_root = os.path.join(base_dir, obj_dir)
    if not os.path.exists(objective_root):
         print(f"[WARN] Objective root directory not found: {objective_root}")
         return []

    print(f"--- Scanning for objective assets in: {objective_root} ---")
    discovered_paths = []
    
    try:
        folder_names = [d for d in os.listdir(objective_root) if os.path.isdir(os.path.join(objective_root, d))]
    except Exception as e:
        print(f"[ERROR] Could not list directories: {e}")
        return []

    for folder_name in folder_names:
        folder_path = os.path.join(objective_root, folder_name)
        
        # STRONG STRATEGY:
        expected_usd = os.path.join(folder_path, f"{folder_name}.usd")
        
        if os.path.exists(expected_usd):
            discovered_paths.append(expected_usd)
            print(f"   -> Found asset (exact match): {folder_name}")
        else:
            # If the exact file doesn't exist, we search for any .usd inside the folder.
            fallback_search = glob.glob(os.path.join(folder_path, "*.usd*"))
            if fallback_search:
                discovered_paths.append(fallback_search[0])
                print(f"   -> Found asset (fallback): {os.path.basename(fallback_search[0])} in {folder_name}")
            else:
                print(f"[WARN] Skipping folder '{folder_name}': No USD file found inside.")

    print(f"--- Total discovered objectives: {len(discovered_paths)} ({obj_dir}) ---")
    return discovered_paths


def update_camera_pose(camera_prim, eye, target):
    """
    Manually sets the camera pose using USD APIs.
    eye: (x, y, z) position of camera
    target: (x, y, z) point to look at
    """
    eye_gf = Gf.Vec3d(*eye)
    target_gf = Gf.Vec3d(*target)
    up_axis = Gf.Vec3d(0, 0, 1) # Z-up

    # Calculate View Matrix (World -> Camera)
    view_matrix = Gf.Matrix4d().SetLookAt(eye_gf, target_gf, up_axis)
    
    # Camera Transform is Inverse of View (Camera -> World)
    # We set this transform on the camera prim
    xform_api = UsdGeom.Xformable(camera_prim)
    
    xform_api.ClearXformOpOrder()
    xform_api.AddTransformOp().Set(view_matrix.GetInverse())


def load_pbr_materials(stage):
    """
    Loads PBR materials and manually assigns textures to avoid API errors.
    Returns a list of UsdShade.Material.
    """
    if not os.path.exists(TEXTURES_ROOT_DIR):
        print(f"[ERROR] Texture directory not found: {TEXTURES_ROOT_DIR}")
        return []

    material_folders = [f.path for f in os.scandir(TEXTURES_ROOT_DIR) if f.is_dir()]
    print(f"--- Found {len(material_folders)} texture folders ---")
    
    # 1. ROBUST CREATION
    for i, folder_path in enumerate(material_folders):
        try:
            files = glob.glob(os.path.join(folder_path, "*.*"))
            found_maps = {"diffuse": None, "normal": None, "roughness": None, "ao": None, "emission": None}
            normal_gl = None
            normal = None
            
            for f_path in files:
                name = os.path.basename(f_path).lower()
                if any(x in name for x in ["color", "diff", "alb"]): found_maps["diffuse"] = f_path
                elif "rough" in name: found_maps["roughness"] = f_path
                elif "norm" in name:
                    if "gl" in name:
                        normal_gl = f_path
                    else: normal = f_path
                elif "ao" in name: found_maps["ao"] = f_path
                elif any(x in name for x in ["emiss", "emit"]): found_maps["emission"] = f_path

            if not found_maps["diffuse"]: continue
            found_maps["normal"] = normal_gl if normal_gl else normal
            
            # Create material ONLY with diffuse texture
            rep_mat = rep.create.material_omnipbr(
                diffuse_texture=found_maps["diffuse"],
                roughness=0.5,
                specular=0.5,
                count=1
            )
            
            # Manually inject the rest of the maps using internal Shader names.
            with rep_mat:
                if found_maps["roughness"]:
                    rep.modify.attribute("inputs:reflectionroughness_texture", found_maps["roughness"])
                if found_maps["normal"]:
                    rep.modify.attribute("inputs:normalmap_texture", found_maps["normal"])
                if found_maps["ao"]:
                    rep.modify.attribute("inputs:ao_texture", found_maps["ao"])
                if found_maps["emission"]:
                    rep.modify.attribute("inputs:enable_emission", True)
                    rep.modify.attribute("inputs:emissive_color_texture", found_maps["emission"])
                    rep.modify.attribute("inputs:emissive_intensity", 50.0)

        except Exception as e:
            print(f"Error creating material from {folder_path}: {e}")
            
    # 2. COLLECTION
    loaded_materials_prims = []
    looks_scope = stage.GetPrimAtPath("/Replicator/Looks")
    
    if looks_scope and looks_scope.IsValid():
        for child in looks_scope.GetChildren():
            if child.IsA(UsdShade.Material):
                loaded_materials_prims.append(UsdShade.Material(child))
    
    print(f"--- Successfully loaded {len(loaded_materials_prims)} USD materials ---")
    return loaded_materials_prims

def setup_scene_materials_initial(stage, terrain_paths_map, loaded_materials):
    """
    Assigns random materials to meshes dynamically.
    Iterates through all detected keys in terrain_paths_map
    and assigns a random material to each group.
    """
    if not loaded_materials:
        print("[WARN] No textures loaded. Skipping initial assignment.")
        return

    print(f"--- Assigning initial materials for found groups ---")

    for key, paths in terrain_paths_map.items():
        if not paths:
            continue
            
        chosen_mat = random.choice(loaded_materials)
        print(f"   -> Group '{key}': Assigning material to {len(paths)} objects.")

        for path in paths:
            prim = stage.GetPrimAtPath(path)
            if prim.IsValid():
                UsdShade.MaterialBindingAPI(prim).Bind(chosen_mat)
            
    print("--- Initial materials assigned successfully ---")


def randomize_and_assign_new_materials(stage, terrain_paths_map, loaded_materials):
    """
    Choose UNIQUE materials for each terrain section in this frame.
    """
    if not loaded_materials:
        return

    # Adjusted scales
    scale_flat = (1.0, 2.0)      
    scale_mountain = (0.03, 0.05)

    # 1. FILTER ACTIVE GROUPS
    active_keys = [k for k, v in terrain_paths_map.items() if v]
    num_needed = len(active_keys)

    # 2. SELECTION OF MATERIALS WITHOUT REPETITION
    if len(loaded_materials) >= num_needed:
        selected_materials = random.sample(loaded_materials, k=num_needed)
    else:
        print(f"[WARN] Not enough unique materials ({len(loaded_materials)}) for groups ({num_needed}). Collisions may occur.")
        selected_materials = random.choices(loaded_materials, k=num_needed)

    # 3. ASSIGNMENT
    for key, chosen_material in zip(active_keys, selected_materials):
        paths = terrain_paths_map[key]

        shader = None
        for child in chosen_material.GetPrim().GetChildren():
            if child.IsA(UsdShade.Shader):
                shader = UsdShade.Shader(child)
                break
        
        if shader:
            # Scale logic based on terrain type
            if "flat" in key.lower():
                s_min, s_max = scale_flat
            else:
                s_min, s_max = scale_mountain

            scale_val = random.uniform(s_min, s_max)
            rot_val = random.uniform(0, 360)

            # color_val = Gf.Vec3f(
            #     random.uniform(0.4, 1.0), 
            #     random.uniform(0.4, 1.0), 
            #     random.uniform(0.4, 1.0)
            # )
            
            # Apply changes to the shader attributes
            shader.CreateInput("texture_scale", Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(scale_val, scale_val))
            shader.CreateInput("texture_rotate", Sdf.ValueTypeNames.Float).Set(rot_val)
            # shader.CreateInput("diffuse_tint", Sdf.ValueTypeNames.Color3f).Set(color_val)
            shader.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(random.uniform(0.4, 0.9))

            normal_strength = random.uniform(1.5, 2.5) 
            shader.CreateInput("bump_factor", Sdf.ValueTypeNames.Float).Set(normal_strength)

        # --- BIND ---
        for path in paths:
            prim = stage.GetPrimAtPath(path)
            if prim.IsValid():
                binding_api = UsdShade.MaterialBindingAPI(prim)
                binding_api.Bind(chosen_material)


def get_ground_height(x, y):
    """
    Cast a ray from high up (z=500) downwards to find the ground.
    Returns the Z height of the hit point, or 0 if no hit.
    """
    origin = carb.Float3(x, y, 500.0)
    direction = carb.Float3(0, 0, -1.0)
    distance = 1000.0 # Sufficient to cover the range
    
    hit = get_physx_scene_query_interface().raycast_closest(origin, direction, distance)
    
    if hit["hit"]:
        return hit["position"][2] # Return Z component
    return -9999.0

def get_drone_camera_pose(focus_target):
    """
    Calculates the position of the camera orbiting around a point of interest (focus_target).
    Simulates a drone flight.
    """
    tx, ty, tz = focus_target
    
    # --- FLIGHT PARAMETERS ---
    # Distance to target (hypotenuse)
    distance = random.uniform(5.0, 10.0) 
    
    # Angular height (Pitch):
    elevation_deg = random.uniform(30.0, 60.0)
    elevation_rad = math.radians(elevation_deg)
    
    # Angle around the target (Azimuth)
    azimuth_deg = random.uniform(0, 360)
    azimuth_rad = math.radians(azimuth_deg)
    
    # --- POSITION CALCULATION (SPHERICAL COORDINATES) ---
    cam_z = tz + distance * math.sin(elevation_rad)
    
    # Projection on the XY plane (how far we move horizontally)
    dist_xy = distance * math.cos(elevation_rad)
    
    cam_x = tx + dist_xy * math.cos(azimuth_rad)
    cam_y = ty + dist_xy * math.sin(azimuth_rad)
    
    # --- EXTRA SAFETY ---
    ground_under_cam = get_ground_height(cam_x, cam_y)
    
    if cam_z < ground_under_cam + 5.0:
        cam_z = ground_under_cam + 5.0

    return (cam_x, cam_y, cam_z)


def sample_assets_from_pool(asset_pool, num_samples, allow_duplicates=True):
    """
    Selects random assets from a list.
    """
    if not asset_pool:
        return []
    
    if allow_duplicates:
        return random.choices(asset_pool, k=num_samples)
    else:
        # Ensure we don't ask for more than available if duplicates are not allowed
        k = min(num_samples, len(asset_pool))
        return random.sample(asset_pool, k)


def calculate_pitch_on_terrain(stage, x, y, yaw_degrees, wheelbase):
    """
    Calculates the pitch and adjusted Z height so that an object
    rests correctly on the terrain.
    
    Args:
        x, y: Center coordinates of the object.
        yaw_degrees: Object orientation (where it faces).
        wheelbase: Distance between axles in METERS.
    
    Returns:
        (pitch_degrees, z_center_adjusted): Pitch angle and new Z height.
        Returns (None, None) if any raycast fails.
    """
    # 1. Convert direction angle (Yaw) to unit vector
    yaw_rad = math.radians(yaw_degrees)
    dir_x = math.cos(yaw_rad)
    dir_y = math.sin(yaw_rad)
    
    # 2. Calculate wheel positions (Front and Rear)
    # Assume (x,y) is the center, so displace half the wheelbase
    half_len = wheelbase / 2.0
    
    front_x = x + dir_x * half_len
    front_y = y + dir_y * half_len
    
    back_x = x - dir_x * half_len
    back_y = y - dir_y * half_len
    
    # 3. Raycast at each wheel
    z_front = get_ground_height(front_x, front_y)
    z_back = get_ground_height(back_x, back_y)
    
    # If any raycast returns the error value, abort
    if z_front == -9999.0 or z_back == -9999.0: 
        return None, None

    # --- VISUAL DEBUG ---
    if DEBUG_WHEEL_CONTACT:
        draw_debug_cube(stage, (front_x, front_y, z_front))
        draw_debug_cube(stage, (back_x, back_y, z_back))

    # 4. Calculate pitch angle
    delta_z = z_front - z_back
    pitch_rad = math.atan2(delta_z, wheelbase)
    pitch_deg = math.degrees(pitch_rad)
    
    # 5. Calculate center Z
    z_center_adjusted = (z_front + z_back) / 2.0
    
    return pitch_deg, z_center_adjusted


def draw_debug_cube(stage, position):
    """ 
    Draws a red cube at the given position.
    Generates its own random name to avoid depending on external logic.
    """
    rand_id = random.randint(0, 20)
    path = f"/World/Debug/DebugCube_{rand_id}"
    
    cube_geom = UsdGeom.Cube.Define(stage, path)
    cube_geom.ClearXformOpOrder() # Clear previous transformations to avoid errors
    
    cube_geom.AddTranslateOp().Set(Gf.Vec3d(*position))
    cube_geom.AddScaleOp().Set(Gf.Vec3d(0.1, 0.1, 0.1)) 
    cube_geom.GetDisplayColorAttr().Set([(1, 0, 0)])


def get_multiple_poses_near_target(stage, target_pos, num_objects, min_dist=2.0, max_radius=10.0, existing_obstacles=[], wheelbase= None):
    """
    Generates N valid positions with slope adaptation.
    Avoids collisions with 'existing_obstacles' and with itself.
    
    If wheelbase is not None, it will calculate the pitch angle based on the two wheels.
    """
    valid_poses = []
    tx, ty, tz = target_pos

    obstacles_xy = [(p[0], p[1]) for p in existing_obstacles]
    
    max_attempts = num_objects * 100 
    attempts = 0
    
    while len(valid_poses) < num_objects and attempts < max_attempts:
        attempts += 1
        
        # 1. Generate random candidate
        r = random.uniform(1.0, max_radius) 
        theta = random.uniform(0, 2 * math.pi)
        
        cand_x = tx + r * math.cos(theta)
        cand_y = ty + r * math.sin(theta)
        
        # 2. Check 2D collision
        collision = False
        # A) Check against previous obstacles
        for (ex, ey) in obstacles_xy:
            dist = math.sqrt((cand_x - ex)**2 + (cand_y - ey)**2)
            if dist < min_dist:
                collision = True
                break
        
        # B) Check against the ones we just generated in this same loop
        if not collision:
            for (exist_pos, _) in valid_poses:
                ex, ey, _ = exist_pos
                dist = math.sqrt((cand_x - ex)**2 + (cand_y - ey)**2)
                if dist < min_dist:
                    collision = True
                    break
        
        if not collision:
            angle_yaw = random.uniform(0, 360)
            
            # 3. DIFFERENT LOGIC: Is it a vehicle or static?
            if wheelbase is not None:
                # --- VEHICLE MODE ---
                # Calculate pitch based on the two wheels
                pitch_deg, z_adjusted = calculate_pitch_on_terrain(stage, cand_x, cand_y, angle_yaw, wheelbase)
                
                if pitch_deg is None: 
                    continue
                
                rotation_corrected = (90, -pitch_deg, angle_yaw)
                z_final = z_adjusted
                
            else:
                # --- STATIC MODE ---
                # Only need the ground height at the center
                z_ground = get_ground_height(cand_x, cand_y)
                
                if z_ground == -9999.0: 
                    continue
                
                rotation_corrected = (90, 0, angle_yaw)
                z_final = z_ground

            # 4. Save valid candidate
            valid_poses.append(((cand_x, cand_y, z_final), rotation_corrected))
            
    if len(valid_poses) < num_objects:
        print(f"[WARN] Could only place {len(valid_poses)}/{num_objects} objects safely.")
        
    return valid_poses


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
    loaded_materials = load_pbr_materials(stage) 
    terrain_paths_map = find_prims_by_material_name(stage, ENVIRONMENT_LOOKUP_KEYS)
    setup_scene_materials_initial(stage, terrain_paths_map, loaded_materials)

    # --- 3. LOAD OBJECTS ---
    assets_library = {}
    for obj_class in OBJECTS_CONFIG.keys():
        found_paths = discover_objective_assets(ASSETS_ROOT_DIR, obj_class)
        if found_paths:
            assets_library[obj_class] = found_paths
        else:
            print(f"[WARN] No assets found for class '{obj_class}'. It will be skipped.")
    
    scene_reps = {} 
    
    for obj_class, config in OBJECTS_CONFIG.items():
        if obj_class not in assets_library: continue
        
        # Select randomly which models to use
        paths_to_use = sample_assets_from_pool(assets_library[obj_class], config["count"], allow_duplicates=True)
        
        items_list = []
        for i, path in enumerate(paths_to_use):
            rep_item = rep.create.from_usd(
                path, 
                semantics=[('class', obj_class)],
                name=f"{obj_class}_{i}" 
            )
            items_list.append(rep_item)
            
        scene_reps[obj_class] = items_list
        print(f"--- Instantiated {len(items_list)} objects of class '{obj_class}' ---")

    # Run physics warmup
    for i in range(60):
        simulation_app.update()
    
    # --- 5. LIGHTS AND CAMERA (SETUP) ---
    
    # Ambient Light (Fill)
    rep.create.light(light_type="Dome", intensity=5, texture=None)

    # Sun (Main)
    distant_light = rep.create.light(
        light_type="Distant", 
        intensity=50, 
        rotation=(300, 0, 0)
    )
    with distant_light:
        rep.modify.attribute("inputs:angle", 0.5)
    
    # REPLICATOR CAMERA
    cam_rep = rep.create.camera(
        focal_length=18.0,
        name="DroneCam" 
    )
    
    # Writer
    writer = rep.WriterRegistry.get("KittiWriter")
    writer.initialize(output_dir=args.data_dir, omit_semantic_type=True)
    
    render_product = rep.create.render_product(cam_rep, (CONFIG["width"], CONFIG["height"]))
    writer.attach(render_product)

    # --- 6. MAIN LOOP ---
    print(f"Starting generation of {CONFIG['num_frames']} frames...")
    rep.orchestrator.stop()

    frames_generated = 0
    max_attempts = CONFIG["num_frames"] * 5 # Avoid infinite loops
    attempts = 0
    
    # --- Main Loop ---
    while frames_generated < CONFIG["num_frames"] and attempts < max_attempts:
        attempts += 1
        print(f"\n--- ATTEMPTING FRAME {frames_generated} (Attempt {attempts}) ---")

        # A. APPLY MATERIAL RANDOMIZATION
        randomize_and_assign_new_materials(stage, terrain_paths_map, loaded_materials)

        # B. CHOOSE TARGET
        tx = random.uniform(WORLD_LIMITS[0], WORLD_LIMITS[1])
        ty = random.uniform(WORLD_LIMITS[2], WORLD_LIMITS[3])
        tz = get_ground_height(tx, ty)
        
        # If it returns the error value (-9999) or is out of logical limits
        if tz == -9999.0 or tz < -200.0: 
            print(f"[RETRY] Invalid Ground at ({tx:.1f}, {ty:.1f}). Raycast failed: {tz}.")
            continue
            
        current_target = (tx, ty, tz)
        
        # C. POSITION CAMERA (Using Replicator LookAt)
        cam_x, cam_y, cam_z = get_drone_camera_pose(current_target)
        
        with cam_rep:
            rep.modify.pose(
                position=(cam_x, cam_y, cam_z),
                look_at=current_target
            )

        # D. POSITION OBJECTS
        all_poses_occupied = [] 
        frame_failed = False
        for obj_class, rep_items in scene_reps.items():
            if not rep_items: continue
            
            config = OBJECTS_CONFIG[obj_class]
            
            poses = get_multiple_poses_near_target(
                stage,
                target_pos=current_target,
                num_objects=len(rep_items),
                min_dist=0.5,
                max_radius=4.0,
                existing_obstacles=all_poses_occupied,
                wheelbase=config["wheelbase"]
            )

            if len(poses) < len(rep_items):
                 print(f"[RETRY] Could not place {obj_class} safely.")
                 frame_failed = True
                 break
            
            for rep_item, (pos, rot) in zip(rep_items, poses):
                with rep_item:
                    rep.modify.pose(
                        position=pos,
                        rotation=rot,
                        scale=config["scale"]
                    )
                
                all_poses_occupied.append(pos)
        
        if frame_failed:
            continue

        # E. SHOOT
        simulation_app.update()
        simulation_app.update()
        simulation_app.update()

        rep.orchestrator.step(delta_time=0.0, rt_subframes=128)
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
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()
