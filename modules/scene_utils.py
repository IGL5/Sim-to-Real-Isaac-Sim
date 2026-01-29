import math
import random
import carb
from omni.physx import get_physx_scene_query_interface
from pxr import UsdGeom, Gf, Semantics, UsdShade, Sdf
import omni.replicator.core as rep
from modules import config

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


def run_orchestrator(simulation_app):
    rep.orchestrator.run()

    while not rep.orchestrator.get_is_started():
        simulation_app.update()
    while rep.orchestrator.get_is_started():
        simulation_app.update()

    rep.BackendDispatch.wait_until_done()
    rep.orchestrator.stop()


def get_ground_height(x, y):
    """
    Cast a ray from high up (z=500) downwards to find the ground.
    Returns the Z height of the hit point, or 0 if no hit.
    """
    origin = carb.Float3(x, y, config.RAYCAST_START_HEIGHT)
    direction = carb.Float3(0, 0, -1.0)
    distance = config.RAYCAST_DISTANCE
    
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
    distance = random.uniform(config.CAMERA_DISTANCE_RANGE[0], config.CAMERA_DISTANCE_RANGE[1]) 
    
    # Angular height (Pitch):
    elevation_deg = random.uniform(config.CAMERA_HEIGHT_RANGE[0], config.CAMERA_HEIGHT_RANGE[1])
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
    if config.DEBUG_WHEEL_CONTACT:
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


def update_prim_pose_and_visibility(stage, path, position, rotation, scale, visible=True):
    """
    Modifies the pose and visibility of a prim.
    """
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        return

    # Visibility
    imageable = UsdGeom.Imageable(prim)
    if visible:
        imageable.MakeVisible()

        # Pose
        xform = UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()
        
        # Add operations
        try:
            xform.AddTranslateOp().Set(Gf.Vec3d(float(position[0]), float(position[1]), float(position[2])))
            xform.AddRotateXYZOp().Set(Gf.Vec3f(float(rotation[0]), float(rotation[1]), float(rotation[2])))
            xform.AddScaleOp().Set(Gf.Vec3f(float(scale), float(scale), float(scale)))
        except Exception as e:
            print(f"[ERROR] Failed to set pose for {path}: {e}")
    else:
        imageable.MakeInvisible()
