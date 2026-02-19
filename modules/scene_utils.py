import math
import random
import carb
from omni.physx import get_physx_scene_query_interface
from pxr import UsdGeom, Gf, Semantics, UsdShade, Sdf
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


def select_objects_by_budget(available_keys, config_map, total_budget):
    """
    Selects objects until the budget of 'units' is filled.
    Returns a list of keys (e.g: ['truck', 'rock', 'rock', 'sign'])
    """
    selected_keys = []
    current_usage = 0
    
    # Avoid infinite loops if nothing fits
    active_costs = [config_map[k]['cost_units'] for k in available_keys]
    if not active_costs: return []
    min_cost = min(active_costs)
    
    attempts = 0
    while (total_budget - current_usage) >= min_cost and attempts < 100:
        attempts += 1
        
        # Weighted selection
        keys = list(available_keys)
        weights = [config_map[k]['selection_weight'] for k in keys]
        
        choice = random.choices(keys, weights=weights, k=1)[0]
        cost = config_map[choice]['cost_units']
        
        if current_usage + cost <= total_budget:
            selected_keys.append(choice)
            current_usage += cost
            
    return selected_keys


def get_smart_poses_near_target(stage, target_pos, candidates_specs, max_radius=10.0, existing_obstacles=[]):
    """
    Generates non-overlapping positions and rotations for a set of objects around a target point.
    Adjusts height and pitch based on terrain geometry.
    """
    valid_results = []
    tx, ty, _ = target_pos
    
    # Local copy of obstacles (x, y, radius)
    current_obstacles = [(p[0], p[1], p[2]) for p in existing_obstacles] 

    for i, spec in enumerate(candidates_specs):
        obj_radius = spec['radius']
        obj_wheelbase = spec.get('wheelbase', None)
        
        placed = False
        attempts = 0
        max_attempts = 100 # Attempts per object
        
        while not placed and attempts < max_attempts:
            attempts += 1
            
            # 1. Generate candidate
            r = random.uniform(0.0, max_radius) # Can be 0 to be very close
            theta = random.uniform(0, 2 * math.pi)
            
            cand_x = tx + r * math.cos(theta)
            cand_y = ty + r * math.sin(theta)
            
            # 2. Collision check (Variable Radius)
            collision = False
            for (ex, ey, er) in current_obstacles:
                dist = math.sqrt((cand_x - ex)**2 + (cand_y - ey)**2)
                # The minimum distance is the sum of the two radius
                min_separation = obj_radius + er
                
                if dist < min_separation:
                    collision = True
                    break
            
            if not collision:
                # 3. Calculate Height / Pitch
                angle_yaw = random.uniform(0, 360)
                
                if obj_wheelbase is not None:
                    # Vehicle mode
                    pitch_deg, z_adjusted = calculate_pitch_on_terrain(stage, cand_x, cand_y, angle_yaw, obj_wheelbase)
                    if pitch_deg is None: continue
                    rotation = (90, -pitch_deg, angle_yaw)
                    z_final = z_adjusted
                else:
                    # Static mode
                    z_ground = get_ground_height(cand_x, cand_y)
                    if z_ground == -9999.0: continue
                    rotation = (90, 0, angle_yaw)
                    z_final = z_ground
                
                # Save valid candidate
                valid_results.append( ((cand_x, cand_y, z_final), rotation, i) )
                current_obstacles.append( (cand_x, cand_y, obj_radius) )
                placed = True
                
        if not placed:
            print(f"[WARN] Could not place object with radius {obj_radius}")

    return valid_results


def place_objects_from_config(stage, target_pos, config_map, pools_paths_map, budget_range, max_radius, previous_obstacles=[]):
    """
    Master Orchestrator.
    Handles: Selection (Budget) -> Unique Assignment (Stack) -> Placement -> Cleanup.
    """
    # 1. Budget and Theoretical Selection
    budget = random.uniform(budget_range[0], budget_range[1])
    active_keys = [k for k, v in config_map.items() if v.get('active', True)]
    
    # List of desired objects (e.g: ['rock', 'rock', 'tree', ...])
    chosen_keys = select_objects_by_budget(active_keys, config_map, budget)
    
    # 2. Sort by radius (Biggest first)
    chosen_keys.sort(key=lambda k: config_map[k]['radius'], reverse=True)
    
    # Sampling without replacement
    working_pools = {}
    for key, paths in pools_paths_map.items():
        if paths:
            pool_copy = list(paths)
            random.shuffle(pool_copy)
            working_pools[key] = pool_copy

    # 3. Resource Assignment
    candidates_specs = []
    paths_candidates = []
    keys_candidates = []
    
    for key in chosen_keys:
        # Skip if no stock or pool is empty (Dynamic pruning)
        if not working_pools.get(key):
            continue
            
        path = working_pools[key].pop()
        cfg = config_map[key]
        
        candidates_specs.append({'radius': cfg['radius'], 'wheelbase': cfg.get('wheelbase')})
        paths_candidates.append(path)
        keys_candidates.append(key)
        
    # 4. Calculate Poses (Mathematics)
    # results devuelve: [ ((x,y,z), rot, original_index), ... ]
    results = get_smart_poses_near_target(stage, target_pos, candidates_specs, max_radius, previous_obstacles)
    
    # 5. Move successfully placed objects
    new_obstacles = [] 
    successfully_placed_paths = set()

    for (pos, rot, original_idx) in results:
        path = paths_candidates[original_idx]
        
        key = keys_candidates[original_idx]
        cfg = config_map[key]
        
        s_min, s_max = cfg.get('scale_range', (1.0, 1.0))
        scale = random.uniform(s_min, s_max)
        
        update_prim_pose_and_visibility(stage, path, pos, rot, scale, visible=True)
        
        # Register success
        new_obstacles.append( (pos[0], pos[1], cfg['radius']) )
        successfully_placed_paths.add(path)
        
    # 6. Clean up unused objects
    for key, pool in pools_paths_map.items():
        for path in pool:
            if path not in successfully_placed_paths:
                update_prim_pose_and_visibility(stage, path, None, None, None, visible=False)

    return new_obstacles


def update_prim_pose_and_visibility(stage, path, position, rotation, scale, visible=True):
    """
    Modifies the pose and visibility of a prim.
    """
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        return

    # Visibility
    imageable = UsdGeom.Imageable(prim)

    # Optimized visibility update
    current_vis = imageable.GetVisibilityAttr().Get()
    target_vis = UsdGeom.Tokens.inherited if visible else UsdGeom.Tokens.invisible
    if current_vis != target_vis:
        if visible:
            imageable.MakeVisible()
        else:
            imageable.MakeInvisible()

    if visible:
        xform = UsdGeom.Xformable(prim)
        ops = xform.GetOrderedXformOps()
        
        # Add operations
        if len(ops) >= 3:
            # Op 0: Translate
            ops[0].Set(Gf.Vec3d(float(position[0]), float(position[1]), float(position[2])))
            # Op 1: Rotate
            ops[1].Set(Gf.Vec3f(float(rotation[0]), float(rotation[1]), float(rotation[2])))
            # Op 2: Scale
            ops[2].Set(Gf.Vec3f(float(scale), float(scale), float(scale)))
        else:
            # Fallback
            xform.ClearXformOpOrder()
            xform.AddTranslateOp().Set(Gf.Vec3d(float(position[0]), float(position[1]), float(position[2])))
            xform.AddRotateXYZOp().Set(Gf.Vec3f(float(rotation[0]), float(rotation[1]), float(rotation[2])))
            xform.AddScaleOp().Set(Gf.Vec3f(float(scale), float(scale), float(scale)))


def validate_placement_config(config_map, budget_max, container_radius, context_name="Config"):
    """
    Analizes if objects fit in the assigned space based on the maximum budget
    and selection probabilities.
    
    Returns:
        (is_safe, message): Boolean and string with the diagnosis.
    """
    total_weight = 0
    weighted_area_sum = 0
    weighted_cost_sum = 0
    
    # Only consider active items
    active_items = {k: v for k, v in config_map.items() if v.get('active', True)}
    
    if not active_items:
        return True, f"[{context_name}] No active items."

    # 1. Calculate weighted averages
    for k, v in active_items.items():
        weight = v.get('selection_weight', 1)
        radius = v.get('radius', 1.0)
        cost = v.get('cost_units', 1.0)
        
        area = math.pi * (radius ** 2)
        
        weighted_area_sum += weight * area
        weighted_cost_sum += weight * cost
        total_weight += weight
        
    if total_weight == 0:
        return True, f"[{context_name}] Total weights are 0."

    avg_area_per_item = weighted_area_sum / total_weight
    avg_cost_per_item = weighted_cost_sum / total_weight
    
    # 2. Estimation of worst case (Maximum budget)
    estimated_num_items = budget_max / avg_cost_per_item if avg_cost_per_item > 0 else 0
    required_area = estimated_num_items * avg_area_per_item
    
    # 3. Available area
    available_area = math.pi * (container_radius ** 2)
    
    # 4. Packing Factor
    # Perfect circles fill ~90%. Random placement is ~40-50%.
    # If we exceed 60%, we will start having many failures.
    fill_ratio = required_area / available_area if available_area > 0 else 999.0
    
    packing_limit_safe = 0.45  # Green: Very safe
    packing_limit_warn = 0.65  # Yellow: Possible failures, but acceptable
    
    msg = (f"[{context_name}] Ratio of Occupation: {fill_ratio*100:.1f}% "
           f"(Req: {required_area:.0f}m² / Disp: {available_area:.0f}m²)")
    
    if fill_ratio > 1.0:
        return False, f"🔴 {msg} -> IMPOSSIBLE (Overload > 100%)"
    elif fill_ratio > packing_limit_warn:
        return False, f"🟠 {msg} -> CRITICAL (High probability of failure)"
    elif fill_ratio > packing_limit_safe:
        return True, f"🟡 {msg} -> DENSE (May have some warnings)"
    else:
        return True, f"🟢 {msg} -> OK"


def update_camera_pose(stage, cam_path, eye, target):
    """
    Moves the camera manually using USD, avoiding that the Replicator graph grows.
    Calculates the transformation matrix for 'LookAt'.
    """
    prim = stage.GetPrimAtPath(cam_path)
    if not prim.IsValid(): return
    
    # 1. Vectors
    eye_vec = Gf.Vec3d(*eye)
    target_vec = Gf.Vec3d(*target)
    up_vec = Gf.Vec3d(0, 0, 1)
    
    # 2. Calculate View Matrix (World -> Camera)
    view_mtx = Gf.Matrix4d().SetLookAt(eye_vec, target_vec, up_vec)
    
    # 3. Invert for getting Transform (Camera -> World) that we move
    xform_mtx = view_mtx.GetInverse()
    
    # 4. Apply to the camera
    xform = UsdGeom.Xformable(prim)
    
    # Search if it already has a Transformation operation
    ops = xform.GetOrderedXformOps()
    found_op = None
    for op in ops:
        if op.GetOpType() == UsdGeom.XformOp.TypeTransform:
            found_op = op
            break
            
    if found_op:
        found_op.Set(xform_mtx)
    else:
        # If it's the first time, clear and create
        xform.ClearXformOpOrder()
        xform.AddTransformOp().Set(xform_mtx)