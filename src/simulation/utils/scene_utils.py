import math
import random
import carb
from omni.physx import get_physx_scene_query_interface
from pxr import UsdGeom, Gf, UsdShade, Sdf
from src.simulation.utils import sim_config

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
    origin = carb.Float3(x, y, sim_config.RAYCAST_START_HEIGHT)
    direction = carb.Float3(0, 0, -1.0)
    distance = sim_config.RAYCAST_DISTANCE
    
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
    distance = random.uniform(sim_config.CAMERA_3D_DIST_RANGE[0], sim_config.CAMERA_3D_DIST_RANGE[1]) 
    
    # Angular height (Pitch):
    elevation_deg = random.uniform(sim_config.CAMERA_ELEVATION_DEG_RANGE[0], sim_config.CAMERA_ELEVATION_DEG_RANGE[1])
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
    
    if cam_z < ground_under_cam + 1.0:
        cam_z = ground_under_cam + 1.0

    return (cam_x, cam_y, cam_z)


def calculate_vehicle_orientation_on_terrain(stage, x, y, yaw_degrees, wheelbase, track_width=None):
    """
    Calculates Pitch (camber), Roll (lateral tilt) and Z height.
    Works for 2 wheels (bikes) and 4 wheels (cars).
    """
    yaw_rad = math.radians(yaw_degrees)
    # Forward vector
    dir_x = math.cos(yaw_rad)
    dir_y = math.sin(yaw_rad)
    
    half_wb = wheelbase / 2.0

    # --- 2-WHEEL LOGIC (Bikes) ---
    if track_width is None:
        front_x = x + dir_x * half_wb
        front_y = y + dir_y * half_wb
        back_x = x - dir_x * half_wb
        back_y = y - dir_y * half_wb
        
        z_front = get_ground_height(front_x, front_y)
        z_back = get_ground_height(back_x, back_y)
        
        if z_front == -9999.0 or z_back == -9999.0: return None, None, None

        # --- DEBUG DRAWING (Bikes) ---
        if sim_config.DEBUG_WHEEL_CONTACT:
            draw_debug_cube(stage, (front_x, front_y, z_front))
            draw_debug_cube(stage, (back_x, back_y, z_back))

        # Pitch and Height
        pitch_deg = math.degrees(math.atan2(z_front - z_back, wheelbase))
        z_center = (z_front + z_back) / 2.0
        
        return pitch_deg, 0.0, z_center # Roll is 0 for bikes

    # --- 4-WHEEL LOGIC (Cars) ---
    else:
        half_tw = track_width / 2.0
        
        # Right vector (perpendicular to forward vector)
        right_x = dir_y
        right_y = -dir_x
        
        # 4-wheel coordinates
        fl_x = x + dir_x * half_wb - right_x * half_tw # Front-Left
        fl_y = y + dir_y * half_wb - right_y * half_tw
        
        fr_x = x + dir_x * half_wb + right_x * half_tw # Front-Right
        fr_y = y + dir_y * half_wb + right_y * half_tw
        
        bl_x = x - dir_x * half_wb - right_x * half_tw # Back-Left
        bl_y = y - dir_y * half_wb - right_y * half_tw
        
        br_x = x - dir_x * half_wb + right_x * half_tw # Back-Right
        br_y = y - dir_y * half_wb + right_y * half_tw
        
        # 4 Raycasts
        z_fl = get_ground_height(fl_x, fl_y)
        z_fr = get_ground_height(fr_x, fr_y)
        z_bl = get_ground_height(bl_x, bl_y)
        z_br = get_ground_height(br_x, br_y)
        
        if -9999.0 in (z_fl, z_fr, z_bl, z_br): return None, None, None

        # --- VISUAL DEBUG (Cars) ---
        if sim_config.DEBUG_WHEEL_CONTACT:
            draw_debug_cube(stage, (fl_x, fl_y, z_fl))
            draw_debug_cube(stage, (fr_x, fr_y, z_fr))
            draw_debug_cube(stage, (bl_x, bl_y, z_bl))
            draw_debug_cube(stage, (br_x, br_y, z_br))
        
        # 1. Front vs Rear average to calculate Pitch
        z_front_avg = (z_fl + z_fr) / 2.0
        z_back_avg = (z_bl + z_br) / 2.0
        pitch_deg = math.degrees(math.atan2(z_front_avg - z_back_avg, wheelbase))
        
        # 2. Left vs Right average to calculate Roll
        z_left_avg = (z_fl + z_bl) / 2.0
        z_right_avg = (z_fr + z_br) / 2.0
        roll_deg = math.degrees(math.atan2(z_right_avg - z_left_avg, track_width))
        
        # 3. Exact center height
        z_center = (z_fl + z_fr + z_bl + z_br) / 4.0
        
        return pitch_deg, roll_deg, z_center


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


def estimate_items_by_budget(active_items, budget_max):
    """
    Estimates the number of items of each category that will be selected for budget_max,
    taking into account selection weights, costs, and pool sizes.
    """
    counts = {k: 0.0 for k in active_items.keys()}
    remaining_budget = budget_max
    active_keys = list(active_items.keys())
    
    while remaining_budget > 0 and active_keys:
        total_weight = sum(active_items[k]['selection_weight'] for k in active_keys)
        if total_weight == 0:
            break
            
        next_active_keys = []
        budget_shares = {}
        for k in active_keys:
            share = remaining_budget * (active_items[k]['selection_weight'] / total_weight)
            budget_shares[k] = share
            
        used_in_round = 0
        for k in active_keys:
            cost = active_items[k]['cost_units']
            pool_size = active_items[k].get('pool_size', 999999)
            
            items_share = budget_shares[k] / cost
            items_remaining = pool_size - counts[k]
            
            if items_share >= items_remaining:
                counts[k] = pool_size
                used_in_round += items_remaining * cost
            else:
                counts[k] += items_share
                used_in_round += budget_shares[k]
                next_active_keys.append(k)
                
        remaining_budget -= used_in_round
        if abs(used_in_round) < 1e-5 or len(next_active_keys) == len(active_keys):
            break
            
        active_keys = next_active_keys
        
    return counts


def select_objects_by_budget(available_keys, config_map, total_budget):
    """
    Selects objects until the budget of 'units' is filled.
    Ensures that we do not select more items of a category than its pool_size.
    Returns a list of keys (e.g: ['truck', 'rock', 'rock', 'sign'])
    """
    selected_keys = []
    current_usage = 0
    counts = {k: 0 for k in available_keys}
    
    # Avoid infinite loops if nothing fits
    active_costs = [config_map[k]['cost_units'] for k in available_keys]
    if not active_costs: return []
    min_cost = min(active_costs)
    
    attempts = 0
    while (total_budget - current_usage) >= min_cost and attempts < 100:
        valid_keys = [k for k in available_keys if counts[k] < config_map[k].get('pool_size', 999999)]
        if not valid_keys:
            break
            
        # Weighted selection
        weights = [config_map[k]['selection_weight'] for k in valid_keys]
        
        choice = random.choices(valid_keys, weights=weights, k=1)[0]
        cost = config_map[choice]['cost_units']
        
        if current_usage + cost <= total_budget:
            selected_keys.append(choice)
            current_usage += cost
            counts[choice] += 1
            attempts = 0  # Reset attempts on successful selection
        else:
            attempts += 1
            
    return selected_keys


def get_smart_poses_near_target(stage, target_pos, candidates_specs, max_radius=10.0, existing_obstacles=[], cam_pos=None, fov_margin=60, look_at_target=None):
    """
    Generates non-overlapping positions and rotations for a set of objects around a target point.
    Adjusts height and pitch based on terrain geometry.
    """
    valid_results = []
    tx, ty, tz = target_pos
    
    # Calculate look direction vectors
    lx, ly, lz = look_at_target if look_at_target is not None else target_pos
    look_target = (lx, ly, lz)
    
    # Local copy of obstacles (x, y, radius)
    current_obstacles = [(p[0], p[1], p[2]) for p in existing_obstacles] 

    for i, spec in enumerate(candidates_specs):
        obj_radius = spec['radius']
        obj_wheelbase = spec.get('wheelbase', None)
        obj_track_width = spec.get('track_width', None)

        s_min, s_max = spec.get('spawn_radius', (0.0, max_radius))
        
        placed = False
        attempts = 0
        max_attempts = 100 # Attempts per object
        
        reject_dist = 0
        reject_fov = 0
        reject_collision = 0
        reject_ground = 0

        total_tries = 0
        max_tries = 500

        # Calculate camera heading direction for FOV-directed theta sampling
        angle_cam = 0.0
        if cam_pos is not None:
            angle_cam = math.atan2(look_target[1] - cam_pos[1], look_target[0] - cam_pos[0])

        while not placed and attempts < max_attempts and total_tries < max_tries:
            total_tries += 1
            
            # Circle Method: sample distance r
            r = random.uniform(s_min, s_max)
            
            # Smart theta sampling: try uniform first. If it fails 5 times, target the camera FOV direction.
            if total_tries <= 5 or cam_pos is None:
                theta = random.uniform(0, 2 * math.pi)
            else:
                margin_rad = math.radians(fov_margin)
                theta = random.uniform(angle_cam - margin_rad, angle_cam + margin_rad)
                
            cand_x = tx + r * math.cos(theta)
            cand_y = ty + r * math.sin(theta)
            
            # Verify camera FOV
            if cam_pos:
                if not is_in_camera_fov(cam_pos, look_target, (cand_x, cand_y, tz), fov_margin):
                    reject_fov += 1
                    continue

            # Candidate is within FOV and distance limits, so we increment attempts
            attempts += 1
            
            # 2. Collision check (Variable Radius)
            collision = False
            for (ex, ey, er) in current_obstacles:
                dist = math.sqrt((cand_x - ex)**2 + (cand_y - ey)**2)
                # The minimum distance is the sum of the two radius
                min_separation = obj_radius + er
                
                if dist < min_separation:
                    collision = True
                    break
            
            if collision:
                reject_collision += 1
                continue
                
            if not collision:
                # 3. Calculate Height / Pitch
                angle_yaw = random.uniform(0, 360)
                
                if obj_wheelbase is not None:
                    # Vehicle mode
                    pitch_deg, roll_deg, z_adjusted = calculate_vehicle_orientation_on_terrain(
                        stage, cand_x, cand_y, angle_yaw, obj_wheelbase, obj_track_width
                    )
                    if pitch_deg is None:
                        reject_ground += 1
                        continue
                    
                    rotation = (90 - roll_deg, -pitch_deg, angle_yaw)
                    z_final = z_adjusted
                else:
                    # Static mode
                    z_ground = get_ground_height(cand_x, cand_y)
                    if z_ground == -9999.0:
                        reject_ground += 1
                        continue
                    rotation = (90, 0, angle_yaw)
                    z_final = z_ground
                
                # Save valid candidate
                valid_results.append( ((cand_x, cand_y, z_final), rotation, i) )
                current_obstacles.append( (cand_x, cand_y, obj_radius) )
                placed = True
                
        if not placed:
            print(f"[WARN] Could not place object with radius {obj_radius} after {total_tries} tries")

    return valid_results


def randomize_precalculated_shaders(stage, shader_paths, soft_colors):
    """
    Applies a random tint directly to pre-calculated shader paths.
    """
    if not shader_paths: return

    color_val = None
    if  soft_colors:
        base_light = random.uniform(0.4, 1.0)
        color_val = Gf.Vec3f(
            base_light * random.uniform(0.60, 1.0), # R
            base_light * random.uniform(0.60, 1.0), # G
            base_light * random.uniform(0.60, 1.0)  # B
        )
    else:
        color_val = Gf.Vec3f(
            random.uniform(0.05, 1.0),
            random.uniform(0.05, 1.0),
            random.uniform(0.05, 1.0)
        )
    
    for spath in shader_paths:
        shader_prim = stage.GetPrimAtPath(spath)
        if shader_prim.IsValid():
            shader = UsdShade.Shader(shader_prim)
            shader.CreateInput("base_color_factor", Sdf.ValueTypeNames.Color3f).Set(color_val)


def place_objects_from_config(stage, target_pos, config_map, pools_paths_map, budget_range, max_radius, previous_obstacles=[], cam_pos=None, fov_margin=60, look_at_target=None):
    """
    Master Orchestrator.
    Handles: Selection (Budget) -> Unique Assignment (Stack) -> Placement -> Cleanup.
    Randomizes materials if needed.
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
    shaders_candidates = []
    
    for key in chosen_keys:
        # Skip if no stock or pool is empty (Dynamic pruning)
        if not working_pools.get(key):
            continue
            
        obj_data = working_pools[key].pop()
        cfg = config_map[key]
        
        spawn_r = cfg.get('spawn_radius', (0.0, max_radius))
        s_min, s_max = spawn_r
        s_max = min(s_max, max_radius)
        s_min = min(s_min, s_max)

        candidates_specs.append({
            'radius': cfg['radius'],
            'wheelbase': cfg.get('wheelbase', None),
            'track_width': cfg.get('track_width', None),
            'spawn_radius': (s_min, s_max)
        })
        paths_candidates.append(obj_data["path"])
        shaders_candidates.append(obj_data["shaders"])
        keys_candidates.append(key)
        
    # 4. Calculate Poses (Mathematics)
    # results devuelve: [ ((x,y,z), rot, original_index), ... ]
    results = get_smart_poses_near_target(stage, target_pos, candidates_specs, max_radius, previous_obstacles, cam_pos, fov_margin, look_at_target)
    
    # 5. Move successfully placed objects
    new_obstacles = [] 
    successfully_placed_paths = set()

    for (pos, rot, original_idx) in results:
        path = paths_candidates[original_idx]
        shaders = shaders_candidates[original_idx]
        key = keys_candidates[original_idx]
        cfg = config_map[key]
        
        s_min, s_max = cfg.get('scale_range', (1.0, 1.0))
        scale = random.uniform(s_min, s_max)
        
        update_prim_pose_and_visibility(stage, path, pos, rot, scale, visible=True)

        if "randomize_materials" in cfg and cfg["randomize_materials"]:
            randomize_precalculated_shaders(stage, shaders, cfg.get("randomize_soft_colors", False))
        
        # Register success
        dist_to_target = math.sqrt((pos[0] - target_pos[0])**2 + (pos[1] - target_pos[1])**2)
        new_obstacles.append( (pos[0], pos[1], cfg['radius']) )
        successfully_placed_paths.add(path)
        
    # 6. Clean up unused objects
    for key, pool_list in pools_paths_map.items():
        for obj_data in pool_list:
            path = obj_data["path"]
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


def validate_placement_config(config_map, budget_max, container_radius, fov_factor=0.30, context_name="Config"):
    """
    Analyses if the objects fit in the assigned space based on the maximum budget,
    taking into account that the camera only sees a percentage (fov_factor) of the total circle.
    """
    # Only consider active items
    active_items = {k: v for k, v in config_map.items() if v.get('active', True)}
    
    if not active_items:
        return "green", f"[{context_name}] No active items."

    # 1. Estimate expected counts under budget and pool limits
    expected_counts = estimate_items_by_budget(active_items, budget_max)
    
    # 2. Calculate required area based on estimated counts
    required_area = 0.0
    for k, count in expected_counts.items():
        radius = active_items[k].get('radius', 1.0)
        area = math.pi * (radius ** 2)
        required_area += count * area

    # 3. Available area
    total_theoretical_area = math.pi * (container_radius ** 2)
    available_area = total_theoretical_area * (fov_factor + 0.1)
    
    # 4. Packing Factor
    # Perfect circles fill ~90%. Random placement is ~40-50%.
    # If we exceed 60%, we will start having many failures.
    fill_ratio = required_area / available_area if available_area > 0 else 999.0
    
    packing_limit_safe = 0.45  # Green: Very safe
    packing_limit_warn = 0.65  # Yellow: Possible failures, but acceptable
    
    msg = (f"[{context_name}] Ratio of Occupation: {fill_ratio*100:.1f}% "
           f"(Req: {required_area:.0f}m² / Disp (Visible): {available_area:.0f}m²)")
    
    if fill_ratio > 1.0:
        return "red", f"[IMPOSSIBLE] {msg} -> IMPOSSIBLE (Overload > 100%)"
    elif fill_ratio > packing_limit_warn:
        return "orange", f"[CRITICAL] {msg} -> CRITICAL (High probability of failure)"
    elif fill_ratio > packing_limit_safe:
        return "yellow", f"[DENSE] {msg} -> DENSE (May have some warnings)"
    else:
        return "green", f"[OK] {msg} -> OK"


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


def is_in_camera_fov(cam_pos, target_pos, obj_pos, fov_margin_degrees=60.0):
    """
    Checks if an object is within the camera's field of view.
    """
    cx, cy, cz = cam_pos
    tx, ty, tz = target_pos
    ox, oy, oz = obj_pos

    # Camera -> Target Vector (Central axis of vision)
    v_cam_target = (tx - cx, ty - cy, tz - cz)
    norm_ct = math.sqrt(v_cam_target[0]**2 + v_cam_target[1]**2 + v_cam_target[2]**2)

    # Camera -> Object Vector
    v_cam_obj = (ox - cx, oy - cy, oz - cz)
    norm_co = math.sqrt(v_cam_obj[0]**2 + v_cam_obj[1]**2 + v_cam_obj[2]**2)

    if norm_ct == 0 or norm_co == 0:
        return True

    # Dot product to isolate the cosine of the angle
    dot_product = (v_cam_target[0]*v_cam_obj[0] + v_cam_target[1]*v_cam_obj[1] + v_cam_target[2]*v_cam_obj[2])
    cos_angle = dot_product / (norm_ct * norm_co)
    
    # Clamp to avoid floating point errors
    cos_angle = max(min(cos_angle, 1.0), -1.0)
    angle_rad = math.acos(cos_angle)
    
    # If angle is smaller than our margin, it is "in sight"
    return math.degrees(angle_rad) <= fov_margin_degrees