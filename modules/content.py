import os
import glob
import random
from pxr import UsdShade, Gf, Sdf, Usd
import omni.replicator.core as rep
from modules import config


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
        else:
            # If the exact file doesn't exist, we search for any .usd inside the folder.
            fallback_search = glob.glob(os.path.join(folder_path, "*.usd*"))
            if fallback_search:
                discovered_paths.append(fallback_search[0])
            else:
                print(f"[WARN] Skipping folder '{folder_name}': No USD file found inside.")

    print(f"--- Total discovered objectives: {len(discovered_paths)} ({obj_dir}) ---")
    return discovered_paths


def load_pbr_materials(stage):
    """
    Loads PBR materials and manually assigns textures to avoid API errors.
    Returns a list of UsdShade.Material.
    """
    if not os.path.exists(config.TEXTURES_ROOT_DIR):
        print(f"[ERROR] Texture directory not found: {config.TEXTURES_ROOT_DIR}")
        return []

    material_folders = [f.path for f in os.scandir(config.TEXTURES_ROOT_DIR) if f.is_dir()]
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

            color_val = Gf.Vec3f(
                random.uniform(0.4, 1.0), 
                random.uniform(0.4, 1.0), 
                random.uniform(0.4, 1.0)
            )
            
            # Apply changes to the shader attributes
            shader.CreateInput("texture_scale", Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(scale_val, scale_val))
            shader.CreateInput("texture_rotate", Sdf.ValueTypeNames.Float).Set(rot_val)
            shader.CreateInput("diffuse_tint", Sdf.ValueTypeNames.Color3f).Set(color_val)
            shader.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(random.uniform(0.4, 0.9))

            normal_strength = random.uniform(1.5, 2.5) 
            shader.CreateInput("bump_factor", Sdf.ValueTypeNames.Float).Set(normal_strength)

        # --- BIND ---
        for path in paths:
            prim = stage.GetPrimAtPath(path)
            if prim.IsValid():
                binding_api = UsdShade.MaterialBindingAPI(prim)
                binding_api.Bind(chosen_material)


def discover_assets(root_dir, category_name, recursive=False):
    """
    Looks for USD files. Unifies the logic of 'Objetivos' and 'Distractores'.
    """
    found_paths = []
    
    # 1. Recursive Mode (Distractors): assets/objects/distractors/cat/**/*.usd
    if recursive:
        search_path = os.path.join(root_dir, "distractors", category_name)
        if os.path.exists(search_path):
            found_paths = glob.glob(os.path.join(search_path, "**", "*.usd*"), recursive=True)
    
    # 2. Standard Mode (Objectives): assets/objects/cat/cat.usd
    if not found_paths:
        search_path = os.path.join(root_dir, category_name)
        if os.path.exists(search_path):
            # Priority: File with the same name as the folder
            # (Ex: assets/objects/cyclist/cyclist.usd)
            subfolders = [f.path for f in os.scandir(search_path) if f.is_dir()]
            for folder in subfolders:
                folder_name = os.path.basename(folder)
                expected_usd = os.path.join(folder, f"{folder_name}.usd")
                if os.path.exists(expected_usd):
                    found_paths.append(expected_usd)
                else:
                    # Fallback: any usd
                    any_usd = glob.glob(os.path.join(folder, "*.usd*"))
                    if any_usd: found_paths.append(any_usd[0])
            
            # If no subfolders, look in the root of category
            if not found_paths:
                found_paths = glob.glob(os.path.join(search_path, "*.usd*"))

    return found_paths

def create_class_pool(stage, config_map, root_dir):
    """
    Creates the object pool in Replicator (initially hidden).
    Serves both OBJECTS_CONFIG and DISTRACTOR_CONFIG.
    """
    pools_paths = {}
    print(f"--- Creating Asset Pools ---")

    for category, cfg in config_map.items():
        if not cfg.get("active", True): continue
            
        # Determine if it is recursive (if it is in distractors)
        is_recursive = category in config.DISTRACTOR_CONFIG
        
        asset_files = discover_assets(root_dir, category, recursive=is_recursive)
        
        if not asset_files:
            print(f"[WARN] No assets found for '{category}'")
            continue
            
        target_pool_size = cfg["pool_size"]
        created_paths = []
        
        # Fill list to meet pool size
        assets_to_use = []
        while len(assets_to_use) < target_pool_size:
            assets_to_use.extend(asset_files)
        assets_to_use = assets_to_use[:target_pool_size]
        random.shuffle(assets_to_use)
        
        print(f"   -> Instantiating {target_pool_size} objects for '{category}'...")
        
        for i, usd_path in enumerate(assets_to_use):
            prim_name = f"{category}_{i}"
            
            # Create instance
            rep.create.from_usd(usd_path, semantics=[('class', category)], name=prim_name)
            
            # Find real path
            found_path = None
            for p in stage.Traverse():
                if p.GetName() == prim_name:
                    found_path = str(p.GetPath())
                    break
            
            if found_path:
                # Hide
                prim = stage.GetPrimAtPath(found_path)
                UsdGeom.Imageable(prim).MakeInvisible()
                created_paths.append(found_path)
            else:
                print(f"[ERROR] Path not found for {prim_name}")

        pools_paths[category] = created_paths
        
    return pools_paths