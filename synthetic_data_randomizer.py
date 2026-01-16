import os
import argparse
import random
import math


parser = argparse.ArgumentParser("Dataset generator")
parser.add_argument("--headless", type=bool, default=False, help="Launch script headless, default is False")
parser.add_argument("--height", type=int, default=544, help="Height of image")
parser.add_argument("--width", type=int, default=960, help="Width of image")
parser.add_argument("--num_frames", type=int, default=1, help="Number of frames to record")
parser.add_argument("--distractors", type=str, default="None",
                    help="Options are 'warehouse', 'additional' or None (default)")
parser.add_argument("--data_dir", type=str, default=os.getcwd() + "/_output_data",
                    help="Location where data will be output")

args, unknown_args = parser.parse_known_args()

# This is the config used to launch simulation.
CONFIG = {"renderer": "RayTracedLighting", "headless": args.headless,
          "width": args.width, "height": args.height, "num_frames": args.num_frames}


from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp(launch_config=CONFIG)


import carb
from isaacsim.core.utils.nucleus import get_assets_root_path
from isaacsim.core.utils.stage import get_current_stage, open_stage
from omni.timeline import get_timeline_interface
from pxr import UsdPhysics, PhysxSchema, Semantics, UsdShade, Sdf, UsdGeom, Gf
import omni.replicator.core as rep
from omni.physx import get_physx_scene_query_interface


# Increase subframes if shadows/ghosting appears of moving objects
rep.settings.carb_settings("/omni/replicator/RTSubframes", 4)


# --- Areas of Interest (AOI) Configuration ---
WORLD_LIMITS = (-1300, 1300, -1300, 1300)

# Lista maestra de tus modelos (rutas relativas o absolutas)
# Imagina que tienes varias carpetas o archivos
CYCLIST_ASSET_POOL = [
    os.path.join(os.getcwd(), "assets", "cyclist", "cyclist_road_bike_black.usd"),
    # ... puedes añadir más aquí
]
CYCLIST_SCALE_FACTOR = 0.01


def find_prims_by_material_name(stage, material_names):
    """
    Finds prims that have a material binding matching one of the given names.
    Returns a dictionary matching material_name -> list of prim paths.
    """
    found_paths = {name: [] for name in material_names}
    
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh): # Only check meshes
            continue
            
        # Check direct binding
        binding_api = UsdShade.MaterialBindingAPI(prim)
        if binding_api:
            # We check the direct binding for simplicity
            direct_binding = binding_api.GetDirectBinding()
            material = direct_binding.GetMaterial()
            if material:
                mat_path = material.GetPath()
                mat_name = mat_path.name
                
                # Check if this material matches any of our targets
                for target_name in material_names:
                    if target_name in mat_name:
                        found_paths[target_name].append(str(prim.GetPath()))
    
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
                        # Skip repeated instance, instances are iterated twice due to their two semantic properties (class, data)
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


# needed for loading textures correctly
def prefix_with_isaac_asset_server(relative_path):
    assets_root_path = get_assets_root_path()
    if assets_root_path is None:
        raise Exception("Nucleus server not found, could not access Isaac Sim assets folder")
    return assets_root_path + relative_path


# This will handle replicator
def run_orchestrator():
    rep.orchestrator.run()

    # Wait until started
    while not rep.orchestrator.get_is_started():
        simulation_app.update()

    # Wait until stopped
    while rep.orchestrator.get_is_started():
        simulation_app.update()

    rep.BackendDispatch.wait_until_done()
    rep.orchestrator.stop()


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
    
    # Clear existing ops to be safe/clean or just set transform
    xform_api.ClearXformOpOrder()
    xform_api.AddTransformOp().Set(view_matrix.GetInverse())


def get_ground_height(x, y):
    """
    Cast a ray from high up (z=200) downwards to find the ground.
    Returns the Z height of the hit point, or 0 if no hit.
    """
    origin = carb.Float3(x, y, 500.0)
    direction = carb.Float3(0, 0, -1.0)
    distance = 700.0 # Sufficient to cover the range
    
    # Raycast using PhysX
    hit = get_physx_scene_query_interface().raycast_closest(origin, direction, distance)
    
    if hit["hit"]:
        return hit["position"][2] # Return Z component
    return 100.0 # Default if no ground found (e.g. hole or no collider)

def get_drone_camera_pose(focus_target):
    """
    Calcula la posición de la cámara orbitando alrededor de un punto de interés (focus_target).
    Simula un vuelo de dron.
    """
    tx, ty, tz = focus_target
    
    # --- PARÁMETROS DE VUELO ---
    # Distancia al objetivo (hipotenusa)
    distance = random.uniform(15.0, 30.0) 
    
    # Altura angular (Pitch): 
    # 0 grados = a ras de suelo (malo)
    # 90 grados = vista cenital perfecta (top-down)
    # Dron típico = 30 a 60 grados
    elevation_deg = random.uniform(30.0, 60.0)
    elevation_rad = math.radians(elevation_deg)
    
    # Ángulo alrededor del objetivo (Azimut)
    azimuth_deg = random.uniform(0, 360)
    azimuth_rad = math.radians(azimuth_deg)
    
    # --- CÁLCULO DE POSICIÓN (COORDENADAS ESFÉRICAS) ---
    # Z (Altura) = Target Z + (Distancia * seno de la elevación)
    cam_z = tz + distance * math.sin(elevation_rad)
    
    # Proyección en el plano XY (cuánto nos alejamos horizontalmente)
    dist_xy = distance * math.cos(elevation_rad)
    
    cam_x = tx + dist_xy * math.cos(azimuth_rad)
    cam_y = ty + dist_xy * math.sin(azimuth_rad)
    
    # --- SEGURIDAD EXTRA ---
    # Verificamos la altura del suelo justo DEBAJO de la cámara para no estar bajo tierra
    # (por si hay una montaña justo detrás del dron)
    ground_under_cam = get_ground_height(cam_x, cam_y)
    
    if cam_z < ground_under_cam + 5.0:
        # Si la órbita nos dejó muy bajos respecto a la montaña de atrás,
        # forzamos la altura para estar 5 metros sobre el suelo real.
        print("DEBUG: Camera clipped terrain, adjusting altitude.")
        cam_z = ground_under_cam + 5.0

    return (cam_x, cam_y, cam_z)


def sample_assets_from_pool(asset_pool, num_samples, allow_duplicates=True):
    """
    Selecciona assets aleatorios de una lista.
    """
    if not asset_pool:
        return []
    
    if allow_duplicates:
        return random.choices(asset_pool, k=num_samples)
    else:
        # Aseguramos no pedir más de los que hay si no permitimos duplicados
        k = min(num_samples, len(asset_pool))
        return random.sample(asset_pool, k)


def get_multiple_poses_near_target(target_pos, ground_func, num_objects, min_dist=2.0, max_radius=10.0):
    """
    Genera N posiciones válidas alrededor de un target sin que se superpongan.
    """
    valid_poses = []
    tx, ty, tz = target_pos
    
    # Límite de intentos para no colgar el programa si no caben
    max_attempts = num_objects * 50 
    attempts = 0
    
    while len(valid_poses) < num_objects and attempts < max_attempts:
        attempts += 1
        
        # 1. Generar candidato aleatorio
        r = random.uniform(1.0, max_radius) 
        theta = random.uniform(0, 2 * math.pi)
        
        cand_x = tx + r * math.cos(theta)
        cand_y = ty + r * math.sin(theta)
        
        # 2. Chequear colisión con los YA aceptados
        collision = False
        for (exist_pos, _) in valid_poses:
            ex, ey, ez = exist_pos
            dist = math.sqrt((cand_x - ex)**2 + (cand_y - ey)**2)
            if dist < min_dist:
                collision = True
                break
        
        # 3. Si no choca, calculamos Z y guardamos
        if not collision:
            cand_z = ground_func(cand_x, cand_y)
            rot_z = random.uniform(0, 360)
            valid_poses.append(((cand_x, cand_y, cand_z), (0, 0, rot_z)))
            
    if len(valid_poses) < num_objects:
        print(f"Warning: Could only place {len(valid_poses)}/{num_objects} objects without overlap.")
        
    return valid_poses


def main():
    # --- 1. CARGA DEL MAPA (Escalado y Centrado) ---
    map_path = os.path.join(os.getcwd(), "map", "Environment_variable.usd")
    open_stage(map_path)
    stage = get_current_stage()
    
    # Aplicamos escala 0.5 al mapa
    root_prim = stage.GetDefaultPrim()
    xform = UsdGeom.Xformable(root_prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0)) # Aseguramos centro
    xform.AddScaleOp().Set(Gf.Vec3d(0.5, 0.5, 0.5))     # Reducimos tamaño

    # Physics Scene
    if not stage.GetPrimAtPath("/PhysicsScene"):
        scene = UsdPhysics.Scene.Define(stage, Sdf.Path("/PhysicsScene"))
        scene.CreateGravityDirectionAttr().Set((0, 0, -1))
        scene.CreateGravityMagnitudeAttr().Set(9.81)

    timeline = get_timeline_interface()
    timeline.play()

    # --- 2. ASIGNACIÓN DE MATERIALES ---
    # Create simple colored materials
    # Green for "Terrain_flat" (Grass/Valley)
    mat_grass = rep.create.material_omnipbr(diffuse=(0.2, 0.5, 0.2), roughness=0.8)
    # Brown/Grey for "Terrain" (Mountain/Rocky)
    mat_rock = rep.create.material_omnipbr(diffuse=(0.6, 0.4, 0.25), roughness=0.9)
    
    targets = ["Terrain", "Terrain_flat"]
    found_paths_map = find_prims_by_material_name(stage, targets)
            
    # Apply Green to Flat
    flat_paths = found_paths_map.get("Terrain_flat", [])
    if flat_paths:
        print(f"Found {len(flat_paths)} meshes for Terrain_flat. Applying Green.")
        flat_group = rep.create.group(flat_paths)
        with flat_group:
            rep.modify.material(mat_grass)
            
    # Apply Rock to the rest (Terrain)
    terrain_paths = found_paths_map.get("Terrain", [])
    # Filter duplicates if any
    terrain_paths = [p for p in terrain_paths if p not in flat_paths]
    
    if terrain_paths:
        print(f"Found {len(terrain_paths)} meshes for Terrain. Applying Rock.")
        terrain_group = rep.create.group(terrain_paths)
        with terrain_group:
            rep.modify.material(mat_rock)


    # --- 3. CARGA DE CICLISTAS ---
    NUM_CYCLISTS = 2 
    selected_paths = sample_assets_from_pool(CYCLIST_ASSET_POOL, NUM_CYCLISTS, allow_duplicates=True)
    cyclist_reps = []
    
    for i, path in enumerate(selected_paths):
        rep_item = rep.create.from_usd(
            path, 
            semantics=[('class', 'cyclist')],
            name=f"cyclist_{i}" 
        )
        cyclist_reps.append(rep_item)

    # --- 4. DEBUG MARKERS (Esferas) ---
    # Roja = Target (Suelo)
    target_marker = rep.create.sphere(scale=0.5, visible=True)
    with target_marker:
        rep.modify.material(rep.create.material_omnipbr(diffuse=(1, 0, 0))) # Rojo
        rep.modify.pose(position=(0,0,-1000)) # Ocultar inicialmente

    # Run physics warmup
    for i in range(60):
        simulation_app.update()
    
    # --- 5. LUCES Y CÁMARA (SETUP) ---
    
    # Luz Ambiental (Relleno)
    rep.create.light(light_type="Dome", intensity=10, texture=None)

    # Sol (Principal) - Ajusta intensidad si se quema
    rep.create.light(
        light_type="Distant", 
        intensity=20, 
        rotation=(300, 0, 0)
    )
    
    # CÁMARA REPLICATOR (La clave para arreglar la rotación)
    cam_rep = rep.create.camera(
        focal_length=18.0,
        name="DroneCam" 
    )
    
    # Writer
    writer = rep.WriterRegistry.get("KittiWriter")
    writer.initialize(output_dir=args.data_dir, omit_semantic_type=True)
    
    render_product = rep.create.render_product(cam_rep, (CONFIG["width"], CONFIG["height"]))
    writer.attach(render_product)

    # --- 6. BUCLE PRINCIPAL ---
    print(f"Starting generation of {CONFIG['num_frames']} frames...")
    rep.orchestrator.stop()
    
    # --- Main Loop ---
    for i in range(CONFIG["num_frames"]):
        print(f"\n--- GENERATING FRAME {i} ---")
        
        # A. ELEGIR TARGET (Dentro de los nuevos límites -600 a 600)
        tx = random.uniform(WORLD_LIMITS[0], WORLD_LIMITS[1])
        ty = random.uniform(WORLD_LIMITS[2], WORLD_LIMITS[3])
        tz = get_ground_height(tx, ty)
        
        # Si no encontramos suelo (valor por defecto 100.0 o hueco), saltamos
        # Nota: Ajusta esto según lo que devuelva tu get_ground_height cuando falla
        if tz > 500.0 or tz < -200.0: 
            print(f"Skipping target ({tx:.1f}, {ty:.1f}, {tz:.1f}) - Invalid Ground")
            continue
            
        current_target = (tx, ty, tz)
        
        # B. POSICIONAR CÁMARA (Usando LookAt de Replicator)
        # Calculamos dónde debe estar el dron
        cam_x, cam_y, cam_z = get_drone_camera_pose(current_target) # Tu función modificada antes
        
        with cam_rep:
            rep.modify.pose(
                position=(cam_x, cam_y, cam_z),
                look_at=current_target  # <--- ESTO ARREGLA LA ROTACIÓN Y EL CIELO
            )

        # Mover esfera roja al target para confirmar visualmente
        with target_marker:
            rep.modify.pose(position=current_target)

        # C. POSICIONAR CICLISTAS (Evitando clipping)
        # Aumentamos min_dist a 4.0 metros para que no se fusionen
        poses = get_multiple_poses_near_target(
            target_pos=current_target,
            ground_func=get_ground_height,
            num_objects=len(cyclist_reps),
            min_dist=2.5,   # Ajustado para bicis de tamaño real
            max_radius=8.0  # Radio más pequeño para asegurar que salgan en plano
        )
        
        for rep_item, (pos, rot) in zip(cyclist_reps, poses):
            with rep_item:
                rep.modify.pose(
                    position=pos,
                    rotation=rot,
                    scale=CYCLIST_SCALE_FACTOR
                )

        # D. DISPARAR
        simulation_app.update()
        rep.orchestrator.step(delta_time=0.0, rt_subframes=20)

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
