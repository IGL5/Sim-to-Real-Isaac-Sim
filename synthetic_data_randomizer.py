from omni.isaac.kit import SimulationApp
import os
import argparse

import carb
from isaacsim.core.utils.nucleus import get_assets_root_path
from isaacsim.core.utils.stage import get_current_stage, open_stage
from omni.timeline import get_timeline_interface
from pxr import UsdPhysics, PhysxSchema
from pxr import Semantics
import omni.replicator.core as rep
from pxr import UsdShade, Sdf, UsdGeom, UsdPhysics, Gf
import random
from omni.physx import get_physx_scene_query_interface


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

simulation_app = SimulationApp(launch_config=CONFIG)

# Increase subframes if shadows/ghosting appears of moving objects
rep.settings.carb_settings("/omni/replicator/RTSubframes", 4)


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


def main():
    # Open the environment in a new stage
    map_path = os.path.join(os.getcwd(), "map", "Environment_variable.usd")
    open_stage(map_path)
    stage = get_current_stage()

    # Ensure a PhysicsScene exists
    if not stage.GetPrimAtPath("/PhysicsScene"):
        scene = UsdPhysics.Scene.Define(stage, Sdf.Path("/PhysicsScene"))
        scene.CreateGravityDirectionAttr().Set((0, 0, -1))
        scene.CreateGravityMagnitudeAttr().Set(9.81)

    timeline = get_timeline_interface()
    timeline.play()

    # --- Material Assignment for Terrain ---
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

    # Run physics warmup
    for i in range(60):
        simulation_app.update()

    # --- Manual Camera Setup ---
    # We create a specific USD camera so we can control it easily
    camera_path = "/World/Camera"
    cam_prim = UsdGeom.Camera.Define(stage, camera_path).GetPrim()
    
    # Add a Dome Light for basic illumination
    def add_dome_light():
        rep.create.light(light_type="Dome", intensity=10, texture=None)
    
    add_dome_light()

    # --- Areas of Interest (AOI) Configuration ---
    WORLD_LIMITS = (-2800, 2800, -2800, 2800)


    def get_ground_height(x, y):
        """
        Cast a ray from high up (z=200) downwards to find the ground.
        Returns the Z height of the hit point, or 0 if no hit.
        """
        origin = carb.Float3(x, y, 200.0)
        direction = carb.Float3(0, 0, -1.0)
        distance = 400.0 # Sufficient to cover the range
        
        # Raycast using PhysX
        hit = get_physx_scene_query_interface().raycast_closest(origin, direction, distance)
        
        if hit["hit"]:
            return hit["position"][2] # Return Z component
        return 0.0 # Default if no ground found (e.g. hole or no collider)

    def get_random_pose():
        """
        Genera una posición totalmente aleatoria dentro de los límites globales del mapa.
        """
        # 1. Generar X e Y aleatorios en todo el mapa
        pos_x = random.uniform(WORLD_LIMITS[0], WORLD_LIMITS[1])
        pos_y = random.uniform(WORLD_LIMITS[2], WORLD_LIMITS[3])
        
        # 2. Raycast para encontrar la altura (Z) del terreno en ese punto
        ground_z = get_ground_height(pos_x, pos_y)
        
        # 3. Configurar altura de cámara (10 a 30 metros sobre el suelo)
        height_offset = random.uniform(10, 30)
        
        # Si el raycast devuelve 0 (posible mar profundo o fallo), 
        # asumimos altura 0 del agua y sumamos el offset.
        pos_z = ground_z + height_offset
        position = (pos_x, pos_y, pos_z)
        
        # 4. Mirar a un punto cercano en el suelo
        target_x = pos_x + random.uniform(-5, 5)
        target_y = pos_y + random.uniform(-5, 5)
        look_at = (target_x, target_y, ground_z)

        return position, look_at

    # --- Setup Writer ---
    writer = rep.WriterRegistry.get("KittiWriter")
    output_directory = args.data_dir
    print("Outputting data to ", output_directory)
    writer.initialize(output_dir=output_directory, omit_semantic_type=True)
    
    # Attach render product to our manually created camera
    RESOLUTION = (CONFIG["width"], CONFIG["height"])
    render_product = rep.create.render_product(camera_path, RESOLUTION)
    writer.attach(render_product)

    # --- Main Loop (Manual Execution) ---
    print(f"Starting generation of {CONFIG['num_frames']} frames...")
    
    # Asegúrate de que no haya restos de ejecuciones anteriores
    rep.orchestrator.stop() 
    
    for i in range(CONFIG["num_frames"]):
        print(f"Generating Frame {i}...")
        
        pos, target = get_random_pose()
        
        update_camera_pose(cam_prim, pos, target)
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
