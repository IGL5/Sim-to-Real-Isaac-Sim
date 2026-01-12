from omni.isaac.kit import SimulationApp
import os
import argparse

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

import carb
import omni
import omni.usd
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import get_current_stage, open_stage
from pxr import Semantics
import omni.replicator.core as rep

from omni.isaac.core.utils.semantics import get_semantics

# Increase subframes if shadows/ghosting appears of moving objects
rep.settings.carb_settings("/omni/replicator/RTSubframes", 4)

# This is the location of the palletjacks in the simready asset library
CAR_ASSETS = [ # SUPONEMOS QUE ENCUENTRO .USD DE COCHES
    ]

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


def add_cars():
    rep_obj_list = [
        rep.create.from_usd(
            usd=car_path,
            semantics=[("class", "car")],
            count=1
        )
        for car_path in CAR_ASSETS
    ]

    rep_car_group = rep.create.group(rep_obj_list)

    return rep_car_group


def full_distractors_list(type):
    return []


def add_distractors(distractor_type="warehouse"):
    full_distractors = full_distractors_list(distractor_type)
    distractors = [rep.create.from_usd(distractor_path, count=1) for distractor_path in full_distractors]
    distractor_group = rep.create.group(distractors)
    return distractor_group


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


def main():
    # Open the environment in a new stage
    # Open the environment in a new stage
    # Load the specific map USD
    # NOTE: Ensure this path is correct relative to where you run the script or use absolute path if needed.
    # We use os.getcwd() to find it relative to the script location.
    map_path = os.path.join(os.getcwd(), "map", "Environment_variable.usd")
    open_stage(map_path)
    stage = get_current_stage()

    # Run some app updates to make sure things are properly loaded
    for i in range(100):
        if i % 10 == 0:
            print(f"App update {i}..")
        simulation_app.update()

    # rep_cars_group = add_cars()
    # rep_distractor_group = add_distractors(distractor_type=args.distractors)

    # We only need labels for the palletjack objects
    # update_semantics(stage=stage, keep_semantics=["car"])

    # Create camera with Replicator API for gathering data
    cam = rep.create.camera(clipping_range=(0.1, 1000000))

    # Add a Dome Light for basic illumination
    def add_dome_light():
        rep.create.light(light_type="Dome", intensity=1000, texture=None)
    
    add_dome_light()

    # --- Areas of Interest (AOI) Configuration ---
    # Define zones with weights and 3D bounds: ((min_x, min_y, min_z), (max_x, max_y, max_z))
    # TODO: USER MUST UPDATE THESE COORDINATES based on the actual map layout
    ZONES = [
        {
            "name": "Flat Area / City",
            "weight": 0.2, # 20% chance
            "bounds": ((-10, -10, 15), (0, 0, 25)) 
        },
        {
            "name": "Variable Terrain",
            "weight": 0.6, # 60% chance
            "bounds": ((10, 10, 10), (50, 50, 30))
        },
        {
            "name": "Boundary / Transition",
            "weight": 0.2, # 20% chance
            "bounds": ((0, 0, 15), (10, 10, 25))
        }
    ]

    import random
    from omni.physx import get_physx_scene_query_interface

    def get_ground_height(x, y):
        """
        Cast a ray from high up (z=1000) downwards to find the ground.
        Returns the Z height of the hit point, or 0 if no hit.
        """
        origin = carb.Float3(x, y, 1000.0)
        direction = carb.Float3(0, 0, -1.0)
        distance = 2000.0 # Sufficient to cover the range
        
        # Raycast using PhysX
        hit = get_physx_scene_query_interface().raycast_closest(origin, direction, distance)
        
        if hit["hit"]:
            return hit["position"][2] # Return Z component
        return 0.0 # Default if no ground found (e.g. hole or no collider)

    def get_random_zone_pose():
        # Select a zone based on weights
        total_weight = sum(exclude["weight"] for exclude in ZONES)
        r = random.uniform(0, total_weight)
        uplimit = 0
        selected_zone = ZONES[0]
        
        for zone in ZONES:
            uplimit += zone["weight"]
            if r <= uplimit:
                selected_zone = zone
                break
        
        # Generate random position within bounds
        bounds = selected_zone["bounds"]
        min_pt, max_pt = bounds[0], bounds[1]
        
        pos_x = random.uniform(min_pt[0], max_pt[0])
        pos_y = random.uniform(min_pt[1], max_pt[1])
        
        # --- Raycast Logic ---
        # Find the actual ground height at this X, Y
        ground_z = get_ground_height(pos_x, pos_y)
        
        # Set camera 10-20m above the detected ground
        height_offset = random.uniform(10, 20)
        pos_z = ground_z + height_offset
        
        # Camera position
        position = (pos_x, pos_y, pos_z)
        
        # Look at a point on the ground (roughly)
        target_x = pos_x + random.uniform(-5, 5)
        target_y = pos_y + random.uniform(-5, 5)
        # Look at the ground height found
        look_at = (target_x, target_y, ground_z)

        return position, look_at

    # Register the randomizer
    def randomize_camera():
        pos, target = get_random_zone_pose()
        with cam:
            rep.modify.pose(position=pos, look_at=target)

    rep.randomizer.register(randomize_camera)

    # trigger replicator pipeline
    with rep.trigger.on_frame(num_frames=CONFIG["num_frames"]):
        rep.randomizer.randomize_camera()

        # TODO:
        # Add distractors
        # Randomize positions and rotations
        # Randomize textures
        # Randomize the lighting of the scene

    # Set up the writer
    writer = rep.WriterRegistry.get("KittiWriter")

    # output directory of writer
    output_directory = args.data_dir
    print("Outputting data to ", output_directory)

    # use writer for bounding boxes, rgb and segmentation
    writer.initialize(output_dir=output_directory,
                      omit_semantic_type=True, )

    # attach camera render products to wrieter so that data is outputted
    RESOLUTION = (CONFIG["width"], CONFIG["height"])
    render_product = rep.create.render_product(cam, RESOLUTION)
    writer.attach(render_product)

    # run rep pipeline
    run_orchestrator()
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
