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
    open_stage() # Añadir el environment a la función con prefix_with_isaac_asset_server(ENV_URL)
    stage = get_current_stage()

    # Run some app updates to make sure things are properly loaded
    for i in range(100):
        if i % 10 == 0:
            print(f"App uppdate {i}..")
        simulation_app.update()

    rep_cars_group = add_cars()
    # rep_distractor_group = add_distractors(distractor_type=args.distractors)

    # We only need labels for the palletjack objects
    update_semantics(stage=stage, keep_semantics=["car"])

    # Create camera with Replicator API for gathering data
    cam = rep.create.camera(clipping_range=(0.1, 1000000))

    # trigger replicator pipeline
    with rep.trigger.on_frame(num_frames=CONFIG["num_frames"]):

        # Move the camera around in the scene, focus on the center of warehouse
        with cam:
            rep.modify.pose(position=rep.distribution.uniform((-9.2, -11.8, 0.4), (7.2, 15.8, 4)),
                            look_at=(0, 0, 0))

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
