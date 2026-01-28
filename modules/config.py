import os
import argparse

parser = argparse.ArgumentParser("Dataset generator")
parser.add_argument("--headless", type=bool, default=False, help="Launch script headless, default is False")
parser.add_argument("--height", type=int, default=816, help="Height of image") # 544
parser.add_argument("--width", type=int, default=1440, help="Width of image") # 960
parser.add_argument("--num_frames", type=int, default=1, help="Number of frames to record")
parser.add_argument("--distractors", type=str, default="None",
                    help="Options are ")
parser.add_argument("--data_dir", type=str, default=os.getcwd() + "/_output_data",
                    help="Location where data will be output")

args, unknown_args = parser.parse_known_args()

# "renderer": "RayTracedLighting" is another option to consider
CONFIG = {"renderer": "PathTracing", "headless": args.headless,
          "width": args.width, "height": args.height, "num_frames": args.num_frames}

# GENERAL CONSTANTS
WORLD_LIMITS = (-1300, 1300, -1300, 1300)
TEXTURES_ROOT_DIR = os.path.join(os.getcwd(), "assets", "textures")

# CAMERA CONSTANTS
CAMERA_HEIGHT_RANGE = (30.0, 80.0)
CAMERA_DISTANCE_RANGE = (10.0, 20.0)
LOOKAT_JITTER_RADIUS = 2.5

# ASSET POOLS
ASSETS_ROOT_DIR = os.path.join(os.getcwd(), "assets", "objects")
OBJECTS_CONFIG = {
    "cyclist": {
        "pool_size": 10,                # Memory pool size
        "num_visible_range": (1, 7),    # Number of visible objects per frame
        "wheelbase": 0.6,               # Physics: For incline calculation (None if not applicable)
        "scale": 1.0                    # Scale factor
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
