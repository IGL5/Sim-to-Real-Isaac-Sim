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

# RAYCAST SETTINGS
RAYCAST_START_HEIGHT = 2000.0
RAYCAST_DISTANCE = 4000.0

# --- CONFIGURATION: ENVIRONMENT TARGETS ---
ENVIRONMENT_LOOKUP_KEYS = [
    "Terrain",
    "Terrain_flat",
    # "Road",
    # "Path",
    # "Lake"
]

# ASSET POOLS
ASSETS_ROOT_DIR = os.path.join(os.getcwd(), "assets", "objects")
OBJECTS_CONFIG = {
    "cyclist": {
        "pool_size": 10,
        "active": True,
        "radius": 1.0,           # Radius of safety (bicycle + person)
        "cost_units": 2.0,       # High cost (main character)
        "selection_weight": 100, # Always want to appear if there's space
        "wheelbase": 0.6,        # For incline calculation (None if not applicable)
        "scale_range": (1.0, 1.0) # Fixed scale for rigorous datasets
    },
}

# DISTRACTOR CONFIGURATION (Scatter)
# Keys must match folder names in assets/objects/distractors/
DISTRACTOR_CONFIG = {
    "vegetation": {
        "active": True,
        "pool_size": 20,
        "spawn_radius": (5.0, 25.0),
        "radius": 1.5,          # Physical radius (m)
        "cost_units": 3.0,      # Cost units (large)
        "selection_weight": 10, # Medium Frecuency 
        "scale_range": (0.8, 1.5)
    },
    "debris": {
        "active": True,
        "pool_size": 20, # Increase pool for variety
        "spawn_radius": (2.0, 15.0),
        "radius": 0.3,          # Physical radius (m)
        "cost_units": 0.5,      # Cost units (small)
        "selection_weight": 50, # High Frecuency 
        "scale_range": (0.2, 0.5)
    },
    "manmade": {
        "active": True,
        "pool_size": 5,
        "spawn_radius": (10.0, 40.0),
        "radius": 0.8,
        "cost_units": 1.0,
        "selection_weight": 2, # Low Frecuency 
        "scale_range": (0.8, 1.2)
    }
}

# --- DEBUGGING ---
DEBUG_WHEEL_CONTACT = False
