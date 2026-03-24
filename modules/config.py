import os
import argparse

parser = argparse.ArgumentParser("Dataset generator")
parser.add_argument("--headless", action="store_true", help="Launch script headless")
parser.add_argument("--width", type=int, default=640, help="Width of image")     # 1280x720 (HD Resolution)
parser.add_argument("--height", type=int, default=480, help="Height of image")    # 640x480 (SD Resolution)
parser.add_argument("--num_frames", type=int, default=1, help="Number of frames to record")
parser.add_argument("--data_dir", type=str, default=os.getcwd() + "/_output_data",
                    help="Location where data will be output")

args, unknown_args = parser.parse_known_args()


# Sky config
HDR_MAPS_DIR = os.path.join(os.getcwd(), "assets", "hdr")
AVAILABLE_HDRS = []
HDR_INTENSITY_RANGE = (0.8, 1.5)

# "renderer": "RayTracedLighting" is another option to consider
CONFIG = {"renderer": "PathTracing", "headless": args.headless,
          "width": args.width, "height": args.height, "num_frames": args.num_frames}

# GENERAL CONSTANTS
MAP_NAME = "Environment_variable.usd"
WORLD_LIMITS = (-1300, 1300, -1300, 1300)
MATERIAL_SCALE_FLAT = (0.6, 1.0)
MATERIAL_SCALE_MOUNTAIN = (0.05, 0.1)
RT_SUBFRAMES = 32

TEXTURES_ROOT_DIR = os.path.join(os.getcwd(), "assets", "textures")

# CAMERA CONSTANTS
CAMERA_HEIGHT_RANGE = (0.0, 1.0)
CAMERA_DISTANCE_RANGE = (2.0, 6.0)
LOOKAT_JITTER_RADIUS = 0.5

# RAYCAST SETTINGS
RAYCAST_START_HEIGHT = 2000.0
RAYCAST_DISTANCE = 4000.0

# OBJECT BUDGET
OBJECTS_BUDGET_RANGE = (2.0, 10.0) # (0.5, 7.0), (2.0, 10.0), (1.85, 3.9)
OBJECTS_MAX_RADIUS = 3.5

# DISTRACTOR BUDGET
DISTRACTOR_BUDGET_RANGE = (800.0, 1000.0) # (15.0, 30.0), (30.0, 75.0)
DISTRACTOR_MAX_RADIUS = 30.0

# --- CONFIGURATION: ENVIRONMENT TARGETS ---
ENVIRONMENT_LOOKUP_KEYS = [
    "Terrain",
    "Terrain_flat",
    # "Road",
    # "Path",
    # "Lake"
]

# --- DOMAIN RANDOMIZATION LIMITS ---
MAX_PBR_MATERIALS = 30
MAX_HDR_MAPS = 15
RANDOMIZE_SKY = True
RANDOMIZE_TERRAIN = True

# ASSET POOLS
ASSETS_ROOT_DIR = os.path.join(os.getcwd(), "assets", "objects")
OBJECTS_CONFIG = {
    "bicycle": {
        "active": True,                     # Enable this object type
        "pool_size": 13,                    # Number of bicycles in the pool
        "radius": 0.8,                      # Radius of safety
        "cost_units": 2.0,                  # High cost (main character)
        "selection_weight": 100,            # Always want to appear if there's space
        "wheelbase": 0.6,                   # For incline calculation (None if not applicable)
        "scale_range": (1.0, 1.0),          # Fixed scale for rigorous datasets
        "randomize_materials": ["frame"],   # List of prim names to randomize materials
        "randomize_soft_colors": True       # Randomize colors but keep the same material
    },
}

# DISTRACTOR CONFIGURATION (Scatter)
# Keys must match folder names in assets/objects/distractors/
DISTRACTOR_CONFIG = {
    "vegetation": {
        "active": True,
        "pool_size": 30,
        "spawn_radius": (0.0, 10.0),
        "radius": 0.5,
        "cost_units": 2.0,
        "selection_weight": 80,
        "scale_range": (0.7, 1.5),
        "randomize_materials": ["main"],
        "randomize_soft_colors": True
    },
    "trees": {
        "active": True,
        "pool_size": 50,
        "spawn_radius": (3.0, 30.0),
        "radius": 2.0,
        "cost_units": 5.0,
        "selection_weight": 150,
        "scale_range": (0.6, 1.2),
        "randomize_materials": ["main"],
        "randomize_soft_colors": True
    },
    "debris": {
        "active": True,
        "pool_size": 100,
        "spawn_radius": (0.0, 8.0),
        "radius": 0.5,
        "cost_units": 0.5,
        "selection_weight": 70,
        "scale_range": (0.3, 1.0),
        "randomize_materials": ["main"],
        "randomize_soft_colors": True
    },
    "manmade": {
        "active": True,
        "pool_size": 60,
        "spawn_radius": (4.0, 10.0),
        "radius": 1.0,
        "cost_units": 2.0,
        "selection_weight": 40,
        "scale_range": (0.8, 1.2),
        "randomize_materials": ["main", "sec"],
        "randomize_soft_colors": True
    }
}

# --- DEBUGGING ---
DEBUG_WHEEL_CONTACT = False
