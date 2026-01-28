# Sim-to-Real-Isaac-Sim

A synthetic data generation tool leveraging **Nvidia Isaac Sim Replicator** to bridge the sim-to-real gap. This project generates high-quality synthetic datasets with extensive domain randomization, including variations in lighting, materials, object placement, and camera perspectives, specifically tailored for Sim-to-Real applications.

## 🚀 Features

*   **Isaac Sim Replicator Integration:** Built on top of `omni.replicator` for efficient and scalable synthetic data generation.
*   **Physics-Aware Placement:** Utilizes physics queries (raycasting) to ensure objects are correctly placed on complex terrains, avoiding floating or intersecting objects.
*   **Domain Randomization:**
    *   **Materials:** Randomizes textures on terrain and objects to improve model robustness.
    *   **Lighting:** Varies sunlight intensity, angle, and ambient lighting.
    *   **Camera:** Randomizes camera position, distance, and look-at targets with jitter.
    *   **Objects:** Randomly selects, places, and manages visibility of objects from a pool (e.g., cyclists).
*   **Modular Architecture:** Cleanly organized codebase separating configuration, scene utilities, and content management.
*   **Kitti Writer:** Exports data in the Kitti format for easy integration with standard computer vision pipelines.

## 📂 Project Structure

```text
.
├── synthetic_data_randomizer.py  # Main entry point script
├── modules/                      # Modularized Python logic
│   ├── config.py                 # Configuration settings & CLI args
│   ├── scene_utils.py            # Physics, camera, and positioning helpers
│   ├── content.py                # Asset handling and material randomization
│   └── __init__.py
├── assets/                       # Objects and Textures
├── map/                          # Environment USD files
└── _output_data/                 # Generated dataset output (created automatically)
```

## 🛠️ Prerequisites

*   **Nvidia Isaac Sim:** This tool is designed to run within the Isaac Sim environment.
*   **Python:** Uses the Python interpreter bundled with Isaac Sim.

## ⚙️ Configuration

Key settings can be modified in **`modules/config.py`**:

*   **`WORLD_LIMITS`**: Define the valid area for object placement.
*   **`OBJECTS_CONFIG`**: Configure object classes, pool sizes, visibility ranges, and physical properties (wheelbase, scale).
*   **`CAMERA_CONSTANTS`**: Adjust camera height, distance ranges, and jitter.
*   **`CONFIG`**: Rendering settings (PathTracing vs RayTracedLighting).

## 🏃 Usage

Run the script using the Isaac Sim Python interpreter (typically found in your Isaac Sim installation folder):

```bash
# Example from Isaac Sim root directory
./python.sh path/to/synthetic_data_randomizer.py --num_frames 100 --width 1280 --height 720
```

### Command Line Arguments

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--num_frames` | Number of frames to generate | `1` |
| `--width` | Image width | `1440` |
| `--height` | Image height | `816` |
| `--headless` | Run in headless mode (no UI) | `False` |
| `--data_dir` | Directory to save output data | `./_output_data` |
