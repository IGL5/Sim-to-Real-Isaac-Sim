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
*   **Data Traceability & Analytics:** Automatically generates a comprehensive `generation_metadata.json` for every run, tracking performance metrics, theoretical object distribution, spatial coverage (camera limits vs object placement), and domain randomization coverage.
*   **Data Quality Control:** Includes an automated cleaning tool (`clean_dataset.py`) to mathematically detect and purge corrupted, completely flat, or excessively dark frames before they reach the AI.
*   **Kitti Writer:** Exports data in the Kitti format for easy integration with standard computer vision pipelines.

## 📂 Project Structure

```text
.
├── synthetic_data_randomizer.py  # Main entry point script
├── clean_dataset.py              # Automated quality control and purge script
├── modules/                      # Modularized Python logic
│   ├── config.py                 # Configuration settings & CLI args
│   ├── scene_utils.py            # Physics, camera, and positioning helpers
│   ├── content.py                # Asset handling and material randomization
│   └── __init__.py
├── assets/                       # Objects, Distractors, Map and Textures
├── _output_data/                 # Generated dataset output (created automatically)
│   ├── CameraName/               # Camera folders
│   └── generation_metadata.json  # Auto-generated traceability report
└── YOLO/                         # YOLOv8 training and validation pipeline
```

## 🛠️ Prerequisites

*   **Nvidia Isaac Sim:** This tool is designed to run within the Isaac Sim environment.
*   **Python:** Uses the Python interpreter bundled with Isaac Sim.

## ⚙️ Configuration

Key settings can be modified in **`modules/config.py`**:

*   **`WORLD_LIMITS`**: Define the valid area for object placement.
*   **`OBJECTS_CONFIG` & `OBJECTS_BUDGET_RANGE`**: Configure main target classes, pool sizes, scale, and placement density.
*   **`DISTRACTOR_CONFIG` & `DISTRACTOR_BUDGET_RANGE`**: Manage the density and types of background objects (vegetation, debris, etc.) to increase scene complexity.
*   **`CAMERA_CONSTANTS`**: Adjust camera height, distance ranges, and jitter.
*   **`CONFIG`**: Rendering settings (PathTracing vs RayTracedLighting) and image resolution.

## 🏃 Usage

Run the script using the Isaac Sim Python interpreter (typically found in your Isaac Sim installation folder):

```bash
# Example from project folder
C:\isaac-sim\python.bat .\synthetic_data_randomizer.py --headless --num_frames 100 --width 640 --height 480
```

### Command Line Arguments

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--num_frames` | Number of frames to generate | `1` |
| `--width` | Image width | `1440` |
| `--height` | Image height | `810` |
| `--headless` | Run in headless mode (no UI) | `False` |
| `--data_dir` | Directory to save output data | `./_output_data` |


### 🧹 Cleaning the Dataset (Optional but Recommended)

Before passing the data to any AI pipeline, you can automatically purge corrupted, completely flat, or dark frames. This will also update the `generation_metadata.json`.

```bash
# Preview what would be deleted without actually deleting anything
python clean_dataset.py --dir _output_data --dry

# Execute the actual cleanup
python clean_dataset.py --dir _output_data
```


## 🧠 Model Training (YOLOv8)

This repository includes a complete pipeline to train a YOLOv8 object detector using the generated synthetic data. The training tools are located in the YOLO/ directory.

The pipeline includes:

- **Automatic ETL:** Converts Isaac Sim (Kitti) data to YOLO format and manages Train/Val/Test splits with session history tracking.
- **Training Management:** Automates fine-tuning, handles early stopping, and logs hyperparameter history.
- **Interactive Auditing:** Generates comprehensive, tabbed HTML reports (using Jinja2) containing Confusion Matrices, Heatmaps, confidence distributions, and Sim-to-Real metadata to validate model performance.
- **Model Benchmarking:** A built-in CLI tool to compare multiple experiments side-by-side, generating a dynamic and interactive JavaScript dashboard (Chart.js) to track metric improvements and model drift.

👉 [Go to Training Documentation](YOLO/README.md)