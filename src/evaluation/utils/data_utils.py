from pathlib import Path
from src.core.utils import math_utils as mu

def parse_kitti_label(label_path, width, height, return_polygons=False):
    """
    Parses YOLO/KITTI label files supporting both detection (5 tokens: class_id xc yc w h)
    and segmentation (7+ tokens: class_id x1 y1 x2 y2 ...).
    Returns list of boxes [class_id, x1, y1, x2, y2].
    If return_polygons=True, returns tuple (boxes, polygons).
    """
    boxes = []
    polygons = []
    if not Path(label_path).exists():
        return (boxes, polygons) if return_polygons else boxes

    try:
        with open(label_path, 'r') as f:
            for line in f:
                p = line.strip().split()
                if len(p) < 5:
                    continue

                class_id = int(p[0])
                coords = [float(x) for x in p[1:]]

                # Segmentation polygon format (7+ elements and even number of coordinates)
                if len(coords) >= 6 and len(coords) % 2 == 0:
                    xs_norm = coords[0::2]
                    ys_norm = coords[1::2]

                    xmin_n, xmax_n = min(xs_norm), max(xs_norm)
                    ymin_n, ymax_n = min(ys_norm), max(ys_norm)

                    x1 = int(xmin_n * width)
                    y1 = int(ymin_n * height)
                    x2 = int(xmax_n * width)
                    y2 = int(ymax_n * height)

                    boxes.append([class_id, x1, y1, x2, y2])

                    abs_poly = []
                    for i in range(0, len(coords), 2):
                        abs_poly.extend([coords[i] * width, coords[i+1] * height])
                    polygons.append([class_id, abs_poly])

                # Standard Bounding Box format (4 coordinates: cx cy w h)
                elif len(coords) == 4:
                    xc, yc, w, h = coords
                    x1, y1, x2, y2 = mu.yolo_to_corners(xc, yc, w, h, width, height)
                    boxes.append([class_id, x1, y1, x2, y2])

                    # Equivalent rectangular polygon
                    rect_poly = [x1, y1, x2, y1, x2, y2, x1, y2]
                    polygons.append([class_id, rect_poly])

    except Exception as e:
        print(f"⚠️ Error reading label {label_path}: {e}")

    return (boxes, polygons) if return_polygons else boxes