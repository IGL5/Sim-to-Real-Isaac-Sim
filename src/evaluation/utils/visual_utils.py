import cv2
import numpy as np

def draw_boxes(img, boxes, color=(0, 255, 0), label="", confidences=None, classes=None, class_names=None):
    """
    Draws bounding boxes on an image.
    """
    for i, b in enumerate(boxes):
        # Extract safely the last 4 positions (coordinates)
        x1, y1, x2, y2 = map(int, b[-4:])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        text_to_draw = label
        
        # If Ground Truth (has 5 elements), extract its real class
        if len(b) == 5 and class_names:
            c_id = int(b[0])
            text_to_draw = class_names.get(c_id, f"Class {c_id}")
        
        # If Prediction, look at the list of classes we passed
        elif classes is not None and i < len(classes) and class_names:
            c_id = int(classes[i])
            text_to_draw = class_names.get(c_id, f"Class {c_id}")
            
        # Add confidence if it exists
        if confidences is not None and i < len(confidences):
            text_to_draw += f" {confidences[i]:.2f}"
            
        if text_to_draw: 
            # Put a colored background to make the text always legible
            (w, h), _ = cv2.getTextSize(text_to_draw, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            if y1 < 20:
                cv2.rectangle(img, (x1, y1), (x1 + w, y1 + 20), color, -1)
                cv2.putText(img, text_to_draw, (x1, y1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            else:
                cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), color, -1)
                cv2.putText(img, text_to_draw, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img


def draw_segmentation(img, boxes=None, polygons=None, color=(0, 255, 0), label="", confidences=None, classes=None, class_names=None, alpha=0.35):
    """
    Draws polygon masks and bounding boxes with semi-transparent overlay.
    - polygons: list of [class_id, [x1, y1, x2, y2, ...]] or list of point arrays/lists
    - boxes: list of [x1, y1, x2, y2] or [class_id, x1, y1, x2, y2]
    """
    if polygons is None or len(polygons) == 0:
        if boxes is not None:
            return draw_boxes(img, boxes, color, label, confidences, classes, class_names)
        return img

    overlay = img.copy()
    h_img, w_img = img.shape[:2]

    # Render translucent filled polygons
    for i, poly_item in enumerate(polygons):
        if isinstance(poly_item, list) and len(poly_item) == 2 and isinstance(poly_item[1], (list, np.ndarray)):
            pts_data = poly_item[1]
        else:
            pts_data = poly_item

        if pts_data is None or len(pts_data) < 6:
            continue

        pts = np.array(pts_data, dtype=np.float32).reshape(-1, 2)
        if np.max(pts) <= 1.05:
            pts[:, 0] *= w_img
            pts[:, 1] *= h_img
        pts_int = pts.astype(np.int32)
        cv2.fillPoly(overlay, [pts_int], color)

    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    items_count = len(polygons)

    for i in range(items_count):
        poly_item = polygons[i]
        if isinstance(poly_item, list) and len(poly_item) == 2 and isinstance(poly_item[1], (list, np.ndarray)):
            c_id_item = poly_item[0]
            pts_data = poly_item[1]
        else:
            c_id_item = None
            pts_data = poly_item

        pts = np.array(pts_data, dtype=np.float32).reshape(-1, 2)
        if np.max(pts) <= 1.05:
            pts[:, 0] *= w_img
            pts[:, 1] *= h_img
        pts_int = pts.astype(np.int32)

        # Draw contour line
        cv2.polylines(img, [pts_int], isClosed=True, color=color, thickness=2)

        # Determine label position
        if boxes is not None and i < len(boxes):
            b = boxes[i]
            x1, y1 = map(int, b[-4:-2])
            x2, y2 = map(int, b[-2:])
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
        else:
            x1, y1 = int(np.min(pts[:, 0])), int(np.min(pts[:, 1]))

        # Text construction
        text_to_draw = label
        if c_id_item is not None and class_names:
            text_to_draw = class_names.get(int(c_id_item), f"Class {c_id_item}")
        elif classes is not None and i < len(classes) and class_names:
            c_id = int(classes[i])
            text_to_draw = class_names.get(c_id, f"Class {c_id}")
        elif boxes is not None and i < len(boxes) and len(boxes[i]) == 5 and class_names:
            c_id = int(boxes[i][0])
            text_to_draw = class_names.get(c_id, f"Class {c_id}")

        if confidences is not None and i < len(confidences):
            text_to_draw += f" {confidences[i]:.2f}"

        if text_to_draw:
            (w, h), _ = cv2.getTextSize(text_to_draw, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            if y1 < 20:
                cv2.rectangle(img, (x1, y1), (x1 + w, y1 + 20), color, -1)
                cv2.putText(img, text_to_draw, (x1, y1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            else:
                cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), color, -1)
                cv2.putText(img, text_to_draw, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return img


def draw_overlapping_pairs(img, pred_boxes, pairs_indices, confidences=None):
    """
    Draws ONLY the overlapping pairs of boxes to highlight them.
    pairs_indices: List of tuples [(idx1, idx2), (idx3, idx4)...]
    """
    alert_color = (0, 165, 255) 
    thickness = 3
    
    boxes_to_draw_idx = set()
    for i, j in pairs_indices:
        boxes_to_draw_idx.add(i)
        boxes_to_draw_idx.add(j)
        
    for idx in boxes_to_draw_idx:
        box = pred_boxes[idx]
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), alert_color, thickness)
        
        if confidences:
            conf = confidences[idx]
            label = f"{conf:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            
            if y1 < 20:
                cv2.rectangle(img, (x1, y1), (x1 + w, y1 + 20), alert_color, -1)
                cv2.putText(img, label, (x1, y1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            else:
                cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), alert_color, -1)
                cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
    return img