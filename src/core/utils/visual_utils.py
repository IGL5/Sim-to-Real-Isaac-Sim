import cv2

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
            cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), alert_color, -1)
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
    return img