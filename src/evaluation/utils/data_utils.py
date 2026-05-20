import os

def parse_kitti_label(label_path, width, height):
    """ Converts txt to list of boxes [class_id, x1, y1, x2, y2] """
    boxes = []
    if not os.path.exists(label_path): return boxes
    
    try:
        with open(label_path, 'r') as f:
            for line in f:
                p = line.strip().split()
                # YOLO format: class_id xc yc w h
                class_id = int(p[0])
                xc, yc, w, h = float(p[1]), float(p[2]), float(p[3]), float(p[4])
                
                x1 = (xc - w/2) * width
                y1 = (yc - h/2) * height
                x2 = (xc + w/2) * width
                y2 = (yc + h/2) * height
                
                # <--- NUEVO: Añadimos class_id al principio de la lista
                boxes.append([class_id, x1, y1, x2, y2]) 
    except Exception as e:
        print(f"⚠️ Error reading label {label_path}: {e}")
        
    return boxes