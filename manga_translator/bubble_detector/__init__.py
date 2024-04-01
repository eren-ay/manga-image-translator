import numpy as np
from ultralytics import YOLO

def coord_format(coordinat):
    bubble_coord = coordinat.astype(int)
    x1 = bubble_coord[0]
    x2 = bubble_coord[2]
    y1 = bubble_coord[1]
    y2 = bubble_coord[3]
    arr = [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
    return arr

    

def dispatch(image: np.ndarray):
    model = YOLO('models/bubble_detect/best.pt')

    results = model.predict(image, save=False, imgsz=640, conf=0.45)
    result = results[0].boxes
    result = result.numpy()
    bubble_coord=[]
    
    for i in result:
        bubble_coord.append(coord_format(i.xyxy[0]))
    return bubble_coord