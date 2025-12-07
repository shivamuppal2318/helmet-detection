import cv2
import numpy as np
from ultralytics import YOLO

class HelmetDetector:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)
        self.class_names = ['helmet', 'no-helmet']  # Update with your classes

    def detect(self, frame: np.ndarray):
        results = self.model.predict(frame, conf=0.7)
        violations = []
        
        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            cls_ids = r.boxes.cls.cpu().numpy().astype(int)
            
            for box, conf, cls_id in zip(boxes, confs, cls_ids):
                x1, y1, x2, y2 = map(int, box)
                label = self.class_names[cls_id]
                
                if label == 'no-helmet':
                    violations.append({
                        "bbox": [x1, y1, x2, y2],
                        "confidence": float(conf),
                        "class": label
                    })
        
        return violations