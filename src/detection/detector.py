from ultralytics import YOLO
from src.utils.config import Config

class Detector:
    def __init__(self):
        self.model = YOLO(Config.MODEL_PATH)

    def detect(self, frame):
        results = self.model(
            frame,
            conf=Config.CONFIDENCE,
            iou=Config.IOU,
            device=Config.DEVICE
        )[0]

        detections = []
        for box in results.boxes:
            cls = int(box.cls[0])
            if cls == Config.TARGET_CLASS:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                detections.append(([x1, y1, x2 - x1, y2 - y1], conf, "bird"))
        return detections