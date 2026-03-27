import cv2
from src.utils.helpers import generate_color

class Visualizer:
    def __init__(self):
        self.colors = {}

    def get_color(self, track_id):
        if track_id not in self.colors:
            self.colors[track_id] = generate_color(track_id)
        return self.colors[track_id]

    def draw(self, frame, objects):
        for obj in objects:
            x, y, w, h = obj["bbox"]
            track_id = obj["id"]
            color = self.get_color(track_id)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"Bird {track_id}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return frame