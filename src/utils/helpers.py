import cv2
import time
import numpy as np

class FPSCounter:
    def __init__(self):
        self.start_time = time.time()
        self.frame_count = 0
        self.fps = 0

    def update(self):
        self.frame_count += 1
        elapsed = time.time() - self.start_time

        if elapsed >= 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.start_time = time.time()

        return round(self.fps, 2)


def preprocess_frame(frame):
    """Improve frame quality for better detection"""
    frame = cv2.convertScaleAbs(frame, alpha=1.1, beta=10)
    frame = cv2.GaussianBlur(frame, (3, 3), 0)
    return frame


def smooth_bbox(prev_bbox, curr_bbox, alpha=0.7):
    """Smooth bounding boxes to reduce jitter"""
    if prev_bbox is None:
        return curr_bbox
    x = int(alpha * prev_bbox[0] + (1 - alpha) * curr_bbox[0])
    y = int(alpha * prev_bbox[1] + (1 - alpha) * curr_bbox[1])
    w = int(alpha * prev_bbox[2] + (1 - alpha) * curr_bbox[2])
    h = int(alpha * prev_bbox[3] + (1 - alpha) * curr_bbox[3])
    return (x, y, w, h)


def generate_color(track_id):
    """
    Generate deterministic consistent color for each object ID
    Fixed for DeepSORT (track_id is sometimes str)
    """
    # DeepSORT returns track_id as string → convert safely to int
    if isinstance(track_id, str):
        try:
            track_id = int(track_id)
        except ValueError:
            track_id = hash(track_id) % (2**32)

    np.random.seed(track_id)
    color = tuple(np.random.randint(0, 255, 3).tolist())
    return color


def draw_fps(frame, fps):
    """Draw FPS on frame"""
    cv2.putText(frame, f"FPS: {fps}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return frame


def draw_total_count(frame, total):
    """Draw total unique birds count"""
    cv2.putText(frame, f"Total Birds: {total}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    return frame


def validate_video(path):
    """Validate input video"""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    width = int(cap.get(3))
    height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    return width, height, fps