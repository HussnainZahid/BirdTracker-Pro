import torch

class Config:
    VIDEO_PATH = "data/input/input.mp4"
    OUTPUT_PATH = "data/output/output.mp4"
    MODEL_PATH = "models/yolov8s.pt"

    CONFIDENCE = 0.4
    IOU = 0.5

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    TARGET_CLASS = 14  # bird class in COCO dataset

    MAX_AGE = 30
    N_INIT = 3

    # Extra settings for better UX
    SHOW_FPS = True
    SHOW_TOTAL_COUNT = True