import cv2
import os
from src.detection.detector import Detector
from src.tracking.tracker import Tracker
from src.visualization.visualizer import Visualizer
from src.utils.config import Config
from src.utils.helpers import (
    FPSCounter, preprocess_frame, draw_fps,
    draw_total_count, validate_video
)

def main():
    # Validate input video
    validate_video(Config.VIDEO_PATH)
    print(f"✅ Processing video: {Config.VIDEO_PATH}")

    # Ensure output directory exists
    os.makedirs("data/output", exist_ok=True)

    cap = cv2.VideoCapture(Config.VIDEO_PATH)
    width = int(cap.get(3))
    height = int(cap.get(4))
    fps_video = int(cap.get(cv2.CAP_PROP_FPS))

    out = cv2.VideoWriter(
        Config.OUTPUT_PATH,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps_video,
        (width, height)
    )

    detector = Detector()
    tracker = Tracker()
    visualizer = Visualizer()
    fps_counter = FPSCounter()

    seen_ids = set()          # For accurate total unique bird count
    frame_count = 0

    print("🚀 BirdTracker-Pro started! Press ESC to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Pre-process frame for better detection
        frame = preprocess_frame(frame)

        # Detection + Tracking
        detections = detector.detect(frame)
        tracked_objects = tracker.update(detections, frame)

        # Update unique count
        for obj in tracked_objects:
            seen_ids.add(obj["id"])

        # Visualization
        frame = visualizer.draw(frame, tracked_objects)

        # Draw FPS and Total Count
        if Config.SHOW_FPS:
            frame = draw_fps(frame, fps_counter.update())
        if Config.SHOW_TOTAL_COUNT:
            frame = draw_total_count(frame, len(seen_ids))

        # Save and display
        out.write(frame)
        cv2.imshow("BirdTracker-Pro - Real-Time Bird Detection", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC key
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"\n🎉 Processing complete!")
    print(f"   📁 Output saved: {Config.OUTPUT_PATH}")
    print(f"   🐦 Total unique birds tracked: {len(seen_ids)}")
    print(f"   📊 Total frames processed: {frame_count}")


if __name__ == "__main__":
    main()