from deep_sort_realtime.deepsort_tracker import DeepSort
from src.utils.config import Config

class Tracker:
    def __init__(self):
        self.tracker = DeepSort(
            max_age=Config.MAX_AGE,
            n_init=Config.N_INIT
        )

    def update(self, detections, frame):
        tracks = self.tracker.update_tracks(detections, frame=frame)

        tracked_objects = []
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            # FIXED: Correct unpacking of to_ltrb() → (left, top, right, bottom)
            l, t, r, b = map(int, track.to_ltrb())
            w = r - l
            h = b - t

            tracked_objects.append({
                "id": track_id,
                "bbox": (l, t, w, h)
            })

        return tracked_objects