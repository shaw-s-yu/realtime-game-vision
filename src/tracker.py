"""Simple movement tracker using history of ByteTrack IDs."""

from collections import defaultdict, deque
from typing import Dict, List
import time


class MovementTracker:
    def __init__(self, trail_length=15):
        self.trail_length = trail_length
        self.history = defaultdict(
            lambda: deque(maxlen=trail_length)
        )  # track_id -> deque of (t, cx, cy)
        self.velocities = {}  # track_id -> (vx, vy)

    def update(self, detections: List[Dict], timestamp=None):
        if timestamp is None:
            timestamp = time.time()
        seen_ids = set()
        for d in detections:
            tid = d.get("track_id", -1)
            if tid < 0:
                continue
            seen_ids.add(tid)
            cx = d["cx"]
            cy = d["cy"]
            hist = self.history[tid]
            hist.append((timestamp, cx, cy))
            if len(hist) >= 2:
                t0, x0, y0 = hist[-2]
                t1, x1, y1 = hist[-1]
                dt = max(t1 - t0, 1e-3)
                vx = (x1 - x0) / dt
                vy = (y1 - y0) / dt
                self.velocities[tid] = (vx, vy)
                d["vx"] = vx
                d["vy"] = vy
                d["speed"] = (vx**2 + vy**2) ** 0.5
            else:
                d["vx"] = 0.0
                d["vy"] = 0.0
                d["speed"] = 0.0
        # cleanup old tracks not seen
        stale = [tid for tid in list(self.history.keys()) if tid not in seen_ids]
        # keep history for a bit for trail drawing, optional cleanup after long time
        return detections

    def get_trail(self, track_id):
        return list(self.history.get(track_id, []))
