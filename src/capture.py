"""DXGI Desktop Duplication capture for Windows via dxcam, fallback to mss."""

import time
import platform
from typing import Optional, Tuple
import numpy as np

try:
    import dxcam  # type: ignore

    DXCAM_AVAILABLE = True
except Exception:
    DXCAM_AVAILABLE = False

try:
    import mss  # type: ignore

    MSS_AVAILABLE = True
except Exception:
    MSS_AVAILABLE = False

import cv2


class ScreenCapture:
    def __init__(
        self,
        target_fps=30,
        region=None,
        monitor=0,
        output_width=1280,
        output_height=None,
    ):
        self.target_fps = target_fps
        self.region = region  # [left, top, w, h] or None
        self.monitor = monitor
        self.output_width = output_width
        self.output_height = output_height
        self.backend = None
        self.cam = None
        self.sct = None
        self._last_frame_time = 0

        if platform.system() == "Windows" and DXCAM_AVAILABLE:
            try:
                self.cam = dxcam.create(output_color="BGR", output_idx=monitor)
                if self.region:
                    l, t, w, h = self.region
                    self.cam.region = (l, t, l + w, t + h)
                self.cam.start(target_fps=target_fps, video_mode=True)
                self.backend = "dxcam"
                print(f"[capture] dxcam started monitor={monitor} region={self.region}")
            except Exception as e:
                print(f"[capture] dxcam failed: {e}, falling back to mss")
                self.cam = None

        if self.backend is None and MSS_AVAILABLE:
            self.sct = mss.mss()
            if self.region:
                l, t, w, h = self.region
                self.mon = {"left": l, "top": t, "width": w, "height": h}
            else:
                mons = self.sct.monitors
                idx = monitor + 1 if monitor + 1 < len(mons) else 1
                self.mon = mons[idx]
            self.backend = "mss"
            print(f"[capture] mss fallback using {self.mon}")

        if self.backend is None:
            raise RuntimeError(
                "No capture backend available. Install dxcam on Windows or mss."
            )

    def read_latest(self) -> Optional[np.ndarray]:
        """Return latest frame BGR uint8 or None if not ready."""
        if self.backend == "dxcam":
            frame = self.cam.get_latest_frame()
            if frame is None:
                return None
        else:  # mss
            now = time.time()
            # throttle mss to ~target_fps to avoid overload
            if now - self._last_frame_time < 1.0 / self.target_fps:
                return None
            self._last_frame_time = now
            sct_img = self.sct.grab(self.mon)
            frame = np.array(sct_img)[:, :, :3]  # BGRA to BGR
            frame = (
                cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                if frame.shape[2] == 4
                else frame
            )

        # resize keeping aspect
        h, w = frame.shape[:2]
        if self.output_width and w != self.output_width:
            if self.output_height:
                nh, nw = self.output_height, self.output_width
            else:
                scale = self.output_width / w
                nw = self.output_width
                nh = int(h * scale)
            frame = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
        return frame

    def stop(self):
        if self.cam:
            try:
                self.cam.stop()
            except Exception:
                pass
        if self.sct:
            try:
                self.sct.close()
            except Exception:
                pass
