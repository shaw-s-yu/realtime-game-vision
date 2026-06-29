"""
VisionEngine encapsulates the realtime screen capture + detection + tracking + OCR pipeline
for embedding inside UI without spawning separate process.
No hot reload needed per spec - engine reads config once at start(), runs until stop().
Emits frames and log messages via callbacks for UI to display in Screen tab.
"""

import time
import json
import threading
from pathlib import Path
import cv2
import numpy as np

from .utils import load_config, FPSMeter
from .capture import ScreenCapture
from .detector import DetectorTracker
from .tracker import MovementTracker
from .ocr import OCRProcessor
from .overlay import Overlay
from .vlm_client import VLMWorker


class VisionEngine:
    def __init__(
        self, config_path="config.yaml", frame_callback=None, log_callback=None
    ):
        """
        frame_callback: callable(img_bgr: np.ndarray) called from engine thread each processed frame - UI must handle thread-safety (use queue or Qt signal).
        log_callback: callable(str) called for log messages.
        """
        self.config_path = Path(config_path)
        self.frame_callback = frame_callback
        self.log_callback = log_callback
        self._stop_event = threading.Event()
        self._thread = None
        self._running = False

    def log(self, msg):
        if self.log_callback:
            try:
                self.log_callback(msg)
            except:
                pass
        else:
            print(msg)

    def start(self):
        if self._running:
            return False
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self._running = True
        return True

    def stop(self, timeout=3):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                # can't force kill thread in Python, but loop checks stop event frequently
                return False
        self._running = False
        return True

    def is_running(self):
        return self._running and self._thread and self._thread.is_alive()

    def _run_loop(self):
        try:
            cfg = load_config(str(self.config_path))
        except Exception as e:
            self.log(f"[VisionEngine] Failed to load config {self.config_path}: {e}")
            return

        cap_cfg = cfg.get("capture", {})
        det_cfg = cfg.get("detector", {})
        trk_cfg = cfg.get("tracker", {})
        ocr_cfg = cfg.get("ocr", {})
        vlm_cfg = cfg.get("vlm", {})
        over_cfg = cfg.get("overlay", {})
        out_cfg = cfg.get("output", {})

        process_fps = cap_cfg.get("process_fps", 10)
        process_interval = 1.0 / max(1, process_fps)

        # GPU status logging similar to main.py
        try:
            import torch

            if torch.cuda.is_available():
                self.log(
                    f"[GPU] torch CUDA available True device {torch.cuda.get_device_name(0)} torch {torch.__version__}"
                )
            else:
                self.log(f"[GPU] torch CUDA False - using CPU")
        except Exception as e:
            self.log(f"[GPU] torch check failed {e}")

        try:
            import onnxruntime as ort

            prov = ort.get_available_providers()
            self.log(f"[GPU] onnxruntime providers: {prov}")
        except Exception as e:
            self.log(f"[GPU] onnxruntime check failed {e}")

        try:
            capture = ScreenCapture(
                target_fps=cap_cfg.get("target_fps", 30),
                region=cap_cfg.get("region"),
                monitor=cap_cfg.get("monitor", 0),
                output_width=cap_cfg.get("output_width", 1280),
                output_height=cap_cfg.get("output_height"),
            )
            self.log("[VisionEngine] Capture initialized")
        except Exception as e:
            self.log(f"[VisionEngine] Capture init failed: {e}")
            return

        try:
            detector = DetectorTracker(
                model_path=det_cfg.get("model", "yolov8s-worldv2.pt"),
                classes=det_cfg.get("classes"),
                conf=det_cfg.get("conf", 0.25),
                iou=det_cfg.get("iou", 0.45),
                device=det_cfg.get("device", "cuda"),
                half=det_cfg.get("half", True),
                max_det=det_cfg.get("max_det", 100),
                tracker=trk_cfg.get("type", "bytetrack") + ".yaml",
                track_buffer=trk_cfg.get("track_buffer", 30),
            )
            self.log(f"[VisionEngine] Detector loaded {det_cfg.get('model')}")
        except Exception as e:
            self.log(f"[VisionEngine] Detector init failed: {e}")
            capture.stop()
            return

        movement = MovementTracker(trail_length=over_cfg.get("trail_length", 15))
        ocr = OCRProcessor(
            enabled=ocr_cfg.get("enabled", True),
            lang=ocr_cfg.get("lang", "ch"),
            det_thresh=ocr_cfg.get("det_thresh", 0.3),
            rec_thresh=ocr_cfg.get("rec_thresh", 0.5),
            use_gpu=ocr_cfg.get("use_gpu", True),
            roi_only=ocr_cfg.get("roi_only", True),
            text_classes=ocr_cfg.get("text_classes"),
            diff_threshold=ocr_cfg.get("diff_threshold", 0.6),
        )

        vlm = VLMWorker(
            enabled=vlm_cfg.get("enabled", False),
            provider=vlm_cfg.get("provider", "ollama"),
            model=vlm_cfg.get("model", "moondream:latest"),
            base_url=vlm_cfg.get("base_url", "http://localhost:11434"),
            prompt=vlm_cfg.get("prompt", ""),
            interval=vlm_cfg.get("interval", 3),
            timeout_ms=vlm_cfg.get("timeout_ms", 2000),
        )
        if vlm.enabled:
            vlm.start()
            self.log("[VisionEngine] VLM worker started")

        overlay = Overlay(
            show=False,  # never show cv2 window when embedded in UI, UI will display returned image
            show_fps=over_cfg.get("show_fps", True),
            show_trails=over_cfg.get("show_trails", True),
            trail_length=over_cfg.get("trail_length", 15),
            show_labels=over_cfg.get("show_labels", True),
            show_ocr=over_cfg.get("show_ocr", True),
        )

        fps_meter = FPSMeter()
        save_dir = Path(out_cfg.get("save_dir", "captures"))
        save_dir.mkdir(exist_ok=True)

        self.log(
            f"[VisionEngine] Starting main loop at target process_fps={process_fps}"
        )
        last_process = 0
        frame_idx = 0

        try:
            while not self._stop_event.is_set():
                frame = capture.read_latest()
                if frame is None:
                    time.sleep(0.001)
                    continue

                now = time.time()
                if now - last_process < process_interval:
                    # even when skipping heavy processing, we can still emit preview frame occasionally for UI responsiveness at lower cost
                    # For simplicity, just sleep; UI will show last annotated frame until next process tick.
                    time.sleep(0.001)
                    continue

                last_process = now
                frame_idx += 1
                fps_meter.tick()

                # Detector + tracker
                try:
                    detections = detector.predict_track(frame)
                    detections = movement.update(detections, timestamp=now)
                except Exception as e:
                    self.log(f"[VisionEngine] detection error: {e}")
                    detections = []

                # OCR
                try:
                    ocr_result = ocr.process(frame, detections)
                except Exception as e:
                    self.log(f"[VisionEngine] OCR error: {e}")
                    ocr_result = {"texts": [], "new_notices": [], "changed": False}

                # VLM async submit
                try:
                    vlm.submit(frame, frame_idx)
                    vlm_text = vlm.get_latest() if vlm.enabled else ""
                except Exception:
                    vlm_text = ""

                # Overlay draw returns annotated BGR image, no cv2.imshow because show=False
                try:
                    vis = overlay.draw(
                        frame,
                        detections,
                        ocr_result,
                        movement,
                        fps=fps_meter.fps(),
                        vlm_text=vlm_text,
                    )
                except Exception as e:
                    self.log(f"[VisionEngine] overlay error: {e}")
                    vis = frame

                # Emit frame to UI callback
                if self.frame_callback:
                    try:
                        # copy to avoid threading issues with numpy mutable buffer being reused by next capture
                        self.frame_callback(vis.copy())
                    except Exception as e:
                        self.log(f"[VisionEngine] frame callback error: {e}")

                # Log new notices
                if ocr_result.get("new_notices"):
                    self.log(
                        f"[VisionEngine] New text notices: {ocr_result['new_notices']}"
                    )

        except Exception as e:
            self.log(f"[VisionEngine] fatal error in loop: {e}")
        finally:
            self.log("[VisionEngine] shutting down...")
            try:
                if vlm.enabled:
                    vlm.stop()
            except:
                pass
            try:
                capture.stop()
            except:
                pass
            self._running = False
            self.log("[VisionEngine] stopped")
