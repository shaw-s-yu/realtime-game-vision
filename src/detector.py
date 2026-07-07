"""Ultralytics YOLO detector + tracker wrapper."""

import logging
import warnings
from typing import List, Dict, Any
import numpy as np

# The "'half' is deprecated" spam is emitted by Ultralytics' own logger (not the
# python `warnings` module) once per predict() call — silence it at the source.
warnings.filterwarnings("ignore", message=r".*'half' is deprecated.*")
logging.getLogger("ultralytics").setLevel(logging.ERROR)

try:
    from ultralytics import YOLO

    ULTRALYTICS_AVAILABLE = True
except Exception:
    ULTRALYTICS_AVAILABLE = False


class DetectorTracker:
    def __init__(
        self,
        model_path="yolov8s-worldv2.pt",
        classes=None,
        conf=0.25,
        iou=0.45,
        device="cuda",
        half=True,
        max_det=100,
        tracker="bytetrack.yaml",
        track_buffer=30,
    ):
        if not ULTRALYTICS_AVAILABLE:
            raise RuntimeError("ultralytics not installed. pip install ultralytics")
        # Validate CUDA availability once at init to avoid per-frame "Invalid CUDA device" spam seen in UI logs.
        # Root cause observed: torch.cuda.is_available() True in standalone check_gpu.py but False inside PySide6 UI process due to Qt DLL load order.
        # Workaround applied in ui_app.py early torch import before PySide6, but double-check here and fallback gracefully with clear single warning.
        try:
            import torch

            cuda_available = torch.cuda.is_available()
            cuda_count = torch.cuda.device_count() if cuda_available else 0
            req = str(device).lower()
            if req in ("cuda", "0", "cuda:0", "cuda0"):
                if not cuda_available or cuda_count == 0:
                    print(
                        '[detector] WARNING: CUDA requested but torch.cuda.is_available() is False inside UI process. Falling back to CPU - FPS will drop to <1. Fix: ensure ui_app.py imports torch before PySide6 (already in repo after fix), check nvidia-smi driver >=528, ensure CUDA_VISIBLE_DEVICES not empty string, reinstall torch with --index-url https://download.pytorch.org/whl/cu121 --force-reinstall, launch UI from activated venv PowerShell not pythonw, and verify python scripts/check_gpu.py shows True outside UI then same python -c "import torch; from PySide6 import QtWidgets; print(torch.cuda.is_available())" should also show True after fix.'
                    )
                    device = "cpu"
                    half = False
                else:
                    device = 0 if req in ("cuda", "0") else device
        except Exception as e:
            print(f"[detector] CUDA check failed, falling back to cpu: {e}")
            device = "cpu"
            half = False

        self.model = YOLO(model_path)
        self.classes = classes or []
        self.conf = conf
        self.iou = iou
        self.device = device
        self.half = half
        self.max_det = max_det
        self.tracker_cfg = tracker
        # set classes for YOLO-World open vocab if provided
        if classes and "world" in model_path.lower():
            try:
                self.model.set_classes(classes)
                print(f"[detector] YOLO-World classes set: {classes}")
            except Exception as e:
                print(f"[detector] set_classes failed, using default COCO: {e}")

        # warmup
        print(f"[detector] loading {model_path} on {device} ...")

    def predict_track(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Run detection + tracking, return list of dicts with id, cls, conf, xyxy, xywh normalized."""
        h, w = frame.shape[:2]
        results = self.model.track(
            source=frame,
            persist=True,
            tracker=self.tracker_cfg,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            half=self.half,
            verbose=False,
            max_det=self.max_det,
        )
        out = []
        if not results or len(results) == 0:
            return out
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return out
        boxes = r.boxes
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy() if boxes.conf is not None else [1.0] * len(xyxy)
        cls_ids = (
            boxes.cls.cpu().numpy().astype(int)
            if boxes.cls is not None
            else [0] * len(xyxy)
        )
        ids = (
            boxes.id.cpu().numpy().astype(int)
            if boxes.id is not None
            else [-1] * len(xyxy)
        )
        names = r.names

        for i in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[i]
            track_id = int(ids[i]) if i < len(ids) else -1
            cls_id = int(cls_ids[i])
            cls_name = (
                names.get(cls_id, str(cls_id))
                if isinstance(names, dict)
                else str(cls_id)
            )
            # filter by classes list if provided and not world model (world already filtered)
            if (
                self.classes
                and cls_name not in self.classes
                and "world" not in str(self.model.ckpt_path).lower()
            ):
                # allow partial match
                if not any(
                    c.lower() in cls_name.lower() or cls_name.lower() in c.lower()
                    for c in self.classes
                ):
                    continue
            out.append(
                {
                    "track_id": track_id,
                    "cls_id": cls_id,
                    "cls_name": cls_name,
                    "conf": float(conf[i]),
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2),
                    "cx": (x1 + x2) / 2 / w,
                    "cy": (y1 + y2) / 2 / h,
                    "w_norm": (x2 - x1) / w,
                    "h_norm": (y2 - y1) / h,
                }
            )
        return out
