"""Ultralytics YOLO detector + tracker wrapper."""

from typing import List, Dict, Any
import numpy as np

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
        # Validate CUDA availability early and fallback to CPU gracefully to avoid repeated per-frame errors in UI log
        # This addresses issue observed in UI logs where torch.cuda.is_available() returns False inside PySide6 process
        # even though check_gpu.py shows True in standalone terminal due to Qt DLL loading order interference.
        # We check here once at init time and adjust device accordingly with clear warning.
        try:
            import torch

            cuda_available = torch.cuda.is_available()
            cuda_count = torch.cuda.device_count() if cuda_available else 0
            # Normalize device string: "cuda" -> "0" or "cpu" fallback, "cuda:0" stays, etc.
            requested = str(device).lower()
            if requested in ("cuda", "0", "cuda:0", "cuda0"):
                if not cuda_available or cuda_count == 0:
                    print(
                        "[detector] WARNING: CUDA requested but torch.cuda.is_available() is False or device_count 0. "
                        "Falling back to CPU. This significantly reduces FPS. "
                        "Common causes on Windows with PySide6 UI: Qt loaded before torch CUDA init. "
                        "Fix attempted in UI by early torch import, but if still failing check: "
                        "1) nvidia-smi shows driver, 2) python -c 'import torch; print(torch.cuda.is_available())' in same venv outside UI returns True, "
                        "3) no CUDA_VISIBLE_DEVICES='' empty string in environment, 4) reinstall torch with --index-url https://download.pytorch.org/whl/cu121 --force-reinstall"
                    )
                    device = "cpu"
                    half = False  # FP16 not supported on CPU
                else:
                    # Normalize to cuda:0 for ultralytics clarity
                    device = 0 if requested in ("cuda", "0") else device
            # else keep user specified cpu or other device as is
        except Exception as e:
            print(f"[detector] CUDA check failed during init, falling back to cpu: {e}")
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
