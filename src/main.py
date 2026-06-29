"""Main orchestration loop for realtime game vision with hot-reload config and optional UI panel."""

import argparse
import time
import json
from pathlib import Path
import cv2
import numpy as np

from .utils import setup_logging, FPSMeter
from .config_manager import ConfigManager
from .capture import ScreenCapture
from .detector import DetectorTracker
from .tracker import MovementTracker
from .ocr import OCRProcessor
from .overlay import Overlay
from .vlm_client import VLMWorker

try:
    from .ui_panel import ControlPanel

    UI_AVAILABLE = True
except Exception:
    UI_AVAILABLE = False


def main():
    parser = argparse.ArgumentParser(
        description="Realtime Game Vision - local screen agent"
    )
    parser.add_argument("--config", default="config.yaml", help="path to config yaml")
    parser.add_argument(
        "--ui",
        action="store_true",
        help="launch Dear PyGui control panel for live tuning",
    )
    parser.add_argument("--no-ui", dest="ui", action="store_false")
    parser.set_defaults(ui=False)
    args = parser.parse_args()

    cm = ConfigManager(args.config, auto_reload=True)
    cfg = cm.get()
    log = setup_logging(cfg.get("output", {}).get("log_level", "INFO"))

    # GPU status check at startup
    try:
        import torch

        cuda_avail = torch.cuda.is_available()
        if cuda_avail:
            dev_count = torch.cuda.device_count()
            dev_name = torch.cuda.get_device_name(0)
            dev_cap = torch.cuda.get_device_capability(0)
            log.info(
                "[GPU] torch CUDA available: True device_count=%d device0=%s capability=%s torch=%s cuda_runtime=%s",
                dev_count,
                dev_name,
                dev_cap,
                torch.__version__,
                torch.version.cuda,
            )
        else:
            log.warning(
                "[GPU] torch CUDA available: False torch=%s — detector will fall back to CPU. Check torch install with CUDA wheel and NVIDIA driver >=528.",
                torch.__version__,
            )
    except Exception as e:
        log.warning("[GPU] torch check failed: %s", e)

    try:
        import onnxruntime as ort

        providers = ort.get_available_providers()
        log.info(
            "[GPU] onnxruntime %s providers available: %s", ort.__version__, providers
        )
        if (
            "CUDAExecutionProvider" not in providers
            and "TensorrtExecutionProvider" not in providers
        ):
            log.warning(
                "[GPU] onnxruntime GPU providers NOT found — OCR will run on CPU. Uninstall onnxruntime and pip install onnxruntime-gpu==1.18.1 with CUDA 12 feed. See README troubleshooting."
            )
        else:
            log.info("[GPU] onnxruntime GPU provider found — OCR should use GPU")
    except Exception as e:
        log.warning("[GPU] onnxruntime check failed: %s", e)

    def get_cfg():
        return cm.get()

    cfg = get_cfg()
    cap_cfg = cfg["capture"]
    det_cfg = cfg["detector"]
    trk_cfg = cfg["tracker"]
    ocr_cfg = cfg["ocr"]
    vlm_cfg = cfg["vlm"]
    over_cfg = cfg["overlay"]
    out_cfg = cfg.get("output", {})
    perf_cfg = cfg.get("performance", {})

    process_fps = cap_cfg.get("process_fps", 10)
    process_interval = 1.0 / process_fps

    log.info("Initializing capture...")
    capture = ScreenCapture(
        target_fps=cap_cfg.get("target_fps", 30),
        region=cap_cfg.get("region"),
        monitor=cap_cfg.get("monitor", 0),
        output_width=cap_cfg.get("output_width", 1280),
        output_height=cap_cfg.get("output_height"),
    )

    log.info("Loading detector %s ...", det_cfg["model"])
    detector = DetectorTracker(
        model_path=det_cfg["model"],
        classes=det_cfg.get("classes"),
        conf=det_cfg.get("conf", 0.25),
        iou=det_cfg.get("iou", 0.45),
        device=det_cfg.get("device", "cuda"),
        half=det_cfg.get("half", True),
        max_det=det_cfg.get("max_det", 100),
        tracker=trk_cfg.get("type", "bytetrack") + ".yaml",
        track_buffer=trk_cfg.get("track_buffer", 30),
    )

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

    overlay = Overlay(
        show=over_cfg.get("show", True),
        show_fps=over_cfg.get("show_fps", True),
        show_trails=over_cfg.get("show_trails", True),
        trail_length=over_cfg.get("trail_length", 15),
        show_labels=over_cfg.get("show_labels", True),
        show_ocr=over_cfg.get("show_ocr", True),
    )

    # Optional UI panel thread
    ui_panel = None
    if args.ui:
        if UI_AVAILABLE:
            ui_panel = ControlPanel(cm)
            ui_panel.start()
            log.info(
                "Control panel UI started — tune parameters live, changes apply next frame. Use --no-ui to disable."
            )
        else:
            log.warning(
                "UI requested but dearpygui not installed. pip install dearpygui"
            )

    fps_meter = FPSMeter()
    save_dir = Path(out_cfg.get("save_dir", "captures"))
    save_dir.mkdir(exist_ok=True)

    log.info(
        "Starting main loop at target process_fps=%s (use --ui flag for live tuning panel, or edit config.yaml and it hot-reloads)",
        process_fps,
    )
    last_process = 0
    frame_idx = 0
    last_cfg_check = 0
    try:
        while True:
            frame = capture.read_latest()
            if frame is None:
                time.sleep(0.001)
                continue

            now = time.time()
            # hot reload config every ~0.5 sec to pick up UI changes or file edits without restart
            if now - last_cfg_check > 0.5:
                last_cfg_check = now
                new_cfg = cm.get()
                # update runtime mutable parameters
                new_process_fps = new_cfg.get("capture", {}).get("process_fps", 10)
                if new_process_fps != process_fps:
                    process_fps = max(1, min(60, new_process_fps))
                    process_interval = 1.0 / process_fps
                    log.info("process_fps updated live to %s", process_fps)
                # update detector thresholds live
                det_conf = new_cfg.get("detector", {}).get("conf", 0.25)
                det_iou = new_cfg.get("detector", {}).get("iou", 0.45)
                det_max = new_cfg.get("detector", {}).get("max_det", 100)
                if hasattr(detector, "model"):
                    # ultralytics allows runtime override via predict kwargs, we store for next call
                    detector.conf = det_conf
                    detector.iou = det_iou
                    detector.max_det = det_max
                # update overlay flags live
                over = new_cfg.get("overlay", {})
                overlay.show_trails = over.get("show_trails", True)
                overlay.show_ocr = over.get("show_ocr", True)
                overlay.show_labels = over.get("show_labels", True)
                overlay.trail_length = over.get("trail_length", 15)
                # update movement trail length
                movement.trail_length = overlay.trail_length
                # update ocr enable flag - full restart of ocr object needed for lang change, simplified to just toggle enabled flag live; lang change requires restart for now
                ocr.enabled = (
                    new_cfg.get("ocr", {}).get("enabled", True) and ocr.enabled
                )  # keep false if init failed
                # update vlm interval live
                vlm.interval = new_cfg.get("vlm", {}).get("interval", 3)

            if now - last_process < process_interval:
                # show lightweight preview without processing occasionally to keep UI responsive
                if overlay.show:
                    overlay.draw(
                        frame, [], {"texts": []}, movement, fps=fps_meter.fps()
                    )
                    key = overlay.wait_key(1)
                    if key == ord("q"):
                        break
                time.sleep(0.001)
                continue

            last_process = now
            frame_idx += 1
            fps_meter.tick()

            # Detector + tracker - use live conf/iou from detector object updated above
            detections = detector.predict_track(frame)
            detections = movement.update(detections, timestamp=now)

            # OCR
            ocr_result = ocr.process(frame, detections)

            # VLM async submit
            vlm.submit(frame, frame_idx)
            vlm_text = vlm.get_latest() if vlm.enabled else ""

            # Overlay
            vis = overlay.draw(
                frame,
                detections,
                ocr_result,
                movement,
                fps=fps_meter.fps(),
                vlm_text=vlm_text,
            )
            key = overlay.wait_key(1)
            if key == ord("q"):
                log.info("Quit requested")
                break
            elif key == ord("s"):
                # save snapshot
                ts = int(time.time() * 1000)
                img_path = save_dir / f"cap_{ts}.jpg"
                json_path = save_dir / f"cap_{ts}.json"
                cv2.imwrite(str(img_path), frame)
                out_json = {
                    "timestamp": ts,
                    "frame_idx": frame_idx,
                    "detections": detections,
                    "ocr": ocr_result,
                    "vlm": vlm_text,
                }
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(out_json, f, indent=2)
                log.info("Saved %s and %s", img_path, json_path)

            # Log new notices
            if ocr_result.get("new_notices"):
                log.info("New text notices: %s", ocr_result["new_notices"])

    except KeyboardInterrupt:
        log.info("Interrupted")
    finally:
        log.info("Shutting down...")
        if ui_panel:
            ui_panel.stop()
        if vlm.enabled:
            vlm.stop()
            vlm.join(timeout=2)
        capture.stop()
        overlay.destroy()


if __name__ == "__main__":
    main()
