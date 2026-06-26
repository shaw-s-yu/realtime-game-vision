"""Main orchestration loop for realtime game vision."""

import argparse
import time
import json
from pathlib import Path
import cv2
import numpy as np

from .utils import load_config, setup_logging, FPSMeter
from .capture import ScreenCapture
from .detector import DetectorTracker
from .tracker import MovementTracker
from .ocr import OCRProcessor
from .overlay import Overlay
from .vlm_client import VLMWorker


def main():
    parser = argparse.ArgumentParser(
        description="Realtime Game Vision - local screen agent"
    )
    parser.add_argument("--config", default="config.yaml", help="path to config yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
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
        lang=ocr_cfg.get("lang", "en"),
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

    fps_meter = FPSMeter()
    save_dir = Path(out_cfg.get("save_dir", "captures"))
    save_dir.mkdir(exist_ok=True)

    log.info("Starting main loop at target process_fps=%s", process_fps)
    last_process = 0
    frame_idx = 0
    try:
        while True:
            frame = capture.read_latest()
            if frame is None:
                time.sleep(0.001)
                continue

            now = time.time()
            if now - last_process < process_interval:
                # still show overlay at capture rate but skip heavy processing? we choose to skip to save CPU
                # but for simplicity we just continue loop waiting for next process slot
                if overlay.show:
                    # show lightweight preview without processing occasionally to keep UI responsive
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

            # Detector + tracker
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
        if vlm.enabled:
            vlm.stop()
            vlm.join(timeout=2)
        capture.stop()
        overlay.destroy()


if __name__ == "__main__":
    main()
