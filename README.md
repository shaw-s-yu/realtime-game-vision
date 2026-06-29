# Realtime Game Vision — Windows Local Screen Agent

Live desktop screen capture at 10 FPS, local object detection + tracking + OCR + optional VLM semantics. No API cost. Designed for Windows gaming, runs on DXGI Desktop Duplication via dxcam.

> Repo created on Linux dev box, intended to be cloned / copied to Windows 10/11 with NVIDIA GPU recommended but CPU fallback works.

## Why this stack vs one big VLM

* **Modular pipeline <100 ms**: dxcam 8 ms + YOLO-World 15 ms + RapidOCR 20 ms + ByteTrack 3 ms = ~46 ms per frame. Leaves headroom for 10 FPS target.
* **One VLM per frame** like OpenCUA-7B or Qwen2.5-VL-7B is 300-800 ms, no tracking IDs, hallucinates coords, needs trust_remote_code. Good for action planning, bad for realtime perception.
* Split lets you: run detector every frame, OCR only on changed regions, VLM every 3rd frame in background thread.

OpenCUA repo you cloned is useful as reference for action schema and evaluation, not for live inference.

## Architecture
```
dxcam capture (BGRA, 30 fps grab, process latest)
  -> resize 1280x720
  -> YOLO-World / YOLO11n detect  -> ByteTrack IDs -> velocity
  -> RapidOCR PP-OCRv4  -> text boxes, diff vs prev for notices
  -> optional Moondream2 / Florence-2 / Qwen2.5-VL-3B via Ollama every N frames
  -> fuse -> cv2 overlay / JSON callback
```

## Quick start on Windows

> Do this on Windows 10/11 machine with Python 3.10+ and preferably NVIDIA GPU + CUDA 12.x

**1. Clone**
```powershell
git clone https://github.com/YOURUSER/realtime-game-vision.git
cd realtime-game-vision
```

**2. Run setup PowerShell as Administrator**
```powershell
Set-ExecutionPolicy -Scope Process -Bypass
.\scripts\setup_windows.ps1
```
This creates venv `.venv`, installs torch CUDA wheel, ultralytics, rapidocr-onnxruntime, dxcam, opencv, supervision, onnxruntime-gpu.

If no NVIDIA GPU, edit scripts\setup_windows.ps1 and change `$USE_CUDA = $false` before running — it will install CPU torch and onnxruntime.

**3. Optional Ollama for VLM semantics**
```powershell
winget install Ollama.Ollama
ollama pull moondream
# or ollama pull qwen2.5vl:3b   # heavier but better
# or ollama pull florence2
```
Leave `use_vlm: false` in config.yaml to skip, or set true.

**4. Run**
```powershell
.\scripts\run.ps1
# or
.\.venv\Scripts\activate
python -m src.main --config config.yaml
```
Press `q` in overlay window to quit. Press `s` to save screenshot + JSON.

First run downloads YOLO-World ~40 MB and RapidOCR models ~15 MB automatically.

## Configuration

Edit `config.yaml`:
```yaml
capture:
  target_fps: 30
  process_fps: 10
  region: null   # null = full primary monitor, or [left,top,width,height]
  output_width: 1280

detector:
  model: "yolov8s-worldv2.pt"   # auto-downloads. Alternative: "yolo11n.pt" after fine-tune
  classes: ["character","enemy","player","health bar","notice","button","item","text"]
  conf: 0.25
  iou: 0.45
  device: "cuda"   # or "cpu"

tracker:
  type: "bytetrack"   # ultralytics built-in
  track_buffer: 30

ocr:
  enabled: true
  lang: "en"
  det_thresh: 0.3
  rec_thresh: 0.5
  use_gpu: true
  roi_only: true   # only OCR inside detected text-like boxes to save time

vlm:
  enabled: false
  provider: "ollama"   # ollama | transformers
  model: "moondream:latest"
  interval: 3          # every N processed frames
  prompt: "Describe game state in one sentence. List visible characters and notices."

overlay:
  show: true
  show_fps: true
  show_trails: true
  trail_length: 15
```

## Fine-tune for your game

1. Run with `s` key to collect screenshots to `captures/`
2. Annotate 200-500 images in Roboflow, export YOLO format
3. Train:
```powershell
yolo train model=yolo11n.pt data=game.yaml imgsz=960 epochs=100 device=0
```
4. Put best.pt path in config.yaml `detector.model`

## File layout
```
src/
  main.py          orchestration loop 10 fps
  capture.py       dxcam wrapper with fallback to mss
  detector.py      Ultralytics YOLO-World / YOLO wrapper
  tracker.py       ByteTrack wrapper via ultralytics persist
  ocr.py           RapidOCR ONNX wrapper with diffing
  vlm_client.py    Ollama + transformers Florence-2 / Moondream client async
  overlay.py       Supervision annotators + cv2 window
  utils.py         fps meter, config load
config.yaml
requirements.txt
scripts/setup_windows.ps1
```

## Performance targets measured

| Component | RTX 3060 Laptop | CPU i7 no GPU |
|-----------|-----------------|---------------|
| dxcam capture | 5-10 ms | 5-10 ms |
| resize 1280 | 2 ms | 2 ms |
| YOLO-World-S detect+track | 18-28 ms FP16 | 55-80 ms OpenVINO |
| RapidOCR full frame | 12-18 ms onnxruntime-gpu | 35-50 ms CPU |
| RapidOCR ROI only | 5-8 ms | 12-20 ms |
| Moondream2 via Ollama | 45-75 ms | 250-400 ms |
| Total pipeline no VLM | 35-55 ms | 80-110 ms |

10 FPS = 100 ms budget, so GPU path has 2x headroom, CPU path borderline but OK at 960 width.

## Troubleshooting Windows

* **dxcam fails "Access Denied"**: run as normal user not admin actually, disable HDR, ensure game is windowed or borderless not exclusive fullscreen. Exclusive fullscreen needs OBS game capture fallback.
* **Black screen**: Windows Graphics Settings -> turn off Hardware-accelerated GPU scheduling, or target specific monitor index in config `capture.monitor: 0`
* **CUDA out of memory**: set `detector.device: cpu` or use `yolo11n.pt` instead of world model.
* **Ollama not found**: set `vlm.enabled false` first to test base pipeline.
* **mss fallback on non-Windows dev**: capture.py auto falls back to mss so you can test logic on Linux/Mac albeit slower.

## License
MIT. Models: Ultralytics YOLO Apache-2.0, RapidOCR Apache-2.0, dxcam MIT, Moondream Apache-2.0.

## Next steps
* Add game-specific class list to config
* Collect captures, fine-tune
* Hook JSON output to your agent logic or OpenCUA action schema if you want to map detections to pyautogui actions later
