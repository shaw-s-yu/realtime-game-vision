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

**4. Run with live UI panel (recommended for tuning)**
```powershell
.\.venv\Scripts\Activate.ps1
python -m src.main --config config.yaml --ui
```
This launches two windows:
* OpenCV overlay showing detections + OCR + FPS — press `q` to quit, `s` to save screenshot JSON to captures\
* Dear PyGui control panel titled "Realtime Game Vision Control" with sliders for process_fps, conf, iou, checkboxes for OCR, trails, etc. Changes apply live next frame without restart. Click Save to persist to config.yaml.

Without UI panel, classic headless mode still works:
```powershell
.\scripts\run.ps1
# or python -m src.main --config config.yaml --no-ui
```

First run downloads YOLO-World ~40 MB and RapidOCR models ~15 MB automatically. Install Dear PyGui once: it's in requirements.txt already, `pip install dearpygui` if missing.

## Configuration

Edit `config.yaml` directly, or use live UI panel `--ui` flag, or use C# WPF app in `ui-csharp/` folder which edits same yaml file and Python hot-reloads every 0.5s.

```yaml
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

## Live UI Control Panel

Two options, pick one based on your stack preference.

### Option A — Python Dear PyGui panel built-in (recommended, single process, pip only)

Already in repo, no Visual Studio needed.

```powershell
# after venv activate and pip install -r requirements.txt which includes dearpygui
python -m src.main --config config.yaml --ui
```

Opens second window "Realtime Game Vision Control" alongside OpenCV overlay. Sliders update these live without restart:

* capture.process_fps 1-30, capture.target_fps 10-120, output_width 640-1920 (width needs restart to take effect, shows orange warning in UI)
* detector.conf 0.05-0.9, iou, max_det, device cuda/cpu
* ocr.enabled, ocr.lang ch/en/japan/korean, ocr roi only toggle, det/rec thresholds
* overlay.show_trails, show_ocr, show_labels, trail_length
* vlm.enabled, vlm.interval

Changes write through thread-safe ConfigManager and main loop polls every 0.5s. Click "Save to config.yaml" in UI to persist for next run, or "Reload from Disk" if you edited yaml manually.

Implementation: `src/ui_panel.py` uses Dear PyGui, runs in daemon thread, calls `config_manager.update(dotpath, value)` on slider callbacks. `src/main.py` polls `cm.get()` each loop and applies mutable parameters live. Parameters needing model reload are marked "(restart required)" in UI.

Install if missing: `pip install dearpygui`

### Option B — C# WPF Config Editor (separate process, native Windows UI)

For teams preferring C# Windows desktop app style as you mentioned. Located in `ui-csharp/`.

How it works: WPF app edits same `config.yaml` on disk using YamlDotNet. Python app has ConfigManager auto_reload=True polling file mtime every 0.5s, so changes apply live without socket IPC. Simple file-based IPC robust for tuning speed.

Build:
```powershell
cd ui-csharp
# requires .NET 8 SDK: winget install Microsoft.DotNet.SDK.8
dotnet restore
dotnet build -c Release
dotnet run --project RealtimeGameVisionConfig
# or open RealtimeGameVisionConfig.sln in Visual Studio 2022 and F5
```

UI shows sliders, checkboxes, combos matching Python panel layout. On value changed -> writes yaml immediately -> Python picks up next frame. Same restart-required notes apply for model path, output_width, ocr lang.

Choose Python Dear PyGui for quickest start (single pip, single process). Choose C# WPF if you want native Windows look, Visual Studio designer drag-drop extensibility, or to integrate into larger C# console app ecosystem later. Both talk to same config.yaml, you can even run both at once — last write wins.

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
  main.py          orchestration loop 10 fps, --ui flag launches control panel
  config_manager.py thread-safe hot reload yaml manager shared between UI and main loop
  ui_panel.py      Dear PyGui live tuning panel, sliders update config in real time
  capture.py       dxcam wrapper with fallback to mss
  detector.py      Ultralytics YOLO-World / YOLO wrapper
  tracker.py       ByteTrack wrapper via ultralytics persist
  ocr.py           RapidOCR ONNX wrapper with diffing, Chinese support via PIL overlay
  vlm_client.py    Ollama + transformers Florence-2 / Moondream client async
  overlay.py       Supervision annotators + cv2 window + PIL Unicode text rendering
  utils.py         fps meter, config load
config.yaml
requirements.txt
scripts/setup_windows.ps1  setup_windows.bat  run.ps1  run.bat  check_gpu.py
ui-csharp/         optional C# WPF config editor alternative, edits same yaml, Python hot-reloads
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
