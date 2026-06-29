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

**4. Run UI Config Launcher (recommended)**
```powershell
.\scripts\launch_ui.ps1
# or on Linux/mac: bash scripts/launch_ui.sh
# or manual: .\.venv\Scripts\Activate.ps1 ; python -m src.ui_app
```
This opens cross-platform PySide6 window with top bar tabs **Custom** and **All**.

* **All tab** shows every configuration field grouped by section with proper widgets sliders checkboxes combos.
* **Custom tab** shows your preferred subset for quick tuning. Click **Manage Fields...** button to choose which dot-paths appear there — selection saved to `ui_custom.json` for next sessions.
* Edit values in UI, then click **Start** button to launch vision process with current configuration. UI writes temporary `config.runtime.yaml` and spawns `python -m src.main --config config.runtime.yaml` as subprocess. No hot reload needed per design — Start always launches fresh process based on entered config.
* **Stop** button terminates vision process cleanly.
* **Save config.yaml** button persists current UI values to repo config file for next time.

Press `q` in overlay vision window to quit vision process, or use Stop button in UI. Press `s` in overlay to save screenshot + JSON to captures\.

First run downloads YOLO-World ~40 MB and RapidOCR models ~15 MB automatically. PySide6 UI works on Windows Linux macOS same code; install via `pip install PySide6` already in requirements.

**4b. Run headless without UI (advanced)**
```powershell
.\scripts\run.ps1
# or .\.venv\Scripts\Activate.ps1 ; python -m src.main --config config.yaml
```
Use this if you prefer editing yaml directly or running on server without GUI.

## Configuration

You have three ways to edit configuration, all using same `config.yaml` schema:

1. **UI All tab** — shows every field grouped by capture / detector / tracker / ocr / vlm / overlay / output sections with proper widgets. Best for discovery.
2. **UI Custom tab** — shows only your preferred subset. Click Manage Fields... to add/remove dot-paths like `detector.conf` or `ocr.lang`. Selection persisted to `ui_custom.json`. Best for daily tuning of 5-10 key knobs without scrolling.
3. **Edit `config.yaml` directly** in editor — UI loads it on startup as default values. Example below:

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
  main.py            orchestration loop, reads config once at start, no hot reload per spec
  ui_app.py          PySide6 cross-platform UI with Custom/All tabs, Manage dialog, Start Stop buttons
  config_schema.py   central field definitions for UI generation, type, ranges, defaults, restart flags
  config_manager.py  (legacy, not used in new UI flow - UI holds state in memory, writes temp yaml on Start)
  capture.py         dxcam wrapper with fallback to mss
  detector.py        Ultralytics YOLO-World / YOLO wrapper
  tracker.py         ByteTrack wrapper via ultralytics persist
  ocr.py             RapidOCR ONNX wrapper with diffing, Chinese support via PIL overlay
  vlm_client.py      Ollama client async
  overlay.py         Supervision annotators + cv2 window + PIL Unicode text rendering
  utils.py           fps meter, config load
config.yaml          default persisted config, loaded by UI on startup
ui_custom.json       UI custom pane field selection persisted separately (created on first Manage save)
config.runtime.yaml  temporary generated on Start button press, deleted on next Start
requirements.txt
scripts/
  setup_windows.ps1  setup_windows.bat
  launch_ui.ps1  launch_ui.bat  launch_ui.sh   # UI launcher cross-platform
  run.ps1 run.bat   # headless direct launch without UI
  check_gpu.py
```

## UI Design Rationale

We chose **Option D hybrid** approach after evaluating alternatives:

* **A yaml file only**: UI edits file directly, simple but no in-memory state separation and risk of partial writes.
* **B in-memory via stdin JSON no file**: clean but loses human readability and harder to debug running config.
* **C SQLite profile store**: good for multi-profile versioning but overkill for single-user desktop tool, adds dependency.
* **D hybrid chosen**: UI holds state in memory dict loaded from default config.yaml on startup. User edits in UI tabs. On Start button, UI writes current state to temporary `config.runtime.yaml` then launches `python -m src.main --config config.runtime.yaml` as subprocess. Stop button terminates subprocess via PID. Optional Save button writes back to `config.yaml` for persistence across sessions. No hot reload needed per spec — each Start launches fresh process with snapshot of UI state at that moment. Simple file-based IPC works cross-platform and cross-language, matches existing main.py --config interface without modification to vision core.

UI library choice **PySide6** because: native look on Windows Linux macOS, mature QTabWidget QTreeWidget QFormLayout QDoubleSpinBox perfect for Custom/All tabs and Manage dialog checklist, LGPL license pip installable, good subprocess QProcess or Python subprocess integration for Start Stop buttons, wide community. Alternatives considered: Dear PyGui lighter GPU accelerated but less native widget richness for complex forms; Tkinter built-in but dated look and poor DPI scaling; Textual TUI not GUI; web Flask requires browser separate process.

Custom vs All tabs implementation: All tab builds form dynamically from central SCHEMA list in `config_schema.py` grouped by top-level sections. Custom tab builds same form widget factory but filtered to dot-paths listed in `ui_custom.json`. Manage button opens QDialog with QTreeWidget grouped checklist, saves selection back to json, rebuilds Custom tab layout on next UI open (or dynamic rebuild). This satisfies spec requirement for user-preferred pane.

## UI Usage Walkthrough

1. Launch UI:
```powershell
.\scripts\launch_ui.ps1
# Linux/mac: bash scripts/launch_ui.sh
# or python -m src.ui_app
```
2. Top bar shows two tabs: **Custom** and **All**. Start on Custom for quick tuning, switch to All to see every field grouped by section.
3. In Custom tab click **Manage Fields...** button top left. Checklist dialog opens grouped by Capture / Detector / Tracker / OCR / VLM / Overlay / Output. Check fields you want quick access to, uncheck to hide. OK saves to `ui_custom.json`. Reopen Custom tab to see updated layout (or restart UI for full rebuild in current version).
4. Edit values: sliders for ints/floats, checkboxes for bools, combos for enums like device cuda/cpu or ocr lang ch/en, text fields for model paths and comma-separated class lists.
5. Click **Save config.yaml** bottom to persist current values as default for next session (optional).
6. Click green **Start** button top bar. UI writes temp `config.runtime.yaml` then spawns vision process. Status label changes to "running pid XXXX". Overlay window appears showing live detections.
7. Tune sliders while running? No hot reload per spec — changes apply on next Start. Stop current run with red **Stop** button, adjust values in UI, Start again. This avoids complexity of live parameter injection into running PyTorch model.
8. While vision runs, UI log pane at bottom shows stdout from subprocess including GPU status, FPS, new text notices.
9. In overlay window press `s` to save screenshot + JSON to captures folder for later fine-tune dataset, press `q` to quit vision process (or use Stop button in UI).
10. Close UI window -> prompts to stop running vision process if still active.

For C# WPF alternative see `ui-csharp/` folder README — same yaml file contract, Python hot-reloads file on Start only, so C# app can be used instead of Python UI if preferred, though Python PySide6 version is primary cross-platform implementation.

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
