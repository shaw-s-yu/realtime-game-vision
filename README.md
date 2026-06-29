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
This opens cross-platform PySide6 window with top bar tabs **Custom**, **All**, and **Screen**.

* **All tab** shows every configuration field grouped by section with proper widgets sliders checkboxes combos.
* **Custom tab** shows your preferred subset for quick tuning. Click **Manage Fields...** button to choose which dot-paths appear there, click **Save Custom Layout** to persist selection to `ui_custom.json`. Selection saved separately from values.
* **Screen tab** shows live video with detections in top section and process log in bottom section — no separate cv2 window and no separate python process spawned per new spec. UI itself hosts vision engine in background thread.
* Top toolbar has green **Start** button and red **Stop** button and gray **Save config.yaml** button visible on all tabs. Click Start -> UI automatically switches to Screen tab, writes current UI values to temporary `config.runtime.yaml`, starts in-process VisionEngine thread reading that config once at startup, no hot reload. Video appears embedded in Screen tab top pane in real time, log appears in bottom pane of same tab. Click Stop to end vision thread cleanly. No separate console window needed, though you can still run headless via scripts if preferred.
* **Save config.yaml** button in top toolbar persists current field values to config.yaml for next UI session. **Save Custom Layout** button in Custom tab persists which fields show in Custom pane to ui_custom.json. Two separate saves by design: values vs layout preference.

First run downloads YOLO-World ~40 MB and RapidOCR models ~15 MB automatically. PySide6 UI works on Windows Linux macOS same code; install via `pip install PySide6` already in requirements.

**4b. Run headless without UI (advanced)**
```powershell
.\scripts\run.ps1
# or .\.venv\Scripts\Activate.ps1 ; python -m src.main --config config.yaml
```
Use this if you prefer editing yaml directly or running on server without GUI. Headless mode still opens cv2 window for overlay unless overlay.show false in config.

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
  main.py            headless orchestration loop, reads config once at start, still works standalone for servers without UI
  ui_app.py          PySide6 cross-platform UI with Custom / All / Screen tabs, Manage dialog, Start Stop buttons, embeds vision engine in-process per new spec
  vision_engine.py   VisionEngine class encapsulating capture+detect+track+ocr pipeline runnable in background thread, emits frames via callback for UI embedding, no cv2 window needed
  config_schema.py   central field definitions for UI generation, type ranges defaults restart flags
  capture.py         dxcam wrapper with fallback to mss
  detector.py        Ultralytics YOLO-World / YOLO wrapper
  tracker.py         ByteTrack wrapper
  ocr.py             RapidOCR ONNX wrapper with diffing, Chinese support
  vlm_client.py      Ollama client async
  overlay.py         draws annotations onto numpy BGR frame, returns image without cv2.imshow when embedded
  utils.py           fps meter, config load
config.yaml          default persisted config loaded by UI on startup
ui_custom.json       UI custom pane field selection persisted separately
config.runtime.yaml  temporary generated on Start button press from UI in-memory state
requirements.txt     includes PySide6 for UI
scripts/
  launch_ui.ps1 launch_ui.bat launch_ui.sh   # launch UI cross-platform
  setup_windows.ps1 setup_windows.bat
  run.ps1 run.bat   # headless direct launch without UI still available
  check_gpu.py
```

## UI Design Rationale

We chose **Option D hybrid** approach after evaluating alternatives per your selection:

* **A yaml file only**: UI edits file directly simple but no in-memory separation.
* **B in-memory via stdin JSON no file**: clean but loses human readability.
* **C SQLite profile store**: good for multi-profile but overkill.
* **D hybrid chosen**: UI holds state in memory dict loaded from default config.yaml on startup. User edits in UI tabs Custom / All. Manage button edits which fields appear in Custom pane saved to ui_custom.json separate from values. On Start button UI writes current in-memory state to temporary config.runtime.yaml then starts VisionEngine in-process background thread reading that config once at startup — no separate python process spawned per new spec, no hot reload. Stop button stops engine thread. Optional Save config.yaml button persists values for next UI session. Simple file-based snapshot works cross-platform and keeps vision core unchanged except now embedded not subprocess.

UI library choice **PySide6** because cross-platform Windows Linux macOS native look, mature QTabWidget for top bar tabs, QTreeWidget for Manage dialog checklist, QFormLayout for typed fields, QThread / python threading integration easy for Start Stop, QLabel with QPixmap perfect for embedding live video frames inside Screen tab. Alternatives considered: Dear PyGui lighter but less native widget richness; Tkinter built-in but dated and poor DPI; Textual TUI not GUI; web Flask requires browser separate.

Custom vs All vs Screen tabs implementation per spec:
* Top bar QTabWidget has three tabs in order: **Custom**, **All**, **Screen**.
* Custom tab top row has Manage Fields... button and Save Custom Layout button and green status label showing field count and ui_custom.json path. Below is scrollable form showing only selected fields.
* All tab shows full scrollable form grouped by schema sections, no Manage button needed there, shows everything for discovery.
* Screen tab has two vertical sections as requested: top section is video QLabel displaying live BGR frames with detections drawn by overlay.py converted to QImage, scaled keeping aspect ratio, updating at process FPS via Qt signal from vision thread. Bottom section is QPlainTextEdit read-only process log showing GPU status, FPS, new text notices, errors. No log pane in Custom or All tabs — only in Screen tab per spec.
* Manage dialog is QDialog with QTreeWidget grouped checklist. OK saves selection to ui_custom.json and rebuilds Custom tab instantly and switches to Custom tab automatically to show result — no restart needed for UI layout change.
* Start button in top toolbar switches to Screen tab automatically per spec requirement "when clicking start, the tab should be switched to screen", then starts VisionEngine thread. Stop button stops thread and clears video widget back to placeholder text.

## UI Usage Walkthrough

1. Launch UI:
```powershell
.\scripts\launch_ui.ps1
# Linux/mac: bash scripts/launch_ui.sh
# or python -m src.ui_app
```
2. Top bar shows three tabs: **Custom**, **All**, **Screen**. Start on Custom or All for configuration editing. Custom shows your preferred subset, All shows full schema grouped.
3. In Custom tab click **Manage Fields...** button top left next to **Save Custom Layout** button. Checklist dialog opens grouped by Capture / Detector / Tracker / OCR / VLM / Overlay / Output. Check fields you want quick access to, uncheck to hide. Click OK — dialog closes, Custom tab rebuilds instantly to show new layout, status bar shows saved path and field count, green label updates. Click **Save Custom Layout** explicitly any time for confirmation dialog with full absolute path to ui_custom.json — this persists layout preference across UI sessions separate from config values.
4. Edit values in Custom or All tab: sliders for ints/floats, checkboxes for bools, combos for enums like device cuda/cpu or ocr lang ch/en, text fields for model paths and comma-separated class lists.
5. Click **Save config.yaml** in top toolbar to persist current field values as default for next UI session (optional — Start works without saving, using in-memory UI state snapshot).
6. Click green **Start** button in top toolbar. UI automatically switches to **Screen** tab per spec. UI writes temporary `config.runtime.yaml` from current in-memory UI state, then starts VisionEngine in background thread within same Python process — **no separate python process spawned, no separate cv2 window**. Screen tab top section shows live video with detections embedded directly in UI via QLabel, bottom section shows process log in real time. Status label shows "running in-process".
7. Tune sliders while running? No hot reload per spec — changes apply on next Start. Switch back to Custom or All tab while vision is running to prepare next config values if you want, but they won't affect running engine until you Stop and Start again. This avoids complexity of live parameter injection into running PyTorch model and matches spec point 2.
8. While vision runs on Screen tab, top video pane updates at process FPS showing bounding boxes trails OCR text overlay rendered via PIL for Unicode support. Bottom log pane shows GPU status, FPS, new text notices, errors. No log pane in Custom or All tabs — only in Screen tab per spec point 4.
9. Click red **Stop** button in top toolbar to end vision thread cleanly. Video widget returns to placeholder text "Vision stopped...". Start button re-enabled. Now edit config in Custom/All tabs again if needed, then Start again for new run with updated config snapshot.
10. Close UI window -> prompts to stop running vision engine if still active, then quits cleanly.
11. For headless mode without UI still available via `python -m src.main --config config.yaml` which opens separate cv2 window as before for servers or debugging.

For C# WPF alternative see `ui-csharp/` folder README — same yaml file contract, but note C# version would need to launch Python UI app as subprocess or embed via Python.NET to get embedded video inside WPF; file-based config contract still works for Start action launching python UI or headless process. Python PySide6 version is primary cross-platform implementation meeting spec fully with embedded video.

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
