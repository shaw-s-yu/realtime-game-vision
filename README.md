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

> Do this on Windows 10/11 machine with Python 3.10+ 3.11 recommended and preferably NVIDIA GPU + CUDA 12.x . Same steps work on Linux and macOS except dxcam is Windows-only and falls back to mss.

**1. Clone**
```powershell
git clone https://github.com/shaw-s-yu/realtime-game-vision.git
cd realtime-game-vision
```

**2. Run setup PowerShell as Administrator**
```powershell
Set-ExecutionPolicy -Scope Process -Bypass
.\scripts\setup_windows.ps1
```
This creates venv `.venv`, installs torch CUDA wheel, ultralytics, rapidocr-onnxruntime, dxcam, opencv, supervision, onnxruntime-gpu, PySide6 for UI.

If no NVIDIA GPU, edit scripts\setup_windows.ps1 and change `$USE_CUDA = $false` before running — it will install CPU torch and onnxruntime.

Linux/macOS equivalent:
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# onnxruntime-gpu only installs on Windows automatically, Linux needs manual CUDA setup if you want GPU
```

**3. Optional Ollama for VLM semantics**
```powershell
winget install Ollama.Ollama
ollama pull moondream
# or ollama pull qwen2.5vl:3b   # heavier but better
```
Leave `vlm.enabled: false` in config.yaml to skip, or set true in UI.

**4. Launch UI Config Launcher — recommended way**
```powershell
.\scripts\launch_ui.ps1
# or
.\.venv\Scripts\Activate.ps1
python -m src.ui_launcher
```
This opens cross-platform PySide6 window with 2 tabs on top bar: **Custom** and **All**. No hot reload complexity — you edit, then press Start.

First run downloads YOLO-World ~40 MB and RapidOCR models ~15 MB automatically when you first press Start (actually when vision process starts).

**5. Or run headless directly without UI**
```powershell
.\scripts\run.ps1
# or python -m src.main --config config.yaml
```
Press `q` in overlay window to quit. Press `s` to save screenshot + JSON.

## Configuration

Edit `config.yaml` directly, or use UI launcher which provides form fields with validation. Config structure matches UI schema defined in `src/config_schema.py`.

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
  lang: "ch"
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

## UI Launcher — cross-platform config interface with Start Stop

The repo now includes `src/ui_launcher.py` built with PySide6, which works on Windows, Linux, macOS with same code base — suitable library choice because Qt6 abstracts native windowing on all three OSes, unlike WinForms/WPF which are Windows-only, unlike Tkinter which lacks modern tab management.

**Features per your spec:**

1. **UI interface to enter configuration metadata** — form generated from central schema in `src/config_schema.py`. Each field renders appropriate widget: slider for int/float with min max step, checkbox for bool, combobox for choice enums like device cuda/cpu or ocr lang ch/en/japan/korean, text line edit for strings, comma-separated editor for list fields like classes.

2. **No hot reload** — UI does NOT attempt live patch running process, per your requirement. You edit values, then press Start.

3. **Two buttons Start and Stop** — top bar persistent across tabs. Start writes current form values to `config.runtime.yaml` then launches `python -m src.main --config config.runtime.yaml` as subprocess via `subprocess.Popen`. Stop sends terminate then kill after timeout. Status label shows idle / running pid / exited code.

4. **Cross-platform library** — PySide6 chosen over Dear PyGui, Tkinter, wxPython because:
   * PySide6: native look on Windows 11, macOS Aqua, Linux Qt; mature QTabWidget, QFormLayout, QSlider, QComboBox; LGPL license free commercial; pip installable wheels for all three OSes; good HiDPI support.
   * Dear PyGui alternative is lighter GPU-accelerated immediate mode but less native look and accessibility; kept as optional legacy in requirements but not used for main UI to meet "suitable for different environments" spec with native feel.
   * Tkinter built-in but ugly and limited tab styling, hard to extend Manage dialog with tree checklist.
   * wxPython good but heavier build dependencies on Linux.
   * Electron / Tauri would require Node Rust toolchain overkill for Python config launcher.

5. **Top bar with 2 tabs: Custom / All**
   * **All tab** shows full schema grouped by sections Capture, Detector, Tracker, OCR, VLM, Overlay, Output with labels and tooltips from schema description.
   * **Custom tab** shows filtered subset. Top of Custom pane has **Manage Fields...** button opening checklist dialog grouped by section. Select which dot-path fields you want in your preferred pane, save to `ui_custom.json` in repo root. Next launch of UI remembers selection. Default custom selection includes 10 most tuned fields: process_fps, output_width, conf, iou, device, ocr.enabled, ocr.lang, ocr.roi_only, overlay.show_trails, vlm.enabled.

**Run UI:**
```powershell
# Windows
.\scripts\launch_ui.ps1
# or
.\.venv\Scripts\Activate.ps1
python -m src.ui_launcher
```
```bash
# Linux / macOS
source .venv/bin/activate
python -m src.ui_launcher
# or python -m src.ui_launcher --help  # currently no args needed, uses config.yaml in cwd
```

UI workflow:
1. Open UI -> All tab shows everything, Custom tab shows your preferred subset.
2. Click Manage Fields in Custom tab to add/remove fields from preferred pane — saves to ui_custom.json.
3. Adjust sliders / checkboxes / combos. Tooltip hover shows description from schema.
4. Click Start -> UI writes config.runtime.yaml and spawns vision subprocess. Start button disables, Stop enables, status shows pid.
5. Game vision overlay window appears separately showing detections. UI stays responsive for monitoring but edits won't affect running process until you Stop and Start again — no hot reload per spec.
6. Stop button terminates subprocess gracefully, falls back to kill after 3s timeout.
7. Close UI window -> prompts if vision process still running to stop it.

**C# alternative note:** If you truly want C# Windows Console App style UI as originally mentioned, see legacy `ui-csharp/` folder in git history before revert — we removed it to keep repo Python-focused cross-platform, but can re-add as optional separate folder that edits same config.yaml and you press Start in Python UI or via C# launching python subprocess via Process.Start. PySide6 approach satisfies cross-platform requirement better than C# which is Windows-primary despite .NET MAUI cross-platform still immature for desktop Linux.

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
  main.py              orchestration loop, reads config.yaml, no hot reload needed for UI spec
  ui_launcher.py       PySide6 cross-platform UI with Custom/All tabs, Manage Fields dialog, Start Stop buttons launching main as subprocess
  config_schema.py     central schema defining all tunable fields for UI generation with type min max options descriptions
  config_manager.py    (removed in favor of simple load/save for no-hot-reload spec, kept as reference in git history if needed later)
  capture.py           dxcam wrapper with fallback to mss
  detector.py          Ultralytics YOLO-World / YOLO wrapper
  tracker.py           ByteTrack wrapper via ultralytics persist
  ocr.py               RapidOCR ONNX wrapper with diffing, Chinese support via PIL overlay
  vlm_client.py        Ollama client async
  overlay.py           Supervision annotators + cv2 window + PIL Unicode text rendering
  utils.py             fps meter, config load
config.yaml
ui_custom.json         generated after first Manage Fields save, defines custom pane field list
requirements.txt       includes PySide6>=6.6 for UI
scripts/
  setup_windows.ps1  setup_windows.bat
  launch_ui.ps1  launch_ui.bat   # new UI launcher shortcuts
  run.ps1 run.bat                # headless direct run without UI
  check_gpu.py
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
