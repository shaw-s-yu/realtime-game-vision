# Realtime Game Vision - Windows Troubleshooting Knowledge Base

Captured from real debugging session on RTX 4070 Windows 11 + Python 3.11 venv, PySide6 UI embedding VisionEngine in-process thread.

## Symptom 1: torch CUDA True in standalone check_gpu.py but False inside UI process

**Log pattern in UI Screen tab log pane:**
```
[GPU] torch CUDA False - using CPU
[VisionEngine] detection error: Invalid CUDA 'device=0' requested...
torch.cuda.is_available(): False
```

But standalone terminal shows:
```
python scripts\check_gpu.py
[torch] cuda available: True device NVIDIA GeForce RTX 4070
```

**Root cause:** PySide6 Qt platform plugin DLL loading order interference with PyTorch CUDA initialization on Windows. When QtWidgets QApplication imports Qt6 DLLs first, later torch import inside VisionEngine thread fails to initialize CUDA context. This is Windows-specific Qt vs CUDA DLL hell.

**Fix applied in repo:**
- src/ui_app.py top imports torch before PySide6 and forces early CUDA init with torch.zeros(1, device="cuda") while no Qt event loop running.
- Sets os.environ CUDA_MODULE_LOADING=LAZY to reduce DLL conflict.
- Cleans empty string CUDA_VISIBLE_DEVICES which means no GPU visible.
- src/detector.py checks CUDA once at init and falls back gracefully with single warning not per-frame spam.

**Verification:**
```powershell
python -c "import torch; print('standalone', torch.cuda.is_available())"
python -c "from PySide6 import QtWidgets; import torch; print('qt first', torch.cuda.is_available())"
python -c "import torch; from PySide6 import QtWidgets; print('torch first', torch.cuda.is_available())"
```
Expect True, likely False, True. Third pattern is what fixed ui_app.py does.

## Symptom 2: onnxruntime providers only CPU, no CUDAExecutionProvider, FPS <1

**Log:** `[GPU] onnxruntime providers: ['AzureExecutionProvider','CPUExecutionProvider']`

**Root cause:** onnxruntime-gpu not installed correctly, or CPU onnxruntime package shadows GPU wheel. Pip treats onnxruntime and onnxruntime-gpu as different names but they provide same module.

**Fix:**
```powershell
pip uninstall -y onnxruntime onnxruntime-gpu onnxruntime-azure rapidocr-onnxruntime -q
pip install onnxruntime-gpu==1.18.1 --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/ --force-reinstall --no-cache-dir --no-deps
pip install coloredlogs flatbuffers "protobuf>=4.25" packaging humanfriendly pyreadline3 "mpmath>=1.1" -q
pip install "numpy>=1.26,<2" "sympy==1.13.1" --force-reinstall --no-deps
pip install rapidocr-onnxruntime --no-deps
pip install pyclipper shapely pypdfium2 -q
pip install -r requirements.txt --no-deps
pip install pillow pyyaml tqdm psutil requests einops timm accelerate transformers ollama PySide6 -q
python scripts\check_gpu.py
```
Expect providers include Tensorrt, CUDA, CPU in that order.

**Pip false positive:** `rapidocr-onnxruntime requires onnxruntime>=1.7.0 which is not installed` is expected false positive because you have onnxruntime-gpu not onnxruntime package name. Safe to ignore if runtime import works and check_gpu shows CUDA provider.

## Symptom 3: pip dependency resolver warnings about sympy version

**Pattern:** `torch 2.5.1+cu121 requires sympy==1.13.1 but you have sympy 1.14.0`

**Fix:** Pin sympy explicitly after any pip install:
```powershell
pip install "sympy==1.13.1" --force-reinstall --no-deps
pip check
```
Repo requirements.txt now pins sympy==1.13.1 to prevent drift. For sympy 1.14 features upgrade torch to >=2.7.

## Symptom 4: Detector init failed ultralytics not installed

**Log:** `[VisionEngine] Detector init failed: ultralytics not installed.`

**Root cause:** ultralytics package missing or installed with --no-deps but its dependencies missing causing import failure.

**Fix:**
```powershell
pip install "ultralytics>=8.3,<9" --no-deps
pip install "defusedxml>=0.7.1" "matplotlib>=3.6" "pydeprecate>=0.9,<0.10" "scipy>=1.10" "polars>=0.20" "ultralytics-thop>=2.0.18" "nvidia-ml-py>=12.0"
pip install "numpy>=1.26,<2" "sympy==1.13.1" "opencv-python>=4.9,<4.12" --force-reinstall --no-deps
pip install "torch>=2.2,<2.7" --index-url https://download.pytorch.org/whl/cu121 --force-reinstall --no-deps
python -c "from ultralytics import YOLO; print('ok')"
```

## Symptom 5: Region selector only allows drag on small rectangle covering text

**Root cause:** Old version used small QLabel centered top with WA_TransparentForMouseEvents which doesn't work reliably on some Windows managers.

**Fix applied:** src/region_selector.py rewritten to QWidget fullscreen transparent overlay with no QLabel child, instructional banner drawn directly in paintEvent full width top 70px, mousePressEvent accepts left button anywhere on widget to start drag. Main UI hides temporarily during selection for clean view then restores and auto-switches to Screen tab.

## Symptom 6: FPS lower than 1 per second after GPU fixed

Even after torch CUDA True and onnxruntime CUDA, FPS can be low due to config.

**Checklist for RTX 4070 target 10-20 fps:**
- process_fps 15 or 20 not 1
- output_width 960 not 1920
- detector.model yolov8s-worldv2.pt or yolo11n.pt, avoid M/L
- detector.conf 0.3, max_det 50
- detector.device cuda
- ocr.roi_only true checked
- overlay.show_trails false for test
- vlm.enabled false for baseline
- Use region crop on Start to select tight area not full 4K desktop
- Check Task Manager GPU CUDA graph 30-60% not 0%
- Compare headless `python -m src.main` vs UI embedded to isolate Qt rendering overhead. If headless 15+ fps but UI <3, optimize VideoWidget to skip frames or lower width to 640.

## Symptom 7: git fetch bad object error

**Error:** `fatal: bad object refs/heads/main` `did not send all necessary objects`

**Fix:**
```powershell
git remote prune origin
Remove-Item -Force .git\refs\remotes\origin\main -ErrorAction SilentlyContinue
git fetch origin main --prune --no-tags --depth=1
git fetch origin main --unshallow
git reset --hard origin/main
git clean -fd
# if still fails, fresh clone to new folder then optionally copy .venv to save time
```

## Symptom 8: pip install -q prints nothing on second run

Expected with -q quiet flag when packages already satisfied. Remove -q to see "Requirement already satisfied" lines. Not an error.

## Full fresh setup checklist

See README Quick Start and scripts/setup_windows.ps1, then apply pip sequence from Symptom 2 and 3 and 4 above in order to avoid sympy/numpy drift, then `python scripts/check_gpu.py` must show torch cuda True and onnx CUDA provider, then `python -m src.ui_app`, click Start, choose Select Region, drag anywhere on fullscreen overlay, release, UI auto switches to Screen tab showing embedded live video with detections and log pane below, no separate process spawned.
