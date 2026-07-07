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

## Symptom 6a: config.runtime.yaml has `detector.device: cpu` even after edits in UI

**Log pattern:** `[detector] loading yolov8s-worldv2.pt on cpu ...` (device string
"cpu" instead of "0"), FPS ~0.5 despite torch/ORT both reporting CUDA available.

**Root cause:** the UI's ComboBox for `detector.device` on the All tab isn't
binding writes back to the in-memory config dict on some setups. Start still
snapshots the stale value and writes `device: cpu` into `config.runtime.yaml`
each time, so every UI edit round-trip silently reverts.

**Fix — edit `config.yaml` directly, close and relaunch the UI:**
```yaml
detector:
  device: cuda
  half: true
```
Then `type config.runtime.yaml` after clicking Start to verify `device: cuda`
was written. The UI reads `config.yaml` at process start so the fresh value
propagates to future Start clicks.

## Symptom 6b: FPS caps at ~2 on text-heavy scenes (Chinese login screens, dialog boxes)

**Log pattern in `[perf]` line:** `overlay=400+ms` while `det=30-60 ms`.

**Root cause:** overlay was doing a full-image `cv2.cvtColor(BGR->RGB) +
Image.fromarray + np.array(pil) + cv2.cvtColor(RGB->BGR)` for every Unicode
label drawn. 30 OCR text boxes = 30 full-image round trips per frame.

**Fix already applied in repo:** `src/overlay.py` collects every Unicode label
(OCR text + notices + VLM caption) into one `pil_items` list and calls
`_draw_texts_pil_batch` once per frame. Overlay drops from ~450 ms to <35 ms.

If you customize `overlay.py`, keep the batched-draw invariant: no per-label
`_draw_text_pil` calls inside the draw loop. Add labels to `pil_items` and
draw them all at the end.

## Symptom 6c: FPS still low after fixing overlay — OCR blocking the main loop

**Log pattern:** `ocr=1500-5000 ms` on `[perf]` line, mostly on Chinese scenes.

**Root cause:** OCR was called synchronously in the main loop. RapidOCR
processes one inference per ROI crop; on 30+ small CJK crops per frame that
compounds to seconds per call, stalling the entire pipeline.

**Fix already applied in repo:** `src/ocr.py` runs RapidOCR in a background
worker thread. `OCRProcessor.submit(frame, detections)` drops any pending job
and enqueues the freshest frame; `get_latest()` returns the last completed
result immediately. The main loop calls both every frame and never blocks.
OCR results lag reality by however long RapidOCR takes (typically 1-3 s on
dense CJK text) — this is fine for text overlay which changes slowly.

If you need synchronous OCR for some reason, `OCRProcessor.process()` is a
compat shim that submits + returns latest, but do not expect it to be
"fresh-for-this-frame".

## Symptom 6d: FPS drops *further* when forcing GPU on RapidOCR sessions

**Log pattern:** `[ocr] session providers after init: session=CUDA, session=CUDA, session=CUDA`
and FPS drops from ~2 (CPU OCR) to ~0.3.

**Root cause:** RapidOCR issues one ONNX inference per ROI crop. On 30+ small
crops per call, per-crop CPU<->GPU sync and cuDNN convolution algorithm
autotune (called for every unique crop shape) dominate the kernel cost.
`onnxruntime-gpu` is only faster for large, batched, static-shape inputs.

**Fix:** keep OCR on CPU. `OCRProcessor(use_gpu=False)` is the default and the
recommended setting for this pipeline. The async worker makes CPU OCR
effectively free from the main loop's perspective.

If you want to experiment with GPU OCR anyway (e.g. batched, single large ROI
workloads), monkey-patching `InferenceSession` on every already-imported
`rapidocr_onnxruntime.*` submodule before calling `RapidOCR()` is the pattern
that works (see git history around `Force RapidOCR sessions onto CUDA`) —
patching only `onnxruntime.InferenceSession` doesn't intercept the
`from onnxruntime import InferenceSession` binding that rapidocr uses
internally.

## Symptom 6e: Ultralytics "'half' is deprecated" floods stdout, one line per predict call

**Log pattern:** dozens of `WARNING 'half' is deprecated and will be removed in the future. Use 'quantize' instead.` lines.

**Root cause:** the message is emitted by Ultralytics' own logger (not the
python `warnings` module). Ultralytics resets the logger level during `YOLO()`
init, so `logging.getLogger("ultralytics").setLevel(ERROR)` gets clobbered.

**Fix already applied in repo:** `src/detector.py` attaches a `logging.Filter`
(runs per-record and survives level resets) to the `ultralytics`, `yolo`, and
root loggers to drop the specific message. `warnings.filterwarnings` is kept
as a belt-and-braces backup.

## Symptom 6f: nvidia-smi shows 8% GPU util and 12W draw while app is running — is GPU really being used?

Both can be true at 10 FPS with a small YOLO model:
- Each inference bursts to 40-80% util for ~30 ms
- Between bursts (100 ms budget - 30 ms = 70 ms idle), GPU sleeps to P8
- `nvidia-smi` sampling often lands in the idle gap → shows near-zero util

**Better diagnostics:**
- Look at the `[perf]` line in the Screen tab log — `det` in ms is the truth
- `nvidia-smi -l 1` for continuous sampling, look at peaks not mean
- The `[detector]` startup line prints `on 0 ...` for cuda:0 or `on cpu ...`;
  device 0 is unambiguous proof YOLO chose GPU

**Also watch out for the venv/base-python nvidia-smi display quirk:** on
Windows the process name column may resolve a venv Python to its base install
image path (`AppData\Local\Programs\Python\Python311\python.exe`) even when
the app is genuinely running from `.venv\Scripts\python.exe`. Not a bug in
your app, just how the process name lookup works.

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
