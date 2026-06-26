"""Quick GPU diagnostic for realtime-game-vision stack on Windows."""

import sys

print("Python:", sys.version)
print("Executable:", sys.executable)
print()

try:
    import torch

    print("[torch] version:", torch.__version__)
    print("[torch] cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("[torch] cuda version:", torch.version.cuda)
        print("[torch] device count:", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            print(
                f"  device {i}:",
                torch.cuda.get_device_name(i),
                torch.cuda.get_device_capability(i),
            )
        print(
            "[torch] cudnn version:",
            torch.backends.cudnn.version()
            if torch.backends.cudnn.is_available()
            else "n/a",
        )
    else:
        print(
            "[torch] WARNING: CUDA not available - check torch install is cu121 wheel not cpu, and NVIDIA driver >=528"
        )
except Exception as e:
    print("[torch] import failed:", e)

print()

try:
    import onnxruntime as ort

    print("[onnxruntime] version:", ort.__version__)
    print("[onnxruntime] get_device():", ort.get_device())
    prov = ort.get_available_providers()
    print("[onnxruntime] available providers:", prov)
    if "CUDAExecutionProvider" in prov:
        print("[onnxruntime] OK: CUDAExecutionProvider found - OCR can use GPU")
    elif "TensorrtExecutionProvider" in prov:
        print("[onnxruntime] OK: TensorRT provider found")
    else:
        print("[onnxruntime] WARNING: only CPUExecutionProvider found.")
        print("  Fix: pip uninstall -y onnxruntime onnxruntime-gpu")
        print(
            "       pip install onnxruntime-gpu==1.18.1 --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/"
        )
        print("       then pip install rapidocr-onnxruntime --no-deps")
except Exception as e:
    print("[onnxruntime] import failed:", e)

print()

try:
    from rapidocr_onnxruntime import RapidOCR
    import inspect

    print("[rapidocr] RapidOCR signature:", inspect.signature(RapidOCR.__init__))
    # try init to see what providers it picks
    try:
        ocr = RapidOCR(lang="ch")
        print("[rapidocr] initialized with lang=ch")
        # try to inspect internal session providers
        for attr in ["text_det", "text_rec", "det", "rec"]:
            obj = getattr(ocr, attr, None)
            if obj and hasattr(obj, "session"):
                print(f"  {attr} session providers:", obj.session.get_providers())
                break
    except Exception as e:
        print("[rapidocr] init test failed:", e)
except Exception as e:
    print("[rapidocr] import failed:", e)

print()
print(
    "Done. If torch cuda True and onnx providers include CUDAExecutionProvider, you're good."
)
print("If not, see README Troubleshooting GPU section.")
