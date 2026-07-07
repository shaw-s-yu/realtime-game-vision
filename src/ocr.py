"""RapidOCR wrapper with ROI filtering and text change detection."""

from typing import List, Dict, Tuple
import numpy as np
import cv2

try:
    from rapidocr_onnxruntime import RapidOCR

    RAPID_AVAILABLE = True
except Exception:
    RAPID_AVAILABLE = False


class OCRProcessor:
    def __init__(
        self,
        enabled=True,
        lang="en",
        det_thresh=0.3,
        rec_thresh=0.5,
        use_gpu=True,
        roi_only=True,
        text_classes=None,
        diff_threshold=0.6,
    ):
        self.enabled = enabled and RAPID_AVAILABLE
        self.roi_only = roi_only
        self.text_classes = set(
            [
                c.lower()
                for c in (
                    text_classes or ["notice", "text", "dialog", "button", "menu"]
                )
            ]
        )
        self.diff_threshold = diff_threshold
        self.prev_texts = set()
        self.lang = lang
        self.use_gpu = use_gpu
        if self.enabled:
            try:
                # rapidocr-onnxruntime ignores use_gpu/device/use_cuda kwargs in most
                # versions and constructs its internal ONNX sessions with
                # CPUExecutionProvider only. To force GPU without reaching into version-
                # specific internals, monkey-patch ort.InferenceSession before RapidOCR
                # init so every session it creates gets CUDA providers by default.
                import onnxruntime as ort

                _orig_session_cls = ort.InferenceSession
                _cuda_providers = [
                    "CUDAExecutionProvider",
                    "CPUExecutionProvider",
                ]

                def _forced_cuda_session(*args, **kwargs):
                    prov = kwargs.get("providers")
                    if (
                        not prov
                        or "CUDAExecutionProvider" not in prov
                        and "TensorrtExecutionProvider" not in prov
                    ):
                        kwargs["providers"] = _cuda_providers
                    return _orig_session_cls(*args, **kwargs)

                if use_gpu and "CUDAExecutionProvider" in ort.get_available_providers():
                    ort.InferenceSession = _forced_cuda_session

                try:
                    ocr_obj = None
                    last_err = None
                    candidates = [
                        {"lang": lang, "device": "cuda" if use_gpu else "cpu"},
                        {"lang": lang, "use_cuda": use_gpu},
                        {"lang": lang, "use_gpu": use_gpu},
                        {"lang": lang},
                        {"device": "cuda" if use_gpu else "cpu"},
                        {"use_cuda": use_gpu},
                        {},
                    ]
                    for kw in candidates:
                        try:
                            ocr_obj = RapidOCR(**kw)
                            break
                        except TypeError as te:
                            last_err = te
                            continue
                        except Exception as e:
                            last_err = e
                            continue
                    if ocr_obj is None:
                        raise RuntimeError(
                            f"RapidOCR init failed with all signatures, last error: {last_err}"
                        )
                finally:
                    # Always restore, even on failure.
                    ort.InferenceSession = _orig_session_cls

                self.ocr = ocr_obj
                print(f"[ocr] RapidOCR initialized lang={lang} use_gpu={use_gpu}")

                # Verify which providers RapidOCR's internal sessions actually
                # ended up with — walk the object graph looking for any
                # onnxruntime.InferenceSession instance.
                if use_gpu:
                    try:
                        found = []
                        seen = set()

                        def _walk(obj, path, depth=0):
                            if depth > 4 or id(obj) in seen:
                                return
                            seen.add(id(obj))
                            if isinstance(obj, _orig_session_cls):
                                found.append((path, obj.get_providers()[0]))
                                return
                            for name in dir(obj):
                                if name.startswith("__"):
                                    continue
                                try:
                                    child = getattr(obj, name)
                                except Exception:
                                    continue
                                if callable(child) and not hasattr(child, "__dict__"):
                                    continue
                                if isinstance(
                                    child, (str, int, float, bool, bytes, type(None))
                                ):
                                    continue
                                _walk(child, f"{path}.{name}", depth + 1)

                        _walk(self.ocr, "ocr")
                        if found:
                            summary = ", ".join(
                                f"{p.split('.')[-1]}={ep.replace('ExecutionProvider','')}"
                                for p, ep in found
                            )
                            print(f"[ocr] session providers after init: {summary}")
                            on_cpu = [
                                p for p, ep in found if ep == "CPUExecutionProvider"
                            ]
                            if on_cpu:
                                print(
                                    f"[ocr] WARNING {len(on_cpu)} session(s) still on CPU: {on_cpu}"
                                )
                        else:
                            print(
                                "[ocr] WARNING no InferenceSession objects found via walk"
                            )
                    except Exception as e:
                        print(f"[ocr] provider verify failed: {e}")

                # Log which onnx providers RapidOCR actually ended up using
                try:
                    import onnxruntime as ort

                    prov = ort.get_available_providers()
                    print(f"[ocr] onnxruntime available providers: {prov}")
                    # Try to inspect internal sessions if accessible
                    sess_providers = []
                    for attr in [
                        "text_det",
                        "text_rec",
                        "text_cls",
                        "det",
                        "rec",
                        "cls",
                    ]:
                        try:
                            obj = getattr(self.ocr, attr, None)
                            if obj and hasattr(obj, "session"):
                                sess_providers.append(obj.session.get_providers())
                            elif (
                                obj
                                and hasattr(obj, "rec")
                                and hasattr(obj.rec, "session")
                            ):
                                sess_providers.append(obj.rec.session.get_providers())
                        except Exception:
                            pass
                    if sess_providers:
                        print(
                            f"[ocr] RapidOCR sessions providers sample: {sess_providers[0]}"
                        )
                        if use_gpu and all(
                            "CUDAExecutionProvider" not in p
                            and "TensorrtExecutionProvider" not in p
                            for p in sess_providers
                        ):
                            print(
                                "[ocr] WARNING: RapidOCR sessions are on CPU despite use_gpu=True. Check onnxruntime-gpu install and CUDA driver. See README troubleshooting."
                            )
                except Exception:
                    pass

            except Exception as e:
                print(f"[ocr] init failed: {e}")
                self.enabled = False
        else:
            print("[ocr] disabled or rapidocr not available")

    def _iou(self, a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        inter = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(
            0, min(ay2, by2) - max(ay1, by1)
        )
        union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter + 1e-6
        return inter / union

    def process(self, frame: np.ndarray, detections: List[Dict]) -> Dict:
        """Return dict with texts list and new_notices list."""
        if not self.enabled:
            return {"texts": [], "new_notices": [], "changed": False}

        h, w = frame.shape[:2]
        rois = []
        if self.roi_only and detections:
            for d in detections:
                if (
                    d["cls_name"].lower() in self.text_classes
                    or "text" in d["cls_name"].lower()
                ):
                    # expand a bit
                    pad = 4
                    x1 = max(0, d["x1"] - pad)
                    y1 = max(0, d["y1"] - pad)
                    x2 = min(w, d["x2"] + pad)
                    y2 = min(h, d["y2"] + pad)
                    rois.append((x1, y1, x2, y2))
            # merge overlapping rois simple
            if not rois:
                rois = [(0, 0, w, h)]
        else:
            rois = [(0, 0, w, h)]

        texts = []
        for x1, y1, x2, y2 in rois:
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            try:
                result, _ = self.ocr(crop)
                if result:
                    for item in result:
                        # rapidocr returns [box, text, score] or similar
                        if len(item) >= 2:
                            box, txt = item[0], item[1]
                            score = item[2] if len(item) > 2 else 1.0
                            # adjust box to full frame coords
                            if isinstance(box, (list, np.ndarray)) and len(box) == 4:
                                # box is 4 points or rect; simplify to bbox
                                try:
                                    xs = [p[0] for p in box]
                                    ys = [p[1] for p in box]
                                    bx1, by1, bx2, by2 = (
                                        min(xs) + x1,
                                        min(ys) + y1,
                                        max(xs) + x1,
                                        max(ys) + y1,
                                    )
                                except:
                                    bx1, by1, bx2, by2 = x1, y1, x2, y2
                            else:
                                bx1, by1, bx2, by2 = x1, y1, x2, y2
                            texts.append(
                                {
                                    "text": str(txt).strip(),
                                    "score": float(score),
                                    "x1": int(bx1),
                                    "y1": int(by1),
                                    "x2": int(bx2),
                                    "y2": int(by2),
                                }
                            )
            except Exception as e:
                # silent fail per ROI
                continue

        # deduplicate by text content
        uniq_texts = {}
        for t in texts:
            key = t["text"].lower()
            if key not in uniq_texts or t["score"] > uniq_texts[key]["score"]:
                uniq_texts[key] = t
        texts = list(uniq_texts.values())

        curr_set = set(t["text"].lower() for t in texts)
        new_notices = list(curr_set - self.prev_texts)
        # simple change detection: jaccard distance
        inter = len(curr_set & self.prev_texts)
        union = len(curr_set | self.prev_texts) + 1e-6
        changed = (1 - inter / union) > self.diff_threshold
        self.prev_texts = curr_set

        return {"texts": texts, "new_notices": new_notices, "changed": changed}
