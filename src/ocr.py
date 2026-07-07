"""RapidOCR wrapper with ROI filtering, text change detection, and async worker.

OCR runs in a background thread — the main pipeline loop submits the latest
frame+detections and immediately reads the last completed result. This keeps
FPS bound by the detector, not by RapidOCR (which can take 1–5 seconds per
call on text-heavy Chinese scenes).
"""

import queue
import threading
import time
from typing import List, Dict, Optional
import numpy as np

try:
    from rapidocr_onnxruntime import RapidOCR

    RAPID_AVAILABLE = True
except Exception:
    RAPID_AVAILABLE = False


_EMPTY_RESULT = {"texts": [], "new_notices": [], "changed": False}


class OCRProcessor:
    def __init__(
        self,
        enabled=True,
        lang="en",
        det_thresh=0.3,
        rec_thresh=0.5,
        use_gpu=False,  # GPU forcing consistently slower than CPU on ROI-sized crops
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

        self._latest = dict(_EMPTY_RESULT)
        self._latest_lock = threading.Lock()
        self._q: "queue.Queue" = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._worker: Optional[threading.Thread] = None

        if self.enabled:
            try:
                # Instantiate RapidOCR with whichever kwargs the installed version
                # accepts. We don't force GPU: on typical ROI-crop workloads,
                # per-crop CPU↔GPU sync overhead in onnxruntime-gpu makes GPU
                # slower than CPU for this pipeline.
                ocr_obj = None
                last_err = None
                candidates = [
                    {"lang": lang},
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
                self.ocr = ocr_obj
                print(f"[ocr] RapidOCR initialized lang={lang}")
                self._worker = threading.Thread(
                    target=self._run_worker, daemon=True, name="OCRWorker"
                )
                self._worker.start()
                print("[ocr] async worker started")
            except Exception as e:
                print(f"[ocr] init failed: {e}")
                self.enabled = False
        else:
            print("[ocr] disabled or rapidocr not available")

    def stop(self):
        self._stop.set()

    # --- public API used by vision_engine ---

    def submit(self, frame: np.ndarray, detections: List[Dict]) -> None:
        """Submit latest frame+detections; drops any pending job so worker
        always processes the freshest frame."""
        if not self.enabled:
            return
        if self._q.full():
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
        try:
            self._q.put_nowait((frame.copy(), detections))
        except queue.Full:
            pass

    def get_latest(self) -> Dict:
        """Return the last-computed OCR result. Safe to call every frame."""
        with self._latest_lock:
            latest = self._latest
        # Return a shallow copy so caller can mutate new_notices without racing
        # the worker's next update; blank one-shot signals so they don't re-emit.
        return {
            "texts": latest.get("texts", []),
            "new_notices": [],
            "changed": False,
        }

    def process(self, frame: np.ndarray, detections: List[Dict]) -> Dict:
        """Compatibility shim: sync-style call that submits and returns the
        last completed result. Prefer submit()/get_latest() directly."""
        self.submit(frame, detections)
        return self.get_latest()

    # --- worker loop ---

    def _run_worker(self):
        while not self._stop.is_set():
            try:
                frame, detections = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                result = self._process_sync(frame, detections)
                with self._latest_lock:
                    # Preserve one-shot signals: new_notices/changed live in the
                    # authoritative _latest so vision_engine can consume them
                    # once before we overwrite on the next tick. We hand them
                    # out via _consume_one_shots below.
                    self._latest = result
                self._consume_one_shots(result)
            except Exception as e:
                print(f"[ocr] worker error: {e}")

    def _consume_one_shots(self, result: Dict):
        """Called after each successful OCR pass. Currently a no-op hook —
        vision_engine reads texts from get_latest(); new_notices logging is
        handled by the vision_engine itself when it observes _latest change."""
        # We intentionally don't drain new_notices here so a future frame's
        # get_latest could report them, but for now vision_engine treats
        # get_latest as texts-only and uses its own diffing on displayed texts.
        return

    # --- core OCR (runs on worker thread) ---

    def _process_sync(self, frame: np.ndarray, detections: List[Dict]) -> Dict:
        h, w = frame.shape[:2]
        rois = []
        if self.roi_only and detections:
            for d in detections:
                if (
                    d["cls_name"].lower() in self.text_classes
                    or "text" in d["cls_name"].lower()
                ):
                    pad = 4
                    x1 = max(0, d["x1"] - pad)
                    y1 = max(0, d["y1"] - pad)
                    x2 = min(w, d["x2"] + pad)
                    y2 = min(h, d["y2"] + pad)
                    rois.append((x1, y1, x2, y2))
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
                if not result:
                    continue
                for item in result:
                    if len(item) < 2:
                        continue
                    box, txt = item[0], item[1]
                    score = item[2] if len(item) > 2 else 1.0
                    if isinstance(box, (list, np.ndarray)) and len(box) == 4:
                        try:
                            xs = [p[0] for p in box]
                            ys = [p[1] for p in box]
                            bx1, by1, bx2, by2 = (
                                min(xs) + x1,
                                min(ys) + y1,
                                max(xs) + x1,
                                max(ys) + y1,
                            )
                        except Exception:
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
            except Exception:
                continue

        uniq_texts = {}
        for t in texts:
            key = t["text"].lower()
            if key not in uniq_texts or t["score"] > uniq_texts[key]["score"]:
                uniq_texts[key] = t
        texts = list(uniq_texts.values())

        curr_set = set(t["text"].lower() for t in texts)
        new_notices = list(curr_set - self.prev_texts)
        inter = len(curr_set & self.prev_texts)
        union = len(curr_set | self.prev_texts) + 1e-6
        changed = (1 - inter / union) > self.diff_threshold
        self.prev_texts = curr_set

        return {"texts": texts, "new_notices": new_notices, "changed": changed}
