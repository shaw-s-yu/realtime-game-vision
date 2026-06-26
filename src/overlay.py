"""Overlay drawing using OpenCV and supervision annotators."""

import cv2
import numpy as np
from typing import List, Dict

try:
    import supervision as sv

    SV_AVAILABLE = True
except Exception:
    SV_AVAILABLE = False


class Overlay:
    def __init__(
        self,
        show=True,
        show_fps=True,
        show_trails=True,
        trail_length=15,
        show_labels=True,
        show_ocr=True,
    ):
        self.show = show
        self.show_fps = show_fps
        self.show_trails = show_trails
        self.trail_length = trail_length
        self.show_labels = show_labels
        self.show_ocr = show_ocr
        if SV_AVAILABLE:
            self.box_annotator = sv.BoxAnnotator(thickness=2)
            self.label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5)
            self.trace_annotator = sv.TraceAnnotator(
                thickness=2, trace_length=trail_length
            )
        self.window_name = "Realtime Game Vision - press q to quit, s to save"

    def draw(
        self, frame, detections, ocr_result, movement_tracker, fps=None, vlm_text=""
    ):
        img = frame.copy()
        h, w = img.shape[:2]

        if SV_AVAILABLE and detections:
            # convert to supervision Detections
            xyxy = np.array(
                [[d["x1"], d["y1"], d["x2"], d["y2"]] for d in detections],
                dtype=np.float32,
            )
            conf = np.array([d["conf"] for d in detections], dtype=np.float32)
            class_id = np.array([d["cls_id"] for d in detections], dtype=int)
            tracker_id = np.array(
                [d["track_id"] if d["track_id"] >= 0 else -1 for d in detections],
                dtype=int,
            )
            det_sv = sv.Detections(
                xyxy=xyxy, confidence=conf, class_id=class_id, tracker_id=tracker_id
            )
            labels = (
                [f"{d['cls_name']} {d['track_id']} {d['conf']:.2f}" for d in detections]
                if self.show_labels
                else []
            )
            img = self.box_annotator.annotate(img, det_sv)
            if labels:
                img = self.label_annotator.annotate(img, det_sv, labels)
            if self.show_trails:
                img = self.trace_annotator.annotate(img, det_sv)
        else:
            # fallback cv2 drawing
            for d in detections:
                x1, y1, x2, y2 = d["x1"], d["y1"], d["x2"], d["y2"]
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                if self.show_labels:
                    label = f"{d['cls_name']} {d['track_id']} {d['conf']:.2f} s:{d.get('speed', 0):.2f}"
                    cv2.putText(
                        img,
                        label,
                        (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )

        # OCR boxes
        if self.show_ocr and ocr_result and ocr_result.get("texts"):
            for t in ocr_result["texts"]:
                x1, y1, x2, y2 = t["x1"], t["y1"], t["x2"], t["y2"]
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 180, 0), 1)
                txt = t["text"][:30]
                cv2.putText(
                    img,
                    txt,
                    (x1, y2 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 180, 0),
                    1,
                    cv2.LINE_AA,
                )

        # HUD
        y = 25
        if self.show_fps and fps:
            cv2.putText(
                img,
                f"FPS process: {fps:.1f}",
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
            y += 28
        if ocr_result and ocr_result.get("new_notices"):
            notice = ", ".join(ocr_result["new_notices"][:3])
            cv2.putText(
                img,
                f"NEW TEXT: {notice}",
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 200, 255),
                2,
            )
            y += 24
        if vlm_text:
            # wrap
            for line in vlm_text.split("\n")[:2]:
                cv2.putText(
                    img,
                    f"VLM: {line[:80]}",
                    (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (200, 200, 255),
                    1,
                )
                y += 22
        cv2.putText(
            img,
            f"Detections: {len(detections)}",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (180, 255, 180),
            1,
        )

        if self.show:
            cv2.imshow(self.window_name, img)
        return img

    def wait_key(self, delay=1):
        return cv2.waitKey(delay) & 0xFF

    def destroy(self):
        cv2.destroyAllWindows()
