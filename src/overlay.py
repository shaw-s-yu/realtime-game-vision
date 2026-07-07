"""Overlay drawing using OpenCV and supervision annotators."""

import cv2
import numpy as np
from typing import List, Dict
import os
import platform

try:
    import supervision as sv

    SV_AVAILABLE = True
except Exception:
    SV_AVAILABLE = False

# PIL for Unicode text rendering (Chinese, Japanese, Korean etc.)
try:
    from PIL import Image, ImageDraw, ImageFont

    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False


def _find_cjk_font():
    """Find a CJK capable TTF/OTF/TTC font on Windows, macOS, Linux in order."""
    candidates = []
    if platform.system() == "Windows":
        win_dir = os.environ.get("WINDIR", r"C:\Windows")
        candidates += [
            os.path.join(
                win_dir, "Fonts", "msyh.ttc"
            ),  # Microsoft YaHei UI - best for Simplified Chinese
            os.path.join(win_dir, "Fonts", "msyhbd.ttc"),
            os.path.join(win_dir, "Fonts", "msyhl.ttc"),
            os.path.join(win_dir, "Fonts", "simsun.ttc"),  # SimSun
            os.path.join(win_dir, "Fonts", "simhei.ttf"),  # SimHei
            os.path.join(win_dir, "Fonts", "simkai.ttf"),
            os.path.join(win_dir, "Fonts", "malgun.ttf"),  # Korean fallback
            os.path.join(win_dir, "Fonts", "YuGothM.ttc"),  # Japanese
            os.path.join(win_dir, "Fonts", "msgothic.ttc"),
            os.path.join(win_dir, "Fonts", "seguisym.ttf"),  # Segoe UI Symbol partial
            os.path.join(win_dir, "Fonts", "seguiemj.ttf"),
        ]
    elif platform.system() == "Darwin":
        candidates += [
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/STHeiti Light.ttc",
            "/Library/Fonts/Arial Unicode.ttf",
        ]
    else:  # Linux
        candidates += [
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansSC-Regular.otf",
        ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None


_CJK_FONT_PATH = _find_cjk_font()
_CJK_FONT_CACHE = {}


def _get_font(size=18):
    if not PIL_AVAILABLE:
        return None
    key = size
    if key in _CJK_FONT_CACHE:
        return _CJK_FONT_CACHE[key]
    try:
        if _CJK_FONT_PATH:
            font = ImageFont.truetype(_CJK_FONT_PATH, size)
        else:
            font = ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()
    _CJK_FONT_CACHE[key] = font
    return font


def _draw_text_pil(img_bgr, text, org, color=(255, 255, 255), font_size=18):
    """Draw Unicode text on BGR OpenCV image using PIL. Returns BGR image.

    NOTE: prefer _draw_texts_pil_batch when drawing many labels — this
    single-text variant does a full-image BGR↔RGB round trip each call
    and is only kept for backwards compatibility.
    """
    return _draw_texts_pil_batch(img_bgr, [(text, org, color, font_size)])


def _draw_texts_pil_batch(img_bgr, items):
    """Draw many Unicode texts in one BGR->PIL->BGR round trip.

    items: iterable of (text, (x, y), (b, g, r), font_size).
    Rendering 30 CJK labels this way is ~30x cheaper than calling
    _draw_text_pil per label, because color-space conversion and PIL
    wrapping dominate a single draw call on a 1280x720 frame.
    """
    items = [it for it in items if it and it[0]]
    if not items:
        return img_bgr
    if not PIL_AVAILABLE:
        # ASCII fallback — CJK characters will render as '?' but at least fast.
        for text, org, color, font_size in items:
            try:
                cv2.putText(
                    img_bgr,
                    str(text).encode("ascii", "replace").decode(),
                    org,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    max(0.3, font_size / 30.0),
                    color,
                    1,
                    cv2.LINE_AA,
                )
            except Exception:
                pass
        return img_bgr
    try:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil)
        for text, org, color, font_size in items:
            font = _get_font(font_size)
            rgb_color = (color[2], color[1], color[0])
            try:
                draw.text(org, str(text), fill=rgb_color, font=font)
            except Exception:
                continue
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except Exception:
        return img_bgr


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

        # Collect every Unicode text into one batch — drawing them in a single
        # BGR->PIL->BGR round trip is ~30x cheaper than one round trip per label.
        pil_items = []

        # OCR boxes: rectangles now, text queued for batch
        if self.show_ocr and ocr_result and ocr_result.get("texts"):
            for t in ocr_result["texts"]:
                x1, y1, x2, y2 = t["x1"], t["y1"], t["x2"], t["y2"]
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 180, 0), 1)
                pil_items.append((t["text"][:30], (x1, y2 + 2), (255, 180, 0), 16))

        # HUD - ASCII FPS stays on cv2 (fast), notices/VLM join the PIL batch
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
            pil_items.append(
                (f"NEW TEXT: {notice}", (10, y - 5), (0, 200, 255), 20)
            )
            y += 28
        if vlm_text:
            for line in vlm_text.split("\n")[:2]:
                pil_items.append(
                    (f"VLM: {line[:80]}", (10, y - 4), (200, 200, 255), 18)
                )
                y += 24

        if pil_items:
            img = _draw_texts_pil_batch(img, pil_items)

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
