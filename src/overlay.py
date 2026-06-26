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
    """Draw Unicode text on BGR OpenCV image using PIL. Returns BGR image."""
    if not PIL_AVAILABLE or not text:
        # fallback to cv2 (will show ??? for CJK but better than crash)
        cv2.putText(
            img_bgr,
            text.encode("ascii", "replace").decode(),
            org,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_size / 30.0,
            color,
            1,
            cv2.LINE_AA,
        )
        return img_bgr
    try:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil)
        font = _get_font(font_size)
        # PIL uses RGB, convert BGR color to RGB
        rgb_color = (color[2], color[1], color[0])
        draw.text(org, str(text), fill=rgb_color, font=font)
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except Exception:
        # fallback
        try:
            cv2.putText(
                img_bgr,
                text.encode("ascii", "replace").decode(),
                org,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
        except:
            pass
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

        # OCR boxes - use PIL for Unicode text rendering to support Chinese
        if self.show_ocr and ocr_result and ocr_result.get("texts"):
            for t in ocr_result["texts"]:
                x1, y1, x2, y2 = t["x1"], t["y1"], t["x2"], t["y2"]
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 180, 0), 1)
                txt = t["text"][:30]
                img = _draw_text_pil(
                    img, txt, (x1, y2 + 2), color=(255, 180, 0), font_size=16
                )

        # HUD - use PIL for parts that may contain Unicode (notices, VLM), cv2 for ASCII FPS
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
            img = _draw_text_pil(
                img,
                f"NEW TEXT: {notice}",
                (10, y - 5),
                color=(0, 200, 255),
                font_size=20,
            )
            y += 28
        if vlm_text:
            # wrap
            for line in vlm_text.split("\n")[:2]:
                img = _draw_text_pil(
                    img,
                    f"VLM: {line[:80]}",
                    (10, y - 4),
                    color=(200, 200, 255),
                    font_size=18,
                )
                y += 24
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
