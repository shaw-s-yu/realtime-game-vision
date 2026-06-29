"""Dear PyGui based realtime control panel for tuning config without restart.
Run as separate thread. Updates ConfigManager live; main loop polls get() each frame.
Install: pip install dearpygui
"""

import threading
import time

try:
    import dearpygui.dearpygui as dpg

    DPG_AVAILABLE = True
except Exception:
    DPG_AVAILABLE = False


class ControlPanel(threading.Thread):
    def __init__(self, config_manager, port=0):
        super().__init__(daemon=True)
        self.cm = config_manager
        self.running = False
        self._stop = False

    def run(self):
        if not DPG_AVAILABLE:
            print("[ui] dearpygui not installed, UI disabled. pip install dearpygui")
            return
        self.running = True
        dpg.create_context()
        dpg.create_viewport(title="Realtime Game Vision Control", width=520, height=680)
        dpg.setup_dearpygui()

        cfg = self.cm.get()

        def get(path, default):
            cur = cfg
            for p in path.split("."):
                cur = cur.get(p, {}) if isinstance(cur, dict) else default
            return cur if cur != {} else default

        with dpg.window(label="Live Tuning", width=500, height=660, tag="main_win"):
            dpg.add_text(
                "Realtime Game Vision — live config. Changes apply next frame."
            )
            dpg.add_separator()
            dpg.add_text("Capture")
            dpg.add_slider_int(
                label="process_fps",
                tag="process_fps",
                min_value=1,
                max_value=30,
                default_value=get("capture.process_fps", 10),
                callback=lambda s, a: self.cm.update("capture.process_fps", a),
            )
            dpg.add_slider_int(
                label="target_fps",
                tag="target_fps",
                min_value=10,
                max_value=120,
                default_value=get("capture.target_fps", 30),
                callback=lambda s, a: self.cm.update("capture.target_fps", a),
            )
            dpg.add_slider_int(
                label="output_width",
                tag="output_width",
                min_value=640,
                max_value=1920,
                default_value=get("capture.output_width", 1280),
                callback=lambda s, a: self.cm.update("capture.output_width", a),
            )

            dpg.add_separator()
            dpg.add_text("Detector")
            dpg.add_slider_float(
                label="conf threshold",
                tag="det_conf",
                min_value=0.05,
                max_value=0.9,
                format="%.2f",
                default_value=get("detector.conf", 0.25),
                callback=lambda s, a: self.cm.update("detector.conf", a),
            )
            dpg.add_slider_float(
                label="iou threshold",
                tag="det_iou",
                min_value=0.1,
                max_value=0.9,
                format="%.2f",
                default_value=get("detector.iou", 0.45),
                callback=lambda s, a: self.cm.update("detector.iou", a),
            )
            dpg.add_slider_int(
                label="max_det",
                tag="max_det",
                min_value=10,
                max_value=200,
                default_value=get("detector.max_det", 100),
                callback=lambda s, a: self.cm.update("detector.max_det", a),
            )
            dpg.add_combo(
                label="device",
                tag="det_device",
                items=["cuda", "cpu"],
                default_value=get("detector.device", "cuda"),
                callback=lambda s, a: self.cm.update("detector.device", a),
            )

            dpg.add_separator()
            dpg.add_text("OCR")
            dpg.add_checkbox(
                label="ocr enabled",
                tag="ocr_en",
                default_value=get("ocr.enabled", True),
                callback=lambda s, a: self.cm.update("ocr.enabled", a),
            )
            dpg.add_combo(
                label="ocr lang",
                tag="ocr_lang",
                items=["ch", "en", "ch_server", "en_server", "japan", "korean"],
                default_value=get("ocr.lang", "ch"),
                callback=lambda s, a: self.cm.update("ocr.lang", a),
            )
            dpg.add_checkbox(
                label="ocr roi only (faster)",
                tag="ocr_roi",
                default_value=get("ocr.roi_only", True),
                callback=lambda s, a: self.cm.update("ocr.roi_only", a),
            )
            dpg.add_slider_float(
                label="ocr det thresh",
                tag="ocr_det",
                min_value=0.1,
                max_value=0.9,
                format="%.2f",
                default_value=get("ocr.det_thresh", 0.3),
                callback=lambda s, a: self.cm.update("ocr.det_thresh", a),
            )
            dpg.add_slider_float(
                label="ocr rec thresh",
                tag="ocr_rec",
                min_value=0.1,
                max_value=0.9,
                format="%.2f",
                default_value=get("ocr.rec_thresh", 0.5),
                callback=lambda s, a: self.cm.update("ocr.rec_thresh", a),
            )

            dpg.add_separator()
            dpg.add_text("Overlay")
            dpg.add_checkbox(
                label="show trails",
                tag="show_trails",
                default_value=get("overlay.show_trails", True),
                callback=lambda s, a: self.cm.update("overlay.show_trails", a),
            )
            dpg.add_checkbox(
                label="show ocr boxes",
                tag="show_ocr",
                default_value=get("overlay.show_ocr", True),
                callback=lambda s, a: self.cm.update("overlay.show_ocr", a),
            )
            dpg.add_checkbox(
                label="show labels",
                tag="show_labels",
                default_value=get("overlay.show_labels", True),
                callback=lambda s, a: self.cm.update("overlay.show_labels", a),
            )
            dpg.add_slider_int(
                label="trail length",
                tag="trail_len",
                min_value=5,
                max_value=60,
                default_value=get("overlay.trail_length", 15),
                callback=lambda s, a: self.cm.update("overlay.trail_length", a),
            )

            dpg.add_separator()
            dpg.add_text("VLM")
            dpg.add_checkbox(
                label="vlm enabled",
                tag="vlm_en",
                default_value=get("vlm.enabled", False),
                callback=lambda s, a: self.cm.update("vlm.enabled", a),
            )
            dpg.add_slider_int(
                label="vlm interval frames",
                tag="vlm_int",
                min_value=1,
                max_value=30,
                default_value=get("vlm.interval", 3),
                callback=lambda s, a: self.cm.update("vlm.interval", a),
            )

            dpg.add_separator()
            dpg.add_button(label="Save to config.yaml", callback=lambda: self.cm.save())
            dpg.add_button(
                label="Reload from disk", callback=lambda: self.cm.load(force=True)
            )
            dpg.add_text(
                "Tip: changes apply live next processed frame. Save persists to disk for next run.",
                wrap=480,
            )

        dpg.show_viewport()
        while dpg.is_dearpygui_running() and not self._stop:
            dpg.render_dearpygui_frame()
            time.sleep(0.016)
        dpg.destroy_context()
        self.running = False

    def stop(self):
        self._stop = True
        if DPG_AVAILABLE:
            try:
                import dearpygui.dearpygui as dpg

                if dpg.is_dearpygui_running():
                    dpg.stop_dearpygui()
            except:
                pass
