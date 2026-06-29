"""
Cross-platform configuration UI for Realtime Game Vision.
PySide6 based — works on Windows, Linux, macOS.
Top bar has 3 tabs: Custom / All / Screen.
Custom pane has Manage button to select which config fields to show, and Save Custom Layout button.
All pane shows all configuration fields.
Screen tab shows live video with detections and process log, no separate process spawned — vision runs in-process thread started by Start button, stopped by Stop button.
No hot reload per spec — Start launches fresh in-process engine with snapshot config.
"""

import sys
import os
import json
import time
from pathlib import Path
from copy import deepcopy
import threading

try:
    from PySide6 import QtWidgets, QtCore, QtGui

    PYSIDE_AVAILABLE = True
except Exception as e:
    PYSIDE_AVAILABLE = False
    _missing_err = e

import yaml
import numpy as np
import cv2

try:
    from .config_schema import SCHEMA, DEFAULT_CUSTOM_SELECTION
except ImportError:
    from config_schema import SCHEMA, DEFAULT_CUSTOM_SELECTION  # type: ignore

try:
    from .vision_engine import VisionEngine
except ImportError:
    from vision_engine import VisionEngine  # type: ignore

CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
CUSTOM_PATH = Path(__file__).parent.parent / "ui_custom.json"
RUNTIME_CONFIG = Path(__file__).parent.parent / "config.runtime.yaml"


def load_yaml(path: Path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}


def save_yaml(path: Path, data: dict):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def get_nested(d: dict, dotpath: str, default=None):
    cur = d
    for p in dotpath.split("."):
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return default
    return cur


def set_nested(d: dict, dotpath: str, value):
    parts = dotpath.split(".")
    cur = d
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = value


class ManageDialog(QtWidgets.QDialog):
    """Dialog to select which fields appear in Custom tab."""

    def __init__(self, current_selection, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Manage Custom Pane Fields")
        self.resize(520, 600)
        self.selection = set(current_selection)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(
            QtWidgets.QLabel(
                "Check fields to show in Custom tab. Unchecked fields remain available in All tab."
            )
        )

        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setRootIsDecorated(True)
        layout.addWidget(self.tree)

        # group by schema group
        grouped = {}
        for item in SCHEMA:
            grouped.setdefault(item["group"], []).append(item)

        for group in sorted(grouped.keys()):
            group_item = QtWidgets.QTreeWidgetItem([group.upper()])
            group_item.setFlags(group_item.flags() & ~QtCore.Qt.ItemIsUserCheckable)
            font = group_item.font(0)
            font.setBold(True)
            group_item.setFont(0, font)
            group_item.setBackground(0, QtGui.QColor("#e0e0e0"))
            self.tree.addTopLevelItem(group_item)
            for it in grouped[group]:
                child = QtWidgets.QTreeWidgetItem([f"{it['label']}  ({it['path']})"])
                child.setFlags(child.flags() | QtCore.Qt.ItemIsUserCheckable)
                child.setCheckState(
                    0,
                    QtCore.Qt.Checked
                    if it["path"] in self.selection
                    else QtCore.Qt.Unchecked,
                )
                child.setData(0, QtCore.Qt.UserRole, it["path"])
                child.setToolTip(0, it.get("description", ""))
                group_item.addChild(child)
            group_item.setExpanded(True)

        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def get_selection(self):
        sel = []
        root = self.tree.invisibleRootItem()
        for gi in range(root.childCount()):
            group_item = root.child(gi)
            for ci in range(group_item.childCount()):
                child = group_item.child(ci)
                if child.checkState(0) == QtCore.Qt.Checked:
                    path = child.data(0, QtCore.Qt.UserRole)
                    if path:
                        sel.append(path)
        return sel


class ConfigFormWidget(QtWidgets.QWidget):
    """Dynamically builds form based on schema subset."""

    def __init__(self, schema_items, config_dict, parent=None):
        super().__init__(parent)
        self.schema_items = schema_items
        self.widgets = {}  # path -> widget
        layout = QtWidgets.QVBoxLayout(self)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        container = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(container)
        form.setLabelAlignment(QtCore.Qt.AlignRight)
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.ExpandingFieldsGrow)

        # group preserving order
        grouped = {}
        order = []
        for it in schema_items:
            g = it["group"]
            if g not in grouped:
                grouped[g] = []
                order.append(g)
            grouped[g].append(it)

        for group in order:
            group_label = QtWidgets.QLabel(f"<b>{group.upper()}</b>")
            form.addRow(group_label)
            for item in grouped[group]:
                path = item["path"]
                label = item["label"]
                typ = item["type"]
                desc = item.get("description", "")
                default_val = get_nested(config_dict, path, item.get("default"))
                w = self._make_widget(item, default_val)
                self.widgets[path] = w
                lbl = QtWidgets.QLabel(label)
                lbl.setToolTip(desc)
                w.setToolTip(desc)
                form.addRow(lbl, w)
                if item.get("restart"):
                    note = QtWidgets.QLabel(
                        "requires restart on change – applies on next Start"
                    )
                    note.setStyleSheet("color:#b36b00; font-size:9pt;")
                    form.addRow("", note)

        scroll.setWidget(container)
        layout.addWidget(scroll)

    def _make_widget(self, item, value):
        typ = item["type"]
        if typ == "bool":
            cb = QtWidgets.QCheckBox()
            cb.setChecked(bool(value))
            return cb
        elif typ == "int":
            sb = QtWidgets.QSpinBox()
            sb.setMinimum(item.get("min", 0))
            sb.setMaximum(item.get("max", 10000))
            sb.setSingleStep(item.get("step", 1))
            try:
                sb.setValue(
                    int(value) if value is not None else int(item.get("default", 0))
                )
            except:
                sb.setValue(0)
            return sb
        elif typ == "float":
            dsb = QtWidgets.QDoubleSpinBox()
            dsb.setMinimum(item.get("min", 0.0))
            dsb.setMaximum(item.get("max", 1.0))
            dsb.setSingleStep(item.get("step", 0.01))
            dsb.setDecimals(3)
            try:
                dsb.setValue(
                    float(value)
                    if value is not None
                    else float(item.get("default", 0.0))
                )
            except:
                dsb.setValue(0.0)
            return dsb
        elif typ == "choice":
            cb = QtWidgets.QComboBox()
            opts = [str(o) for o in item.get("options", [])]
            cb.addItems(opts)
            try:
                idx = opts.index(str(value))
                cb.setCurrentIndex(idx)
            except ValueError:
                cb.setEditable(True)
                cb.setCurrentText(str(value) if value is not None else "")
            return cb
        elif typ == "list_str":
            le = QtWidgets.QLineEdit()
            if isinstance(value, list):
                le.setText(",".join(map(str, value)))
            else:
                le.setText(str(value) if value is not None else "")
            le.setPlaceholderText("comma separated list")
            return le
        else:  # str
            le = QtWidgets.QLineEdit()
            le.setText(str(value) if value is not None else "")
            return le

    def collect_values(self):
        out = {}
        for path, w in self.widgets.items():
            if isinstance(w, QtWidgets.QCheckBox):
                val = w.isChecked()
            elif isinstance(w, QtWidgets.QSpinBox):
                val = w.value()
            elif isinstance(w, QtWidgets.QDoubleSpinBox):
                val = round(w.value(), 6)
            elif isinstance(w, QtWidgets.QComboBox):
                val = w.currentText()
            elif isinstance(w, QtWidgets.QLineEdit):
                txt = w.text().strip()
                schema_item = next(
                    (it for it in self.schema_items if it["path"] == path), None
                )
                if schema_item and schema_item["type"] == "list_str":
                    val = [s.strip() for s in txt.split(",") if s.strip()]
                else:
                    val = txt
            else:
                val = None
            out[path] = val
        return out

    def apply_values(self, values_dict):
        for path, w in self.widgets.items():
            if path not in values_dict:
                continue
            v = values_dict[path]
            try:
                if isinstance(w, QtWidgets.QCheckBox):
                    w.setChecked(bool(v))
                elif isinstance(w, QtWidgets.QSpinBox):
                    w.setValue(int(v))
                elif isinstance(w, QtWidgets.QDoubleSpinBox):
                    w.setValue(float(v))
                elif isinstance(w, QtWidgets.QComboBox):
                    idx = w.findText(str(v))
                    if idx >= 0:
                        w.setCurrentIndex(idx)
                    else:
                        w.setCurrentText(str(v))
                elif isinstance(w, QtWidgets.QLineEdit):
                    if isinstance(v, list):
                        w.setText(",".join(map(str, v)))
                    else:
                        w.setText(str(v))
            except:
                pass


class VideoWidget(QtWidgets.QLabel):
    """QLabel that displays BGR numpy frames scaled keeping aspect ratio."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(640, 360)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setStyleSheet("background-color: black; color: white;")
        self.setText(
            "Vision not running.\nClick Start to begin screen capture and detection.\n\nGame window should be in borderless windowed mode for dxcam to capture."
        )
        self.setScaledContents(False)

    def set_frame(self, img_bgr):
        if img_bgr is None:
            return
        h, w, ch = img_bgr.shape
        # convert BGR to RGB
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        qimg = QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888)
        # scale pixmap to fit label keeping aspect ratio
        pix = QtGui.QPixmap.fromImage(qimg)
        scaled = pix.scaled(
            self.width(),
            self.height(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        )
        self.setPixmap(scaled)

    def resizeEvent(self, event):
        # keep current pixmap scaled on resize; actual frame update happens via set_frame
        super().resizeEvent(event)


class MainWindow(QtWidgets.QMainWindow):
    # Qt signal for thread-safe frame updates from vision engine thread
    frame_signal = QtCore.Signal(object)  # will emit numpy array
    log_signal = QtCore.Signal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Realtime Game Vision - Config UI")
        self.resize(1100, 800)
        self.vision_engine = None
        self.base_config_path = Path(__file__).parent.parent / "config.yaml"
        self.custom_path = Path(__file__).parent.parent / "ui_custom.json"
        self.runtime_path = Path(__file__).parent.parent / "config.runtime.yaml"

        # load base config and custom selection
        self.base_config = load_yaml(self.base_config_path)
        try:
            with open(self.custom_path, "r", encoding="utf-8") as f:
                self.custom_selection = json.load(f).get("fields", [])
                if not self.custom_selection:
                    raise ValueError
        except:
            old_path = Path(__file__).parent.parent / "ui_custom_fields.json"
            try:
                with open(old_path, "r", encoding="utf-8") as f:
                    self.custom_selection = json.load(f).get("fields", [])
            except:
                self.custom_selection = []
        if not self.custom_selection:
            try:
                from .config_schema import DEFAULT_CUSTOM_SELECTION

                self.custom_selection = DEFAULT_CUSTOM_SELECTION.copy()
            except:
                try:
                    from config_schema import DEFAULT_CUSTOM_SELECTION

                    self.custom_selection = DEFAULT_CUSTOM_SELECTION.copy()
                except:
                    self.custom_selection = [it["path"] for it in SCHEMA[:10]]

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        vbox = QtWidgets.QVBoxLayout(central)

        # top toolbar with Start Stop Save
        toolbar = QtWidgets.QHBoxLayout()
        self.start_btn = QtWidgets.QPushButton("Start")
        self.start_btn.setStyleSheet(
            "background-color:#4CAF50;color:white;font-weight:bold;padding:8px 16px;"
        )
        self.start_btn.clicked.connect(self.on_start)
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.stop_btn.setStyleSheet(
            "background-color:#f44336;color:white;font-weight:bold;padding:8px 16px;"
        )
        self.stop_btn.clicked.connect(self.on_stop)
        self.stop_btn.setEnabled(False)
        self.save_btn = QtWidgets.QPushButton("Save config.yaml")
        self.save_btn.setToolTip(
            "Save current field values to config.yaml for persistence across UI sessions"
        )
        self.save_btn.clicked.connect(self.on_save)
        self.status_label = QtWidgets.QLabel("Status: idle")
        self.status_label.setStyleSheet("font-weight:bold;")
        toolbar.addWidget(self.start_btn)
        toolbar.addWidget(self.stop_btn)
        toolbar.addWidget(self.save_btn)
        toolbar.addWidget(self.status_label)
        toolbar.addStretch()
        vbox.addLayout(toolbar)

        # tabs: Custom / All / Screen  as per spec top bar tabs
        self.tabs = QtWidgets.QTabWidget()
        vbox.addWidget(self.tabs)

        # All tab - full schema
        self.all_form = ConfigFormWidget(SCHEMA, self.base_config)
        all_tab = QtWidgets.QWidget()
        all_tab_layout = QtWidgets.QVBoxLayout(all_tab)
        all_tab_layout.setContentsMargins(0, 0, 0, 0)
        all_tab_layout.addWidget(self.all_form)
        self.tabs.addTab(all_tab, "All")

        # Custom tab with Manage button on top
        custom_tab = QtWidgets.QWidget()
        custom_vbox = QtWidgets.QVBoxLayout(custom_tab)
        manage_hbox = QtWidgets.QHBoxLayout()
        manage_btn = QtWidgets.QPushButton("Manage Fields...")
        manage_btn.setToolTip("Choose which configuration fields appear in Custom tab")
        manage_btn.clicked.connect(self.on_manage)
        manage_hbox.addWidget(manage_btn)
        self.save_custom_btn = QtWidgets.QPushButton("Save Custom Layout")
        self.save_custom_btn.setToolTip(
            "Save current custom field selection to ui_custom.json"
        )
        self.save_custom_btn.clicked.connect(self.on_save_custom)
        manage_hbox.addWidget(self.save_custom_btn)
        self.custom_status_label = QtWidgets.QLabel("")
        self.custom_status_label.setStyleSheet("color:#2a7d2a; font-style:italic;")
        manage_hbox.addWidget(self.custom_status_label)
        manage_hbox.addWidget(
            QtWidgets.QLabel(
                "Custom pane shows selected fields for quick tuning. All pane shows everything."
            )
        )
        manage_hbox.addStretch()
        custom_vbox.addLayout(manage_hbox)
        self.custom_form_container_holder = QtWidgets.QVBoxLayout()
        custom_vbox.addLayout(self.custom_form_container_holder)
        self.tabs.addTab(custom_tab, "Custom")

        # Screen tab with 2 sections: video display on top, process log below. No log elsewhere per spec.
        screen_tab = QtWidgets.QWidget()
        screen_vbox = QtWidgets.QVBoxLayout(screen_tab)
        # video display area
        video_group = QtWidgets.QGroupBox("Screen and Detection - Realtime")
        video_layout = QtWidgets.QVBoxLayout()
        self.video_widget = VideoWidget()
        self.video_widget.setMinimumHeight(400)
        video_layout.addWidget(self.video_widget)
        video_group.setLayout(video_layout)
        screen_vbox.addWidget(video_group, stretch=3)

        # process log area inside screen tab only
        log_group = QtWidgets.QGroupBox("Process Log")
        log_layout = QtWidgets.QVBoxLayout()
        self.log_edit = QtWidgets.QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setMaximumBlockCount(1000)
        self.log_edit.setPlaceholderText(
            "Process log will appear here after clicking Start..."
        )
        log_layout.addWidget(self.log_edit)
        log_group.setLayout(log_layout)
        screen_vbox.addWidget(log_group, stretch=1)

        self.tabs.addTab(screen_tab, "Screen")

        # default to Custom tab on open per spec preference pane concept, user can switch to All or Screen
        self.tabs.setCurrentIndex(0)

        # connect signals for thread-safe UI updates from vision engine
        self.frame_signal.connect(self.on_frame_received)
        self.log_signal.connect(self.on_log_received)

        # timer to check process status for old subprocess mode fallback - not needed now but keep for safety
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.check_process)
        self.timer.start(1000)

    def rebuild_custom_form(self):
        while self.custom_form_container_holder.count():
            child = self.custom_form_container_holder.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        filtered_schema = [it for it in SCHEMA if it["path"] in self.custom_selection]
        if not filtered_schema:
            lbl = QtWidgets.QLabel(
                "No fields selected. Click Manage Fields to add configuration fields to custom pane."
            )
            lbl.setWordWrap(True)
            self.custom_form_container_holder.addWidget(lbl)
            self.custom_form = None
            if hasattr(self, "custom_status_label"):
                self.custom_status_label.setText("Custom: 0 fields")
        else:
            current_vals = (
                self.all_form.collect_values() if hasattr(self, "all_form") else {}
            )
            temp_cfg = {}
            for path, val in current_vals.items():
                parts = path.split(".")
                cur = temp_cfg
                for p in parts[:-1]:
                    cur = cur.setdefault(p, {})
                cur[parts[-1]] = val
            if not temp_cfg:
                temp_cfg = load_yaml(CONFIG_PATH)
            self.custom_form = ConfigFormWidget(filtered_schema, temp_cfg)
            self.custom_form_container_holder.addWidget(self.custom_form)
            if hasattr(self, "custom_status_label"):
                self.custom_status_label.setText(
                    f"Custom: {len(filtered_schema)} fields | ui_custom.json"
                )

    def on_manage(self):
        from .config_schema import SCHEMA as schema_ref

        dlg = ManageDialog(self.custom_selection, self)
        if dlg.exec() == QtWidgets.QDialog.Accepted:
            self.custom_selection = dlg.get_selection()
            try:
                with open(self.custom_path, "w", encoding="utf-8") as f:
                    import json

                    json.dump({"fields": self.custom_selection}, f, indent=2)
            except Exception as e:
                QtWidgets.QMessageBox.warning(
                    self, "Error", f"Failed to save custom selection: {e}"
                )
                return
            self.rebuild_custom_form()
            # switch to Custom tab automatically to show result immediately per improved UX spec
            for i in range(self.tabs.count()):
                if self.tabs.tabText(i).lower() == "custom":
                    self.tabs.setCurrentIndex(i)
                    break
            msg = f"Custom layout saved to {self.custom_path.name} with {len(self.custom_selection)} fields — Custom tab updated live"
            self.status_label.setText(msg)
            self.custom_status_label.setText(
                f"Custom: {len(self.custom_selection)} fields | ui_custom.json"
            )
            self.log_edit.appendPlainText(f"[UI] {msg}")

    def on_save_custom(self):
        try:
            with open(self.custom_path, "w", encoding="utf-8") as f:
                import json

                json.dump({"fields": self.custom_selection}, f, indent=2)
            self.custom_status_label.setText(
                f"Custom layout saved to ui_custom.json ({len(self.custom_selection)} fields)"
            )
            self.status_label.setText("Status: ui_custom.json saved")
            self.log_edit.appendPlainText(
                f"[UI] Custom layout explicitly saved to {self.custom_path.resolve()} with {len(self.custom_selection)} fields"
            )
            QtWidgets.QMessageBox.information(
                self,
                "Saved",
                f"Custom field selection saved to:\n{self.custom_path.resolve()}\n\n{len(self.custom_selection)} fields will appear in Custom tab on next UI launch.\nCurrent Custom tab already reflects selection.",
            )
        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self, "Error", f"Failed to save ui_custom.json: {e}"
            )

    def collect_all_values(self):
        vals = self.all_form.collect_values()
        if hasattr(self, "custom_form") and self.custom_form:
            try:
                custom_vals = self.custom_form.collect_values()
                vals.update(custom_vals)
            except:
                pass
        return vals

    def apply_values_to_config(self, values_dict):
        cfg = load_yaml(CONFIG_PATH)
        if not cfg:
            cfg = {}
        for dotpath, val in values_dict.items():
            parts = dotpath.split(".")
            cur = cfg
            for p in parts[:-1]:
                cur = cur.setdefault(p, {})
            cur[parts[-1]] = val
        return cfg

    def on_save(self):
        try:
            vals = self.collect_all_values()
            cfg = self.apply_values_to_config(vals)
            save_yaml(CONFIG_PATH, cfg)
            self.status_label.setText("Status: config.yaml saved")
            self.log_edit.appendPlainText(
                "[UI] config.yaml saved with current values for next session persistence"
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Save Error", str(e))

    def on_frame_received(self, img):
        # img is numpy BGR array from vision engine thread via signal
        try:
            self.video_widget.set_frame(img)
        except Exception as e:
            # avoid crashing UI on occasional bad frame
            pass

    def on_log_received(self, msg):
        try:
            self.log_edit.appendPlainText(msg)
        except:
            pass

    def on_start(self):
        # check if already running
        if (
            hasattr(self, "vision_engine")
            and self.vision_engine
            and self.vision_engine.is_running()
        ):
            QtWidgets.QMessageBox.information(
                self, "Already Running", "Vision process already running. Stop first."
            )
            return
        # collect current UI values and write to runtime config, then start in-process vision engine per new spec (no separate process)
        try:
            vals = self.collect_all_values()
            cfg = self.apply_values_to_config(vals)
            runtime_path = Path(__file__).parent.parent / "config.runtime.yaml"
            save_yaml(runtime_path, cfg)
            self.log_edit.appendPlainText(f"[UI] Config prepared at {runtime_path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Error", f"Failed to prepare config: {e}"
            )
            return
        try:
            # switch to Screen tab automatically per spec
            for i in range(self.tabs.count()):
                if self.tabs.tabText(i).lower() == "screen":
                    self.tabs.setCurrentIndex(i)
                    break
            # create vision engine with callbacks to UI thread via signals
            from .vision_engine import VisionEngine

            def frame_cb(img):
                # emit signal to UI thread for safe QLabel update
                self.frame_signal.emit(img)

            def log_cb(msg):
                self.log_signal.emit(msg)

            self.vision_engine = VisionEngine(
                config_path=str(runtime_path),
                frame_callback=frame_cb,
                log_callback=log_cb,
            )
            started = self.vision_engine.start()
            if not started:
                raise RuntimeError("VisionEngine failed to start thread")
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.status_label.setText("Status: running in-process vision engine")
            self.log_edit.appendPlainText(
                "[UI] Vision engine started in-process thread. Screen tab should now show live video with detections."
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Start Failed", str(e))
            import traceback

            self.log_edit.appendPlainText(
                "[UI] Start failed traceback:\n" + traceback.format_exc()
            )

    def on_stop(self):
        try:
            if hasattr(self, "vision_engine") and self.vision_engine:
                stopped = self.vision_engine.stop(timeout=3)
                if stopped:
                    self.status_label.setText("Status: stopped")
                    self.log_edit.appendPlainText("[UI] Vision engine stopped by user")
                else:
                    self.status_label.setText("Status: stop requested, waiting...")
                    self.log_edit.appendPlainText(
                        "[UI] Vision engine stop timeout, may still be shutting down background threads"
                    )
            else:
                self.status_label.setText("Status: idle")
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Stop Error", str(e))
        finally:
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.vision_engine = None
            # clear video widget to placeholder text
            try:
                self.video_widget.clear()
                self.video_widget.setText(
                    "Vision stopped.\nClick Start to begin screen capture and detection.\n\nGame window should be in borderless windowed mode for dxcam to capture."
                )
            except:
                pass

    def check_process(self):
        # kept for compatibility; with in-process engine we rely on is_running check via timer maybe not needed, but keep for UI status refresh
        if hasattr(self, "vision_engine") and self.vision_engine:
            if not self.vision_engine.is_running() and not self.start_btn.isEnabled():
                # engine died unexpectedly
                self.status_label.setText("Status: exited unexpectedly")
                self.start_btn.setEnabled(True)
                self.stop_btn.setEnabled(False)
                self.vision_engine = None

    def closeEvent(self, event):
        if (
            hasattr(self, "vision_engine")
            and self.vision_engine
            and self.vision_engine.is_running()
        ):
            reply = QtWidgets.QMessageBox.question(
                self,
                "Quit",
                "Vision engine is running. Stop it and quit UI?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            )
            if reply == QtWidgets.QMessageBox.Yes:
                self.on_stop()
            else:
                event.ignore()
                return
        event.accept()


def main():
    if not PYSIDE_AVAILABLE:
        print("PySide6 not installed. Run: pip install PySide6")
        print("Or install full requirements: pip install -r requirements.txt")
        sys.exit(1)
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
