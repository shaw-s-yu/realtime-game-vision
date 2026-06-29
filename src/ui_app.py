"""
Cross-platform configuration UI for Realtime Game Vision.
PySide6 based — works on Windows, Linux, macOS.
Top bar has 2 tabs: Custom / All.
Custom pane has Manage button to select which config fields to show.
Start button launches python -m src.main with generated config, Stop terminates process.
No hot reload — UI is source of truth until Start is pressed.
"""

import sys
import os
import json
import subprocess
import time
from pathlib import Path
from copy import deepcopy

try:
    from PySide6 import QtWidgets, QtCore, QtGui

    PYSIDE_AVAILABLE = True
except Exception as e:
    PYSIDE_AVAILABLE = False
    _missing_err = e

import yaml

# import schema from same package or standalone fallback
try:
    from .config_schema import SCHEMA, DEFAULT_CUSTOM_SELECTION
except ImportError:
    from config_schema import SCHEMA, DEFAULT_CUSTOM_SELECTION  # type: ignore

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
            group_item.setBackground(0, QtGui.QColor("#e8e8e8"))
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

        outer = QtWidgets.QVBoxLayout(self)
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
        outer.addWidget(scroll)

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
                # find schema to decide list vs str
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
        """Update widgets from dict keyed by dotpath, used when reloading."""
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


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Realtime Game Vision - Config UI")
        self.resize(1000, 760)
        self.proc = None
        self.base_config_path = CONFIG_PATH
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
            # try old path name for backward compatibility
            old_path = Path(__file__).parent.parent / "ui_custom_fields.json"
            try:
                with open(old_path, "r", encoding="utf-8") as f:
                    self.custom_selection = json.load(f).get("fields", [])
            except:
                self.custom_selection = []
        if not self.custom_selection:
            # fallback to default from schema module
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
        self.save_btn.clicked.connect(self.on_save)
        self.status_label = QtWidgets.QLabel("Status: idle")
        self.status_label.setStyleSheet("font-weight:bold;")
        toolbar.addWidget(self.start_btn)
        toolbar.addWidget(self.stop_btn)
        toolbar.addWidget(self.save_btn)
        toolbar.addWidget(self.status_label)
        toolbar.addStretch()
        vbox.addLayout(toolbar)

        # tabs: Custom / All
        self.tabs = QtWidgets.QTabWidget()
        vbox.addWidget(self.tabs)

        # All tab - full schema
        self.all_form = ConfigFormWidget(SCHEMA, self.base_config)
        all_scroll = QtWidgets.QScrollArea()
        all_scroll.setWidgetResizable(True)
        all_container = QtWidgets.QWidget()
        all_layout = QtWidgets.QVBoxLayout(all_container)
        all_layout.addWidget(self.all_form)
        all_scroll.setWidget(
            all_container
        )  # actually ConfigFormWidget already has scroll, simplify
        # We'll just use all_form directly inside tab with its own scroll
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
        self.rebuild_custom_form()

        # bottom log area read-only for process output
        log_group = QtWidgets.QGroupBox("Process Log")
        log_layout = QtWidgets.QVBoxLayout()
        self.log_edit = QtWidgets.QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setMaximumBlockCount(500)
        log_layout.addWidget(self.log_edit)
        log_group.setLayout(log_layout)
        vbox.addWidget(log_group, stretch=1)

        # timer to check process status
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.check_process)
        self.timer.start(1000)

        # default to Custom tab
        self.tabs.setCurrentIndex(0)

    def rebuild_custom_form(self):
        # clear existing layout in holder
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
            # load current values from All form to keep in sync, or from base config
            current_vals = (
                self.all_form.collect_values() if hasattr(self, "all_form") else {}
            )
            # build temp config dict for initial values
            temp_cfg = {}
            for path, val in current_vals.items():
                # set nested structure for ConfigFormWidget init expects nested dict
                parts = path.split(".")
                cur = temp_cfg
                for p in parts[:-1]:
                    cur = cur.setdefault(p, {})
                cur[parts[-1]] = val
            # fallback to base config file values if empty
            if not temp_cfg:
                temp_cfg = load_yaml(CONFIG_PATH)
            self.custom_form = ConfigFormWidget(filtered_schema, temp_cfg)
            self.custom_form_container_holder.addWidget(self.custom_form)
            if hasattr(self, "custom_status_label"):
                self.custom_status_label.setText(
                    f"Custom: {len(filtered_schema)} fields | ui_custom.json"
                )

    def on_manage(self):
        dlg = ManageDialog(self.custom_selection, self)
        if dlg.exec() == QtWidgets.QDialog.Accepted:
            self.custom_selection = dlg.get_selection()
            # save to ui_custom.json in repo root for persistence across sessions
            try:
                with open(self.custom_path, "w", encoding="utf-8") as f:
                    json.dump({"fields": self.custom_selection}, f, indent=2)
            except Exception as e:
                QtWidgets.QMessageBox.warning(
                    self, "Error", f"Failed to save custom selection: {e}"
                )
                return
            self.rebuild_custom_form()
            # switch to Custom tab automatically to show result immediately
            # find Custom tab index by label
            for i in range(self.tabs.count()):
                if self.tabs.tabText(i).lower() == "custom":
                    self.tabs.setCurrentIndex(i)
                    break
            msg = f"Custom layout saved to {self.custom_path.name} with {len(self.custom_selection)} fields — Custom tab updated live"
            self.status_label.setText(msg)
            self.custom_status_label.setText(
                f"Saved {len(self.custom_selection)} fields to ui_custom.json"
            )
            self.log_edit.appendPlainText(f"[UI] {msg}")

    def on_save_custom(self):
        """Explicit save custom layout button handler - redundant with Manage OK auto-save but provides explicit UI per user feedback."""
        try:
            # ensure current selection is saved even if Manage dialog not reopened
            with open(self.custom_path, "w", encoding="utf-8") as f:
                json.dump({"fields": self.custom_selection}, f, indent=2)
            self.custom_status_label.setText(
                f"Custom layout saved to ui_custom.json ({len(self.custom_selection)} fields)"
            )
            self.status_label.setText("Status: ui_custom.json saved")
            self.log_edit.appendPlainText(
                f"[UI] Custom layout explicitly saved to {self.custom_path} with {len(self.custom_selection)} fields"
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
        # All form is authoritative as superset; custom form values should sync to all form ideally, but for simplicity we collect from all form and then overlay custom form values if custom tab active to ensure latest edits not lost.
        vals = self.all_form.collect_values()
        # if custom form exists and maybe user edited there last, overlay those values (they should be subset)
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
            # find schema to handle list_str properly already done in collect, but ensure type coercion
            cur[parts[-1]] = val
        return cfg

    def on_save(self):
        try:
            vals = self.collect_all_values()
            cfg = self.apply_values_to_config(vals)
            save_yaml(CONFIG_PATH, cfg)
            self.status_label.setText("Status: config.yaml saved")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Save Error", str(e))

    def on_start(self):
        if self.proc and self.proc.poll() is None:
            QtWidgets.QMessageBox.information(
                self, "Already Running", "Vision process already running. Stop first."
            )
            return
        # collect current UI values and write to runtime config, then launch subprocess
        try:
            vals = self.collect_all_values()
            cfg = self.apply_values_to_config(vals)
            runtime_path = Path(__file__).parent.parent / "config.runtime.yaml"
            save_yaml(runtime_path, cfg)
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Error", f"Failed to prepare config: {e}"
            )
            return
        try:
            import sys

            python_exe = sys.executable
            # run as module src.main with --config pointing to runtime yaml
            cmd = [python_exe, "-m", "src.main", "--config", str(runtime_path)]
            # Use cwd as repo root so src module resolves
            cwd = str(Path(__file__).parent.parent)
            self.proc = subprocess.Popen(
                cmd,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.status_label.setText(f"Status: running pid {self.proc.pid}")
            # start thread to read output non-blocking
            self._start_log_reader()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Start Failed", str(e))

    def _start_log_reader(self):
        import threading

        def reader():
            try:
                assert self.proc is not None and self.proc.stdout is not None
                for line in iter(self.proc.stdout.readline, ""):
                    if not line:
                        break
                    # Use Qt signal via QMetaObject to update UI thread safe - simplified using timer poll approach below actually we use direct append with invokeMethod
                    QtCore.QMetaObject.invokeMethod(
                        self.log_edit,
                        "appendPlainText",
                        QtCore.Qt.QueuedConnection,
                        QtCore.Q_ARG(str, line.rstrip()),
                    )
            except Exception:
                pass

        t = threading.Thread(target=reader, daemon=True)
        t.start()

    def on_stop(self):
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                self.status_label.setText("Status: stopped")
                self.log_edit.appendPlainText("[UI] Process terminated by user")
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Stop Error", str(e))
        else:
            self.status_label.setText("Status: idle")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.proc = None

    def check_process(self):
        if self.proc:
            ret = self.proc.poll()
            if ret is not None:
                self.status_label.setText(f"Status: exited code {ret}")
                self.start_btn.setEnabled(True)
                self.stop_btn.setEnabled(False)
                self.log_edit.appendPlainText(f"[UI] Process exited with code {ret}")
                self.proc = None

    def closeEvent(self, event):
        if self.proc and self.proc.poll() is None:
            reply = QtWidgets.QMessageBox.question(
                self,
                "Quit",
                "Vision process is running. Stop it and quit UI?",
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
