"""Thread-safe hot-reloadable configuration manager for realtime UI tuning."""

import threading
import time
import yaml
from pathlib import Path
from copy import deepcopy


class ConfigManager:
    def __init__(self, path="config.yaml", auto_reload=True):
        self.path = Path(path)
        self._lock = threading.RLock()
        self._cfg = {}
        self._last_mtime = 0
        self.auto_reload = auto_reload
        self.load(force=True)

    def load(self, force=False):
        try:
            mtime = self.path.stat().st_mtime
        except FileNotFoundError:
            return False
        if not force and mtime == self._last_mtime:
            return False
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            with self._lock:
                self._cfg = data or {}
                self._last_mtime = mtime
            return True
        except Exception:
            return False

    def get(self):
        if self.auto_reload:
            self.load()
        with self._lock:
            return deepcopy(self._cfg)

    def get_nested(self, *keys, default=None):
        cfg = self.get()
        cur = cfg
        for k in keys:
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                return default
        return cur

    def update(self, dot_path: str, value):
        """Update nested value by dot path like 'detector.conf' and optionally write back to file."""
        with self._lock:
            keys = dot_path.split(".")
            cur = self._cfg
            for k in keys[:-1]:
                cur = cur.setdefault(k, {})
            # try to coerce type based on existing value
            old = cur.get(keys[-1])
            if isinstance(old, bool):
                value = bool(value)
            elif isinstance(old, int) and not isinstance(old, bool):
                try:
                    value = int(value)
                except:
                    pass
            elif isinstance(old, float):
                try:
                    value = float(value)
                except:
                    pass
            cur[keys[-1]] = value
        return True

    def save(self, path=None):
        p = Path(path or self.path)
        with self._lock:
            with open(p, "w", encoding="utf-8") as f:
                yaml.safe_dump(self._cfg, f, allow_unicode=True, sort_keys=False)
            self._last_mtime = p.stat().st_mtime
