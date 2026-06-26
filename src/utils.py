"""Utility helpers: config load, fps meter, logging."""

import yaml
import time
import logging
from pathlib import Path


def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logging(level="INFO"):
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("rgv")


class FPSMeter:
    def __init__(self, avg_window=30):
        self.times = []
        self.avg_window = avg_window

    def tick(self):
        now = time.time()
        self.times.append(now)
        # keep last N
        if len(self.times) > self.avg_window + 1:
            self.times = self.times[-(self.avg_window + 1) :]

    def fps(self):
        if len(self.times) < 2:
            return 0.0
        dt = self.times[-1] - self.times[0]
        if dt <= 0:
            return 0.0
        return (len(self.times) - 1) / dt
