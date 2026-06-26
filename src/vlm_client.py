"""Async VLM client for Ollama or transformers Florence-2 / Moondream."""

import threading
import queue
import time
import base64
import cv2
from typing import Optional

try:
    import requests

    REQUESTS_AVAILABLE = True
except Exception:
    REQUESTS_AVAILABLE = False


class VLMWorker(threading.Thread):
    def __init__(
        self,
        enabled=False,
        provider="ollama",
        model="moondream:latest",
        base_url="http://localhost:11434",
        prompt="",
        interval=3,
        timeout_ms=2000,
    ):
        super().__init__(daemon=True)
        self.enabled = enabled
        self.provider = provider
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.prompt = prompt
        self.interval = interval
        self.timeout = timeout_ms / 1000.0
        self.q = queue.Queue(maxsize=1)
        self.latest_result = ""
        self._stop = False
        self.frame_counter = 0
        if enabled:
            print(f"[vlm] enabled provider={provider} model={model}")

    def submit(self, frame, frame_idx):
        if not self.enabled:
            return
        if frame_idx % self.interval != 0:
            return
        # drop if queue full
        if self.q.full():
            try:
                self.q.get_nowait()
            except:
                pass
        try:
            self.q.put_nowait((frame.copy(), frame_idx))
        except:
            pass

    def run(self):
        while not self._stop:
            try:
                frame, idx = self.q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                if self.provider == "ollama":
                    result = self._ollama_infer(frame)
                else:
                    result = self._transformers_infer(frame)
                self.latest_result = result
            except Exception as e:
                self.latest_result = f"[vlm error] {e}"

    def _encode_jpeg_base64(self, frame):
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buf).decode()

    def _ollama_infer(self, frame):
        if not REQUESTS_AVAILABLE:
            return "requests not available"
        b64 = self._encode_jpeg_base64(frame)
        payload = {
            "model": self.model,
            "prompt": self.prompt,
            "stream": False,
            "images": [b64],
            "options": {"temperature": 0.2, "num_predict": 128},
        }
        try:
            r = requests.post(
                f"{self.base_url}/api/generate", json=payload, timeout=self.timeout + 5
            )
            if r.status_code == 200:
                return r.json().get("response", "").strip()
            else:
                return f"ollama http {r.status_code}"
        except Exception as e:
            return f"ollama error {e}"

    def _transformers_infer(self, frame):
        # Lazy load Florence-2 or Moondream via transformers on first use
        # Simplified placeholder: implement if needed. For now return not implemented.
        return "transformers provider not implemented in skeleton, use ollama"

    def get_latest(self):
        return self.latest_result

    def stop(self):
        self._stop = True
