"""Small terminal-only UI helpers for the interactive CodeFlow CLI."""

import sys
import threading
from contextlib import contextmanager


class ThinkingIndicator:
    """Show a lightweight progress animation while a model request is running.

    The indicator is deliberately generic. It tells the user that CodeFlow is
    waiting on model/tool work without exposing or pretending to display the
    model's private reasoning.
    """

    _FRAMES = ("|", "/", "-", "\\")

    def __init__(self, stream=None, interval=0.25, enabled=None):
        self.stream = stream or sys.stdout
        self.interval = interval
        self.enabled = self.stream.isatty() if enabled is None else bool(enabled)
        self._stop = threading.Event()
        self._thread = None
        self._last_width = 0

    def start(self):
        if not self.enabled or self._thread is not None:
            return self
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="codeflow-thinking", daemon=True)
        self._thread.start()
        return self

    def stop(self):
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=max(self.interval * 2, 0.1))
        self._thread = None
        self._clear()

    def _run(self):
        frame_index = 0
        while not self._stop.is_set():
            self._write(f"CodeFlow 思考中 {self._FRAMES[frame_index % len(self._FRAMES)]}")
            frame_index += 1
            self._stop.wait(self.interval)

    def _write(self, text):
        padding = max(self._last_width - len(text), 0)
        self._last_width = len(text)
        self.stream.write("\r" + text + (" " * padding))
        self.stream.flush()

    def _clear(self):
        if not self.enabled:
            return
        self.stream.write("\r" + (" " * self._last_width) + "\r")
        self.stream.flush()
        self._last_width = 0


@contextmanager
def thinking_indicator(stream=None, interval=0.25, enabled=None):
    indicator = ThinkingIndicator(stream=stream, interval=interval, enabled=enabled)
    indicator.start()
    try:
        yield indicator
    finally:
        indicator.stop()
