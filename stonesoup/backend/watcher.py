"""Debounced filesystem watch for a single file (watchdog)."""

from __future__ import annotations

import logging
import threading
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

logger = logging.getLogger(__name__)


class _DebouncedHandler(FileSystemEventHandler):
    def __init__(self, target: Path, callback, debounce_s: float = 0.25) -> None:
        self.target = target.resolve()
        self.callback = callback
        self.debounce_s = debounce_s
        self._timer: threading.Timer | None = None
        self._lock = threading.Lock()

    def on_modified(self, event) -> None:
        if getattr(event, "is_directory", False):
            return
        p = Path(str(event.src_path)).resolve()
        if p != self.target:
            return
        self._schedule()

    def _schedule(self) -> None:
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(self.debounce_s, self._run_cb)
            self._timer.daemon = True
            self._timer.start()

    def _run_cb(self) -> None:
        with self._lock:
            self._timer = None
        try:
            self.callback()
        except Exception:
            logger.exception("stonesoup watch callback failed")


class FileWatcher:
    """Watch one file's parent directory for modifications to that file."""

    def __init__(self) -> None:
        self._observer: Observer | None = None
        self._handler: _DebouncedHandler | None = None

    def start(self, path: Path, callback) -> None:
        self.stop()
        path = path.resolve()
        parent = path.parent
        self._handler = _DebouncedHandler(path, callback)
        self._observer = Observer()
        self._observer.schedule(self._handler, str(parent), recursive=False)
        self._observer.start()

    def stop(self) -> None:
        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=3.0)
            self._observer = None
        self._handler = None
