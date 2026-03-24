"""Debounced filesystem watch for a single file (watchdog)."""

from __future__ import annotations

import logging
import os
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

    def _is_target(self, path_str: str | bytes) -> bool:
        try:
            p = Path(os.fsdecode(path_str)).resolve()
        except OSError:
            return False
        return p == self.target

    def on_modified(self, event) -> None:
        if getattr(event, "is_directory", False):
            return
        if self._is_target(getattr(event, "src_path", "")):
            self._schedule()

    def on_created(self, event) -> None:
        """Atomic save (write temp + rename) often emits *created* for the final file."""
        if getattr(event, "is_directory", False):
            return
        if self._is_target(getattr(event, "src_path", "")):
            self._schedule()

    def on_moved(self, event) -> None:
        """Same as *created* when the editor renames a temp file over the watched path."""
        if getattr(event, "is_directory", False):
            return
        if self._is_target(getattr(event, "dest_path", "")):
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
