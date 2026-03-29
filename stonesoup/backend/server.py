"""FastAPI server: watch file, parse cells, run cells, WebSocket push."""

from __future__ import annotations

import asyncio
import logging
import os
import queue
from collections import OrderedDict, defaultdict
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from typing import Any

from pydantic import BaseModel

from stonesoup.backend.kernel import Cell, Kernel, parse_cells
from stonesoup.backend.watcher import FileWatcher

logger = logging.getLogger(__name__)


def _restart_server_process() -> None:
    """Replace this process with a new Python interpreter (same command line and environment).

    Stops in-flight cell runs and returns RAM to the OS. Requires no external supervisor.
    """
    import sys

    exe = sys.executable
    argv = sys.argv
    if not argv:
        os.execv(exe, [exe])
    try:
        same_interpreter = os.path.samefile(argv[0], exe)
    except OSError:
        same_interpreter = os.path.abspath(argv[0]) == os.path.abspath(exe)
    new_argv = [exe, *argv[1:]] if same_interpreter else [exe, *argv]
    logger.info("Restarting Stonesoup server via exec (same argv as this process)")
    os.execv(exe, new_argv)


# --- path policy ---


def stonesoup_root() -> Path:
    env = os.environ.get("STONESOUP_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    # stonesoup/backend/server.py -> repo root is parent of stonesoup/
    return Path(__file__).resolve().parent.parent.parent


def safe_py_path(path_str: str) -> Path:
    root = stonesoup_root()
    raw = Path(path_str).expanduser()
    candidate = (raw.resolve() if raw.is_absolute() else (root / raw).resolve())
    try:
        candidate.relative_to(root)
    except ValueError:
        raise HTTPException(status_code=400, detail="Path must be under STONESOUP_ROOT")
    if candidate.suffix.lower() != ".py":
        raise HTTPException(status_code=400, detail="Only .py files are allowed")
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return candidate


def safe_dir_under_root(path_str: str) -> Path:
    """Resolve a repo-relative (or absolute under root) directory; reject escapes."""
    root = stonesoup_root()
    raw = Path(path_str.strip()).expanduser()
    candidate = (raw.resolve() if raw.is_absolute() else (root / raw).resolve())
    try:
        candidate.relative_to(root)
    except ValueError:
        raise HTTPException(status_code=400, detail="Path must be under STONESOUP_ROOT")
    if not candidate.exists():
        raise HTTPException(status_code=404, detail="Directory not found")
    if not candidate.is_dir():
        raise HTTPException(status_code=400, detail="Not a directory")
    return candidate


_DEFAULT_KERNEL_CACHE_MAX = 32


def _kernel_cache_max() -> int:
    raw = os.environ.get("STONESOUP_KERNEL_CACHE_MAX", "").strip()
    if not raw:
        return _DEFAULT_KERNEL_CACHE_MAX
    try:
        n = int(raw)
        return max(1, min(n, 10_000))
    except ValueError:
        return _DEFAULT_KERNEL_CACHE_MAX


def _kernel_cache_key(path: Path) -> str:
    return path.resolve().as_posix()


def _repo_relative_display(path: Path) -> str:
    root = stonesoup_root().resolve()
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return path.name


def _get_or_create_kernel(path: Path) -> Kernel:
    """LRU-cached ``Kernel`` for this resolved script path (MRU at OrderedDict end)."""
    key = _kernel_cache_key(path)
    kc = state.kernel_cache
    if key not in kc:
        while len(kc) >= _kernel_cache_max():
            oldest_key, _ = kc.popitem(last=False)
            logger.info(
                "Evicted kernel cache for %s (cap=%d)",
                oldest_key,
                _kernel_cache_max(),
            )
        kc[key] = Kernel()
    kc.move_to_end(key)
    return kc[key]


def _kernel_for_watched_path(path: Path | None) -> Kernel | None:
    if path is None or not path.is_file():
        return None
    return _get_or_create_kernel(path)


def _kernel_sessions_payload() -> tuple[str | None, list[dict[str, Any]]]:
    watched_path_str = str(state.watched_path) if state.watched_path else None
    current_key: str | None = None
    if state.watched_path is not None and state.watched_path.is_file():
        current_key = _kernel_cache_key(state.watched_path)

    sessions: list[dict[str, Any]] = []
    ordered_keys = list(state.kernel_cache.keys())
    seen: set[str] = set()
    if current_key and current_key in state.kernel_cache:
        kernel = state.kernel_cache[current_key]
        sessions.append(
            {
                "path": _repo_relative_display(Path(current_key)),
                "n_vars": len(kernel.snapshot_globals_for_ui()),
                "current": True,
            }
        )
        seen.add(current_key)
    for key in reversed(ordered_keys):
        if key in seen:
            continue
        kernel = state.kernel_cache[key]
        sessions.append(
            {
                "path": _repo_relative_display(Path(key)),
                "n_vars": len(kernel.snapshot_globals_for_ui()),
                "current": False,
            }
        )

    return watched_path_str, sessions


# --- app state ---


class AppState:
    def __init__(self) -> None:
        self.watched_path: Path | None = None
        self.cells: list[Cell] = []
        self.revision: int = 0
        self.last_changed_cell_indices: list[int] = []
        self.kernel_cache: OrderedDict[str, Kernel] = OrderedDict()
        # One lock per resolved script path — Kernel is shared mutable state per file.
        self.kernel_run_locks: dict[str, asyncio.Lock] = {}
        self.watcher = FileWatcher()
        self.ws_clients: set[WebSocket] = set()
        self.loop: asyncio.AbstractEventLoop | None = None


state = AppState()


def _kernel_run_lock(path: Path) -> asyncio.Lock:
    key = _kernel_cache_key(path)
    lock = state.kernel_run_locks.get(key)
    if lock is None:
        lock = asyncio.Lock()
        state.kernel_run_locks[key] = lock
    return lock


def _changed_cell_indices(old: list[Cell], new: list[Cell]) -> list[int]:
    """
    Indices in ``new`` whose body changed or that are new slots for a given ``marker_key``.

    Cells are matched by ``marker_key`` (hash of the opening ``# %%`` line, or an implicit-head
    sentinel), then paired in file order within each key — so inserting/deleting other cells does
    not falsely mark unrelated markers as updated.
    """
    if not old:
        return []
    by_old: dict[str, list[Cell]] = defaultdict(list)
    for c in old:
        by_old[c.marker_key].append(c)
    by_new: dict[str, list[tuple[int, Cell]]] = defaultdict(list)
    for i, c in enumerate(new):
        by_new[c.marker_key].append((i, c))
    changed: set[int] = set()
    for fp, new_items in by_new.items():
        olds = by_old.get(fp, [])
        for j, (ni, nc) in enumerate(new_items):
            if j < len(olds):
                if olds[j].source != nc.source:
                    changed.add(ni)
            else:
                changed.add(ni)
    return sorted(changed)


def _reload_from_disk_sync() -> None:
    if state.watched_path is None or not state.watched_path.is_file():
        return
    text = state.watched_path.read_text(encoding="utf-8", errors="replace")
    old_cells = state.cells
    new_cells = parse_cells(text)
    state.last_changed_cell_indices = _changed_cell_indices(old_cells, new_cells)
    state.cells = new_cells
    state.revision += 1
    logger.info(
        "Reloaded %s -> %d cells rev=%d changed=%s",
        state.watched_path,
        len(state.cells),
        state.revision,
        state.last_changed_cell_indices,
    )


def _schedule_broadcast() -> None:
    loop = state.loop
    if loop is None or not loop.is_running():
        return
    asyncio.run_coroutine_threadsafe(_broadcast_cells(), loop)


def _on_file_changed() -> None:
    try:
        _reload_from_disk_sync()
    except Exception:
        logger.exception("reload failed")
    _schedule_broadcast()


async def _broadcast_ws_json(payload: dict[str, Any]) -> None:
    dead: list[WebSocket] = []
    for sock in state.ws_clients:
        try:
            await sock.send_json(payload)
        except Exception:
            dead.append(sock)
    for sock in dead:
        state.ws_clients.discard(sock)


def _cells_payload() -> dict[str, Any]:
    return {
        "type": "cells",
        "revision": state.revision,
        "path": str(state.watched_path) if state.watched_path else None,
        "cells": [c.to_dict() for c in state.cells],
        "changed_cell_indices": list(state.last_changed_cell_indices),
    }


async def _broadcast_cells() -> None:
    await _broadcast_ws_json(_cells_payload())


# --- FastAPI ---


class WatchBody(BaseModel):
    path: str


class RunBody(BaseModel):
    cell_index: int
    # Optional names merged into the kernel globals before this cell runs.
    inject: dict[str, Any] | None = None


app = FastAPI(title="Stonesoup", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "http://127.0.0.1:5174",
        "http://localhost:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def _startup() -> None:
    state.loop = asyncio.get_running_loop()


@app.get("/api/health")
async def health() -> dict:
    return {"ok": True}


@app.get("/api/py-files")
async def api_py_files(
    subdir: str = Query(
        "",
        alias="dir",
        description="Repo-relative directory; lists *.py there, or recursively if recursive=true.",
    ),
    recursive: bool = Query(
        False,
        description="If true, list every *.py under dir (nested subfolders).",
    ),
) -> dict:
    """List ``.py`` files under a subdirectory of the repo (for UI dropdown)."""
    d = subdir.strip()
    if not d:
        return {"dir": "", "files": [], "recursive": recursive}
    folder = safe_dir_under_root(d)
    root = stonesoup_root()
    files: list[str] = []
    if recursive:
        for p in sorted(folder.rglob("*.py")):
            if not p.is_file():
                continue
            try:
                rp = p.resolve()
                rel = rp.relative_to(root.resolve())
            except ValueError:
                continue
            files.append(rel.as_posix())
    else:
        for p in sorted(folder.glob("*.py")):
            if p.is_file():
                files.append(p.relative_to(root).as_posix())
    return {"dir": d.replace("\\", "/"), "files": files, "recursive": recursive}


@app.post("/api/watch")
async def api_watch(body: WatchBody) -> dict:
    path = safe_py_path(body.path)
    state.watcher.stop()
    state.watched_path = path
    state.cells = []  # do not diff against a different previously watched file
    _reload_from_disk_sync()
    state.watcher.start(path, _on_file_changed)
    await _broadcast_cells()
    cp = _cells_payload()
    return {
        "ok": True,
        "path": cp["path"],
        "revision": cp["revision"],
        "n_cells": len(state.cells),
        "cells": cp["cells"],
        "changed_cell_indices": cp["changed_cell_indices"],
    }


@app.get("/api/cells")
async def api_cells() -> dict:
    return {
        "revision": state.revision,
        "path": str(state.watched_path) if state.watched_path else None,
        "cells": [c.to_dict() for c in state.cells],
        "changed_cell_indices": list(state.last_changed_cell_indices),
    }


@app.post("/api/run")
async def api_run(body: RunBody) -> dict:
    if not state.cells:
        raise HTTPException(status_code=400, detail="No cells loaded; POST /api/watch first")
    idx = body.cell_index
    if idx < 0 or idx >= len(state.cells):
        raise HTTPException(status_code=400, detail=f"cell_index out of range 0..{len(state.cells)-1}")
    watched = state.watched_path
    if watched is None or not watched.is_file():
        raise HTTPException(status_code=400, detail="Watched file is not available")

    async with _kernel_run_lock(watched.resolve()):
        source = state.cells[idx].source
        inject = dict(body.inject or {})
        source_path = str(watched.resolve())

        chunk_queue: queue.Queue[tuple[str, str] | None] = queue.Queue()

        def on_stdout(s: str) -> None:
            if s:
                chunk_queue.put(("stdout", s))

        def on_stderr(s: str) -> None:
            if s:
                chunk_queue.put(("stderr", s))

        kernel = _kernel_for_watched_path(watched)
        if kernel is None:
            raise HTTPException(status_code=400, detail="Watched file is not available")

        def worker() -> tuple[str, str, bool]:
            try:
                return kernel.run_cell(
                    source,
                    inject=inject,
                    source_path=source_path,
                    start_line=state.cells[idx].start_line,
                    on_stdout_chunk=on_stdout,
                    on_stderr_chunk=on_stderr,
                )
            finally:
                chunk_queue.put(None)

        stdout, stderr = "", ""
        ok = False
        await _broadcast_ws_json({"type": "run_start", "cell_index": idx})
        try:
            fut = asyncio.create_task(asyncio.to_thread(worker))
            while True:
                item = await asyncio.to_thread(chunk_queue.get)
                if item is None:
                    break
                stream_name, text = item
                await _broadcast_ws_json(
                    {
                        "type": "run_stream",
                        "cell_index": idx,
                        "stream": stream_name,
                        "text": text,
                    }
                )
            stdout, stderr, ok = await fut
            return {"ok": ok, "cell_index": idx, "stdout": stdout, "stderr": stderr}
        finally:
            await _broadcast_ws_json({"type": "run_end", "cell_index": idx, "ok": ok})


@app.post("/api/reset")
async def api_reset(background_tasks: BackgroundTasks) -> dict:
    """Restart the backend process (fresh interpreter); UI should reconnect after a short wait."""
    background_tasks.add_task(_restart_server_process)
    return {"ok": True, "restarting": True}


@app.get("/api/kernel/vars")
async def api_kernel_vars() -> dict:
    """Variable table for the current watch target; ``sessions`` lists all cached kernels (per file)."""
    watched_path_str, sessions = _kernel_sessions_payload()
    ker = _kernel_for_watched_path(state.watched_path)
    vars_rows = ker.snapshot_globals_for_ui() if ker is not None else []
    return {
        "watched_path": watched_path_str,
        "vars": vars_rows,
        "sessions": sessions,
    }


@app.websocket("/ws")
async def websocket_cells(ws: WebSocket) -> None:
    await ws.accept()
    state.ws_clients.add(ws)
    try:
        await ws.send_json(_cells_payload())
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        state.ws_clients.discard(ws)


def main() -> None:
    import uvicorn

    host = os.environ.get("STONESOUP_HOST", "127.0.0.1")
    port = int(os.environ.get("STONESOUP_PORT", "8765"))
    reload = os.environ.get("STONESOUP_RELOAD", "").strip().lower() in ("1", "true", "yes")
    # Default uvicorn reload watches the process cwd (often repo root), which would restart on every
    # experiment ``*.py`` save. Only watch the ``stonesoup`` package so reload is for server code only.
    reload_dirs: list[str] | None = None
    if reload:
        stonesoup_package = Path(__file__).resolve().parent.parent
        reload_dirs = [str(stonesoup_package)]
    uvicorn.run(
        "stonesoup.backend.server:app",
        host=host,
        port=port,
        reload=reload,
        reload_dirs=reload_dirs,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
