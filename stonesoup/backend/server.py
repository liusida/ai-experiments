"""FastAPI server: watch file, parse cells, run cells, WebSocket push."""

from __future__ import annotations

import asyncio
import logging
import os
import queue
from collections import defaultdict
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from typing import Any

from pydantic import BaseModel

from stonesoup.backend.kernel import Cell, Kernel, parse_cells
from stonesoup.backend.watcher import FileWatcher

logger = logging.getLogger(__name__)

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


# --- app state ---


class AppState:
    def __init__(self) -> None:
        self.watched_path: Path | None = None
        self.cells: list[Cell] = []
        self.revision: int = 0
        self.last_changed_cell_indices: list[int] = []
        self.kernel = Kernel()
        self.watcher = FileWatcher()
        self.ws_clients: set[WebSocket] = set()
        self.loop: asyncio.AbstractEventLoop | None = None


state = AppState()


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
    source = state.cells[idx].source
    inject = dict(body.inject or {})
    source_path = str(state.watched_path.resolve()) if state.watched_path is not None else None

    chunk_queue: queue.Queue[tuple[str, str] | None] = queue.Queue()

    def on_stdout(s: str) -> None:
        if s:
            chunk_queue.put(("stdout", s))

    def on_stderr(s: str) -> None:
        if s:
            chunk_queue.put(("stderr", s))

    def worker() -> tuple[str, str, bool]:
        try:
            return state.kernel.run_cell(
                source,
                inject=inject,
                source_path=source_path,
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
async def api_reset() -> dict:
    state.kernel.reset()
    return {"ok": True}


@app.get("/api/kernel/vars")
async def api_kernel_vars() -> dict:
    """Current user-visible names in the shared kernel namespace (repr previews, truncated)."""
    return {"vars": state.kernel.snapshot_globals_for_ui()}


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
