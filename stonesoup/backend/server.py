"""FastAPI server: watch file, parse cells, run cells, WebSocket push."""

from __future__ import annotations

import asyncio
import logging
import os
import queue
import threading
import time
import traceback
from collections import OrderedDict, defaultdict
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import Any, Callable

from pydantic import BaseModel

from stonesoup.experiment.paths import repo_outputs_root, repo_root as stonesoup_root

from stonesoup.backend.hf_models import (
    ModelLoadRuntimeError,
    clear_experiment_kernel_namespace,
    list_hf_hub_cached_model_repo_ids,
    list_loaded_models,
    list_loaded_models_globally,
    load_models_into_kernel,
    pool_key_from_b64,
    release_all_managed_bindings,
    set_unload_diag_sink,
    unload_all_models_from_all_kernels,
    unload_binding_names_from_all_kernels,
    unload_models_from_kernel,
    unload_pool_keys_from_all_kernels,
)
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


def _repo_root_abs_posix() -> str:
    """Absolute repo root for editor deeplinks (POSIX slashes)."""
    return stonesoup_root().resolve().as_posix()


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
            oldest_key, oldest_kernel = kc.popitem(last=False)
            try:
                release_all_managed_bindings(oldest_kernel)
            except Exception:
                logger.exception("Evicted kernel model cleanup failed for %s", oldest_key)
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


def _require_watched_path() -> Path:
    watched = state.watched_path
    if watched is None or not watched.is_file():
        raise HTTPException(status_code=400, detail="Watch a Python file first.")
    return watched.resolve()


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
        # Set for the duration of ``POST /api/run``; ``POST /api/run/abort`` sets this event.
        self.run_cancel_event: threading.Event | None = None


state = AppState()


def _kernel_run_lock(path: Path) -> asyncio.Lock:
    key = _kernel_cache_key(path)
    lock = state.kernel_run_locks.get(key)
    if lock is None:
        lock = asyncio.Lock()
        state.kernel_run_locks[key] = lock
    return lock


@asynccontextmanager
async def _all_kernel_run_locks() -> AsyncIterator[None]:
    """Hold every cached kernel lock (sorted path keys) so global model unload cannot race cell runs."""
    keys = sorted(state.kernel_cache.keys())
    if not keys:
        yield
        return
    async with AsyncExitStack() as stack:
        for key in keys:
            lock = _kernel_run_lock(Path(key))
            await stack.enter_async_context(lock)
        yield


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
    # Repo-relative path matches script picker / Stonesoup localStorage keys (not absolute
    # ``Path`` strings, which broke layout restore on the first refresh after save).
    rel = (
        _repo_relative_display(state.watched_path)
        if state.watched_path and state.watched_path.is_file()
        else None
    )
    return {
        "type": "cells",
        "revision": state.revision,
        "path": rel,
        "repo_root": _repo_root_abs_posix(),
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


class ModelLoadItemBody(BaseModel):
    repo_id: str
    name: str | None = None
    # auto: pick by config (causal LM vs vision-language, etc.); causal_lm / image_text force one path
    model_kind: str | None = None


class ModelsLoadBody(BaseModel):
    items: list[ModelLoadItemBody]
    device_map: str | None = "auto"
    torch_dtype: str | None = None
    trust_remote_code: bool = False
    model_kind: str | None = "auto"


class ModelsUnloadBody(BaseModel):
    names: list[str] | None = None
    #: Decoded via :func:`stonesoup.backend.hf_models.pool_key_from_b64`; removes those checkpoints from **every** cached kernel (frees memory when refcounts hit zero).
    pool_keys_b64: list[str] | None = None


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    state.loop = asyncio.get_running_loop()
    try:
        yield
    finally:
        state.loop = None


app = FastAPI(title="Stonesoup", version="0.1.0", lifespan=_lifespan)

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

_data_images_dir = stonesoup_root() / "data" / "images"
if _data_images_dir.is_dir():
    app.mount(
        "/data/image",
        StaticFiles(directory=str(_data_images_dir.resolve()), check_dir=True),
        name="data_images",
    )
else:
    logger.warning("Static data/images directory not found; skipping /data/image mount: %s", _data_images_dir)

_outputs_root = repo_outputs_root()
try:
    app.mount(
        "/outputs",
        StaticFiles(directory=str(_outputs_root.resolve()), check_dir=True),
        name="outputs",
    )
except OSError as exc:
    logger.warning("Could not mount /outputs (cell images): %s", exc)


@app.get("/api/health")
async def health() -> dict:
    return {"ok": True, "repo_root": _repo_root_abs_posix()}


def _mtime_or_zero(path: Path) -> float:
    try:
        return float(path.stat().st_mtime)
    except OSError:
        return 0.0


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
        return {"dir": "", "files": [], "mtimes": [], "recursive": recursive}
    folder = safe_dir_under_root(d)
    root = stonesoup_root()
    files: list[str] = []
    mtimes: list[float] = []
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
            mtimes.append(_mtime_or_zero(p))
    else:
        for p in sorted(folder.glob("*.py")):
            if p.is_file():
                files.append(p.relative_to(root).as_posix())
                mtimes.append(_mtime_or_zero(p))
    return {
        "dir": d.replace("\\", "/"),
        "files": files,
        "mtimes": mtimes,
        "recursive": recursive,
    }


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
        "repo_root": cp["repo_root"],
    }


@app.get("/api/cells")
async def api_cells() -> dict:
    rel = (
        _repo_relative_display(state.watched_path)
        if state.watched_path and state.watched_path.is_file()
        else None
    )
    return {
        "revision": state.revision,
        "path": rel,
        "repo_root": _repo_root_abs_posix(),
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
        state.run_cancel_event = threading.Event()
        try:
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

            cancel_ev = state.run_cancel_event

            def worker() -> tuple[str, str, bool, float]:
                try:
                    t0 = time.perf_counter()
                    stdout, stderr, ok = kernel.run_cell(
                        source,
                        inject=inject,
                        source_path=source_path,
                        start_line=state.cells[idx].start_line,
                        on_stdout_chunk=on_stdout,
                        on_stderr_chunk=on_stderr,
                        cancel_event=cancel_ev,
                    )
                    elapsed_s = time.perf_counter() - t0
                    return stdout, stderr, ok, elapsed_s
                finally:
                    chunk_queue.put(None)

            stdout, stderr = "", ""
            ok = False
            elapsed_s = 0.0
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
                stdout, stderr, ok, elapsed_s = await fut
            except Exception:
                logger.exception("api_run failed")
                ok = False
                tb = traceback.format_exc()
                stderr = f"{stderr}\n{tb}" if stderr else tb
            finally:
                await _broadcast_ws_json({"type": "run_end", "cell_index": idx, "ok": ok})

            return {
                "ok": ok,
                "cell_index": idx,
                "stdout": stdout,
                "stderr": stderr,
                "duration_sec": elapsed_s,
            }
        finally:
            state.run_cancel_event = None


@app.post("/api/run/abort")
async def api_run_abort() -> dict[str, Any]:
    """Set the cooperative cancel flag for the in-flight ``POST /api/run`` (if any)."""
    ev = state.run_cancel_event
    signaled = ev is not None
    if signaled:
        ev.set()
    return {"ok": True, "signaled": signaled}


@app.get("/api/models")
async def api_models() -> dict:
    watched_path = state.watched_path.resolve() if state.watched_path is not None else None
    kernel = _kernel_for_watched_path(watched_path)
    return {
        "watched_path": str(watched_path) if watched_path is not None else None,
        "loaded": list_loaded_models(kernel),
        "loaded_globally": list_loaded_models_globally(),
    }


@app.get("/api/models/hf-cache")
async def api_models_hf_cache() -> dict:
    """Repo ids for models found under the local Hugging Face Hub cache (on this machine)."""
    repo_ids = await asyncio.to_thread(list_hf_hub_cached_model_repo_ids)
    return {"repo_ids": repo_ids}


@app.post("/api/models/load")
async def api_models_load(body: ModelsLoadBody) -> dict:
    if not body.items:
        raise HTTPException(status_code=400, detail="Provide at least one model to load.")
    watched = _require_watched_path()
    items_arg = [
        {"repo_id": item.repo_id, "name": item.name, "model_kind": item.model_kind}
        for item in body.items
    ]

    async def _models_load_job() -> None:
        """Serialize on ``_kernel_run_lock`` with cell runs; errors go to WS ``app_log_end`` only."""
        log_ok = True
        log_error: str | None = None
        try:
            async with _kernel_run_lock(watched):
                kernel = _kernel_for_watched_path(watched)
                if kernel is None:
                    log_ok = False
                    log_error = "Watched file is not available"
                else:
                    chunk_queue: queue.Queue[tuple[str, str] | None] = queue.Queue()

                    def on_stdout(s: str) -> None:
                        if s:
                            chunk_queue.put(("stdout", s))

                    def on_stderr(s: str) -> None:
                        if s:
                            chunk_queue.put(("stderr", s))

                    await _broadcast_ws_json({"type": "app_log_start", "op": "model_load"})
                    try:

                        def worker() -> dict[str, Any]:
                            try:
                                return load_models_into_kernel(
                                    kernel,
                                    items=items_arg,
                                    device_map=body.device_map,
                                    torch_dtype=body.torch_dtype,
                                    trust_remote_code=body.trust_remote_code,
                                    default_model_kind=body.model_kind,
                                    on_stdout_chunk=on_stdout,
                                    on_stderr_chunk=on_stderr,
                                )
                            finally:
                                chunk_queue.put(None)

                        fut = asyncio.create_task(asyncio.to_thread(worker))
                        while True:
                            item = await asyncio.to_thread(chunk_queue.get)
                            if item is None:
                                break
                            stream_name, text = item
                            await _broadcast_ws_json(
                                {"type": "app_log", "stream": stream_name, "text": text}
                            )
                        await fut
                    except (ModelLoadRuntimeError, ValueError) as exc:
                        log_ok = False
                        log_error = str(exc)
                    except Exception as exc:
                        log_ok = False
                        log_error = str(exc)
                        logger.exception("Failed to load Hugging Face models")
        except Exception as exc:
            log_ok = False
            if log_error is None:
                log_error = str(exc)
            logger.exception("Model load task failed before or during lock")
        finally:
            end_payload: dict[str, Any] = {"type": "app_log_end", "ok": log_ok, "op": "model_load"}
            if log_error:
                end_payload["error"] = log_error
            await _broadcast_ws_json(end_payload)

    asyncio.create_task(_models_load_job())
    return {"ok": True, "accepted": True}


async def _run_unload_in_thread_with_diag(
    unload_diag: bool,
    fn: Callable[..., dict[str, Any]],
    *args: Any,
) -> dict[str, Any]:
    """Run unload in a worker thread; optionally stream ``STONESOUP_UNLOAD_DEBUG`` lines to the UI Console."""
    log_queue: queue.Queue[str | None] = queue.Queue()

    def sink(line: str) -> None:
        log_queue.put(line)

    def worker() -> dict[str, Any]:
        if unload_diag:
            set_unload_diag_sink(sink)
        try:
            return fn(*args)
        finally:
            if unload_diag:
                set_unload_diag_sink(None)
                log_queue.put(None)

    task = asyncio.create_task(asyncio.to_thread(worker))
    if unload_diag:
        await _broadcast_ws_json({"type": "app_log_start", "op": "model_unload"})
        while True:
            item = await asyncio.to_thread(log_queue.get)
            if item is None:
                break
            text = item if item.endswith("\n") else item + "\n"
            await _broadcast_ws_json({"type": "app_log", "stream": "stdout", "text": text})
    try:
        result = await task
    except Exception as exc:
        if unload_diag:
            await _broadcast_ws_json(
                {"type": "app_log_end", "ok": False, "op": "model_unload", "error": str(exc)}
            )
        raise
    if unload_diag:
        await _broadcast_ws_json({"type": "app_log_end", "ok": True, "op": "model_unload"})
    return result


@app.post("/api/models/unload")
async def api_models_unload(body: ModelsUnloadBody) -> dict:
    pool_raw = body.pool_keys_b64
    use_pool = pool_raw is not None and any(str(x).strip() for x in pool_raw)
    if use_pool:
        if body.names is not None:
            raise HTTPException(
                status_code=400,
                detail="Use either pool_keys_b64 or names, not both.",
            )
        decoded: list[str] = []
        for x in pool_raw or []:
            s = str(x).strip()
            if not s:
                continue
            try:
                decoded.append(pool_key_from_b64(s))
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        if not decoded:
            raise HTTPException(status_code=400, detail="Provide at least one pool_key_b64.")
    elif body.names is not None and not body.names:
        raise HTTPException(status_code=400, detail="Provide at least one model name or null for all.")
    unload_diag = os.environ.get("STONESOUP_UNLOAD_DEBUG", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    async with _all_kernel_run_locks():
        kernels = list(state.kernel_cache.values())
        try:
            if kernels:
                if use_pool:
                    result = await _run_unload_in_thread_with_diag(
                        unload_diag,
                        unload_pool_keys_from_all_kernels,
                        kernels,
                        decoded,
                    )
                elif body.names is None:
                    result = await _run_unload_in_thread_with_diag(
                        unload_diag,
                        unload_all_models_from_all_kernels,
                        kernels,
                    )
                else:
                    result = await _run_unload_in_thread_with_diag(
                        unload_diag,
                        unload_binding_names_from_all_kernels,
                        kernels,
                        body.names,
                    )
            else:
                result = {"unloaded": [], "missing": []}
        except Exception as exc:
            logger.exception("Failed to unload Hugging Face models")
            raise HTTPException(status_code=500, detail=f"Failed to unload model(s): {exc}") from exc
        watched_path = state.watched_path.resolve() if state.watched_path is not None and state.watched_path.is_file() else None
        watched_k = _kernel_for_watched_path(watched_path)
        result["loaded_now"] = list_loaded_models(watched_k)
    return {"ok": True, **result}


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


@app.post("/api/kernel/free-memory")
async def api_kernel_free_memory() -> dict[str, Any]:
    """Clear the **current** watched file's kernel: all variables and HF bindings for this session.

    Shared checkpoints may stay resident if other experiments still reference them.
    """
    watched = _require_watched_path()
    async with _kernel_run_lock(watched):
        kernel = _kernel_for_watched_path(watched)
        if kernel is None:
            return {"ok": True, "cleared": [], "loaded_now": []}
        result = await asyncio.to_thread(clear_experiment_kernel_namespace, kernel)
    return {"ok": True, **result}


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
