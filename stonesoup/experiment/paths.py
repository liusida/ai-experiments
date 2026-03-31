"""Repository root and watched-script paths for experiment code."""

from __future__ import annotations

import inspect
import os
from pathlib import Path

# Repo-relative; served under ``/outputs/stonesoup/`` (see ``stonesoup.show`` and backend ``/outputs`` mount).
CELL_WEB_OUTPUT_SUBPATH = Path("outputs") / "stonesoup"


def repo_root() -> Path:
    """Directory that contains ``stonesoup/`` and (in this repo) ``experiments/``.

    If :envvar:`STONESOUP_ROOT` is set, returns that path (same policy as the FastAPI server).
    Otherwise uses this file's location: ``stonesoup/experiment/paths.py`` → three parents up to
    the repo root — correct for an **editable** install of this project. For a non-editable install
    from a wheel, set ``STONESOUP_ROOT`` to your checkout.
    """
    raw = os.environ.get("STONESOUP_ROOT", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return Path(__file__).resolve().parent.parent.parent


def _stonesoup_package_dir() -> Path:
    """``stonesoup/`` install directory (parent of ``experiment/``)."""
    return Path(__file__).resolve().parent.parent


def script_path() -> Path:
    """Absolute path to the watched experiment ``.py`` (Stonesoup) or the running module's ``__file__``.

    In a Stonesoup cell the kernel sets ``__file__`` to the watched script. Otherwise walks the stack
    and returns the first ``__file__`` outside the ``stonesoup`` package (so wrappers like
    ``script_dir()`` are not confused with this module's ``__file__``).
    """
    from stonesoup.backend.kernel import active_kernel

    k = active_kernel.get()
    if k is not None:
        p = k.globals.get("__file__")
        if isinstance(p, str):
            return Path(p).resolve()
    pkg = _stonesoup_package_dir()
    frame = inspect.currentframe()
    try:
        f = frame.f_back if frame is not None else None
        while f is not None:
            p = f.f_globals.get("__file__")
            if isinstance(p, str):
                cand = Path(p).resolve()
                try:
                    cand.relative_to(pkg)
                except ValueError:
                    return cand
            f = f.f_back
    finally:
        del frame
    raise RuntimeError(
        "stonesoup.script_path(): run inside a Stonesoup cell, or call from a .py file "
        "(not an interactive REPL) so __file__ exists."
    )


def script_dir() -> Path:
    """Directory containing the experiment file (e.g. local ``reports/`` not web-served by default)."""
    return script_path().parent


def _plot_segment_relative_to_repo() -> Path:
    """``outputs/stonesoup/<path/to/script_stem>`` relative to repo root.

    Example: ``experiments/Demo/tmp_1.py`` → ``outputs/stonesoup/experiments/Demo/tmp_1``.
    """
    root = repo_root()
    rel_script = script_path().relative_to(root)
    return CELL_WEB_OUTPUT_SUBPATH / rel_script.with_suffix("")


def plot_dir() -> Path:
    """Per-script directory under ``outputs/stonesoup/`` (mirrors ``__file__`` relative to the repo).

    Same tree :func:`stonesoup.experiment.display.show` uses. Served under ``/outputs/…``. Created if
    missing (``mkdir(parents=True, exist_ok=True)``).
    """
    d = repo_root() / _plot_segment_relative_to_repo()
    d.mkdir(parents=True, exist_ok=True)
    return d


def data_dir() -> Path:
    """``repo_root() / "data"``, created if missing (shared checkout data)."""
    d = repo_root() / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d
