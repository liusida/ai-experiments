"""Repository root and watched-script paths for experiment code."""

from __future__ import annotations

import inspect
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Repo layout (see FastAPI ``/outputs`` mount in ``stonesoup.backend.server``)
# ---------------------------------------------------------------------------

# Directory under :func:`repo_root` served at HTTP ``/outputs``.
OUTPUTS_SEGMENT = Path("outputs")

# Cell artifacts live directly under ``outputs/`` (no ``stonesoup/`` segment): strip a leading
# ``experiments/`` from the repo-relative script path, drop ``.py``, mirror remaining dirs.
# Example: ``experiments/Demo/foo.py`` → ``outputs/Demo/foo``;
# ``experiments/2026-04-06-Hypersphere/embeddings.py`` → ``outputs/2026-04-06-Hypersphere/embeddings``.
STONESOUP_OUTPUTS_SEGMENT = OUTPUTS_SEGMENT

# Back-compat name: same tree as cell PNGs / caches (formerly suggested a ``stonesoup`` subdir).
CELL_WEB_OUTPUT_SUBPATH = OUTPUTS_SEGMENT


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


def repo_outputs_root() -> Path:
    """``repo_root() / "outputs"`` — directory mounted at HTTP ``/outputs``."""
    d = repo_root() / OUTPUTS_SEGMENT
    d.mkdir(parents=True, exist_ok=True)
    return d


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


def _per_script_outputs_subpath_relative_to_repo() -> Path:
    """``outputs/<…>/<script_stem>`` relative to repo root (leading ``experiments/`` stripped)."""
    root = repo_root()
    rel_script = script_path().relative_to(root)
    parts = rel_script.parts
    if parts[:1] == ("experiments",):
        parts = parts[1:]
    stem_rel = Path(*parts).with_suffix("") if parts else rel_script.with_suffix("")
    return STONESOUP_OUTPUTS_SEGMENT / stem_rel


def _plot_segment_relative_to_repo() -> Path:
    """Backward-compat name; same as :func:`_per_script_outputs_subpath_relative_to_repo`."""
    return _per_script_outputs_subpath_relative_to_repo()


def outputs_dir() -> Path:
    """Per-script directory under ``outputs/`` (mirrors ``__file__`` relative to the repo).

    Use this for any cell-produced files you want under the ``/outputs`` mount (plots, checkpoints,
    HTML assets, caches, etc.). Same tree as :func:`stonesoup.experiment.display.show`.

    Repo-relative path shape: ``outputs/<…>/<stem>/`` where a leading ``experiments/`` segment is
    removed from the script path and ``.py`` is dropped (same layout as on-disk cell outputs in
    this repo). Created if missing.
    """
    d = repo_root() / _per_script_outputs_subpath_relative_to_repo()
    d.mkdir(parents=True, exist_ok=True)
    return d


def plot_dir() -> Path:
    """Synonym for :func:`outputs_dir` (historical name: matplotlib figures)."""
    return outputs_dir()


def data_dir() -> Path:
    """``repo_root() / "data"``, created if missing (shared checkout data)."""
    d = repo_root() / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d
