"""Stonesoup: watched # %% cells + local Python kernel + floating UI.

- ``stonesoup.backend``: FastAPI server, file watcher, in-process kernel.
- ``stonesoup.frontend``: Vite browser UI (not importable as Python).
- ``stonesoup.experiment``: helpers for experiment code running *inside* a Stonesoup cell.

Inside a cell, call :func:`check_abort` periodically so the UI **Abort** button can stop long loops.
"""

from __future__ import annotations

from pathlib import Path

from stonesoup.backend.kernel import (
    RunAborted,
    StonesoupRunCancelled,
    check_abort,
)

__version__ = "0.1.0"

# First stdout line: UI strips it. Default output is plain text; use HTML/MD lines only for rich panes.
# See EXPERIMENT_PYTHON.md (stdout render hint).
STONESOUP_RENDER_AUTO = "# stonesoup:render=auto\n"
STONESOUP_RENDER_TEXT = "# stonesoup:render=text\n"
STONESOUP_RENDER_HTML = "# stonesoup:render=html\n"
STONESOUP_RENDER_MARKDOWN = "# stonesoup:render=markdown\n"
STONESOUP_RENDER_MD = "# stonesoup:render=md\n"


def repo_root() -> Path:
    """Return the repository root; same rules as :func:`stonesoup.experiment.paths.repo_root`."""
    from stonesoup.experiment.paths import repo_root as _repo_root

    return _repo_root()


def script_path() -> Path:
    """Absolute path to the watched/running experiment ``.py``; see :func:`stonesoup.experiment.paths.script_path`."""
    from stonesoup.experiment.paths import script_path as _sp

    return _sp()


def script_dir() -> Path:
    """Directory of the watched experiment file; see :func:`stonesoup.experiment.paths.script_dir`."""
    from stonesoup.experiment.paths import script_dir as _sd

    return _sd()


def outputs_dir() -> Path:
    """Per-script directory under ``outputs/…`` (HTTP ``/outputs``); see :func:`stonesoup.experiment.paths.outputs_dir`."""
    from stonesoup.experiment.paths import outputs_dir as _od

    return _od()


def plot_dir() -> Path:
    """Synonym for :func:`outputs_dir`; see :func:`stonesoup.experiment.paths.plot_dir`."""
    from stonesoup.experiment.paths import plot_dir as _pd

    return _pd()


def data_dir() -> Path:
    """``repo_root/data`` ensured on disk; see :func:`stonesoup.experiment.paths.data_dir`."""
    from stonesoup.experiment.paths import data_dir as _dd

    return _dd()


def stonesoup_render_prefix(mode: str) -> str:
    """First stdout line for Stonesoup (trailing newline). Rich: html, markdown/md; plain: text, auto."""
    m = mode.strip().lower()
    if m in ("auto", "text", "html", "markdown", "md"):
        return f"# stonesoup:render={m}\n"
    raise ValueError(f"Unknown stonesoup render mode: {mode!r}")


def list_loaded_models():
    """Re-export: bindings in this script's kernel (``name`` + ``repo_id``); weights may be shared globally."""
    from stonesoup.experiment import list_loaded_models as _list_loaded_models

    return _list_loaded_models()


def list_loaded_models_globally():
    """Re-export: every in-memory checkpoint in this Stonesoup process (``pool_key_b64``, ``repo_id``, …)."""
    from stonesoup.experiment import list_loaded_models_globally as _llg

    return _llg()


def list_hf_hub_cached_repo_ids():
    """Re-export: repo ids under the local Hugging Face Hub cache."""
    from stonesoup.experiment import list_hf_hub_cached_repo_ids as _list_cached

    return _list_cached()


def load_model(ref: str):
    """Re-export: (model, processor) from the shared pool; adds a binding in this kernel; Hub load if needed."""
    from stonesoup.experiment import load_model as _load_model

    return _load_model(ref)


def show(fig=None, *, basename=None, dpi=120, format="png", emit_render_hint=True, **kwargs):
    """Save a Matplotlib figure via :func:`outputs_dir` (optional ``basename=`` stem), print HTML, then ``plt.close(fig)``."""
    from stonesoup.experiment import show as _show

    return _show(fig, basename=basename, dpi=dpi, format=format, emit_render_hint=emit_render_hint, **kwargs)
