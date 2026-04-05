"""Experiment-side API: helpers for ``# %%`` cells that talk to the Stonesoup kernel.

This layer is separate from ``stonesoup.backend`` (server/kernel implementation) and
``stonesoup/frontend`` (browser UI). Import it from notebooks and experiment scripts
to reduce boilerplate—e.g. resolving models already loaded in the UI without duplicating
weights. It is only meaningful while a cell is executing under Stonesoup (see each helper's docs).

Heavy imports (``models``, ``display.show``) load on first use so ``stonesoup.experiment.paths`` stays
light for the FastAPI server.
"""

from __future__ import annotations

import typing as _t

__all__ = [
    "data_dir",
    "list_hf_hub_cached_repo_ids",
    "list_loaded_models",
    "load_model",
    "outputs_dir",
    "plot_dir",
    "repo_root",
    "script_dir",
    "script_path",
    "show",
]


def __getattr__(name: str) -> _t.Any:
    if name == "repo_root":
        from stonesoup.experiment.paths import repo_root as _repo_root

        return _repo_root
    if name == "script_path":
        from stonesoup.experiment.paths import script_path as _script_path

        return _script_path
    if name == "script_dir":
        from stonesoup.experiment.paths import script_dir as _script_dir

        return _script_dir
    if name == "outputs_dir":
        from stonesoup.experiment.paths import outputs_dir as _outputs_dir

        return _outputs_dir
    if name == "plot_dir":
        from stonesoup.experiment.paths import plot_dir as _plot_dir

        return _plot_dir
    if name == "data_dir":
        from stonesoup.experiment.paths import data_dir as _data_dir

        return _data_dir
    if name == "load_model":
        from stonesoup.experiment.models import load_model as _load_model

        return _load_model
    if name == "list_loaded_models":
        from stonesoup.experiment.models import list_loaded_models as _list_lm

        return _list_lm
    if name == "list_hf_hub_cached_repo_ids":
        from stonesoup.experiment.models import list_hf_hub_cached_repo_ids as _list_hf

        return _list_hf
    if name == "show":
        from stonesoup.experiment.display import show as _show

        return _show
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
