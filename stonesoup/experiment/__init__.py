"""Experiment-side API: helpers for ``# %%`` cells that talk to the Stonesoup kernel.

This layer is separate from ``stonesoup.backend`` (server/kernel implementation) and
``stonesoup/frontend`` (browser UI). Import it from notebooks and experiment scripts
to reduce boilerplate—e.g. binding to weights already in the Stonesoup **shared pool** (toolbar **Load**
or another script) without a second `from_pretrained`. It is only meaningful while a cell is executing
under Stonesoup (see each helper's docs).

Heavy imports (``models``, ``display.show``) load on first use so ``stonesoup.experiment.paths`` stays
light for the FastAPI server.
"""

from __future__ import annotations

import typing as _t

__all__ = [
    "STONESOUP_RENDER_AUTO",
    "STONESOUP_RENDER_HTML",
    "STONESOUP_RENDER_MARKDOWN",
    "STONESOUP_RENDER_MD",
    "STONESOUP_RENDER_TEXT",
    "capture_embed_and_post_blocks",
    "capture_pre_block0_and_post_blocks",
    "configure_matplotlib_agg",
    "data_dir",
    "decoder_blocks",
    "encode_text_inputs",
    "ensure_pad_token_via_eos",
    "hf_repo_id_safe_stem",
    "inner_tokenizer",
    "list_hf_hub_cached_repo_ids",
    "list_loaded_models",
    "list_loaded_models_globally",
    "load_model",
    "outputs_dir",
    "output_url_path",
    "plot_dir",
    "repo_root",
    "script_dir",
    "script_path",
    "show",
    "stonesoup_render_prefix",
    "unload_loaded_names_for_repo",
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
    if name == "unload_loaded_names_for_repo":
        from stonesoup.experiment.models import unload_loaded_names_for_repo as _ul

        return _ul
    if name == "list_loaded_models":
        from stonesoup.experiment.models import list_loaded_models as _list_lm

        return _list_lm
    if name == "list_loaded_models_globally":
        from stonesoup.experiment.models import list_loaded_models_globally as _llg

        return _llg
    if name == "list_hf_hub_cached_repo_ids":
        from stonesoup.experiment.models import list_hf_hub_cached_repo_ids as _list_hf

        return _list_hf
    if name == "show":
        from stonesoup.experiment.display import show as _show

        return _show
    if name == "output_url_path":
        from stonesoup.experiment.display import output_url_path as _oup

        return _oup
    if name == "configure_matplotlib_agg":
        from stonesoup.experiment.display import configure_matplotlib_agg as _cma

        return _cma
    if name == "hf_repo_id_safe_stem":
        from stonesoup.experiment.names import hf_repo_id_safe_stem as _stem

        return _stem
    if name == "inner_tokenizer":
        from stonesoup.experiment.hf_inputs import inner_tokenizer as _it

        return _it
    if name == "encode_text_inputs":
        from stonesoup.experiment.hf_inputs import encode_text_inputs as _eti

        return _eti
    if name == "ensure_pad_token_via_eos":
        from stonesoup.experiment.hf_inputs import ensure_pad_token_via_eos as _ep

        return _ep
    if name == "decoder_blocks":
        from stonesoup.experiment.lm_stack import decoder_blocks as _db

        return _db
    if name == "capture_embed_and_post_blocks":
        from stonesoup.experiment.hidden_hooks import capture_embed_and_post_blocks as _ceb

        return _ceb
    if name == "capture_pre_block0_and_post_blocks":
        from stonesoup.experiment.hidden_hooks import capture_pre_block0_and_post_blocks as _cpb

        return _cpb
    if name == "STONESOUP_RENDER_AUTO":
        from stonesoup import STONESOUP_RENDER_AUTO as _v

        return _v
    if name == "STONESOUP_RENDER_HTML":
        from stonesoup import STONESOUP_RENDER_HTML as _v

        return _v
    if name == "STONESOUP_RENDER_MARKDOWN":
        from stonesoup import STONESOUP_RENDER_MARKDOWN as _v

        return _v
    if name == "STONESOUP_RENDER_MD":
        from stonesoup import STONESOUP_RENDER_MD as _v

        return _v
    if name == "STONESOUP_RENDER_TEXT":
        from stonesoup import STONESOUP_RENDER_TEXT as _v

        return _v
    if name == "stonesoup_render_prefix":
        from stonesoup import stonesoup_render_prefix as _v

        return _v
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
