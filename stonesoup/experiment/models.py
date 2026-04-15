"""Model handles: experiment code uses the same shared HF weights as the Stonesoup UI."""

from __future__ import annotations

from typing import Any


def list_loaded_models_globally() -> list[dict[str, Any]]:
    """Return every checkpoint currently resident in this Stonesoup server process (shared pool).

    Rows include ``pool_key_b64``, ``repo_id``, ``model_kind``, ``torch_dtype``, etc.—the same
    data that powers the toolbar dropdown. Meaningful only while the Stonesoup backend is running.
    """
    from stonesoup.backend.hf_models import list_loaded_models_globally as _lg

    return _lg()


def list_loaded_models() -> list[dict[str, str]]:
    """Return ``[{name, repo_id}, ...]`` for **this script's kernel** bindings to shared weights.

    Weights may already live in the process-wide pool (toolbar **Load**, another file, or an
    earlier ``load_model``); this lists the names bound into the **current** watched file's
    namespace only. Outside a Stonesoup cell, ``active_kernel`` is unset and this returns [].
    """
    from stonesoup.backend.hf_models import list_loaded_models as _list
    from stonesoup.backend.kernel import active_kernel

    return _list(active_kernel.get())


def list_hf_hub_cached_repo_ids() -> list[str]:
    """Hugging Face Hub repo ids found under the local cache (same scan as the UI datalist)."""
    from stonesoup.backend.hf_models import list_hf_hub_cached_model_repo_ids

    return list_hf_hub_cached_model_repo_ids()


def load_model(ref: str, *, use_offline: bool = True) -> tuple[Any, Any]:
    """Return ``(model, processor)`` for a Stonesoup-managed bundle (shared in-memory weights).

    Checkpoints live in a **process-wide pool** (toolbar **Load** and cell loads share one copy).
    ``load_model`` ensures **this script's kernel** has a binding: if the pool already has weights
    for ``ref``, you get the **same** tensors/tokenizers and only this namespace is updated; if
    not, it runs the same load path as **Load** (HF Hub), then returns the bundle.

    ``ref`` with ``/`` is a Hugging Face repo id (e.g. ``Qwen/Qwen3-VL-8B-Instruct``). Without ``/``,
    ``ref`` is the internal **binding name** only; that name must already exist in this kernel
    (load the repo id first in the UI or from a cell so auto-bind can run).

    For text-only causal LMs the second value is the tokenizer (no multimodal processor).

    When ``use_offline`` is True (default), all HF Hub calls use ``local_files_only=True``
    to skip network requests and rely on cached files only.
    """
    from stonesoup.backend.hf_models import (
        ModelLoadRuntimeError,
        load_models_into_kernel,
        resolve_loaded_bundle,
    )
    from stonesoup.backend.kernel import active_kernel

    kernel = active_kernel.get()
    if kernel is None:
        raise RuntimeError(
            "stonesoup.load_model() only works while a Stonesoup cell is running in the Stonesoup UI."
        )
    ref_stripped = ref.strip()
    try:
        bundle = resolve_loaded_bundle(kernel, ref_stripped)
    except KeyError as exc:
        msg = str(exc)
        if "Multiple Stonesoup bundles" in msg:
            raise
        if "/" not in ref_stripped:
            raise
        try:
            load_models_into_kernel(
                kernel,
                items=[{"repo_id": ref_stripped, "name": None, "model_kind": None}],
                device_map="auto",
                torch_dtype=None,
                trust_remote_code=False,
                default_model_kind="auto",
                local_files_only=use_offline,
            )
        except (ModelLoadRuntimeError, ValueError) as err:
            raise RuntimeError(
                f"stonesoup.load_model({ref_stripped!r}): failed to load into kernel: {err}"
            ) from err
        bundle = resolve_loaded_bundle(kernel, ref_stripped)
    processor = getattr(bundle, "processor", None) or bundle.tokenizer
    return bundle.model, processor


def unload_loaded_names_for_repo(repo_id: str) -> None:
    """Unload all kernel bindings whose ``repo_id`` matches (frees pool refs when refcount hits zero).

    No-op if not running inside a Stonesoup cell (no active kernel). Only affects bindings in
    **this** kernel whose listed ``repo_id`` equals the given string.
    """
    from stonesoup.backend.hf_models import unload_models_from_kernel
    from stonesoup.backend.kernel import active_kernel

    kernel = active_kernel.get()
    if kernel is None:
        return
    names = [r["name"] for r in list_loaded_models() if r.get("repo_id") == repo_id]
    if names:
        unload_models_from_kernel(kernel, names=names)
