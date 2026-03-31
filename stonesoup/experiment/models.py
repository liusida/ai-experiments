"""Model handles: tie experiment code to weights loaded in the Stonesoup UI."""

from __future__ import annotations

from typing import Any


def list_loaded_models() -> list[dict[str, str]]:
    """Return ``[{name, repo_id}, ...]`` for weights currently loaded into this kernel (toolbar / ``load_model``).

    Outside a Stonesoup cell, ``active_kernel`` is unset and this returns an empty list.
    """
    from stonesoup.backend.hf_models import list_loaded_models as _list
    from stonesoup.backend.kernel import active_kernel

    return _list(active_kernel.get())


def list_hf_hub_cached_repo_ids() -> list[str]:
    """Hugging Face Hub repo ids found under the local cache (same scan as the UI datalist)."""
    from stonesoup.backend.hf_models import list_hf_hub_cached_model_repo_ids

    return list_hf_hub_cached_model_repo_ids()


def load_model(ref: str) -> tuple[Any, Any]:
    """Return ``(model, processor)`` for a Stonesoup-managed bundle.

    ``ref`` may be a Hugging Face repo id (e.g. ``Qwen/Qwen3-VL-8B-Instruct``). If it contains
    ``/`` and nothing is loaded yet for that id, weights are **loaded into the kernel from this
    cell** (same path as the toolbar **Load**), then the bundle is returned.

    If ``ref`` has no ``/``, it is the internal kernel binding name only; you must load that name
    in the UI first (or use the repo id so auto-load can run).

    For text-only causal LMs there is no multimodal processor; the second value is the tokenizer.
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
            )
        except (ModelLoadRuntimeError, ValueError) as err:
            raise RuntimeError(
                f"stonesoup.load_model({ref_stripped!r}): failed to load into kernel: {err}"
            ) from err
        bundle = resolve_loaded_bundle(kernel, ref_stripped)
    processor = getattr(bundle, "processor", None) or bundle.tokenizer
    return bundle.model, processor
