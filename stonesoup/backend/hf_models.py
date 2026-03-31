"""Helpers for loading Hugging Face models into a Stonesoup kernel."""

from __future__ import annotations

import gc
import keyword
import os
import re
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from pathlib import Path
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Callable
from weakref import WeakKeyDictionary

if TYPE_CHECKING:
    from stonesoup.backend.kernel import Kernel


class ModelLoadRuntimeError(RuntimeError):
    """Raised when the ML runtime dependencies are unavailable."""


@dataclass(slots=True)
class LoadedModelInfo:
    name: str
    repo_id: str

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "repo_id": self.repo_id}


@dataclass(slots=True)
class _ManagedModelsState:
    bundles: dict[str, str] = field(default_factory=dict)
    convenience_aliases: set[str] = field(default_factory=set)


_MANAGED_MODEL_STATE: WeakKeyDictionary[Kernel, _ManagedModelsState] = WeakKeyDictionary()
_IDENTIFIER_RE = re.compile(r"[^0-9A-Za-z_]+")
_RESERVED_GLOBALS = frozenset({"model", "tokenizer", "processor", "MODEL_REPO_ID"})


class _ChunkStreamSink:
    """Forward ``sys.stdout`` / ``sys.stderr`` writes to an optional per-chunk callback."""

    __slots__ = ("_on_chunk",)

    def __init__(self, on_chunk: Callable[[str], None] | None) -> None:
        self._on_chunk = on_chunk

    def write(self, s: str) -> int:
        if s and self._on_chunk is not None:
            self._on_chunk(s)
        return len(s)

    def flush(self) -> None:
        pass


def _state_for(kernel: Kernel) -> _ManagedModelsState:
    state = _MANAGED_MODEL_STATE.get(kernel)
    if state is None:
        state = _ManagedModelsState()
        _MANAGED_MODEL_STATE[kernel] = state
    return state


def _clear_convenience_aliases(kernel: Kernel, state: _ManagedModelsState) -> None:
    for alias in state.convenience_aliases:
        kernel.globals.pop(alias, None)
    state.convenience_aliases.clear()


def _to_identifier(raw: str) -> str:
    name = _IDENTIFIER_RE.sub("_", raw.strip().lower()).strip("_")
    name = re.sub(r"_+", "_", name)
    if not name:
        name = "model"
    if name[0].isdigit():
        name = f"model_{name}"
    if keyword.iskeyword(name):
        name = f"{name}_model"
    return name


def _unique_binding_name(*, repo_id: str, requested_name: str | None, taken: set[str]) -> str:
    base = _to_identifier(requested_name or repo_id)
    candidate = base
    suffix = 2
    while candidate in taken or candidate.startswith("__"):
        candidate = f"{base}_{suffix}"
        suffix += 1
    taken.add(candidate)
    return candidate


def _load_runtime() -> tuple[Any, Any]:
    try:
        import torch
    except Exception as exc:  # noqa: BLE001 - import errors vary by install
        raise ModelLoadRuntimeError(
            "PyTorch is required to load models in Stonesoup. Install a compatible torch build first."
        ) from exc
    try:
        from transformers import AutoTokenizer
    except Exception as exc:  # noqa: BLE001 - import errors vary by install
        raise ModelLoadRuntimeError(
            'transformers is required to load models in Stonesoup. Install it with `uv pip install -e ".[models]"`.'
        ) from exc
    return torch, AutoTokenizer


def _normalize_model_kind(raw: str | None) -> str:
    k = (raw or "auto").strip().lower()
    if k not in {"auto", "causal_lm", "image_text"}:
        raise ValueError(
            "model_kind must be one of: auto, causal_lm, image_text "
            f"(vision-language models such as Qwen3-VL need auto or image_text)."
        )
    return k


def _load_pretrained_model(
    repo_id: str,
    *,
    model_kind: str,
    trust_remote_code: bool,
    torch_dtype: Any,
    device_map_value: str | None,
) -> tuple[Any, str]:
    """Pick an appropriate HF Auto class; return ``(model, resolved_branch)``.

    ``resolved_branch`` is ``causal_lm``, ``image_text``, or ``generic`` (for tokenizer/processor setup).
    """
    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoModelForImageTextToText
    from transformers.models.auto.modeling_auto import (
        MODEL_FOR_CAUSAL_LM_MAPPING,
        MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING,
    )

    kind = _normalize_model_kind(model_kind)
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "torch_dtype": torch_dtype,
    }
    if device_map_value:
        model_kwargs["device_map"] = device_map_value

    if kind == "causal_lm":
        return AutoModelForCausalLM.from_pretrained(repo_id, **model_kwargs), "causal_lm"
    if kind == "image_text":
        return AutoModelForImageTextToText.from_pretrained(repo_id, **model_kwargs), "image_text"

    config = AutoConfig.from_pretrained(repo_id, trust_remote_code=trust_remote_code)
    cfg_type = type(config)
    # Prefer image+text when both mappings list the same config (e.g. Qwen3.5-4B: causal loads
    # ``Qwen3_5ForCausalLM`` (text-only); image-text loads ``Qwen3_5ForConditionalGeneration``).
    if cfg_type in MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING:
        return AutoModelForImageTextToText.from_pretrained(repo_id, **model_kwargs), "image_text"
    if cfg_type in MODEL_FOR_CAUSAL_LM_MAPPING:
        return AutoModelForCausalLM.from_pretrained(repo_id, **model_kwargs), "causal_lm"
    return AutoModel.from_pretrained(repo_id, **model_kwargs), "generic"


def _resolve_torch_dtype(torch: Any, raw: str | None) -> Any:
    if raw is None or not raw.strip():
        return torch.float16 if torch.cuda.is_available() else torch.float32
    key = raw.strip().lower()
    mapping = {
        "auto": "auto",
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "float": torch.float32,
    }
    try:
        return mapping[key]
    except KeyError as exc:
        raise ValueError(
            "Unsupported torch_dtype. Use one of: auto, float16, bfloat16, float32."
        ) from exc


def list_loaded_models(kernel: Kernel | None) -> list[dict[str, str]]:
    if kernel is None:
        return []
    state = _state_for(kernel)
    return [
        LoadedModelInfo(name=name, repo_id=repo_id).to_dict()
        for name, repo_id in sorted(state.bundles.items(), key=lambda item: item[0])
    ]


def _hf_hub_cache_root() -> Path:
    raw = os.environ.get("HF_HUB_CACHE", "").strip()
    if raw:
        return Path(raw).expanduser()
    hf_home = os.environ.get("HF_HOME", "").strip()
    base = Path(hf_home).expanduser() if hf_home else Path.home() / ".cache" / "huggingface"
    return base / "hub"


def _repo_id_from_models_cache_folder(name: str) -> str | None:
    """Decode ``models--org--repo`` directory name under the Hugging Face hub cache."""
    if not name.startswith("models--"):
        return None
    rest = name[8:]
    idx = rest.find("--")
    if idx <= 0:
        return None
    org = rest[:idx]
    repo_name = rest[idx + 2 :]
    if not org or not repo_name:
        return None
    return f"{org}/{repo_name}"


def _list_cached_model_repo_ids_from_disk(hub: Path) -> list[str]:
    if not hub.is_dir():
        return []
    seen: set[str] = set()
    try:
        for p in hub.iterdir():
            if not p.is_dir():
                continue
            rid = _repo_id_from_models_cache_folder(p.name)
            if rid:
                seen.add(rid)
    except OSError:
        return []
    return sorted(seen)


def list_hf_hub_cached_model_repo_ids() -> list[str]:
    """Return Hugging Face **model** repo ids present in the local Hub cache.

    Prefer :func:`huggingface_hub.scan_cache_dir` when available; otherwise scan
    ``models--*`` folders under the default hub directory.
    """
    try:
        from huggingface_hub import scan_cache_dir
    except ImportError:
        scan_cache_dir = None
    if scan_cache_dir is not None:
        try:
            info = scan_cache_dir()
            ids = sorted(
                {
                    r.repo_id
                    for r in info.repos
                    if getattr(r, "repo_type", None) == "model"
                }
            )
            if ids:
                return ids
        except Exception:
            pass
    return _list_cached_model_repo_ids_from_disk(_hf_hub_cache_root())


def resolve_loaded_bundle(kernel: Kernel, ref: str) -> Any:
    """Return the Stonesoup bundle (namespace with ``model``, ``tokenizer``, …) for a repo id or bundle name.

    ``ref`` is either a Hugging Face ``repo_id`` (contains ``/``) or the Python binding name created when loading
    (e.g. ``qwen`` from ``qwen=Qwen/...``). Does **not** download or instantiate a second copy.
    """
    ref_stripped = ref.strip()
    if not ref_stripped:
        raise ValueError("ref must be a non-empty bundle name or Hugging Face repo_id.")

    state = _state_for(kernel)
    bundle: Any = None
    if "/" not in ref_stripped and ref_stripped in state.bundles:
        bundle = kernel.globals.get(ref_stripped)
    else:
        matches = [n for n, rid in state.bundles.items() if rid == ref_stripped]
        if len(matches) == 1:
            bundle = kernel.globals.get(matches[0])
        elif len(matches) > 1:
            raise KeyError(
                f"Multiple Stonesoup bundles for repo_id {ref_stripped!r}: {matches}. "
                "Pass the bundle name (left-hand side in name=repo) instead of the repo id."
            )

    if bundle is None:
        loaded = [f"{n}={r}" for n, r in sorted(state.bundles.items())]
        raise KeyError(
            f"No Stonesoup-loaded model matches {ref_stripped!r}. "
            f"Load it in the Stonesoup UI first. Currently loaded: {loaded or 'none'}."
        )
    if not hasattr(bundle, "model"):
        raise TypeError(f"Global {ref_stripped!r} is not a Stonesoup model bundle.")
    return bundle


def load_models_into_kernel(
    kernel: Kernel,
    *,
    items: list[dict[str, str | None]],
    device_map: str | None = "auto",
    torch_dtype: str | None = None,
    trust_remote_code: bool = False,
    default_model_kind: str | None = "auto",
    on_stdout_chunk: Callable[[str], None] | None = None,
    on_stderr_chunk: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    if not items:
        raise ValueError("Provide at least one model to load.")

    def _body() -> dict[str, Any]:
        torch, AutoTokenizer = _load_runtime()
        resolved_dtype = _resolve_torch_dtype(torch, torch_dtype)
        state = _state_for(kernel)
        _clear_convenience_aliases(kernel, state)

        taken_names = set(kernel.globals.keys()) | set(_RESERVED_GLOBALS)
        prepared: list[tuple[str, str, str]] = []
        for item in items:
            repo_id = str(item.get("repo_id") or "").strip()
            if not repo_id:
                raise ValueError("Each model entry must include a non-empty repo_id.")
            requested_name = item.get("name")
            model_kind = _normalize_model_kind(
                str(item.get("model_kind") or default_model_kind or "auto")
            )
            name = _unique_binding_name(
                repo_id=repo_id,
                requested_name=str(requested_name).strip() if requested_name else None,
                taken=taken_names,
            )
            prepared.append((name, repo_id, model_kind))

        device_map_value = device_map.strip() if isinstance(device_map, str) else device_map
        if device_map_value == "":
            device_map_value = None

        loaded: list[dict[str, str]] = []
        for name, repo_id, model_kind in prepared:
            model, branch = _load_pretrained_model(
                repo_id,
                model_kind=model_kind,
                trust_remote_code=trust_remote_code,
                torch_dtype=resolved_dtype,
                device_map_value=device_map_value,
            )
            processor = None
            tokenizer = None
            if branch == "image_text":
                try:
                    from transformers import AutoProcessor

                    processor = AutoProcessor.from_pretrained(
                        repo_id, trust_remote_code=trust_remote_code
                    )
                    tokenizer = getattr(processor, "tokenizer", None)
                except Exception:
                    processor = None
            if tokenizer is None:
                tokenizer = AutoTokenizer.from_pretrained(
                    repo_id, trust_remote_code=trust_remote_code
                )
            bundle = SimpleNamespace(
                model=model, tokenizer=tokenizer, processor=processor, repo_id=repo_id
            )
            kernel.globals[name] = bundle
            state.bundles[name] = repo_id
            loaded.append(LoadedModelInfo(name=name, repo_id=repo_id).to_dict())

        convenience_aliases: list[str] = []
        if len(prepared) == 1:
            only_name, only_repo_id, _mk = prepared[0]
            only_bundle = kernel.globals[only_name]
            kernel.globals["model"] = only_bundle.model
            kernel.globals["tokenizer"] = only_bundle.tokenizer
            kernel.globals["processor"] = only_bundle.processor or only_bundle.tokenizer
            kernel.globals["MODEL_REPO_ID"] = only_repo_id
            state.convenience_aliases = {"model", "tokenizer", "processor", "MODEL_REPO_ID"}
            convenience_aliases = sorted(state.convenience_aliases)

        return {
            "loaded": loaded,
            "convenience_aliases": convenience_aliases,
            "loaded_now": list_loaded_models(kernel),
        }

    if on_stdout_chunk is not None or on_stderr_chunk is not None:
        with ExitStack() as stack:
            if on_stdout_chunk is not None:
                stack.enter_context(redirect_stdout(_ChunkStreamSink(on_stdout_chunk)))
            if on_stderr_chunk is not None:
                stack.enter_context(redirect_stderr(_ChunkStreamSink(on_stderr_chunk)))
            return _body()
    return _body()


def unload_models_from_kernel(kernel: Kernel, *, names: list[str] | None = None) -> dict[str, Any]:
    state = _state_for(kernel)
    if names is None:
        target_names = sorted(state.bundles.keys())
        missing: list[str] = []
    else:
        target_names = []
        missing = []
        for raw in names:
            name = str(raw).strip()
            if not name:
                continue
            if name in state.bundles:
                target_names.append(name)
            else:
                missing.append(name)

    if target_names:
        _clear_convenience_aliases(kernel, state)

    unloaded: list[dict[str, str]] = []
    for name in target_names:
        repo_id = state.bundles.pop(name, "")
        bundle = kernel.globals.pop(name, None)
        if bundle is not None:
            model = getattr(bundle, "model", None)
            tokenizer = getattr(bundle, "tokenizer", None)
            processor = getattr(bundle, "processor", None)
            del bundle
            del model
            del tokenizer
            del processor
        unloaded.append(LoadedModelInfo(name=name, repo_id=repo_id).to_dict())

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    return {
        "unloaded": unloaded,
        "missing": missing,
        "loaded_now": list_loaded_models(kernel),
    }
