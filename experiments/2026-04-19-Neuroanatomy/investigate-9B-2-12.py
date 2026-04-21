# %% Imports, relayer helpers & capture
from __future__ import annotations

import copy
import json
import re
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import stonesoup
import torch
import torch.nn as nn
from accelerate.hooks import add_hook_to_module

from stonesoup.experiment import (
    configure_matplotlib_agg,
    decoder_blocks,
    ensure_pad_token_via_eos,
    inner_tokenizer,
)

configure_matplotlib_agg()

# --- Layer relayer (same as sweep-qwen3.5-9B.py: swap decoder ModuleList, shallow-copy, shared weights) ---


def _get_text_layer_owner(base_model: nn.Module) -> tuple[nn.Module, str]:
    if hasattr(base_model, "model") and hasattr(base_model.model, "layers"):
        return base_model.model, "layers"
    if (
        hasattr(base_model, "model")
        and hasattr(base_model.model, "language_model")
        and hasattr(base_model.model.language_model, "layers")
    ):
        return base_model.model.language_model, "layers"
    if hasattr(base_model, "language_model") and hasattr(base_model.language_model, "layers"):
        return base_model.language_model, "layers"
    raise AttributeError(
        "Could not find decoder layers (expected model.layers or *.language_model.layers)."
    )


def _rebind_accelerate_hook(original_module: nn.Module, copied_module: nn.Module) -> None:
    source_hook = getattr(original_module, "_hf_hook", None)
    if source_hook is None:
        return
    for attr in ("_hf_hook", "_old_forward"):
        if hasattr(copied_module, attr):
            delattr(copied_module, attr)
    copied_module.forward = type(copied_module).forward.__get__(copied_module, type(copied_module))
    add_hook_to_module(copied_module, copy.copy(source_hook), append=False)


# copy.copy shares the hook OrderedDicts with the source module (and with other copies of the
# same source). Without this reset, registering a forward hook on one copy also registers it on
# every sibling, so repeated layers cross-talk during capture.
_HOOK_DICT_ATTRS = (
    "_forward_hooks",
    "_forward_pre_hooks",
    "_forward_hooks_with_kwargs",
    "_forward_pre_hooks_with_kwargs",
    "_forward_hooks_always_called",
    "_backward_hooks",
    "_backward_pre_hooks",
    "_state_dict_hooks",
    "_state_dict_pre_hooks",
    "_load_state_dict_pre_hooks",
    "_load_state_dict_post_hooks",
)


def _reset_module_hooks(module: nn.Module) -> None:
    for attr in _HOOK_DICT_ATTRS:
        if hasattr(module, attr):
            current = getattr(module, attr)
            setattr(module, attr, type(current)())


def _shallow_copy_decoder_layer(layer: nn.Module, new_layer_idx: int) -> nn.Module:
    new_layer = copy.copy(layer)
    _reset_module_hooks(new_layer)
    new_layer._modules = dict(layer._modules)
    if hasattr(layer, "self_attn"):
        new_attn = copy.copy(layer.self_attn)
        _reset_module_hooks(new_attn)
        new_attn._modules = dict(layer.self_attn._modules)
        new_attn.layer_idx = new_layer_idx
        _rebind_accelerate_hook(layer.self_attn, new_attn)
        new_layer._modules["self_attn"] = new_attn
    if hasattr(layer, "linear_attn"):
        new_la = copy.copy(layer.linear_attn)
        _reset_module_hooks(new_la)
        new_la._modules = dict(layer.linear_attn._modules)
        new_la.layer_idx = new_layer_idx
        _rebind_accelerate_hook(layer.linear_attn, new_la)
        new_layer._modules["linear_attn"] = new_la
    _rebind_accelerate_hook(layer, new_layer)
    return new_layer


def expand_block_with_repeats(
    num_layers: int, block: tuple[int, int], exec_times: int
) -> list[int]:
    """Relayer layer index list; third arg is ``RELAYER_EXEC_TIMES`` (see config).

    - ``0``: **skip the block** — ``range(0, i) + range(j, N)`` (layers ``i..j-1`` never run; residual jumps).
    - ``1``: **baseline** — identity ``0..N-1`` (``block`` ignored except ``(0,0)``).
    - ``2``: **execute block once** — ``range(0, j) + range(i, N)`` (``i..j-1`` appear twice; usual relayer).
    - ``k >= 3``: **execute the block ``k-1`` times** in the overlap sense — ``range(0,j)`` + ``(k-2)`` copies of
      ``range(i,j)`` + ``range(i,N)`` (same as internal ``repeat_times = k-1`` in the construction below).
    """
    if exec_times == 1:
        return list(range(num_layers))
    if exec_times < 0:
        raise ValueError(f"exec_times must be >= 0, got {exec_times}")
    i, j = int(block[0]), int(block[1])
    if i == 0 and j == 0:
        return list(range(num_layers))
    if i < 0 or j < 0 or i >= j or j > num_layers:
        raise ValueError(f"Invalid block {(i, j)} for {num_layers} layers.")
    if exec_times == 0:
        return list(range(0, i)) + list(range(j, num_layers))
    # exec_times >= 2: overlap relayer; internal repeat_times = exec_times - 1
    repeat_times = exec_times - 1
    middle: list[int] = []
    for _ in range(repeat_times - 1):
        middle.extend(range(i, j))
    return list(range(0, j)) + middle + list(range(i, num_layers))


def relayer_repeat_bands_for_plot(exec_times: int) -> int:
    """Overlap bands (each width ``repeat_len``) for ``_s_after_repeat`` (exec ``>= 2`` only)."""
    if exec_times <= 1:
        return 0
    return exec_times - 1


def expand_single_block(num_layers: int, block: tuple[int, int]) -> list[int]:
    """Same as ``expand_block_with_repeats(..., exec_times=2)`` (one overlap; ``i..j-1`` twice)."""
    return expand_block_with_repeats(num_layers, block, 2)


class LayerDuplicatedModel(nn.Module):
    """Wrap a HF causal LM: temporarily replace decoder layers with a repeated index sequence (shared weights)."""

    def __init__(self, base_model: nn.Module, layer_indices: list[int]):
        super().__init__()
        self.base_model = base_model
        self.layer_indices = list(layer_indices)
        self._layers_owner, self._layers_attr = _get_text_layer_owner(base_model)
        self._original_layers = list(getattr(self._layers_owner, self._layers_attr))
        self._original_num_layers = len(self._original_layers)
        for idx in self.layer_indices:
            if idx < 0 or idx >= self._original_num_layers:
                raise ValueError(f"Layer index {idx} out of range [0, {self._original_num_layers})")
        self._new_num_layers = len(self.layer_indices)
        new_layers = [
            _shallow_copy_decoder_layer(self._original_layers[orig_i], new_pos)
            for new_pos, orig_i in enumerate(self.layer_indices)
        ]
        self._new_layers = nn.ModuleList(new_layers)

    @contextmanager
    def _apply_layer_config(self):
        owner = self._layers_owner
        attr = self._layers_attr
        orig_layers = getattr(owner, attr)
        cfg = self.base_model.config
        orig_n = getattr(cfg, "num_hidden_layers", None)
        orig_lt = getattr(cfg, "layer_types", None)
        tc = getattr(cfg, "text_config", None)
        orig_tn = getattr(tc, "num_hidden_layers", None) if tc is not None else None
        orig_tlt = getattr(tc, "layer_types", None) if tc is not None else None
        try:
            setattr(owner, attr, self._new_layers)
            if orig_n is not None:
                cfg.num_hidden_layers = self._new_num_layers
            if orig_lt is not None:
                if len(orig_lt) == self._original_num_layers:
                    cfg.layer_types = [orig_lt[i] for i in self.layer_indices]
                else:
                    cfg.layer_types = list(orig_lt)
            if tc is not None and orig_tn is not None:
                tc.num_hidden_layers = self._new_num_layers
            if tc is not None and orig_tlt is not None:
                if len(orig_tlt) == self._original_num_layers:
                    tc.layer_types = [orig_tlt[i] for i in self.layer_indices]
                else:
                    tc.layer_types = list(orig_tlt)
            yield
        finally:
            setattr(owner, attr, orig_layers)
            if orig_n is not None:
                cfg.num_hidden_layers = orig_n
            if orig_lt is not None:
                cfg.layer_types = orig_lt
            if tc is not None and orig_tn is not None:
                tc.num_hidden_layers = orig_tn
            if tc is not None and orig_tlt is not None:
                tc.layer_types = orig_tlt

    def forward(self, *args, **kwargs):
        if kwargs.get("past_key_values") is not None:
            cache = kwargs["past_key_values"]
            if hasattr(cache, "key_cache"):
                n_cache = len(cache.key_cache)
            elif isinstance(cache, tuple):
                n_cache = len(cache)
            else:
                n_cache = 0
            if n_cache != self._new_num_layers:
                kwargs["past_key_values"] = None
        with self._apply_layer_config():
            return self.base_model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        kwargs = dict(kwargs)
        kwargs.pop("past_key_values", None)
        with self._apply_layer_config():
            return self.base_model.generate(*args, **kwargs)


def build_model_with_layers(model: nn.Module, layer_indices: list[int]) -> LayerDuplicatedModel:
    return LayerDuplicatedModel(model, layer_indices)


def decoder_num_layers(model: nn.Module) -> int:
    cfg = model.config
    tc = getattr(cfg, "text_config", None)
    if tc is not None and getattr(tc, "num_hidden_layers", None) is not None:
        return int(tc.num_hidden_layers)
    return int(getattr(cfg, "num_hidden_layers"))


class _FirstActivationOnly:
    """Allow one successful ``take()`` per ``reset()`` — used to ignore extra forwards on the same module."""

    __slots__ = ("_armed",)

    def __init__(self) -> None:
        self._armed = True

    def take(self) -> bool:
        if not self._armed:
            return False
        self._armed = False
        return True

    def reset(self) -> None:
        self._armed = True


def _capture_embed_and_blocks_first_only(
    emb: nn.Module,
    blocks: list[nn.Module],
    merged: dict[str, Any],
    run: Callable[..., Any],
) -> tuple[torch.Tensor, list[str]]:
    """First activation only per module (Qwen3.x may call embed/layers multiple times per forward)."""
    gate_embed = _FirstActivationOnly()
    gate_layers = [_FirstActivationOnly() for _ in blocks]
    captured: list[torch.Tensor] = []

    def grab_embed(_mod: nn.Module, _inp: Any, x: torch.Tensor) -> None:
        if not gate_embed.take():
            return
        captured.append(x.detach())

    def make_grab_layer(i: int):
        def grab_layer(_mod: nn.Module, _inp: Any, x: Any) -> None:
            if not gate_layers[i].take():
                return
            h = x[0] if isinstance(x, tuple) else x
            captured.append(h.detach())

        return grab_layer

    hooks: list[Any] = [emb.register_forward_hook(grab_embed)]
    hooks += [blocks[i].register_forward_hook(make_grab_layer(i)) for i in range(len(blocks))]
    kw = dict(merged)
    kw.pop("output_hidden_states", None)
    gate_embed.reset()
    for g in gate_layers:
        g.reset()
    try:
        with torch.inference_mode():
            run(**kw)
    finally:
        for h in hooks:
            h.remove()

    names = ["embedding"] + [f"layer_{i}" for i in range(len(blocks))]
    if len(captured) != len(names):
        raise RuntimeError(
            f"embed+blocks capture: expected {len(names)} tensors (first-only hooks), got {len(captured)}"
        )
    return torch.stack(captured, dim=0), names


def capture_causal_lm_embed_and_post_blocks(
    model: nn.Module,
    inputs: dict[str, Any],
    **forward_kw: Any,
) -> tuple[torch.Tensor, list[str]]:
    """Baseline (no relayer): embed + one post-hook per ``decoder_blocks(model)``."""
    merged = dict(inputs, **forward_kw)
    blocks = list(decoder_blocks(model))
    emb = model.get_input_embeddings()
    return _capture_embed_and_blocks_first_only(emb, blocks, merged, model)


def capture_relayer_embed_and_post_blocks(
    dup: LayerDuplicatedModel,
    inputs: dict[str, Any],
    **forward_kw: Any,
) -> tuple[torch.Tensor, list[str]]:
    """Relayer: embed from ``base_model`` + hooks on ``dup._new_layers`` (same first-only semantics)."""
    merged = dict(inputs, **forward_kw)
    blocks = list(dup._new_layers)
    emb = dup.base_model.get_input_embeddings()
    return _capture_embed_and_blocks_first_only(emb, blocks, merged, dup)


def parse_first_integer(completion: str) -> int:
    """First integer in the model output (commas allowed). Unparseable → 0."""
    compact = completion.replace(",", "")
    m = re.search(r"-?\d+", compact)
    if not m:
        return 0
    return int(m.group(0))


def mean_token_l2_norm_per_stage(stack: torch.Tensor, batch_index: int = 0) -> np.ndarray:
    """stack (S, batch, seq, hidden) → mean L2 norm over **all** sequence positions per stage.

    For single-sequence inputs with no padding, this matches a masked mean over real tokens.
    """
    norms = stack.float().norm(dim=-1)
    b = int(batch_index)
    return norms[:, b, :].mean(dim=-1).detach().cpu().numpy()


def token_l2_norm_per_stage(
    stack: torch.Tensor,
    token_index: int,
    *,
    batch_index: int = 0,
) -> np.ndarray:
    """stack (S, batch, seq, hidden) → L2 norm at token index ``t`` per stage (batch row ``batch_index``)."""
    norms = stack.float().norm(dim=-1)
    b = int(batch_index)
    t = int(token_index)
    _seq = int(norms.shape[2])
    if t < 0 or t >= _seq:
        raise IndexError(f"token_index {t} out of range for seq_len {_seq}")
    return norms[:, b, t].detach().cpu().numpy()


# Same system prompt as sweep-qwen3.5-9B.py; thinking off via ``enable_thinking=False``.
MATH_SYSTEM_PROMPT = (
    "You are a highly intelligent AI. You have extraordinary intuition and can "
    "easily make accurate estimations. For the following questions, you will "
    "always provide an answer, even if you are not certain."
)

# %% Config
MODEL_ID = "Qwen/Qwen3.5-9B"
RELAYER_EXEC_TIMES = 0  # 0: skip block (shortcut), 1: baseline, 2: block once (treatment), 3+: extra repeats
RELAYER_BLOCK: tuple[int, int] = (2, 12)
RELAYER_BLOCK: tuple[int, int] = (13, 15)
DATA_PATH = stonesoup.data_dir() / "rys-dataset" / "simple_math_16.json"
FIRST_KEY = "4"

# %% Load model
torch.set_grad_enabled(False)

model, processor = stonesoup.load_model(MODEL_ID)
model.eval()
tokenizer = inner_tokenizer(processor)
ensure_pad_token_via_eos(tokenizer)
device = next(model.parameters()).device

# %% Relayer wrapper (``RELAYER_BLOCK``, ``RELAYER_EXEC_TIMES``)
_N = decoder_num_layers(model)
_layer_indices = expand_block_with_repeats(_N, RELAYER_BLOCK, RELAYER_EXEC_TIMES)
dup = build_model_with_layers(model, _layer_indices)
dup.eval()
print(
    f"decoder layers: base={_N} RELAYER_EXEC_TIMES={RELAYER_EXEC_TIMES} "
    f"relayer stack depth={len(_layer_indices)} indices[:]={_layer_indices[:]!r}",
    flush=True,
)

# %% First prompt from simple_math_16 (chat template)
with open(DATA_PATH, encoding="utf-8") as f:
    _dataset = json.load(f)
_item = _dataset[FIRST_KEY]
_msgs = [
    {"role": "system", "content": MATH_SYSTEM_PROMPT},
    {"role": "user", "content": _item["question"]},
]
inputs = tokenizer.apply_chat_template(
    _msgs,
    enable_thinking=False,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
)
_ids_row = inputs["input_ids"][0]
print("--- chat messages (before template) ---", flush=True)
print(json.dumps(_msgs, indent=2, ensure_ascii=False), flush=True)
print("--- prompt after chat template (decoded, with special tokens) ---", flush=True)
print(tokenizer.decode(_ids_row, skip_special_tokens=False), flush=True)
print("--- same, skip_special_tokens=True ---", flush=True)
print(tokenizer.decode(_ids_row, skip_special_tokens=True), flush=True)
inputs = {k: v.to(device) for k, v in inputs.items()}
print(f"key={FIRST_KEY!r} seq_len={inputs['input_ids'].shape[1]}", flush=True)

# %% Generate: baseline vs relayer vs ground truth
# Same greedy decode budget as sweep-qwen3.5-9B (integer answers).
MAX_NEW_TOKENS = 32
_prompt_len = int(inputs["input_ids"].shape[1])
_pad_id_gen = int(tokenizer.pad_token_id or tokenizer.eos_token_id)
_ground_truth = int(_item["answer"])
stonesoup.check_abort()
with torch.inference_mode():
    _gen_base = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=_pad_id_gen,
    )
    _gen_rel = dup.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=_pad_id_gen,
    )
_completion_base = tokenizer.decode(_gen_base[0, _prompt_len:], skip_special_tokens=True)
_completion_rel = tokenizer.decode(_gen_rel[0, _prompt_len:], skip_special_tokens=True)
_pred_base = parse_first_integer(_completion_base)
_pred_rel = parse_first_integer(_completion_rel)
print(
    f"key={FIRST_KEY!r}  ground_truth={_ground_truth}\n"
    f"  baseline model: pred={_pred_base}  match={_pred_base == _ground_truth}  "
    f"completion={_completion_base!r}\n"
    f"  relayer {RELAYER_BLOCK}×{RELAYER_EXEC_TIMES}: pred={_pred_rel}  match={_pred_rel == _ground_truth}  "
    f"completion={_completion_rel!r}",
    flush=True,
)

# %% Forward + hidden capture, mean norm per stage, plot (baseline vs relayer) # stonesoup:cell-input
# CELL_INPUT: 0-based index into the **baseline greedy completion** (first new token = 0).
token_idx = int(globals().get("CELL_INPUT", "0"))
_n_new = int(_gen_base.shape[1] - _prompt_len)
if _n_new <= 0:
    raise RuntimeError("No completion tokens in _gen_base; run the generate cell first.")
if token_idx < 0 or token_idx >= _n_new:
    raise IndexError(
        f"response token index {token_idx} out of range [0, {_n_new}) ({_n_new} completion tokens)"
    )
_abs_idx = _prompt_len + token_idx
_continuation = _gen_base[:, _prompt_len:]
inputs_capture = {
    "input_ids": torch.cat([inputs["input_ids"], _continuation], dim=1),
    "attention_mask": torch.cat(
        [
            inputs["attention_mask"],
            torch.ones(
                _continuation.shape,
                dtype=inputs["attention_mask"].dtype,
                device=inputs["attention_mask"].device,
            ),
        ],
        dim=1,
    ),
}
stack_base, _names_base = capture_causal_lm_embed_and_post_blocks(model, inputs_capture, use_cache=False)
mean_norms_base = mean_token_l2_norm_per_stage(stack_base)
token_norm_base = token_l2_norm_per_stage(stack_base, _abs_idx)
base = token_norm_base

stack_rel, _names_rel = capture_relayer_embed_and_post_blocks(dup, inputs_capture, use_cache=False)
mean_norms_rel = mean_token_l2_norm_per_stage(stack_rel)
token_norm_rel = token_l2_norm_per_stage(stack_rel, _abs_idx)
relayer = token_norm_rel

x_rel = np.arange(len(mean_norms_rel))
_orig_line_rel = ["embed"] + [str(_layer_indices[j]) for j in range(len(_layer_indices))]
_xtick_labels_rel = [f"{i}\n{_orig_line_rel[i]}" for i in range(len(mean_norms_rel))]

# Baseline y for relayer x: ``1`` = same stage index; ``0`` = map relayer stage → baseline by ``_layer_indices``;
# ``>= 2`` = compressed x across overlap bands (same as before).
_cj = int(RELAYER_BLOCK[1])
_repeat_len = _cj - int(RELAYER_BLOCK[0])
if RELAYER_EXEC_TIMES <= 1:
    x_base_plot = x_rel.astype(np.float64)
    if RELAYER_EXEC_TIMES == 1:
        y_base_plot = np.asarray(base, dtype=np.float64)
    else:
        y_base_plot = np.asarray(
            [base[0] if _s == 0 else base[_layer_indices[_s - 1] + 1] for _s in range(len(mean_norms_rel))],
            dtype=np.float64,
        )
    _s_after_repeat = -1  # unused (cos sim uses per-mode mapping)
else:
    _s_after_repeat = _cj + relayer_repeat_bands_for_plot(RELAYER_EXEC_TIMES) * _repeat_len + 1
    _n_tail = len(mean_norms_rel) - _s_after_repeat
    x_base_plot = np.concatenate(
        [
            np.arange(_cj + 1, dtype=np.float64),
            np.arange(_s_after_repeat, len(mean_norms_rel), dtype=np.float64),
        ]
    )
    y_base_plot = np.concatenate(
        [
            np.asarray(base[: _cj + 1], dtype=np.float64),
            np.asarray(base[_cj + 1 : _cj + 1 + _n_tail], dtype=np.float64),
        ]
    )

# Plot — decode ``token_idx``-th **completion** token (same ids as ``inputs_capture`` suffix).
_resp_tok_id = int(_gen_base[0, _prompt_len + token_idx])
_token_fragment = tokenizer.decode([_resp_tok_id])

fig_w = min(22, max(10.0, 0.28 * len(mean_norms_rel)))
fig, ax = plt.subplots(figsize=(fig_w, 5.2))
ax.plot(
    x_rel,
    relayer,
    marker="o",
    ms=3,
    color="tab:orange",
    label=f"relayer {RELAYER_BLOCK}×{RELAYER_EXEC_TIMES}",
)
ax.plot(
    x_base_plot,
    y_base_plot,
    marker="s",
    ms=2,
    color="tab:blue",
    alpha=0.9,
    label=f"baseline",
)
ax.set_xticks(x_rel)
ax.set_xticklabels(_xtick_labels_rel, fontsize=6, ha="center")
ax.set_xlabel('stage (top) / base layer index or "embed" (bottom) — relayer stack')
ax.set_ylabel("mean L2 norm (masked mean over tokens)")
ax.set_title(
    f"{MODEL_ID}  relayer {RELAYER_BLOCK}×{RELAYER_EXEC_TIMES} vs baseline — completion token {token_idx} "
    f"{repr(_token_fragment)}"
)
ax.legend(loc="upper right", fontsize=8)
ax.grid(True, alpha=0.3)
fig.tight_layout()
stonesoup.show(basename="investigate_9b_baseline_aligned_vs_relayer_norm", fig=fig)

# %% Cosine similarity baseline vs relayer (aligned x, same token as norm plot)
# Requires the previous cell: ``stack_*``, ``_abs_idx``, ``_cj``, ``_s_after_repeat``, ``_xtick_labels_rel``.
_len_rel = int(stack_rel.shape[0])


def _baseline_idx_for_rel_stage(s: int) -> int | None:
    """Baseline ``stack_base`` index aligned to relayer stage ``s`` (same convention as norm plot)."""
    if RELAYER_EXEC_TIMES == 1:
        return s
    if RELAYER_EXEC_TIMES == 0:
        if s == 0:
            return 0
        return _layer_indices[s - 1] + 1
    if s <= _cj:
        return s
    if s >= _s_after_repeat:
        return _cj + 1 + (s - _s_after_repeat)
    return None


_cos = np.full(_len_rel, np.nan, dtype=np.float64)
for _s in range(_len_rel):
    _bi = _baseline_idx_for_rel_stage(_s)
    if _bi is None:
        continue
    if _bi < 0 or _bi >= stack_base.shape[0]:
        raise IndexError(
            f"aligned baseline stage {_bi} out of range [0, {stack_base.shape[0]}) "
            f"(rel stage {_s})"
        )
    _vb = stack_base[_bi, 0, _abs_idx].float()
    _vr = stack_rel[_s, 0, _abs_idx].float()
    _cos[_s] = torch.nn.functional.cosine_similarity(
        _vb.unsqueeze(0), _vr.unsqueeze(0), dim=1, eps=1e-12
    ).item()

fig_w2 = min(22, max(10.0, 0.28 * _len_rel))
fig2, ax2 = plt.subplots(figsize=(fig_w2, 5.2))
ax2.plot(x_rel, _cos, marker="o", ms=3, color="tab:green")
ax2.set_xticks(x_rel)
ax2.set_xticklabels(_xtick_labels_rel, fontsize=6, ha="center")
ax2.set_xlabel('stage (top) / base layer index or "embed" (bottom) — relayer stack')
ax2.set_ylabel("cosine similarity (hidden at completion token)")
ax2.set_ylim(0.7, 1.05)
ax2.axhline(0.0, color="0.5", lw=0.8, ls="--", alpha=0.6)
ax2.set_title(
    f"{MODEL_ID}  relayer {RELAYER_BLOCK}×{RELAYER_EXEC_TIMES}: cos(h_base, h_rel) vs aligned stage — "
    f"completion token {token_idx} {repr(_token_fragment)}"
)
ax2.grid(True, alpha=0.3)
fig2.tight_layout()
stonesoup.show(basename="investigate_9b_rel_vs_base_cos_sim", fig=fig2)
