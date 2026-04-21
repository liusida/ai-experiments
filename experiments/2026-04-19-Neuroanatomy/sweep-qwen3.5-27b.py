# %% Imports & helpers
from __future__ import annotations

import copy
import json
import re
from contextlib import contextmanager

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from accelerate.hooks import add_hook_to_module

import stonesoup
from stonesoup.experiment import ensure_pad_token_via_eos, inner_tokenizer


def parse_first_integer(completion: str) -> int:
    """First integer in the model output (commas allowed). Unparseable → 0."""
    compact = completion.replace(",", "")
    m = re.search(r"-?\d+", compact)
    if not m:
        return 0
    return int(m.group(0))


def similarity_score(pred: int, ans: int) -> float:
    """clamp(0, 1, 1 - |pred-ans|/|ans|). Call only when ans != 0."""
    return max(0.0, min(1.0, 1.0 - abs(pred - ans) / abs(ans)))


# RYS-aligned math system prompt; thinking off via ``apply_chat_template(..., enable_thinking=False)``.
MATH_SYSTEM_PROMPT = (
    "You are a highly intelligent AI. You have extraordinary intuition and can "
    "easily make accurate estimations. For the following questions, you will "
    "always provide an answer, even if you are not certain."
)


# --- Standalone layer relayer (RYS-style: swap decoder ModuleList, shallow-copy layers, shared weights) ---


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


def _shallow_copy_decoder_layer(layer: nn.Module, new_layer_idx: int) -> nn.Module:
    new_layer = copy.copy(layer)
    new_layer._modules = dict(layer._modules)
    if hasattr(layer, "self_attn"):
        new_attn = copy.copy(layer.self_attn)
        new_attn._modules = dict(layer.self_attn._modules)
        new_attn.layer_idx = new_layer_idx
        _rebind_accelerate_hook(layer.self_attn, new_attn)
        new_layer._modules["self_attn"] = new_attn
    if hasattr(layer, "linear_attn"):
        new_la = copy.copy(layer.linear_attn)
        new_la._modules = dict(layer.linear_attn._modules)
        new_la.layer_idx = new_layer_idx
        _rebind_accelerate_hook(layer.linear_attn, new_la)
        new_layer._modules["linear_attn"] = new_la
    _rebind_accelerate_hook(layer, new_layer)
    return new_layer


def expand_single_block(num_layers: int, block: tuple[int, int]) -> list[int]:
    """(i, j) shorthand: ``range(0, j) + range(i, N)`` — layers ``i..j-1`` appear twice."""
    i, j = int(block[0]), int(block[1])
    if i == 0 and j == 0:
        return list(range(num_layers))
    if i < 0 or j < 0 or i >= j or j > num_layers:
        raise ValueError(f"Invalid block {(i, j)} for {num_layers} layers.")
    return list(range(0, j)) + list(range(i, num_layers))


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
        self._new_layers = nn.ModuleList(
            [
                _shallow_copy_decoder_layer(self._original_layers[orig_i], new_pos)
                for new_pos, orig_i in enumerate(self.layer_indices)
            ]
        )

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


# %% Config # stonesoup:cell-input
MODEL_ID = "Qwen/Qwen3.5-27B"
# Longest reference answers are ~8-digit products. Qwen tokenizers often use ~1 token per
# digit for long bare integers; 32 covers the digit string plus a little slack.
MAX_NEW_TOKENS = 32
DATA_PATH = stonesoup.data_dir() / "rys-dataset" / "math_16.json"
MATH16_NUM_ITEMS = 16
# Restrict sweep region: i > 24 and j < 50. Baseline (0, 0) always runs first.
SWEEP_I_MIN = 1
SWEEP_J_MAX = 999
# HF generate on padded batches (decoder-only: tokenizer padding_side must be "left").
SWEEP_BATCH_SIZE = 16
SWEEP_DEBUG_PRINT_SCORE = True
# Outside Stonesoup: empty. In Stonesoup: optional cell-input — further cap (e.g. 1 for smoke test).
_cell_in = globals().get("CELL_INPUT", "").strip()
ITEM_LIMIT: int | None = int(_cell_in) if _cell_in.isdigit() else None

# %% Load model
model, processor = stonesoup.load_model(MODEL_ID, use_offline=False)
model.eval()
tokenizer = inner_tokenizer(processor)
ensure_pad_token_via_eos(tokenizer)
tokenizer.padding_side = "left"
device = next(model.parameters()).device

# %% Relayer (i,j) sweep — setup (matrix + pair list)
# Pipeline: put only the next two cells in a Stonesoup loop. Loop JSON: array of ``[i, j]``
# pairs (e.g. copy of ``pairs`` or a subset). Each iteration sets ``LOOP_ITEM`` / ``LOOP_INDEX``.
num_layers = decoder_num_layers(model)
j_upper = min(SWEEP_J_MAX, num_layers + 1)
region_pairs = [
    (i, j)
    for i in range(SWEEP_I_MIN, num_layers)
    for j in range(i + 1, j_upper)
]
pairs = [(0, 0)] + region_pairs

with open(DATA_PATH, encoding="utf-8") as f:
    dataset_sweep = json.load(f)
keys_sweep = sorted(dataset_sweep.keys(), key=lambda k: int(k))[:MATH16_NUM_ITEMS]
if ITEM_LIMIT is not None:
    keys_sweep = keys_sweep[:ITEM_LIMIT]

score_mat = np.full((num_layers, num_layers + 1), np.nan, dtype=np.float64)
records: list[dict] = []
sweep_keys_valid = [k for k in keys_sweep if int(dataset_sweep[k]["answer"]) != 0]
sweep_step_i = 0

print(json.dumps(pairs))

# %% Relayer (i,j) sweep — one pair
# Uses ``LOOP_ITEM`` as ``[i, j]`` when the Stonesoup pipeline injects it; otherwise advances
# through ``pairs`` starting at ``sweep_step_i`` (manual runs without a loop).
loop_item = globals().get("LOOP_ITEM")
print(json.dumps(loop_item))
if loop_item is not None:
    if not isinstance(loop_item, (list, tuple)) or len(loop_item) < 2:
        raise TypeError(f"LOOP_ITEM must be [i, j], got {type(loop_item).__name__}: {loop_item!r}")
    i, j = int(loop_item[0]), int(loop_item[1])
else:
    if sweep_step_i >= len(pairs):
        raise IndexError(
            "Sweep exhausted: re-run setup to reset, or use a pipeline loop with LOOP_ITEM."
        )
    i, j = pairs[sweep_step_i]
    sweep_step_i += 1

stonesoup.check_abort()
layer_indices = expand_single_block(num_layers, (i, j))
dup = build_model_with_layers(model, layer_indices)
dup.eval()
scores: list[float] = []
pad_id = int(tokenizer.pad_token_id or tokenizer.eos_token_id)
for b0 in range(0, len(sweep_keys_valid), SWEEP_BATCH_SIZE):
    stonesoup.check_abort()
    chunk_keys = sweep_keys_valid[b0 : b0 + SWEEP_BATCH_SIZE]
    row_ids: list[torch.Tensor] = []
    row_mask: list[torch.Tensor] = []
    for key in chunk_keys:
        item = dataset_sweep[key]
        msgs = [
            {"role": "system", "content": MATH_SYSTEM_PROMPT},
            {"role": "user", "content": item["question"]},
        ]
        one = tokenizer.apply_chat_template(
            msgs,
            enable_thinking=False,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        one = {k: v.to(device) for k, v in one.items()}
        row_ids.append(one["input_ids"])
        row_mask.append(one["attention_mask"])
    max_len = max(int(t.shape[1]) for t in row_ids)
    batched_ids: list[torch.Tensor] = []
    batched_mask: list[torch.Tensor] = []
    for ids, am in zip(row_ids, row_mask, strict=True):
        pad = max_len - ids.shape[1]
        if pad:
            ids = torch.cat(
                [torch.full((1, pad), pad_id, dtype=ids.dtype, device=device), ids],
                dim=1,
            )
            am = torch.cat(
                [torch.zeros((1, pad), dtype=am.dtype, device=device), am],
                dim=1,
            )
        batched_ids.append(ids)
        batched_mask.append(am)
    batch_input_ids = torch.cat(batched_ids, dim=0)
    batch_attn = torch.cat(batched_mask, dim=0)
    with torch.inference_mode():
        out_ids = dup.generate(
            input_ids=batch_input_ids,
            attention_mask=batch_attn,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=pad_id,
        )
    for bi, key in enumerate(chunk_keys):
        ans = int(dataset_sweep[key]["answer"])
        completion = tokenizer.decode(out_ids[bi, max_len:], skip_special_tokens=True)
        pred = parse_first_integer(completion)
        item_score = similarity_score(pred, ans)
        scores.append(item_score)
        if SWEEP_DEBUG_PRINT_SCORE:
            out1 = " ".join(completion.split())
            print(
                f"sweep (i,j)=({i},{j}) key={key} pred={pred} ans={ans} "
                f"score={item_score:.6f} output={out1!r}",
                flush=True,
            )
del dup
mean_score = sum(scores) / len(scores) if scores else float("nan")
score_mat[i, j] = mean_score
records.append(
    {"i": i, "j": j, "mean_score": float(mean_score) if mean_score == mean_score else None}
)
print(
    f"relayer sweep (i,j)=({i},{j}) mean_score={mean_score:.6f} "
    f"LOOP_INDEX={globals().get('LOOP_INDEX')!r}",
    flush=True,
)

# %% Relayer (i,j) heatmap — plot the swept region (y = start i, x = end j)
baseline_score = score_mat[0, 0]
i_lo = SWEEP_I_MIN
i_hi = min(SWEEP_J_MAX - 1, num_layers)
j_lo = SWEEP_I_MIN + 1
j_hi = min(SWEEP_J_MAX, num_layers + 1)
region = np.asarray(score_mat[i_lo:i_hi, j_lo:j_hi], dtype=float)
region_plot = region - float(baseline_score)
title = f"{MODEL_ID} − math_16 mean score − baseline (0,0)={baseline_score:.3f}; similarity clipped [0,1]"
cb_label = "Δ mean score"
vrange = 0.35
fig, ax = plt.subplots(figsize=(12, 12))
im = ax.imshow(
    np.ma.masked_invalid(region_plot),
    origin="upper",
    aspect="equal",
    cmap="RdBu_r",
    vmin=-vrange,
    vmax=vrange,
)
ax.set_xlabel("j (block end)")
ax.set_ylabel("i (block start)")
ax.set_title(
    f"{title}\nN={num_layers} layers, n={len(keys_sweep)} items; "
    f"i∈[{i_lo},{i_hi}), j∈[{j_lo},{j_hi})"
)
ax.set_xticks(np.arange(region_plot.shape[1]))
ax.set_yticks(np.arange(region_plot.shape[0]))
ax.set_xticklabels([str(v) for v in range(j_lo, j_hi)])
ax.set_yticklabels([str(v) for v in range(i_lo, i_hi)])
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=cb_label)
fig.tight_layout()
stonesoup.show(basename=f"relayer_ij_math16_Qwen3.5-27B", fig=fig)


# %% Baseline (0,0) — question, answer, prediction per item
layer_indices = expand_single_block(num_layers, (0, 0))
dup = build_model_with_layers(model, layer_indices)
dup.eval()
pad_id = int(tokenizer.pad_token_id or tokenizer.eos_token_id)
baseline_rows: list[dict[str, str | int]] = []
for b0 in range(0, len(sweep_keys_valid), SWEEP_BATCH_SIZE):
    stonesoup.check_abort()
    chunk_keys = sweep_keys_valid[b0 : b0 + SWEEP_BATCH_SIZE]
    row_ids: list[torch.Tensor] = []
    row_mask: list[torch.Tensor] = []
    for key in chunk_keys:
        item = dataset_sweep[key]
        msgs = [
            {"role": "system", "content": MATH_SYSTEM_PROMPT},
            {"role": "user", "content": item["question"]},
        ]
        one = tokenizer.apply_chat_template(
            msgs,
            enable_thinking=False,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        one = {k: v.to(device) for k, v in one.items()}
        row_ids.append(one["input_ids"])
        row_mask.append(one["attention_mask"])
    max_len = max(int(t.shape[1]) for t in row_ids)
    batched_ids: list[torch.Tensor] = []
    batched_mask: list[torch.Tensor] = []
    for ids, am in zip(row_ids, row_mask, strict=True):
        pad = max_len - ids.shape[1]
        if pad:
            ids = torch.cat(
                [torch.full((1, pad), pad_id, dtype=ids.dtype, device=device), ids],
                dim=1,
            )
            am = torch.cat(
                [torch.zeros((1, pad), dtype=am.dtype, device=device), am],
                dim=1,
            )
        batched_ids.append(ids)
        batched_mask.append(am)
    batch_input_ids = torch.cat(batched_ids, dim=0)
    batch_attn = torch.cat(batched_mask, dim=0)
    with torch.inference_mode():
        out_ids = dup.generate(
            input_ids=batch_input_ids,
            attention_mask=batch_attn,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=pad_id,
        )
    for bi, key in enumerate(chunk_keys):
        ans = int(dataset_sweep[key]["answer"])
        completion = tokenizer.decode(out_ids[bi, max_len:], skip_special_tokens=True)
        pred = parse_first_integer(completion)
        q = dataset_sweep[key]["question"]
        baseline_rows.append(
            {
                "key": key,
                "question": q,
                "answer": ans,
                "prediction": pred,
                "output": " ".join(completion.split()),
            }
        )
del dup
print(
    f"Baseline (i,j)=(0,0); mean_score={score_mat[0, 0]!r} (NaN if not run yet)",
    flush=True,
)
for row in baseline_rows:
    print(
        f"\n--- key={row['key']} score={similarity_score(int(row['prediction']), int(row['answer'])):.6f} ---\n"
        f"Q: {row['question']}\n"
        f"answer: {row['answer']!r}  prediction: {row['prediction']!r}\n"
        f"output: {row['output']!r}",
        flush=True,
    )
