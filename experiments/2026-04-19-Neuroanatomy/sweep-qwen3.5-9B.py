# %% Imports & helpers
from __future__ import annotations

import copy
import json
import random
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


def _first_non_whitespace(s: str) -> str | None:
    for ch in s:
        if not ch.isspace():
            return ch
    return None


def should_stop_generating(completion: str) -> bool:
    """True when further tokens cannot improve a single leading integer (invalid prefix or junk after digits)."""
    if not completion.strip():
        return False
    c = _first_non_whitespace(completion)
    if c is None:
        return False
    if c not in "-0123456789":
        return True
    m = re.match(r"^\s*(-?[\d,]+)(.*)$", completion, re.DOTALL)
    if not m:
        return False
    num_part, rest = m.group(1), m.group(2)
    if not num_part or num_part == "-":
        return False
    tail = rest.lstrip()
    if not tail:
        return False
    if tail[0].isdigit():
        return False
    return True


# --- Standalone layer relayer (same idea as RYS: swap decoder ``ModuleList``, shallow-copy layers, shared weights) ---


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
    """(i, j) shorthand: ``range(0,j) + range(i, N)`` — layers ``i..j-1`` appear twice."""
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
        """Delegate to ``base_model.generate`` under relayer layer swap (same idea as RYS ``LayerDuplicatedModel``)."""
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


def _generation_prep_model(model: nn.Module) -> nn.Module:
    """Module with ``prepare_inputs_for_generation`` / ``_update_model_kwargs_for_generation`` (GenerationMixin).

    Do not use ``model.base_model`` for generic HF modules — that often points at the inner backbone
    (e.g. ``Qwen3_5Model``) which is not the generative wrapper. ``LayerDuplicatedModel`` is unwrapped
    explicitly via ``.base_model`` to the original causal LM.
    """
    prep = model.base_model if isinstance(model, LayerDuplicatedModel) else model
    if not hasattr(prep, "prepare_inputs_for_generation"):
        raise TypeError(
            f"{type(prep).__name__} has no prepare_inputs_for_generation; "
            "expected a HuggingFace causal LM (GenerationMixin), not an inner backbone."
        )
    return prep


def generate_integer_completion(
    model,
    tokenizer,
    inputs: dict[str, torch.Tensor],
    prompt_len: int,
    max_new_tokens: int,
) -> str:
    """Greedy decode one token at a time with KV cache; stop early on EOS or should_stop_generating.

    Uses ``prepare_inputs_for_generation`` + ``_update_model_kwargs_for_generation`` like ``.generate()``.

    **Qwen3.5:** Do *not* pass generic 2D ``cumsum`` ``position_ids``. ``Qwen3_5ForCausalLM`` overrides
    ``_prepare_position_ids_for_generation`` to build the multi-row mRoPE layout (``generate()`` only does this
    when ``position_ids`` is omitted). Forcing 2D ids matches neither ``.generate()`` nor the model's RoPE path
    and badly hurts quality — especially with KV cache.
    """
    prep = _generation_prep_model(model)
    eos_id = tokenizer.eos_token_id
    input_ids = inputs["input_ids"].clone()
    attention_mask = inputs["attention_mask"]

    past_key_values = None
    model_kwargs: dict = {
        "attention_mask": attention_mask,
        "use_cache": True,
    }
    model_kwargs["position_ids"] = prep._prepare_position_ids_for_generation(
        input_ids,
        {
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "input_ids": input_ids,
        },
    )

    for _ in range(max_new_tokens):
        stonesoup.check_abort()
        with torch.inference_mode():
            next_seq = 1 if past_key_values is not None else None
            model_inputs = prep.prepare_inputs_for_generation(
                input_ids,
                past_key_values=past_key_values,
                attention_mask=model_kwargs["attention_mask"],
                position_ids=model_kwargs["position_ids"],
                use_cache=True,
                next_sequence_length=next_seq,
            )
            outputs = model(**model_inputs, return_dict=True)

        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_token], dim=1)
        model_kwargs = prep._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=False
        )
        past_key_values = model_kwargs.get("past_key_values")

        if eos_id is not None and next_token.item() == eos_id:
            break

        completion = tokenizer.decode(input_ids[0, prompt_len:], skip_special_tokens=True)
        if should_stop_generating(completion):
            break

    return tokenizer.decode(input_ids[0, prompt_len:], skip_special_tokens=True)


# %% Config # stonesoup:cell-input
MODEL_ID = "Qwen/Qwen3.5-9B"
# Longest reference answers are 28-digit products (keys 8–11). Qwen tokenizers often use ~1 token per
# digit for long bare integers, so ~12 would truncate; 32 covers the digit string plus a little slack.
MAX_NEW_TOKENS = 32
DATA_PATH = stonesoup.data_dir() / "rys-dataset" / "math_16.json"
MATH16_NUM_ITEMS = 16
# Shuffled non-baseline (i,j) pairs; (0,0) baseline always runs first, then this many.
RELAYER_SWEEP_NUM_PAIRS = 2080
# HF ``generate`` on padded batches (decoder-only: tokenizer ``padding_side`` must be ``left``).
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
# pairs (e.g. copy of ``_pairs`` or a subset). Each iteration sets ``LOOP_ITEM`` / ``LOOP_INDEX``.
_N = decoder_num_layers(model)
_rest_pairs = [(i, j) for i in range(_N) for j in range(i + 1, _N + 1)]
random.shuffle(_rest_pairs)
_pairs = [(0, 0)] + _rest_pairs[:RELAYER_SWEEP_NUM_PAIRS]

with open(DATA_PATH, encoding="utf-8") as f:
    _dataset_sweep = json.load(f)
_keys_sweep = sorted(_dataset_sweep.keys(), key=lambda k: int(k))[:MATH16_NUM_ITEMS]
if ITEM_LIMIT is not None:
    _keys_sweep = _keys_sweep[:ITEM_LIMIT]

_mat = np.full((_N, _N + 1), np.nan, dtype=np.float64)
_records: list[dict] = []
_sweep_keys_valid = [
    _k
    for _k in _keys_sweep
    if int(_dataset_sweep[_k]["answer"]) != 0
]
_sweep_step_i = 0

print(json.dumps(_pairs))

# %% Relayer (i,j) sweep — one pair
# Uses ``LOOP_ITEM`` as ``[i, j]`` when the Stonesoup pipeline injects it; otherwise advances
# through ``_pairs`` starting at ``_sweep_step_i`` (manual runs without a loop).
_li = globals().get("LOOP_ITEM")
if _li is not None:
    if not isinstance(_li, (list, tuple)) or len(_li) < 2:
        raise TypeError(f"LOOP_ITEM must be [i, j], got {type(_li).__name__}: {_li!r}")
    i, j = int(_li[0]), int(_li[1])
else:
    if _sweep_step_i >= len(_pairs):
        raise IndexError(
            "Sweep exhausted: re-run setup to reset, or use a pipeline loop with LOOP_ITEM."
        )
    i, j = _pairs[_sweep_step_i]
    _sweep_step_i += 1

stonesoup.check_abort()
_layer_indices = expand_single_block(_N, (i, j))
_dup = build_model_with_layers(model, _layer_indices)
_dup.eval()
_scores: list[float] = []
_pad_id = int(tokenizer.pad_token_id or tokenizer.eos_token_id)
for _b0 in range(0, len(_sweep_keys_valid), SWEEP_BATCH_SIZE):
    stonesoup.check_abort()
    _chunk_keys = _sweep_keys_valid[_b0 : _b0 + SWEEP_BATCH_SIZE]
    _row_ids: list[torch.Tensor] = []
    _row_m: list[torch.Tensor] = []
    for _key in _chunk_keys:
        _item = _dataset_sweep[_key]
        _msgs = [
            {"role": "system", "content": MATH_SYSTEM_PROMPT},
            {"role": "user", "content": _item["question"]},
        ]
        _one = tokenizer.apply_chat_template(
            _msgs,
            enable_thinking=False,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        _one = {k: v.to(device) for k, v in _one.items()}
        _row_ids.append(_one["input_ids"])
        _row_m.append(_one["attention_mask"])
    _maxL = max(int(t.shape[1]) for t in _row_ids)
    _batched_ids: list[torch.Tensor] = []
    _batched_m: list[torch.Tensor] = []
    for _ids, _am in zip(_row_ids, _row_m, strict=True):
        _pad = _maxL - _ids.shape[1]
        if _pad:
            _ids = torch.cat(
                [
                    torch.full(
                        (1, _pad),
                        _pad_id,
                        dtype=_ids.dtype,
                        device=device,
                    ),
                    _ids,
                ],
                dim=1,
            )
            _am = torch.cat(
                [
                    torch.zeros(
                        (1, _pad),
                        dtype=_am.dtype,
                        device=device,
                    ),
                    _am,
                ],
                dim=1,
            )
        _batched_ids.append(_ids)
        _batched_m.append(_am)
    _batch_input_ids = torch.cat(_batched_ids, dim=0)
    _batch_attn = torch.cat(_batched_m, dim=0)
    with torch.inference_mode():
        _out_ids = _dup.generate(
            input_ids=_batch_input_ids,
            attention_mask=_batch_attn,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=_pad_id,
        )
    for _bi, _key in enumerate(_chunk_keys):
        _ans = int(_dataset_sweep[_key]["answer"])
        _completion = tokenizer.decode(
            _out_ids[_bi, _maxL:], skip_special_tokens=True
        )
        _pred = parse_first_integer(_completion)
        _item_score = similarity_score(_pred, _ans)
        _scores.append(_item_score)
        if SWEEP_DEBUG_PRINT_SCORE:
            _out1 = " ".join(_completion.split())
            print(
                f"sweep (i,j)=({i},{j}) key={_key} pred={_pred} ans={_ans} "
                f"score={_item_score:.6f} output={_out1!r}",
                flush=True,
            )
del _dup
_m = sum(_scores) / len(_scores) if _scores else float("nan")
_mat[i, j] = _m
_records.append({"i": i, "j": j, "mean_score": float(_m) if _m == _m else None})
print(
    f"relayer sweep (i,j)=({i},{j}) mean_score={_m:.6f} LOOP_INDEX={globals().get('LOOP_INDEX')!r}",
    flush=True,
)

## %% Save sweep data (npz + json)
#
# _baseline_score = _mat[0, 0]
# _mat_delta = _mat - _baseline_score
#
# _out_dir = stonesoup.outputs_dir()
# _out_dir.mkdir(parents=True, exist_ok=True)
# _npz_path = _out_dir / "relayer_ij_math16.npz"
# _json_path = _out_dir / "relayer_ij_math16.json"
# np.savez_compressed(
#     _npz_path,
#     mean_score=_mat,
#     mean_score_delta=_mat_delta,
#     baseline_score=_baseline_score,
#     num_layers=_N,
# )
# with open(_json_path, "w", encoding="utf-8") as f:
#     json.dump(
#         {
#             "num_layers": _N,
#             "keys": _keys_sweep,
#             "baseline_score": float(_baseline_score)
#             if _baseline_score == _baseline_score
#             else None,
#             "results": _records,
#         },
#         f,
#         indent=2,
#     )
# print(f"Saved sweep data: {_npz_path}\n{_json_path}", flush=True)
#
## %% Relayer (i,j) heatmap — load saved sweep
# _out_dir = stonesoup.outputs_dir()
# _npz_path = _out_dir / "relayer_ij_math16.npz"
# _json_path = _out_dir / "relayer_ij_math16.json"
# with np.load(_npz_path) as _z:
#     _N = int(_z["num_layers"])
#     if "mean_score_delta" in _z:
#         _mat_plot = np.array(_z["mean_score_delta"], copy=True)
#     elif "baseline_score" in _z:
#         _mat_plot = np.array(_z["mean_score"], copy=True) - float(_z["baseline_score"])
#     else:
#         _ms = np.array(_z["mean_score"], copy=True)
#         _mat_plot = _ms - _ms[0, 0]
# with open(_json_path, encoding="utf-8") as f:
#     _meta = json.load(f)
# _keys_sweep = _meta["keys"]

# %% Relayer (i,j) heatmap — plot (y = start i, x = end j)
_baseline_score = _mat[0, 0]
_mat_f = np.asarray(_mat, dtype=float)
if _baseline_score == _baseline_score:
    _mat_plot = _mat_f - float(_baseline_score)
    _heatmap_title = "math_16 mean score − baseline (0,0); similarity clipped [0,1]"
    _cb_label = "Δ mean score"
else:
    _mat_plot = _mat_f
    _heatmap_title = "math_16 mean score (baseline (0,0) not filled yet)"
    _cb_label = "mean score"
_abs = float(np.nanmax(np.abs(_mat_plot)))
if _abs != _abs or _abs == 0.0:
    _abs = 0.01

_fig, _ax = plt.subplots(figsize=(12, 12))
_im = _ax.imshow(
    np.ma.masked_invalid(_mat_plot),
    origin="upper",
    aspect="equal",
    cmap="RdBu_r",
    vmin=-_abs,
    vmax=_abs,
)
_ax.set_xlabel("j (block end)")
_ax.set_ylabel("i (block start)")
_ax.set_title(f"{_heatmap_title}\nN={_N} layers, n={len(_keys_sweep)} items")
_ax.set_xticks(np.arange(_mat_plot.shape[1]))
_ax.set_yticks(np.arange(_mat_plot.shape[0]))
_ax.set_xticklabels([str(j) for j in range(_mat_plot.shape[1])])
_ax.set_yticklabels([str(i) for i in range(_mat_plot.shape[0])])
_fig.colorbar(_im, ax=_ax, fraction=0.046, pad=0.04, label=_cb_label)
_fig.tight_layout()
stonesoup.show(basename="relayer_ij_math16", fig=_fig)
