# %% Imports, relayer & capture helpers
from __future__ import annotations

import copy
import json
import re
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import stonesoup
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate.hooks import add_hook_to_module

from stonesoup.experiment import (
    configure_matplotlib_agg,
    data_dir,
    decoder_blocks,
    ensure_pad_token_via_eos,
    hf_repo_id_safe_stem,
    inner_tokenizer,
)

configure_matplotlib_agg()


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
    """Relayer layer index list; ``exec_times`` is ``RELAYER_EXEC_TIMES`` (see config)."""
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
    repeat_times = exec_times - 1
    middle: list[int] = []
    for _ in range(repeat_times - 1):
        middle.extend(range(i, j))
    return list(range(0, j)) + middle + list(range(i, num_layers))


def relayer_repeat_bands_for_plot(exec_times: int) -> int:
    if exec_times <= 1:
        return 0
    return exec_times - 1


class LayerDuplicatedModel(nn.Module):
    """HF causal LM with decoder ``ModuleList`` replaced by a repeated index list (shared weights)."""

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
    merged = dict(inputs, **forward_kw)
    blocks = list(decoder_blocks(model))
    emb = model.get_input_embeddings()
    return _capture_embed_and_blocks_first_only(emb, blocks, merged, model)


def capture_relayer_embed_and_post_blocks(
    dup: LayerDuplicatedModel,
    inputs: dict[str, Any],
    **forward_kw: Any,
) -> tuple[torch.Tensor, list[str]]:
    merged = dict(inputs, **forward_kw)
    blocks = list(dup._new_layers)
    emb = dup.base_model.get_input_embeddings()
    return _capture_embed_and_blocks_first_only(emb, blocks, merged, dup)


def parse_first_integer(completion: str) -> int:
    compact = completion.replace(",", "")
    m = re.search(r"-?\d+", compact)
    if not m:
        return 0
    return int(m.group(0))


def mean_token_l2_norm_per_stage(stack: torch.Tensor, batch_index: int = 0) -> np.ndarray:
    norms = stack.float().norm(dim=-1)
    b = int(batch_index)
    return norms[:, b, :].mean(dim=-1).detach().cpu().numpy()


def token_l2_norm_per_stage(
    stack: torch.Tensor,
    token_index: int,
    *,
    batch_index: int = 0,
) -> np.ndarray:
    norms = stack.float().norm(dim=-1)
    b, t = int(batch_index), int(token_index)
    _seq = int(norms.shape[2])
    if t < 0 or t >= _seq:
        raise IndexError(f"token_index {t} out of range for seq_len {_seq}")
    return norms[:, b, t].detach().cpu().numpy()


def aligned_baseline_stack_idx(
    s: int,
    *,
    relayer_exec_times: int,
    layer_indices: list[int],
    cj: int,
    s_after_repeat: int,
) -> int | None:
    """Baseline ``stack_base`` row index aligned to relayer stack row ``s``."""
    if relayer_exec_times == 1:
        return s
    if relayer_exec_times == 0:
        if s == 0:
            return 0
        return layer_indices[s - 1] + 1
    if s <= cj:
        return s
    if s >= s_after_repeat:
        return cj + 1 + (s - s_after_repeat)
    return None


def stack_idx_to_layer_label(bi: int | None) -> str:
    if bi is None:
        return "—"
    if bi == 0:
        return "embed"
    return str(bi - 1)


class AffineTranslator(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.weight = nn.Parameter(torch.eye(d))
        self.bias = nn.Parameter(torch.zeros(d))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        w, b = self.weight.float(), self.bias.float()
        return (h.float() @ w.T + b).to(h.dtype)


class TunedLens(nn.Module):
    def __init__(self, n_probed: int, d: int):
        super().__init__()
        self.translators = nn.ModuleList([AffineTranslator(d) for _ in range(n_probed)])


def load_tuned_lens(
    pt: Path,
    model: nn.Module,
    device: torch.device,
) -> tuple[TunedLens, int, nn.Module, nn.Module] | None:
    """Return ``(lens, n_probed, final_norm, lm_head)`` or ``None`` if ``pt`` missing."""
    if not pt.is_file():
        return None
    ckpt = torch.load(pt, map_location=device)
    n = int(ckpt["n_probed_layers"])
    d = int(ckpt["d_model"])
    text_mod = getattr(model.model, "language_model", model.model)
    lens = TunedLens(n, d).to(device=device, dtype=next(model.parameters()).dtype)
    lens.load_state_dict(ckpt["state_dict"], strict=True)
    lens.eval()
    return lens, n, text_mod.norm, model.lm_head


def lens_logits_argmax_prob(logits_1d: torch.Tensor, tok: Any) -> tuple[str, float]:
    lf = logits_1d.float()
    tid = int(lf.argmax().item())
    p = float(torch.softmax(lf, dim=-1)[tid].item())
    return tok.decode([tid]), p


def format_lens_plot_caption(text: str, prob: float, max_chars: int = 20) -> str:
    t = text.replace("\n", " ").strip() or "∅"
    if len(t) > max_chars:
        t = t[: max_chars - 1] + "…"
    return f"[{t}] {prob:.2f}"


def build_investigate_tuned_lens_table_rows(
    *,
    len_rel: int,
    stack_base: torch.Tensor,
    stack_rel: torch.Tensor,
    abs_idx: int,
    relayer_exec_times: int,
    layer_indices: list[int],
    cj: int,
    s_after_repeat: int,
    orig_line_rel: list[str],
    lens_mod: TunedLens,
    n_lens: int,
    final_norm: nn.Module,
    lm_head: nn.Module,
    tokenizer: Any,
) -> list[dict[str, Any]]:
    """Decode rows for the tuned-lens HTML table; each dict includes ``rel_stage`` (drop before display)."""
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for s in range(len_rel):
            bi = aligned_baseline_stack_idx(
                s,
                relayer_exec_times=relayer_exec_times,
                layer_indices=layer_indices,
                cj=cj,
                s_after_repeat=s_after_repeat,
            )
            rel_lbl = orig_line_rel[s] if s < len(orig_line_rel) else "?"
            base_lbl = stack_idx_to_layer_label(bi)
            hr = stack_rel[s, 0, abs_idx]
            if bi is None:
                lr = lm_head(final_norm(hr.unsqueeze(0).unsqueeze(0)))[0, 0].float()
                txt_r, p_r = lens_logits_argmax_prob(lr, tokenizer)
                rows.append(
                    {
                        "rel_stage": s,
                        "rel_layer": rel_lbl,
                        "base_layer": base_lbl,
                        "lens_argmax_base_h": None,
                        "lens_prob_base_h": None,
                        "lens_argmax_rel_h": txt_r,
                        "lens_prob_rel_h": round(p_r, 6),
                        "note": "overlap (rel raw)",
                    }
                )
                continue
            if bi < 0 or bi >= stack_base.shape[0]:
                continue
            hb = stack_base[bi, 0, abs_idx]
            if bi < n_lens:
                tr = lens_mod.translators[bi]
                lb = lm_head(final_norm(tr(hb.unsqueeze(0).unsqueeze(0))))[0, 0].float()
                lr = lm_head(final_norm(tr(hr.unsqueeze(0).unsqueeze(0))))[0, 0].float()
                note = ""
            else:
                lb = lm_head(final_norm(hb.unsqueeze(0).unsqueeze(0)))[0, 0].float()
                lr = lm_head(final_norm(hr.unsqueeze(0).unsqueeze(0)))[0, 0].float()
                note = "raw (no translator idx)"
            txt_b, p_b = lens_logits_argmax_prob(lb, tokenizer)
            txt_r, p_r = lens_logits_argmax_prob(lr, tokenizer)
            rows.append(
                {
                    "rel_stage": s,
                    "rel_layer": rel_lbl,
                    "base_layer": base_lbl,
                    "lens_argmax_base_h": txt_b,
                    "lens_prob_base_h": round(p_b, 6),
                    "lens_argmax_rel_h": txt_r,
                    "lens_prob_rel_h": round(p_r, 6),
                    "note": note,
                }
            )
    return rows


MATH_SYSTEM_PROMPT = (
    "You are a highly intelligent AI. You have extraordinary intuition and can "
    "easily make accurate estimations. For the following questions, you will "
    "always provide an answer, even if you are not certain."
)

# %% Config # stonesoup:cell-input
MODEL_ID = "Qwen/Qwen3.5-9B"
# CELL_INPUT: relayer mode — 0 skip block, 1 baseline, 2 one overlap, 3+ extra repeats
RELAYER_EXEC_TIMES = int(str(globals().get("CELL_INPUT", "") or "").strip() or "0")
RELAYER_BLOCK: tuple[int, int] = (13, 15)
TUNED_LENS_PT = stonesoup.script_dir() / f"tuned_lens_math_{hf_repo_id_safe_stem(MODEL_ID)}.pt"
DATA_PATH = data_dir() / "rys-dataset" / "mid_math_16.json"
FIRST_KEY = "4"

# %% Load model
torch.set_grad_enabled(False)

model, processor = stonesoup.load_model(MODEL_ID)
model.eval()
tokenizer = inner_tokenizer(processor)
ensure_pad_token_via_eos(tokenizer)
device = next(model.parameters()).device

# %% Relayer wrapper
N_DEC = decoder_num_layers(model)
layer_indices = expand_block_with_repeats(N_DEC, RELAYER_BLOCK, RELAYER_EXEC_TIMES)
dup = build_model_with_layers(model, layer_indices)
dup.eval()
print(
    f"decoder layers: base={N_DEC} RELAYER_EXEC_TIMES={RELAYER_EXEC_TIMES} "
    f"relayer depth={len(layer_indices)} head={layer_indices[:8]!r}…",
    flush=True,
)

# %% RYS item + chat template
with open(DATA_PATH, encoding="utf-8") as f:
    dataset = json.load(f)
item = dataset[FIRST_KEY]
msgs = [
    {"role": "system", "content": MATH_SYSTEM_PROMPT},
    {"role": "user", "content": item["question"]},
]
inputs = tokenizer.apply_chat_template(
    msgs,
    enable_thinking=False,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
)
print(json.dumps(msgs, indent=2, ensure_ascii=False), flush=True)
print(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True), flush=True)
inputs = {k: v.to(device) for k, v in inputs.items()}
print(f"key={FIRST_KEY!r} seq_len={inputs['input_ids'].shape[1]}", flush=True)

# %% Generate baseline vs relayer
MAX_NEW_TOKENS = 32
prompt_len = int(inputs["input_ids"].shape[1])
pad_id_gen = int(tokenizer.pad_token_id or tokenizer.eos_token_id)
ground_truth = int(item["answer"])
stonesoup.check_abort()
with torch.inference_mode():
    gen_base = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=pad_id_gen,
    )
    gen_rel = dup.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=pad_id_gen,
    )
completion_base = tokenizer.decode(gen_base[0, prompt_len:], skip_special_tokens=True)
completion_rel = tokenizer.decode(gen_rel[0, prompt_len:], skip_special_tokens=True)
pred_base = parse_first_integer(completion_base)
pred_rel = parse_first_integer(completion_rel)
print(
    f"ground_truth={ground_truth}  baseline pred={pred_base} ok={pred_base == ground_truth} {completion_base!r}\n"
    f"relayer pred={pred_rel} ok={pred_rel == ground_truth} {completion_rel!r}",
    flush=True,
)

# %% Capture + L2 norm plot # stonesoup:cell-input
# CELL_INPUT: which generated token to study (0 = first); readout is the hidden state
# *before* that token (same position used to predict it), not after it is processed.
_raw_tok = str(globals().get("CELL_INPUT", "") or "").strip()
token_idx = int(_raw_tok) if _raw_tok else 0
n_new = int(gen_base.shape[1] - prompt_len)
if n_new <= 0:
    raise RuntimeError("No completion tokens; run Generate cell first.")
if token_idx < 0 or token_idx >= n_new:
    raise IndexError(f"token_idx {token_idx} not in [0, {n_new})")

abs_idx = prompt_len + token_idx - 1
if abs_idx < 0:
    raise ValueError("abs_idx < 0 (prompt must be non-empty to predict the first completion token)")

continuation = gen_base[:, prompt_len:]
inputs_capture = {
    "input_ids": torch.cat([inputs["input_ids"], continuation], dim=1),
    "attention_mask": torch.cat(
        [
            inputs["attention_mask"],
            torch.ones(
                continuation.shape,
                dtype=inputs["attention_mask"].dtype,
                device=inputs["attention_mask"].device,
            ),
        ],
        dim=1,
    ),
}
stack_base, _ = capture_causal_lm_embed_and_post_blocks(model, inputs_capture, use_cache=False)
mean_norms_base = mean_token_l2_norm_per_stage(stack_base)
token_norm_base = token_l2_norm_per_stage(stack_base, abs_idx)
base = token_norm_base

stack_rel, _ = capture_relayer_embed_and_post_blocks(dup, inputs_capture, use_cache=False)
mean_norms_rel = mean_token_l2_norm_per_stage(stack_rel)
token_norm_rel = token_l2_norm_per_stage(stack_rel, abs_idx)
relayer_norms = token_norm_rel

x_rel = np.arange(len(mean_norms_rel))
orig_line_rel = ["embed"] + [str(layer_indices[j]) for j in range(len(layer_indices))]
xtick_labels_rel = [f"{i}\n{orig_line_rel[i]}" for i in range(len(mean_norms_rel))]

cj = int(RELAYER_BLOCK[1])
repeat_len = cj - int(RELAYER_BLOCK[0])
if RELAYER_EXEC_TIMES <= 1:
    x_base_plot = x_rel.astype(np.float64)
    if RELAYER_EXEC_TIMES == 1:
        y_base_plot = np.asarray(base, dtype=np.float64)
    else:
        y_base_plot = np.asarray(
            [base[0] if s == 0 else base[layer_indices[s - 1] + 1] for s in range(len(mean_norms_rel))],
            dtype=np.float64,
        )
    s_after_repeat = -1
else:
    s_after_repeat = cj + relayer_repeat_bands_for_plot(RELAYER_EXEC_TIMES) * repeat_len + 1
    n_tail = len(mean_norms_rel) - s_after_repeat
    x_base_plot = np.concatenate(
        [np.arange(cj + 1, dtype=np.float64), np.arange(s_after_repeat, len(mean_norms_rel), dtype=np.float64)]
    )
    y_base_plot = np.concatenate(
        [
            np.asarray(base[: cj + 1], dtype=np.float64),
            np.asarray(base[cj + 1 : cj + 1 + n_tail], dtype=np.float64),
        ]
    )

resp_tok_id = int(gen_base[0, prompt_len + token_idx])
token_fragment = tokenizer.decode([resp_tok_id])

fig_w = min(22, max(10.0, 0.28 * len(mean_norms_rel)))
fig, ax = plt.subplots(figsize=(fig_w, 5.2))
ax.plot(x_rel, relayer_norms, marker="o", ms=3, color="tab:orange", label=f"relayer {RELAYER_BLOCK}×{RELAYER_EXEC_TIMES}")
ax.plot(x_base_plot, y_base_plot, marker="s", ms=2, color="tab:blue", alpha=0.9, label="baseline")
ax.set_xticks(x_rel)
ax.set_xticklabels(xtick_labels_rel, fontsize=6, ha="center")
ax.set_xlabel('stage / base layer or "embed" (relayer stack)')
ax.set_ylabel("L2 norm at predict position (pre-target-token state)")
ax.set_title(
    f"{MODEL_ID}  L2 vs stage — predicting token {token_idx} {repr(token_fragment)} (seq pos {abs_idx})"
)
ax.legend(loc="upper right", fontsize=8)
ax.grid(True, alpha=0.3)
fig.tight_layout()
stonesoup.show(basename="investigate_9b_baseline_aligned_vs_relayer_norm", fig=fig)

# %% Cosine similarity + tuned-lens captions
len_rel = int(stack_rel.shape[0])
cos_vals = np.full(len_rel, np.nan, dtype=np.float64)
for s in range(len_rel):
    bi = aligned_baseline_stack_idx(
        s,
        relayer_exec_times=RELAYER_EXEC_TIMES,
        layer_indices=layer_indices,
        cj=cj,
        s_after_repeat=s_after_repeat,
    )
    if bi is None:
        continue
    if bi < 0 or bi >= stack_base.shape[0]:
        raise IndexError(f"baseline stack {bi} out of range for rel stage {s}")
    vb = stack_base[bi, 0, abs_idx].float()
    vr = stack_rel[s, 0, abs_idx].float()
    cos_vals[s] = F.cosine_similarity(vb.unsqueeze(0), vr.unsqueeze(0), dim=1, eps=1e-12).item()

loaded = load_tuned_lens(TUNED_LENS_PT, model, device)
INVESTIGATE_TUNED_LENS = loaded
if loaded is None:
    print(f"Cos plot: no {TUNED_LENS_PT.name} — no lens captions.", flush=True)
    lens_mod, n_lens, final_norm, lm_head = None, 0, None, None
else:
    lens_mod, n_lens, final_norm, lm_head = loaded

fig_w2 = min(22, max(10.0, 0.28 * len_rel))
fig2, ax2 = plt.subplots(figsize=(fig_w2, 6.8))
ax2.plot(x_rel, cos_vals, marker="o", ms=4, color="tab:green")
ax2.set_xticks(x_rel)
ax2.set_xticklabels(xtick_labels_rel, fontsize=6, ha="center")
ax2.set_xlabel('stage / base layer or "embed" (relayer stack)')
ax2.set_ylabel("cos(h_base, h_rel) at predict position")
ax2.set_ylim(0.4, 1.08)
ax2.axhline(0.0, color="0.5", lw=0.8, ls="--", alpha=0.6)
ax2.set_title(
    f"{MODEL_ID}  cos sim — predicting token {token_idx} {repr(token_fragment)} (seq pos {abs_idx})"
    + (
        "\n(blue above: base; orange below: rel; no affine → raw h→norm→lm_head)"
        if lens_mod is not None
        else ""
    ),
    fontsize=10,
)
ax2.grid(True, alpha=0.3)

if lens_mod is not None:
    INVESTIGATE_LENS_TABLE_ROWS = build_investigate_tuned_lens_table_rows(
        len_rel=len_rel,
        stack_base=stack_base,
        stack_rel=stack_rel,
        abs_idx=abs_idx,
        relayer_exec_times=RELAYER_EXEC_TIMES,
        layer_indices=layer_indices,
        cj=cj,
        s_after_repeat=s_after_repeat,
        orig_line_rel=orig_line_rel,
        lens_mod=lens_mod,
        n_lens=n_lens,
        final_norm=final_norm,
        lm_head=lm_head,
        tokenizer=tokenizer,
    )
    for r in INVESTIGATE_LENS_TABLE_ROWS:
        s = int(r["rel_stage"])
        if np.isnan(cos_vals[s]) or r["lens_argmax_base_h"] is None:
            continue
        tb = r["lens_argmax_base_h"]
        pb = float(r["lens_prob_base_h"])
        trt = r["lens_argmax_rel_h"]
        pr = float(r["lens_prob_rel_h"])
        x, y = float(x_rel[s]), float(cos_vals[s])
        ax2.annotate(
            format_lens_plot_caption(tb, pb),
            xy=(x, y),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=5,
            color="tab:blue",
        )
        ax2.annotate(
            format_lens_plot_caption(trt, pr),
            xy=(x, y),
            xytext=(0, -10),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=5,
            color="tab:orange",
        )
else:
    INVESTIGATE_LENS_TABLE_ROWS = None

fig2.tight_layout()
stonesoup.show(basename="investigate_9b_rel_vs_base_cos_sim", fig=fig2)

# %% Tuned lens table (HTML)
rows = globals().get("INVESTIGATE_LENS_TABLE_ROWS")
if rows is None:
    loaded = globals().get("INVESTIGATE_TUNED_LENS")
    if loaded is None:
        loaded = load_tuned_lens(TUNED_LENS_PT, model, device)
        INVESTIGATE_TUNED_LENS = loaded
    if loaded is None:
        raise FileNotFoundError(f"Train/save tuned lens first: {TUNED_LENS_PT}")
    lens_mod, n_lens, final_norm, lm_head = loaded
    rows = build_investigate_tuned_lens_table_rows(
        len_rel=len_rel,
        stack_base=stack_base,
        stack_rel=stack_rel,
        abs_idx=abs_idx,
        relayer_exec_times=RELAYER_EXEC_TIMES,
        layer_indices=layer_indices,
        cj=cj,
        s_after_repeat=s_after_repeat,
        orig_line_rel=orig_line_rel,
        lens_mod=lens_mod,
        n_lens=n_lens,
        final_norm=final_norm,
        lm_head=lm_head,
        tokenizer=tokenizer,
    )
else:
    loaded = globals().get("INVESTIGATE_TUNED_LENS")
    if loaded is None:
        raise FileNotFoundError(f"Train/save tuned lens first: {TUNED_LENS_PT}")
    _, n_lens, _, _ = loaded

df_lens = pd.DataFrame([{k: v for k, v in r.items() if k != "rel_stage"} for r in rows])
stonesoup.html()
print(
    f"Tuned lens {TUNED_LENS_PT.name}  n_probed={n_lens}  predict pos={abs_idx} (target completion idx {token_idx})\n"
    "rel_layer / base_layer = decoder id; lens_* = argmax after Affine→norm→lm_head "
    "(overlap: no aligned base row — rel via raw h→norm→lm_head; final stack row: both raw if no translator).",
    flush=True,
)
stonesoup.display(df_lens, max_rows=len(df_lens), max_cols=len(df_lens.columns), emit_render_hint=False)
