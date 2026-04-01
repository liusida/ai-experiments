# %% Imports & eager-attention ablation helpers

from __future__ import annotations

import copy
from contextlib import contextmanager
from typing import Any, Literal

import torch
import torch.nn.functional as F

import stonesoup
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen35_m
from transformers.models.qwen3_vl import modeling_qwen3_vl as qwen3vl_m
from transformers.models.qwen3_5.modeling_qwen3_5 import repeat_kv

# Snapshots before any monkey-patch (both families share the same ``repeat_kv`` math).
_HF_EAGER_Q35 = qwen35_m.eager_attention_forward
_HF_EAGER_Q3VL = qwen3vl_m.eager_attention_forward

# Set in **Load model** to the module whose ``eager_attention_forward`` is patched (`qwen35_m` or `qwen3vl_m`).
ACTIVE_ATTN_MODULE: Any = qwen35_m
HF_EAGER_ORIGINAL: Any = _HF_EAGER_Q35

# Mode: ``none`` | ``block_all`` | ``single_layer`` (all heads at one layer keep full softmax on forbidden cells)
# | ``single_head`` (one head at one layer keeps full softmax).
_AblationMode = Literal["none", "block_all", "single_layer", "single_head"]
ABLATION_STATE: dict[str, Any] = {
    "mode": "none",
    "image_key_indices": None,  # LongTensor 1D, CPU or device
    "first_post_image_q": 0,  # int — forbid (q >= this, k in image_keys)
    "only_layer": None,  # int — target decoder index for ``single_layer`` / ``single_head``
    "only_head": None,  # int | None — head index for ``single_head`` only
}

# Per ``greedy_decode_ablated`` run (mode != ``none``): sum of (softmax − final) on forbidden cells, LM layers only.
ABLATION_MASS_ACC: dict[str, float | int] = {"sum_delta": 0.0, "n": 0}
LAST_ABLATION_MASS_STATS: dict[str, float | int] | None = None

# For query rows ≥ first post-image index: scale image-key weights by α, then row-renormalize.
# Set to ``1.0`` to disable. Does not run in ``mode == "none"`` (HF path unchanged).
# Printed ``mass Δ`` uses weights **before** this boost (ablation-only).
IMAGE_KEY_ATTN_BOOST_ALPHA = 1024


def _boost_postimage_image_cols_renorm(
    w: torch.Tensor,
    *,
    image_key_indices: torch.Tensor,
    first_post_image_q: int,
    alpha: float,
) -> torch.Tensor:
    """Boost attention to image keys from post-image queries only; keeps each row a distribution."""
    if alpha == 1.0:
        return w
    _b, _h, tq, tk = w.shape
    device = w.device
    idx = image_key_indices.to(device=device, dtype=torch.long)
    idx = idx[(idx >= 0) & (idx < tk)]
    if idx.numel() == 0 or first_post_image_q >= tq:
        return w
    out = w.clone()
    out[:, :, first_post_image_q:, idx] *= alpha
    denom = out[:, :, first_post_image_q:, :].sum(dim=-1, keepdim=True).clamp_min(1e-10)
    out[:, :, first_post_image_q:, :] = out[:, :, first_post_image_q:, :] / denom
    return out


def _forbidden_postimage_to_image(
    *,
    Tq: int,
    Tk: int,
    device: torch.device,
    image_key_indices: torch.Tensor,
    first_post_image_q: int,
) -> torch.Tensor:
    """Boolean [Tq, Tk] — True where we zero attention mass (post-image query → image key)."""
    img_k = torch.zeros(Tk, dtype=torch.bool, device=device)
    idx = image_key_indices.to(device=device, dtype=torch.long)
    idx = idx[(idx >= 0) & (idx < Tk)]
    if idx.numel() > 0:
        img_k[idx] = True
    post_q = torch.zeros(Tq, dtype=torch.bool, device=device)
    if first_post_image_q < Tq:
        post_q[first_post_image_q:] = True
    return post_q.unsqueeze(1) & img_k.unsqueeze(0)


def _renormalize_attn_rows(
    w: torch.Tensor,
    forbidden: torch.Tensor,
) -> torch.Tensor:
    """``w`` [B,H,Tq,Tk]; ``forbidden`` [Tq,Tk]. Zero forbidden entries; renormalize along keys."""
    B, H, Tq, Tk = w.shape
    dtype = w.dtype
    m = forbidden.view(1, 1, Tq, Tk)
    w2 = w.masked_fill(m, 0.0)
    denom = w2.sum(dim=-1, keepdim=True)
    eps = 1e-10
    allow = ~forbidden
    fallback = allow.to(dtype=dtype) / allow.sum(dim=-1, dtype=torch.float32, keepdim=True).clamp_min(eps)
    fallback = fallback.to(dtype).clamp_min(0.0)
    safe = denom.squeeze(-1) > eps
    safe_exp = safe.unsqueeze(-1)
    out = torch.where(safe_exp, w2 / denom.clamp_min(eps), fallback.view(1, 1, Tq, Tk))
    return out


def eager_attention_forward_ablated(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    st = ABLATION_STATE
    mode: _AblationMode = st["mode"]
    if mode == "none" or st["image_key_indices"] is None:
        return HF_EAGER_ORIGINAL(module, query, key, value, attention_mask, scaling, dropout, **kwargs)

    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)

    img_idx = st["image_key_indices"]
    first_q = int(st["first_post_image_q"])
    B, H, Tq, Tk = attn_weights.shape
    device = attn_weights.device
    forbidden = _forbidden_postimage_to_image(
        Tq=Tq, Tk=Tk, device=device, image_key_indices=img_idx, first_post_image_q=first_q
    )

    orig = attn_weights
    blocked = _renormalize_attn_rows(orig, forbidden)

    layer_idx = int(getattr(module, "layer_idx", -1))
    if mode == "block_all" or st["only_layer"] is None:
        w_out = blocked
    elif mode == "single_layer":
        only_l = int(st["only_layer"])
        w_out = orig if layer_idx == only_l else blocked
    else:  # single_head
        only_l = int(st["only_layer"])
        only_h = st["only_head"]
        if only_h is None:
            w_out = blocked
        elif layer_idx == only_l:
            oh = int(only_h)
            w_out = blocked.clone()
            w_out[:, oh, :, :] = orig[:, oh, :, :]
        else:
            w_out = blocked

    w_after_ablation = w_out
    if IMAGE_KEY_ATTN_BOOST_ALPHA != 1.0:
        w_out = _boost_postimage_image_cols_renorm(
            w_after_ablation,
            image_key_indices=img_idx,
            first_post_image_q=first_q,
            alpha=IMAGE_KEY_ATTN_BOOST_ALPHA,
        )

    if forbidden.any() and layer_idx >= 0:
        m = forbidden.view(1, 1, Tq, Tk).to(dtype=torch.float32)
        delta = (orig.float() - w_after_ablation.float()).mul(m).sum().item()
        ABLATION_MASS_ACC["sum_delta"] = float(ABLATION_MASS_ACC["sum_delta"]) + delta
        ABLATION_MASS_ACC["n"] = int(ABLATION_MASS_ACC["n"]) + 1

    w_out = F.dropout(w_out, p=dropout, training=module.training)
    attn_output = torch.matmul(w_out, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, w_out


@contextmanager
def ablation_scope(
    *,
    mode: _AblationMode,
    image_key_indices: torch.Tensor,
    first_post_image_q: int,
    only_layer: int | None = None,
    only_head: int | None = None,
):
    prev = copy.copy(ABLATION_STATE)
    ABLATION_STATE["mode"] = mode
    ABLATION_STATE["image_key_indices"] = image_key_indices
    ABLATION_STATE["first_post_image_q"] = first_post_image_q
    ABLATION_STATE["only_layer"] = only_layer
    ABLATION_STATE["only_head"] = only_head
    ACTIVE_ATTN_MODULE.eager_attention_forward = eager_attention_forward_ablated
    try:
        yield
    finally:
        for k, v in prev.items():
            ABLATION_STATE[k] = v
        ACTIVE_ATTN_MODULE.eager_attention_forward = HF_EAGER_ORIGINAL


def set_ablation_none():
    """Restore HF eager attention (no post-softmax edits). Useful between cells."""
    ABLATION_STATE["mode"] = "none"
    ACTIVE_ATTN_MODULE.eager_attention_forward = HF_EAGER_ORIGINAL


print(
    "Cells: imports | load | tokenize | greedy | baseline | block-all | sweep (layers then heads)\n",
    flush=True,
)

# %% Load model — Qwen3.5‑9B or Qwen3‑VL‑8B‑Instruct (Stonesoup **Load** or this cell)

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
model, processor = stonesoup.load_model(MODEL_ID)
tokenizer = getattr(processor, "tokenizer", processor)
device = next(model.parameters()).device
img_tok = int(model.config.image_token_id)
lm_cfg = model.model.language_model.config

_model_type = getattr(model.config, "model_type", "")
if _model_type == "qwen3_vl":
    ACTIVE_ATTN_MODULE = qwen3vl_m
    HF_EAGER_ORIGINAL = _HF_EAGER_Q3VL
else:
    # Qwen3.5 multimodal / text stack uses ``modeling_qwen3_5`` eager attention.
    ACTIVE_ATTN_MODULE = qwen35_m
    HF_EAGER_ORIGINAL = _HF_EAGER_Q35
ACTIVE_ATTN_MODULE.eager_attention_forward = HF_EAGER_ORIGINAL

_layer_types = getattr(lm_cfg, "layer_types", None)
if _layer_types:
    full_attn_layers = [i for i, t in enumerate(_layer_types) if t == "full_attention"]
    n_layers_msg = len(_layer_types)
else:
    full_attn_layers = list(range(lm_cfg.num_hidden_layers))
    n_layers_msg = int(lm_cfg.num_hidden_layers)
n_heads = int(lm_cfg.num_attention_heads)

model.eval()
model.set_attn_implementation("eager")

print(
    f"Loaded {MODEL_ID}  model_type={_model_type!r}  device={device}  image_token_id={img_tok}\n"
    f"attention patch module: {ACTIVE_ATTN_MODULE.__name__}\n"
    f"decoder layers={n_layers_msg}  sweep layers (full MHA)={len(full_attn_layers)}  indices={full_attn_layers}\n"
    f"num_attention_heads={n_heads}",
    flush=True,
)

# %% Paths, prompt, expected answer

ROOT = stonesoup.repo_root()
IMAGE_PATH = ROOT / "data" / "images" / "MoD" / "0.png"
USER_PROMPT = "Respond with a single word. Is it a cat or a dog?"
EXPECTED_SUBSTR = "dog"
MAX_NEW_TOKENS = 1

_messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": str(IMAGE_PATH.resolve())},
            {"type": "text", "text": USER_PROMPT},
        ],
    },
]

if _model_type == "qwen2_5_vl":
    from qwen_vl_utils import process_vision_info

    _chat_text = processor.apply_chat_template(_messages, tokenize=False, add_generation_prompt=True)
    _image_inputs, _video_inputs = process_vision_info(_messages)
    _template_inputs = processor(
        text=[_chat_text],
        images=_image_inputs,
        videos=_video_inputs,
        padding=True,
        return_tensors="pt",
    )
else:
    try:
        _template_inputs = processor.apply_chat_template(
            _messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=True,
            return_tensors="pt",
        )
    except TypeError:
        _template_inputs = processor.apply_chat_template(
            _messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

print(f"IMAGE_PATH={IMAGE_PATH} exists={IMAGE_PATH.is_file()}", flush=True)
print(f"PROMPT={USER_PROMPT!r}  expect output contains {EXPECTED_SUBSTR!r}", flush=True)

# %% Greedy decode (no KV cache; full sequence each step — matches ablation indices)

def _vision_meta(input_ids_1d: torch.Tensor) -> tuple[torch.Tensor, int]:
    vis = (input_ids_1d == img_tok).nonzero(as_tuple=False).flatten()
    if vis.numel() == 0:
        raise RuntimeError("no vision placeholder tokens in input_ids")
    first_post = int(vis.max().item()) + 1
    return vis, first_post


def prefill_last_logits(
    *,
    mode: _AblationMode,
    only_layer: int | None = None,
    only_head: int | None = None,
    token_ids: list[int],
) -> dict[int, float]:
    """Last-position logits on the **prompt** (no new tokens), same masking as ``greedy_decode_ablated``."""
    if mode != "none":
        ABLATION_MASS_ACC["sum_delta"] = 0.0
        ABLATION_MASS_ACC["n"] = 0
    inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in _template_inputs.items()}
    ids = inputs["input_ids"]
    mm = inputs["mm_token_type_ids"]
    vis_pos, first_post = _vision_meta(ids[0])
    try:
        with ablation_scope(
            mode=mode,
            image_key_indices=vis_pos,
            first_post_image_q=first_post,
            only_layer=only_layer,
            only_head=only_head,
        ):
            with torch.inference_mode():
                attn = torch.ones(ids.shape[:2], device=device, dtype=torch.long)
                out = model(
                    input_ids=ids,
                    attention_mask=attn,
                    mm_token_type_ids=mm,
                    pixel_values=inputs["pixel_values"],
                    image_grid_thw=inputs["image_grid_thw"],
                    use_cache=False,
                )
        vec = out.logits[0, -1, :].float()
        return {tid: float(vec[tid].item()) for tid in token_ids}
    finally:
        if mode != "none":
            ABLATION_MASS_ACC["sum_delta"] = 0.0
            ABLATION_MASS_ACC["n"] = 0


def prefill_logits_for_block_and_base_tokens(
    *,
    mode: _AblationMode,
    only_layer: int | None = None,
    only_head: int | None = None,
    tok_from_block_all: int,
    tok_from_baseline: int,
) -> tuple[float, float]:
    """Last-position logit for **block-all’s first token** and **baseline’s first token** under this ablation."""
    tid = [tok_from_block_all]
    if tok_from_baseline != tok_from_block_all:
        tid.append(tok_from_baseline)
    d = prefill_last_logits(mode=mode, only_layer=only_layer, only_head=only_head, token_ids=tid)
    return float(d[tok_from_block_all]), float(d[tok_from_baseline])


def _finalize_mass_stats(mode: _AblationMode) -> None:
    global LAST_ABLATION_MASS_STATS
    if mode == "none":
        LAST_ABLATION_MASS_STATS = None
    else:
        n = int(ABLATION_MASS_ACC["n"])
        s = float(ABLATION_MASS_ACC["sum_delta"])
        LAST_ABLATION_MASS_STATS = {
            "sum_zeroed_mass": s,
            "n_lm_attn_forwards": n,
            "mean_zeroed_mass_per_forward": s / n if n else 0.0,
        }


def format_mass_stats(st: dict[str, float | int] | None) -> str:
    if not st or int(st.get("n_lm_attn_forwards", 0)) == 0:
        return "mass Δ: n/a"
    return (
        f"mass Δ sum={float(st['sum_zeroed_mass']):.4f} "
        f"mean/attn={float(st['mean_zeroed_mass_per_forward']):.6f} "
        f"({int(st['n_lm_attn_forwards'])} LM self-attn forwards)"
    )


def greedy_decode_ablated(
    *,
    mode: _AblationMode,
    only_layer: int | None = None,
    only_head: int | None = None,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> tuple[str, int]:
    if mode != "none":
        ABLATION_MASS_ACC["sum_delta"] = 0.0
        ABLATION_MASS_ACC["n"] = 0

    inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in _template_inputs.items()}
    ids = inputs["input_ids"].clone()
    mm = inputs["mm_token_type_ids"].clone()
    vis_pos, first_post = _vision_meta(ids[0])
    eos_id = tokenizer.eos_token_id

    gen_ids: list[int] = []
    with ablation_scope(
        mode=mode,
        image_key_indices=vis_pos,
        first_post_image_q=first_post,
        only_layer=only_layer,
        only_head=only_head,
    ):
        with torch.inference_mode():
            for _ in range(max_new_tokens):
                attn = torch.ones(ids.shape[:2], device=device, dtype=torch.long)
                out = model(
                    input_ids=ids,
                    attention_mask=attn,
                    mm_token_type_ids=mm,
                    pixel_values=inputs["pixel_values"],
                    image_grid_thw=inputs["image_grid_thw"],
                    use_cache=False,
                )
                nxt = int(out.logits[0, -1, :].argmax().item())
                gen_ids.append(nxt)
                nxt_t = torch.tensor([[nxt]], device=device, dtype=ids.dtype)
                ids = torch.cat([ids, nxt_t], dim=1)
                mm = torch.cat([mm, torch.zeros_like(nxt_t, dtype=mm.dtype)], dim=1)
                if eos_id is not None and nxt == eos_id:
                    break

    out = tokenizer.decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    _finalize_mass_stats(mode)
    first_id = int(gen_ids[0]) if gen_ids else -1
    return out, first_id


def prints_answer(label: str, text: str) -> bool:
    ok = EXPECTED_SUBSTR.lower() in text.lower().strip()
    print(f"{label}: {text!r}  →  {'OK' if ok else 'no match'}", flush=True)
    return ok


def answer_ok(text: str) -> bool:
    return EXPECTED_SUBSTR.lower() in text.lower().strip()


# %% Baseline (no ablation)

set_ablation_none()
text0, FIRST_TOK_BASELINE = greedy_decode_ablated(mode="none")
prints_answer("baseline", text0)

# %% Block all post-image → image attention (all heads, all full-attention layers)

text_block, FIRST_TOK_BLOCK_ALL = greedy_decode_ablated(mode="block_all")
prints_answer("block_all_postimage_to_image", text_block)
print(format_mass_stats(LAST_ABLATION_MASS_STATS), flush=True)

# Anchor prefill only (pure ``block_all`` vs ``none``). Each sweep row re-measures both ids under that row’s ablation.
if FIRST_TOK_BASELINE < 0 or FIRST_TOK_BLOCK_ALL < 0:
    raise RuntimeError("baseline or block_all produced no token; check MAX_NEW_TOKENS / EOS")
REF_LOGIT_BLOCK_TOK_AT_BLOCK = prefill_last_logits(mode="block_all", token_ids=[FIRST_TOK_BLOCK_ALL])[
    FIRST_TOK_BLOCK_ALL
]
REF_LOGIT_BASE_TOK_AT_BASELINE = prefill_last_logits(mode="none", token_ids=[FIRST_TOK_BASELINE])[
    FIRST_TOK_BASELINE
]
_lbl_blk = tokenizer.decode([FIRST_TOK_BLOCK_ALL], skip_special_tokens=False, clean_up_tokenization_spaces=False)
_lbl_base = tokenizer.decode([FIRST_TOK_BASELINE], skip_special_tokens=False, clean_up_tokenization_spaces=False)
print(
    f"anchor prefill (last pos): id={FIRST_TOK_BLOCK_ALL} {_lbl_blk!r} @block_all={REF_LOGIT_BLOCK_TOK_AT_BLOCK:.4f} | "
    f"id={FIRST_TOK_BASELINE} {_lbl_base!r} @baseline={REF_LOGIT_BASE_TOK_AT_BASELINE:.4f}",
    flush=True,
)
print("(sweep rows: same two vocab ids, logits under **that** row’s ``single_layer`` / ``single_head``.)", flush=True)

# %% Sweep: layer pass (all heads at that layer) then heads only inside passing layers

hits: list[tuple[int, int, str, bool]] = []
layers_pass: list[int] = []
n_layer_runs = 0
n_head_runs = 0

for layer in full_attn_layers:
    stonesoup.check_abort()
    n_layer_runs += 1
    t_layer, _ = greedy_decode_ablated(mode="single_layer", only_layer=layer, only_head=None)
    ok_layer = answer_ok(t_layer)
    _lb, _lbase = prefill_logits_for_block_and_base_tokens(
        mode="single_layer",
        only_layer=layer,
        only_head=None,
        tok_from_block_all=FIRST_TOK_BLOCK_ALL,
        tok_from_baseline=FIRST_TOK_BASELINE,
    )
    print(
        f"layer sweep L{layer:02d} (all heads) → {t_layer!r}  {'OK' if ok_layer else '--'}  |  "
        f"{format_mass_stats(LAST_ABLATION_MASS_STATS)}  |  "
        f"blk_tok@{FIRST_TOK_BLOCK_ALL}={_lb:.4f} base_tok@{FIRST_TOK_BASELINE}={_lbase:.4f}",
        flush=True,
    )
    if ok_layer:
        layers_pass.append(layer)

if not layers_pass:
    print(
        f"\nNo layer passed the coarse test — skipping head sweep "
        f"(try block_all vs baseline, or lower {EXPECTED_SUBSTR!r} check).",
        flush=True,
    )

for layer in layers_pass:
    for head in range(n_heads):
        stonesoup.check_abort()
        n_head_runs += 1
        t, _ = greedy_decode_ablated(mode="single_head", only_layer=layer, only_head=head)
        ok = answer_ok(t)
        hits.append((layer, head, t, ok))
        _lb, _lbase = prefill_logits_for_block_and_base_tokens(
            mode="single_head",
            only_layer=layer,
            only_head=head,
            tok_from_block_all=FIRST_TOK_BLOCK_ALL,
            tok_from_baseline=FIRST_TOK_BASELINE,
        )
        print(
            f"L{layer:02d} H{head:02d} → {t!r}  {'OK' if ok else '--'}  |  "
            f"{format_mass_stats(LAST_ABLATION_MASS_STATS)}  |  "
            f"blk_tok@{FIRST_TOK_BLOCK_ALL}={_lb:.4f} base_tok@{FIRST_TOK_BASELINE}={_lbase:.4f}",
            flush=True,
        )

_naive_total = len(full_attn_layers) * n_heads
_runs_total = n_layer_runs + n_head_runs
_delta = _naive_total - _runs_total
ok_pairs = [(L, H, s) for L, H, s, o in hits if o]
_saved_note = (
    f"saved {_delta} vs naive L×H"
    if _delta > 0
    else (f"+{-_delta} extra vs naive" if _delta < 0 else "same as naive")
)
print(
    f"\nCoarse: {len(layers_pass)}/{len(full_attn_layers)} layers match.\n"
    f"Runs: {n_layer_runs} layer + {n_head_runs} head = {_runs_total} "
    f"(naive grid {len(full_attn_layers)}×{n_heads}={_naive_total}; {_saved_note}).\n"
    f"Summary: {len(ok_pairs)}/{len(hits)} head-level hits recover {EXPECTED_SUBSTR!r}.\n"
    f"Hits (layer, head, answer): first 10 of {len(ok_pairs)}:",
    flush=True,
)
for row in ok_pairs[:10]:
    print(f"  {row}", flush=True)
if len(ok_pairs) > 10:
    print("  ...", flush=True)

set_ablation_none()
print("Restored default eager attention.", flush=True)
