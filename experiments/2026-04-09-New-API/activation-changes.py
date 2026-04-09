# %% Imports and config
from __future__ import annotations

from typing import Any

import stonesoup
import torch
import torch.nn.functional as F
from stonesoup.experiment import (
    decoder_blocks,
    encode_text_inputs,
    ensure_pad_token_via_eos,
    inner_tokenizer,
)

# MODEL_ID = "Qwen/Qwen3-8B-Base"
MODEL_ID = "google/gemma-2-2b"
# MODEL_ID = "allenai/OLMo-2-0425-1B"
# MODEL_ID = "meta-llama/llama-3.2-3B"
# MODEL_ID = "mistralai/Ministral-3-3B-Base-2512"

PROMPT = "The capital of France is"
# Single sequence position for all vectors below (not a mean over tokens).
# Default -1 = last prompt token; use 0, 1, … for other positions.
TOKEN_INDEX = -2

# %% Load model
model, proc = stonesoup.load_model(MODEL_ID)
model.eval()
tokenizer = inner_tokenizer(proc)
ensure_pad_token_via_eos(tokenizer)
device = next(model.parameters()).device
print("Loaded:", MODEL_ID, device, flush=True)

# %% Capture hooks: embedding, per-layer block_in, LN→attn, attn_out, LN→mlp, mlp_out
def attention_submodule(block: torch.nn.Module) -> torch.nn.Module | None:
    for name in ("attn", "self_attn", "attention"):
        if hasattr(block, name):
            return getattr(block, name)
    return None


def ln1_submodule(block: torch.nn.Module) -> torch.nn.Module | None:
    """Pre-attention RMSNorm; OLMo-2 has none (attention runs on the residual directly)."""
    return getattr(block, "input_layernorm", None) or getattr(block, "ln_1", None)


def ln2_submodule(block: torch.nn.Module) -> torch.nn.Module | None:
    return getattr(block, "post_attention_layernorm", None) or getattr(block, "ln_2", None)


def mlp_submodule(block: torch.nn.Module) -> torch.nn.Module | None:
    return getattr(block, "mlp", None)


def post_feedforward_norm_submodule(block: torch.nn.Module) -> torch.nn.Module | None:
    """Gemma2 applies RMSNorm to MLP output before the second residual add."""
    return getattr(block, "post_feedforward_layernorm", None)


def embed_module(model: torch.nn.Module) -> torch.nn.Module:
    """Token embedding table (RoPE: no additive pos in this tensor).

    Prefer ``get_input_embeddings()`` so multimodal wrappers (e.g. Qwen3.5 ``*ForConditionalGeneration``)
    resolve the text embedder; then common ``model.embed_tokens`` / ``model.language_model.embed_tokens``.
    """
    ge = getattr(model, "get_input_embeddings", None)
    if callable(ge):
        emb = ge()
        if emb is not None:
            return emb
    inner = getattr(model, "model", None)
    if inner is not None:
        if hasattr(inner, "embed_tokens"):
            return inner.embed_tokens
        lm = getattr(inner, "language_model", None)
        if lm is not None and hasattr(lm, "embed_tokens"):
            return lm.embed_tokens
    raise TypeError(
        f"Could not resolve input embeddings on {type(model).__name__} "
        "(get_input_embeddings / model.embed_tokens / model.language_model.embed_tokens)."
    )


def vec_at_token(h: torch.Tensor, idx: int) -> torch.Tensor:
    """[batch, seq, d] → [d] for batch 0."""
    return h[0, idx, :].float()


def l2_norm(v: torch.Tensor) -> float:
    return float(v.norm().item())


def cos_pair(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=-1).item())


def residual_mid_block_out(L: dict[str, torch.Tensor], ti: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Reconstruct mid; prefer hooked full decoder output as ``block_out`` when present.

    Llama-style: mid = block_in + attn_out; block_out = mid + mlp_out (norms sit *inside* the
    submodule path before the delta is formed).

    Gemma2-style (post_feedforward_layernorm present): the residual adds are
    ``x + RMSNorm(attn_out)`` then ``x + RMSNorm(mlp_out)``; we capture those normed deltas as
    ``ln2_pre_mlp`` (post-attention RMSNorm output) and ``mlp_post_norm``.

    Gemma-4 / E2B may add MoE branches, extra norms, and ``layer_scalar``; submodule hooks alone
    cannot reconstruct that — we install a forward hook on the whole decoder block and store
    ``block_output``; when set, it overrides ``block_out`` (mid is still from attn path).
    """
    block_in = vec_at_token(L["block_in"], ti)
    attn_o = vec_at_token(L["attn_out"], ti)
    ln2_pm = vec_at_token(L["ln2_pre_mlp"], ti)
    mlp_o = vec_at_token(L["mlp_out"], ti)
    if "mlp_post_norm" in L:
        mid = block_in + ln2_pm
        mlp_post = vec_at_token(L["mlp_post_norm"], ti)
        block_out = mid + mlp_post
    else:
        mid = block_in + attn_o
        block_out = mid + mlp_o
    if "block_output" in L:
        block_out = vec_at_token(L["block_output"], ti)
    return mid, block_out


class CaptureState:
    def __init__(self) -> None:
        self.embed: torch.Tensor | None = None
        self.layers: dict[int, dict[str, torch.Tensor]] = {}

    def set_layer(self, li: int, key: str, t: torch.Tensor) -> None:
        self.layers.setdefault(li, {})[key] = t.detach()


state = CaptureState()
handles: list[Any] = []


def install_captures(model: torch.nn.Module) -> None:
    remove_handles()
    emb = embed_module(model)

    def embed_hook(mod: torch.nn.Module, inp: Any, out: torch.Tensor) -> None:
        state.embed = out.detach()

    handles.append(emb.register_forward_hook(embed_hook))

    blocks = decoder_blocks(model)
    for li, block in enumerate(blocks):
        ln1 = ln1_submodule(block)
        ln2 = ln2_submodule(block)
        attn = attention_submodule(block)
        mlp = mlp_submodule(block)
        post_ff = post_feedforward_norm_submodule(block)
        if attn is None or mlp is None or ln2 is None:
            raise RuntimeError(
                f"Layer {li} ({type(block).__name__}): missing ln2/attn/mlp for hooking."
            )

        def make_pre(li_: int):
            def pre_hook(mod: torch.nn.Module, inp: tuple[Any, ...]) -> None:
                t = inp[0].detach()
                state.set_layer(li_, "block_in", t)
                # OLMo-2: no input_layernorm; attention sees this same residual (attn() is kw-only, so no reliable attn pre-hook).
                if ln1 is None:
                    state.set_layer(li_, "ln1_pre_attn", t)

            return pre_hook

        def make_ln1(li_: int):
            def ln1_hook(mod: torch.nn.Module, inp: Any, out: torch.Tensor) -> None:
                state.set_layer(li_, "ln1_pre_attn", out.detach())

            return ln1_hook

        def make_attn(li_: int):
            def attn_hook(mod: torch.nn.Module, inp: Any, out: Any) -> None:
                first = out[0] if isinstance(out, tuple) else out
                state.set_layer(li_, "attn_out", first.detach())

            return attn_hook

        def make_ln2(li_: int):
            def ln2_hook(mod: torch.nn.Module, inp: Any, out: torch.Tensor) -> None:
                state.set_layer(li_, "ln2_pre_mlp", out.detach())

            return ln2_hook

        def make_mlp(li_: int):
            def mlp_hook(mod: torch.nn.Module, inp: Any, out: torch.Tensor) -> None:
                state.set_layer(li_, "mlp_out", out.detach())

            return mlp_hook

        def make_post_ff(li_: int):
            def post_ff_hook(mod: torch.nn.Module, inp: Any, out: torch.Tensor) -> None:
                state.set_layer(li_, "mlp_post_norm", out.detach())

            return post_ff_hook

        def make_block_output(li_: int):
            def block_out_hook(mod: torch.nn.Module, inp: Any, out: Any) -> None:
                t = out[0] if isinstance(out, tuple) else out
                state.set_layer(li_, "block_output", t.detach())

            return block_out_hook

        handles.append(block.register_forward_pre_hook(make_pre(li)))
        if ln1 is not None:
            handles.append(ln1.register_forward_hook(make_ln1(li)))
        handles.append(attn.register_forward_hook(make_attn(li)))
        handles.append(ln2.register_forward_hook(make_ln2(li)))
        handles.append(mlp.register_forward_hook(make_mlp(li)))
        if post_ff is not None:
            handles.append(post_ff.register_forward_hook(make_post_ff(li)))
        handles.append(block.register_forward_hook(make_block_output(li)))


def remove_handles() -> None:
    for h in handles:
        h.remove()
    handles.clear()


# %% One forward pass (no generation)
enc = encode_text_inputs(proc, PROMPT, device=device)
seq_len = int(enc["input_ids"].shape[1])
ti = TOKEN_INDEX if TOKEN_INDEX >= 0 else seq_len + TOKEN_INDEX
if not (0 <= ti < seq_len):
    raise ValueError(f"TOKEN_INDEX {TOKEN_INDEX} out of range for seq_len {seq_len}")

state = CaptureState()
install_captures(model)
stonesoup.check_abort()
try:
    with torch.inference_mode():
        model(**enc, use_cache=False)
finally:
    remove_handles()

if state.embed is None:
    raise RuntimeError("Embedding hook did not run.")

# %% Metrics: norms and cos sim along the residual stream (per layer, one token)
lines: list[str] = []
lines.append(f"Prompt: {PROMPT!r}  |  token index: {ti} / seq_len={seq_len}")
lines.append("")

emb_v = vec_at_token(state.embed, ti)
lines.append(f"embedding (embed_tokens)  ||h|| = {l2_norm(emb_v):.4f}")
if 0 in state.layers:
    bi0 = vec_at_token(state.layers[0]["block_in"], ti)
    lines.append(f"cos(embedding, L0 block_in)={cos_pair(emb_v, bi0):.4f}  (expect ~1)")
lines.append("")

n_layers = len(state.layers)
for li in range(n_layers):
    L = state.layers[li]
    block_in = vec_at_token(L["block_in"], ti)
    ln1_pa = vec_at_token(L["ln1_pre_attn"], ti)
    attn_o = vec_at_token(L["attn_out"], ti)
    ln2_pm = vec_at_token(L["ln2_pre_mlp"], ti)
    mlp_o = vec_at_token(L["mlp_out"], ti)
    mid, block_out = residual_mid_block_out(L, ti)
    if "mlp_post_norm" in L:
        attn_delta_res = ln2_pm
        mlp_delta_res = vec_at_token(L["mlp_post_norm"], ti)
    else:
        attn_delta_res = attn_o
        mlp_delta_res = mlp_o

    lines.append(f"--- Layer {li} ---")
    lines.append(
        "  ||·|| : "
        f"block_in={l2_norm(block_in):.4f}  "
        f"ln1(pre_attn)={l2_norm(ln1_pa):.4f}  "
        f"attn_delta={l2_norm(attn_o):.4f}  "
        f"mid(after_attn_res)={l2_norm(mid):.4f}  "
        f"ln2(pre_mlp)={l2_norm(ln2_pm):.4f}  "
        f"mlp_delta={l2_norm(mlp_o):.4f}  "
        f"block_out={l2_norm(block_out):.4f}"
    )
    lines.append(
        "  cos successive (same direction as residual adds): "
        f"cos(block_in, mid)={cos_pair(block_in, mid):.4f}  "
        f"cos(mid, block_out)={cos_pair(mid, block_out):.4f}  "
        f"cos(block_in, block_out)={cos_pair(block_in, block_out):.4f}"
    )
    lines.append(
        "  cos updates vs stream: "
        f"cos(block_in, attn_to_res)={cos_pair(block_in, attn_delta_res):.4f}  "
        f"cos(mid, mlp_to_res)={cos_pair(mid, mlp_delta_res):.4f}  "
        f"cos(attn_to_res, mlp_to_res)={cos_pair(attn_delta_res, mlp_delta_res):.4f}"
    )
    lines.append(
        "  cos LN path: "
        f"cos(block_in, ln1)={cos_pair(block_in, ln1_pa):.4f}  "
        f"cos(ln1, attn_delta)={cos_pair(ln1_pa, attn_o):.4f}  "
        f"cos(mid, ln2)={cos_pair(mid, ln2_pm):.4f}  "
        f"cos(ln2, mlp_delta)={cos_pair(ln2_pm, mlp_o):.4f}"
    )
    lines.append("")

report = "\n".join(lines)
print(report, flush=True)

# %% Plot data: arrays + sanity checks (run after forward + metrics; then run each plot cell below)
from stonesoup.experiment import configure_matplotlib_agg, hf_repo_id_safe_stem

configure_matplotlib_agg()
import matplotlib.pyplot as plt
import numpy as np

if state.embed is None or not state.layers:
    raise RuntimeError("Run the forward and metrics cells first so state is populated.")

seq_len_plot = int(state.embed.shape[1])
ti_plot = TOKEN_INDEX if TOKEN_INDEX >= 0 else seq_len_plot + TOKEN_INDEX
if not (0 <= ti_plot < seq_len_plot):
    raise ValueError(f"TOKEN_INDEX out of range for plot (seq_len={seq_len_plot}).")

emb_vp = vec_at_token(state.embed, ti_plot)
n_l = len(state.layers)

# --- Depth: ||embed||, then ||block_out|| after each layer (RoPE: embed matches L0 block_in)
norm_depth: list[float] = [l2_norm(emb_vp)]
for li in range(n_l):
    L = state.layers[li]
    _, block_out = residual_mid_block_out(L, ti_plot)
    norm_depth.append(l2_norm(block_out))

x_depth = np.arange(len(norm_depth))
labels_depth = ["embed"] + [f"L{i} out" for i in range(n_l)]

# --- Layer 0: norm at each submodule boundary (along one block)
L0 = state.layers[0]
b0 = vec_at_token(L0["block_in"], ti_plot)
ln1_0 = vec_at_token(L0["ln1_pre_attn"], ti_plot)
attn_0 = vec_at_token(L0["attn_out"], ti_plot)
mlp_0 = vec_at_token(L0["mlp_out"], ti_plot)
ln2_0 = vec_at_token(L0["ln2_pre_mlp"], ti_plot)
mid_0, out_0 = residual_mid_block_out(L0, ti_plot)
if "mlp_post_norm" in L0:
    mlp_post_0 = vec_at_token(L0["mlp_post_norm"], ti_plot)
    norm_l0_path = [
        l2_norm(b0),
        l2_norm(ln1_0),
        l2_norm(attn_0),
        l2_norm(ln2_0),
        l2_norm(mid_0),
        l2_norm(mlp_0),
        l2_norm(mlp_post_0),
        l2_norm(out_0),
    ]
    labels_l0 = [
        "block_in",
        "ln1",
        "attnΔ",
        "post_attn_norm",
        "mid",
        "mlpΔ",
        "post_mlp_norm",
        "block_out",
    ]
else:
    norm_l0_path = [
        l2_norm(b0),
        l2_norm(ln1_0),
        l2_norm(attn_0),
        l2_norm(mid_0),
        l2_norm(ln2_0),
        l2_norm(mlp_0),
        l2_norm(out_0),
    ]
    labels_l0 = ["block_in", "ln1", "attnΔ", "mid", "ln2", "mlpΔ", "block_out"]
x_l0 = np.arange(len(norm_l0_path))

# --- Per-layer cos sim (vs layer index)
cos_block_mid: list[float] = []
cos_mid_out: list[float] = []
cos_block_out: list[float] = []
# Submodule updates: mid = block_in + δ_attn, block_out = mid + δ_mlp (normed deltas on Gemma)
cos_attn_delta_vs_in: list[float] = []
cos_attn_delta_vs_mid: list[float] = []
cos_mlp_delta_vs_mid: list[float] = []
cos_mlp_delta_vs_out: list[float] = []
for li in range(n_l):
    L = state.layers[li]
    block_in = vec_at_token(L["block_in"], ti_plot)
    mid, block_out = residual_mid_block_out(L, ti_plot)
    cos_block_mid.append(cos_pair(block_in, mid))
    cos_mid_out.append(cos_pair(mid, block_out))
    cos_block_out.append(cos_pair(block_in, block_out))
    delta_attn = mid - block_in
    delta_mlp = block_out - mid
    na = float(delta_attn.norm().item())
    nm = float(delta_mlp.norm().item())
    if na < 1e-12:
        cos_attn_delta_vs_in.append(float("nan"))
        cos_attn_delta_vs_mid.append(float("nan"))
    else:
        cos_attn_delta_vs_in.append(cos_pair(block_in, delta_attn))
        cos_attn_delta_vs_mid.append(cos_pair(mid, delta_attn))
    if nm < 1e-12:
        cos_mlp_delta_vs_mid.append(float("nan"))
        cos_mlp_delta_vs_out.append(float("nan"))
    else:
        cos_mlp_delta_vs_mid.append(cos_pair(mid, delta_mlp))
        cos_mlp_delta_vs_out.append(cos_pair(block_out, delta_mlp))

# --- h after embed and after each layer (for multi-reference cosine plot)
h_at_depth: list[torch.Tensor] = [emb_vp]
for li in range(n_l):
    L = state.layers[li]
    _, block_out = residual_mid_block_out(L, ti_plot)
    h_at_depth.append(block_out)

# Sanity: per-layer cos plot vs depth plot must use the same residual endpoints.
# Hooked activations are often bf16; we rebuild with float() and sums — expect ~1e-3–1e-2
# elementwise noise vs float32, not ~O(1) gaps from wrong residual math.
_STREAM_ATOL = 1e-2
_STREAM_RTOL = 1e-3
_STREAM_COS_MIN = 1.0 - 1e-5
# Two cos_pair() paths on the same geometry can differ by ~1e-4 in late layers (float32).
_COS_TOL = 1e-3
for li in range(n_l):
    bi = vec_at_token(state.layers[li]["block_in"], ti_plot)
    L = state.layers[li]
    _, block_out_re = residual_mid_block_out(L, ti_plot)
    hd = h_at_depth[li]
    if cos_pair(bi, hd) < _STREAM_COS_MIN:
        raise AssertionError(
            f"Layer {li}: block_in vs h_at_depth[{li}] cosine {cos_pair(bi, hd):.8f} "
            f"(expect ~1; wrong indexing or residual wiring)."
        )
    if not torch.allclose(bi, hd, atol=_STREAM_ATOL, rtol=_STREAM_RTOL):
        md = float((bi - hd).abs().max().item())
        rel = float((torch.norm(bi - hd) / (torch.norm(bi) + 1e-8)).item())
        _prev = "embed" if li == 0 else f"L{li - 1} out"
        raise AssertionError(
            f"Layer {li}: block_in must match h_at_depth[{li}] (max abs {md:g}, rel L2 {rel:g}; "
            f"expect block_in == {_prev})."
        )
    if not torch.allclose(block_out_re, h_at_depth[li + 1], atol=_STREAM_ATOL, rtol=_STREAM_RTOL):
        md = float((block_out_re - h_at_depth[li + 1]).abs().max().item())
        raise AssertionError(
            f"Layer {li}: reconstructed block_out must match h_at_depth[{li + 1}] "
            f"(max abs diff {md:g})."
        )
    c_plot = cos_block_out[li]
    c_depth = cos_pair(h_at_depth[li], h_at_depth[li + 1])
    if abs(c_plot - c_depth) > _COS_TOL:
        raise AssertionError(
            f"Layer {li}: cos_block_out ({c_plot:.8f}) != "
            f"cos(h_at_depth[{li}], h_at_depth[{li + 1}]) ({c_depth:.8f}); "
            "per-layer plot vs multi-ref plot would disagree."
        )

x_layer = np.arange(n_l)
x_depth_embed = np.arange(n_l + 1)
labels_depth_embed = ["embed"] + [f"L{i} out" for i in range(n_l)]

# Reference directions: embed + block_out after selected layers (skip if model shallower)
REFERENCE_BLOCK_LAYERS = (0, 2, 10, 20, 25)
cos_ref_specs: list[tuple[str, torch.Tensor]] = [("embed", emb_vp)]
for li in REFERENCE_BLOCK_LAYERS:
    if li < n_l:
        cos_ref_specs.append((f"L{li} out", h_at_depth[li + 1]))

stem = hf_repo_id_safe_stem(MODEL_ID)

# %% Plot: residual norms (depth + layer 0 submodule path)
fig_n, (ax_d, ax_p) = plt.subplots(2, 1, figsize=(8, 7), constrained_layout=True)
fig_n.suptitle(
    f"{MODEL_ID}\n(single token index {ti_plot}, not mean over sequence)",
    fontsize=10,
    fontweight="medium",
)
ax_d.plot(x_depth, norm_depth, marker="o", ms=5, color="C0")
ax_d.set_xticks(x_depth)
ax_d.set_xticklabels(labels_depth, rotation=35, ha="right")
ax_d.set_ylabel("L2 norm")
ax_d.set_title("Residual stream norm: embedding → each layer block output")
ax_d.grid(True, alpha=0.3)

ax_p.plot(x_l0, norm_l0_path, marker="o", ms=5, color="C1")
ax_p.set_xticks(x_l0)
ax_p.set_xticklabels(labels_l0, rotation=30, ha="right")
ax_p.set_ylabel("L2 norm")
ax_p.set_title("Norm along submodules (layer 0 only)")
ax_p.grid(True, alpha=0.3)

stonesoup.show(fig_n, basename=f"{stem}_activation_norms", dpi=120)

# %% Plot: delta vs hidden (attention + MLP sub-steps)
fig_cd, (ax_cd_a, ax_cd_m) = plt.subplots(2, 1, figsize=(8, 7), constrained_layout=True)
fig_cd.suptitle(
    f"{MODEL_ID}\n(single token index {ti_plot}, not mean over sequence)",
    fontsize=10,
    fontweight="medium",
)
ax_cd_a.plot(
    x_layer,
    cos_attn_delta_vs_in,
    marker="o",
    color="C0",
    label=r"cos(block_in, $\delta_{\mathrm{attn}}$)",
)
ax_cd_a.plot(
    x_layer,
    cos_attn_delta_vs_mid,
    marker="s",
    color="C1",
    label=r"cos(mid, $\delta_{\mathrm{attn}}$)",
)
ax_cd_a.plot(
    x_layer,
    cos_block_mid,
    marker="^",
    color="C2",
    label=r"cos(block_in, mid)",
)
ax_cd_a.set_ylabel("cosine similarity")
ax_cd_a.set_title(
    r"Attention sub-step: mid $=$ block_in $+\,\delta_{\mathrm{attn}}$ "
    r"($\delta_{\mathrm{attn}}$ is normed attn residual)"
)
ax_cd_a.set_xticks(x_layer)
ax_cd_a.legend(loc="best")
ax_cd_a.grid(True, alpha=0.3)
ax_cd_a.set_ylim(-1.05, 1.05)
ax_cd_a.axhline(0.0, color="gray", lw=0.6, alpha=0.4)

ax_cd_m.plot(
    x_layer,
    cos_mlp_delta_vs_mid,
    marker="o",
    color="C0",
    label=r"cos(mid, $\delta_{\mathrm{mlp}}$)",
)
ax_cd_m.plot(
    x_layer,
    cos_mlp_delta_vs_out,
    marker="s",
    color="C1",
    label=r"cos(block_out, $\delta_{\mathrm{mlp}}$)",
)
ax_cd_m.plot(
    x_layer,
    cos_mid_out,
    marker="^",
    color="C2",
    label=r"cos(mid, block_out)",
)
ax_cd_m.set_xlabel("Layer")
ax_cd_m.set_ylabel("cosine similarity")
ax_cd_m.set_title(
    r"MLP sub-step: block_out $=$ mid $+\,\delta_{\mathrm{mlp}}$ "
    r"($\delta_{\mathrm{mlp}}$ is normed MLP residual)"
)
ax_cd_m.set_xticks(x_layer)
ax_cd_m.legend(loc="best")
ax_cd_m.grid(True, alpha=0.3)
ax_cd_m.set_ylim(-1.05, 1.05)
ax_cd_m.axhline(0.0, color="gray", lw=0.6, alpha=0.4)

stonesoup.show(fig_cd, basename=f"{stem}_activation_cos_delta_submodules", dpi=120)

# %% Norm comparison: relative to ||h_in|| (= ||block_in||) per layer
rel_h_mid: list[float] = []
rel_h_out: list[float] = []
rel_delta_attn: list[float] = []
rel_delta_mlp: list[float] = []
for li in range(n_l):
    L = state.layers[li]
    block_in_v = vec_at_token(L["block_in"], ti_plot)
    mid_v, block_out_v = residual_mid_block_out(L, ti_plot)
    n_in = l2_norm(block_in_v)
    if n_in < 1e-12:
        rel_h_mid.append(float("nan"))
        rel_h_out.append(float("nan"))
        rel_delta_attn.append(float("nan"))
        rel_delta_mlp.append(float("nan"))
    else:
        rel_h_mid.append(l2_norm(mid_v) / n_in)
        rel_h_out.append(l2_norm(block_out_v) / n_in)
        rel_delta_attn.append(l2_norm(mid_v - block_in_v) / n_in)
        rel_delta_mlp.append(l2_norm(block_out_v - mid_v) / n_in)

fig_ncomp, (ax_nh, ax_nd) = plt.subplots(2, 1, figsize=(8, 7), constrained_layout=True)
fig_ncomp.suptitle(
    f"{MODEL_ID}\n(single token index {ti_plot}, not mean over sequence)",
    fontsize=10,
    fontweight="medium",
)
ax_nh.axhline(1.0, color="C0", ls="--", lw=1.2, label=r"$\Vert h_{\mathrm{in}}\Vert$ ref (=1)")
# ax_nh.plot(
#     x_layer,
#     rel_h_mid,
#     marker="s",
#     ms=5,
#     color="C1",
#     label=r"$\Vert h_{\mathrm{mid}}\Vert \,/\, \Vert h_{\mathrm{in}}\Vert$",
# )
ax_nh.plot(
    x_layer,
    rel_h_out,
    marker="^",
    ms=5,
    color="C2",
    label=r"$\Vert h_{\mathrm{out}}\Vert \,/\, \Vert h_{\mathrm{in}}\Vert$",
)
ax_nh.set_ylabel("relative L2 norm")
ax_nh.set_title(r"Hidden states vs $\Vert h_{\mathrm{in}}\Vert$ at this layer (block_in)")
ax_nh.set_xticks(x_layer)
ax_nh.legend(loc="best", fontsize=8)
ax_nh.grid(True, alpha=0.3)

ax_nd.plot(
    x_layer,
    rel_delta_attn,
    marker="o",
    ms=5,
    color="C0",
    label=r"$\Vert\delta_{\mathrm{attn}}\Vert \,/\, \Vert h_{\mathrm{in}}\Vert$",
)
ax_nd.plot(
    x_layer,
    rel_delta_mlp,
    marker="s",
    ms=5,
    color="C1",
    label=r"$\Vert\delta_{\mathrm{mlp}}\Vert \,/\, \Vert h_{\mathrm{in}}\Vert$",
)
ax_nd.set_xlabel("Layer")
ax_nd.set_ylabel("relative L2 norm")
ax_nd.set_title(r"Submodule $\delta$ size vs $\Vert h_{\mathrm{in}}\Vert$ (same layer)")
ax_nd.set_xticks(x_layer)
ax_nd.set_ylim(0.0, 1.05)
ax_nd.legend(loc="best", fontsize=8)
ax_nd.grid(True, alpha=0.3)

stonesoup.show(fig_ncomp, basename=f"{stem}_activation_norm_rel_hin", dpi=120)

# %% Plot: cos(h at depth, reference directions)
fig_e, ax_e = plt.subplots(figsize=(10, 5), constrained_layout=True)
fig_e.suptitle(
    f"{MODEL_ID}\n(single token index {ti_plot}, not mean over sequence)",
    fontsize=10,
    fontweight="medium",
)
for idx, (ref_name, ref_vec) in enumerate(cos_ref_specs):
    y_ref = [cos_pair(h_at_depth[i], ref_vec) for i in range(len(h_at_depth))]
    ax_e.plot(
        x_depth_embed,
        y_ref,
        marker="o",
        ms=4,
        label=f"vs {ref_name}",
        color=f"C{idx % 10}",
    )
ax_e.set_xticks(x_depth_embed)
ax_e.set_xticklabels(labels_depth_embed, rotation=35, ha="right")
ax_e.set_xlabel("Depth h moves along stack (embed → L0 out → …)")
ax_e.set_ylabel("cosine similarity")
ax_e.set_title(
    "cos(h at depth, reference direction): each line fixes one ref vector; "
    "x is current h along depth"
)
ax_e.set_ylim(0.0, 1.05)
ax_e.grid(True, alpha=0.3)
ax_e.axhline(0.0, color="gray", lw=0.6, alpha=0.4)
ax_e.legend(loc="best", fontsize=8, ncol=2)

stonesoup.show(fig_e, basename=f"{stem}_activation_cos_to_embed", dpi=120)
