# %% Imports & load model
from __future__ import annotations

import hashlib

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from matplotlib.patches import Patch

import stonesoup
from stonesoup.experiment import (
    configure_matplotlib_agg,
    decoder_blocks,
    hf_repo_id_safe_stem,
    inner_tokenizer,
)

configure_matplotlib_agg()

MODEL_ID = "Qwen/Qwen3-8B"
PROMPT = "Name the birthday of the Queen of Denmark."
MAX_NEW_TOKENS = 48
KL_CAP = 2.0
# For post–last-block plot: `1 - cos(h_ablated, h_baseline)` at last pos (same as KL: larger = more effect).
COS_DIV_CAP = 0.2
# Ablations grid: at most this many response-token “cuts” (columns) per row of subplots.
PLOT_TOKENS_PER_ROW = 12
# Bar colors: KL plot + KL half of combined; 1-cos plot + 1-cos half (attn / mlp).
_C_KL_ATTN, _C_KL_MLP = "#1B9E77", "#D95F02"
_C_COS_ATTN, _C_COS_MLP = "#7570B3", "#E7298A"
# Used in the inspect cell when CELL_INPUT is not set to "cut,layer".
INSPECT_CUT = 0
INSPECT_LAYER = 0

model, proc = stonesoup.load_model(MODEL_ID)
model.eval()
tokenizer = inner_tokenizer(proc)
device = next(model.parameters()).device

blocks = decoder_blocks(model)
num_layers = len(blocks)
safe = hf_repo_id_safe_stem(MODEL_ID)
print(f"num_layers: {num_layers}")

# Scale self_attn / mlp outputs to ablate each submodule (Llama-style residuals).

attn_scales = [1.0] * num_layers
mlp_scales = [1.0] * num_layers


def make_scale_hook(scales: list[float], i: int):
    def hook(_m, _in, out):
        s = scales[i]
        if isinstance(out, tuple):
            return (out[0] * s,) + out[1:]
        return out * s

    return hook


attn_handles = [
    b.self_attn.register_forward_hook(make_scale_hook(attn_scales, i))
    for i, b in enumerate(blocks)
]
mlp_handles = [
    b.mlp.register_forward_hook(make_scale_hook(mlp_scales, i))
    for i, b in enumerate(blocks)
]
print("hooks: self_attn + mlp per block")


def logits_for_prefix(pref: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return model(pref).logits[0, -1].detach().float().cpu()


def logits_and_h_post_last(
    pref: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Last-position lm logits and last-position hidden **out of final decoder block** (scales=hooks)."""
    cap: list[torch.Tensor | None] = [None]
    b_last = blocks[num_layers - 1]

    def _post(_m, _in, o):
        t = o[0] if isinstance(o, tuple) else o
        cap[0] = t[:, -1, :].detach().float()

    h = b_last.register_forward_hook(_post)
    try:
        with torch.no_grad():
            out = model(pref)
    finally:
        h.remove()
    if cap[0] is None:
        raise RuntimeError("h_post_last hook did not run")
    return out.logits[0, -1].detach().float().cpu(), cap[0].detach().float().cpu()


def _short_alt_label(tid: int) -> str:
    s = tokenizer.decode([tid]) if tid else ""
    s = s.replace("\n", "↵").replace("\r", "")
    if len(s) > 12:
        s = s[:11] + "…"
    return s


def _ablation_paged_figure(
    a_mat: np.ndarray,
    m_mat: np.ndarray,
    targets: list[int],
    title: str,
    basename: str,
    *,
    pred_attn: np.ndarray,
    pred_mlp: np.ndarray,
    cap: float,
    xlabel: str,
    ylabel: str,
    bar_attn: str,
    bar_mlp: str,
) -> None:
    """Shared KL / 1-cos ablation bar grid; multiple subplot rows, ≤ ``PLOT_TOKENS_PER_ROW`` cuts per row."""
    n_tok = a_mat.shape[0]
    bar_half = 0.45
    bar_h = 0.55
    cap_f = float(cap) if cap > 0 else 0.0
    scale = bar_half / cap_f if cap_f > 0 else 0.0
    ann_fs = max(2.0, min(3.6, 8.0 - 0.1 * num_layers))
    spine_text_pad = 0.02
    layers_y = np.arange(num_layers)
    per = int(PLOT_TOKENS_PER_ROW)
    n_row = max(1, (n_tok + per - 1) // per)
    row_w = max(8.0, 0.9 * n_tok) if n_row == 1 else max(8.0, 0.9 * per)
    # ~Half the previous per-row height; width unchanged.
    row_h = max(2.5, 0.11 * num_layers)
    fig, axes = plt.subplots(
        n_row,
        1,
        figsize=(row_w, row_h * n_row),
        sharey=True,
        sharex=False,
    )
    if n_row == 1:
        ax_list: list = [axes]  # type: ignore[list-item]
    else:
        ax_list = list(axes)  # type: ignore[assignment]

    for row, ax in enumerate(ax_list):
        c0 = row * per
        c1 = min(c0 + per, n_tok)
        n_this = c1 - c0
        for k, cut in enumerate(range(c0, c1)):
            a = np.minimum(np.nan_to_num(a_mat[cut], nan=0.0), cap_f) * scale
            _mrow = np.asarray(m_mat[cut], dtype=np.float64)
            m = np.minimum(np.where(np.isfinite(_mrow), _mrow, 0.0), cap_f) * scale
            ax.barh(layers_y, -a, left=k, height=bar_h, color=bar_attn, edgecolor="none")
            ax.barh(layers_y, m, left=k, height=bar_h, color=bar_mlp, edgecolor="none")
            ax.axvline(k, color="#cccccc", linewidth=0.4, zorder=0)
            gt = int(targets[cut])
            x_left = float(k) - spine_text_pad
            x_right = float(k) + spine_text_pad
            for j in range(num_layers):
                yj = float(j)
                pid_a = int(pred_attn[cut, j])
                pid_m = int(pred_mlp[cut, j])
                if pid_a != gt:
                    ax.text(
                        x_left,
                        yj,
                        _short_alt_label(pid_a),
                        ha="right",
                        va="center",
                        fontsize=ann_fs,
                        color="#0a2540",
                        clip_on=False,
                        zorder=3,
                    )
                if pid_m != gt:
                    ax.text(
                        x_right,
                        yj,
                        _short_alt_label(pid_m),
                        ha="left",
                        va="center",
                        fontsize=ann_fs,
                        color="#4a2a00",
                        clip_on=False,
                        zorder=3,
                    )
        ax.set_xticks(np.arange(n_this))
        ax.set_xticklabels(
            [f"[{c}] {tokenizer.decode([targets[c]])!r}" for c in range(c0, c1)],
            rotation=-30,
            ha="left",
            fontsize=8,
        )
        ax.set_yticks(layers_y)
        ax.set_yticklabels([f"L{i}" for i in layers_y], fontsize=6)
        ax.set_xlim(-0.6, n_this - 0.4)
        ax.set_ylim(num_layers - 0.25, -0.6)
        ax.grid(alpha=0.25, axis="y", linestyle=":")

    ax_list[0].set_ylabel(ylabel)
    ax_list[-1].set_xlabel(xlabel)
    ax_list[0].legend(
        handles=[
            Patch(color=bar_attn, label="attn (left)"),
            Patch(color=bar_mlp, label="mlp (right)"),
        ],
        loc="upper right",
    )
    fig.suptitle(
        f"{title}  (≤{per} positions/row, {n_row} row(s))",
        fontsize=10,
        y=1.0,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.99), h_pad=0.5)
    stonesoup.show(fig, basename=basename, dpi=140)
    plt.close(fig)


def plot_kl_centered(
    kl_attn: np.ndarray,
    kl_mlp: np.ndarray,
    targets: list[int],
    title: str,
    basename: str,
    *,
    pred_attn: np.ndarray,
    pred_mlp: np.ndarray,
) -> None:
    _ablation_paged_figure(
        kl_attn,
        kl_mlp,
        targets,
        title,
        basename,
        pred_attn=pred_attn,
        pred_mlp=pred_mlp,
        cap=KL_CAP,
        xlabel=(
            f"token position in row (← attn | mlp →, bar ∝ KL, cap={KL_CAP}; "
            f"text when ablated top-1 ≠ ground truth)"
        ),
        ylabel="layer",
        bar_attn=_C_KL_ATTN,
        bar_mlp=_C_KL_MLP,
    )


def plot_cos_diverge_centered(
    div_attn: np.ndarray,
    div_mlp: np.ndarray,
    targets: list[int],
    title: str,
    basename: str,
    *,
    pred_attn: np.ndarray,
    pred_mlp: np.ndarray,
) -> None:
    """Left/right bar grid like ``plot_kl_centered`` but for ``1 - cos`` to unablated post–last h."""
    _ablation_paged_figure(
        div_attn,
        div_mlp,
        targets,
        title,
        basename,
        pred_attn=pred_attn,
        pred_mlp=pred_mlp,
        cap=COS_DIV_CAP,
        xlabel=(
            f"token position in row (← attn | mlp →, bar ∝ 1−cos, cap={COS_DIV_CAP}; "
            f"post–L{num_layers - 1} h vs unablated; text when ablated top-1 ≠ ground truth)"
        ),
        ylabel="layer ablated to 0",
        bar_attn=_C_COS_ATTN,
        bar_mlp=_C_COS_MLP,
    )


def plot_kl_and_cos_paged(
    kl_attn: np.ndarray,
    kl_mlp: np.ndarray,
    cosdiv_attn: np.ndarray,
    cosdiv_mlp: np.ndarray,
    targets: list[int],
    title: str,
    basename: str,
) -> None:
    """Same page layout as the separate KL/cos figures: four sub-rows per layer (KL pair, then cos pair)."""
    n_tok = kl_attn.shape[0]
    bar_half = 0.45
    cap_kl = float(KL_CAP) if KL_CAP > 0 else 0.0
    cap_cos = float(COS_DIV_CAP) if COS_DIV_CAP > 0 else 0.0
    scale_kl = bar_half / cap_kl if cap_kl > 0 else 0.0
    scale_cos = bar_half / cap_cos if cap_cos > 0 else 0.0
    per = int(PLOT_TOKENS_PER_ROW)
    n_row = max(1, (n_tok + per - 1) // per)
    row_w = max(8.0, 0.9 * n_tok) if n_row == 1 else max(8.0, 0.9 * per)
    # Slightly taller per row so four sub-bars per layer are readable.
    row_h = max(2.8, 0.12 * num_layers)
    bh = 0.14
    y_off = (0.08, 0.28, 0.48, 0.68)  # KL attn, KL mlp, cos attn, cos mlp

    fig, axes = plt.subplots(
        n_row,
        1,
        figsize=(row_w, row_h * n_row),
        sharey=True,
        sharex=False,
    )
    if n_row == 1:
        ax_list: list = [axes]  # type: ignore[list-item]
    else:
        ax_list = list(axes)  # type: ignore[assignment]
    layers_y = np.arange(num_layers)

    for row, ax in enumerate(ax_list):
        c0 = row * per
        c1 = min(c0 + per, n_tok)
        n_this = c1 - c0
        for k, cut in enumerate(range(c0, c1)):
            ak = np.minimum(np.nan_to_num(kl_attn[cut], nan=0.0), cap_kl) * scale_kl
            mk = np.minimum(np.nan_to_num(kl_mlp[cut], nan=0.0), cap_kl) * scale_kl
            ac = (
                np.minimum(np.nan_to_num(cosdiv_attn[cut], nan=0.0), cap_cos)
                * scale_cos
            )
            mco = (
                np.minimum(np.nan_to_num(cosdiv_mlp[cut], nan=0.0), cap_cos)
                * scale_cos
            )
            ax.axvline(k, color="#cccccc", linewidth=0.4, zorder=0)
            for j in range(num_layers):
                y1 = float(j) + y_off[0]
                y2 = float(j) + y_off[1]
                y3 = float(j) + y_off[2]
                y4 = float(j) + y_off[3]
                ax.barh(
                    y1, -ak[j], left=float(k), height=bh, color=_C_KL_ATTN, edgecolor="none", zorder=1
                )
                ax.barh(
                    y2, mk[j], left=float(k), height=bh, color=_C_KL_MLP, edgecolor="none", zorder=1
                )
                ax.barh(
                    y3, -ac[j], left=float(k), height=bh, color=_C_COS_ATTN, edgecolor="none", zorder=2
                )
                ax.barh(
                    y4, mco[j], left=float(k), height=bh, color=_C_COS_MLP, edgecolor="none", zorder=2
                )
        ax.set_xticks(np.arange(n_this))
        ax.set_xticklabels(
            [f"[{c}] {tokenizer.decode([targets[c]])!r}" for c in range(c0, c1)],
            rotation=-30,
            ha="left",
            fontsize=8,
        )
        ax.set_yticks(layers_y)
        ax.set_yticklabels([f"L{i}" for i in layers_y], fontsize=6)
        ax.set_xlim(-0.6, n_this - 0.4)
        ax.set_ylim(num_layers - 0.1, -0.45)
        ax.grid(alpha=0.25, axis="y", linestyle=":")

    ax_list[-1].set_xlabel(
        f"token in row: per layer 4 bars — "
        f"KL attn|mlp (←/→, cap {KL_CAP}) then 1−cos attn|mlp (cap {COS_DIV_CAP}); same palette as separate plots"
    )
    ax_list[0].set_ylabel("layer (four sub-bars: KL then cos)")
    ax_list[0].legend(
        handles=[
            Patch(color=_C_KL_ATTN, label="KL attn"),
            Patch(color=_C_KL_MLP, label="KL mlp"),
            Patch(color=_C_COS_ATTN, label="1-cos attn"),
            Patch(color=_C_COS_MLP, label="1-cos mlp"),
        ],
        loc="upper right",
        fontsize=7,
        ncol=2,
    )
    fig.suptitle(
        f"{title}  (≤{per} pos/row, {n_row} row(s))",
        fontsize=10,
        y=1.0,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.99), h_pad=0.5)
    stonesoup.show(fig, basename=basename, dpi=140)
    plt.close(fig)


# %% Generate → sweep → plot
# When this cell is run alone, the first cell’s ``tokenizer`` binding is missing — rebuild from ``proc``.
from stonesoup.experiment import inner_tokenizer

tokenizer = inner_tokenizer(proc)
# Stonesoup: set CELL_INPUT to override PROMPT for this cell.
prompt_text = (str(globals().get("CELL_INPUT", "") or "").strip()) or PROMPT
print(f"prompt: {prompt_text!r}")

messages = [{"role": "user", "content": prompt_text}]
try:
    chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
except Exception:
    chat_text = prompt_text
prompt_ids = tokenizer(chat_text, return_tensors="pt").input_ids.to(device)

with torch.no_grad():
    gen = model.generate(
        prompt_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
response_ids = gen[0, prompt_ids.shape[1] :]
print("=== response ===")
print(tokenizer.decode(response_ids, skip_special_tokens=True))

n_tokens = int(response_ids.shape[0])
kl_attn_all = np.zeros((n_tokens, num_layers))
kl_mlp_all = np.zeros((n_tokens, num_layers), dtype=np.float64)
cosdiv_attn_all = np.zeros((n_tokens, num_layers))
cosdiv_mlp_all = np.zeros((n_tokens, num_layers), dtype=np.float64)
pred_attn_all = np.zeros((n_tokens, num_layers), dtype=np.int64)
pred_mlp_all = np.zeros((n_tokens, num_layers), dtype=np.int64)
targets: list[int] = []

for cut in tqdm(
    range(n_tokens),
    desc="e2e ablation (response tokens)",
    unit="pos",
    leave=True,
):
    stonesoup.check_abort()
    pref = torch.cat([prompt_ids[0], response_ids[:cut]]).unsqueeze(0)
    targets.append(int(response_ids[cut].item()))

    attn_scales[:] = [1.0] * num_layers
    mlp_scales[:] = [1.0] * num_layers
    _raw0, h_base_2d = logits_and_h_post_last(pref)
    log_p_cut = F.log_softmax(_raw0, dim=-1)

    for i in range(num_layers):
        attn_scales[:] = [1.0] * num_layers
        mlp_scales[:] = [1.0] * num_layers
        attn_scales[i] = 0.0
        raw_q, h_q = logits_and_h_post_last(pref)
        log_q = F.log_softmax(raw_q, dim=-1)
        kl_attn_all[cut, i] = F.kl_div(log_q, log_p_cut, reduction="sum", log_target=True).item()
        cosdiv_attn_all[cut, i] = 1.0 - F.cosine_similarity(
            h_base_2d, h_q, dim=1, eps=1e-8
        ).item()
        pred_attn_all[cut, i] = int(raw_q.argmax().item())

    for i in range(num_layers):
        attn_scales[:] = [1.0] * num_layers
        mlp_scales[:] = [1.0] * num_layers
        mlp_scales[i] = 0.0
        raw_q, h_q = logits_and_h_post_last(pref)
        log_q = F.log_softmax(raw_q, dim=-1)
        kl_mlp_all[cut, i] = F.kl_div(log_q, log_p_cut, reduction="sum", log_target=True).item()
        cosdiv_mlp_all[cut, i] = 1.0 - F.cosine_similarity(
            h_base_2d, h_q, dim=1, eps=1e-8
        ).item()
        pred_mlp_all[cut, i] = int(raw_q.argmax().item())

    tqdm.write(
        f"cut={cut:3d} target={tokenizer.decode([targets[cut]])!r}  "
        f"max KL_attn={kl_attn_all[cut].max():.3f}  max KL_mlp={kl_mlp_all[cut].max():.3f}  "
        f"max 1-cos_attn={cosdiv_attn_all[cut].max():.3f}  max 1-cos_mlp={cosdiv_mlp_all[cut].max():.3f}"
    )

attn_scales[:] = [1.0] * num_layers
mlp_scales[:] = [1.0] * num_layers

# %% Plot

prompt_tag = hashlib.md5(prompt_text.encode()).hexdigest()[:8]
plot_kl_centered(
    kl_attn_all,
    kl_mlp_all,
    targets,
    title=f"{MODEL_ID} — Ablation to all submodules, measured by KL",
    basename=f"{safe}_ablation_kl_e2e_{prompt_tag}",
    pred_attn=pred_attn_all,
    pred_mlp=pred_mlp_all,
)

# %% Plot: post–last block hidden (1 − cos) vs unablated (same layout as KL)
plot_cos_diverge_centered(
    cosdiv_attn_all,
    cosdiv_mlp_all,
    targets,
    title=f"{MODEL_ID} — Ablation to all submodules, measured by 1−cos to unablated post–L{num_layers - 1} h",
    basename=f"{safe}_ablation_hpost_cosdiv_e2e_{prompt_tag}",
    pred_attn=pred_attn_all,
    pred_mlp=pred_mlp_all,
)

# %% Plot: KL + 1-cos combined (four bars per layer; darker = KL, brighter = 1-cos to unablated h)
plot_kl_and_cos_paged(
    kl_attn_all,
    kl_mlp_all,
    cosdiv_attn_all,
    cosdiv_mlp_all,
    targets,
    title=f"{MODEL_ID} — {prompt_text!r}  logit KL + post–L{num_layers - 1} 1-cos  (all submodules)",
    basename=f"{safe}_ablation_kl_and_cos_e2e_{prompt_tag}",
)

# %% Inspect attention at (cut, layer)
stonesoup.html()
from stonesoup.experiment import inner_tokenizer

tokenizer = inner_tokenizer(proc)
# Stonesoup: CELL_INPUT as "cut,layer" (e.g. "3,12"). Otherwise INSPECT_CUT / INSPECT_LAYER.
_parts = (
    str(globals().get("CELL_INPUT", "") or "").strip()
    or f"{INSPECT_CUT},{INSPECT_LAYER}"
).split(",")
inspect_cut, inspect_layer = int(_parts[0].strip()), int(_parts[1].strip())

attn_scales[:] = [1.0] * num_layers
mlp_scales[:] = [1.0] * num_layers

pref = torch.cat([prompt_ids[0], response_ids[:inspect_cut]]).unsqueeze(0)
target = int(response_ids[inspect_cut].item())
print(f"cut={inspect_cut}  target={tokenizer.decode([target])!r}  layer=L{inspect_layer}")

captured_attn: dict[int, torch.Tensor] = {}


def make_attn_capture_hook(i: int):
    def hook(_mod, _in, output):
        if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
            captured_attn[i] = output[1].detach().float().cpu()

    return hook


capture_handles = [
    b.self_attn.register_forward_hook(make_attn_capture_hook(i))
    for i, b in enumerate(blocks)
]

submodule_configs: list[object] = []
for b in blocks:
    if hasattr(b.self_attn, "config"):
        submodule_configs.append(b.self_attn.config)
orig_impls = [
    model.config._attn_implementation,
    *[c._attn_implementation for c in submodule_configs],
]
try:
    model.config._attn_implementation = "eager"
    for c in submodule_configs:
        c._attn_implementation = "eager"
    with torch.no_grad():
        model(pref, output_attentions=True)
finally:
    model.config._attn_implementation = orig_impls[0]
    for c, impl in zip(submodule_configs, orig_impls[1:]):
        c._attn_implementation = impl
    for h in capture_handles:
        h.remove()

attn_layer = captured_attn.get(inspect_layer)

if attn_layer is None:
    print(
        f"could not capture attention weights for L{inspect_layer} — "
        f"attention impl ({getattr(model.config, '_attn_implementation', '?')}) "
        f"did not expose them."
    )
else:
    attn_mat = attn_layer[0].numpy()
    num_heads, seq_len, _ = attn_mat.shape
    q_pos = seq_len - 1
    avg_attn = attn_mat[:, q_pos, :].mean(axis=0)

    token_ids = pref[0].tolist()
    token_strs = [tokenizer.decode([t]) for t in token_ids]

    topk = 15
    top_idx = np.argsort(avg_attn)[::-1][:topk]
    print(f"num_heads={num_heads}  seq_len={seq_len}  query_pos={q_pos}")
    print(f"top-{topk} attended source tokens (head-averaged) from q_pos={q_pos}:")
    for i in top_idx:
        print(f"  src[{i:4d}]  w={avg_attn[i]:.4f}  {token_strs[i]!r}")

    fig, ax = plt.subplots(figsize=(max(8.0, 0.22 * seq_len), 3.8))
    ax.bar(np.arange(seq_len), avg_attn, color="#1f77b4", edgecolor="none")
    for i in top_idx[:8]:
        ax.annotate(
            token_strs[i],
            (i, avg_attn[i]),
            fontsize=6,
            ha="center",
            va="bottom",
            rotation=-35,
        )
    ax.set_xlabel("source token position (query = last)")
    ax.set_ylabel("avg attention weight (over heads)")
    ax.set_title(
        f"{MODEL_ID} — L{inspect_layer} attention at cut={inspect_cut}  "
        f"target={tokenizer.decode([target])!r}"
    )
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    stonesoup.show(
        fig,
        basename=f"{safe}_attn_inspect_cut{inspect_cut}_L{inspect_layer}",
        dpi=140,
    )
    plt.close(fig)

# %% Remove scale hooks
for h in attn_handles + mlp_handles:
    h.remove()
print("hooks removed")
