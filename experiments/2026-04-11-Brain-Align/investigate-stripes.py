# %% Imports & helpers
from __future__ import annotations

import math
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import stonesoup
import torch
import torch.nn.functional as F
from ckatorch import cka_base
from stonesoup.experiment import (
    apply_matplotlib_fonts_to_figure,
    capture_embed_and_post_blocks,
    configure_matplotlib_agg,
    configure_matplotlib_unicode_fonts,
    ensure_pad_token_via_eos,
    hf_repo_id_safe_stem,
    inner_tokenizer,
)

configure_matplotlib_agg()
configure_matplotlib_unicode_fonts()


def _assert_all_finite(t: torch.Tensor, *, what: str) -> None:
    assert bool(torch.isfinite(t).all().item()), f"non-finite values in {what}"


def forward_tok_stage_dim(
    model: Any,
    proc: Any,
    device: torch.device,
    prompt: str,
    *,
    max_length: int,
) -> tuple[torch.Tensor, list[str], list[str], int]:
    """Return ``(tok_stage_dim, stage_names, token_labels, seq_len)``.

    ``tok_stage_dim`` is ``(seq_len, n_stages, dim)`` (valid tokens only, no pad).
    ``token_labels`` are short strings for axis ticks: per-id ``batch_decode`` so CJK renders as
    real Unicode (``convert_ids_to_tokens`` is often byte-level / unreadable for Han).
    """
    tok = inner_tokenizer(proc)
    ensure_pad_token_via_eos(tok)
    enc = tok(
        prompt,
        return_tensors="pt",
        return_attention_mask=True,
        add_special_tokens=True,
        max_length=max_length,
        truncation=True,
    )
    inputs = {k: v.to(device) for k, v in enc.items()}
    stack, stage_names = capture_embed_and_post_blocks(model, inputs, use_cache=False)
    seq_len = int(inputs["attention_mask"][0].sum().item())
    sl = int(min(seq_len, stack.shape[2]))
    st = stack[:, 0, :sl, :].detach().float()
    tok_stage_dim = st.permute(1, 0, 2).contiguous()
    ids = enc["input_ids"][0, :sl].tolist()
    # Per-token decode gives proper Unicode for plots; convert_ids_to_tokens is not display-safe for CJK.
    decoded = tok.batch_decode([[tid] for tid in ids], skip_special_tokens=False)
    labels = [_short_tok_label(s) for s in decoded]
    _assert_all_finite(tok_stage_dim, what="forward_tok_stage_dim")
    return tok_stage_dim, stage_names, labels, sl


def _short_tok_label(s: str, *, max_len: int = 12) -> str:
    t = s.replace("Ġ", " ").replace("▁", " ")
    if len(t) > max_len:
        return t[: max_len - 1] + "…"
    return t


def stage_subplot_title(li: int, stage_names: list[str]) -> str:
    """Label for one forward stage: ``li`` matches CKA matrix row/column index (0 = embedding)."""
    name = stage_names[li] if li < len(stage_names) else f"missing_{li}"
    # return f"stage {li}: {name}"
    return f"stage {li}"


def layer_layer_linear_cka(
    tok_stage_dim: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Unbiased linear CKA between stages; same token rows as columns of ``tok_stage_dim``."""
    n_stages = int(tok_stage_dim.shape[1])
    mat = torch.eye(n_stages, dtype=torch.float64, device=device)
    A = tok_stage_dim.detach().to(dtype=torch.float64)
    A = F.normalize(A, dim=-1, eps=1e-8) # make cos sim as CKA kernel
    for i in range(n_stages):
        stonesoup.check_abort()
        for j in range(i + 1, n_stages):
            v = cka_base(A[:, i, :], A[:, j, :], unbiased=True)
            assert bool(torch.isfinite(v).item())
            mat[i, j] = v
            mat[j, i] = v
    _assert_all_finite(mat, what="layer_layer_linear_cka")
    return mat


def token_token_cosine_matrix(h: torch.Tensor) -> torch.Tensor:
    """Cosine similarity ``(n_tok, n_tok)`` for hidden states ``h`` of shape ``(n_tok, dim)``."""
    u = F.normalize(h.float(), dim=-1, eps=1e-8)
    return u @ u.T


def token_token_dot_matrix(h: torch.Tensor) -> torch.Tensor:
    """Raw dot product ``(n_tok, n_tok)`` for ``h`` of shape ``(n_tok, dim)`` (not length-normalized)."""
    x = h.float()
    return x @ x.T


TokenPairMetric = Literal["cosine", "dot"]


def token_token_pair_matrix(h: torch.Tensor, *, metric: TokenPairMetric) -> torch.Tensor:
    if metric == "cosine":
        return token_token_cosine_matrix(h)
    if metric == "dot":
        return token_token_dot_matrix(h)
    raise ValueError(f"metric must be 'cosine' or 'dot', got {metric!r}")


def mean_offdiag_pairwise(h: torch.Tensor, *, metric: TokenPairMetric) -> float:
    """Mean token×token score over distinct pairs (upper triangle, excludes diagonal)."""
    sim = token_token_pair_matrix(h, metric=metric).detach().cpu().numpy()
    n = int(sim.shape[0])
    if n < 2:
        return float("nan")
    ii, jj = np.triu_indices(n, k=1)
    return float(sim[ii, jj].mean())


def plot_mean_pairwise_metric_by_stage(
    tok_stage_dim: torch.Tensor,
    *,
    basename: str,
    title: str,
    metric: TokenPairMetric,
) -> None:
    """Line plot: x = stage index, y = mean off-diagonal pairwise score for that stage."""
    n_stages = int(tok_stage_dim.shape[1])
    means: list[float] = []
    for li in range(n_stages):
        stonesoup.check_abort()
        h = tok_stage_dim[:, li, :]
        means.append(mean_offdiag_pairwise(h, metric=metric))
    x = np.arange(n_stages, dtype=np.float64)
    fig_w = min(16.0, 5.0 + n_stages * 0.22)
    fig, ax = plt.subplots(figsize=(fig_w, 4.5))
    fs = 11.0
    ax.plot(x, means, marker="o", ms=4, lw=1.25, color="0.2")
    ax.set_xlabel("stage index (= CKA matrix index; 0 = embedding)", fontsize=fs)
    if metric == "cosine":
        ax.set_ylabel("mean pairwise cosine (off-diagonal)", fontsize=fs)
        ax.set_ylim(-1.05, 1.05)
    else:
        ax.set_ylabel("mean pairwise dot product (off-diagonal)", fontsize=fs)
        lo, hi = min(means), max(means)
        pad = 0.05 * (hi - lo + 1e-9)
        ax.set_ylim(lo - pad, hi + pad)
    ax.set_title(title, fontsize=fs + 1)
    step = max(1, n_stages // 24)
    ax.set_xticks(np.arange(0, n_stages, step))
    ax.axhline(0.0, color="0.75", lw=0.8, ls="--")
    ax.grid(True, alpha=0.35)
    ax.tick_params(labelsize=fs - 1)
    apply_matplotlib_fonts_to_figure(fig)
    fig.tight_layout()
    stonesoup.show(fig, basename=basename, dpi=144)


def plot_layer_layer_heatmap(
    mat: torch.Tensor,
    *,
    title: str,
    basename: str,
    vmin: float,
    vmax: float,
) -> None:
    arr = mat.detach().cpu().numpy()
    n = arr.shape[0]
    fig_w = min(22, 5 + n * 0.09)
    fig_h = min(20, 4 + n * 0.09)
    fs = float(max(9.0, min(16.0, 4.0 + fig_w * 0.35)))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(
        arr,
        vmin=vmin,
        vmax=vmax,
        cmap="Blues",
        aspect="equal",
        interpolation="nearest",
    )
    ax.set_title(title, fontsize=fs + 2)
    _cka_ax = "CKA matrix row / column index (same as token-plot stage; 0 = embedding)"
    ax.set_xlabel(_cka_ax, fontsize=fs)
    ax.set_ylabel(_cka_ax, fontsize=fs)
    tick_step = max(1, n // 24)
    ticks = np.arange(0, n, tick_step)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.tick_params(labelsize=fs - 1)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("linear CKA (unbiased)", fontsize=fs)
    apply_matplotlib_fonts_to_figure(fig)
    fig.tight_layout()
    stonesoup.show(fig, basename=basename, dpi=144)


def plot_token_token_all_layers(
    tok_stage_dim: torch.Tensor,
    token_labels: list[str],
    stage_names: list[str],
    *,
    basename_prefix: str,
    ncols: int,
    metric: TokenPairMetric,
) -> None:
    """One figure: every stage as a subplot. Cosine uses one global [-1, 1] scale + shared colorbar.

    Dot product uses **per-layer** symmetric ``vmin/vmax = ±max|G|`` so layer-wise magnitude differences
    do not wash out contrast; each subplot gets its own small colorbar.
    """
    n_stages = int(tok_stage_dim.shape[1])
    n_tok = int(tok_stage_dim.shape[0])
    assert len(token_labels) == n_tok
    assert len(stage_names) == n_stages

    per_layer_scale = metric == "dot"
    if metric == "cosine":
        cbar_label = "cosine similarity (token × token)"
        metric_tag = "cos"
    else:
        cbar_label = "dot product (token × token)"
        metric_tag = "dot"

    nrows = int(math.ceil(n_stages / ncols))
    fig_w = 4.2 * ncols
    # Dot: per-layer symmetric scale + small colorbar per subplot (no shared bar — scales differ).
    fig = plt.figure(figsize=(fig_w, 3.8 * nrows + (0.65 if per_layer_scale else 1.15)))
    if per_layer_scale:
        gs = fig.add_gridspec(
            nrows,
            ncols,
            hspace=0.45,
            wspace=0.42,
            left=0.07,
            right=0.96,
            top=0.985,
            bottom=0.06,
        )
    else:
        gs = fig.add_gridspec(
            nrows + 1,
            ncols,
            height_ratios=[1.0] * nrows + [0.22],
            hspace=0.45,
            wspace=0.35,
            left=0.07,
            right=0.97,
            top=0.985,
            bottom=0.06,
        )
    axes = np.empty((nrows, ncols), dtype=object)
    for r in range(nrows):
        for c in range(ncols):
            axes[r, c] = fig.add_subplot(gs[r, c])

    fs = 9.0
    tick_fs = max(6.0, 7.5 - 0.1 * n_tok)
    im_last = None

    for li in range(n_stages):
        stonesoup.check_abort()
        r, c = divmod(li, ncols)
        ax = axes[r, c]
        h = tok_stage_dim[:, li, :]
        sim = token_token_pair_matrix(h, metric=metric).detach().cpu().numpy()
        if per_layer_scale:
            m = max(float(np.max(np.abs(sim))), 1e-9)
            vmin, vmax = -m, m
        else:
            vmin, vmax = -1.0, 1.0
        im_last = ax.imshow(
            sim,
            vmin=vmin,
            vmax=vmax,
            cmap="RdBu_r",
            aspect="equal",
            interpolation="nearest",
        )
        if per_layer_scale:
            _cb = fig.colorbar(im_last, ax=ax, fraction=0.075, pad=0.08)
            _cb.ax.tick_params(labelsize=max(5.0, fs - 2.5))
        ax.set_title(stage_subplot_title(li, stage_names), fontsize=fs)
        ax.set_xticks(np.arange(n_tok))
        ax.set_yticks(np.arange(n_tok))
        ax.set_xticklabels(token_labels, rotation=65, ha="right", fontsize=tick_fs)
        ax.set_yticklabels(token_labels, fontsize=tick_fs)

    for k in range(n_stages, nrows * ncols):
        r, c = divmod(k, ncols)
        axes[r][c].set_visible(False)

    if (not per_layer_scale) and im_last is not None:
        cax = fig.add_subplot(gs[nrows, :])
        cb = fig.colorbar(im_last, cax=cax, orientation="horizontal")
        cb.set_label(cbar_label, fontsize=fs)
        cb.ax.tick_params(labelsize=fs - 1)
    metric_title = "cosine similarity" if metric == "cosine" else "dot product"
    _extra = (
        ""
        if not per_layer_scale
        else " — per-layer symmetric scale ±max|G|; colorbar ticks show range"
    )
    fig.suptitle(
        f"Token×token {metric_title} (stage index = CKA axis; stages 0–{n_stages - 1}){_extra}",
        fontsize=fs + 2,
        y=1.0,
        va="top",
    )
    apply_matplotlib_fonts_to_figure(fig)
    stonesoup.show(fig, basename=f"{basename_prefix}_tokpair_{metric_tag}_all_layers", dpi=144)


# %% Config — Qwen3.5-9B stripes (layer CKA + per-layer token geometry)
REPO_ID = "Qwen/Qwen3.5-9B"
# Short plain prompt so token ticks stay readable (raise ``MAX_LENGTH`` if you truncate).

# Twenty short passages: mixed genres, registers, and languages (non-sink tokens concatenated).
SENTENCES: list[str] = [
    # Expository English (science)
    "Chloroplasts use chlorophyll to absorb photons and store energy in ATP and NADPH. "
    "Those carriers then power the Calvin cycle, which fixes carbon into sugars the cell can use.",
    # Spoken dialogue
    '"Did you remember the keys?"\n"On the hook—unless the cat knocked them down again."',
    # Chinese (informative)
    "月球围绕地球公转，同一面始终朝向地球；潮汐主要由月球引力引起。",
    # Children’s storybook tone
    "The little boat wished for wings, so the wind stitched clouds into sails and pushed it upstream.",
    # Statutory / legal style
    "Where a party fails to perform without excuse, the non-breaching party may seek damages as provided herein.",
    # Recipe / procedural
    "Whisk eggs with salt, fold in warm rice off the heat, then sprinkle nori without over-stirring.",
    # Poetry-ish (line breaks as in source)
    "Fog on the pier—\nA gull borrows the moon\nAnd flies away.",
    # Kid-friendly fable
    "The fox promised grapes were sour anyway, but the crow still laughed from the high branch.",
    # News headline + lead
    "City council delays vote: residents packed the hall, some holding signs that read “Fix the pipes first.”",
    # Text-message / informal
    "omw — grab a table near the window?? coffee’s on me if traffic eats me alive lol",
    # Second Chinese (colloquial narrative)
    "周末我想去爬山，如果下雨就改在家里煮火锅、看电影。",
    # Technical / spec tone
    "Requirement: latency p99 under 120 ms; fallback path must degrade gracefully without data loss.",
    # Courtroom dialogue
    "Your Honor, the exhibit is authenticated under Rule 902—the chain of custody is unbroken.",
    # Sports play-by-play
    "She fakes left, splits two defenders, and curls one into the top corner—stadium erupts.",
    # Academic philosophy (dense)
    "Normative claims concern what ought to be; descriptive claims concern what is—confusing them risks the is-ought gap.",
    # Product blurb / marketing
    "This jacket repels drizzle, packs into its pocket, and weighs less than your phone—trail-tested.",
    # Medical chart note style
    "Patient reports intermittent vertigo; differential includes BPPV versus orthostatic hypotension.",
    # Email closings / formal
    "Please find the revised figures attached. I remain available for a brief call next Tuesday.",
    # Myth / epic register
    "When the river refused the oath, the old king broke his crown and scattered the shards downstream.",
    # Code-adjacent comment (natural language)
    "# TODO: replace O(n^2) pairing with hash map once we confirm key distribution in prod logs.",
]
PROMPT = SENTENCES[0]

MAX_LENGTH = 96
SAFE = hf_repo_id_safe_stem(REPO_ID)
TOKEN_TOK_NCOLS = 3
# Token×token panels & mean line: ``cosine`` (L2-normalized) vs raw ``dot`` product.
TOKEN_PAIR_METRIC: TokenPairMetric = "cosine"
# TOKEN_PAIR_METRIC: TokenPairMetric = "dot"

print(f"REPO_ID={REPO_ID!r}", flush=True)
print(f"TOKEN_PAIR_METRIC={TOKEN_PAIR_METRIC!r}", flush=True)
print(f"PROMPT ({len(PROMPT)} chars): {PROMPT[:120]!r}…", flush=True)

# %% Load model & forward (embedding + post-block stack)
model, proc = stonesoup.load_model(REPO_ID)
model.eval()
device = next(model.parameters()).device
ensure_pad_token_via_eos(inner_tokenizer(proc))

tok_stage_dim, stage_names, token_labels, seq_len = forward_tok_stage_dim(
    model,
    proc,
    device,
    PROMPT,
    max_length=MAX_LENGTH,
)
print(
    f"seq_len={seq_len}  n_stages={tok_stage_dim.shape[1]}  dim={tok_stage_dim.shape[2]}",
    flush=True,
)
print(f"stage_names: {stage_names[:3]} … {stage_names[-2:]}", flush=True)
print(f"token_labels ({len(token_labels)}): {token_labels}", flush=True)

# %% Layer-vs-layer linear CKA (single model) — reproduces stripe heatmap
ck_layer = layer_layer_linear_cka(tok_stage_dim, device=device)
ck_lo = float(torch.min(ck_layer).item())
ck_hi = float(torch.max(ck_layer).item())
ck_pad = 0.02 * (ck_hi - ck_lo + 1e-9)
vmin_cka = max(0.0, ck_lo - ck_pad)
vmax_cka = min(1.0, ck_hi + ck_pad)
plot_layer_layer_heatmap(
    ck_layer,
    title=f"{REPO_ID.split('/')[-1]} — layer vs layer linear CKA (n_tokens={seq_len})",
    basename=f"{SAFE}_layer_vs_layer_cka",
    vmin=0,
    vmax=vmax_cka,
)

# %% Per-layer token×token similarity (single figure, all stages)
plot_token_token_all_layers(
    tok_stage_dim,
    token_labels,
    stage_names,
    basename_prefix=SAFE,
    ncols=TOKEN_TOK_NCOLS,
    metric=TOKEN_PAIR_METRIC,
)

# %% Mean pairwise token score vs stage (off-diagonal mean per layer)
plot_mean_pairwise_metric_by_stage(
    tok_stage_dim,
    basename=f"{SAFE}_mean_pairwise_{TOKEN_PAIR_METRIC}_by_stage",
    title=(
        f"{REPO_ID.split('/')[-1]} — mean off-diagonal token×token "
        f"({'cosine' if TOKEN_PAIR_METRIC == 'cosine' else 'dot product'}, n_tokens={seq_len})"
    ),
    metric=TOKEN_PAIR_METRIC,
)
