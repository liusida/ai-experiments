# %% Config & imports
from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import stonesoup

# Setup 1 (IOI)
PREFIX = "This morning,"
SUFFIX = " went there, and she was mad"
A_REPLACEMENTS_CANDIDATES: list[str] = [
    " she",
    " he",
]
# Setup 2
# PREFIX = "This"
# SUFFIX = ", she went to the beach."
# A_REPLACEMENTS_CANDIDATES: list[str] = [
#     " afternoon",
#     " night",
# ]
# Setup 3
# PREFIX = "This morning, she"
# SUFFIX = " to work, she was"
# A_REPLACEMENTS_CANDIDATES: list[str] = [
#     " drove",
#     " ran",
# ]

# Max suffix tokens after <A> to plot (from first token of " went" onward).
MAX_SUFFIX_TOKENS = 8

# %% Helpers: encode, decoder blocks, capture stack


def encode_prompt(proc: Any, prompt: str, device: torch.device) -> dict[str, Any]:
    tok = getattr(proc, "tokenizer", None) or proc
    enc = tok(prompt, return_tensors="pt", return_attention_mask=True, add_special_tokens=True)
    return {k: v.to(device) for k, v in enc.items()}


def inner_tokenizer(proc: Any) -> Any:
    return getattr(proc, "tokenizer", None) or proc


def decoder_blocks(model: Any) -> list[Any]:
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return list(model.gpt_neox.layers)
    inner = getattr(model, "model", None)
    if inner is None:
        raise TypeError(f"No .transformer / .model on {type(model).__name__}")
    if hasattr(inner, "language_model") and hasattr(inner.language_model, "layers"):
        return list(inner.language_model.layers)
    return list(inner.layers)


def capture_stack(model: Any, inputs: dict[str, Any]) -> tuple[torch.Tensor, list[str]]:
    blocks = decoder_blocks(model)
    out: list[torch.Tensor] = []

    def grab_embed(_mod, _in, x: torch.Tensor) -> None:
        out.append(x.detach())

    def grab_layer(_mod, _in, x: Any) -> None:
        h = x[0] if isinstance(x, tuple) else x
        out.append(h.detach())

    emb = model.get_input_embeddings()
    hooks = [emb.register_forward_hook(grab_embed)]
    hooks += [L.register_forward_hook(grab_layer) for L in blocks]
    try:
        with torch.inference_mode():
            model(**inputs, use_cache=False)
    finally:
        for h in hooks:
            h.remove()

    names = ["embedding"] + [f"layer_{i}" for i in range(len(blocks))]
    assert len(out) == len(names)
    return torch.stack(out, dim=0), names


def token_range_for_a(inner: Any, prefix: str, a: str) -> tuple[int, int]:
    """Half-open ``[n0, n1)`` token span for ``a`` in ``prefix + a`` (matches full sentence)."""
    pre = inner(prefix, add_special_tokens=True, return_tensors="pt")
    pa = inner(prefix + a, add_special_tokens=True, return_tensors="pt")
    n0 = int(pre["input_ids"].shape[1])
    n1 = int(pa["input_ids"].shape[1])
    return n0, n1


def a_token_index(inner: Any, prefix: str, a: str) -> int:
    """Index of the single token for ``a`` after ``prefix`` (raises if not exactly one token)."""
    n0, n1 = token_range_for_a(inner, prefix, a)
    if n1 - n0 != 1:
        raise ValueError(f"expected single-token <A>, got span [{n0}, {n1}) for a={a!r}")
    return n0


def verify_full_prefix(inner: Any, prefix: str, a: str, suffix: str) -> None:
    full = prefix + a + suffix
    pa = inner(prefix + a, add_special_tokens=True, return_tensors="pt")["input_ids"][0]
    full_ids = inner(full, add_special_tokens=True, return_tensors="pt")["input_ids"][0]
    if not torch.equal(full_ids[: pa.shape[0]], pa):
        raise RuntimeError(
            f"Token mismatch: full prefix+a slice != prefix+a alone. "
            f"Try different spacing or a smaller suffix for this tokenizer."
        )


def pairwise_cos_upper_mean(vecs: torch.Tensor) -> float:
    """``vecs``: (n_var, d) → mean cos over pairs i<j."""
    n = int(vecs.shape[0])
    if n < 2:
        return float("nan")
    x = F.normalize(vecs, p=2, dim=-1, eps=1e-12)
    g = (x @ x.T).float()
    tri_i, tri_j = torch.triu_indices(n, n, offset=1)
    return float(g[tri_i, tri_j].mean().cpu())


def pairwise_cos_matrix(vecs: torch.Tensor) -> np.ndarray:
    """``vecs``: (n_var, d) → (n_var, n_var) cosine similarity."""
    x = F.normalize(vecs, p=2, dim=-1, eps=1e-12)
    return (x @ x.T).float().cpu().numpy()


def pairwise_cos_matrices_all_layers(per_variant: list[torch.Tensor], seq_index: int) -> np.ndarray:
    """``(n_stages, n_var, n_var)`` pairwise cos at each layer at token ``seq_index`` (same for all variants)."""
    n_var = len(per_variant)
    n_stages = int(per_variant[0].shape[0])
    out = np.zeros((n_stages, n_var, n_var), dtype=np.float64)
    for li in range(n_stages):
        stonesoup.check_abort()
        vecs = torch.stack([per_variant[v][li, 0, seq_index, :] for v in range(n_var)], dim=0)
        out[li] = pairwise_cos_matrix(vecs)
    return out


def plot_pairwise_cos_heatmap_per_layer(
    mats: np.ndarray,
    stage_names: list[str],
    labels: list[str],
    suptitle: str,
    basename: str,
    ncols: int = 6,
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> None:
    """``mats``: (n_stages, n_var, n_var) pairwise cosine per stage; one heatmap per stage."""
    n_stages, n_var, n2 = mats.shape
    assert n_var == n2
    nrows = int(np.ceil(n_stages / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(2.2 * ncols, 2.0 * nrows),
        squeeze=False,
        layout="constrained",
    )
    lbl = [repr(s)[1:-1] for s in labels]
    im = None
    for li in range(n_stages):
        stonesoup.check_abort()
        r, c = divmod(li, ncols)
        ax = axes[r][c]
        im = ax.imshow(
            mats[li],
            vmin=vmin,
            vmax=vmax,
            cmap="Blues",
            aspect="equal",
        )
        ax.set_xticks(range(n_var))
        ax.set_yticks(range(n_var))
        if r == nrows - 1:
            ax.set_xticklabels(lbl, rotation=45, ha="right", fontsize=5)
        else:
            ax.set_xticklabels([])
        if c == 0:
            ax.set_yticklabels(lbl, fontsize=5)
        else:
            ax.set_yticklabels([])
        name = stage_names[li] if li < len(stage_names) else f"stage_{li}"
        ax.set_title(name, fontsize=7)
    for k in range(n_stages, nrows * ncols):
        r, c = divmod(k, ncols)
        axes[r][c].set_visible(False)
    assert im is not None
    axes_used = [axes[r][c] for r in range(nrows) for c in range(ncols) if r * ncols + c < n_stages]
    fig.suptitle(suptitle + "\nrows/cols = <A> pairs (i vs j)", fontsize=10)
    fig.colorbar(im, ax=axes_used, fraction=0.02, label="cosine similarity")
    stonesoup.show(fig, basename=basename)


# %% Load model & collect hidden states per <A> variant
MODEL_IDs = [
    # Baseline
    "openai-community/gpt2-xl",

    # Tiny
    "EleutherAI/pythia-1.4b",
    "google/gemma-2-2b",
    "meta-llama/llama-3.2-3B",
    "Qwen/Qwen2.5-3B",
    "google/gemma-4-E2B",
    "mistralai/Ministral-3-3B-Base-2512",

    # Small
    "meta-llama/Llama-2-7b-hf",
    "Qwen/Qwen3-8B-Base",
    "tiiuae/falcon-7b",
    "allenai/Olmo-3-1025-7B",
]
for MODEL_ID in MODEL_IDs:
    stonesoup.check_abort()
    model, proc = stonesoup.load_model(MODEL_ID)
    model.eval()
    device = next(model.parameters()).device
    inner = inner_tokenizer(proc)

    A_REPLACEMENTS: list[str] = []
    for a in A_REPLACEMENTS_CANDIDATES:
        stonesoup.check_abort()
        n0, n1 = token_range_for_a(inner, PREFIX, a)
        if n1 - n0 == 1:
            A_REPLACEMENTS.append(a)
    if len(A_REPLACEMENTS) < 2:
        raise RuntimeError(
            "Need at least two one-token <A> values; expand A_REPLACEMENTS_CANDIDATES or switch MODEL_ID."
        )

    for a in A_REPLACEMENTS:
        stonesoup.check_abort()
        verify_full_prefix(inner, PREFIX, a, SUFFIX)

    # (n_var, n_stages, seq, hidden) on CPU float — single-token <A> ⇒ same token index for every variant
    per_variant: list[torch.Tensor] = []

    for a in A_REPLACEMENTS:
        stonesoup.check_abort()
        full = PREFIX + a + SUFFIX
        inputs = encode_prompt(proc, full, device)
        stack, stage_names = capture_stack(model, inputs)
        hs = stack.float().cpu()
        per_variant.append(hs)

    STAGE_NAMES = stage_names
    N_STAGES = int(per_variant[0].shape[0])
    N_VAR = len(A_REPLACEMENTS)
    assert all(v.shape[0] == N_STAGES for v in per_variant)

    idx_a = a_token_index(inner, PREFIX, A_REPLACEMENTS[0])
    for a in A_REPLACEMENTS[1:]:
        if a_token_index(inner, PREFIX, a) != idx_a:
            raise RuntimeError(f"<A> token index mismatch for a={a!r} vs first variant (idx_a={idx_a})")

    seq_len = int(per_variant[0].shape[2])
    assert all(int(v.shape[2]) == seq_len for v in per_variant)
    n_suffix = seq_len - idx_a - 1
    n_suffix_plot = min(MAX_SUFFIX_TOKENS, n_suffix)

    layer_labels = [f"L{i}" for i in range(N_STAGES)]

    # Plot 1: mean pairwise cos between <A> variants at last token of <A> (per layer)
    # mean_cos_last_a: list[float] = []
    # for li in range(N_STAGES):
    #     stonesoup.check_abort()
    #     vecs = torch.stack([per_variant[v][li, 0, idx_a, :] for v in range(N_VAR)], dim=0)
    #     mean_cos_last_a.append(pairwise_cos_upper_mean(vecs))

    # fig1, ax1 = plt.subplots(figsize=(8, 4))
    # ax1.plot(range(N_STAGES), mean_cos_last_a, marker="o", ms=3)
    # ax1.set_xticks(range(0, N_STAGES, max(1, N_STAGES // 12)))
    # ax1.set_xticklabels([layer_labels[i] for i in range(0, N_STAGES, max(1, N_STAGES // 12))], rotation=45, ha="right")
    # ax1.set_ylim(-0.05, 1.05)
    # ax1.set_xlabel("stage")
    # ax1.set_ylabel("<-[Different]        Cosine Similarity       [Similar]->")
    # ax1.set_title(f"Token Difference {MODEL_ID}:\n\n{PREFIX}[{A_REPLACEMENTS[0]}/{A_REPLACEMENTS[1]}]")
    # ax1.grid(True, alpha=0.3)
    # fig1.tight_layout()
    # stonesoup.show(fig1, basename=f"Token_Difference_{MODEL_ID.replace('/', '__').replace(':', '-')}_last_token_of_A")

    # Plot 2: one line per suffix token (first token after <A>, then next, …)
    _ref_ids = inner(
        PREFIX + A_REPLACEMENTS[0] + SUFFIX,
        return_tensors="pt",
        add_special_tokens=True,
    )["input_ids"][0]
    suffix_legend_labels: list[str] = []
    for k in range(-1,n_suffix_plot):
        tid = int(_ref_ids[idx_a + 1 + k].item())
        piece = inner.decode([tid], skip_special_tokens=False)
        suffix_legend_labels.append(repr(piece))

    fig2, ax2 = plt.subplots(figsize=(9, 4.5))
    for k in range(-1,n_suffix_plot):
        stonesoup.check_abort()
        curve: list[float] = []
        for li in range(N_STAGES):
            vecs = torch.stack([per_variant[v][li, 0, idx_a + 1 + k, :] for v in range(N_VAR)], dim=0)
            curve.append(pairwise_cos_upper_mean(vecs))
        ax2.plot(range(N_STAGES), curve, marker=".", ms=2, label=suffix_legend_labels[k+1])

    ax2.set_xticks(range(0, N_STAGES, max(1, N_STAGES // 12)))
    ax2.set_xticklabels([layer_labels[i] for i in range(0, N_STAGES, max(1, N_STAGES // 12))], rotation=45, ha="right")
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_xlabel("stage")
    ax2.set_ylabel("<-[Different]        Cosine Similarity       [Similar]->")
    ax2.set_title(f"Activation Difference {MODEL_ID}:\n\n{PREFIX}[{A_REPLACEMENTS[0]}/{A_REPLACEMENTS[1]}]{SUFFIX}")
    ax2.legend(loc="best", fontsize=8, ncol=2)
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    stonesoup.show(fig2, basename=f"Activation_Difference_{MODEL_ID.replace('/', '__').replace(':', '-')}")
    plt.close("all")
