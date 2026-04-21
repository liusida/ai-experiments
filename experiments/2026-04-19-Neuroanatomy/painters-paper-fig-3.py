# %% Imports, config & helpers
from __future__ import annotations

import stonesoup
from stonesoup.experiment import (
    capture_embed_and_post_blocks,
    configure_matplotlib_agg,
    encode_text_inputs,
    ensure_pad_token_via_eos,
    hf_repo_id_safe_stem,
    inner_tokenizer,
)

configure_matplotlib_agg()

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import numpy as np
import torch
import torch.nn.functional as F

# Llama 2 7B — gated on Hugging Face; log in with `huggingface-cli login` if load fails.
MODEL_ID = "meta-llama/Llama-2-7b-hf"

PHYSICS_PARAGRAPH = (
    "Classical mechanics describes motion using Newton's laws: a body accelerates when a net force "
    "acts on it, and momentum is conserved in isolated systems. Energy can change form yet is often "
    "conserved, which lets us predict orbits, collisions, and pendulums with surprising accuracy."
)


def mean_vectors_over_tokens(
    stack_bt_h: torch.Tensor,
    *,
    start_token: int = 0,
) -> torch.Tensor:
    """Mean hidden state per layer; ``stack`` per layer is ``(B, T, H)``."""
    if start_token < 0:
        raise ValueError("start_token must be >= 0")
    seq = stack_bt_h[:, start_token:, :]
    if seq.shape[1] == 0:
        raise ValueError("No tokens left after start_token; sequence is too short.")
    return seq.mean(dim=(0, 1)).float()


def mean_vectors_single_token(stack_bt_h: torch.Tensor, token_idx: int) -> torch.Tensor:
    """Per layer: activation at ``token_idx`` (0-based); mean over batch if ``B`` > 1."""
    if token_idx < 0:
        raise ValueError("token_idx must be >= 0")
    seq_len = stack_bt_h.shape[1]
    if seq_len <= token_idx:
        raise ValueError(f"Sequence length {seq_len} too short for token index {token_idx}")
    return stack_bt_h[:, token_idx, :].mean(dim=0).float()


def mean_vectors_first_token_only(stack_bt_h: torch.Tensor) -> torch.Tensor:
    """Per layer: activation at token index 0 only (mean over batch if ``B`` > 1)."""
    return mean_vectors_single_token(stack_bt_h, 0)


def cosine_sim_matrix(mean_per_layer: torch.Tensor) -> torch.Tensor:
    """Pairwise cosine similarity; ``mean_per_layer`` is ``(num_layers, hidden)``."""
    z = F.normalize(mean_per_layer, dim=1, eps=1e-8)
    return z @ z.T


def plot_cosine_heatmap(
    cos: torch.Tensor,
    labels: list[str],
    title: str,
    basename: str,
) -> None:
    arr = cos.detach().cpu().numpy()
    n = len(labels)
    fig, ax = plt.subplots(figsize=(0.35 * n + 2, 0.35 * n + 2))
    im = ax.imshow(arr, vmin=0.0, vmax=1.0, cmap="viridis", aspect="equal")
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="cosine similarity")
    fig.tight_layout()
    stonesoup.show(fig, basename=basename, dpi=140)


def _trunc(s: str, max_len: int) -> str:
    s = s.replace("\n", "↵")
    return s if len(s) <= max_len else s[: max_len - 1] + "…"


def plot_per_token_cosine_subplot_grid(
    cos_per_token: list[torch.Tensor],
    subtitles: list[str],
    basename: str,
    *,
    ncols: int = 8,
    dpi: int = 110,
) -> None:
    """One figure: small ``imshow`` heatmaps in a grid, shared colorbar (0–1)."""
    n = len(cos_per_token)
    if n != len(subtitles):
        raise ValueError("cos_per_token and subtitles length mismatch")
    nrows = (n + ncols - 1) // ncols
    fig_w = min(2.1 * ncols, 48)
    fig_h = min(2.0 * nrows, 80)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
    cmap = plt.get_cmap("viridis")
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    for idx, ax in enumerate(axes.flat):
        if idx >= n:
            ax.set_axis_off()
            continue
        arr = cos_per_token[idx].detach().cpu().numpy()
        ax.imshow(arr, norm=norm, cmap=cmap, aspect="equal", interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(subtitles[idx], fontsize=5)
    fig.subplots_adjust(left=0.04, right=0.89, bottom=0.03, top=0.94, wspace=0.35, hspace=0.5)
    cax = fig.add_axes([0.91, 0.15, 0.015, 0.7])
    fig.colorbar(mappable, cax=cax, label="cosine sim.")
    fig.suptitle(f"{MODEL_ID}: Per-token layer mean cosine similarity (single position)", fontsize=11, y=0.995)
    stonesoup.show(fig, basename=basename, dpi=dpi)
    plt.close(fig)

# %% Load Llama 2 7B (toolbar Load or cell load)
model, proc = stonesoup.load_model(MODEL_ID)
model.eval()
tokenizer = inner_tokenizer(proc)
ensure_pad_token_via_eos(tokenizer)
device = next(model.parameters()).device
dtype = next(model.parameters()).dtype
print("device:", device, "dtype:", dtype, flush=True)

# %% Encode physics paragraph & capture embedding + post-block activations
inputs = encode_text_inputs(proc, PHYSICS_PARAGRAPH, device=device)
stack, layer_names = capture_embed_and_post_blocks(model, inputs, use_cache=False)
# stack: (num_stages, batch, seq, hidden); stage 0 = embedding output, then each decoder block.
print("stack shape:", tuple(stack.shape), "stages:", len(layer_names), flush=True)
print("layer names (first/last):", layer_names[0], "…", layer_names[-1], flush=True)

# %% Cosine similarity of per-layer mean activations — all prompt tokens
safe = hf_repo_id_safe_stem(MODEL_ID)
short_labels = ["emb"] + [f"L{i}" for i in range(len(layer_names) - 1)]

means_all = torch.stack(
    [mean_vectors_over_tokens(stack[s], start_token=0) for s in range(stack.shape[0])],
    dim=0,
)
cos_all = cosine_sim_matrix(means_all)
plot_cosine_heatmap(
    cos_all,
    short_labels,
    title="Cosine similarity of mean hidden states (all prompt tokens)",
    basename=f"{safe}_layer_mean_cosine_all_tokens",
)

# %% Same analysis excluding the first token (compare to previous heatmap)
means_skip0 = torch.stack(
    [mean_vectors_over_tokens(stack[s], start_token=1) for s in range(stack.shape[0])],
    dim=0,
)
cos_skip0 = cosine_sim_matrix(means_skip0)
plot_cosine_heatmap(
    cos_skip0,
    short_labels,
    title="Cosine similarity of mean hidden states (excluding first token)",
    basename=f"{safe}_layer_mean_cosine_skip_first_token",
)

# %% First token only — cosine similarity of per-position-0 activations across layers
means_first = torch.stack(
    [mean_vectors_first_token_only(stack[s]) for s in range(stack.shape[0])],
    dim=0,
)
cos_first = cosine_sim_matrix(means_first)
plot_cosine_heatmap(
    cos_first,
    short_labels,
    title="Cosine similarity of hidden states (first prompt token only)",
    basename=f"{safe}_layer_mean_cosine_first_token_only",
)

# %% Per-token cosine heatmaps — one figure, small subplots (token index + text in titles)
input_ids_1d = inputs["input_ids"][0]
seq_len_tokens = int(input_ids_1d.shape[0])
if stack.shape[2] != seq_len_tokens:
    raise ValueError(
        f"stack seq dim {stack.shape[2]} != tokenizer seq len {seq_len_tokens}"
    )
cos_per_tok: list[torch.Tensor] = []
subtitles: list[str] = []
for token_idx in range(seq_len_tokens):
    stonesoup.check_abort()
    tid = int(input_ids_1d[token_idx].item())
    piece = tokenizer.convert_ids_to_tokens([tid])[0]
    subtitles.append(f"{token_idx}: {_trunc(piece, 28)}")
    means_one = torch.stack(
        [mean_vectors_single_token(stack[s], token_idx) for s in range(stack.shape[0])],
        dim=0,
    )
    cos_per_tok.append(cosine_sim_matrix(means_one))
plot_per_token_cosine_subplot_grid(
    cos_per_tok,
    subtitles,
    basename=f"{safe}_layer_mean_cosine_all_tokens_subplots",
    ncols=8,
)
print(f"Per-token subplot grid done: {seq_len_tokens} panels.", flush=True)
