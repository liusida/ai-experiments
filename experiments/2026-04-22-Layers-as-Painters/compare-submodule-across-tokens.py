# %% Imports, config & helpers
from __future__ import annotations

import stonesoup
from stonesoup.experiment import (
    configure_matplotlib_agg,
    decoder_blocks,
    hf_repo_id_safe_stem,
    inner_tokenizer,
)

configure_matplotlib_agg()

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

MODEL_ID = "Qwen/Qwen3-8B"
MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"

# Qwen3's chat template accepts enable_thinking=False; Llama's doesn't.
_IS_QWEN = "qwen" in MODEL_ID.lower()
_CHAT_KW = {"enable_thinking": False} if _IS_QWEN else {}

QUESTIONS = [
    "What is the captial of France?",
    "What is the captial of Spain?",
    "What is the captial of Brazil?",
    "What about Japan?",
    "Thank you.",
]
MAX_NEW_TOKENS = 32


def _first(x):
    """HF block/attn returns may be tuples; grab the hidden-states element."""
    return x[0] if isinstance(x, tuple) else x

# %% Load model (toolbar Load or cell load)
model, proc = stonesoup.load_model(MODEL_ID, use_offline=False)
model.eval()
tokenizer = inner_tokenizer(proc)
device = next(model.parameters()).device
dtype = next(model.parameters()).dtype
print("device:", device, "dtype:", dtype, flush=True)

# %% Multi-turn chat: for each question, generate a response and append it
messages_full: list[dict] = []
for turn_idx, question in enumerate(QUESTIONS):
    messages_full.append({"role": "user", "content": question})
    prompt_text_step = tokenizer.apply_chat_template(
        messages_full,
        tokenize=False,
        add_generation_prompt=True,
        **_CHAT_KW,
    )
    prompt_inputs_step = tokenizer(prompt_text_step, return_tensors="pt").to(device)
    with torch.no_grad():
        gen = model.generate(
            **prompt_inputs_step,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    q_len_step = int(prompt_inputs_step["input_ids"].shape[1])
    response_ids = gen[0, q_len_step:].tolist()
    response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
    print(f"Q{turn_idx + 1}: {question}")
    print(f"A{turn_idx + 1}: {response_text!r}")
    messages_full.append({"role": "assistant", "content": response_text})

# %% Tokenize the full conversation for the hooked forward pass
full_text = tokenizer.apply_chat_template(
    messages_full,
    tokenize=False,
    add_generation_prompt=False,
    **_CHAT_KW,
)
print(full_text)

inputs = tokenizer(full_text, return_tensors="pt").to(device)
token_ids = inputs["input_ids"][0].tolist()
token_pieces = [tokenizer.decode([t]) for t in token_ids]
seq_len = len(token_ids)

# Turn-start boundaries. Qwen uses <|im_start|>; Llama 3 uses <|start_header_id|>.
im_start_id = None
for _candidate in ("<|im_start|>", "<|start_header_id|>"):
    _id = tokenizer.convert_tokens_to_ids(_candidate)
    if _id is not None and _id != tokenizer.unk_token_id:
        im_start_id = _id
        break
turn_starts = [i for i, t in enumerate(token_ids) if t == im_start_id]
if not _IS_QWEN:
    turn_starts = turn_starts[1:] # Llama has an extra segment of hidden prompt
print("full length (tokens):", seq_len, " turn-start positions:", turn_starts, flush=True)

# Shade spans: (start_pos, end_pos, role) for each turn (last one runs to seq_len).
turn_roles = [m["role"] for m in messages_full]
turn_spans = [
    (
        turn_starts[i],
        turn_starts[i + 1] if i + 1 < len(turn_starts) else seq_len,
        turn_roles[i],
    )
    for i in range(len(turn_roles))
]

# %% Forward pass with hooks: capture residual stream (h_pre/post) and submodule
# contributions (Δ_attn, Δ_mlp). Names follow the diagram.
blocks = decoder_blocks(model)
num_layers = len(blocks)
print(f"num decoder layers: {num_layers}", flush=True)

h_pre_buf:     list[torch.Tensor | None] = [None] * num_layers  # residual before attn
h_post_buf:    list[torch.Tensor | None] = [None] * num_layers  # residual after mlp
delta_attn_buf: list[torch.Tensor | None] = [None] * num_layers  # self_attn out (pre-add)
delta_mlp_buf:  list[torch.Tensor | None] = [None] * num_layers  # mlp out (pre-add)


def make_block_pre_hook(i):
    def hook(_module, inputs):
        h_pre_buf[i] = _first(inputs).detach().to(torch.float32).cpu()
    return hook


def make_block_post_hook(i):
    def hook(_module, _inputs, output):
        h_post_buf[i] = _first(output).detach().to(torch.float32).cpu()
    return hook


def make_attn_hook(i):
    def hook(_module, _inputs, output):
        delta_attn_buf[i] = _first(output).detach().to(torch.float32).cpu()
    return hook


def make_mlp_hook(i):
    def hook(_module, _inputs, output):
        delta_mlp_buf[i] = _first(output).detach().to(torch.float32).cpu()
    return hook


handles = []
for i, block in enumerate(blocks):
    handles.append(block.register_forward_pre_hook(make_block_pre_hook(i)))
    handles.append(block.register_forward_hook(make_block_post_hook(i)))
    handles.append(block.self_attn.register_forward_hook(make_attn_hook(i)))
    handles.append(block.mlp.register_forward_hook(make_mlp_hook(i)))

try:
    with torch.no_grad():
        model(**inputs, use_cache=False)
finally:
    for h in handles:
        h.remove()

# Stack into (num_layers, seq, hidden) tensors.
h_pre_stack      = torch.stack([h_pre_buf[i][0]      for i in range(num_layers)], dim=0)
h_post_stack     = torch.stack([h_post_buf[i][0]     for i in range(num_layers)], dim=0)
delta_attn_stack = torch.stack([delta_attn_buf[i][0] for i in range(num_layers)], dim=0)
delta_mlp_stack  = torch.stack([delta_mlp_buf[i][0]  for i in range(num_layers)], dim=0)
# h_mid = h_pre + Δ_attn (the residual after attention, before MLP).
h_mid_stack = h_pre_stack + delta_attn_stack

# Drop the first token (Qwen has no <bos>; position 0 is <|im_start|> and acts as
# an attention sink → far off-scale in most metrics). Sync token-level metadata.
h_pre_stack      = h_pre_stack[:, 1:]
h_mid_stack      = h_mid_stack[:, 1:]
h_post_stack     = h_post_stack[:, 1:]
delta_attn_stack = delta_attn_stack[:, 1:]
delta_mlp_stack  = delta_mlp_stack[:, 1:]
token_ids    = token_ids[1:]
token_pieces = token_pieces[1:]
seq_len      = len(token_ids)
turn_starts  = [p - 1 for p in turn_starts if p - 1 >= 0]
turn_spans   = [
    (max(s - 1, 0), e - 1, r) for (s, e, r) in turn_spans if e - 1 > 0
]
print("stacks:", tuple(h_pre_stack.shape), "(h_pre/h_mid/h_post/Δ_attn/Δ_mlp)", flush=True)


# %% Plot helper (one cell per (feature, metric) pair below)
safe = hf_repo_id_safe_stem(MODEL_ID)
x_positions = np.arange(seq_len)


def _escape(p: str) -> str:
    """Make newlines / tabs visible in tick labels."""
    return p.replace("\n", "\\n").replace("\t", "\\t")


# Top labels: the token at position t.
tick_labels = [_escape(p) for p in token_pieces]
# Bottom labels: the next-token at position t (shifted by +1, last is blank).
pred_labels = [
    _escape(token_pieces[t + 1]) if t + 1 < seq_len else ""
    for t in range(seq_len)
]

# One color per feature from matplotlib's default tab10 palette.
PALETTE = plt.get_cmap("tab20").colors

def plot_grid_combined(
    series: list[tuple[str, np.ndarray, str]],
    metric_name: str,
    ylim: tuple[float] = None,
) -> None:
    # Blocked order: all kind-0 layers first (L0..Llast), then kind-1, ...
    n_kinds = len(series)
    rows = n_kinds * num_layers
    fig, axes = plt.subplots(
        rows, 1,
        figsize=(seq_len / 10, rows * 0.35),
        sharex=True,
        gridspec_kw={"hspace": 0.03},
    )
    axes = np.asarray(axes).reshape(rows)
    if ylim is None:
        if metric_name.startswith("cos"):
            y_min = -1.0
            y_max = 1.0
        else:
            # Clip y-range to mean ± 3·std so outliers don't flatten the rest of the plot.
            # Drop token 0 (BOS-like) — its values are usually far off-scale.
            combined = np.concatenate([m[:, 1:].ravel() for _, m, _ in series])
            mu = float(combined.mean())
            sd = float(combined.std())
            y_min = mu - 3 * sd
            y_max = mu + 3 * sd
    else:
        y_min, y_max = ylim
    frame_color = "#bbbbbb"
    user_bg = "#aaaaaa"
    asst_bg = "#ffffff"
    for kind_idx, (kind, mat, color) in enumerate(series):
        row_base = kind_idx * num_layers
        for i in range(num_layers):
            ax = axes[row_base + i]
            # Shade per-turn background by role (behind the data).
            for start, end, role in turn_spans:
                ax.axvspan(
                    start, end,
                    color=user_bg if role == "user" else asst_bg,
                    alpha=0.1, linewidth=0, zorder=0,
                )
            ax.plot(x_positions, mat[i], "o-", color=color, markersize=1.3, linewidth=0.8)
            ax.axhline(0.0, color="gray", linestyle=":", linewidth=0.8)
            for pos in turn_starts:
                ax.axvline(pos, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
            ax.set_ylim(y_min, y_max)
            ax.grid(alpha=0.2, linewidth=0.3)
            ax.set_xticks(x_positions)
            for spine in ax.spines.values():
                spine.set_linewidth(0.4)
                spine.set_color(frame_color)
            ax.tick_params(
                axis="y", colors="#888888", width=0.4, length=2, labelsize=5,
            )
            # Hide x tick labels everywhere; they'll be forced onto axes[0] below.
            ax.tick_params(
                axis="x", labelbottom=False, labeltop=False,
                bottom=False, top=False,
            )
            ax.set_ylabel(f"{kind}\nL{i}", fontsize=7)

    # With sharex=True, matplotlib auto-shows x labels on the bottom-most axis.
    # Force them onto the top-most axis instead.
    top_ax = axes[0]
    top_ax.set_xticks(x_positions)
    top_ax.set_xticklabels(
        tick_labels,
        rotation=-45, ha="right", va="bottom",
        rotation_mode="anchor",
        fontsize=5,
    )
    top_ax.xaxis.tick_top()
    top_ax.xaxis.set_label_position("top")
    top_ax.tick_params(
        axis="x", labeltop=True, labelbottom=False, top=True, bottom=False,
        colors="#888888", width=0.4, length=2, labelsize=5,
    )

    # Bottom labels on the last axis: next-token (what follows position t in
    # the input). Use a *secondary* x-axis because sharex=True makes all axes
    # share a single formatter — calling set_xticklabels on the primary axis
    # would clobber the top-axis labels we just set above.
    bottom_ax = axes[-1]
    bot_sec = bottom_ax.secondary_xaxis("bottom")
    bot_sec.set_xticks(x_positions)
    bot_sec.set_xticklabels(
        pred_labels,
        rotation=-45, ha="left", va="top",
        rotation_mode="anchor",
        fontsize=5,
    )
    bot_sec.tick_params(
        axis="x", colors="#888888", width=0.4, length=2, labelsize=5,
    )

    kinds_str = ", ".join(k for k, _, _ in series)
    fig.suptitle(
        f"{MODEL_ID} — {kinds_str}: {metric_name}",
        fontsize=11,
    )
    fig_h = rows * 0.35
    # Top carries the suptitle + rotated tick labels above axes[0].
    # Bottom carries the next-token prediction labels below axes[-1].
    fig.subplots_adjust(
        left=0.07, right=0.99,
        top=1 - 0.8 / fig_h,
        bottom=0.6 / fig_h,
        hspace=0.03,
    )
    kinds_slug = "_".join(k.replace("Δ_", "delta_") for k, _, _ in series)
    stonesoup.show(fig, basename=f"{safe}_{kinds_slug}_{metric_name}", dpi=140)
    plt.close(fig)

# %% Per-token metric vs the per-layer mean (swap METRIC to change the y-axis)
"""
We have five main variables:
delta_attn, delta_mlp, and h_pre, h_mid, h_post.
Define a metric here to capture one aspect of the data.
"""

def metric_norm_by_layer(stack: torch.Tensor) -> np.ndarray:
    normed = stack.norm(dim=-1)
    normed_by_layer = F.layer_norm(normed, normalized_shape=(normed.shape[-1],))
    return normed_by_layer


def metric_cos_to_mean(stack: torch.Tensor) -> np.ndarray:
    """Cosine similarity of each (layer, token) vector to the per-layer mean."""
    mean = stack.mean(dim=1)                                 # (L, H)
    s_n = F.normalize(stack, dim=-1, eps=1e-8)               # (L, T, H)
    m_n = F.normalize(mean,  dim=-1, eps=1e-8).unsqueeze(1)  # (L, 1, H)
    return (s_n * m_n).sum(dim=-1).numpy()                   # (L, T)

def metric_cos(stack_1: torch.Tensor, stack_2: torch.Tensor) -> np.ndarray:
    """Cos sim of two variables"""
    n_1 = F.normalize(stack_1, dim=-1, eps=1e-8)
    n_2 = F.normalize(stack_2, dim=-1, eps=1e-8)
    return (n_1 * n_2).sum(dim=-1).numpy()

def metric_norm_ratio(stack_1: torch.Tensor, stack_2: torch.Tensor) -> np.ndarray:
    """Ratio of norms of two variables"""
    n_1 = stack_1.norm(dim=-1)
    n_2 = stack_2.norm(dim=-1)
    return (n_1 / n_2).numpy()

# %% Δ_attn: cos_to_mean
plot_grid_combined(
    [("Δ_attn", metric_cos_to_mean(delta_attn_stack), PALETTE[0])],
    metric_name="cos_to_mean",
)

# %% Δ_mlp: cos_to_mean
plot_grid_combined(
    [("Δ_mlp", metric_cos_to_mean(delta_mlp_stack), PALETTE[2])],
    metric_name="cos_to_mean",
)

# %% h_pre: norm
plot_grid_combined(
    [("h_pre", metric_norm_by_layer(h_pre_stack), PALETTE[4])],
    metric_name="norm",
)

# %% h_mid: norm
plot_grid_combined(
    [("h_mid", metric_norm_by_layer(h_mid_stack), PALETTE[6])],
    metric_name="norm",
)

# %% h_post: norm
plot_grid_combined(
    [("h_post", metric_norm_by_layer(h_post_stack), PALETTE[8])],
    metric_name="norm",
)

# %% cos( Δ_attn, h_pre )
plot_grid_combined(
    [("", metric_cos(delta_attn_stack, h_pre_stack), PALETTE[10])],
    metric_name="cos( Δ_attn, h_pre ): does Δ amplify or remove",
)

# %% cos( Δ_mlp, h_mid )
plot_grid_combined(
    [("", metric_cos(delta_mlp_stack, h_mid_stack), PALETTE[12])],
    metric_name="cos( Δ_mlp, h_mid ): does Δ amplify or remove",
)

# %% cos( h_pre, h_mid )
plot_grid_combined(
    [("", metric_cos(h_pre_stack, h_mid_stack), PALETTE[14])],
    metric_name="cos( h_pre, h_mid ): how does Δ_attn change h (0.0, 1.0)",
    ylim=[0.0, 1.0],
)

# %% cos( h_mid, h_post )
plot_grid_combined(
    [("", metric_cos(h_mid_stack, h_post_stack), PALETTE[16])],
    metric_name="cos( h_mid, h_post ): how does Δ_mlp change h (0.0, 1.0)",
    ylim=[0.0, 1.0],
)

# %% Δ_attn norm relative to h_pre (-1.0, 1.0)
plot_grid_combined(
    [("", metric_norm_ratio(delta_attn_stack, h_pre_stack), PALETTE[18])],
    metric_name="Δ_attn norm relative to h_pre (-1.0, 1.0)",
    ylim=[-1.0, 1.0],
)

# %% Δ_attn norm relative to h_pre (-10.0, 10.0)
plot_grid_combined(
    [("", metric_norm_ratio(delta_attn_stack, h_pre_stack), PALETTE[18])],
    metric_name="Δ_attn norm relative to h_pre (-10.0, 10.0)",
    ylim=[-10.0, 10.0],
)

# %% Δ_mlp norm relative to h_mid (-1.0, 1.0)
plot_grid_combined(
    [("", metric_norm_ratio(delta_mlp_stack, h_mid_stack), PALETTE[0])],
    metric_name="Δ_mlp norm relative to h_mid (-1.0, 1.0)",
    ylim=[-1.0, 1.0],
)

# %% Δ_mlp norm relative to h_mid (-10.0, 10.0)
plot_grid_combined(
    [("", metric_norm_ratio(delta_mlp_stack, h_mid_stack), PALETTE[0])],
    metric_name="Δ_mlp norm relative to h_mid (-10.0, 10.0)",
    ylim=[-10.0, 10.0],
)
