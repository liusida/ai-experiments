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

MODEL_ID = "Qwen/Qwen3-0.6B"
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
model, proc = stonesoup.load_model(MODEL_ID)
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
        enable_thinking=False,
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
    enable_thinking=False,
)
print(full_text)

inputs = tokenizer(full_text, return_tensors="pt").to(device)
token_ids = inputs["input_ids"][0].tolist()
token_pieces = [tokenizer.decode([t]) for t in token_ids]
seq_len = len(token_ids)

# Turn-start boundaries = positions of <|im_start|> tokens in the actual input.
im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
turn_starts = [i for i, t in enumerate(token_ids) if t == im_start_id]
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

# %% Forward pass with hooks: capture attn and mlp submodule contributions
blocks = decoder_blocks(model)
num_layers = len(blocks)
print(f"num decoder layers: {num_layers}", flush=True)

attn_contrib: list[torch.Tensor | None] = [None] * num_layers  # self_attn out (pre-add)
mlp_contrib:  list[torch.Tensor | None] = [None] * num_layers  # mlp out (pre-add)


def make_attn_hook(i):
    def hook(_module, _inputs, output):
        attn_contrib[i] = _first(output).detach().to(torch.float32).cpu()
    return hook


def make_mlp_hook(i):
    def hook(_module, _inputs, output):
        mlp_contrib[i] = _first(output).detach().to(torch.float32).cpu()
    return hook


handles = []
for i, block in enumerate(blocks):
    handles.append(block.self_attn.register_forward_hook(make_attn_hook(i)))
    handles.append(block.mlp.register_forward_hook(make_mlp_hook(i)))

try:
    with torch.no_grad():
        model(**inputs, use_cache=False)
finally:
    for h in handles:
        h.remove()

# Stack into (num_layers, seq, hidden) tensors.
attn_stack = torch.stack([attn_contrib[i][0] for i in range(num_layers)], dim=0)
mlp_stack  = torch.stack([mlp_contrib[i][0]  for i in range(num_layers)], dim=0)
print("attn_stack:", tuple(attn_stack.shape), "mlp_stack:", tuple(mlp_stack.shape), flush=True)

# %% Per-token metric vs the per-layer mean (swap METRIC to change the y-axis)
def metric_cos_to_mean(stack: torch.Tensor) -> np.ndarray:
    """Cosine similarity of each (layer, token) vector to the per-layer mean."""
    mean = stack.mean(dim=1)                                 # (L, H)
    s_n = F.normalize(stack, dim=-1, eps=1e-8)               # (L, T, H)
    m_n = F.normalize(mean,  dim=-1, eps=1e-8).unsqueeze(1)  # (L, 1, H)
    return (s_n * m_n).sum(dim=-1).numpy()                   # (L, T)


def metric_diff_norm(stack: torch.Tensor) -> np.ndarray:
    """L2 norm of (token vector - per-layer mean over tokens)."""
    mean = stack.mean(dim=1, keepdim=True)                   # (L, 1, H)
    return (stack - mean).norm(dim=-1).numpy()               # (L, T)


def metric_diff_norm_ln(stack: torch.Tensor) -> np.ndarray:
    """LayerNorm each vector (zero-mean / unit-var across hidden), then diff norm.

    LayerNorm puts every (layer, token) vector on the same scale, so the per-layer
    magnitude growth with depth is factored out before comparing.
    """
    normed = F.layer_norm(stack, normalized_shape=(stack.shape[-1],))
    mean = normed.mean(dim=1, keepdim=True)                  # (L, 1, H)
    return (normed - mean).norm(dim=-1).numpy()              # (L, T)


def metric_diff_norm_rel(stack: torch.Tensor) -> np.ndarray:
    """Diff norm divided by per-layer mean token-norm → dimensionless."""
    mean = stack.mean(dim=1, keepdim=True)                   # (L, 1, H)
    diff = (stack - mean).norm(dim=-1)                       # (L, T)
    scale = stack.norm(dim=-1).mean(dim=1, keepdim=True)     # (L, 1)
    return (diff / scale.clamp(min=1e-8)).numpy()


METRIC = metric_cos_to_mean        # swap to metric_diff_norm / _ln / _rel
# METRIC = metric_diff_norm_ln
METRIC_NAME = METRIC.__name__.removeprefix("metric_")

attn_y = METRIC(attn_stack)  # (L, T)
mlp_y  = METRIC(mlp_stack)

# %% Plot per-layer cos-sim across token positions
safe = hf_repo_id_safe_stem(MODEL_ID)
x_positions = np.arange(seq_len)
# Escape/clean token text for tick labels (show newlines/special chars visibly).
tick_labels = [p.replace("\n", "\\n").replace("\t", "\\t") for p in token_pieces]


def plot_grid_combined(
    attn_mat: np.ndarray,
    mlp_mat: np.ndarray,
    attn_color: str = "#9467bd",
    mlp_color: str = "#ff7f0e",
) -> None:
    # Blocked order: all attn layers first (L0..Llast), then all mlp layers.
    rows = 2 * num_layers
    fig, axes = plt.subplots(
        rows, 1,
        figsize=(seq_len / 10, rows * 0.35),
        sharex=True,
        gridspec_kw={"hspace": 0.03},
    )
    axes = np.asarray(axes).reshape(rows)
    # Clip y-range to mean ± 3·std so outliers don't flatten the rest of the plot.
    combined = np.concatenate([attn_mat.ravel(), mlp_mat.ravel()])
    mu = float(combined.mean())
    sd = float(combined.std())
    y_min = mu - 3 * sd
    y_max = mu + 3 * sd
    frame_color = "#bbbbbb"
    blocks_spec = [
        ("attn", attn_mat, attn_color, 0),
        ("mlp",  mlp_mat,  mlp_color,  num_layers),
    ]
    user_bg = "#aaaaaa"
    asst_bg = "#ffffff"
    for kind, mat, color, row_base in blocks_spec:
        for i in range(num_layers):
            ax = axes[row_base + i]
            # Shade per-turn background by role (behind the data).
            for start, end, role in turn_spans:
                ax.axvspan(
                    start - 0.5, end - 0.5,
                    color=user_bg if role == "user" else asst_bg,
                    alpha=0.05, linewidth=0, zorder=0,
                )
            ax.plot(x_positions, mat[i], "o-", color=color, markersize=2, linewidth=0.8)
            ax.axhline(0.0, color="gray", linestyle=":", linewidth=0.8)
            for pos in turn_starts:
                ax.axvline(pos - 0.5, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
            ax.set_ylim(y_min, y_max)
            ax.grid(alpha=0.2, linewidth=0.3)
            ax.set_xticks(x_positions)
            ax.set_xticklabels(tick_labels, rotation=90, fontsize=5)
            for spine in ax.spines.values():
                spine.set_linewidth(0.4)
                spine.set_color(frame_color)
            ax.tick_params(colors="#888888", width=0.4, length=2, labelsize=5)
            ax.set_ylabel(f"{kind}\nL{i}", fontsize=7)

    fig.suptitle(
        f'{MODEL_ID} — attn & mlp contribution: {METRIC_NAME}(per-token vs layer-mean)',
        fontsize=11,
    )
    fig_h = rows * 0.35
    fig.subplots_adjust(
        left=0.07, right=0.99,
        top=1 - 0.75 / fig_h,
        bottom=0.6 / fig_h,
        hspace=0.03,
    )
    stonesoup.show(fig, basename=f"{safe}_contrib_{METRIC_NAME}_vs_mean", dpi=140)
    plt.close(fig)


plot_grid_combined(attn_y, mlp_y)
