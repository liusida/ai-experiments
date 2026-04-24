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

MODEL_ID = "google/gemma-4-E4B-it"

# Per-layer forward (see modeling_gemma4.Gemma4TextDecoderLayer.forward):
#
#   h_pre  ──►[input_ln ► self_attn ► post_attn_ln]────► Δ_attn
#   h_mid  = h_pre  + Δ_attn
#          ──►[pre_ffn_ln ► mlp ► post_ffn_ln]───────► Δ_mlp
#   h_mlp  = h_mid  + Δ_mlp
#          ──►[gate(h) * per_layer_input ► proj ► ln]► Δ_ple
#   h_ple  = h_mlp  + Δ_ple
#   h_post = h_ple * layer_scalar
#
# Five h's (h_pre/h_mid/h_mlp/h_ple/h_post), three Δ's (Δ_attn/Δ_mlp/Δ_ple),
# one scalar (layer_scalar).

QUESTIONS = [
    "What is the capital of France?",
    "What is the capital of Spain?",
    "What is the capital of Brazil?",
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
)
print(full_text)

inputs = tokenizer(full_text, return_tensors="pt").to(device)
token_ids = inputs["input_ids"][0].tolist()
token_pieces = [tokenizer.decode([t]) for t in token_ids]
seq_len = len(token_ids)

# Gemma 4 uses <|turn> as the per-turn delimiter (Gemma 2 used <start_of_turn>).
im_start_id = tokenizer.convert_tokens_to_ids("<|turn>")
assert im_start_id is not None and im_start_id != tokenizer.unk_token_id, (
    f"<|turn> not found in tokenizer for {MODEL_ID}"
)
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

# %% Forward pass with hooks: capture residual stream and submodule contributions.
# Hooks read the tensors that are actually added to the residual — i.e. the
# outputs of the *post* norms, not the raw attn/mlp outputs.
blocks = decoder_blocks(model)
num_layers = len(blocks)
print(f"num decoder layers: {num_layers}", flush=True)

h_pre_buf:      list[torch.Tensor | None] = [None] * num_layers
h_post_buf:     list[torch.Tensor | None] = [None] * num_layers
delta_attn_buf: list[torch.Tensor | None] = [None] * num_layers
delta_mlp_buf:  list[torch.Tensor | None] = [None] * num_layers
delta_ple_buf:  list[torch.Tensor | None] = [None] * num_layers


def make_block_pre_hook(i):
    def hook(_module, inputs):
        h_pre_buf[i] = _first(inputs).detach().to(torch.float32).cpu()
    return hook


def make_block_post_hook(i):
    def hook(_module, _inputs, output):
        h_post_buf[i] = _first(output).detach().to(torch.float32).cpu()
    return hook


def make_out_hook(buf, i):
    def hook(_module, _inputs, output):
        buf[i] = _first(output).detach().to(torch.float32).cpu()
    return hook


handles = []
for i, block in enumerate(blocks):
    handles.append(block.register_forward_pre_hook(make_block_pre_hook(i)))
    handles.append(block.register_forward_hook(make_block_post_hook(i)))
    handles.append(block.post_attention_layernorm.register_forward_hook(make_out_hook(delta_attn_buf, i)))
    handles.append(block.post_feedforward_layernorm.register_forward_hook(make_out_hook(delta_mlp_buf, i)))
    handles.append(block.post_per_layer_input_norm.register_forward_hook(make_out_hook(delta_ple_buf, i)))

try:
    with torch.no_grad():
        model(**inputs, use_cache=False)
finally:
    for h in handles:
        h.remove()

# Stack (num_layers, seq, hidden).
h_pre_stack      = torch.stack([h_pre_buf[i][0]      for i in range(num_layers)], dim=0)
h_post_stack     = torch.stack([h_post_buf[i][0]     for i in range(num_layers)], dim=0)
delta_attn_stack = torch.stack([delta_attn_buf[i][0] for i in range(num_layers)], dim=0)
delta_mlp_stack  = torch.stack([delta_mlp_buf[i][0]  for i in range(num_layers)], dim=0)
delta_ple_stack  = torch.stack([delta_ple_buf[i][0]  for i in range(num_layers)], dim=0)

# Per-layer scalar (shape (num_layers,)).
layer_scalar = torch.stack(
    [blocks[i].layer_scalar.detach().to(torch.float32).cpu().reshape(()) for i in range(num_layers)]
)

# Derived residual-stream snapshots.
h_mid_stack = h_pre_stack + delta_attn_stack              # after attn add
h_mlp_stack = h_mid_stack + delta_mlp_stack               # after mlp add
h_ple_stack = h_mlp_stack + delta_ple_stack               # after ple add  (== h_post / layer_scalar)

# Sanity: (h_pre + Δ_attn + Δ_mlp + Δ_ple) * layer_scalar ≈ h_post
_recon = h_ple_stack * layer_scalar.view(-1, 1, 1)
_err = (_recon - h_post_stack).abs().max().item()
_scale = h_post_stack.abs().mean().item()
print(f"reconstruction max-abs err: {_err:.2e}  (mean |h_post|: {_scale:.2e})", flush=True)

# Drop the first token (BOS acts as an attention sink → far off-scale). Sync metadata.
h_pre_stack      = h_pre_stack[:, 1:]
h_mid_stack      = h_mid_stack[:, 1:]
h_mlp_stack      = h_mlp_stack[:, 1:]
h_ple_stack      = h_ple_stack[:, 1:]
h_post_stack     = h_post_stack[:, 1:]
delta_attn_stack = delta_attn_stack[:, 1:]
delta_mlp_stack  = delta_mlp_stack[:, 1:]
delta_ple_stack  = delta_ple_stack[:, 1:]
token_ids    = token_ids[1:]
token_pieces = token_pieces[1:]
seq_len      = len(token_ids)
turn_starts  = [p - 1 for p in turn_starts if p - 1 >= 0]
turn_spans   = [
    (max(s - 1, 0), e - 1, r) for (s, e, r) in turn_spans if e - 1 > 0
]
print("stacks:", tuple(h_pre_stack.shape), "(h_pre/h_mid/h_mlp/h_ple/h_post + Δ_attn/Δ_mlp/Δ_ple)", flush=True)


# %% Plot helper (one cell per (feature, metric) pair below)
safe = hf_repo_id_safe_stem(MODEL_ID)
x_positions = np.arange(seq_len)


def _escape(p: str) -> str:
    """Make newlines / tabs visible in tick labels."""
    return p.replace("\n", "\\n").replace("\t", "\\t")


tick_labels = [_escape(p) for p in token_pieces]
pred_labels = [
    _escape(token_pieces[t + 1]) if t + 1 < seq_len else ""
    for t in range(seq_len)
]

PALETTE = plt.get_cmap("tab20").colors

def plot_grid_combined(
    series: list[tuple[str, np.ndarray, str]],
    metric_name: str,
    ylim: tuple[float],
) -> None:
    n_kinds = len(series)
    rows = n_kinds * num_layers
    fig, axes = plt.subplots(
        rows, 1,
        figsize=(seq_len / 10, rows * 0.35),
        sharex=True,
        gridspec_kw={"hspace": 0.03},
    )
    axes = np.asarray(axes).reshape(rows)
    y_min, y_max = ylim
    frame_color = "#bbbbbb"
    user_bg = "#aaaaaa"
    asst_bg = "#ffffff"
    for kind_idx, (kind, mat, color) in enumerate(series):
        row_base = kind_idx * num_layers
        for i in range(num_layers):
            ax = axes[row_base + i]
            for start, end, role in turn_spans:
                ax.axvspan(
                    start, end,
                    color=user_bg if role == "user" else asst_bg,
                    alpha=0.1, linewidth=0, zorder=0,
                )
            ax.axhline(0.0, color="#eeeeee", linestyle=":", linewidth=0.8)
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
            ax.tick_params(
                axis="x", labelbottom=False, labeltop=False,
                bottom=False, top=False,
            )
            ax.set_ylabel(f"{kind}\nL{i}", fontsize=7)
            ax.plot(x_positions, mat[i], "o-", color=color, markersize=1.3, linewidth=0.8)

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
    fig.subplots_adjust(
        left=0.07, right=0.99,
        top=1 - 0.8 / fig_h,
        bottom=0.6 / fig_h,
        hspace=0.03,
    )
    kinds_slug = "_".join(k.replace("Δ_", "delta_") for k, _, _ in series)
    stonesoup.show(fig, basename=f"{safe}_{kinds_slug}_{metric_name}", dpi=140)
    plt.close(fig)

# %% Generated-token h_pre / h_mid capture
"""Pass "Repeat <tok><tok><tok>", generate n tokens, assert they are all <tok>,
and return (h_pre, h_mid) at each layer at the generated positions.
"""
def generate_token_representation(token_str: str, n_generate: int = 8, n_repeat: int = 6):
    prompt = "Repeat," + token_str * n_repeat
    pids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    target_id = int(pids[0, -1])
    with torch.no_grad():
        gen = model.generate(pids, max_new_tokens=n_generate, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    new = gen[0, pids.shape[1]:].tolist()
    print("prompt:", repr(prompt), "→ generated:", repr(tokenizer.decode(new)),
          "pieces:", [tokenizer.decode([t]) for t in new])
    assert all(t == target_id for t in new), [tokenizer.decode([t]) for t in new]

    hs = []
    for i, b in enumerate(blocks):
        hs.append(b.register_forward_pre_hook(make_block_pre_hook(i)))
        hs.append(b.post_attention_layernorm.register_forward_hook(make_out_hook(delta_attn_buf, i)))
    try:
        with torch.no_grad():
            model(input_ids=gen, use_cache=False)
    finally:
        for h in hs: h.remove()

    pre = torch.stack([h_pre_buf[i][0] for i in range(num_layers)])
    mid = pre + torch.stack([delta_attn_buf[i][0] for i in range(num_layers)])
    sl = slice(pids.shape[1], gen.shape[1])
    return pre[:, sl].contiguous(), mid[:, sl].contiguous()


# %% Per-token metric vs the per-layer mean (swap METRIC to change the y-axis)
"""
================== !!! Below is the REAL JUICE !!! ==================

Nine variables per layer:
  h_pre, h_mid, h_mlp, h_ple, h_post,  Δ_attn, Δ_mlp, Δ_ple,  layer_scalar.
"""

def metric_norm_by_layer(stack: torch.Tensor) -> np.ndarray:
    normed = stack.norm(dim=-1)
    normed_by_layer = F.layer_norm(normed, normalized_shape=(normed.shape[-1],))
    return normed_by_layer


def metric_cos_to_mean(stack: torch.Tensor) -> np.ndarray:
    """Cosine similarity of each (layer, token) vector to the per-layer mean."""
    mean = stack.mean(dim=1)
    s_n = F.normalize(stack, dim=-1, eps=1e-8)
    m_n = F.normalize(mean,  dim=-1, eps=1e-8).unsqueeze(1)
    return (s_n * m_n).sum(dim=-1).numpy()

def metric_cos(stack_1: torch.Tensor, stack_2: torch.Tensor) -> np.ndarray:
    """Cos sim of two variables."""
    n_1 = F.normalize(stack_1, dim=-1, eps=1e-8)
    n_2 = F.normalize(stack_2, dim=-1, eps=1e-8)
    return (n_1 * n_2).sum(dim=-1).numpy()

def metric_norm_ratio(stack_1: torch.Tensor, stack_2: torch.Tensor) -> np.ndarray:
    """Ratio of norms of two variables."""
    n_1 = stack_1.norm(dim=-1)
    n_2 = stack_2.norm(dim=-1)
    return (n_1 / n_2).numpy()

def metric_cos_single_token(stack: torch.Tensor, token: torch.Tensor) -> np.ndarray:
    """Cosine similarity of each (layer, token) vector to a per-layer reference (L, H)."""
    s_n = F.normalize(stack, dim=-1, eps=1e-8)
    t_n = F.normalize(token, dim=-1, eps=1e-8).unsqueeze(1)
    return (s_n * t_n).sum(dim=-1).numpy()

# %% layer_scalar per layer
fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(np.arange(num_layers), layer_scalar.numpy(), "o-", color=PALETTE[14])
ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8)
ax.set_xlabel("layer"); ax.set_ylabel("layer_scalar")
ax.set_title(f"{MODEL_ID} — per-layer output scalar")
ax.grid(alpha=0.3)
stonesoup.show(fig, basename=f"{safe}_layer_scalar", dpi=140)
plt.close(fig)

# %% Δ_attn: cos_to_mean
plot_grid_combined(
    [("", metric_cos_to_mean(delta_attn_stack), PALETTE[0])],
    metric_name="2.4 Δ_attn cos_to_mean",
    ylim=[-1.0, 1.0]
)

# %% Δ_mlp: cos_to_mean
plot_grid_combined(
    [("", metric_cos_to_mean(delta_mlp_stack), PALETTE[2])],
    metric_name="3.4 Δ_mlp cos_to_mean",
    ylim=[-1.0, 1.0]
)

# %% Δ_ple: cos_to_mean
plot_grid_combined(
    [("", metric_cos_to_mean(delta_ple_stack), PALETTE[8])],
    metric_name="4.4 Δ_ple cos_to_mean",
    ylim=[-1.0, 1.0]
)

# %% h_pre: norm
plot_grid_combined(
    [("", metric_norm_by_layer(h_pre_stack), PALETTE[4])],
    metric_name="1.1 h_pre norm",
    ylim=[-3, 3]
)

# %% h_mid: norm
plot_grid_combined(
    [("", metric_norm_by_layer(h_mid_stack), PALETTE[6])],
    metric_name="1.2 h_mid norm",
    ylim=[-3, 3]
)

# %% h_mlp: norm
plot_grid_combined(
    [("", metric_norm_by_layer(h_mlp_stack), PALETTE[10])],
    metric_name="1.3 h_mlp norm",
    ylim=[-3, 3]
)

# %% h_ple: norm
plot_grid_combined(
    [("", metric_norm_by_layer(h_ple_stack), PALETTE[12])],
    metric_name="1.4 h_ple norm",
    ylim=[-3, 3]
)

# %% h_post: norm
plot_grid_combined(
    [("", metric_norm_by_layer(h_post_stack), PALETTE[16])],
    metric_name="1.5 h_post norm",
    ylim=[-3, 3]
)

# %% cos( Δ_attn, h_pre )
plot_grid_combined(
    [("", metric_cos(delta_attn_stack, h_pre_stack), PALETTE[10])],
    metric_name="2.2 Δ_attn cos( Δ_attn, h_pre ): does Δ amplify or remove",
    ylim=[-1.0, 1.0],
)

# %% cos( Δ_mlp, h_mid )
plot_grid_combined(
    [("", metric_cos(delta_mlp_stack, h_mid_stack), PALETTE[12])],
    metric_name="3.2 Δ_mlp cos( Δ_mlp, h_mid ): does Δ amplify or remove",
    ylim=[-1.0, 1.0],
)

# %% cos( Δ_ple, h_mlp )
plot_grid_combined(
    [("", metric_cos(delta_ple_stack, h_mlp_stack), PALETTE[18])],
    metric_name="4.2 Δ_ple cos( Δ_ple, h_mlp ): does Δ amplify or remove",
    ylim=[-1.0, 1.0],
)

# %% cos( h_pre, h_mid )
plot_grid_combined(
    [("", metric_cos(h_pre_stack, h_mid_stack), PALETTE[14])],
    metric_name="2.1 Δ_attn cos( h_pre, h_mid ): Overall, how does Δ_attn change h (0.0, 1.0)",
    ylim=[0.0, 1.0],
)

# %% cos( h_mid, h_mlp )
plot_grid_combined(
    [("", metric_cos(h_mid_stack, h_mlp_stack), PALETTE[16])],
    metric_name="3.1 Δ_mlp cos( h_mid, h_mlp ): Overall, how does Δ_mlp change h (0.0, 1.0)",
    ylim=[0.0, 1.0],
)

# %% cos( h_mlp, h_ple )
plot_grid_combined(
    [("", metric_cos(h_mlp_stack, h_ple_stack), PALETTE[4])],
    metric_name="4.1 Δ_ple cos( h_mlp, h_ple ): Overall, how does Δ_ple change h (0.0, 1.0)",
    ylim=[0.0, 1.0],
)

# %% Δ_attn norm relative to h_pre (0.0, 1.0)
plot_grid_combined(
    [("", metric_norm_ratio(delta_attn_stack, h_pre_stack), PALETTE[18])],
    metric_name="2.3 Δ_attn norm relative to h_pre (0.0, 1.0)",
    ylim=[0.0, 1.0],
)

# %% Δ_attn norm relative to h_pre (0.0, 20.0)
plot_grid_combined(
    [("", metric_norm_ratio(delta_attn_stack, h_pre_stack), PALETTE[18])],
    metric_name="2.3.1 Δ_attn norm relative to h_pre (0.0, 20.0)",
    ylim=[0.0, 20.0],
)

# %% Δ_mlp norm relative to h_mid (0.0, 1.0)
plot_grid_combined(
    [("", metric_norm_ratio(delta_mlp_stack, h_mid_stack), PALETTE[0])],
    metric_name="3.3 Δ_mlp norm relative to h_mid (0.0, 1.0)",
    ylim=[0.0, 1.0],
)

# %% Δ_mlp norm relative to h_mid (0.0, 5.0)
plot_grid_combined(
    [("", metric_norm_ratio(delta_mlp_stack, h_mid_stack), PALETTE[0])],
    metric_name="3.3.1 Δ_mlp norm relative to h_mid (0.0, 5.0)",
    ylim=[0.0, 5.0],
)

# %% Δ_ple norm relative to h_mlp (-1.0, 1.0)
plot_grid_combined(
    [("", metric_norm_ratio(delta_ple_stack, h_mlp_stack), PALETTE[2])],
    metric_name="4.3 Δ_ple norm relative to h_mlp (-1.0, 1.0)",
    ylim=[-1.0, 1.0],
)

# %% Δ_ple norm relative to h_mlp (0.0, 5.0)
plot_grid_combined(
    [("", metric_norm_ratio(delta_ple_stack, h_mlp_stack), PALETTE[2])],
    metric_name="4.3.1 Δ_ple norm relative to h_mlp (0.0, 5.0)",
    ylim=[0.0, 5.0],
)

# %% cos sim to Tokyo
stonesoup.html()
tokyo_h_pres, tokyo_h_mids = generate_token_representation(" Tokyo")
print("tokyo_h_pres:", tuple(tokyo_h_pres.shape), "tokyo_h_mids:", tuple(tokyo_h_mids.shape))

first_pos = -2
second_pos = -1
cos_pre = F.cosine_similarity(tokyo_h_pres[:, first_pos], tokyo_h_pres[:, second_pos], dim=-1).numpy()
cos_mid = F.cosine_similarity(tokyo_h_mids[:, first_pos], tokyo_h_mids[:, second_pos], dim=-1).numpy()
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(np.arange(num_layers), cos_pre, "o-", label="h_pre",  color=PALETTE[4])
ax.plot(np.arange(num_layers), cos_mid, "o-", label="h_mid",  color=PALETTE[6])
ax.set_xlabel("layer"); ax.set_ylabel("cos( Tokyo[0], Tokyo[1] )")
ax.set_title(f"{MODEL_ID} — cos sim between 1st and 2nd generated ' Tokyo' per layer")
ax.axhline(0.0, color="gray", linestyle=":", linewidth=0.2)
ax.grid(alpha=0.3); ax.legend()
stonesoup.show(fig, basename=f"{safe}_tokyo_cos_t0_t1", dpi=140)
plt.close(fig)

tokyo_h_pre, tokyo_h_mid = tokyo_h_pres[:, -1], tokyo_h_mids[:, -1]

plot_grid_combined(
    [("", metric_cos_single_token(h_pre_stack, tokyo_h_pre), PALETTE[16])],
    metric_name="6.1 cos to Tokyo",
    ylim=[0.0, 1.0],
)

# %% verify the cos sim between the token position that predicts " Tokyo" in h_pre_stack (one token earlier than the actual " Tokyo") and tokyo_h_pre. # stonesoup:cell-input
stonesoup.html()
offset = int(str(globals().get("CELL_INPUT", "") or "").strip() or "0")
_tid = tokenizer(" Tokyo", add_special_tokens=False, return_tensors="pt").input_ids[0, -1].item()
_pred_pos = [t for t in range(seq_len - 1) if token_ids[t + 1] == _tid]
fig, ax = plt.subplots(figsize=(8, 4))
for p in _pred_pos:
    p = p+1
    print(f"p={p} {token_pieces[p]!r}  p-offset={p-offset} {token_pieces[p-offset]!r}")
    ax.plot(F.cosine_similarity(h_pre_stack[:, p], h_pre_stack[:, p-offset], dim=-1).numpy(), "o-", label=f"p")
    ax.plot(F.cosine_similarity(h_pre_stack[:, p], tokyo_h_pre, dim=-1).numpy(), "o-", label=f"Tokyo")
ax.set_ylim((-1,1))
ax.set_xlabel("layer"); ax.set_ylabel("cos"); ax.grid(alpha=0.3); ax.legend(fontsize=7)
stonesoup.show(fig, basename=f"{safe}_verify_tokyo_cos")

# %% Observation Note
note = """

# Some Analysis of Gemma4-E4B-it

Architecturally:

* Uses an additional `PLE` submodule to remind the residual stream what the token was.

* Uses `layer scalar` to keep norm of residual stream at the same scale.

Behaviorally:

* L23 does something big, the residual stream before L23 and after L23 is almost completely different.

    Attention Submodule:

        * Attn at L0, L1, and L23 changes residual stream significantly (almost rewriting), while at some layers like L31, L39, and so on, barely touches the residual stream. (Might be redundancy.)

        * L23 is a global attention layer. (local:global = 5:1)

        * Attn copies `capital`, `France`, and `**` at later layer, but not very interesting overall.

        * Attn mostly adds information to the residual stream, not deleting.

    MLP Submodule:

        * MLP at L8, L17~L22, and also L23, and L41 (last layer) changes residual stream significantly (but not so extreme like Attn), while at L6, L12, L31 and so on barely touches it.

        * MLP at L41 last layer might be actually responsible for providing the actual answer Pairs, Madrid, Bazilia, and Tokyo. And it finalizes the token selection at L41 as well. Very important.

    PLE Submodule:

        * Overall, it doesn't change the residual stream much. Maybe not a very useful innovation.

"""
print(note)
