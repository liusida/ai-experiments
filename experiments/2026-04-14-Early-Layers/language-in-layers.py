# %% Imports & config
from __future__ import annotations
from re import S

import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F

import stonesoup
from stonesoup.experiment import (
    configure_matplotlib_agg,
    decoder_blocks,
    ensure_pad_token_via_eos,
    hf_repo_id_safe_stem,
    inner_tokenizer,
)

configure_matplotlib_agg()

# MODEL = "google/gemma-2-2b"
MODEL = "tiiuae/falcon-7b"

SECTIONS: list[tuple[str, str]] = [
    # Unrelated English prefix
    ("", "Chloroplasts use chlorophyll to absorb photons and store energy in ATP and NADPH."),
    ("", "The little boat wished for wings, so the wind stitched clouds into sails."),
    # Same meaning in multiple languages
    ("Chinese", "月球围绕地球公转，同一面始终朝向地球。"),
    ("English", "The Moon orbits the Earth, always showing the same face toward Earth."),
    # ("Hebrew", "הירח סובב סביב כדור הארץ, כשאותו צד פונה תמיד לכיוון כדור הארץ."),
    ("Japanese", "月は地球の周りを公転しており、常に同じ面を地球に向けている。"),
    # ("Hindi", "चंद्रमा पृथ्वी की परिक्रमा करता है।"),
    ("French", "La Lune orbite autour de la Terre, présentant toujours la même face."),
    # Unrelated English suffix
    ("", "Where a party fails to perform, the non-breaching party may seek damages as provided herein."),
    ("", "Patient reports intermittent vertigo; differential includes BPPV versus orthostatic hypotension."),
]

prompt_pieces: list[str] = []
section_char_ranges: list[tuple[int, int, str]] = []
pos = 0
for label, text in SECTIONS:
    if prompt_pieces:
        prompt_pieces.append(" ")
        pos += 1
    start = pos
    prompt_pieces.append(text)
    pos += len(text)
    section_char_ranges.append((start, pos, label))

PROMPT = "".join(prompt_pieces)
MAX_LENGTH = 512

LANG_COLORS: dict[str, str] = {
    "Chinese": "#e41a1c",
    "English": "#377eb8",
    "Hebrew": "#4daf4a",
    "Japanese": "#984ea3",
    "Hindi": "#ff7f00",
    "French": "#a65628",
}


# %% Tokenize & capture activations

def _mid_hook_target(block: torch.nn.Module) -> torch.nn.Module | None:
    if hasattr(block, "pre_feedforward_layernorm"):
        return block.pre_feedforward_layernorm
    if hasattr(block, "post_attention_layernorm"):
        return block.post_attention_layernorm
    return None


def capture_embed_mid_post(
    model: torch.nn.Module, inputs: dict, **forward_kw,
) -> tuple[torch.Tensor, list[str]]:
    blocks = decoder_blocks(model)
    captured: list[torch.Tensor] = []

    def _post(_mod, _inp, x):
        captured.append((x[0] if isinstance(x, tuple) else x).detach())

    def _pre(_mod, inp):
        captured.append((inp[0] if isinstance(inp, tuple) else inp).detach())

    emb = model.get_input_embeddings()
    hooks = [emb.register_forward_hook(_post)]
    names = ["embed"]
    for i, block in enumerate(blocks):
        mid = _mid_hook_target(block)
        if mid is not None:
            hooks.append(mid.register_forward_pre_hook(_pre))
            names.append(f"L{i}_mid")
        hooks.append(block.register_forward_hook(_post))
        names.append(f"L{i}_post")
    try:
        with torch.inference_mode():
            model(**{**inputs, **forward_kw})
    finally:
        for h in hooks:
            h.remove()
    if len(captured) != len(names):
        raise RuntimeError(f"expected {len(names)} stages, got {len(captured)}")
    return torch.stack(captured, dim=0), names


model, proc = stonesoup.load_model(MODEL)
model.eval()
device = next(model.parameters()).device
tok = inner_tokenizer(proc)
ensure_pad_token_via_eos(tok)

enc = tok(
    PROMPT,
    return_offsets_mapping=True,
    return_tensors="pt",
    return_attention_mask=True,
    add_special_tokens=True,
    max_length=MAX_LENGTH,
    truncation=True,
)
seq_len = int(enc["attention_mask"][0].sum().item())
offsets = enc["offset_mapping"][0]

# Map each token to its section index
token_section: list[int] = []
for t in range(seq_len):
    a, b = int(offsets[t][0]), int(offsets[t][1])
    if b <= a:
        token_section.append(-1)
        continue
    mid_char = (a + b) / 2
    assigned = -1
    for si, (s_start, s_end, _) in enumerate(section_char_ranges):
        if s_start <= mid_char < s_end:
            assigned = si
            break
    token_section.append(assigned)

# Identify language block token ranges (for ticks) and all section boundaries (for lines)
lang_blocks: list[tuple[str, int, int]] = []
prev_si = -2
block_start = 0
for t in range(seq_len):
    si = token_section[t]
    if si != prev_si:
        if prev_si >= 0 and section_char_ranges[prev_si][2]:
            lang_blocks.append((section_char_ranges[prev_si][2], block_start, t - 1))
        block_start = t
        prev_si = si
if prev_si >= 0 and section_char_ranges[prev_si][2]:
    lang_blocks.append((section_char_ranges[prev_si][2], block_start, seq_len - 1))

section_boundaries: list[float] = []
for t in range(1, seq_len):
    if token_section[t] != token_section[t - 1]:
        section_boundaries.append(t - 0.5)

tick_pos = [(first + last) / 2.0 for _, first, last in lang_blocks]
tick_text = [label for label, _, _ in lang_blocks]
tick_colors = [LANG_COLORS.get(label, "black") for label, _, _ in lang_blocks]

# Forward pass
inputs_gpu = {k: v.to(device) for k, v in enc.items() if k != "offset_mapping"}
stack, stage_names = capture_embed_mid_post(model, inputs_gpu, use_cache=False)
acts = stack[:, 0, :seq_len, :].detach().float()

print(f"Prompt: {len(PROMPT)} chars, {seq_len} tokens, {acts.shape[0]} stages", flush=True)
print(f"Language blocks: {lang_blocks}", flush=True)

# %% Helper function

def _decorate_rsm(ax: plt.Axes, *, small: bool = True) -> None:
    fontsize = 5 if small else 9
    lw = 0.4 if small else 0.8
    for b in section_boundaries:
        ax.axhline(b, color="white", linewidth=lw, alpha=0.8)
        ax.axvline(b, color="white", linewidth=lw, alpha=0.8)
    ax.set_xticks(tick_pos)
    ax.set_yticks(tick_pos)
    ax.set_xticklabels(tick_text, rotation=90, fontsize=fontsize)
    ax.set_yticklabels(tick_text, fontsize=fontsize)
    for tl, c in zip(ax.get_xticklabels(), tick_colors):
        tl.set_color(c)
        tl.set_fontweight("bold")
    for tl, c in zip(ax.get_yticklabels(), tick_colors):
        tl.set_color(c)
        tl.set_fontweight("bold")


# %% RSM for L18_post
si = stage_names.index("L18_post")
normed_single = F.normalize(acts[si], dim=-1, eps=1e-8)
sim_single = (normed_single @ normed_single.T).cpu().numpy()
fig_size = max(8, seq_len * 0.08)
fig, ax = plt.subplots(figsize=(fig_size, fig_size))
ax.imshow(sim_single, vmin=0, vmax=1, cmap="Blues", aspect="equal", interpolation="none")
_decorate_rsm(ax, small=False)
short = MODEL.split("/")[-1]
ax.set_title(f"Token RSM (cosine sim) — {short} L18_post", fontsize=13, pad=10)
cb = fig.colorbar(ax.images[0], ax=ax, fraction=0.03, pad=0.02)
cb.set_label("cosine similarity", fontsize=10)
fig.tight_layout()
stonesoup.show(fig, basename=f"rsm_L18_post_{hf_repo_id_safe_stem(MODEL)}", dpi=144)


# %% RSM grid (without mid-layers)

post_idx = [i for i, n in enumerate(stage_names) if not n.endswith("_mid")]
post_acts = acts[post_idx]
post_names = [stage_names[i] for i in post_idx]

n_post = len(post_idx)
n_cols = 5
n_rows = -(-n_post // n_cols)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.8, n_rows * 2.8))
axes_flat = axes.flatten()

for i in range(n_post):
    stonesoup.check_abort()
    normed = F.normalize(post_acts[i], dim=-1, eps=1e-8)
    sim = (normed @ normed.T).cpu().numpy()
    ax = axes_flat[i]
    ax.imshow(sim, vmin=0, vmax=1, cmap="Blues", aspect="equal", interpolation="none")
    ax.set_title(post_names[i], fontsize=7)
    _decorate_rsm(ax, small=True)

for i in range(n_post, len(axes_flat)):
    axes_flat[i].set_visible(False)

short = MODEL.split("/")[-1]
fig.suptitle(
    f"Token RSM (cosine sim) along residual stream — {short}",
    fontsize=14,
    y=1.01,
)
fig.tight_layout()
stonesoup.show(fig, basename=f"rsm_lang_layers_{hf_repo_id_safe_stem(MODEL)}", dpi=120)

# %% Cross-layer mega RSM
stonesoup.html()
SELECTED_STAGES = [
    "embed", "L0_post", "L2_post", "L5_post", "L12_post",
    "L21_post", "L23_post", "L25_post",
]
print(stage_names)
SELECTED_STAGES = stage_names[::4]
# SELECTED_STAGES += ["L34_post"]
print(SELECTED_STAGES)

sel_idx = [stage_names.index(s) for s in SELECTED_STAGES]

mega = torch.cat([acts[i] for i in sel_idx], dim=0)
normed_mega = F.normalize(mega, dim=-1, eps=1e-8)
sim_mega = (normed_mega @ normed_mega.T).cpu().numpy()

n_sel = len(SELECTED_STAGES)
block_size = seq_len

layer_bounds = [bi * block_size - 0.5 for bi in range(1, n_sel)]
lang_bounds = [bi * block_size + b for bi in range(n_sel) for b in section_boundaries]

y_tick_pos = [(bi + 0.5) * block_size for bi in range(n_sel)]

x_tick_pos: list[float] = []
x_tick_labels: list[str] = []
x_tick_clrs: list[str] = []
for bi in range(n_sel):
    off = bi * block_size
    for label, first, last in lang_blocks:
        x_tick_pos.append(off + (first + last) / 2.0)
        x_tick_labels.append(label)
        x_tick_clrs.append(LANG_COLORS.get(label, "black"))

total = n_sel * block_size
fig_size = max(12, total * 0.06)
fig, ax = plt.subplots(figsize=(fig_size, fig_size))
ax.imshow(sim_mega, vmin=0, vmax=1, cmap="Blues", aspect="equal", interpolation="none")

for b in layer_bounds:
    ax.axhline(b, color="black", linewidth=1.5)
    ax.axvline(b, color="black", linewidth=1.5)
for b in lang_bounds:
    ax.axhline(b, color="white", linewidth=0.3, alpha=0.6)
    ax.axvline(b, color="white", linewidth=0.3, alpha=0.6)

ax.set_yticks(y_tick_pos)
ax.set_yticklabels(SELECTED_STAGES, fontsize=9, fontweight="bold")
ax.set_xticks(x_tick_pos)
ax.set_xticklabels(x_tick_labels, rotation=90, fontsize=5)
for tl, c in zip(ax.get_xticklabels(), x_tick_clrs):
    tl.set_color(c)
    tl.set_fontweight("bold")

short = MODEL.split("/")[-1]
ax.set_title(f"Cross-layer token RSM (cosine sim) — {short}", fontsize=14, pad=12)
# cb = fig.colorbar(ax.images[0], ax=ax, fraction=0.025, pad=0.02)
# cb.set_label("cosine similarity", fontsize=10)
fig.tight_layout()
stonesoup.show(fig, basename=f"rsm_cross_layer_{hf_repo_id_safe_stem(MODEL)}", dpi=120)
