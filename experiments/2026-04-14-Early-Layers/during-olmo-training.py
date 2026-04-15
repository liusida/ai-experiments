# %% Imports & config
from __future__ import annotations

import gc
import re

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F

import stonesoup
from stonesoup.experiment import (
    capture_embed_and_post_blocks,
    configure_matplotlib_agg,
    configure_matplotlib_unicode_fonts,
    apply_matplotlib_fonts_to_figure,
    decoder_blocks,
    ensure_pad_token_via_eos,
    hf_repo_id_safe_stem,
    inner_tokenizer,
    output_url_path,
)

configure_matplotlib_agg()
configure_matplotlib_unicode_fonts()

REPO_ID = "allenai/Olmo-3-1025-7B"

SECTIONS: list[tuple[str, str]] = [
    ("", "Chloroplasts use chlorophyll to absorb photons and store energy in ATP and NADPH."),
    ("", "The little boat wished for wings, so the wind stitched clouds into sails."),
    ("Chinese", "月球围绕地球公转，同一面始终朝向地球。"),
    ("English", "The Moon orbits the Earth, always showing the same face toward Earth."),
    ("Japanese", "月は地球の周りを公転しており、常に同じ面を地球に向けている。"),
    ("French", "La Lune orbite autour de la Terre, présentant toujours la même face."),
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
    "Japanese": "#984ea3",
    "French": "#a65628",
}


# %% Checkpoints to compare
# stage1 (pretraining, 5.93T tokens):  step0 … step1413814
# stage2 (midtraining, 100B tokens):   step0 … step47684
# stage3 (long context, 50B tokens):   step0 … step11921
chosen = [
    # "stage1-step0",
    # "stage1-step1000",
    # "stage1-step2000",
    # "stage1-step4000",
    # "stage1-step8000",
    # "stage1-step16000",
    # "stage1-step32000",
    # "stage1-step64000",
    "stage1-step128000",
    "stage1-step256000",
    "stage1-step512000",
    "stage1-step1024000",
    # "stage1-step1413814",
]

print(f"Chosen {len(chosen)} checkpoints:")
for c in chosen:
    print(f"  {c}")


# %% Tokenize prompt (shared across checkpoints)
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(REPO_ID, revision=chosen[0])
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

inputs_pt = {k: v for k, v in enc.items() if k != "offset_mapping"}

print(f"Prompt: {len(PROMPT)} chars → {seq_len} tokens")
print(f"Language blocks: {lang_blocks}")


# %% Load each checkpoint, capture activations, unload
from transformers import AutoModelForCausalLM

all_sims: dict[str, np.ndarray] = {}

for ci, revision in enumerate(chosen):
    stonesoup.check_abort()
    tag = revision
    print(f"[{ci+1}/{len(chosen)}] Loading {REPO_ID} @ {revision} …", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        REPO_ID,
        revision=revision,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=False,
    )
    model.eval()
    device = next(model.parameters()).device

    inputs_gpu = {k: v.to(device) for k, v in inputs_pt.items()}

    stack, stage_names = capture_embed_and_post_blocks(model, inputs_gpu, use_cache=False)
    acts = stack[:, 0, :seq_len, :].detach().float()

    post_idx = [i for i, n in enumerate(stage_names) if n.startswith("layer_")]
    sel_idx = [0] + post_idx
    sel_names = [stage_names[i] for i in sel_idx]

    mega = torch.cat([acts[i] for i in sel_idx], dim=0)
    normed = F.normalize(mega, dim=-1, eps=1e-8)
    sim = (normed @ normed.T).cpu().numpy()
    all_sims[tag] = sim

    del model, stack, acts, mega, normed, sim, inputs_gpu
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"  → captured {len(sel_names)} layers × {seq_len} tokens, unloaded.", flush=True)

n_layers_total = len(sel_names)
print(f"\nDone. {len(all_sims)} checkpoints captured, {n_layers_total} layers each, {seq_len} tokens.")


# %% Cross-layer RSM heatmaps — one figure per checkpoint
stonesoup.html()
block_size = seq_len
n_sel = n_layers_total

layer_bounds = [bi * block_size - 0.5 for bi in range(1, n_sel)]
lang_bounds = [bi * block_size + b for bi in range(n_sel) for b in section_boundaries]
y_tick_pos = [(bi + 0.5) * block_size for bi in range(n_sel)]
y_tick_labels = sel_names

x_tick_pos: list[float] = []
x_tick_labels: list[str] = []
x_tick_clrs: list[str] = []
for bi in range(n_sel):
    off = bi * block_size
    for label, first, last in lang_blocks:
        x_tick_pos.append(off + (first + last) / 2.0)
        x_tick_labels.append(label)
        x_tick_clrs.append(LANG_COLORS.get(label, "black"))

short = REPO_ID.split("/")[-1]

for tag, sim in all_sims.items():
    stonesoup.check_abort()

    step_match = re.search(r"step(\d+)", tag)
    step_str = f"step {step_match.group(1)}" if step_match else tag
    stage_match = re.search(r"(stage\d)", tag)
    stage_str = stage_match.group(1) if stage_match else ""
    title = f"{stage_str} {step_str}".strip()

    fig_size = max(10, n_sel * block_size * 0.02)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    ax.imshow(sim, vmin=0, vmax=1, cmap="Blues", aspect="equal", interpolation="none")

    for b in layer_bounds:
        ax.axhline(b, color="black", linewidth=1.0)
        ax.axvline(b, color="black", linewidth=1.0)
    for b in lang_bounds:
        ax.axhline(b, color="white", linewidth=0.2, alpha=0.5)
        ax.axvline(b, color="white", linewidth=0.2, alpha=0.5)

    ax.set_yticks(y_tick_pos)
    ax.set_yticklabels(y_tick_labels, fontsize=6)
    ax.set_xticks(x_tick_pos)
    ax.set_xticklabels(x_tick_labels, rotation=90, fontsize=5)
    for tl, c in zip(ax.get_xticklabels(), x_tick_clrs):
        tl.set_color(c)
        tl.set_fontweight("bold")

    ax.set_title(f"Cross-layer RSM — {short} — {title}", fontsize=13, pad=10)
    apply_matplotlib_fonts_to_figure(fig)
    fig.tight_layout()
    stonesoup.show(fig, basename=f"rsm_{hf_repo_id_safe_stem(REPO_ID)}_{tag}", dpi=120)
