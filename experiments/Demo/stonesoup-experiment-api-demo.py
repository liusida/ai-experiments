# %% Imports and matplotlib (Agg)
from __future__ import annotations

import stonesoup
from stonesoup.experiment import configure_matplotlib_agg

configure_matplotlib_agg()  # before pyplot — non-interactive backend

import matplotlib.pyplot as plt
import numpy as np
import torch

from stonesoup.experiment import (
    STONESOUP_RENDER_HTML,
    capture_embed_and_post_blocks,
    capture_pre_block0_and_post_blocks,
    encode_text_inputs,
    ensure_pad_token_via_eos,
    hf_repo_id_safe_stem,
    inner_tokenizer,
    output_url_path,
)

# %% Config — swap this Hub id for a model you have cached (default is small GPT-2)
MODEL_ID = "openai-community/gpt2"
DEMO_PROMPT = "The capital of France is"

# %% Paths — repo layout and per-script outputs (HTTP /outputs/…)
root = stonesoup.repo_root()
out_dir = stonesoup.outputs_dir()
script_folder = stonesoup.script_dir()
print("repo_root:", root, flush=True)
print("outputs_dir (web):", out_dir.relative_to(root), flush=True)
print("script_dir:", script_folder.relative_to(root), flush=True)

# %% Safe filename stem for Hub ids (plots, basenames)
safe_stem = hf_repo_id_safe_stem(MODEL_ID)
example_stem = hf_repo_id_safe_stem("Org/Model:tag")
print("safe stem:", safe_stem, flush=True)
print("example Org/Model:tag →", example_stem, flush=True)

# %% Load model from the Stonesoup shared pool (toolbar Load or first cell load)
model, proc = stonesoup.load_model(MODEL_ID)
model.eval()
tokenizer = inner_tokenizer(proc)
ensure_pad_token_via_eos(tokenizer)
device = next(model.parameters()).device
print("device:", device, "dtype:", next(model.parameters()).dtype, flush=True)

# %% Encode text and capture embedding + post-block hidden states
inputs = encode_text_inputs(proc, DEMO_PROMPT, device=device)
stack_embed, names_embed = capture_embed_and_post_blocks(model, inputs, use_cache=False)
print("capture_embed_and_post_blocks:", stack_embed.shape, "names:", names_embed[:3], "…", flush=True)

# %% Alternate capture: pre-block-0 inputs vs post each block (different stage-0 semantics)
stack_pre, names_pre = capture_pre_block0_and_post_blocks(model, inputs, use_cache=False)
print("capture_pre_block0_and_post_blocks:", stack_pre.shape, "names:", names_pre[:3], "…", flush=True)
# Stage 0 here is the tensor entering decoder block 0; stage 0 in the previous cell is the embedding output.

# %% Plot last-token hidden norm per stage → stonesoup.show with a safe basename
last_token_norms = stack_embed[:, 0, -1, :].float().norm(dim=-1).cpu().numpy()
fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(np.arange(len(last_token_norms)), last_token_norms, marker="o", ms=3)
ax.set_xlabel("stage index (0 = embedding)")
ax.set_ylabel("L2 norm (last token)")
ax.set_title("Demo: norms across stack")
stonesoup.show(fig, basename=f"{safe_stem}_demo_last_token_norm", dpi=120)

# %% Hand-built HTML: write a PNG and reference it via output_url_path
manual_png = out_dir / f"{safe_stem}_demo_manual.png"
fig2, ax2 = plt.subplots(figsize=(4, 2))
ax2.bar(np.arange(min(8, len(last_token_norms))), last_token_norms[:8])
fig2.savefig(manual_png, dpi=100, bbox_inches="tight")
plt.close(fig2)
img_url = output_url_path(manual_png, cache_bust=True)
print(STONESOUP_RENDER_HTML, end="", flush=True)
print(
    f'<p class="stonesoup-show"><img src="{img_url}" alt="manual png" '
    'style="max-width:100%;height:auto" /></p>',
    flush=True,
)

# %% Cooperative abort — call inside long loops (toolbar Abort)
for step in range(3):
    stonesoup.check_abort()
print("Abort demo loop finished (no cancel).", flush=True)

# %% Introspection — bindings in this kernel (optional)
loaded_models = stonesoup.list_loaded_models()
print("list_loaded_models (this file):", loaded_models, flush=True)
