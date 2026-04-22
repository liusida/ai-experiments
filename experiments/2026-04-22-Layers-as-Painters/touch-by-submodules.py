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

MODEL_ID = "google/gemma-4-E2B-it"
MODEL_ID = "Qwen/Qwen3-0.6B"
PROMPT = "Hello!"


def last_token_norm(t: torch.Tensor) -> float:
    """L2 norm of the final-position vector: ``t`` is ``(1, seq, hidden)``."""
    return float(t[0, -1].norm().item())


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

# %% Build chat-templated prompt and tokenize
messages = [{"role": "user", "content": PROMPT}]
prompt_text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)
print(prompt_text)

inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
prefill_len = int(inputs["input_ids"].shape[1])
print("prefill length (tokens):", prefill_len, flush=True)

# %% Forward pass with hooks: capture residual stream + submodule contributions
blocks = decoder_blocks(model)
num_layers = len(blocks)
print(f"num decoder layers: {num_layers}", flush=True)

residual_in:  list[torch.Tensor | None] = [None] * num_layers  # block input
residual_out: list[torch.Tensor | None] = [None] * num_layers  # block output
attn_contrib: list[torch.Tensor | None] = [None] * num_layers  # self_attn out (pre-add)
mlp_contrib:  list[torch.Tensor | None] = [None] * num_layers  # mlp out (pre-add)


def make_block_hook(i):
    def hook(_module, inputs_, output):
        x_in = _first(inputs_[0] if isinstance(inputs_, tuple) else inputs_)
        x_out = _first(output)
        residual_in[i] = x_in.detach().to(torch.float32).cpu()
        residual_out[i] = x_out.detach().to(torch.float32).cpu()
    return hook


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
    handles.append(block.register_forward_hook(make_block_hook(i)))
    handles.append(block.self_attn.register_forward_hook(make_attn_hook(i)))
    handles.append(block.mlp.register_forward_hook(make_mlp_hook(i)))

try:
    with torch.no_grad():
        out = model(**inputs, use_cache=False)
finally:
    for h in handles:
        h.remove()

next_id = int(out.logits[0, -1].argmax().item())
print(
    "predicted first response token id:", next_id,
    "→", repr(tokenizer.decode([next_id])),
    flush=True,
)

# %% Greedy-generate up to 8 response tokens (for context; hooks already removed)
MAX_NEW_TOKENS = 8
with torch.no_grad():
    gen = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
response_ids = gen[0, inputs["input_ids"].shape[1]:].tolist()
response_pieces = [tokenizer.decode([t]) for t in response_ids]
response_text = tokenizer.decode(response_ids, skip_special_tokens=False)
print("response token ids:", response_ids)
print("response pieces   :", response_pieces)
print("response text     :", repr(response_text))

# Sanity check: the first greedy token should match the argmax from the hooked prefill.
assert response_ids[0] == next_id, (response_ids[0], next_id)

# %% Norms at the last prefill position (the state that drives the first response token)
layers = np.arange(num_layers)

pre_block_norms  = np.array([last_token_norm(residual_in[i])  for i in range(num_layers)])
post_block_norms = np.array([last_token_norm(residual_out[i]) for i in range(num_layers)])
attn_norms       = np.array([last_token_norm(attn_contrib[i]) for i in range(num_layers)])
mlp_norms        = np.array([last_token_norm(mlp_contrib[i])  for i in range(num_layers)])

# Derived: post-attn residual = pre-block residual + attn contribution (at last token).
post_attn_norms = np.array([
    float((residual_in[i][0, -1] + attn_contrib[i][0, -1]).norm().item())
    for i in range(num_layers)
])

# Relative magnitudes: how much each submodule rewrites the stream it reads from.
attn_rel = attn_norms / np.maximum(pre_block_norms, 1e-8)
mlp_rel  = mlp_norms  / np.maximum(post_attn_norms, 1e-8)

print("mean |attn| / |pre-block| :", float(attn_rel.mean()))
print("mean |mlp|  / |post-attn| :", float(mlp_rel.mean()))

# %% Plot norms over layers
safe = hf_repo_id_safe_stem(MODEL_ID)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

ax = axes[0]
ax.plot(layers, pre_block_norms,  "o-", color="#1f77b4", label="pre-block (residual in)")
ax.plot(layers, post_attn_norms,  "s-", color="#2ca02c", label="post-attn residual")
ax.plot(layers, post_block_norms, "^-", color="#d62728", label="post-mlp (block out)")
ax.set_xlabel("layer index")
ax.set_ylabel("L2 norm")
ax.set_title("Residual stream norm")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)

ax = axes[1]
ax.plot(layers, attn_norms, "s-", color="#9467bd", label="attn contribution")
ax.plot(layers, mlp_norms,  "^-", color="#ff7f0e", label="mlp contribution")
ax.set_xlabel("layer index")
ax.set_ylabel("L2 norm")
ax.set_title("Submodule contribution norm")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)

ax = axes[2]
ax.plot(layers, attn_rel, "s-", color="#9467bd", label="|attn| / |pre-block|")
ax.plot(layers, mlp_rel,  "^-", color="#ff7f0e", label="|mlp| / |post-attn|")
ax.axhline(1.0, color="gray", linestyle=":", linewidth=1)
ax.set_xlabel("layer index")
ax.set_ylabel("ratio")
ax.set_title("Relative contribution (|Δ| / |residual read|)")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)

fig.suptitle(
    f'{MODEL_ID} — norms at last prefill token (drives first response token)\n'
    f'prompt: "{PROMPT}"',
    fontsize=11,
)
fig.tight_layout()
stonesoup.show(fig, basename=f"{safe}_norms_last_prefill_token", dpi=140)
plt.close(fig)

# %% Cosine similarity of attention contributions across layers (heatmap)
import torch.nn.functional as F

# Per-layer attention contribution vector at the last prefill token: (num_layers, hidden).
attn_last = torch.stack([attn_contrib[i][0, -1] for i in range(num_layers)], dim=0)
cos_attn_last = (F.normalize(attn_last, dim=1, eps=1e-8)
                 @ F.normalize(attn_last, dim=1, eps=1e-8).T).numpy()

# Also mean across all prefill tokens — gives the "average direction" each layer writes.
attn_mean = torch.stack(
    [attn_contrib[i][0].mean(dim=0) for i in range(num_layers)], dim=0
)
cos_attn_mean = (F.normalize(attn_mean, dim=1, eps=1e-8)
                 @ F.normalize(attn_mean, dim=1, eps=1e-8).T).numpy()

labels = [f"L{i}" for i in range(num_layers)]
fig, axes = plt.subplots(1, 2, figsize=(0.3 * num_layers * 2 + 4, 0.3 * num_layers + 2))
for ax, mat, sub in [
    (axes[0], cos_attn_last, "last prefill token"),
    (axes[1], cos_attn_mean, "mean over prefill tokens"),
]:
    im = ax.imshow(mat, vmin=-1.0, vmax=1.0, cmap="coolwarm", aspect="equal")
    ax.set_xticks(np.arange(num_layers))
    ax.set_yticks(np.arange(num_layers))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_title(f"attn contribution cos-sim ({sub})")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="cosine similarity")

fig.suptitle(f'{MODEL_ID} — prompt: "{PROMPT}"', fontsize=11)
fig.tight_layout()
stonesoup.show(fig, basename=f"{safe}_attn_contrib_cosine", dpi=140)
plt.close(fig)
