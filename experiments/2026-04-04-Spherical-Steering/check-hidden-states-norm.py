# %% Load model

import stonesoup

model, processor = stonesoup.load_model("Qwen/Qwen3.5-2B")

# %% Prompt

PROMPT = "What is the spiciest part of a chili pepper? The spiciest part of a chili pepper is the placenta"

# %% Hidden state L2 norm per token, per layer

import torch

torch.set_printoptions(linewidth=200, sci_mode=False)
tokenizer = getattr(processor, "tokenizer", processor)
_device = next(model.parameters()).device

batch = tokenizer(PROMPT, return_tensors="pt", add_special_tokens=True)
batch = {k: v.to(_device) for k, v in batch.items()}

with torch.inference_mode():
    out = model(**batch, output_hidden_states=True)

# ``hidden_states[i]``: ``[batch, seq, hidden]`` — norm over ``hidden`` for each token.
for i, h in enumerate(out.hidden_states):
    per_token = h.float().norm(dim=-1).squeeze(0).cpu()
    print(f"layer {i:2d}  shape={tuple(h.shape)}  L2 per token: {per_token}")

# %% Heatmap: layer × token (L2 norm)

import matplotlib.pyplot as plt
import numpy as np
import torch

import stonesoup

_layer_token_norms = torch.stack(
    [h.float().norm(dim=-1).squeeze(0).cpu() for h in out.hidden_states],
    dim=0,
).numpy()

# Each row ÷ ‖h‖ at token 0 (same layer); first column is 1.0.
_ref = np.maximum(_layer_token_norms[:, 0:1], 1e-12)
_layer_token_norms_rel = _layer_token_norms / _ref

_fig, _ax = plt.subplots(figsize=(14, 6))
_im = _ax.imshow(_layer_token_norms_rel, aspect="auto", origin="lower", cmap="magma")
_ax.set_xlabel("token index")
_ax.set_ylabel("hidden_states index (0 = embeddings)")
_ax.set_title("‖h‖₂ at each token / ‖h‖₂ at token 0 (per layer)")
_fig.colorbar(_im, ax=_ax, label="relative L2 norm")
_fig.tight_layout()
stonesoup.show(_fig)

# %%
