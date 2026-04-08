# %% Imports & model
from __future__ import annotations

import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import stonesoup

MODEL_ID = "Qwen/Qwen3-8B-Base"
NUM_SAMPLES = 200
SEED = 0

model, tokenizer = stonesoup.load_model(MODEL_ID)
model.eval()
device = next(model.parameters()).device
inner_tok = getattr(tokenizer, "tokenizer", tokenizer)
MODEL_BASENAME = MODEL_ID.replace("/", "__").replace(":", "-")
print("Loaded:", MODEL_ID, device, flush=True)

# %% Forward: all hidden stages (pre block 0 + post each block)


def decoder_blocks(m: torch.nn.Module) -> torch.nn.ModuleList:
    if hasattr(m, "transformer") and hasattr(m.transformer, "h"):
        return m.transformer.h
    if hasattr(m, "model"):
        inner = m.model
        if hasattr(inner, "language_model") and hasattr(inner.language_model, "layers"):
            return inner.language_model.layers
        if hasattr(inner, "layers"):
            return inner.layers
    raise TypeError(f"No decoder stack on {type(m).__name__}")


def run_hidden_streams(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """(num_stages, seq_len, hidden) — stage 0 = pre block 0."""
    blocks = decoder_blocks(model)
    captured: list[torch.Tensor] = []

    def pre0(_mod, inputs: tuple) -> None:
        captured.append(inputs[0].detach())

    def post(_mod, _inp, out: torch.Tensor | tuple) -> None:
        captured.append((out[0] if isinstance(out, tuple) else out).detach())

    hooks = [blocks[0].register_forward_pre_hook(pre0)]
    hooks += [layer.register_forward_hook(post) for layer in blocks]
    try:
        with torch.inference_mode():
            model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    finally:
        for h in hooks:
            h.remove()
    stacked = torch.stack(captured, dim=0)
    if stacked.dim() == 4 and stacked.shape[1] == 1:
        stacked = stacked.squeeze(1)
    return stacked


# %% Sample: two different prompts, random positions, cosine per layer
SENTENCES = [
    "The bank closed before noon.",
    "She played bass in a jazz quartet.",
    "Minute details matter in watch repair.",
    "I need to deposit cash at the counter.",
    "Wikipedia documents historical events neutrally.",
    "Python lists append in amortized constant time.",
    "Neural networks learn representations from data.",
    "The patient showed minute improvement overnight.",
    "Engineers object to the proposed zoning map.",
    "Snow covered the river bank by morning.",
]

rng = random.Random(SEED)
n_stages: int | None = None
cos_samples: list[list[float]] = []

for s in range(NUM_SAMPLES):
    stonesoup.check_abort()
    i_a, i_b = rng.sample(range(len(SENTENCES)), 2)
    text_a, text_b = SENTENCES[i_a], SENTENCES[i_b]
    enc_a = inner_tok(
        text_a, return_tensors="pt", return_attention_mask=True, add_special_tokens=True
    )
    enc_b = inner_tok(
        text_b, return_tensors="pt", return_attention_mask=True, add_special_tokens=True
    )
    la = int(enc_a["input_ids"].shape[1])
    lb = int(enc_b["input_ids"].shape[1])
    if la < 2 or lb < 2:
        continue
    pa, pb = rng.randrange(la), rng.randrange(lb)
    ha = run_hidden_streams(
        enc_a["input_ids"].to(device),
        enc_a["attention_mask"].to(device),
    )
    hb = run_hidden_streams(
        enc_b["input_ids"].to(device),
        enc_b["attention_mask"].to(device),
    )
    if n_stages is None:
        n_stages = int(ha.shape[0])
    elif int(ha.shape[0]) != n_stages:
        raise RuntimeError(f"Inconsistent stage count: {ha.shape[0]} vs {n_stages}")
    row: list[float] = []
    va = ha[:, pa].float()
    vb = hb[:, pb].float()
    for li in range(n_stages):
        row.append(
            F.cosine_similarity(va[li : li + 1], vb[li : li + 1], dim=-1).item()
        )
    cos_samples.append(row)

cos_mat = np.asarray(cos_samples, dtype=np.float64)
print(f"samples={cos_mat.shape[0]} layers={cos_mat.shape[1]}", flush=True)

# %% Plot mean ± std of cosine vs layer
assert n_stages is not None
x = np.arange(n_stages)
mean = cos_mat.mean(axis=0)
std = cos_mat.std(axis=0)
fig, ax = plt.subplots(figsize=(11, 4.2))
ax.plot(x, mean, color="0.2", linewidth=1.2, label="mean cos")
ax.fill_between(x, mean - std, mean + std, color="0.45", alpha=0.28, label="±1 std")
ax.set_xlabel("layer stage (0 = pre block 0, k = post block k-1)")
ax.set_ylabel("cosine similarity")
ax.set_title(
    f"{MODEL_ID}\nRandom pair: two prompts · random positions (lower-reference cloud)"
)
ax.set_ylim(-1.05, 1.05)
ax.grid(True, alpha=0.25)
ax.legend(loc="upper right", fontsize=9)
fig.tight_layout()
stonesoup.show(fig, basename=f"{MODEL_BASENAME}_diff_context_random_token_pair_cos")
