# %% Imports, model, knobs
from __future__ import annotations

import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache
import stonesoup

MODEL_ID = "Qwen/Qwen3-0.6B-Base"
LEFT_CONTEXT_1 = (
    "Shiva is one of the principal deities of Hinduism and the supreme god"
)
LEFT_CONTEXT_2 = (
    "Artemis II is an ongoing United States spaceflight mission sending four astronauts "
    "on a flyby around the Moon. It launched"
)
N_RANDOM_TOKENS = 100
SEED = 0

model, tokenizer = stonesoup.load_model(MODEL_ID)
model.eval()
device = next(model.parameters()).device
inner_tok = getattr(tokenizer, "tokenizer", tokenizer)
MODEL_BASENAME = MODEL_ID.replace("/", "__").replace(":", "-")
print("Loaded:", MODEL_ID, device, flush=True)

# %% Helpers: KV clone, collect last-token hiddens for a prefix + shared token list


def clone_kv_cache(past) -> DynamicCache:
    rows: list[tuple] = []
    for tup in past:
        k, v = tup[0], tup[1]
        slide = tup[2] if len(tup) > 2 else None
        if slide is not None:
            rows.append((k.clone(), v.clone(), slide.clone()))
        else:
            rows.append((k.clone(), v.clone()))
    return DynamicCache(ddp_cache_data=rows, config=model.config)


@torch.inference_mode()
def last_pos_hiddens_per_layer(left_text: str, token_ids: list[int]) -> torch.Tensor:
    """``(n_stages, n_tok, hidden)`` — last position after each one-token decode from ``left_text``."""
    enc_pre = inner_tok(
        left_text,
        return_tensors="pt",
        return_attention_mask=True,
        add_special_tokens=True,
    )
    prefix_ids = enc_pre["input_ids"].to(device)
    prefix_len = int(prefix_ids.shape[1])
    attn_one = torch.ones(1, prefix_len + 1, device=device, dtype=torch.long)
    out_pre = model(
        input_ids=prefix_ids,
        attention_mask=enc_pre["attention_mask"].to(device),
        use_cache=True,
    )
    past_prefix = out_pre.past_key_values
    vecs: list[torch.Tensor] = []
    for tid in token_ids:
        stonesoup.check_abort()
        out = model(
            input_ids=torch.tensor([[tid]], device=device, dtype=torch.long),
            attention_mask=attn_one,
            past_key_values=clone_kv_cache(past_prefix),
            use_cache=False,
            output_hidden_states=True,
        )
        hs = out.hidden_states
        vecs.append(torch.stack([h[0, -1].float() for h in hs], dim=0))
    H = torch.stack(vecs, dim=0).transpose(0, 1).contiguous()
    return H


def pairwise_upper_mean_std(H: torch.Tensor, tri_i: torch.Tensor, tri_j: torch.Tensor):
    """Mean / std of cos(h_i, h_j) over all pairs i < j at each layer."""
    n_st = int(H.shape[0])
    mean_c = np.zeros(n_st, dtype=np.float64)
    std_c = np.zeros(n_st, dtype=np.float64)
    for li in range(n_st):
        stonesoup.check_abort()
        x = F.normalize(H[li], dim=-1)
        gram = x @ x.T
        c = gram[tri_i, tri_j].float()
        mean_c[li] = float(c.mean().cpu())
        std_c[li] = float(c.std().cpu())
    return mean_c, std_c


def same_token_two_prefixes_mean_std(H_a: torch.Tensor, H_b: torch.Tensor):
    """Aligned rows: cos between same token id after prefix A vs prefix B (mean/std over tokens)."""
    assert H_a.shape == H_b.shape
    n_st = int(H_a.shape[0])
    mean_c = np.zeros(n_st, dtype=np.float64)
    std_c = np.zeros(n_st, dtype=np.float64)
    for li in range(n_st):
        stonesoup.check_abort()
        a = F.normalize(H_a[li], dim=-1)
        b = F.normalize(H_b[li], dim=-1)
        c = (a * b).sum(dim=-1).float()
        mean_c[li] = float(c.mean().cpu())
        std_c[li] = float(c.std().cpu())
    return mean_c, std_c


# %% Sample token ids once; run both prefixes; within-prefix pairs + cross-prefix aligned cos
bad = {inner_tok.pad_token_id, inner_tok.eos_token_id, inner_tok.bos_token_id, inner_tok.unk_token_id}
bad.discard(None)
vocab = int(getattr(inner_tok, "vocab_size", len(inner_tok)))
rng = random.Random(SEED)
trial_ids: list[int] = []
while len(trial_ids) < N_RANDOM_TOKENS:
    stonesoup.check_abort()
    t = rng.randrange(vocab)
    if t in bad:
        continue
    trial_ids.append(t)

print("Collecting ctx1 …", flush=True)
H1 = last_pos_hiddens_per_layer(LEFT_CONTEXT_1, trial_ids)
print("Collecting ctx2 …", flush=True)
H2 = last_pos_hiddens_per_layer(LEFT_CONTEXT_2, trial_ids)

n_stages = int(H1.shape[0])
n_tok = int(H1.shape[1])
tri_i, tri_j = torch.triu_indices(n_tok, n_tok, offset=1, device=device)
n_pairs = n_tok * (n_tok - 1) // 2
print(
    f"tokens={n_tok}  stages={n_stages}  pairs/layer (within-ctx)={n_pairs}",
    flush=True,
)

mean_w1, std_w1 = pairwise_upper_mean_std(H1, tri_i, tri_j)
mean_w2, std_w2 = pairwise_upper_mean_std(H2, tri_i, tri_j)
mean_x, std_x = same_token_two_prefixes_mean_std(H1, H2)

# %% One figure: two within-context curves + same-token cross-context
layer_x = np.arange(n_stages)
fig, ax = plt.subplots(figsize=(11, 4.3))
ax.plot(layer_x, mean_w1, color="steelblue", linewidth=1.3, label="ctx1: diff tok, pairwise")
ax.fill_between(layer_x, mean_w1 - std_w1, mean_w1 + std_w1, color="steelblue", alpha=0.18)
ax.plot(layer_x, mean_w2, color="darkorange", linewidth=1.3, label="ctx2: diff tok, pairwise")
ax.fill_between(layer_x, mean_w2 - std_w2, mean_w2 + std_w2, color="darkorange", alpha=0.18)
ax.plot(layer_x, mean_x, color="seagreen", linewidth=1.3, label="same tok: ctx1 vs ctx2")
ax.fill_between(layer_x, mean_x - std_x, mean_x + std_x, color="seagreen", alpha=0.22)
ax.set_xlabel("layer stage (0 = pre block 0, k = post block k-1)")
ax.set_ylabel("cosine similarity")
ax.set_title(
    f"{MODEL_ID}\n"
    f"{N_RANDOM_TOKENS} random next-token ids · within-prefix pairwise (blue, orange) · "
    f"aligned same id across prefixes (green)"
)
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.25)
ax.legend(loc="best", fontsize=9)
fig.tight_layout()
stonesoup.show(fig, basename=f"{MODEL_BASENAME}_same_left_ctx_random_tok_cos_pairs")
