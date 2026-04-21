# %% Imports & matplotlib
from __future__ import annotations

import numpy as np
import stonesoup
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt

from stonesoup.experiment import (
    capture_embed_and_post_blocks,
    configure_matplotlib_agg,
    encode_text_inputs,
    ensure_pad_token_via_eos,
    hf_repo_id_safe_stem,
    inner_tokenizer,
    show,
)

configure_matplotlib_agg()

# %% Config — prompts and cosine mode
MODEL_ID = "Qwen/Qwen3.5-9B"
# Subtract mean over the hidden dimension per vector before L2 norm + dot (Pearson / “centered” cosine).
CENTER_HIDDEN_BEFORE_COS = False

PROMPTS: dict[str, str] = {
    "en_fact": (
        "Caffeine binds to adenosine receptors in the brain, blocking the signal that "
        "normally builds sleep pressure, so alertness lasts for several hours after a dose."
    ),
    "zh_fact": (
        "咖啡因会与大脑中的腺苷受体结合，阻断通常累积“睡眠压力”的信号，因此在摄入后的数小时内警觉性仍会维持。"
    ),
    "en_poem": (
        "The harbor exhales fog; gulls trade cries above rusted chains, "
        "and somewhere a horn answers the climbing sun."
    ),
    "zh_poem": "港口吐出雾来；海鸥在锈链上方互换啼叫，远处一声汽笛回应着爬升的太阳。",
}

# (label, key_a, key_b, color, linestyle) — solid for same-language or same-content pairs; dashed for cross mismatch.
PAIR_SPECS: list[tuple[str, str, str, str, str]] = [
    ("cos(EN fact, EN poem)", "en_fact", "en_poem", "tab:red", "-"),
    ("cos(ZH fact, ZH poem)", "zh_fact", "zh_poem", "tab:brown", "-"),
    ("cos(EN fact, ZH fact)", "en_fact", "zh_fact", "tab:orange", "-"),
    ("cos(EN poem, ZH poem)", "en_poem", "zh_poem", "tab:blue", "-"),
    ("cos(EN fact, ZH poem)", "en_fact", "zh_poem", "tab:purple", "--"),
    ("cos(EN poem, ZH fact)", "en_poem", "zh_fact", "tab:green", "--"),
]

# %% Load model
# If this cell raises (OOM, disk, HF), the assignment below never runs — later cells will see
# NameError for `processor` until load succeeds. Larger checkpoints (e.g. 9B) fail more often.
torch.set_grad_enabled(False)

model, processor = stonesoup.load_model(MODEL_ID)
model.eval()
tokenizer = inner_tokenizer(processor)
ensure_pad_token_via_eos(tokenizer)
device = next(model.parameters()).device

# %% Capture sequence-mean hidden states per stage (embedding + post each block)
# Re-bind here so this cell works if Load was skipped or failed before `processor` existed
# (same `load_model` call is cheap when weights are already in the Stonesoup pool).
model, processor = stonesoup.load_model(MODEL_ID)
model.eval()
ensure_pad_token_via_eos(inner_tokenizer(processor))
device = next(model.parameters()).device


def masked_mean_hidden_per_stage(
    stack: torch.Tensor, attention_mask_row: torch.Tensor
) -> torch.Tensor:
    """Mean over sequence positions where mask is 1. stack (S, batch, seq, H), one row of mask (seq,) → (S, H)."""
    m = attention_mask_row.to(stack.device).float().view(1, -1, 1)
    h = stack[:, 0, :, :].float()
    return ((h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-8)).detach()


seq_mean_hiddens: dict[str, torch.Tensor] = {}

for key, text in PROMPTS.items():
    stonesoup.check_abort()
    inputs = encode_text_inputs(processor, text, device=device)
    stack, stage_names = capture_embed_and_post_blocks(model, inputs, use_cache=False)
    # stack: (num_stages, batch, seq, hidden)
    mask = inputs["attention_mask"][0]
    seq_mean_hiddens[key] = masked_mean_hidden_per_stage(stack, mask)
    n_tok = int(mask.sum().item())
    print(f"{key}: n_stages={stack.shape[0]} n_real_toks={n_tok} (mean over positions)", flush=True)

print("stage_names:", stage_names[:3], "…", stage_names[-1], flush=True)

# %% Pairwise cosine (optionally centered) vs layer index


def cosine_per_stage(h_a: torch.Tensor, h_b: torch.Tensor, *, centered: bool) -> torch.Tensor:
    """h_a, h_b: (num_stages, hidden) → (num_stages,) cosine similarities in [-1, 1]."""
    x = h_a.float()
    y = h_b.float()
    if centered:
        x = x - x.mean(dim=-1, keepdim=True)
        y = y - y.mean(dim=-1, keepdim=True)
    x = F.normalize(x, dim=-1, eps=1e-8)
    y = F.normalize(y, dim=-1, eps=1e-8)
    return (x * y).sum(dim=-1).cpu()


curves: dict[str, np.ndarray] = {}
for label, ka, kb, _color, _ls in PAIR_SPECS:
    stonesoup.check_abort()
    curves[label] = cosine_per_stage(
        seq_mean_hiddens[ka],
        seq_mean_hiddens[kb],
        centered=CENTER_HIDDEN_BEFORE_COS,
    ).numpy()

n_stages = next(iter(seq_mean_hiddens.values())).shape[0]
layer_x = np.arange(n_stages)

# %% Grand-mean centering + two-panel figure
KEY_ORDER = tuple(PROMPTS.keys())


def per_layer_mean_across_inputs_then_renorm(
    h_by_key: dict[str, torch.Tensor], keys: tuple[str, ...]
) -> dict[str, torch.Tensor]:
    """Each stage: subtract mean(hidden) over the |keys| prompts, then L2-normalize per prompt."""
    stacked = torch.stack([h_by_key[k].float() for k in keys], dim=1)  # (num_stages, K, hidden)
    mean_across_inputs = stacked.mean(dim=1, keepdim=True)
    centered = stacked - mean_across_inputs
    normed = F.normalize(centered, dim=-1, eps=1e-8)
    return {k: normed[:, i, :] for i, k in enumerate(keys)}


h_per_layer_centered = per_layer_mean_across_inputs_then_renorm(seq_mean_hiddens, KEY_ORDER)

curves_per_layer_centered: dict[str, np.ndarray] = {}
for label, ka, kb, _color, _ls in PAIR_SPECS:
    stonesoup.check_abort()
    curves_per_layer_centered[label] = (
        (h_per_layer_centered[ka] * h_per_layer_centered[kb]).sum(dim=-1).cpu().numpy()
    )

suffix = "hidden-mean" if CENTER_HIDDEN_BEFORE_COS else "raw"
fig, (ax_top, ax_bottom) = plt.subplots(
    2,
    1,
    sharex=True,
    figsize=(9, 9),
    # gridspec_kw={"hspace": 0.22},
    constrained_layout=False,
)

for label, _ka, _kb, color, ls in PAIR_SPECS:
    ax_top.plot(layer_x, curves[label], color=color, ls=ls, lw=1.8, label=label)
ax_top.set_ylabel("cos")
ax_top.set_title(f"(a) Pairwise cos ({suffix})", fontsize=10)
ax_top.set_ylim(-1.0, 1.0)
ax_top.grid(True, alpha=0.3)
ax_top.legend(loc="lower left", ncol=2, fontsize=7)
ax_top.tick_params(axis="x", labelbottom=False)

for label, _ka, _kb, color, ls in PAIR_SPECS:
    ax_bottom.plot(layer_x, curves_per_layer_centered[label], color=color, ls=ls, lw=1.8, label=label)
ax_bottom.set_xlabel("Layer")
ax_bottom.set_ylabel("cos")
ax_bottom.set_title("(b) Centered cos — 4-prompt layer mean removed", fontsize=10)
ax_bottom.set_ylim(-1.0, 1.0)
ax_bottom.grid(True, alpha=0.3)
ax_bottom.legend(loc="lower left", ncol=2, fontsize=7)

fig.suptitle(f"EN/ZH fact–poem · {MODEL_ID}", fontsize=11, y=0.995)
fig.tight_layout()

_safe = hf_repo_id_safe_stem(MODEL_ID)
show(fig, basename=f"{_safe}_en_zh_fact_poem_cos_twopanel_seqmean", dpi=140)
