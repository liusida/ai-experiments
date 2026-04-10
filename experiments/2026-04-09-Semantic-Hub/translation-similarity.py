# %% Setup
from __future__ import annotations

import stonesoup
import torch
import torch.nn.functional as F
from stonesoup.experiment import (
    capture_embed_and_post_blocks,
    encode_text_inputs,
    ensure_pad_token_via_eos,
    inner_tokenizer,
)

MODEL_ID = "bigscience/bloom-7b1"
# Short parallel EN–ZH pairs (period / full stop aligned for the '.' vs '。' plot).
TEXT_PAIRS: list[tuple[str, str]] = [
    ("The cat sat on the mat.", "猫坐在垫子上。"),
    ("The dog runs fast.", "狗跑得很快。"),
    ("She reads a book.", "她在读一本书。"),
    ("Water is essential for life.", "水对生命至关重要。"),
    ("The sun rises in the east.", "太阳从东方升起。"),
    ("I love learning new things.", "我喜欢学习新事物。"),
    ("Spring is a beautiful season.", "春天是美丽的季节。"),
    ("Computers can process data.", "计算机可以处理数据。"),
    ("Music brings joy to people.", "音乐给人们带来快乐。"),
    ("Tomorrow will be a good day.", "明天会是美好的一天。"),
]
# Prepend BOS so the first *content* token is not the only “sink” position.
PREPEND_BOS = True

model, proc = stonesoup.load_model(MODEL_ID)
model.eval()
tok = inner_tokenizer(proc)
ensure_pad_token_via_eos(tok)
device = next(model.parameters()).device


def _bos_token_id() -> int | None:
    bid = getattr(tok, "bos_token_id", None)
    if bid is not None:
        return int(bid)
    cfg = getattr(model, "config", None)
    if cfg is not None:
        bid = getattr(cfg, "bos_token_id", None)
        if bid is not None:
            return int(bid)
    return None


def prepend_bos_tensor(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    bos_id = _bos_token_id()
    if bos_id is None:
        print("PREPEND_BOS: no bos_token_id on tokenizer or model.config; skipped.", flush=True)
        return inputs
    ids = inputs["input_ids"]
    mask = inputs["attention_mask"]
    b = torch.tensor([[bos_id]], dtype=ids.dtype, device=device)
    return {
        **inputs,
        "input_ids": torch.cat([b, ids], dim=1),
        "attention_mask": torch.cat([torch.ones_like(b), mask], dim=1),
    }


def pairwise_cos(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = F.normalize(a.float(), dim=-1)
    b = F.normalize(b.float(), dim=-1)
    return a @ b.T


def _first_label_index(labels: list[str], token: str) -> int:
    for i, t in enumerate(labels):
        if t == token:
            return i
    raise ValueError(f"no token {token!r} in {labels!r}")


per_pair_mean_scalar: list[float] = []
per_layer_mean_over_pairs: list[torch.Tensor] = []
dot_stop_curves: list[torch.Tensor] = []
COS_SIM_LAYERS_EN_ZH: torch.Tensor | None = None
en_labels: list[str] = []
zh_labels: list[str] = []
stage_names: list[str] = []

for pair_idx, (TEXT_EN, TEXT_ZH) in enumerate(TEXT_PAIRS):
    stonesoup.check_abort()
    inputs_en = encode_text_inputs(proc, TEXT_EN, device=device)
    inputs_zh = encode_text_inputs(proc, TEXT_ZH, device=device)
    if PREPEND_BOS:
        inputs_en = prepend_bos_tensor(inputs_en)
        inputs_zh = prepend_bos_tensor(inputs_zh)

    stack_en, stage_names = capture_embed_and_post_blocks(model, inputs_en, use_cache=False)
    stack_zh, _ = capture_embed_and_post_blocks(model, inputs_zh, use_cache=False)

    layers: list[torch.Tensor] = []
    for li in range(len(stage_names)):
        stonesoup.check_abort()
        he = stack_en[li, 0]
        hz = stack_zh[li, 0]
        layers.append(pairwise_cos(he, hz))
    cos_k = torch.stack(layers, dim=0)
    # (num_stages, len_en, len_zh): [layer, en_i, zh_j] = cos(h_en_i, h_zh_j)
    per_pair_mean_scalar.append(float(cos_k.mean().item()))
    per_layer_mean_over_pairs.append(cos_k.mean(dim=(1, 2)))

    if pair_idx == 0:
        COS_SIM_LAYERS_EN_ZH = cos_k
        en_ids = inputs_en["input_ids"][0].tolist()
        zh_ids = inputs_zh["input_ids"][0].tolist()
        en_labels = [tok.decode([i]) for i in en_ids]
        zh_labels = [tok.decode([i]) for i in zh_ids]
        print("tokens EN (pair 0):", en_labels)
        print("tokens ZH (pair 0):", zh_labels)

    _en_ids = inputs_en["input_ids"][0].tolist()
    _zh_ids = inputs_zh["input_ids"][0].tolist()
    _en_labels = [tok.decode([i]) for i in _en_ids]
    _zh_labels = [tok.decode([i]) for i in _zh_ids]

    try:
        _ei_dot = _first_label_index(_en_labels, ".")
        _zj_stop = _first_label_index(_zh_labels, "。")
    except ValueError:
        pass
    else:
        dot_stop_curves.append(cos_k[:, _ei_dot, _zj_stop].detach().float())

assert COS_SIM_LAYERS_EN_ZH is not None
COS_SIM_PER_LAYER_MEAN_10_PAIRS = torch.stack(per_layer_mean_over_pairs, dim=0).mean(dim=0)
print(
    "mean cosine (all positions, all layers), averaged over pairs:",
    float(sum(per_pair_mean_scalar) / len(per_pair_mean_scalar)),
    flush=True,
)
print("per-pair mean cos:", [round(x, 4) for x in per_pair_mean_scalar], flush=True)
print("COS_SIM_LAYERS_EN_ZH (pair 0)", COS_SIM_LAYERS_EN_ZH.shape, flush=True)
print("COS_SIM_PER_LAYER_MEAN_10_PAIRS", COS_SIM_PER_LAYER_MEAN_10_PAIRS.shape, flush=True)

# %% Cosine similarity: each (EN pos, ZH pos) at every layer (pair 0 shown below)

_, n_en, n_zh = COS_SIM_LAYERS_EN_ZH.shape
mean_en_zh = COS_SIM_LAYERS_EN_ZH.float().mean(dim=0).reshape(-1)
k_top = min(3, mean_en_zh.numel())
vals, idx_flat = torch.topk(mean_en_zh, k_top)
TOP3_EN_ZH_MEAN_LAYER: list[tuple[str, str, float]] = []
for rank in range(k_top):
    idx = int(idx_flat[rank].item())
    ei, zj = idx // n_zh, idx % n_zh
    v = float(vals[rank].item())
    TOP3_EN_ZH_MEAN_LAYER.append((en_labels[ei], zh_labels[zj], v))
    pair = f"{en_labels[ei]!r} vs {zh_labels[zj]!r}"
    print(
        f"top-{rank + 1} (pair 0 only, mean over layers): {pair}  mean_cos={v:.4f}",
        flush=True,
    )

# %% Plot: one heatmap per layer (y = English token, x = Chinese token)
import numpy as np

from stonesoup.experiment import configure_matplotlib_agg, hf_repo_id_safe_stem

configure_matplotlib_agg()
import matplotlib.pyplot as plt

arr = COS_SIM_LAYERS_EN_ZH.float().cpu().numpy()
n_st, n_en, n_zh = arr.shape
ncols = min(6, max(1, n_st))
nrows = int(np.ceil(n_st / ncols))
fig, axes = plt.subplots(
    nrows,
    ncols,
    figsize=(ncols * max(2.0, n_zh * 0.35), nrows * max(1.6, n_en * 0.35)),
    constrained_layout=True,
)
axes_flat = np.ravel(np.atleast_1d(axes))
last_im = None
for li in range(n_st):
    stonesoup.check_abort()
    ax = axes_flat[li]
    last_im = ax.imshow(
        arr[li],
        aspect="auto",
        vmin=0.0,
        vmax=1.0,
        cmap="Blues",
        interpolation="nearest",
    )
    row, col = li // ncols, li % ncols
    if col == 0:
        ax.set_yticks(np.arange(n_en))
        ax.set_yticklabels(en_labels, fontsize=7)
    else:
        ax.set_yticks([])
    # if row == nrows - 1:
    ax.set_xticks(np.arange(n_zh))
    ax.set_xticklabels(zh_labels, rotation=40, ha="right", fontsize=7)
    # else:
    #     ax.set_xticks([])
    ax.set_title(stage_names[li], fontsize=8)
for j in range(n_st, len(axes_flat)):
    axes_flat[j].set_visible(False)
fig.supxlabel("Chinese token")
fig.supylabel("English token")
TEXT_EN_0, TEXT_ZH_0 = TEXT_PAIRS[0]
fig.suptitle(
    f"Pair 1/10 (example heatmaps)\nEN: {TEXT_EN_0}\nZH: {TEXT_ZH_0}\n"
    f"Cosine similarity: EN × ZH per layer — {MODEL_ID}",
    fontsize=7,
)
fig.colorbar(last_im, ax=axes_flat[:n_st].tolist(), shrink=0.5, label="cos")
stonesoup.show(fig, basename=f"{hf_repo_id_safe_stem(MODEL_ID)}_en_zh_cos_layers", dpi=120)

# %% Plot: '.' vs '。' cosine similarity vs layer (line), mean over pairs
import numpy as np

from stonesoup.experiment import configure_matplotlib_agg, hf_repo_id_safe_stem

configure_matplotlib_agg()
import matplotlib.pyplot as plt

if not dot_stop_curves:
    print("line plot skipped: no '.' vs '。' curves collected", flush=True)
else:
    _curve = torch.stack(dot_stop_curves, dim=0).mean(dim=0).cpu().numpy()
    COS_SIM_DOT_VS_IDEOGRAPHIC_STOP_VS_LAYER = _curve  # (num_stages,) mean over pairs
    _n_st = len(_curve)
    _fig, _ax = plt.subplots(figsize=(7, 4.0))
    _fig.suptitle(
        f"Mean over {len(dot_stop_curves)} EN–ZH pairs: '.' vs '。' — {MODEL_ID}",
        fontsize=8,
    )
    _ax.plot(np.arange(_n_st), _curve, marker="o", ms=3, label="mean over pairs")
    _ax.plot(
        np.arange(_n_st),
        COS_SIM_PER_LAYER_MEAN_10_PAIRS.detach().float().cpu().numpy(),
        ls="--",
        alpha=0.7,
        label="mean cos (all token pairs, all pairs)",
    )
    _ax.set_xlabel("layer (stage index)")
    _ax.set_ylabel("cosine similarity")
    _ax.set_title("EN '.' vs ZH '。' (averaged) vs global mean per layer")
    _ax.set_xticks(np.linspace(0, _n_st - 1, num=min(_n_st, 12), dtype=int))
    _ax.grid(True, alpha=0.3)
    _ax.legend(loc="best", fontsize=7)
    _fig.tight_layout(rect=[0, 0, 1, 0.90])
    stonesoup.show(_fig, basename=f"{hf_repo_id_safe_stem(MODEL_ID)}_dot_vs_idc_cos_vs_layer", dpi=120)
