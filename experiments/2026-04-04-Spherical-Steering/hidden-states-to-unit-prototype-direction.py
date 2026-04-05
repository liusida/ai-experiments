# %% Load cached answer hidden states

import json
from pathlib import Path

import torch

import stonesoup

MODEL_ID = "Qwen/Qwen3.5-2B"
_R = stonesoup.repo_root()
_EXP = Path("experiments/2026-04-04-Spherical-Steering")

_STAT_FULL = 0
_STAT_FULL_NORM = 1
_STAT_ANSWER = 2
_STAT_ANSWER_NORM = 3
_STAT_LAST = 4

_cache_dir = next(
    (
        _R / "outputs/stonesoup" / _EXP / stem / "truthfulqa_answer_hidden_mean"
        for stem in ("dataset-truthqa-to-hidden-states", "dataset-truthqa")
        if (_R / "outputs/stonesoup" / _EXP / stem / "truthfulqa_answer_hidden_mean" / "meta.json").is_file()
    ),
    None,
)
if _cache_dir is None:
    raise FileNotFoundError(
        f"no TruthfulQA cache under {_R / 'outputs/stonesoup' / _EXP}/*/truthfulqa_answer_hidden_mean/"
    )

_meta = json.loads((_cache_dir / "meta.json").read_text())
assert _meta["model_id"] == MODEL_ID, _meta.get("model_id")
_KIND = _meta.get("kind")
if _KIND != "answer_hidden_five_stats_per_layer_chat_span":
    raise ValueError(
        f"cache kind mismatch (got {_KIND!r}; need answer_hidden_five_stats_per_layer_chat_span). "
        "Re-run ``dataset-truthqa-to-hidden-states`` and re-save."
    )

correct_answer_hiddens = torch.load(_cache_dir / "correct.pt", map_location="cpu", weights_only=False)
incorrect_answer_hiddens = torch.load(_cache_dir / "incorrect.pt", map_location="cpu", weights_only=False)

# ``[N, num_layers+1, 5, H]`` — see ``answer_stat_labels`` in meta.json
print(_cache_dir, tuple(correct_answer_hiddens.shape), tuple(incorrect_answer_hiddens.shape))

# %% Check L2 norms of cached summary vectors (per row, then mean over N)

_labels = _meta.get(
    "answer_stat_labels",
    [
        "full_sequence_mean",
        "full_sequence_normalized_mean",
        "answer_mean",
        "answer_normalized_mean",
        "last_token",
    ],
)

import torch.nn.functional as F

_last_layer = -1
print(f"layer index {_last_layer} (last layer index: embeddings … final block)")
for _si, _name in enumerate(_labels):
    _c = correct_answer_hiddens[:, _last_layer, _si, :].float()
    _ic = incorrect_answer_hiddens[:, _last_layer, _si, :].float()
    _n_c = _c.norm(dim=-1)
    _n_ic = _ic.norm(dim=-1)
    print(
        f"{_name:32s}  correct ‖h‖₂  mean={_n_c.mean():.4f} std={_n_c.std():.4f}  |  "
        f"incorrect ‖h‖₂  mean={_n_ic.mean():.4f} std={_n_ic.std():.4f}"
    )

_n_layers = correct_answer_hiddens.shape[1]
print()
print("cos(full_sequence_mean, full_sequence_normalized_mean)  per example, all layers")
for _li in range(_n_layers):
    _parts = [f"layer {_li:2d}"]
    for _name, _t in ("correct", correct_answer_hiddens), ("incorrect", incorrect_answer_hiddens):
        _cos = F.cosine_similarity(
            _t[:, _li, _STAT_FULL, :].float(),
            _t[:, _li, _STAT_FULL_NORM, :].float(),
            dim=-1,
        )
        _parts.append(
            f"{_name[:3]} μ={_cos.mean():.4f} σ={_cos.std():.4f} [{_cos.min():.4f},{_cos.max():.4f}]"
        )
    print("  " + "  |  ".join(_parts))

print()
print("cos(answer_mean, answer_normalized_mean)  per example, all layers")
for _li in range(_n_layers):
    _parts = [f"layer {_li:2d}"]
    for _name, _t in ("correct", correct_answer_hiddens), ("incorrect", incorrect_answer_hiddens):
        _cos = F.cosine_similarity(
            _t[:, _li, _STAT_ANSWER, :].float(),
            _t[:, _li, _STAT_ANSWER_NORM, :].float(),
            dim=-1,
        )
        _parts.append(
            f"{_name[:3]} μ={_cos.mean():.4f} σ={_cos.std():.4f} [{_cos.min():.4f},{_cos.max():.4f}]"
        )
    print("  " + "  |  ".join(_parts))

print()
print("cos(full_sequence_mean, answer_mean)  per example, all layers")
for _li in range(_n_layers):
    _parts = [f"layer {_li:2d}"]
    for _name, _t in ("correct", correct_answer_hiddens), ("incorrect", incorrect_answer_hiddens):
        _cos = F.cosine_similarity(
            _t[:, _li, _STAT_FULL, :].float(),
            _t[:, _li, _STAT_ANSWER, :].float(),
            dim=-1,
        )
        _parts.append(
            f"{_name[:3]} μ={_cos.mean():.4f} σ={_cos.std():.4f} [{_cos.min():.4f},{_cos.max():.4f}]"
        )
    print("  " + "  |  ".join(_parts))

print()
print("cos(full_sequence_mean, last_token)  per example, all layers")
for _li in range(_n_layers):
    _parts = [f"layer {_li:2d}"]
    for _name, _t in ("correct", correct_answer_hiddens), ("incorrect", incorrect_answer_hiddens):
        _cos = F.cosine_similarity(
            _t[:, _li, _STAT_FULL, :].float(),
            _t[:, _li, _STAT_LAST, :].float(),
            dim=-1,
        )
        _parts.append(
            f"{_name[:3]} μ={_cos.mean():.4f} σ={_cos.std():.4f} [{_cos.min():.4f},{_cos.max():.4f}]"
        )
    print("  " + "  |  ".join(_parts))

print()
print("cos(answer_mean, last_token)  per example, all layers")
for _li in range(_n_layers):
    _parts = [f"layer {_li:2d}"]
    for _name, _t in ("correct", correct_answer_hiddens), ("incorrect", incorrect_answer_hiddens):
        _cos = F.cosine_similarity(
            _t[:, _li, _STAT_ANSWER, :].float(),
            _t[:, _li, _STAT_LAST, :].float(),
            dim=-1,
        )
        _parts.append(
            f"{_name[:3]} μ={_cos.mean():.4f} σ={_cos.std():.4f} [{_cos.min():.4f},{_cos.max():.4f}]"
        )
    print("  " + "  |  ".join(_parts))

# %% Unit prototype directions (answer_mean, one vector per layer)

# Per layer ``ℓ``: grand-mean correct vs incorrect on **answer_mean**, normalize each mean to the
# sphere, subtract, renormalize → **u_prototype[ℓ]**. Shape ``[L+1, H]`` (0 = embeddings, …).
_u_rows: list[torch.Tensor] = []
for _li in range(correct_answer_hiddens.shape[1]):
    u_correct = correct_answer_hiddens[:, _li, _STAT_ANSWER, :].float().mean(dim=0)
    u_incorrect = incorrect_answer_hiddens[:, _li, _STAT_ANSWER, :].float().mean(dim=0)
    u_c = u_correct / u_correct.norm().clamp(min=1e-12)
    u_i = u_incorrect / u_incorrect.norm().clamp(min=1e-12)
    _diff = u_c - u_i
    _u_rows.append(_diff / _diff.norm().clamp(min=1e-12))
u_prototype = torch.stack(_u_rows, dim=0)

print(
    "u_prototype (answer_mean, per layer)",
    tuple(u_prototype.shape),
    "last-layer ‖u‖₂",
    float(u_prototype[-1].norm()),
)

# %%
