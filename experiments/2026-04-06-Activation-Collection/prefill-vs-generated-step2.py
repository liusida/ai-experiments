# %% Imports & load cache
from __future__ import annotations

from typing import Any

import torch
import stonesoup

# Written by ``prefill-vs-generated.py`` (per-script outputs dir for that file).
_CACHE = (
    stonesoup.repo_root()
    / "outputs"
    / "2026-04-06-Activation-Collection"
    / "prefill-vs-generated"
    / "prefill_vs_generated_cache"
    / "activations.pt"
)

payload: dict[str, Any] = torch.load(_CACHE, map_location="cpu", weights_only=False)

prefill_qwen: torch.Tensor = payload["prefill_qwen"]
prefill_correct: torch.Tensor = payload["prefill_correct"]
prefill_incorrect: torch.Tensor = payload["prefill_incorrect"]
generated: torch.Tensor = payload["generated"]
records: list[dict[str, Any]] = payload["records"]

print("cache:", _CACHE.relative_to(stonesoup.repo_root()), flush=True)
for name, t in (
    ("prefill_qwen", prefill_qwen),
    ("prefill_correct", prefill_correct),
    ("prefill_incorrect", prefill_incorrect),
    ("generated", generated),
):
    print(f"  {name}: {tuple(t.shape)} {t.dtype}", flush=True)
print(f"  records: {len(records)}", flush=True)

# %% Pairwise cos sim of global-mean activations (per layer, heatmap grid)
# Mean is over all tokens in each branch’s concatenated tensor for one stage (layer).
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import torch.nn.functional as F

CAT_KEYS = ("prefill_qwen", "prefill_correct", "prefill_incorrect", "generated")
CAT_LABELS = ("qwen", "correct", "inc", "gen")
_BRANCH_TENSORS: dict[str, torch.Tensor] = {
    "prefill_qwen": prefill_qwen,
    "prefill_correct": prefill_correct,
    "prefill_incorrect": prefill_incorrect,
    "generated": generated,
}


def layer_category_means(tensors: dict[str, torch.Tensor], layer: int) -> dict[str, torch.Tensor]:
    return {k: tensors[k][:, layer, :].float().mean(dim=0) for k in CAT_KEYS}


def layer_category_mean_cos_matrix(
    tensors: dict[str, torch.Tensor], layer: int
) -> torch.Tensor:
    means_dict = layer_category_means(tensors, layer)
    V = torch.stack([means_dict[k] for k in CAT_KEYS], dim=0)
    V = F.normalize(V, dim=1, eps=1e-8)
    return V @ V.T


n_layers = int(prefill_qwen.shape[1])
for k in CAT_KEYS:
    assert _BRANCH_TENSORS[k].shape[1] == n_layers

print("\n--- Global-mean hidden states before cos-sim heatmaps ---", flush=True)
for ls in (0, n_layers // 2, n_layers - 1):
    means = layer_category_means(_BRANCH_TENSORS, ls)
    print(f"\nlayer {ls} (unnormalized mean over all tokens in branch):", flush=True)
    for k in CAT_KEYS:
        m = means[k]
        print(
            f"  {k:16s}  L2={float(m.norm()):.6f}  min={float(m.min()):.6f}  "
            f"max={float(m.max()):.6f}  mean={float(m.mean()):.6f}  first12={m[:12].tolist()}",
            flush=True,
        )
    for i, ki in enumerate(CAT_KEYS):
        for kj in CAT_KEYS[i + 1 :]:
            a, b = means[ki], means[kj]
            diff_l2 = float((a - b).norm())
            cos_ab = float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())
            print(f"  cos({ki}, {kj})={cos_ab:.8f}  L2({ki}-{kj})={diff_l2:.6f}", flush=True)

_cos_mats = [
    layer_category_mean_cos_matrix(_BRANCH_TENSORS, li).cpu().numpy()
    for li in range(n_layers)
]
_off_vals: list[float] = []
for _m in _cos_mats:
    for _i in range(4):
        for _j in range(4):
            if _i != _j:
                _off_vals.append(float(_m[_i, _j]))
_cos_lo = min(_off_vals)
_cos_hi = max(_off_vals)
_pad = max(0.001, (_cos_hi - _cos_lo) * 0.12)
# Narrow scale so ~0.97 vs ~0.99 is visible (full [-1,1] makes everything one color).
_cos_vmin = max(_cos_lo - _pad, -1.0)
_cos_vmax = 1.0
print(
    f"\nHeatmap color scale: off-diagonal cos in [{_cos_lo:.6f}, {_cos_hi:.6f}] "
    f"→ imshow [{_cos_vmin:.6f}, {_cos_vmax:.6f}] (labels use .4f so 0.9989 ≠ 1.0000)",
    flush=True,
)
_label_mid = 0.5 * (_cos_vmin + _cos_vmax)

cols = 5
rows = (n_layers + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(2.6 * cols, 2.4 * rows), squeeze=False)
norm = Normalize(vmin=_cos_vmin, vmax=_cos_vmax)
cmap = plt.cm.coolwarm
last_im = None
for li in range(rows * cols):
    r, c = divmod(li, cols)
    ax = axes[r][c]
    if li >= n_layers:
        ax.axis("off")
        continue
    mat = _cos_mats[li]
    last_im = ax.imshow(mat, cmap=cmap, norm=norm, aspect="equal")
    ax.set_title(f"layer {li}", fontsize=9)
    ax.set_xticks(range(len(CAT_LABELS)))
    ax.set_yticks(range(len(CAT_LABELS)))
    if c == 0:
        ax.set_yticklabels(CAT_LABELS, fontsize=7)
    else:
        ax.set_yticklabels([])
    if r == rows - 1:
        ax.set_xticklabels(CAT_LABELS, rotation=45, ha="right", fontsize=7)
    else:
        ax.set_xticklabels([])
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(
                j,
                i,
                f"{mat[i, j]:.4f}",
                ha="center",
                va="center",
                color="white" if mat[i, j] > _label_mid else "black",
                fontsize=5,
            )

fig.suptitle(
    "Cosine similarity: pairwise global-mean activations (per stage / layer)",
    fontsize=11,
)
if last_im is not None:
    fig.subplots_adjust(
        left=0.07, right=0.86, top=0.90, bottom=0.06, wspace=0.28, hspace=0.40
    )
    cax = fig.add_axes([0.89, 0.18, 0.018, 0.52])
    fig.colorbar(last_im, cax=cax, label="cos sim")

# %% show plot
stonesoup.show(fig, basename="category_global_mean_cos_per_layer", dpi=140)

# %% MLP: classify branch from activations at one layer (80% train / 20% test, stratified)
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Same tensors / labels as the heatmap cell (repeat so this cell runs after load-only).
CAT_KEYS = ("prefill_qwen", "prefill_correct", "prefill_incorrect", "generated")
CAT_LABELS = ("qwen", "correct", "inc", "gen")
_BRANCH_TENSORS = {
    "prefill_qwen": prefill_qwen,
    "prefill_correct": prefill_correct,
    "prefill_incorrect": prefill_incorrect,
    "generated": generated,
}

MLP_LAYER = 24
MLP_HIDDEN = 256
MLP_EPOCHS = 25
MLP_LR = 1e-3
MLP_BATCH = 4096
MLP_TEST_FRAC = 0.2
MLP_SEED = 42


class BranchMLP(nn.Module):
    def __init__(self, dim: int, hidden: int, n_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def stratified_train_test_idx(y: torch.Tensor, test_frac: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    y = y.view(-1).long()
    n_cls = int(y.max().item()) + 1
    train_parts: list[torch.Tensor] = []
    test_parts: list[torch.Tensor] = []
    for c in range(n_cls):
        ix = (y == c).nonzero(as_tuple=False).view(-1)
        n = int(ix.numel())
        if n == 0:
            continue
        perm = ix[torch.randperm(n, generator=g)]
        n_te = int(round(n * test_frac))
        if n > 1:
            n_te = min(max(n_te, 1), n - 1)
        else:
            n_te = 0
        if n_te <= 0:
            train_parts.append(perm)
        else:
            test_parts.append(perm[:n_te])
            train_parts.append(perm[n_te:])
    return torch.cat(train_parts), torch.cat(test_parts)


def build_xy_one_layer(
    tensors: dict[str, torch.Tensor], layer: int, keys: tuple[str, ...]
) -> tuple[torch.Tensor, torch.Tensor]:
    xs: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    for lab, k in enumerate(keys):
        t = tensors[k][:, layer, :].float().contiguous()
        xs.append(t)
        ys.append(torch.full((t.shape[0],), lab, dtype=torch.long))
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)


_X_mlp, _y_mlp = build_xy_one_layer(_BRANCH_TENSORS, MLP_LAYER, CAT_KEYS)
_tr_ix, _te_ix = stratified_train_test_idx(_y_mlp, MLP_TEST_FRAC, MLP_SEED)
X_tr, y_tr = _X_mlp[_tr_ix], _y_mlp[_tr_ix]
X_te, y_te = _X_mlp[_te_ix], _y_mlp[_te_ix]

_mu = X_tr.mean(dim=0, keepdim=True)
_sd = X_tr.std(dim=0, keepdim=True).clamp_min(1e-6)
X_tr = (X_tr - _mu) / _sd
X_te = (X_te - _mu) / _sd

_n_cls = len(CAT_KEYS)
_ct = torch.bincount(y_tr, minlength=_n_cls).float()
_class_w = (_ct.sum() / (_ct * _n_cls)).clamp_min(1e-6)

_ld_gen = torch.Generator()
_ld_gen.manual_seed(MLP_SEED)
_train_loader = DataLoader(
    TensorDataset(X_tr, y_tr),
    batch_size=MLP_BATCH,
    shuffle=True,
    generator=_ld_gen,
)

_dev = torch.device("cpu")
_mlp = BranchMLP(X_tr.shape[-1], MLP_HIDDEN, _n_cls).to(_dev)
_opt = torch.optim.AdamW(_mlp.parameters(), lr=MLP_LR)
_crit = nn.CrossEntropyLoss(weight=_class_w.to(_dev))

_mlp.train()
for _ep in range(MLP_EPOCHS):
    _loss_sum = 0.0
    _n = 0
    for _xb, _yb in _train_loader:
        _xb, _yb = _xb.to(_dev), _yb.to(_dev)
        _opt.zero_grad(set_to_none=True)
        _logits = _mlp(_xb)
        _loss = _crit(_logits, _yb)
        _loss.backward()
        _opt.step()
        _loss_sum += float(_loss.item()) * _xb.size(0)
        _n += _xb.size(0)
    if (_ep + 1) % 5 == 0 or _ep == 0:
        print(
            f"MLP epoch {_ep + 1}/{MLP_EPOCHS}  loss={_loss_sum / max(_n, 1):.4f}",
            flush=True,
        )


@torch.inference_mode()
def _acc(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return float((logits.argmax(dim=-1) == labels).float().mean().item())


_mlp.eval()
_tr_acc = _acc(_mlp(X_tr.to(_dev)), y_tr.to(_dev))
_te_acc = _acc(_mlp(X_te.to(_dev)), y_te.to(_dev))
print(
    f"\nMLP layer={MLP_LAYER}  hidden={MLP_HIDDEN}  train acc={_tr_acc:.4f}  test acc={_te_acc:.4f}",
    flush=True,
)
_pred_te = _mlp(X_te.to(_dev)).argmax(dim=-1)
for _ci, _name in enumerate(CAT_LABELS):
    _mask = y_te == _ci
    if int(_mask.sum()) == 0:
        continue
    _a = float((_pred_te[_mask] == y_te.to(_dev)[_mask]).float().mean().item())
    print(f"  test acc [{_name}]: {_a:.4f} (n={int(_mask.sum())})", flush=True)

# %% MLP binary: defs + export one probe (for step3; trains a single layer only)
# Label 0 = prefill (qwen + correct + incorrect); label 1 = generated.
# Requires the 4-class MLP cell: ``BranchMLP``, ``stratified_train_test_idx``, ``_acc``, ``_dev``, ``nn``, ``DataLoader``.

if "BranchMLP" not in globals():
    raise RuntimeError(
        "Run the 4-way MLP cell above first (defines BranchMLP, stratified_train_test_idx, _acc, _dev)."
    )

PREFILL_KEYS = ("prefill_qwen", "prefill_correct", "prefill_incorrect")
_BRANCH_TENSORS_BIN = {
    "prefill_qwen": prefill_qwen,
    "prefill_correct": prefill_correct,
    "prefill_incorrect": prefill_incorrect,
    "generated": generated,
}

BINARY_MLP_HIDDEN = 256
BINARY_MLP_EPOCHS = 25
BINARY_MLP_LR = 1e-3
BINARY_MLP_BATCH = 4096
BINARY_MLP_TEST_FRAC = 0.2
BINARY_MLP_SEED = 43
BINARY_EXPORT_LAYER = 15
_BINARY_CKPT_NAME = "mlp_binary_gen_vs_prefill_layer15.pt"


def build_xy_generated_vs_prefill(
    tensors: dict[str, torch.Tensor], layer: int
) -> tuple[torch.Tensor, torch.Tensor]:
    xs_pref: list[torch.Tensor] = []
    for k in PREFILL_KEYS:
        xs_pref.append(tensors[k][:, layer, :].float().contiguous())
    x_pref = torch.cat(xs_pref, dim=0)
    y_pref = torch.zeros((x_pref.shape[0],), dtype=torch.long)
    x_gen = tensors["generated"][:, layer, :].float().contiguous()
    y_gen = torch.ones((x_gen.shape[0],), dtype=torch.long)
    return torch.cat([x_pref, x_gen], dim=0), torch.cat([y_pref, y_gen], dim=0)


def train_binary_gen_vs_prefill_one_layer(
    layer: int,
    tensors: dict[str, torch.Tensor],
    y_full: torch.Tensor,
    train_ix: torch.Tensor,
    test_ix: torch.Tensor,
) -> tuple[float, float, BranchMLP, torch.Tensor, torch.Tensor, int]:
    """Train acc, test acc, ``BranchMLP`` on CPU, z-score ``mu``/``sd`` (CPU, shape ``(1,H)``), ``in_dim``."""
    X_full, y_check = build_xy_generated_vs_prefill(tensors, layer)
    if not torch.equal(y_check, y_full):
        raise RuntimeError("bug: branch labels changed across layers")
    X_tr, X_te = X_full[train_ix], X_full[test_ix]
    y_tr, y_te = y_full[train_ix], y_full[test_ix]
    in_dim = int(X_tr.shape[-1])
    mu = X_tr.mean(dim=0, keepdim=True)
    sd = X_tr.std(dim=0, keepdim=True).clamp_min(1e-6)
    X_tr = (X_tr - mu) / sd
    X_te = (X_te - mu) / sd
    ctb = torch.bincount(y_tr, minlength=2).float()
    wb = (ctb.sum() / (ctb * 2)).clamp_min(1e-6)
    gen = torch.Generator()
    gen.manual_seed(BINARY_MLP_SEED + int(layer) * 1000)
    loader = DataLoader(
        TensorDataset(X_tr, y_tr),
        batch_size=BINARY_MLP_BATCH,
        shuffle=True,
        generator=gen,
    )
    model = BranchMLP(in_dim, BINARY_MLP_HIDDEN, 2).to(_dev)
    opt = torch.optim.AdamW(model.parameters(), lr=BINARY_MLP_LR)
    crit = nn.CrossEntropyLoss(weight=wb.to(_dev))
    model.train()
    for _ in range(BINARY_MLP_EPOCHS):
        for xb, yb in loader:
            xb, yb = xb.to(_dev), yb.to(_dev)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
    model.eval()
    atr = _acc(model(X_tr.to(_dev)), y_tr.to(_dev))
    ate = _acc(model(X_te.to(_dev)), y_te.to(_dev))
    return atr, ate, model.cpu(), mu.cpu(), sd.cpu(), in_dim


_n_bin_layers = int(prefill_qwen.shape[1])
_export_layer_eff = int(min(BINARY_EXPORT_LAYER, _n_bin_layers - 1))
if _export_layer_eff != int(BINARY_EXPORT_LAYER):
    print(
        f"BINARY_EXPORT_LAYER={BINARY_EXPORT_LAYER} out of range; exporting layer {_export_layer_eff}",
        flush=True,
    )
_, _yb_bin = build_xy_generated_vs_prefill(_BRANCH_TENSORS_BIN, 0)
_tr_bin_export, _te_bin_export = stratified_train_test_idx(
    _yb_bin, BINARY_MLP_TEST_FRAC, BINARY_MLP_SEED
)
_binary_ckpt_path = stonesoup.outputs_dir() / _BINARY_CKPT_NAME
_atr_exp, _ate_exp, _m_exp, _mu_exp, _sd_exp, _in_d_exp = train_binary_gen_vs_prefill_one_layer(
    _export_layer_eff,
    _BRANCH_TENSORS_BIN,
    _yb_bin,
    _tr_bin_export,
    _te_bin_export,
)
print(
    f"binary export layer {_export_layer_eff}  train={_atr_exp:.4f}  test={_ate_exp:.4f}",
    flush=True,
)
_repo_exp = stonesoup.repo_root()
_ckpt_payload_export = {
    "state_dict": _m_exp.state_dict(),
    "mu": _mu_exp,
    "sd": _sd_exp,
    "in_dim": _in_d_exp,
    "hidden_dim": int(BINARY_MLP_HIDDEN),
    "layer_index": int(_export_layer_eff),
    "label0": "prefill_collapsed",
    "label1": "generated",
    "source_script": "prefill-vs-generated-step2",
    "train_config": {
        "epochs": int(BINARY_MLP_EPOCHS),
        "lr": float(BINARY_MLP_LR),
        "batch": int(BINARY_MLP_BATCH),
        "test_frac": float(BINARY_MLP_TEST_FRAC),
        "seed": int(BINARY_MLP_SEED),
    },
    "cache_path_repo_relative": _binary_ckpt_path.relative_to(_repo_exp).as_posix(),
}
torch.save(_ckpt_payload_export, _binary_ckpt_path)
print(
    "Wrote binary MLP checkpoint:",
    _ckpt_payload_export["cache_path_repo_relative"],
    flush=True,
)
del _m_exp, _mu_exp, _sd_exp

# %% MLP binary: train every layer + plot (slow — checkpoint saved in cell above)
import matplotlib.pyplot as plt

if "train_binary_gen_vs_prefill_one_layer" not in globals():
    raise RuntimeError("Run the MLP binary export cell above first (defines helpers and ``_yb_bin``).")
if "_tr_bin_export" not in globals() or "_yb_bin" not in globals():
    raise RuntimeError(
        "Run the MLP binary export cell above first (defines ``_yb_bin``, ``_tr_bin_export``, ``_te_bin_export``)."
    )

_tr_bin, _te_bin = _tr_bin_export, _te_bin_export
_bin_train_acc: list[float] = []
_bin_test_acc: list[float] = []
for _li in range(_n_bin_layers):
    stonesoup.check_abort()
    atr, ate, _m_bin, _mu_bin, _sd_bin, _in_d = train_binary_gen_vs_prefill_one_layer(
        _li, _BRANCH_TENSORS_BIN, _yb_bin, _tr_bin, _te_bin
    )
    _bin_train_acc.append(atr)
    _bin_test_acc.append(ate)
    print(
        f"MLP binary layer {_li}/{_n_bin_layers - 1}  train={atr:.4f}  test={ate:.4f}",
        flush=True,
    )
    del _m_bin, _mu_bin, _sd_bin, _in_d

_layers_axis = list(range(_n_bin_layers))
_fig_bin, _ax_bin = plt.subplots(figsize=(9, 4.5))
_ax_bin.plot(_layers_axis, _bin_train_acc, "o-", label="train acc", ms=4)
_ax_bin.plot(_layers_axis, _bin_test_acc, "s-", label="test acc", ms=4)
_ax_bin.set_xlabel("layer (stage) index")
_ax_bin.set_ylabel("accuracy")
_ax_bin.set_title(
    "Binary MLP: generated vs prefill (collapsed) — acc vs layer\n"
    f"hidden={BINARY_MLP_HIDDEN} epochs={BINARY_MLP_EPOCHS} test_frac={BINARY_MLP_TEST_FRAC}"
)
_ax_bin.set_ylim(0.0, 1.02)
_ax_bin.grid(True, alpha=0.3)
_ax_bin.legend(loc="lower right")
_fig_bin.tight_layout()

# %% show plot
stonesoup.show(_fig_bin, basename="mlp_binary_gen_vs_prefill_acc_per_layer", dpi=120)
