# %% Imports & paths
"""Compare L2 norm distributions of input embedding rows across cached ``.pt`` files in ``data/embedding-layers``.

Caches are written by ``demo.py`` (``weight`` tensor + optional ``model_name``). Run from repo root::

    uv run python 2026-03-23-Embedding/embedding_norm_distributions.py

Or open in Stonesoup and run cells. Figures go under ``2026-03-23-Embedding/plots/``.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
LAYER_DIR = REPO_ROOT / "data" / "embedding-layers"
PLOTS_DIR = Path(__file__).resolve().parent / "plots"
OUT_NAME = "embedding_norm_distributions_cache_layers.png"

# %% Discover .pt files
pt_files = sorted(p for p in LAYER_DIR.glob("*.pt") if p.is_file())
print(f"cache dir: {LAYER_DIR}")
print(f"found {len(pt_files)} file(s)")
for p in pt_files[:20]:
    print(f"  {p.name}")
if len(pt_files) > 20:
    print(f"  ... and {len(pt_files) - 20} more")

# %% Load norms (one file at a time; keep only numpy vectors in memory)


def display_name_from_path(path: Path, payload: dict) -> str:
    m = payload.get("model_name")
    if isinstance(m, str) and m.strip():
        return m.strip()
    return path.stem.replace("__", "/")


def row_l2_norms_from_payload(payload: dict) -> np.ndarray:
    w = payload.get("weight")
    if not torch.is_tensor(w):
        raise TypeError("payload missing tensor 'weight'")
    if w.dim() != 2:
        raise ValueError(f"expected 2D weight, got shape {tuple(w.shape)}")
    return w.float().norm(dim=1).cpu().numpy()


records: list[tuple[str, Path, np.ndarray]] = []
for path in pt_files:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            print(f"skip {path.name}: not a dict checkpoint")
            continue
        name = display_name_from_path(path, payload)
        norms = row_l2_norms_from_payload(payload)
        records.append((name, path, norms))
        print(f"ok {path.name} -> {name!r}  vocab={norms.shape[0]}  mean_norm={float(norms.mean()):.4f}")
    except Exception as e:
        print(f"skip {path.name}: {e}")

# %% Subplot histograms
if not records:
    warnings.warn("No embedding caches loaded — add .pt files under data/embedding-layers or run demo.py to build caches.")
else:
    n = len(records)
    ncols = min(4, max(1, int(np.ceil(np.sqrt(n)))))
    nrows = int(np.ceil(n / ncols))
    fig_w = min(4.2 * ncols, 22)
    fig_h = min(3.0 * nrows, 30)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)
    bins = min(120, max(40, int(np.sqrt(max(r[2].size for r in records)))))

    for ax in axes.ravel():
        ax.set_visible(False)

    for i, (name, _path, norms) in enumerate(records):
        r, c = divmod(i, ncols)
        ax = axes[r][c]
        ax.set_visible(True)
        ax.hist(norms, bins=bins, color="steelblue", alpha=0.85, edgecolor="none")
        short = name if len(name) <= 48 else name[:45] + "…"
        ax.set_title(short, fontsize=9)
        ax.set_xlabel("L2 norm")
        ax.set_ylabel("count")
        ax.tick_params(axis="both", labelsize=8)

    fig.suptitle("Per-row L2 norms of cached input embedding weights", fontsize=12, y=1.02)
    fig.tight_layout()

# %% Save figure
if records:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PLOTS_DIR / OUT_NAME
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")
else:
    print("nothing to plot")
