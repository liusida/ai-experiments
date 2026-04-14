# %% Imports
from __future__ import annotations

from stonesoup.experiment import configure_matplotlib_agg, show

configure_matplotlib_agg()

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from ckatorch import cka_base


def _boxplot_tick_labels_kw(names: list[str]) -> dict[str, list[str]]:
    # tick_labels since Matplotlib 3.9; labels deprecated, removed in 3.11
    major, minor = (int(x) for x in matplotlib.__version__.split(".")[:2])
    key = "tick_labels" if (major, minor) >= (3, 9) else "labels"
    return {key: names}

# %% X, X′ (first row × 10), linear CKA
rng = torch.Generator().manual_seed(20260413)
n_samples = 256
n_features = 64

X = torch.randn(n_samples, n_features, generator=rng, dtype=torch.float64)
X_prime = X.clone()
X_prime[0] = X_prime[0] * 100

cka_self = cka_base(X, X, kernel="linear").item()
cka_xxp = cka_base(X, X_prime, kernel="linear").item()

print(f"Linear CKA(X, X)   = {cka_self:.6f}", flush=True)
print(f"Linear CKA(X, X′)  = {cka_xxp:.6f}   (X′ equals X except row 0 is multiplied by 10)", flush=True)

# %% Box plots: per-row L2 norms (broken y-axis so the outlier does not squash the bulk)
row_norm_x = torch.linalg.vector_norm(X, dim=1).numpy()
row_norm_xp = torch.linalg.vector_norm(X_prime, dim=1).numpy()

# Second-largest norm in X′ is bulk ceiling; largest is the scaled row.
xp_sorted = np.sort(row_norm_xp)
outlier_y = float(xp_sorted[-1])
bulk_hi_xp = float(xp_sorted[-2])
y_lo = float(min(row_norm_x.min(), row_norm_xp.min()) * 0.97)
y_hi_bulk = max(float(row_norm_x.max()), bulk_hi_xp) * 1.06

fig = plt.figure(figsize=(4.5, 4.2))
gs = fig.add_gridspec(2, 1, height_ratios=[1, 2.6], hspace=0.06)
ax_top = fig.add_subplot(gs[0])
ax_bot = fig.add_subplot(gs[1], sharex=ax_top)

bp = ax_bot.boxplot(
    [row_norm_x, row_norm_xp],
    patch_artist=True,
    widths=0.55,
    showfliers=False,
    **_boxplot_tick_labels_kw(["X", "X′"]),
)
for patch, color in zip(bp["boxes"], ["C0", "C2"]):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax_bot.set_ylim(y_lo, y_hi_bulk)
ax_bot.set_ylabel("Per-row ‖·‖₂")

# Outlier alone on the upper segment (same x as the X′ box, position 2)
pad = (outlier_y - bulk_hi_xp) * 0.15 + 1e-6
ax_top.set_ylim(outlier_y - pad, outlier_y + pad)
ax_top.scatter([2], [outlier_y], c="C3", s=42, zorder=5, clip_on=False)
ax_top.tick_params(axis="x", labelbottom=False)
ax_top.spines.bottom.set_visible(False)

ax_bot.spines.top.set_visible(False)

# Diagonal “break” ticks on the left (matplotlib broken-axis pattern)
d = 0.015
kwargs = dict(transform=ax_bot.transAxes, color="k", clip_on=False, linewidth=0.9)
ax_bot.plot((-d, +d), (1 - d, 1 + d), **kwargs)
ax_bot.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
kwargs_t = dict(transform=ax_top.transAxes, color="k", clip_on=False, linewidth=0.9)
ax_top.plot((-d, +d), (-d, +d), **kwargs_t)
ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs_t)

# ax_bot.set_title sits in the thin gap between panels and is often clipped; use suptitle + top margin.
# fig.subplots_adjust(left=0.18, right=0.96, top=0.78, bottom=0.14)
fig.suptitle(
    f"CKA ( X, X' ) = {cka_xxp:.6f}    X.shape = {tuple(X.shape)}",
    fontsize=11,
    y=0.98,
)
# pad_inches: stonesoup.show uses bbox_inches="tight"; padding keeps suptitle inside the saved PNG
show(fig, basename="cka_pitfall_box_row_norms", dpi=130, pad_inches=0.35)
