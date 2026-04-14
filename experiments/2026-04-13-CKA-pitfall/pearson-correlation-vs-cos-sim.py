# %% Imports
from __future__ import annotations

from stonesoup.experiment import configure_matplotlib_agg, show

configure_matplotlib_agg()

import matplotlib.pyplot as plt
import numpy as np

# %% 1000 pairs: Pearson r vs cosine similarity (same tiny length-4 draws each trial)
rng = np.random.default_rng(20260413)
n_trials = 1000
dim = 100

a = rng.standard_normal((n_trials, dim))
b = rng.standard_normal((n_trials, dim))

cos_sim = (a * b).sum(axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))

ac = a - a.mean(axis=1, keepdims=True)
bc = b - b.mean(axis=1, keepdims=True)
pearson_r = (ac * bc).sum(axis=1) / (np.linalg.norm(ac, axis=1) * np.linalg.norm(bc, axis=1))

# %% Scatter: Pearson r vs cosine similarity
fig, ax = plt.subplots(figsize=(4.5, 4.5))
ax.scatter(pearson_r, cos_sim, s=8, alpha=0.45, c="C0", edgecolors="none")
ax.set_aspect("equal", adjustable="box")
lim = (-1.05, 1.05)
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.axhline(0, color="0.5", lw=0.6)
ax.axvline(0, color="0.5", lw=0.6)
ax.plot(lim, lim, "k--", lw=0.8, alpha=0.4, label="y = x")
ax.set_xlabel("Pearson correlation")
ax.set_ylabel("Cosine similarity")
ax.set_title(f"{n_trials} random pairs (dim={dim})")
ax.legend(loc="upper left", fontsize=8)
fig.tight_layout()
show(fig, basename="pearson_vs_cosine_scatter", dpi=130)
