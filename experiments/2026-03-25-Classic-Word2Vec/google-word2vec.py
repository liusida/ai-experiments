"""Explore pre-trained Google News Word2Vec (300d) vectors.

**Deps:** ``gensim`` is listed in the repo ``pyproject.toml``; use ``uv sync`` (and
``--all-extras`` if you use Stonesoup / UMAP experiments).

First run downloads ~1.6 GiB via gensim-data (cached under ``~/gensim-data`` by default).

**Stonesoup:** Watch this file and run cells top to bottom (kernel keeps globals).

**Terminal:** ``uv run python experiments/2026-03-25-Classic-Word2Vec/google-word2vec.py``
"""

from __future__ import annotations

# %% Imports & paths

import gensim.downloader as api
import numpy as np


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity for two 1-D vectors (L2-normalized dot product)."""
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))

# %% Load Google News 300d (download on first use)

# Key name from gensim-data registry; vectors are 300-dim, ~3M vocabulary.
VECTORS = api.load("word2vec-google-news-300")

print(type(VECTORS))
print("Vector size (d):", VECTORS.vector_size)
print("Vocabulary size:", len(VECTORS.key_to_index))

# %% Vector for `king`

word = "king"
vec = VECTORS[word]  # numpy ndarray, float32, shape (300,)

print(f"{word!r}: shape {vec.shape}, dtype {vec.dtype}")
print(f"L2 norm: {float(np.linalg.norm(vec)):.6f}")
print("First 12 values:", vec[:12])
print("Last 12 values:", vec[-12:])
print("Full vector:\n", vec)

# %% Quick sanity: analogy + neighbors

# Classic demo: king - man + woman ≈ queen
pairs = VECTORS.most_similar(positive=["king", "woman"], negative=["man"], topn=8)
print('most_similar(positive=["king","woman"], negative=["man"]):')
for w, score in pairs:
    print(f"  {w:20s}  {score:.4f}")

print()
neighbors = VECTORS.most_similar("king", topn=10)
print('most_similar("king"):')
for w, score in neighbors:
    print(f"  {w:20s}  {score:.4f}")

# %% Explicit king − man + woman vs king and queen

v_king = VECTORS["king"].astype(np.float64)
v_man = VECTORS["man"].astype(np.float64)
v_woman = VECTORS["woman"].astype(np.float64)
v_queen = VECTORS["queen"].astype(np.float64)

# Raw vector analogy (same geometry as most_similar positive/negative, before any mean).
v_analogy = v_king - v_man + v_woman

sim_to_king = _cosine(v_analogy, v_king)
sim_to_queen = _cosine(v_analogy, v_queen)
sim_king_queen = _cosine(v_king, v_queen)

print("Explicit: v_analogy = king - man + woman  (float64 arithmetic)")
print(f"  cos(v_analogy, king) :   {sim_to_king:.6f}")
print(f"  cos(v_analogy, queen):   {sim_to_queen:.6f}")
print(f"  cos(king, queen)       :   {sim_king_queen:.6f}  (reference)")
print("  L2 norms:")
print(f"    ||king||:     {float(np.linalg.norm(v_king)):.6f}")
print(f"    ||queen||:    {float(np.linalg.norm(v_queen)):.6f}")
print(f"    ||man||:      {float(np.linalg.norm(v_man)):.6f}")
print(f"    ||woman||:    {float(np.linalg.norm(v_woman)):.6f}")
print(f"    ||v_analogy||: {float(np.linalg.norm(v_analogy)):.6f}")

# %% v1 = king − queen, v2 = man − woman, v3 = uncle − aunt

v1 = VECTORS["king"].astype(np.float64) - VECTORS["queen"].astype(np.float64)
v2 = VECTORS["man"].astype(np.float64) - VECTORS["woman"].astype(np.float64)
v3 = VECTORS["uncle"].astype(np.float64) - VECTORS["aunt"].astype(np.float64)

print("Difference vectors (float64)")
print(f"  v1 = king − queen     ||v1|| = {float(np.linalg.norm(v1)):.6f}")
print(f"  v2 = man − woman       ||v2|| = {float(np.linalg.norm(v2)):.6f}")
print(f"  v3 = uncle − aunt      ||v3|| = {float(np.linalg.norm(v3)):.6f}")
print("  Pairwise cosines:")
print(f"    cos(v1, v2): {_cosine(v1, v2):.6f}")
print(f"    cos(v1, v3): {_cosine(v1, v3):.6f}")
print(f"    cos(v2, v3): {_cosine(v2, v3):.6f}")

# %% base − (v1+v2+v3)/3 vs matching pair (king/queen, man/woman, uncle/aunt)

vk = VECTORS["king"].astype(np.float64)
vq = VECTORS["queen"].astype(np.float64)
vm = VECTORS["man"].astype(np.float64)
vw = VECTORS["woman"].astype(np.float64)
vu = VECTORS["uncle"].astype(np.float64)
va = VECTORS["aunt"].astype(np.float64)
v1_kq = vk - vq
v2_mw = vm - vw
v3_ua = vu - va
avg_diff = (v1_kq + v2_mw + v3_ua) / 3.0
v_king_adj = vk - avg_diff
v_man_adj = vm - avg_diff
v_uncle_adj = vu - avg_diff

hdr = "base − (v1+v2+v3)/3  (v1=king−queen, v2=man−woman, v3=uncle−aunt)"
print(hdr)
rows = (
    ("king", v_king_adj, vk, vq, "king", "queen"),
    ("man", v_man_adj, vm, vw, "man", "woman"),
    ("uncle", v_uncle_adj, vu, va, "uncle", "aunt"),
)
for label, vadj, ra, rb, na, nb in rows:
    print(f"  [{label}] vs {na} / {nb}")
    print(f"    cos(·, {na}) :       {_cosine(vadj, ra):.6f}")
    print(f"    cos(·, {nb}) :       {_cosine(vadj, rb):.6f}")

# %% Neighbor counts from pairwise cosine (vocab subsample)

# Full ~3M × ~3M cosines is infeasible; sample N words and count, for each word,
# how many *other* sampled words have cos sim above NEIGHBOR_COS_THRESHOLD.
# RAM: similarity matrix S is n×n float32 → ~4 n² bytes (e.g. n=95k → ~34 GiB).

from pathlib import Path

import matplotlib.pyplot as plt

NEIGHBOR_COS_THRESHOLD = 0.8
N_WORDS_SAMPLE = 95_000
RNG_SEED = 0
# Plot: neighbor counts k = 0..HIST_X_MAX exactly (frequencies); tail k > HIST_X_MAX omitted from plot.
HIST_X_MAX = 10
NEIGHBOR_EXAMPLES_PER_K = 3

rng = np.random.default_rng(RNG_SEED)
voc_size = len(VECTORS)
idx = rng.choice(voc_size, size=min(N_WORDS_SAMPLE, voc_size), replace=False)
words_ns = [VECTORS.index_to_key[int(i)] for i in idx]

M = np.asarray(VECTORS[words_ns], dtype=np.float32)
norms = np.linalg.norm(M, axis=1, keepdims=True)
norms = np.where(norms > 0, norms, 1.0)
M_u = M / norms
S = M_u @ M_u.T
mask = S > NEIGHBOR_COS_THRESHOLD
np.fill_diagonal(mask, False)
neighbor_counts = np.sum(mask, axis=1).astype(np.int32)
del S, mask
n_sample = int(neighbor_counts.shape[0])

for k in range(HIST_X_MAX + 1):
    idxs = np.flatnonzero(neighbor_counts == k)
    if idxs.size == 0:
        continue
    n_pick = min(NEIGHBOR_EXAMPLES_PER_K, idxs.size)
    picked = rng.choice(idxs, size=n_pick, replace=False)
    print(f"\n--- Example words with exactly k = {k} neighbor(s) (cos > {NEIGHBOR_COS_THRESHOLD}) — {n_pick} of {idxs.size} ---")
    for pos in picked:
        w = words_ns[int(pos)]
        sims = M_u @ M_u[int(pos)]
        neigh = [
            (words_ns[j], float(sims[j]))
            for j in range(n_sample)
            if j != int(pos) and sims[j] > NEIGHBOR_COS_THRESHOLD
        ]
        neigh.sort(key=lambda x: -x[1])
        print(f"  {w!r}:")
        if not neigh:
            print("    (none — no other sampled word above threshold)")
        else:
            for wj, s in neigh:
                print(f"    {wj!r}  {s:.4f}")

del M_u, M

print(
    f"Sample: {len(words_ns)} words · threshold cos > {NEIGHBOR_COS_THRESHOLD} · "
    f"neighbors: min {neighbor_counts.min()}  max {neighbor_counts.max()}  "
    f"mean {neighbor_counts.mean():.2f}",
)

n_tail = int(np.sum(neighbor_counts > HIST_X_MAX))
xs = np.arange(HIST_X_MAX + 1, dtype=np.int32)
freq = np.array([(neighbor_counts == k).sum() for k in xs], dtype=np.int64)
pct = 100.0 * freq.astype(np.float64) / max(n_sample, 1)
print(
    f"Exact share in 0..{HIST_X_MAX} (of n={n_sample} sampled words) · "
    f"{n_tail} words with k > {HIST_X_MAX} (not plotted)",
)
for k, f, p in zip(xs, freq, pct, strict=True):
    print(f"  k = {k} neighbors → {f} words ({p:.4f}%)")

fig, ax = plt.subplots(figsize=(8, 4))
y_plot = pct.copy()
y_plot[y_plot <= 0] = np.nan
ax.scatter(xs, y_plot, s=36, color="steelblue", edgecolors="black", linewidths=0.6, zorder=3)
ax.plot(xs, y_plot, color="steelblue", alpha=0.35, linewidth=1, zorder=2)
ax.set_xticks(xs)
ax.set_xlim(-0.5, HIST_X_MAX + 0.5)
ax.set_yscale("log")
ax.set_xlabel(
    f"Neighborhood size k — exactly k other sampled words with cos sim > {NEIGHBOR_COS_THRESHOLD}",
)
ax.set_ylabel("% of sampled words (log₁₀ scale)")
ax.set_title(
    "High-similarity neighbors per word (random subsample; k > "
    f"{HIST_X_MAX} omitted; y is log)",
)
ax.grid(True, which="both", alpha=0.35)
fig.tight_layout()
plot_dir = Path(__file__).resolve().parent / "plots"
plot_dir.mkdir(parents=True, exist_ok=True)
out_png = plot_dir / "neighbor_count_histogram_cos08_sample.png"
fig.savefig(out_png, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Wrote {out_png}")
