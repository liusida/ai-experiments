# %% Imports & helpers
from __future__ import annotations

import stonesoup
from stonesoup.experiment import configure_matplotlib_agg

configure_matplotlib_agg()

import matplotlib.pyplot as plt
import numpy as np


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA (column-centered). X, Y: (n, d) with aligned rows."""
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    xty = X.T @ Y
    xtx = X.T @ X
    yty = Y.T @ Y
    num = np.linalg.norm(xty, ord="fro") ** 2
    den = np.linalg.norm(xtx, ord="fro") * np.linalg.norm(yty, ord="fro")
    return float(num / den) if den > 0 else float("nan")


def cos_flattened(A: np.ndarray, B: np.ndarray) -> float:
    """Cosine similarity between flattened (n,d) matrices."""
    a = A.ravel()
    b = B.ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else float("nan")


def layer_cka_matrix(activations: np.ndarray) -> np.ndarray:
    """L×L with K_ij = linear CKA(layer i, layer j); activations shape (n, L, d)."""
    _, L, _ = activations.shape
    out = np.zeros((L, L), dtype=np.float64)
    for i in range(L):
        for j in range(L):
            out[i, j] = linear_cka(activations[:, i, :], activations[:, j, :])
    return out


def layer_cos_flat_matrix(activations: np.ndarray) -> np.ndarray:
    """L×L with C_ij = cos(vec(layer i), vec(layer j))."""
    _, L, _ = activations.shape
    out = np.zeros((L, L), dtype=np.float64)
    for i in range(L):
        for j in range(L):
            out[i, j] = cos_flattened(activations[:, i, :], activations[:, j, :])
    return out


def random_orthogonal(d: int, rng: np.random.Generator) -> np.ndarray:
    q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    return q


def activations_shared_backbone(
    rng: np.random.Generator,
    n_samp: int,
    n_layer: int,
    dim: int,
    *,
    noise_scale: float,
) -> np.ndarray:
    """Build (n, L, d) where each layer shares the same backbone (n, d) plus iid noise.

    IID Gaussian layers in high d look nearly orthogonal; a shared component gives
    meaningful off-diagonal cosine / CKA before any rotation.
    """
    shared = rng.standard_normal((n_samp, dim))
    X = np.empty((n_samp, n_layer, dim), dtype=np.float64)
    for ell in range(n_layer):
        X[:, ell, :] = shared + noise_scale * rng.standard_normal((n_samp, dim))
    return X


# %% Data: activations (n, layer, dim); Y rotates features only on some layers (same R)
rng = np.random.default_rng(0)
n_samp, n_layer, dim = 64, 10, 24
# Smaller noise_scale → stronger layer–layer similarity; larger → closer to iid (near-orthogonal).
NOISE_SCALE = 0.7
X = activations_shared_backbone(rng, n_samp, n_layer, dim, noise_scale=NOISE_SCALE)
R = random_orthogonal(dim, rng)
# e.g. even layers get X @ R, odd layers unchanged — mixed bases across layers
ROTATED_LAYERS = {i for i in range(0, n_layer, 2)}
Y = X.copy()
for ell in ROTATED_LAYERS:
    Y[:, ell, :] = X[:, ell, :] @ R

cka_x = layer_cka_matrix(X)
cka_y = layer_cka_matrix(Y)
cos_x = layer_cos_flat_matrix(X)
cos_y = layer_cos_flat_matrix(Y)

off = cos_x[~np.eye(n_layer, dtype=bool)]
print(
    f"activations shape (n, layer, dim) = {X.shape}, backbone + N(0,{NOISE_SCALE}), R shape = {R.shape}",
)
print(f"mean off-diag cos(X) layers: {float(off.mean()):.3f}  (for reference; iid layers ≈ 0 in high-d)")
print(f"layers with Y[...,l,:] = X[...,l,:] @ R: {sorted(ROTATED_LAYERS)}")
print(f"max |CKA(Y) − CKA(X)|: {np.max(np.abs(cka_y - cka_x)):.3e}  (linear CKA invariant to column orthogonal per layer)")
print(f"max |cos_flat(Y) − cos_flat(X)|: {np.max(np.abs(cos_y - cos_x)):.6f}  (flattened cos can change when layers use different transforms)")

# %% Plot: layer×layer similarity — X vs mixed-rotation Y
fig, axes = plt.subplots(2, 2, figsize=(10.5, 9), constrained_layout=True)

vlim_cka = 1.0
im00 = axes[0, 0].imshow(cka_x, vmin=0, vmax=vlim_cka, cmap="Blues", aspect="equal")
axes[0, 0].set_title("Linear CKA between layers (X)")
axes[0, 0].set_xlabel("layer j")
axes[0, 0].set_ylabel("layer i")
fig.colorbar(im00, ax=axes[0, 0], fraction=0.046, pad=0.04)

im01 = axes[0, 1].imshow(cka_y, vmin=0, vmax=vlim_cka, cmap="Blues", aspect="equal")
axes[0, 1].set_title("Linear CKA between layers (Y: R on even layers only)")
axes[0, 1].set_xlabel("layer j")
axes[0, 1].set_ylabel("layer i")
fig.colorbar(im01, ax=axes[0, 1], fraction=0.046, pad=0.04)

cmin = min(float(cos_x.min()), float(cos_y.min()))
cmax = max(float(cos_x.max()), float(cos_y.max()))
cpad = 0.05 * (cmax - cmin + 1e-9)

im10 = axes[1, 0].imshow(cos_x, vmin=cmin - cpad, vmax=cmax + cpad, cmap="Blues", aspect="equal")
axes[1, 0].set_title(r"Cosine of flattened layers (X): $\cos(\mathrm{vec}(X^{(i)}), \mathrm{vec}(X^{(j)}))$")
axes[1, 0].set_xlabel("layer j")
axes[1, 0].set_ylabel("layer i")
fig.colorbar(im10, ax=axes[1, 0], fraction=0.046, pad=0.04)

im11 = axes[1, 1].imshow(cos_y, vmin=cmin - cpad, vmax=cmax + cpad, cmap="Blues", aspect="equal")
axes[1, 1].set_title("Cosine of flattened layers (Y: mixed rotation)")
axes[1, 1].set_xlabel("layer j")
axes[1, 1].set_ylabel("layer i")
fig.colorbar(im11, ax=axes[1, 1], fraction=0.046, pad=0.04)

fig.suptitle(
    "Orthogonal R on even layers only (odd layers = X). "
    "Linear CKA between layers unchanged; flattened cosine between vec(layer) changes across mixed bases.",
    fontsize=11,
)
stonesoup.show(fig, basename="cka_vs_cos_rotation", dpi=120)
