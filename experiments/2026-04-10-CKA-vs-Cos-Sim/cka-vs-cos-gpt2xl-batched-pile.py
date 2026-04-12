# %% Imports & helpers
from __future__ import annotations

import math
import random
import time

import stonesoup
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from ckatorch import cka_base
from stonesoup.experiment import (
    capture_embed_and_post_blocks,
    configure_matplotlib_agg,
    ensure_pad_token_via_eos,
    hf_repo_id_safe_stem,
    inner_tokenizer,
)

configure_matplotlib_agg()


def _tnanmax(t: torch.Tensor) -> torch.Tensor:
    return torch.max(torch.nan_to_num(t, nan=float("-inf")))


def _tnanmin(t: torch.Tensor) -> torch.Tensor:
    return torch.min(torch.nan_to_num(t, nan=float("inf")))


def _assert_all_finite(t: torch.Tensor, *, what: str) -> None:
    assert bool(torch.isfinite(t).all().item()), f"non-finite values in {what}"


def _assert_finite_float(x: float, *, what: str) -> None:
    assert math.isfinite(x), f"non-finite float in {what}: {x}"


def layer_cka_matrix(activations: torch.Tensor) -> torch.Tensor:
    """L×L linear CKA (unbiased); activations (n_rows, n_layers, dim). Diagonal = 1."""
    if activations.ndim != 3:
        raise ValueError("activations must be (n_samples, n_layers, dim)")
    _assert_all_finite(activations, what="layer_cka_matrix activations")
    _, L, _ = activations.shape
    act = activations.detach()
    out = torch.ones(L, L, dtype=torch.float64, device=act.device)
    for i in range(L):
        stonesoup.check_abort()
        xi = act[:, i, :].to(dtype=torch.float64)
        for j in range(i + 1, L):
            stonesoup.check_abort()
            xj = act[:, j, :].to(dtype=torch.float64)
            v = cka_base(xi, xj, unbiased=True)
            assert bool(torch.isfinite(v).item()), f"non-finite CKA for layer pair ({i}, {j})"
            out[i, j] = v
            out[j, i] = v
    _assert_all_finite(out, what="layer_cka_matrix output")
    return out


def layer_mean_aligned_cos_matrix(activations: torch.Tensor) -> torch.Tensor:
    """L×L mean over samples of cos(h_i[k], h_j[k]) with per-row unit norm; diagonal 1."""
    if activations.ndim != 3:
        raise ValueError("activations must be (n_samples, n_layers, dim)")
    _assert_all_finite(activations, what="layer_mean_aligned_cos_matrix activations")
    n, L, _ = activations.shape
    h = F.normalize(activations.float(), dim=-1)
    out = torch.ones(L, L, dtype=torch.float64, device=h.device)
    for i in range(L):
        stonesoup.check_abort()
        for j in range(i + 1, L):
            stonesoup.check_abort()
            c = (h[:, i] * h[:, j]).sum(dim=-1).mean().to(dtype=torch.float64)
            out[i, j] = c
            out[j, i] = c
    _assert_all_finite(out, what="layer_mean_aligned_cos_matrix output")
    return out


def layer_r2_train_test_matrix(
    activations: torch.Tensor,
    train_idx: torch.Tensor,
    test_idx: torch.Tensor,
    *,
    ridge_alpha: float,
) -> torch.Tensor:
    """Held-out multivariate R² (asymmetric): ``out[i, j] = R²_test`` predicting **target** layer ``j``
    from **source** layer ``i`` via ridge on train rows only.

    Train column means ``μ_X``, ``μ_Y`` are computed on train rows; ``X``, ``Y`` are centered with
    those means for fitting and for test predictions. Ridge:
    ``λ = ridge_alpha * tr(X_c^T X_c) / d`` on **source train** ``X_c``. Test score:
    ``1 - ‖Y_test - Ŷ_test‖_F² / ‖Y_test - μ_Y‖_F²`` (mean baseline = train ``μ_Y``). Values may be
    negative on test. Float64 for stability.
    """
    if activations.ndim != 3:
        raise ValueError("activations must be (n_samples, n_layers, dim)")
    _assert_all_finite(activations, what="layer_r2_train_test_matrix activations")
    if ridge_alpha <= 0:
        raise ValueError("ridge_alpha must be positive")
    _, L, d = activations.shape
    act = activations.detach().to(dtype=torch.float64)
    device = act.device
    out = torch.empty(L, L, dtype=torch.float64, device=device)
    eye = torch.eye(d, dtype=torch.float64, device=device)
    n_tr = int(train_idx.numel())
    n_te = int(test_idx.numel())
    if n_tr < 1 or n_te < 1:
        raise ValueError("train_idx and test_idx must each be non-empty")

    from tqdm import tqdm

    for i in tqdm(range(L), desc="R² rows (source layer)", leave=True):
        stonesoup.check_abort()
        Xi = act.index_select(0, train_idx)[:, i, :]
        mu_x = Xi.mean(dim=0)
        Xc_tr = Xi - mu_x
        g = Xc_tr.T @ Xc_tr
        tr_g = torch.trace(g)
        lam = ridge_alpha * tr_g / float(d)
        lam = lam + 1e-18
        a = g + lam * eye
        for j in range(L):
            stonesoup.check_abort()
            Yj_tr = act.index_select(0, train_idx)[:, j, :]
            mu_y = Yj_tr.mean(dim=0)
            Yc_tr = Yj_tr - mu_y
            rhs = Xc_tr.T @ Yc_tr
            w = torch.linalg.solve(a, rhs)
            X_te = act.index_select(0, test_idx)[:, i, :]
            Y_te = act.index_select(0, test_idx)[:, j, :]
            Xc_te = X_te - mu_x
            y_hat = mu_y + Xc_te @ w
            resid = Y_te - y_hat
            sse = (resid * resid).sum()
            base = Y_te - mu_y
            sst = (base * base).sum()
            if sst <= 0:
                out[i, j] = float("nan")
            else:
                r2 = 1.0 - (sse / sst)
                out[i, j] = r2
    if not bool(torch.logical_or(torch.isfinite(out), torch.isnan(out)).all().item()):
        raise AssertionError("layer_r2_train_test_matrix: unexpected non-finite non-NaN values")
    return out


def _pile_set_name_from_row(row: dict) -> str:
    meta = row.get("meta")
    if isinstance(meta, dict):
        name = meta.get("pile_set_name")
        if name is not None:
            return str(name)
    return "<unknown>"


# %% Config, Pile rows (10 categories × 1 row), token check
# ``datasets``: ``uv pip install datasets``. Hidden size follows the checkpoint (GPT-2 XL → 1600).
# MODEL_ID = "openai-community/gpt2-xl"
# MODEL_ID = "Qwen/Qwen3.5-9B"
MODEL_ID = "google/gemma-2-2b"
# MODEL_ID = "Qwen/Qwen3-8B-Base"
# MODEL_ID = "meta-llama/llama-3.2-3B"

TOKENS_PER_ROW = 51
N_ROWS = 20
PILE_SAMPLE_SEED = 0
SKIP_FIRST_TOKENS = 1
# Held-out ridge R²: one global train/test split of token rows; ``λ = RIDGE_ALPHA * tr(X'X)/d`` on source train.
R2_SPLIT_SEED = 0
R2_TRAIN_FRAC = 0.8
RIDGE_ALPHA = 1e-2
# After drop: (N_ROWS, TOKENS_PER_ROW - 1, n_stages, dim) → N_ROWS * (TOKENS_PER_ROW - 1) position vectors.
PILE_PREVIEW_CHARS = 160

from datasets import load_dataset
from transformers import AutoTokenizer

_ds = load_dataset("NeelNanda/pile-10k", split="train")
_tok_probe = AutoTokenizer.from_pretrained(MODEL_ID)

# Full-document encode without truncation can exceed ``model_max_length`` (e.g. 3180 > 1024 for GPT-2)
# and triggers warnings / inconsistent counts. Use truncated encode only to test “≥ N tokens”.
_mxl = getattr(_tok_probe, "model_max_length", 1024)
if _mxl is None or _mxl > 10_000:
    _mxl = 1024
if _mxl < TOKENS_PER_ROW:
    raise ValueError(f"tokenizer model_max_length {_mxl} < TOKENS_PER_ROW {TOKENS_PER_ROW}")


def _has_enough_tokens(text: str, *, min_tokens: int) -> bool:
    """True if ``text`` has at least ``min_tokens`` ids (GPT-2: first ``_mxl`` tokens suffice)."""
    enc = _tok_probe(
        text,
        add_special_tokens=True,
        truncation=True,
        max_length=_mxl,
    )
    return len(enc["input_ids"]) >= min_tokens


# Index every long-enough row by ``pile_set_name``, then sample ``N_ROWS`` distinct categories and
# one random qualifying row per category (seeded).
_by_cat: dict[str, list[int]] = {}
for idx in range(len(_ds)):
    stonesoup.check_abort()
    row = _ds[idx]
    cat = _pile_set_name_from_row(row)
    text = (row.get("text") or "").strip().replace("\r\n", "\n")
    if not _has_enough_tokens(text, min_tokens=TOKENS_PER_ROW):
        continue
    _by_cat.setdefault(cat, []).append(idx)

_eligible_cats = [c for c, ids in _by_cat.items() if ids]
_rng = random.Random(PILE_SAMPLE_SEED)
if len(_eligible_cats) < N_ROWS:
    raise ValueError(
        f"Need {N_ROWS} categories with ≥1 row of ≥{TOKENS_PER_ROW} tokens; "
        f"only {len(_eligible_cats)} categories qualify in pile-10k. Lower TOKENS_PER_ROW or N_ROWS."
    )
_chosen_cats = _rng.sample(_eligible_cats, k=N_ROWS)
pile_rows: list[tuple[int, str, str]] = []
for cat in _chosen_cats:
    idx = _rng.choice(_by_cat[cat])
    row = _ds[idx]
    text = (row.get("text") or "").strip().replace("\r\n", "\n")
    pile_rows.append((idx, text, cat))

print(
    f"Pile: seed={PILE_SAMPLE_SEED}  sampled {N_ROWS} categories from {len(_eligible_cats)} eligible "
    f"(≥{TOKENS_PER_ROW} tok); one random row per category.",
    flush=True,
)
print(
    f"Model {MODEL_ID}: batch [{N_ROWS}, {TOKENS_PER_ROW}] → vectors "
    f"{N_ROWS}×({TOKENS_PER_ROW}-{SKIP_FIRST_TOKENS})={N_ROWS * (TOKENS_PER_ROW - SKIP_FIRST_TOKENS)} "
    "for CKA / mean aligned cos.",
    flush=True,
)
for idx, text, cat in pile_rows:
    print(
        f"  id={idx}  pile_set_name={cat!r}  preview={text[:PILE_PREVIEW_CHARS]!r}…",
        flush=True,
    )

# %% Load model & batch forward
model, proc = stonesoup.load_model(MODEL_ID)
model.eval()
tok = inner_tokenizer(proc)
ensure_pad_token_via_eos(tok)
device = next(model.parameters()).device
print(f"device={device}", flush=True)

texts = [t for _, t, _ in pile_rows]
enc = tok(
    texts,
    padding=True,
    truncation=True,
    max_length=TOKENS_PER_ROW,
    return_tensors="pt",
    add_special_tokens=True,
)
inputs = {k: v.to(device) for k, v in enc.items()}
masks = inputs["attention_mask"]
assert int(masks.shape[0]) == N_ROWS and int(masks.shape[1]) == TOKENS_PER_ROW
for bi in range(N_ROWS):
    assert int(masks[bi].sum().item()) == TOKENS_PER_ROW, (
        f"row {bi}: expected {TOKENS_PER_ROW} real tokens, got {int(masks[bi].sum().item())}"
    )

t0 = time.perf_counter()
stack, stage_names = capture_embed_and_post_blocks(model, inputs, use_cache=False)
print(
    f"forward {time.perf_counter() - t0:.2f}s  stack {tuple(stack.shape)}  stages={stage_names[0]}…{stage_names[-1]}",
    flush=True,
)

# stack: (n_stages, batch, seq, hidden) → drop first token → (n_stages, batch, seq-1, hidden)
sl0 = int(inputs["attention_mask"][0].sum().item())
st_end = min(sl0, TOKENS_PER_ROW)
stack_trim = stack[:, :, SKIP_FIRST_TOKENS:st_end, :].contiguous()
# (n_stages, N_ROWS, TOKENS_PER_ROW - SKIP_FIRST_TOKENS, dim)
assert stack_trim.shape[2] == TOKENS_PER_ROW - SKIP_FIRST_TOKENS
n_stage = int(stack_trim.shape[0])
hidden = int(stack_trim.shape[3])
# (N_ROWS, seq-1, n_stages, dim) → (N_ROWS * (seq-1), n_stages, dim)
act_bqld = stack_trim.permute(1, 2, 0, 3).contiguous()
ACTIVATIONS_BQ_L = act_bqld  # keep name for inspection
act = act_bqld.reshape(-1, n_stage, hidden).detach().float()
_assert_all_finite(act, what="batched activations")
assert act.shape[0] == N_ROWS * (TOKENS_PER_ROW - SKIP_FIRST_TOKENS)
print(
    f"activations for CKA/cos: {tuple(act.shape)}  (n={act.shape[0]}, L={n_stage}, d={hidden})",
    flush=True,
)

_ns = int(act.shape[0])
if _ns < 2:
    raise ValueError("Need at least 2 token positions for R² train/test split")
_r2_rng = random.Random(R2_SPLIT_SEED)
_ord = list(range(_ns))
_r2_rng.shuffle(_ord)
_n_tr = max(1, min(int(R2_TRAIN_FRAC * _ns), _ns - 1))
train_idx = torch.tensor(_ord[:_n_tr], dtype=torch.long, device=act.device)
test_idx = torch.tensor(_ord[_n_tr:], dtype=torch.long, device=act.device)
print(
    f"R² split: seed={R2_SPLIT_SEED}  train={int(train_idx.numel())}  test={int(test_idx.numel())}  "
    f"RIDGE_ALPHA={RIDGE_ALPHA}  (λ = α·tr(X'X)/d per source layer)",
    flush=True,
)

# %% Linear unbiased CKA + mean aligned cosine + held-out ridge R² (layer × layer)
print(
    f"layer CKA (L×L), n={act.shape[0]}, pairwise Gram work ~ O(n²) in ckatorch …",
    flush=True,
)
t_cka = time.perf_counter()
cka_m = layer_cka_matrix(act)
print(f"CKA done {time.perf_counter() - t_cka:.2f}s", flush=True)

t_cos = time.perf_counter()
cos_m = layer_mean_aligned_cos_matrix(act)
print(f"mean aligned cos done {time.perf_counter() - t_cos:.2f}s", flush=True)

t_r2 = time.perf_counter()
r2_m = layer_r2_train_test_matrix(act, train_idx, test_idx, ridge_alpha=RIDGE_ALPHA)
print(f"held-out ridge R² (test rows) done {time.perf_counter() - t_r2:.2f}s", flush=True)

_ck_lo = float(_tnanmin(cka_m))
_ck_hi = float(_tnanmax(cka_m))
_co_lo = float(_tnanmin(cos_m))
_co_hi = float(_tnanmax(cos_m))
_assert_finite_float(_ck_lo, what="cka min")
_assert_finite_float(_ck_hi, what="cka max")
_assert_finite_float(_co_lo, what="cos min")
_assert_finite_float(_co_hi, what="cos max")

# %% Plot: CKA vs mean aligned cos vs held-out R²
L = int(cka_m.shape[0])
_n_samples = int(act.shape[0])
_n_tr = int(train_idx.numel())
_n_te = int(test_idx.numel())
safe = hf_repo_id_safe_stem(MODEL_ID)
ck_pad = 0.05 * (_ck_hi - _ck_lo + 1e-9)
co_pad = 0.05 * (_co_hi - _co_lo + 1e-9)
_r2_fin = torch.isfinite(r2_m)
if bool(_r2_fin.any().item()):
    _r2_lo = float(torch.min(r2_m[_r2_fin]))
    _r2_hi = float(torch.max(r2_m[_r2_fin]))
else:
    _r2_lo, _r2_hi = -1.0, 1.0
_r2_pad = 0.05 * (_r2_hi - _r2_lo + 1e-9)

fig, axes = plt.subplots(1, 3, figsize=(21.0, 6.2), constrained_layout=True)

im0 = axes[0].imshow(
    cka_m.detach().cpu().numpy(),
    vmin=_ck_lo - ck_pad,
    vmax=_ck_hi + ck_pad,
    cmap="Blues",
    aspect="equal",
    interpolation="nearest",
)
axes[0].set_title(
    f"Linear CKA (unbiased, ckatorch)\n"
    f"same {_n_samples}×d token–position rows per layer; CKA compares layer pairs on these samples"
)
axes[0].set_xlabel("layer j")
axes[0].set_ylabel("layer i")
fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

im1 = axes[1].imshow(
    cos_m.detach().cpu().numpy(),
    vmin=_co_lo - co_pad,
    vmax=_co_hi + co_pad,
    cmap="Blues",
    aspect="equal",
    interpolation="nearest",
)
axes[1].set_title(
    "Mean aligned cosine (layer × layer)\n"
    + rf"$\frac{{1}}{{n}}\sum_k \cos(h_i^{{(k)}}, h_j^{{(k)}})$, same $n={_n_samples}$ positions"
)
axes[1].set_xlabel("layer j")
axes[1].set_ylabel("layer i")
fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

im2 = axes[2].imshow(
    r2_m.detach().cpu().numpy(),
    # vmin=_r2_lo - _r2_pad,
    # vmax=_r2_hi + _r2_pad,
    cmap="Blues",
    aspect="equal",
    interpolation="nearest",
    vmin=0,
    vmax=1,
)
axes[2].set_title(
    f"Held-out multivariate R² (ridge, test rows)\n"
    f"row = source layer i → column = target layer j; "
    f"train μ, λ=α·tr(X'X)/d, α={RIDGE_ALPHA}, split seed={R2_SPLIT_SEED}"
)
axes[2].set_xlabel("target layer j")
axes[2].set_ylabel("source layer i")
fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label="R² (test)")

fig.suptitle(
    f"{MODEL_ID}  |  NeelNanda/pile-10k: {N_ROWS} long rows, one per distinct meta.pile_set_name\n"
    f"Batch [{N_ROWS}×{TOKENS_PER_ROW}] tokens; drop first token → {_n_samples} vectors × "
    f"{L} stages × {hidden}d; R²: n_train={_n_tr}, n_test={_n_te}, frac={R2_TRAIN_FRAC}",
    fontsize=9,
)
stonesoup.show(
    fig,
    basename=f"{safe}_layer_cka_cos_r2ridge_holdout_batch{N_ROWS}x{TOKENS_PER_ROW}_pile10k",
    dpi=120,
)
