# %% Imports & helpers
from __future__ import annotations

import math
import time

import stonesoup
import matplotlib.pyplot as plt
import torch
from ckatorch import cka_base
from stonesoup.experiment import (
    capture_embed_and_post_blocks,
    configure_matplotlib_agg,
    encode_text_inputs,
    ensure_pad_token_via_eos,
    hf_repo_id_safe_stem,
    inner_tokenizer,
)


def _tnanmax(t: torch.Tensor) -> torch.Tensor:
    """Reduce max ignoring NaNs (some torch builds omit ``torch.nanmax``)."""
    return torch.max(torch.nan_to_num(t, nan=float("-inf")))


def _tnanmin(t: torch.Tensor) -> torch.Tensor:
    """Reduce min ignoring NaNs (some torch builds omit ``torch.nanmin``)."""
    return torch.min(torch.nan_to_num(t, nan=float("inf")))


def _assert_all_finite(t: torch.Tensor, *, what: str) -> None:
    assert bool(torch.isfinite(t).all().item()), f"non-finite values in {what}"


def _assert_finite_float(x: float, *, what: str) -> None:
    assert math.isfinite(x), f"non-finite float in {what}: {x}"


configure_matplotlib_agg()

def layer_cka_matrix(activations: torch.Tensor) -> torch.Tensor:
    """L×L linear CKA via ckatorch; activations (n_rows, n_layers, dim).

    Rows may be last-token-per-sentence or **all valid token positions** (concatenated sequences).

    Diagonal is **1** by definition (we do not call ``cka_base(X,X)``: unbiased centering + small *n*
    can make ‖K̃‖_F vanish in float, so the library ratio becomes NaN).

    Off-diagonals use ``cka_base(..., unbiased=True)``; asserts fire if the value is non-finite.
    """
    if activations.ndim != 3:
        raise ValueError("activations must be (n_samples, n_layers, dim)")
    _assert_all_finite(activations, what="layer_cka_matrix activations")
    _, L, _ = activations.shape
    act = activations.detach()
    # ckatorch follows tensor device; keep activations on GPU when captured there (much faster for large n).
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


def capture_last_token_act(
    model,
    proc,
    device,
    sentences: list[str],
) -> tuple[torch.Tensor, torch.Tensor, list[int], list[str] | None]:
    """Return (act, act_full, seq_lens, stage_names) with act (sample, layer, dim) for last real token."""
    per_seq: list[torch.Tensor] = []
    seq_lens: list[int] = []
    stage_names: list[str] | None = None
    for sentence in sentences:
        stonesoup.check_abort()
        inputs = encode_text_inputs(proc, sentence, device=device)
        stack, stage_names = capture_embed_and_post_blocks(model, inputs, use_cache=False)
        seq_len = int(inputs["attention_mask"][0].sum().item())
        sl = max(1, seq_len)
        st = stack[:, 0, :sl, :].detach().float()
        tok_layer_dim = st.permute(1, 0, 2).contiguous()
        _assert_all_finite(tok_layer_dim, what="capture_last_token_act per-sequence stack")
        per_seq.append(tok_layer_dim)
        seq_lens.append(sl)

    n_samp = len(sentences)
    max_seq = max(seq_lens)
    n_stage, dim = per_seq[0].shape[1], per_seq[0].shape[2]
    act_full = torch.full(
        (n_samp, max_seq, n_stage, dim), float("nan"), dtype=torch.float32, device=device
    )
    for si, arr in enumerate(per_seq):
        t = arr.shape[0]
        act_full[si, :t, :, :] = arr

    act = torch.stack([act_full[si, seq_lens[si] - 1, :, :] for si in range(n_samp)], dim=0)
    _assert_all_finite(act, what="capture_last_token_act last-token activations (act)")
    return act, act_full, seq_lens, stage_names


# %% Config & The Pile (1000 rows, grouped by ``meta["pile_set_name"]``)
# Needs: ``uv pip install datasets``. ``NeelNanda/pile-10k`` is a snapshot from The Pile; full ``EleutherAI/pile`` hub streams are often broken.
MODEL_ID = "openai-community/gpt2-xl"

from collections import defaultdict

from datasets import load_dataset

PILE_N_ROWS = 1000
# Keep a category only if it has **strictly more than** this many chunks (i.e. 101+ for default 100).
PILE_MIN_CHUNKS_PER_SET = 100
PILE_MAX_CHARS = 512


def _pile_row_text(row: dict) -> str:
    t = (row.get("text") or "").strip().replace("\r\n", "\n")
    if len(t) > PILE_MAX_CHARS:
        t = t[:PILE_MAX_CHARS] + "…"
    return t


def _pile_set_name(row: dict) -> str:
    meta = row.get("meta")
    if isinstance(meta, dict):
        name = meta.get("pile_set_name")
        if name is not None:
            return str(name)
    return "<unknown>"


_ds_pile = load_dataset("NeelNanda/pile-10k", split=f"train[:{PILE_N_ROWS}]")
_by_name: dict[str, list[str]] = defaultdict(list)
for i in range(len(_ds_pile)):
    stonesoup.check_abort()
    row = _ds_pile[i]
    _by_name[_pile_set_name(row)].append(_pile_row_text(row))

_all_sorted = sorted(_by_name.keys(), key=lambda k: (-len(_by_name[k]), k))
_excluded = [k for k in _all_sorted if len(_by_name[k]) <= PILE_MIN_CHUNKS_PER_SET]
PILE_SET_NAMES = [k for k in _all_sorted if len(_by_name[k]) > PILE_MIN_CHUNKS_PER_SET]
PILE_SETS: list[list[str]] = [_by_name[k] for k in PILE_SET_NAMES]
if not PILE_SET_NAMES:
    raise ValueError(
        f"No categories left after filter (need >{PILE_MIN_CHUNKS_PER_SET} chunks each). "
        "Lower PILE_MIN_CHUNKS_PER_SET or raise PILE_N_ROWS."
    )

print(
    f"Pile: NeelNanda/pile-10k train[:{PILE_N_ROWS}] → {len(_by_name)} raw categories; "
    f"keep {len(PILE_SET_NAMES)} with >{PILE_MIN_CHUNKS_PER_SET} chunks "
    f"(max {PILE_MAX_CHARS} chars/row); order = chunk count desc, then name",
    flush=True,
)
if _excluded:
    print(
        f"Excluded (≤{PILE_MIN_CHUNKS_PER_SET} chunks), desc by count:",
        flush=True,
    )
    for name in _excluded:
        c = len(_by_name[name])
        print(f"  {c:3d}  {name!r}", flush=True)
print(f"Kept categories — chunks (desc by count; ties by name):", flush=True)
_n_total_kept = 0
for name in PILE_SET_NAMES:
    c = len(_by_name[name])
    _n_total_kept += c
    print(f"  {c:3d}  {name!r}", flush=True)
print(
    f"  ---\n  {_n_total_kept} chunks in kept sets (split length {len(_ds_pile)} total rows)",
    flush=True,
)

for s, name in enumerate(PILE_SET_NAMES):
    grp = PILE_SETS[s]
    preview = grp[0].replace("\n", " ")[:100]
    print(f"  [{s}] {name!r} · first chunk preview → {preview!r}…", flush=True)

# %% Load model
model, proc = stonesoup.load_model(MODEL_ID)
model.eval()
tok = inner_tokenizer(proc)
ensure_pad_token_via_eos(tok)
device = next(model.parameters()).device
print(f"device={device}  pile_sets={len(PILE_SETS)}", flush=True)

# %% Capture last-token activations + CKA per ``pile_set_name`` (one heatmap each)
stage_names: list[str] | None = None
cka_by_set: list[torch.Tensor] = []
n_rows_by_set: list[int] = []

_n_sets = len(PILE_SETS)
for si, sentences in enumerate(PILE_SETS):
    stonesoup.check_abort()
    t_step = time.perf_counter()
    label = PILE_SET_NAMES[si]
    print(
        f"[pile CKA] set {si + 1}/{_n_sets} {label!r}: forward {len(sentences)} rows…",
        flush=True,
    )
    t_fwd = time.perf_counter()
    act_last, _act_full, _seq_lens, stage_names = capture_last_token_act(model, proc, device, sentences)
    _assert_all_finite(act_last, what=f"act pile set {si}")
    n_tok = int(act_last.shape[0])
    n_rows_by_set.append(n_tok)
    n2 = n_tok * n_tok
    print(
        f"[pile CKA] set {si + 1}/{_n_sets}: forward {time.perf_counter() - t_fwd:.1f}s → "
        f"n={n_tok} last-token rows (Gram n×n; n²={n2:,}), layer CKA (L×L)…",
        flush=True,
    )
    t_cka = time.perf_counter()
    cka_m = layer_cka_matrix(act_last)
    _assert_all_finite(cka_m, what=f"cka pile set {si}")
    cka_by_set.append(cka_m)
    print(
        f"[pile CKA] set {si + 1}/{_n_sets}: CKA {time.perf_counter() - t_cka:.1f}s  "
        f"step total {time.perf_counter() - t_step:.1f}s  "
        f"n={n_tok} n²={n2:,}  act_last {tuple(act_last.shape)}",
        flush=True,
    )

n_samp, n_stage, dim = act_last.shape
print(
    f"last capture: (sample, layer, dim)=({n_samp}, {n_stage}, {dim})  "
    f"stages: {stage_names[0] if stage_names else '?'} … {stage_names[-1] if stage_names else '?'}",
    flush=True,
)

# %% Plot: one heatmap per ``pile_set_name`` (shared color scale, dynamic grid)
safe = hf_repo_id_safe_stem(MODEL_ID)
ck_lo = min(float(_tnanmin(m)) for m in cka_by_set)
ck_hi = max(float(_tnanmax(m)) for m in cka_by_set)
_assert_finite_float(ck_lo, what="ck_lo")
_assert_finite_float(ck_hi, what="ck_hi")
ck_pad = 0.05 * (ck_hi - ck_lo + 1e-9)
vmin, vmax = ck_lo - ck_pad, ck_hi + ck_pad
_assert_finite_float(ck_pad, what="ck_pad")
_assert_finite_float(vmin, what="vmin")
_assert_finite_float(vmax, what="vmax")

_n_hm = len(cka_by_set)
_ncols = max(1, int(math.ceil(math.sqrt(_n_hm))))
_nrows = int(math.ceil(_n_hm / _ncols))
fig_w = min(24, 4.2 * _ncols)
fig_h = min(3.6 * _nrows + 1.5, 3.6 * _nrows + 2.0)
fig, axes = plt.subplots(_nrows, _ncols, figsize=(fig_w, fig_h), constrained_layout=True)
if _n_hm == 1:
    ax_list = [axes]
elif _nrows == 1 or _ncols == 1:
    ax_list = list(axes)
else:
    ax_list = [axes[i, j] for i in range(_nrows) for j in range(_ncols)]

for ax in ax_list[_n_hm:]:
    ax.set_visible(False)

for ax, cka_m, si, nt in zip(ax_list, cka_by_set, range(_n_hm), n_rows_by_set):
    name = PILE_SET_NAMES[si]
    title = name if len(name) <= 48 else name[:45] + "…"
    im = ax.imshow(cka_m.detach().cpu().numpy(), vmin=vmin, vmax=vmax, cmap="Blues", aspect="equal")
    ax.set_title(f"{title}\n{nt} last-token rows")
    ax.set_xlabel("layer j")
    ax.set_ylabel("layer i")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

fig.suptitle(
    f"{MODEL_ID}\n"
    r"Layer–layer linear CKA (ckatorch, unbiased; $K=XX^\top$). "
    f"NeelNanda/pile-10k train[:{PILE_N_ROWS}], meta.pile_set_name, "
    f"categories with >{PILE_MIN_CHUNKS_PER_SET} chunks only; "
    r"each matrix: last-token activations ($n$ rows).",
    fontsize=10,
)
stonesoup.show(
    fig,
    basename=f"{safe}_layer_cka_ckatorch_lasttok_pile_by_meta_{_n_hm}sets",
    dpi=120,
)
