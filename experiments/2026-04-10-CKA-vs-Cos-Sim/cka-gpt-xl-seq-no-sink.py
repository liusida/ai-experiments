# %% Imports & helpers
from __future__ import annotations

import math
import time
from collections import defaultdict

import stonesoup
import matplotlib.pyplot as plt
import torch
from ckatorch import cka_base
from stonesoup.experiment import (
    capture_embed_and_post_blocks,
    configure_matplotlib_agg,
    ensure_pad_token_via_eos,
    hf_repo_id_safe_stem,
    inner_tokenizer,
)


def _tnanmax(t: torch.Tensor) -> torch.Tensor:
    return torch.max(torch.nan_to_num(t, nan=float("-inf")))


def _tnanmin(t: torch.Tensor) -> torch.Tensor:
    return torch.min(torch.nan_to_num(t, nan=float("inf")))


def _assert_all_finite(t: torch.Tensor, *, what: str) -> None:
    assert bool(torch.isfinite(t).all().item()), f"non-finite values in {what}"


def _assert_finite_float(x: float, *, what: str) -> None:
    assert math.isfinite(x), f"non-finite float in {what}: {x}"


configure_matplotlib_agg()


def layer_cka_matrix(activations: torch.Tensor) -> torch.Tensor:
    """L×L linear CKA; activations (n_rows, n_layers, dim). Diagonal fixed to 1."""
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


def activations_all_layers_skip_sink_prefix(
    model,
    proc,
    device,
    text: str,
    *,
    max_length: int,
    skip_first_tokens: int,
) -> tuple[torch.Tensor, int, int, list[str] | None]:
    """Run one forward; return activations on positions ``[skip_first_tokens : seq_len)``.

    Shapes: ``(n_pos, n_layers, dim)`` with ``n_pos = seq_len - skip_first_tokens``.
    """
    tok = inner_tokenizer(proc)
    enc = tok(
        text,
        return_tensors="pt",
        return_attention_mask=True,
        add_special_tokens=True,
        max_length=max_length,
        truncation=True,
    )
    inputs = {k: v.to(device) for k, v in enc.items()}
    stack, stage_names = capture_embed_and_post_blocks(model, inputs, use_cache=False)
    seq_len = int(inputs["attention_mask"][0].sum().item())
    sl = max(1, seq_len)
    if skip_first_tokens >= sl:
        raise ValueError(
            f"skip_first_tokens={skip_first_tokens} >= seq_len={sl}; "
            "use a longer passage or smaller skip."
        )
    st = stack[:, 0, :sl, :].detach().float()
    tok_layer_dim = st.permute(1, 0, 2).contiguous()
    act = tok_layer_dim[skip_first_tokens:sl, :, :].contiguous()
    _assert_all_finite(act, what="activations_all_layers_skip_sink_prefix")
    return act, sl, int(act.shape[0]), stage_names


def _decorate_cka_heatmap(
    ax: plt.Axes,
    cka_m: torch.Tensor,
    vmin: float,
    vmax: float,
    *,
    title: str,
) -> tuple[plt.Axes, object]:
    """Draw one CKA heatmap with per-layer ticks and thin cell grid; return ``(ax, mappable)``."""
    L = int(cka_m.shape[0])
    # layer_ticks = list(range(L))
    im = ax.imshow(
        cka_m.detach().cpu().numpy(),
        vmin=vmin,
        vmax=vmax,
        cmap="Blues",
        aspect="equal",
        interpolation="nearest",
    )
    # ax.set_xticks(layer_ticks)
    # ax.set_yticks(layer_ticks)
    # if L > 36:
    #     ax.tick_params(axis="both", which="major", labelsize=6)
    #     plt.setp(ax.get_xticklabels(), rotation=90, ha="center", fontsize=6)
    # elif L > 24:
    #     ax.tick_params(axis="both", which="major", labelsize=7)
    #     plt.setp(ax.get_xticklabels(), rotation=90, ha="center", fontsize=7)
    # if L > 1:
    #     b = [i + 0.5 for i in range(L - 1)]
    #     ax.set_xticks(b, minor=True)
    #     ax.set_yticks(b, minor=True)
    #     ax.grid(which="minor", color="0.75", linestyle="-", linewidth=0.35)
    #     ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_title(title)
    ax.set_xlabel("layer j")
    ax.set_ylabel("layer i")
    return ax, im


def _pile_set_name_from_row(row: dict) -> str:
    meta = row.get("meta")
    if isinstance(meta, dict):
        name = meta.get("pile_set_name")
        if name is not None:
            return str(name)
    return "<unknown>"


def _row_title_snippet(
    text: str,
    *,
    max_words: int,
    max_chars: int,
) -> str:
    """First ``max_words`` words, single line; ``…`` if truncated; cap length with ``max_chars``."""
    collapsed = " ".join(text.split())
    if not collapsed:
        return "(empty)"
    parts = collapsed.split(" ")
    body = " ".join(parts[:max_words])
    truncated = len(parts) > max_words
    if truncated:
        body += " …"
    if len(body) > max_chars:
        body = body[: max_chars - 1].rstrip() + "…"
    return body


# %% Config & Pile: scan ``train``, two rows per ``meta.pile_set_name``
# ``datasets``: ``uv pip install datasets``
MODEL_ID = "EleutherAI/pythia-2.8b"
MAX_TOKENS = 512
# Drop the first token(s) (BOS / the usual attention-sink position). Papers often focus on
# position 0 only; increase this if you want a wider prefix dropped for experiments.
SKIP_FIRST_TOKENS = 1
ROWS_PER_CATEGORY = 2
PILE_PREVIEW_CHARS = 200

from datasets import load_dataset

_ds = load_dataset("NeelNanda/pile-10k", split="train")

# Single pass: collect up to ``ROWS_PER_CATEGORY`` indices per category (scan order).
_by_name: dict[str, list[int]] = defaultdict(list)
for idx in range(len(_ds)):
    stonesoup.check_abort()
    row = _ds[idx]
    name = _pile_set_name_from_row(row)
    if len(_by_name[name]) < ROWS_PER_CATEGORY:
        _by_name[name].append(idx)

_all_cats = sorted(_by_name.keys())
_categories_two: list[str] = [c for c in _all_cats if len(_by_name[c]) >= ROWS_PER_CATEGORY]
_categories_short: list[str] = [c for c in _all_cats if len(_by_name[c]) < ROWS_PER_CATEGORY]

pile_samples: list[tuple[int, str, str]] = []
for cat in _categories_two:
    for idx in _by_name[cat]:
        stonesoup.check_abort()
        row = _ds[idx]
        text = (row.get("text") or "").strip().replace("\r\n", "\n")
        pile_samples.append((idx, text, cat))

if not pile_samples:
    raise ValueError(
        "No category has at least ROWS_PER_CATEGORY rows in the scan; "
        "lower ROWS_PER_CATEGORY or check the dataset."
    )

print(
    f"Pile: NeelNanda/pile-10k len={len(_ds)}  scan → {len(_all_cats)} distinct meta.pile_set_name; "
    f"{len(_categories_two)} categories with ≥{ROWS_PER_CATEGORY} examples (plotted); "
    f"max_length={MAX_TOKENS}  skip_first_tokens={SKIP_FIRST_TOKENS}",
    flush=True,
)
if _categories_short:
    print(
        f"Skipped (<{ROWS_PER_CATEGORY} hit in scan): {len(_categories_short)} categories — "
        f"{_categories_short!r}",
        flush=True,
    )
for idx, text, pile_name in pile_samples:
    print(
        f"  pile-10k id:{idx},  pile_set_name:{pile_name!r}  "
        f"preview: {text[:PILE_PREVIEW_CHARS]!r}…",
        flush=True,
    )

# %% Load model
model, proc = stonesoup.load_model(MODEL_ID)
model.eval()
tok = inner_tokenizer(proc)
ensure_pad_token_via_eos(tok)
device = next(model.parameters()).device
print(f"device={device}", flush=True)

# %% Forward + layer CKA (one heatmap per Pile row)
cka_by_row: list[torch.Tensor] = []
seq_len_by_row: list[int] = []
n_pos_by_row: list[int] = []
stage_names: list[str] | None = None

for ri, (ds_idx, text, pile_name) in enumerate(pile_samples):
    stonesoup.check_abort()
    t0 = time.perf_counter()
    print(
        f"[seq no-sink] {ri + 1}/{len(pile_samples)} train[{ds_idx}] {pile_name!r}: forward…",
        flush=True,
    )
    act, seq_len, n_pos, stage_names = activations_all_layers_skip_sink_prefix(
        model,
        proc,
        device,
        text,
        max_length=MAX_TOKENS,
        skip_first_tokens=SKIP_FIRST_TOKENS,
    )
    print(
        f"[seq no-sink] forward {time.perf_counter() - t0:.1f}s  "
        f"seq_len={seq_len}  n_pos={n_pos}  act {tuple(act.shape)}",
        flush=True,
    )
    n2 = n_pos * n_pos
    print(
        f"[seq no-sink] layer CKA (L×L), n={n_pos}, n²={n2:,}…",
        flush=True,
    )
    t_cka = time.perf_counter()
    cka_m = layer_cka_matrix(act)
    print(f"[seq no-sink] CKA {time.perf_counter() - t_cka:.1f}s", flush=True)
    _assert_all_finite(cka_m, what=f"cka matrix row {ds_idx}")
    cka_by_row.append(cka_m)
    seq_len_by_row.append(seq_len)
    n_pos_by_row.append(n_pos)

if stage_names:
    print(
        f"layers: {stage_names[0]} … {stage_names[-1]}  ({len(stage_names)} stages)",
        flush=True,
    )

# %% Plot (grid of heatmaps, shared color scale)
# Subplot title: leading words from the row text (see ``_row_title_snippet``).
PILE_TITLE_MAX_WORDS = 15
PILE_TITLE_MAX_CHARS = 50

safe = hf_repo_id_safe_stem(MODEL_ID)
ck_lo = min(float(_tnanmin(m)) for m in cka_by_row)
ck_hi = max(float(_tnanmax(m)) for m in cka_by_row)
_assert_finite_float(ck_lo, what="ck_lo")
_assert_finite_float(ck_hi, what="ck_hi")
ck_pad = 0.05 * (ck_hi - ck_lo + 1e-9)
vmin, vmax = ck_lo - ck_pad, ck_hi + ck_pad

_sink_note = (
    "after dropping the first token (BOS / sink)."
    if SKIP_FIRST_TOKENS == 1
    else f"after dropping the first {SKIP_FIRST_TOKENS} tokens (sink prefix)."
)
_n_cat = len(_categories_two)
_n_panels = len(pile_samples)
assert _n_panels == _n_cat * ROWS_PER_CATEGORY == len(cka_by_row)
# Four heatmaps per row: two categories × two examples each (``pile_samples`` order is unchanged).
_ncols = 4
_nrows = int(math.ceil(_n_panels / _ncols))
_row_in = 2.55
fig_w = max(28.0, 4.6 * float(_ncols) + 6.0)
fig_h = min(110.0, _row_in * float(_nrows) + 3.5)
fig, axes = plt.subplots(_nrows, _ncols, figsize=(8, 20), constrained_layout=True)
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.12, wspace=0.12, hspace=0.22)
if _nrows == 1:
    ax_list = list(axes)
else:
    ax_list = [axes[i, j] for i in range(_nrows) for j in range(_ncols)]
for _j in range(_n_panels, len(ax_list)):
    ax_list[_j].set_visible(False)

for ax, cka_m, (ds_idx, row_text, pile_name), seq_len, n_pos in zip(
    ax_list[:_n_panels],
    cka_by_row,
    pile_samples,
    seq_len_by_row,
    n_pos_by_row,
):
    pname = pile_name if len(pile_name) <= 32 else pile_name[:29] + "…"
    text_snip = _row_title_snippet(
        row_text,
        max_words=PILE_TITLE_MAX_WORDS,
        max_chars=PILE_TITLE_MAX_CHARS,
    )
    title = (
        f"pile-10k id:{ds_idx}, pile_set_name:{pname!r}\n"
        f"{text_snip}"
        # f"n={n_pos} tok (pos {SKIP_FIRST_TOKENS}…{seq_len - 1}, len≤{MAX_TOKENS})"
    )
    title = pname
    _ax, im = _decorate_cka_heatmap(ax, cka_m, vmin, vmax, title=title)
    # fig.colorbar(im, ax=_ax, fraction=0.046, pad=0.04)

# fig.suptitle(
#     f"{MODEL_ID}\n"
#     r"Layer–layer linear CKA (ckatorch, unbiased). "
#     f"NeelNanda/pile-10k: {ROWS_PER_CATEGORY} examples per meta.pile_set_name; "
#     f"{_n_cat} categories; 4 panels per row (two categories); max {MAX_TOKENS} tok; {_sink_note}",
#     fontsize=10,
# )
fig.tight_layout()
stonesoup.show(
    fig,
    basename=f"{safe}_layer_cka_ckatorch_seq_nosink_2percat_{_n_cat}cat_4col_{MAX_TOKENS}tok",
    dpi=120,
)
