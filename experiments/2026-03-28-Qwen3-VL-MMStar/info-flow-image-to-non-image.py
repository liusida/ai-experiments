# %% Imports & paths

"""Qwen3-VL prefill: print attention from **image keys** to **text queries** (mean over heads).

**Meaning of the printout:** For each layer, ``W`` is attention averaged over heads, with
``W[q, k]`` = how much **query** position *q* (a non-image token) attends to **key** *k* (an image token).
Only **causal** pairs ``k ≤ q`` are shown. Large mass means that patch (see cell 5 HTML) is important for
building the representation at token *q* at that layer.

**Run:** ``uv run python …/info-flow-image-to-non-image.py`` or Stonesoup **Watch** — run **1 → 2 → 3** (optional **4** = post-layer image activation mask + ``generate``, **5** = HTML token map).
Edit **image path, prompt, and print thresholds** in cell 3 before running it.
"""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

# %% Load model & processor

model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    dtype="auto" if DEVICE.type == "cuda" else torch.float32,
    device_map="auto" if DEVICE.type == "cuda" else None,
    attn_implementation="eager",
)
if DEVICE.type != "cuda":
    model = model.to(DEVICE)
processor = AutoProcessor.from_pretrained(MODEL_ID)
print("Loaded:", MODEL_ID)

# %% Encode, forward, print image→text flows

IMAGE_PATH = REPO_ROOT / "data/images/kangaroo-small.png"
USER_TEXT = "Describe the image using only one word: dog, cat, turtle, chicken, kangaroo,or other animal?"
MIN_MASS = 1e-2  # ignore smaller A[query,key]
MAX_PRINT_PER_LAYER = 200
# Only image **key** seq positions listed here contribute to printed edges and ``_keys_above_threshold``.
# ``None`` = all image keys. Must be non-empty intersection with this prompt’s image tokens.
FLOW_ONLY_IMAGE_KEY_POS: frozenset[int] | None = frozenset({8})

if not IMAGE_PATH.is_file():
    raise FileNotFoundError(IMAGE_PATH)

image = Image.open(IMAGE_PATH).convert("RGB")
inputs = processor.apply_chat_template(
    [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": USER_TEXT}]}],
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
).to(DEVICE)

ids = inputs.input_ids[0]
img_tok = int(model.config.image_token_id)
img_cols = {int(i) for i in (ids == img_tok).nonzero(as_tuple=True)[0].tolist()}
if FLOW_ONLY_IMAGE_KEY_POS is not None:
    _flow_key_subset = img_cols.intersection(FLOW_ONLY_IMAGE_KEY_POS)
    if not _flow_key_subset:
        raise ValueError(
            f"FLOW_ONLY_IMAGE_KEY_POS {sorted(FLOW_ONLY_IMAGE_KEY_POS)} has no overlap with image token positions"
        )
else:
    _flow_key_subset = img_cols
T = int(ids.shape[0])
_IMG_PAD = "<.>"
labels = []
_seq_full: list[str] = []
for _si, tid in enumerate(ids.tolist()):
    if int(tid) == img_tok:
        labels.append(_IMG_PAD)
        _seq_full.append(_IMG_PAD)
    else:
        s = processor.tokenizer.decode([tid], skip_special_tokens=False).replace("\n", "↵")
        labels.append((s[:18] + "…") if len(s) > 18 else s)
        _seq_full.append(s)

print("sequence (pos → token; image pads as <.>):")
for _si in range(T):
    print(f"  {_si:>4}  {_seq_full[_si]!r}")

# Hook every layer's self_attn; each forward appends {"layer": i, "attn": [1,H,T,T]}
lm = model.model.language_model
captured: list[dict] = []
handles = []

def capture(layer_i: int):
    def _hook(_m, _args, out):
        captured.append({"layer": layer_i, "attn": out[1].detach()})

    return _hook

for i, layer in enumerate(lm.layers):
    handles.append(layer.self_attn.register_forward_hook(capture(i)))

try:
    with torch.inference_mode():
        model(**inputs)
finally:
    for h in handles:
        h.remove()

captured.sort(key=lambda d: d["layer"])
print(
    f"seq_len={T} | image_slots={len(img_cols)} | layers={len(lm.layers)}"
    + (
        f" | flow keys only: {sorted(_flow_key_subset)}"
        if FLOW_ONLY_IMAGE_KEY_POS is not None
        else ""
    )
)

_keys_above_threshold: set[int] = set()
for row in captured:
    W_pre = row["attn"][0].float().mean(0).cpu()
    for _q in range(T):
        if _q in img_cols:
            continue
        for _k in _flow_key_subset:
            if _k > _q:
                continue
            if float(W_pre[_q, _k].item()) >= MIN_MASS:
                _keys_above_threshold.add(_k)

_img_key_indices = sorted(_keys_above_threshold)
print(
    "image key seq indices (≥ MIN_MASS in some layer):",
    ", ".join(str(k) for k in _img_key_indices) if _img_key_indices else "—",
    f"(n={len(_img_key_indices)})",
)
_fk_note = f" (image keys restricted to {_flow_key_subset})" if FLOW_ONLY_IMAGE_KEY_POS is not None else ""
print("edges: causal only (key_idx ≤ query_idx), image key → non-image query, mass ≥", MIN_MASS, _fk_note)
print()

for row in captured:
    ell = row["layer"]
    # [H,T,T] → mean → [T,T] ; W[q,k] = query q looks at key k
    W = row["attn"][0].float().mean(0).cpu()
    edges: list[tuple[int, int, float]] = []
    for q in range(T):
        if q in img_cols:
            continue
        for k in _flow_key_subset:
            if k > q:
                continue
            mass = float(W[q, k].item())
            if mass >= MIN_MASS:
                edges.append((q, k, mass))
    edges.sort(key=lambda t: -t[2])
    if not edges:
        print(f"layer {ell:>2}: —")
        continue
    print(f"layer {ell:>2}: {len(edges)} edges (showing min({MAX_PRINT_PER_LAYER}, all))")
    for q, k, mass in edges[:MAX_PRINT_PER_LAYER]:
        print(f"   A[{q:>3},{k:>3}]={mass:.5f}   key{k}({labels[k]!s}) → q{q}({labels[q]!s})")
    if len(edges) > MAX_PRINT_PER_LAYER:
        print(f"   … +{len(edges) - MAX_PRINT_PER_LAYER} more")

# %% Hook sweep: keep one image pos × layer cutoff → generate (records matrix for next cell)
#
# Cells 2–3. For each **kept** image seq position (column) and ``ZERO_ALL_AFTER_LAYER`` (row): same pre-hook
# policy as before — partial zero through that layer index, then zero all image rows. Fills ``ABLATION_*``
# globals; scores use ``ABLATION_MATCH_NEGATIVE`` (−1) vs ``ABLATION_MATCH_POSITIVE`` (+1) vs other (0).
# Speed: ``SWEEP_POS_STRIDE`` / ``SWEEP_LAYER_STRIDE`` subsample (e.g. 5 ⇒ every 5th image index & layer).
# Progress: ``tqdm`` (default ``sys.stderr``). Stonesoup folds ``\\r`` in the output UI so the bar stays readable.
# ``use_cache=False``.

import re
from itertools import product

import numpy as np
from tqdm import tqdm

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Substrings in decoded text (lowercased). +1 if positive matches; −1 if negative matches (word + optional ``s``).
ABLATION_MATCH_POSITIVE = "k"
ABLATION_MATCH_NEGATIVE = "cat"


def _score_sweep_decoded(decoded: str) -> float:
    t = decoded.lower()
    if ABLATION_MATCH_POSITIVE and re.search(re.escape(ABLATION_MATCH_POSITIVE.lower()), t):
        return 1.0
    if ABLATION_MATCH_NEGATIVE and re.search(
        rf"\b{re.escape(ABLATION_MATCH_NEGATIVE.lower())}s?\b",
        t,
    ):
        return -1.0
    return 0.0


_layers = model.model.language_model.layers
_n = len(_layers)
_row = inputs.input_ids[0]
_img_set = {int(t) for t in (_row == img_tok).nonzero(as_tuple=True)[0]}
if not _img_set:
    raise ValueError("sweep needs at least one image token in the prompt")

# 1 = full grid; e.g. 5 keeps every 5th sorted image seq index and every 5th decoder layer.
SWEEP_POS_STRIDE = 5
SWEEP_LAYER_STRIDE = 5
if SWEEP_POS_STRIDE < 1 or SWEEP_LAYER_STRIDE < 1:
    raise ValueError("SWEEP_*_STRIDE must be >= 1")

# Horizontal axis: image seq indices (subsampled) we keep in the partial phase.
POS_LIST = sorted(_img_set)[::SWEEP_POS_STRIDE]
LAYER_LIST = list(range(0, _n, SWEEP_LAYER_STRIDE))
if not POS_LIST or not LAYER_LIST:
    raise ValueError("subsampled POS_LIST or LAYER_LIST is empty; lower strides")
SWEEP_VERBOSE = False
# Cap length so the tqdm line stays usable in narrow / streamed UIs (Stonesoup).
SWEEP_TQDM_POSTFIX_MAX = 72


def _sweep_decode_for_postfix(s: str, max_len: int = SWEEP_TQDM_POSTFIX_MAX) -> str:
    t = " ".join(s.replace("\r", " ").replace("\n", " ").split())
    if len(t) > max_len:
        return t[: max_len - 1] + "…"
    return t if t else "—"


def _sweep_max_new_tokens() -> int:
    """Enough headroom for the longer of the two label words under this tokenizer (e.g. kangaroo → 3, cat → 1)."""
    def _enc_len(s: str) -> int:
        return len(processor.tokenizer.encode(s, add_special_tokens=False)) if s else 0

    longest = max(_enc_len(ABLATION_MATCH_POSITIVE), _enc_len(ABLATION_MATCH_NEGATIVE), 1)
    return longest


MAX_NEW_TOKENS_SWEEP = _sweep_max_new_tokens()

n_p, n_z = len(POS_LIST), len(LAYER_LIST)
ABLATION_SCORE = np.zeros((n_z, n_p), dtype=np.float64)
ABLATION_TEXT = np.empty((n_z, n_p), dtype=object)

print(
    f"sweep: {n_z}×{n_p}={n_z * n_p} runs | pos stride {SWEEP_POS_STRIDE} layer stride {SWEEP_LAYER_STRIDE} | "
    f"positions {POS_LIST[0]}…{POS_LIST[-1]} ({n_p} cols) | layers {LAYER_LIST[0]}…{LAYER_LIST[-1]} | "
    f"max_new_tokens={MAX_NEW_TOKENS_SWEEP}"
)

_total_runs = n_z * n_p
_sweep_pbar = tqdm(
    product(range(n_z), range(n_p)),
    total=_total_runs,
    desc="ablation sweep",
    unit="gen",
    dynamic_ncols=True,
)

for zi, pi in _sweep_pbar:
    ZERO_ALL_AFTER_LAYER = LAYER_LIST[zi]
    tok_pos = POS_LIST[pi]
    KEEP_IMAGE_SEQ_POS: set[int] = {tok_pos}
    _keep = KEEP_IMAGE_SEQ_POS
    _zero_idx_partial = sorted(_img_set - _keep)
    _zero_idx_all_img = sorted(_img_set)

    def _zero_vis_cols(hidden: torch.Tensor, layer_i: int) -> None:
        drop = _zero_idx_all_img if layer_i > ZERO_ALL_AFTER_LAYER else _zero_idx_partial
        for p in drop:
            if p < hidden.shape[1]:
                hidden[:, p, :] = 0

    _handles: list = []
    for li in range(_n):

        def _pre(_m, args, _li=li):  # noqa: ANN001
            if args and isinstance(args[0], torch.Tensor):
                _zero_vis_cols(args[0], _li)

        _handles.append(_layers[li].register_forward_pre_hook(_pre))
    try:
        with torch.inference_mode():
            model.model.rope_deltas = None
            _go = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS_SWEEP,
                do_sample=False,
                use_cache=False,
            )
    finally:
        for _h in _handles:
            _h.remove()
    _gen_ids = _go[0, inputs.input_ids.shape[1] :].tolist()
    _txt = processor.tokenizer.decode(_gen_ids, skip_special_tokens=True)
    ABLATION_SCORE[zi, pi] = _score_sweep_decoded(_txt)
    ABLATION_TEXT[zi, pi] = _txt
    _sweep_pbar.set_postfix_str(_sweep_decode_for_postfix(_txt), refresh=True)
    if SWEEP_VERBOSE:
        print(f"  z={ZERO_ALL_AFTER_LAYER} pos={tok_pos} → {ABLATION_SCORE[zi, pi]!r} {_txt!r}")

print("sweep done → use next cell for heatmap; ABLATION_SCORE shape", ABLATION_SCORE.shape)

# %% Plot: ablation sweep heatmap (negative vs positive label)

# Requires the sweep cell. x = kept image seq position, y = ``ZERO_ALL_AFTER_LAYER``. Diverging cmap (see
# ``HEATMAP_CMAP``): cool ≈ negative label, warm ≈ positive, mid ≈ other — **no per-cell text**; light grid;
# **equal aspect** (square cells);
# title on the axes (tight top). ``origin='upper'`` = layer 0 at top.

import matplotlib.pyplot as plt
import numpy as np

if (
    globals().get("ABLATION_SCORE") is None
    or globals().get("ABLATION_TEXT") is None
    or globals().get("POS_LIST") is None
):
    raise RuntimeError("Run the sweep cell first.")

_neg_l = str(globals().get("ABLATION_MATCH_NEGATIVE", "neg"))
_pos_l = str(globals().get("ABLATION_MATCH_POSITIVE", "pos"))

# Diverging map centered at 0 (``vmin``/``vmax``): e.g. ``coolwarm``, ``RdYlBu_r``, ``Spectral_r``.
HEATMAP_CMAP = "RdYlBu_r"

_h_nz, _h_np = ABLATION_SCORE.shape
# Figure size from data shape: **equal aspect** (square cells), room for colorbar + axis labels.
_max_dim = max(_h_nz, _h_np, 1)
_cell_in = min(0.36, 12.0 / _max_dim)
_core_w, _core_h = _cell_in * _h_np, _cell_in * _h_nz
_fig_w = _core_w + 2.15
_fig_h = _core_h + 1.55
_fig, _ax = plt.subplots(figsize=(_fig_w, _fig_h), layout="constrained")
_im = _ax.imshow(
    ABLATION_SCORE,
    aspect="equal",
    origin="upper",
    cmap=HEATMAP_CMAP,
    vmin=-1,
    vmax=1,
    interpolation="nearest",
)
_ax.set_title(
    f"Decoded match: {_neg_l} (−1) vs {_pos_l} (+1) vs kept patch & layer cutoff",
    fontsize=15,
    fontweight="semibold",
    pad=12,
)
_ax.set_xlabel("Image Token Seq Position", fontsize=12, labelpad=12)
_ax.set_ylabel("Layer", fontsize=12, labelpad=12)
_x_step = max(1, _h_np // 30)
_ax.set_xticks(np.arange(0, _h_np, _x_step))
_ax.set_xticklabels([str(POS_LIST[i]) for i in range(0, _h_np, _x_step)], rotation=45, ha="right")
_ys = max(1, _h_nz // 20)
_ax.set_yticks(np.arange(0, _h_nz, _ys))
_ax.set_yticklabels([str(LAYER_LIST[i]) for i in range(0, _h_nz, _ys)])
_ax.tick_params(axis="both", which="major", labelsize=11, length=5, width=0.9, pad=9)

# Subtle cell boundary grid (half-integer lines; drawn above the image).
_ax.set_xticks(np.arange(-0.5, _h_np, 1.0), minor=True)
_ax.set_yticks(np.arange(-0.5, _h_nz, 1.0), minor=True)
_ax.grid(
    which="minor",
    color="0.4",
    linestyle="-",
    linewidth=0.35,
    alpha=0.5,
    zorder=0,
)
_ax.tick_params(which="minor", bottom=False, left=False, length=0)

_cbar = _fig.colorbar(
    _im,
    ax=_ax,
    shrink=0.88,
    fraction=0.045,
    pad=0.03,
    ticks=[-1.0, 0.0, 1.0],
)
_cbar.ax.set_yticklabels([_neg_l, "other", _pos_l])
_cbar.ax.tick_params(labelsize=11, pad=6)

_tight = OUTPUT_DIR / f"ablation_sweep_{_neg_l}_vs_{_pos_l}.png"
_fig.savefig(_tight, dpi=200, bbox_inches="tight", pad_inches=0.12)
plt.close(_fig)
print("saved", _tight)

# %% HTML: image token index → patch on the image

# Requires cells 2–3. Prints HTML (Stonesoup: toggle rendered view): resized canvas + table.
# Rows are **only** seq indices in ``_keys_above_threshold`` from cell 3 (mass ≥ ``MIN_MASS`` in some layer).
# If cell 3 was not run, falls back to all image-token positions.
# ViT uses non-overlapping patch_size² windows; merge_size² of those are pooled → one LM image token.
# Columns: merged region on canvas, and a mosaic of the **base ViT patch** windows that feed that token.

import base64
import io

import numpy as np
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

try:
    from stonesoup import STONESOUP_RENDER_HTML
except ImportError:
    STONESOUP_RENDER_HTML = "# stonesoup:render=html\n"


def _contiguous_runs_sorted(pn: np.ndarray) -> list[tuple[int, int]]:
    if pn.size == 0:
        return []
    br = np.where(np.diff(pn) != 1)[0] + 1
    st = np.concatenate(([0], br))
    en = np.concatenate((br, [pn.size]))
    return list(zip(st.tolist(), en.tolist(), strict=True))


def _merge_hw(k: int, llm_h: int, llm_w: int) -> tuple[int, int]:
    spatial = llm_h * llm_w
    r = int(k) % spatial
    return r // llm_w, r % llm_w


def _processor_min_max_px(image_processor) -> tuple[int, int]:
    default_min, default_max = 56 * 56, 28 * 28 * 1280
    mn = getattr(image_processor, "min_pixels", None)
    mx = getattr(image_processor, "max_pixels", None)
    if mn is None or mx is None:
        sz = getattr(image_processor, "size", None)
        if isinstance(sz, dict):
            mn = mn if mn is not None else int(sz.get("shortest_edge", default_min))
            mx = mx if mx is not None else int(sz.get("longest_edge", default_max))
        else:
            mn = mn if mn is not None else default_min
            mx = mx if mx is not None else default_max
    return int(mn), int(mx)


def _data_url_png(im: Image.Image) -> str:
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.standard_b64encode(buf.getvalue()).decode("ascii")


def _vit_base_patch_mosaic(
    canvas: Image.Image,
    h_m: int,
    w_m: int,
    *,
    merge: int,
    patch_size: int,
    thumb_px: int = 28,
    gap: int = 1,
) -> Image.Image:
    """One image: ``merge×merge`` grid of **ViT** patch crops (each ``patch_size`` square on ``canvas``)."""
    g = gap
    side = merge * thumb_px + (merge - 1) * g
    out = Image.new("RGB", (side, side), (218, 220, 224))
    for br in range(merge):
        for bc in range(merge):
            y0 = (h_m * merge + br) * patch_size
            x0 = (w_m * merge + bc) * patch_size
            tile = canvas.crop((x0, y0, x0 + patch_size, y0 + patch_size)).resize(
                (thumb_px, thumb_px), Image.Resampling.BILINEAR
            )
            out.paste(tile, (bc * (thumb_px + g), br * (thumb_px + g)))
    return out


_ip = processor.image_processor
_merge = int(_ip.merge_size)
_ps = int(_ip.patch_size)
_mn, _mx = _processor_min_max_px(_ip)
_w0, _h0 = image.size
_rh, _rw = smart_resize(_h0, _w0, factor=_ps * _merge, min_pixels=_mn, max_pixels=_mx)
_resized = image.convert("RGB").resize((_rw, _rh), Image.Resampling.BICUBIC)

_ids_cpu = ids.detach().cpu()
_img_id = int(model.config.image_token_id)
_pos = torch.nonzero(_ids_cpu == _img_id, as_tuple=False).squeeze(-1)
_pn = _pos.numpy().astype(np.int64)
_runs = _contiguous_runs_sorted(_pn)
_grid = inputs["image_grid_thw"].detach().cpu().long()
if _grid.dim() == 1:
    _grid = _grid.unsqueeze(0)
if len(_runs) != int(_grid.shape[0]):
    raise ValueError(f"image token runs {len(_runs)} vs image_grid_thw rows {_grid.shape[0]}")

_cell = _merge * _ps
_thumb_merge_max = 72
_mosaic_thumb = 28
try:
    _html_only_keys: frozenset[int] | None = frozenset(_keys_above_threshold)
except NameError:
    _html_only_keys = None

_table_rows: list[tuple[int, str]] = []
for _ri, (_s, _e) in enumerate(_runs):
    t_hw = _grid[_ri].tolist()
    _tp, _hp, _wp = int(t_hw[0]), int(t_hw[1]), int(t_hw[2])
    _n_tok = (_tp * _hp * _wp) // (_merge * _merge)
    _seq_blk = _pn[_s:_e]
    if _seq_blk.size != _n_tok:
        raise ValueError("token count mismatch for image run")
    _lh, _lw_m = _hp // _merge, _wp // _merge
    for _k in range(_n_tok):
        _hm, _wm = _merge_hw(_k, _lh, _lw_m)
        _x0, _y0 = _wm * _cell, _hm * _cell
        _crop = _resized.crop((_x0, _y0, _x0 + _cell, _y0 + _cell))
        _side_m = min(_crop.size[0], _crop.size[1], _thumb_merge_max)
        _thumb_merged = _crop.resize((_side_m, _side_m), Image.Resampling.BILINEAR)
        _mos = _vit_base_patch_mosaic(
            _resized, _hm, _wm, merge=_merge, patch_size=_ps, thumb_px=_mosaic_thumb, gap=1
        )
        _mos_side = _mos.size[0]
        _ix = int(_seq_blk[_k])
        if _html_only_keys is not None and _ix not in _html_only_keys:
            continue
        _row = (
            f"<tr><td>{_ix}</td><td>({_hm},{_wm})</td>"
            f"<td class=\"muted\">{_merge}×{_merge}×{_ps}px<br/>(merged)</td>"
            f"<td><img src=\"{_data_url_png(_thumb_merged)}\" width=\"{_side_m}\" height=\"{_side_m}\" alt=\"merged\"/></td>"
            f"<td><img class=\"mosaic\" src=\"{_data_url_png(_mos)}\" width=\"{_mos_side}\" height=\"{_mos_side}\" "
            f'alt="ViT patches"/></td></tr>'
        )
        _table_rows.append((_ix, _row))

_table_rows.sort(key=lambda t: t[0])
_rows_html = [r for _, r in _table_rows]
_full_src = _data_url_png(_resized)
_filter_note = (
    f" Table lists only <strong>{len(_table_rows)}</strong> seq idx with image→text attention ≥ <strong>{MIN_MASS}</strong> "
    "in some layer (same set as the printed key list)."
    if _html_only_keys is not None
    else " Table lists <strong>all</strong> image-token seq indices (run cell 3 first to restrict to the ≥ MIN_MASS key list)."
)
_html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"/>
<style>
body {{ font-family: system-ui, sans-serif; margin: 1rem; max-width: 1200px; }}
p.note {{ color: #444; font-size: 0.95rem; }}
td.muted {{ font-size: 0.8rem; color: #495057; white-space: nowrap; }}
img.ref {{ display: block; max-width: 100%; height: auto; border: 1px solid #ccc; }}
img.mosaic {{ border: 1px solid #adb5bd; vertical-align: middle; }}
table {{ border-collapse: collapse; margin-top: 1rem; }}
th, td {{ border: 1px solid #ccc; padding: 6px; text-align: left; vertical-align: middle; }}
th {{ background: #f1f3f5; }}
</style></head><body>
<p class="note">Resized vision canvas: <strong>{_rw}×{_rh}</strong> px. The ViT embeds <strong>non-overlapping</strong>
<strong>{_ps}×{_ps}</strong> px patches. <strong>{_merge}×{_merge}</strong> such patch embeddings are spatially merged into
<strong>one</strong> language-model image token (covering <strong>{_cell}×{_cell}</strong> px). <em>seq idx</em> matches
<code>key…</code> in the attention printout. Deeper ViT blocks mix information **within** this merge tile via attention; the
mosaic shows the **pixel windows** that feed the patch embeddings before merge.{_filter_note}</p>
<img class="ref" src="{_full_src}" alt="resized input"/>
<table>
<thead><tr><th>seq idx</th><th>merge cell (h,w)</th><th>layout</th><th>merged window</th><th>ViT {_merge}×{_merge} patches</th></tr></thead>
<tbody>
{chr(10).join(_rows_html)}
</tbody>
</table>
</body></html>"""

print(STONESOUP_RENDER_HTML, _html, sep="")
