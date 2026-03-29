# %% Imports & paths

"""**Qwen3-VL** — capture **language-model self-attention** on a chat prompt (optional image in templates) and plot with **matplotlib**.

Uses **forward hooks** on ``language_model.layers[*].self_attn`` because ``Qwen3VLTextDecoderLayer``
drops ``attn_weights`` from the public forward (see Hugging Face ``modeling_qwen3_vl.py``).
Loads with ``attn_implementation="eager"`` so softmax weights exist.

**Stonesoup:** Watch this file; run cells in order. The diagram reflects **prefill** only; **Short generation**
runs separately. Figures: ``output/attention_vis.png`` (flow) and ``output/attention_matraices/layer_XX.png``
(one heatmap per layer; folder name kept as requested).

**Reading the plot:** Optional ``HIDE_TOKEN_INDICES`` drops those positions from matrix crops and removes any flow
edge touching them (e.g. ``{0}`` if you want to hide BOS / sink). ``OMIT_KEY_INDICES`` only removes flows **from**
listed source keys. Per-layer matrix PNGs can show **patch thumbnails** under image-key columns (``MATRIX_DRAW_IMAGE_KEY_PATCHES``):
each column is one **merged** vision cell (``merge_size``² base patches → one ``image_token_id`` slot).

**Ablations:** ``ZERO_IMAGE_OUTFLOW_AFTER_LAYER`` (0-based LM layer index; ``None`` disables): for layers with index
*strictly greater*, post-softmax attention from **image placeholder keys** to **non-image queries** is zeroed and
rows renormalized (see ``qwen3vl_image_outflow_ablation``). Implemented by patching HF ``eager_attention_forward``
(prefill + generation; needs ``attn_implementation="eager"``).

**Terminal:** run the ``# %%`` cells in order (or **Watch** in Stonesoup).

Requires: recent ``transformers`` with Qwen3-VL; **matplotlib** (see repo ``pyproject.toml``).
"""

from __future__ import annotations

import textwrap
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path

import matplotlib as mpl
import matplotlib.patheffects as mpath_effects
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import numpy as np
from matplotlib.patches import FancyArrowPatch
from PIL import Image

import torch
import torch.nn.functional as F
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
import transformers.models.qwen3_vl.modeling_qwen3_vl as qwen3vl_modeling
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EXPERIMENT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = EXPERIMENT_DIR / "output"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("REPO_ROOT:", REPO_ROOT)
print("EXPERIMENT_DIR:", EXPERIMENT_DIR)
print("DEVICE:", DEVICE)


def build_token_axis_labels(
    input_ids_1d: torch.Tensor,
    tokenizer,
    image_token_id: int,
    *,
    max_chars: int = 18,
) -> list[str]:
    """One string per token position (flow + matrix ticks): readable decode for text, ``'.'`` for image placeholders."""
    out: list[str] = []
    for tid in input_ids_1d.tolist():
        if tid == image_token_id:
            out.append(".")
            continue
        piece = tokenizer.decode([tid], skip_special_tokens=False)
        piece = piece.replace("\n", "↵").replace("\r", "")
        if not piece:
            sub = tokenizer.convert_ids_to_tokens([tid])
            piece = sub[0] if sub else str(tid)
        if len(piece) > max_chars:
            piece = piece[: max_chars - 1] + "…"
        out.append(piece)
    return out


def register_self_attn_hooks(
    lm: torch.nn.Module,
) -> tuple[list[dict], list[Callable[[], None]]]:
    """Return (storage list, [remove_all]). Each forward appends attn_weights [B,H,T,T] or fails."""
    captured: list[dict] = []
    hook_handles: list[torch.utils.hooks.RemovableHandle] = []

    def make_hook(layer_idx: int):
        def hook(
            _module: torch.nn.Module,
            _args: tuple[object, ...],
            output: tuple[torch.Tensor, torch.Tensor | None],
        ) -> None:
            _out, attn_w = output
            if attn_w is None:
                raise RuntimeError(
                    f"layer {layer_idx}: attn_weights is None — use attn_implementation='eager' when loading."
                )
            captured.append({"layer": layer_idx, "attn": attn_w.detach()})

        return hook

    for i, layer in enumerate(lm.layers):
        hook_handles.append(layer.self_attn.register_forward_hook(make_hook(i)))

    def remove_all() -> None:
        for h in hook_handles:
            h.remove()

    return captured, [remove_all]


def register_decoder_output_norm_hooks(
    lm: torch.nn.Module,
) -> tuple[list[dict], list[Callable[[], None]]]:
    """Capture mean L2 norm of hidden vectors after each decoder block (B,S,D → mean ||h||_2 over B,S).

    Runs **before** DeepStack visual residuals in multimodal forwards (those are applied in ``TextModel`` after
    the layer returns).
    """
    captured: list[dict] = []
    hook_handles: list[torch.utils.hooks.RemovableHandle] = []

    def make_hook(layer_idx: int):
        def hook(
            _module: torch.nn.Module,
            _args: tuple[object, ...],
            output: torch.Tensor,
        ) -> None:
            mean_norm = output.detach().float().norm(dim=-1).mean().item()
            captured.append({"layer": layer_idx, "mean_l2_norm": float(mean_norm)})

        return hook

    for i, layer in enumerate(lm.layers):
        hook_handles.append(layer.register_forward_hook(make_hook(i)))

    def remove_all() -> None:
        for h in hook_handles:
            h.remove()

    return captured, [remove_all]


def image_token_positions(input_ids_1d: torch.Tensor, image_token_id: int) -> torch.LongTensor:
    """Indices where ``input_ids_1d == image_token_id`` (vision placeholder tokens in the LM sequence)."""
    return torch.nonzero(input_ids_1d == image_token_id, as_tuple=False).squeeze(-1).detach().cpu()


def _renormalize_attn_block_image_outflow(
    attn_weights: torch.Tensor,
    image_key_indices: torch.Tensor,
) -> torch.Tensor:
    """Zero mass on entries (query ∉ image, key ∈ image), then renormalize each query row over keys.

    ``attn_weights``: [B, H, Tq, Tk]. Rows that end up all-zero (no remaining keys) fall back to uniform over
    non-image keys, or uniform over all keys if every key is an image position.
    """
    B, H, Tq, Tk = attn_weights.shape
    device = attn_weights.device
    dtype = attn_weights.dtype
    idx = image_key_indices.to(device=device, dtype=torch.long)
    idx = idx[(idx >= 0) & (idx < Tk)]
    if idx.numel() == 0:
        return attn_weights

    image_mask_k = torch.zeros(Tk, dtype=torch.bool, device=device)
    image_mask_k[idx] = True

    image_mask_q = torch.zeros(Tq, dtype=torch.bool, device=device)
    q_img = idx[idx < Tq]
    if q_img.numel() > 0:
        image_mask_q[q_img.unique()] = True

    forbidden = (~image_mask_q.unsqueeze(1)) & (image_mask_k.unsqueeze(0))
    w = attn_weights.masked_fill(forbidden.view(1, 1, Tq, Tk), 0.0)

    denom = w.sum(dim=-1, keepdim=True)
    n_non_img = int((~image_mask_k).sum().item())
    if n_non_img == 0:
        fallback_row = torch.full((Tk,), 1.0 / max(Tk, 1), device=device, dtype=dtype)
    else:
        fallback_row = (~image_mask_k).to(dtype) / float(n_non_img)
    fallback = fallback_row.view(1, 1, 1, Tk).expand(B, H, Tq, Tk)

    eps = 1e-10
    safe = denom > eps
    return torch.where(safe, w / denom.clamp_min(eps), fallback)


_ORIG_QWEN3VL_EAGER_ATTN = qwen3vl_modeling.eager_attention_forward


def _eager_attention_forward_with_image_outflow_ablation(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs: object,
):
    """Like HF ``eager_attention_forward``; optionally strips image-key → non-image-query mass after softmax."""
    st = _IMAGE_OUTFLOW_ABLATION
    layer_idx = getattr(module, "layer_idx", None)
    if (
        not st["enabled"]
        or layer_idx is None
        or st["after_layer"] is None
        or layer_idx <= st["after_layer"]
        or st["key_indices"] is None
    ):
        return _ORIG_QWEN3VL_EAGER_ATTN(
            module, query, key, value, attention_mask, scaling, dropout, **kwargs  # type: ignore[arg-type]
        )

    key_indices = st["key_indices"]
    if not isinstance(key_indices, torch.Tensor) or key_indices.numel() == 0:
        return _ORIG_QWEN3VL_EAGER_ATTN(
            module, query, key, value, attention_mask, scaling, dropout, **kwargs  # type: ignore[arg-type]
        )

    key_states = qwen3vl_modeling.repeat_kv(key, module.num_key_value_groups)
    value_states = qwen3vl_modeling.repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = _renormalize_attn_block_image_outflow(attn_weights, key_indices)
    attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


_IMAGE_OUTFLOW_ABLATION: dict[str, object] = {
    "enabled": False,
    "after_layer": None,  # int | None — 0-based; ablate layers with index **strictly greater** than this
    "key_indices": None,  # torch.LongTensor 1D — LM positions of image placeholder tokens (prefill indices)
}


@contextmanager
def qwen3vl_image_outflow_ablation(
    *,
    after_layer: int | None,
    image_key_indices: torch.LongTensor | None,
):
    """Context: patch Qwen3-VL eager attention so layers **after** ``after_layer`` cannot attend from image keys.

    Only affects modules with ``layer_idx`` (language ``Qwen3VLTextAttention``). Vision blocks unchanged.
    Set ``after_layer=None`` or pass ``image_key_indices=None`` to disable (no math change; still patches briefly).
    """
    prev = (
        _IMAGE_OUTFLOW_ABLATION["enabled"],
        _IMAGE_OUTFLOW_ABLATION["after_layer"],
        _IMAGE_OUTFLOW_ABLATION["key_indices"],
    )
    qwen3vl_modeling.eager_attention_forward = _eager_attention_forward_with_image_outflow_ablation
    _IMAGE_OUTFLOW_ABLATION["enabled"] = after_layer is not None and image_key_indices is not None
    _IMAGE_OUTFLOW_ABLATION["after_layer"] = after_layer
    _IMAGE_OUTFLOW_ABLATION["key_indices"] = image_key_indices
    try:
        yield
    finally:
        _IMAGE_OUTFLOW_ABLATION["enabled"] = prev[0]
        _IMAGE_OUTFLOW_ABLATION["after_layer"] = prev[1]
        _IMAGE_OUTFLOW_ABLATION["key_indices"] = prev[2]
        qwen3vl_modeling.eager_attention_forward = _ORIG_QWEN3VL_EAGER_ATTN


def mean_over_heads(attn: torch.Tensor) -> torch.Tensor:
    """[B, H, T, T] -> [T, T] (batch must be 1)."""
    if attn.shape[0] != 1:
        raise ValueError(f"expected batch 1, got {attn.shape[0]}")
    return attn[0].float().mean(dim=0).cpu()


def submatrix_nonimage_queries_image_keys(
    W_tt: torch.Tensor,
    image_key_indices: torch.LongTensor,
) -> tuple[list[int], list[int], np.ndarray]:
    """Crop mean attention ``W_tt[query, key]`` to rows = non-image queries, cols = image keys.

    Returns ``(query_indices, key_indices, M)`` with ``M[r,c] = W_tt[query_indices[r], key_indices[c]]``.
    """
    t = W_tt.shape[0]
    if W_tt.shape[1] != t:
        raise ValueError(f"expected square [T,T], got {tuple(W_tt.shape)}")
    img = sorted({int(i) for i in image_key_indices.tolist() if 0 <= int(i) < t})
    img_set = set(img)
    non = [i for i in range(t) if i not in img_set]
    if not img or not non:
        return non, img, np.zeros((len(non), len(img)), dtype=np.float64)
    w = W_tt.float().cpu().numpy()
    m = w[np.ix_(non, img)]
    return non, img, m


def _contiguous_value_runs(sorted_positions: np.ndarray) -> list[tuple[int, int]]:
    """Consecutive index ranges in ``sorted_positions`` (1-D ascending). Returns ``(run_start_idx, run_end_idx)`` slices."""
    if sorted_positions.size == 0:
        return []
    breaks = np.where(np.diff(sorted_positions) != 1)[0] + 1
    starts = np.concatenate(([0], breaks))
    ends = np.concatenate((breaks, [sorted_positions.size]))
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


def _llm_merge_hw_from_flat_k(k: int, llm_h: int, llm_w: int) -> tuple[int, int]:
    """Map LM vision token flat index ``k`` to merge-cell ``(h_m, w_m)``.

    Order matches ``Qwen2VLImageProcessor`` ``flatten_patches``: temporal block outermost, then row-major over the
    ``llm_h × llm_w`` merged spatial grid (i.e. ``k = t·(llm_h·llm_w) + h_m·llm_w + w_m`` for fixed ``t``).
    """
    spatial = llm_h * llm_w
    rem = int(k) % spatial
    h_m = rem // llm_w
    w_m = rem % llm_w
    if not (0 <= h_m < llm_h and 0 <= w_m < llm_w):
        raise ValueError(f"bad (h_m,w_m) for k={k}, grid h={llm_h} w={llm_w}")
    return h_m, w_m


def _qwen_vl_image_processor_min_max_pixels(image_processor) -> tuple[int, int]:
    """Resolve pixel bounds for ``smart_resize`` (some checkpoints leave ``min_pixels`` / ``max_pixels`` unset)."""
    _default_min = 56 * 56
    _default_max = 28 * 28 * 1280
    mn = getattr(image_processor, "min_pixels", None)
    mx = getattr(image_processor, "max_pixels", None)
    if mn is None or mx is None:
        sz = getattr(image_processor, "size", None)
        if isinstance(sz, dict):
            if mn is None:
                mn = sz.get("shortest_edge", _default_min)
            if mx is None:
                mx = sz.get("longest_edge", _default_max)
        else:
            if mn is None:
                mn = _default_min
            if mx is None:
                mx = _default_max
    return int(mn), int(mx)


def build_qwen_vl_image_key_thumbnails(
    pil_rgb: Image.Image,
    image_processor,
    input_ids_1d: torch.Tensor,
    image_grid_thw: torch.Tensor,
    image_token_id: int,
    *,
    thumb_max_px: int = 26,
) -> dict[int, Image.Image]:
    """Crop one thumbnail per image placeholder **sequence index** (LLM key position).

    Uses the same ``smart_resize`` + patch/merge grid as ``Qwen2VLImageProcessor`` so crops line up with merged
    vision tokens (each thumbnail = ``merge_size × patch_size`` square region on the **resized** canvas).
    """
    merge = int(image_processor.merge_size)
    patch_size = int(image_processor.patch_size)
    _min_px, _max_px = _qwen_vl_image_processor_min_max_pixels(image_processor)
    w0, h0 = pil_rgb.size
    rh, rw = smart_resize(
        h0,
        w0,
        factor=patch_size * merge,
        min_pixels=_min_px,
        max_pixels=_max_px,
    )
    resized = pil_rgb.convert("RGB").resize((rw, rh), Image.Resampling.BICUBIC)

    pos = torch.nonzero(input_ids_1d == image_token_id, as_tuple=False).squeeze(-1)
    if pos.numel() == 0:
        return {}
    pn = pos.detach().cpu().numpy().astype(np.int64)
    runs = _contiguous_value_runs(pn)
    grid = image_grid_thw.detach().cpu().long()
    if grid.dim() == 1:
        grid = grid.unsqueeze(0)
    n_img = int(grid.shape[0])
    if len(runs) != n_img:
        raise ValueError(
            f"image token runs ({len(runs)}) != image_grid_thw rows ({n_img}); "
            "check multimodal inputs or extend run-splitting for your layout."
        )

    thumbs: dict[int, Image.Image] = {}
    cell_px = merge * patch_size
    for run_i, (sli, sri) in enumerate(runs):
        thw = grid[run_i].tolist()
        tp, hp, wp = int(thw[0]), int(thw[1]), int(thw[2])
        n_tok_run = (tp * hp * wp) // (merge * merge)
        seq_positions = pn[sli:sri]
        if seq_positions.size != n_tok_run:
            raise ValueError(
                f"image run {run_i}: {seq_positions.size} positions vs prod(thw)//merge²={n_tok_run}"
            )
        llm_h = hp // merge
        llm_w_m = wp // merge
        if llm_h * merge != hp or llm_w_m * merge != wp:
            raise ValueError(f"grid {hp}x{wp} not divisible by merge_size={merge}")
        if n_tok_run != tp * llm_h * llm_w_m:
            raise ValueError(
                f"token count {n_tok_run} != T·H·W = {tp * llm_h * llm_w_m} "
                f"(thw=({tp},{hp},{wp}), merge={merge})"
            )

        for k in range(n_tok_run):
            h_m, w_m = _llm_merge_hw_from_flat_k(k, llm_h, llm_w_m)
            y0 = h_m * cell_px
            y1 = (h_m + 1) * cell_px
            x0 = w_m * cell_px
            x1 = (w_m + 1) * cell_px
            crop = resized.crop((x0, y0, x1, y1))
            side = min(crop.size[0], crop.size[1], thumb_max_px)
            if side < 1:
                side = 1
            thumb = crop.resize((side, side), Image.Resampling.BILINEAR)
            tok_ix = int(seq_positions[k])
            thumbs[tok_ix] = thumb

    return thumbs


def _data_dx_to_display_px(ax, x0: float, y0: float, x1: float, y1: float) -> float:
    """Pixel length of vector (x0,y0)→(x1,y1) in data space (handles skew / non-linear coords rarely)."""
    p0 = ax.transData.transform((float(x0), float(y0)))
    p1 = ax.transData.transform((float(x1), float(y1)))
    return float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))


def _offsetimage_zoom_for_cell_width(
    ax,
    fig,
    arr_rgb: np.ndarray,
    *,
    thumb_y_data: float,
    patch_px_narrower_than_cell: float,
    zoom_override: float | None,
) -> float:
    """``OffsetImage`` zoom so displayed width ≈ heatmap column width minus ``patch_px_narrower_than_cell`` pixels."""
    if zoom_override is not None:
        return float(zoom_override)
    fig.canvas.draw()
    cell_w = _data_dx_to_display_px(ax, 0.0, thumb_y_data, 1.0, thumb_y_data)
    target_w = max(4.0, cell_w - float(patch_px_narrower_than_cell))
    side = max(arr_rgb.shape[0], arr_rgb.shape[1], 1)
    probe = np.ones((side, side, 3), dtype=float)
    oi_p = OffsetImage(probe, zoom=1.0)
    ab_p = AnnotationBbox(
        oi_p,
        (0.0, float(thumb_y_data)),
        xycoords=ax.transData,
        frameon=False,
        pad=0.0,
    )
    ax.add_artist(ab_p)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    w1 = ab_p.get_window_extent(renderer).width
    ab_p.remove()
    k = w1 / float(side)
    if k < 1e-9:
        return 1.0
    return float(target_w / (float(arr_rgb.shape[1]) * k))


def all_causal_edges(W: torch.Tensor, *, mass_floor: float = 1e-6) -> list[tuple[int, int, float]]:
    """All causal (query i, key j) pairs with j <= i and mass >= ``mass_floor``. W: [T,T]."""
    t = W.shape[0]
    mf = float(mass_floor)
    edges: list[tuple[int, int, float]] = []
    Wf = W.float()
    for i in range(t):
        for j in range(i + 1):
            v = float(Wf[i, j].item())
            if v >= mf:
                edges.append((i, j, v))
    return edges


def filter_edges_by_min_mass(
    edges: list[tuple[int, int, float]],
    min_mass: float,
) -> list[tuple[int, int, float]]:
    """Keep edges with attention mass strictly greater than ``min_mass``."""
    if min_mass < 0.0:
        raise ValueError("min_mass must be >= 0")
    return [e for e in edges if e[2] > min_mass]


def drop_edges_with_source_keys_in(
    edges: list[tuple[int, int, float]],
    omit_key_indices: frozenset[int],
) -> list[tuple[int, int, float]]:
    """Remove edges whose *source key* index is in ``omit_key_indices`` (information-flow: key → query)."""
    if not omit_key_indices:
        return edges
    return [e for e in edges if e[1] not in omit_key_indices]


def keep_edges_with_keys_in(
    edges: list[tuple[int, int, float]],
    key_indices: frozenset[int],
) -> list[tuple[int, int, float]]:
    """Keep edges whose *source key* is in ``key_indices``. Empty ``key_indices`` → no edges kept."""
    if not key_indices:
        return []
    return [e for e in edges if e[1] in key_indices]


def drop_edges_touching_token_indices(
    edges: list[tuple[int, int, float]],
    hidden_indices: frozenset[int],
) -> list[tuple[int, int, float]]:
    """Drop edges whose query (``e[0]``) or key (``e[1]``) is in ``hidden_indices``."""
    if not hidden_indices:
        return edges
    return [e for e in edges if e[0] not in hidden_indices and e[1] not in hidden_indices]


def annotate_attention_matrix_causal(
    ax,
    M_np: np.ndarray,
    query_orig: list[int],
    key_orig: list[int],
    *,
    decimal_places: int | None = None,
) -> None:
    """Label cells where key position <= query position (causal). ``M_np`` shape (n_query, n_key)."""
    n_r, n_c = int(M_np.shape[0]), int(M_np.shape[1])
    if n_r == 0 or n_c == 0:
        return
    if len(query_orig) != n_r or len(key_orig) != n_c:
        raise ValueError("query_orig / key_orig length must match M_np rows / cols")
    nmax = max(n_r, n_c)
    if decimal_places is None:
        decimal_places = 2  # .2f on heatmaps (was 1 when nmax>22, hiding e.g. 0.01 as 0.0)
    fs = float(np.clip(8.0 - 0.25 * nmax, 2.8, 7.2))
    fmt = f"{{:.{decimal_places}f}}"
    outline = [
        mpath_effects.Stroke(linewidth=1.6, foreground="0.12"),
        mpath_effects.Normal(),
    ]
    for ri in range(n_r):
        for ci in range(n_c):
            if key_orig[ci] > query_orig[ri]:
                continue
            v = float(M_np[ri, ci])
            t = ax.text(
                float(ci),
                float(ri),
                fmt.format(v),
                ha="center",
                va="center",
                fontsize=fs,
                color="0.98",
                zorder=6,
            )
            t.set_path_effects(outline)


def annotate_attention_matrix_causal_lower(
    ax,
    M_np: np.ndarray,
    keep_orig: list[int],
    *,
    decimal_places: int | None = None,
) -> None:
    """Square ``W[np.ix_(keep, keep)]``: same as :func:`annotate_attention_matrix_causal` with query=key=``keep_orig``."""
    annotate_attention_matrix_causal(ax, M_np, keep_orig, keep_orig, decimal_places=decimal_places)


def token_index_to_plot_x(token_index: int, hidden_indices: frozenset[int]) -> int | None:
    """Map original token index to compact x (hidden indices omitted). ``None`` if this token is hidden."""
    if token_index in hidden_indices:
        return None
    return int(token_index - sum(1 for h in hidden_indices if h < token_index))


def _flow_edge_draw_stats(
    edges: list[tuple[int, int, float]],
    *,
    hide_token_indices: frozenset[int],
) -> tuple[int, int, int, int]:
    """Return (n_self_loop, n_hidden_skip, n_draw, n_listed) for the flow-plot drawing rules."""
    n_listed = len(edges)
    n_self = 0
    n_hidden = 0
    n_draw = 0
    for qi, kj, _m in edges:
        if qi == kj:
            n_self += 1
            continue
        pk = token_index_to_plot_x(kj, hide_token_indices)
        pq = token_index_to_plot_x(qi, hide_token_indices)
        if pk is None or pq is None:
            n_hidden += 1
            continue
        n_draw += 1
    return n_self, n_hidden, n_draw, n_listed


def _print_flow_edges_debug(
    ell: int,
    edges: list[tuple[int, int, float]],
    token_labels: list[str],
    hide_token_indices: frozenset[int],
) -> None:
    """Explain why edges do or do not become arrows (self-loop / hidden column)."""
    if not edges:
        return
    print(f"    [debug layer {ell}] edges in list (key j → query i; arrow drawn key → query):")
    for qi, kj, w in edges:
        lq = token_labels[qi][:22] + ("…" if qi < len(token_labels) and len(token_labels[qi]) > 22 else "") if qi < len(
            token_labels
        ) else "?"
        lk = token_labels[kj][:22] + ("…" if kj < len(token_labels) and len(token_labels[kj]) > 22 else "") if kj < len(
            token_labels
        ) else "?"
        if qi == kj:
            why = "SKIP plot: self-loop (qi==kj); matrix cell still shows A[q,i] diagonal"
        elif token_index_to_plot_x(kj, hide_token_indices) is None:
            why = f"SKIP plot: key j={kj} is hidden"
        elif token_index_to_plot_x(qi, hide_token_indices) is None:
            why = f"SKIP plot: query i={qi} is hidden"
        else:
            why = "DRAW arrow"
        print(f"      A[{qi},{kj}]={w:.4f}  k{kj}({lk!r})→q{qi}({lq!r})  | {why}")


def _print_flow_drawn_edges_one_line(
    ell: int,
    edges: list[tuple[int, int, float]],
    token_labels: list[str],
    hide_token_indices: frozenset[int],
    *,
    max_show: int = 12,
) -> None:
    """List edges that will actually get an arc (non-self, both endpoints visible)."""
    drawn: list[tuple[int, int, float]] = []
    for qi, kj, w in edges:
        if qi == kj:
            continue
        if token_index_to_plot_x(kj, hide_token_indices) is None or token_index_to_plot_x(qi, hide_token_indices) is None:
            continue
        drawn.append((qi, kj, w))
    if not drawn:
        return
    parts: list[str] = []
    for qi, kj, w in drawn[:max_show]:
        lq = token_labels[qi][:14] + ("…" if qi < len(token_labels) and len(token_labels[qi]) > 14 else "") if qi < len(
            token_labels
        ) else "?"
        lk = token_labels[kj][:14] + ("…" if kj < len(token_labels) and len(token_labels[kj]) > 14 else "") if kj < len(
            token_labels
        ) else "?"
        parts.append(f"A[{qi},{kj}]={w:.3f} k{kj}({lk!r})→q{qi}({lq!r})")
    tail = f" … (+{len(drawn) - max_show} more)" if len(drawn) > max_show else ""
    print(f"    → drawn arcs: {' | '.join(parts)}{tail}")


def save_attention_figure(
    *,
    token_labels: list[str],
    attn_layers: list[torch.Tensor],
    user_prompt: str,
    prediction_text: str,
    out_path: Path,
    dpi: int = 150,
    edge_min_mass: float | None = None,
    omit_key_indices: frozenset[int] = frozenset(),
    edge_visual_norm: str = "per_layer",
    layer_y_scale: float = 1.85,
    fig_height_per_layer_in: float = 0.58,
    fig_height_cap_in: float = 140.0,
    fig_width_base_in: float = 5.0,
    fig_width_per_token_in: float = 0.38,
    fig_width_min_in: float = 6.5,
    fig_width_cap_in: float = 130.0,
    hide_token_indices: frozenset[int] = frozenset(),
    only_flow_key_indices: frozenset[int] | None = None,
    highlight_column_token_indices: frozenset[int] | None = None,
) -> None:
    """Matplotlib: x = token index; y = layers (0 at top); top band = token labels; bottom = prediction.

    **Hidden tokens:** ``hide_token_indices`` removes those columns from the plot and drops every edge touching a
    hidden index (e.g. ``frozenset({0})`` if token 0 is a sink and you want a cleaner diagram). Default is show all.
    **Self-loops** (``query index == key index``) are kept in the filtered edge list for counting but **not** drawn
    as arcs (the diagonal is visible in matrix plots). ``omit_key_indices`` only drops flows whose *source key* is
    listed. If ``edge_min_mass`` is set, only edges with mass strictly ``> edge_min_mass`` are drawn (among causal
    pairs with mass >= 1e-6 before that filter).

    **Image-only arrows:** if ``only_flow_key_indices`` is set, keep only edges whose **key** (arrow tail) is in that
    set—e.g. positions where ``input_ids == image_token_id``—so you see which layers route information from image
    tokens into later queries. Matrix PNGs are unchanged.

    **Column highlight:** if ``highlight_column_token_indices`` is non-empty, draw a light vertical band and stronger
    markers at those **original** token x positions (e.g. all image token indices).
    """
    n_layers = len(attn_layers)
    n_tok = len(token_labels)
    assert all(a.shape == (n_tok, n_tok) for a in attn_layers)
    if edge_visual_norm not in ("global", "per_layer"):
        raise ValueError("edge_visual_norm must be 'global' or 'per_layer'")
    ys = float(layer_y_scale)
    if ys <= 0:
        raise ValueError("layer_y_scale must be > 0")

    per_layer_edges: list[list[tuple[int, int, float]]] = []
    for W in attn_layers:
        edges = all_causal_edges(W, mass_floor=1e-6)
        if edge_min_mass is not None:
            edges = filter_edges_by_min_mass(edges, min_mass=edge_min_mass)
        edges = drop_edges_with_source_keys_in(edges, omit_key_indices)
        edges = drop_edges_touching_token_indices(edges, hide_token_indices)
        if only_flow_key_indices is not None:
            edges = keep_edges_with_keys_in(edges, only_flow_key_indices)
        per_layer_edges.append(edges)

    edge_filter_note = f"mass > {edge_min_mass}" if edge_min_mass is not None else "mass >= 1e-6"
    _imgk = (
        f"; arrow tail (key) ∈ {len(only_flow_key_indices)} image pos."
        if only_flow_key_indices is not None
        else ""
    )
    print(f"attention edges per layer ({edge_filter_note}, source-key filter{_imgk}):")
    print("  (listed = after mass/hide/omit/image-key filters; self-loops are not drawn as arcs; matrix row=query col=key)")
    for ell, edges in enumerate(per_layer_edges):
        n_self, n_hid, n_draw, n_list = _flow_edge_draw_stats(edges, hide_token_indices=hide_token_indices)
        print(
            f"  layer {ell:>3}: {n_list} listed | {n_draw} arrow(s) drawn | "
            f"{n_self} self-loop(s) skipped | {n_hid} cross-edge(s) skipped (hidden tok)"
        )
        if n_draw > 0:
            _print_flow_drawn_edges_one_line(ell, edges, token_labels, hide_token_indices)
        elif edges:
            _print_flow_edges_debug(ell, edges, token_labels, hide_token_indices)
    # Arrow styling: use only **cross-edges** (qi != kj). Self-loops are listed for debug/counts but not drawn;
    # if their masses were included here, a strong diagonal + one weak arc would force the arc to n≈0 (invisible).
    all_w = [w for edges in per_layer_edges for qi, kj, w in edges if qi != kj]
    g_w_min = min(all_w) if all_w else 0.0
    g_w_max = max(all_w) if all_w else 1.0
    per_layer_mass_range: list[tuple[float, float]] = []
    for edges in per_layer_edges:
        ms = [e[2] for e in edges if e[0] != e[1]]
        if not ms:
            per_layer_mass_range.append((0.0, 1.0))
        else:
            per_layer_mass_range.append((min(ms), max(ms)))

    try:
        _edge_cmap = mpl.colormaps["YlOrRd"]
    except (AttributeError, KeyError):
        _edge_cmap = plt.cm.get_cmap("YlOrRd")

    def edge_strength_norm(mass: float, layer_idx: int) -> float:
        """Within-layer contrast; if min==max (single mass / identical edges), use absolute mass so arrows stay visible."""
        if edge_visual_norm == "per_layer":
            w_min, w_max = per_layer_mass_range[layer_idx]
        else:
            w_min, w_max = g_w_min, g_w_max
        spread = w_max - w_min
        if spread <= 1e-6:
            return float(np.clip(mass, 0.0, 1.0)) ** 0.92
        return float(np.clip((mass - w_min) / spread, 0.0, 1.0))

    def edge_visual(mass: float, layer_idx: int) -> tuple[float, float, tuple[float, float, float, float]]:
        """linewidth, mutation_scale, rgba — stronger attention → thicker, darker, more opaque."""
        n = edge_strength_norm(mass, layer_idx)
        lw = 0.45 + 5.2 * (n**0.92)
        alpha = 0.14 + 0.78 * n
        rgb = _edge_cmap(0.12 + 0.88 * n)[:3]
        return lw, 6.0 + 18.0 * n, (*rgb, alpha)

    # y: embedding row on top, then layer 0 .. L-1 (scaled by ``ys`` for vertical separation)
    y_embed = ys * (n_layers + 0.85)

    def y_layer(ell: int) -> float:
        return ys * (n_layers - ell - 0.5)

    _n_hidden = sum(1 for h in hide_token_indices if 0 <= h < n_tok)
    n_vis = max(n_tok - _n_hidden, 1)
    fig_w_in = float(
        np.clip(
            float(fig_width_base_in) + float(fig_width_per_token_in) * n_vis,
            float(fig_width_min_in),
            float(fig_width_cap_in),
        )
    )
    fig_h_in = float(
        np.clip(
            float(fig_height_per_layer_in) * n_layers + 8.0,
            14.0,
            float(fig_height_cap_in),
        )
    )
    print(
        f"attention figure size: {fig_w_in:.1f} x {fig_h_in:.1f} in (W x H), "
        f"n_tok={n_tok} (plot {n_vis} cols), hidden={sorted(hide_token_indices)}, n_layers={n_layers}"
    )
    fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in), layout="constrained")

    _plot_coords: list[tuple[int, int]] = []
    for ti in range(n_tok):
        px = token_index_to_plot_x(ti, hide_token_indices)
        if px is not None:
            _plot_coords.append((px, ti))
    _max_px = max((p[0] for p in _plot_coords), default=0)

    _x_left = float(min(-0.6, -0.85 * ys - 0.35))
    ax.set_xlim(_x_left, max(_max_px + 0.5, 0.4))
    ax.set_ylim(-1.35 * ys, y_embed + 0.55 * ys)
    ax.set_facecolor("#fafafa")
    ax.axhline(y_embed - 0.35 * ys, color="#ddd", lw=0.8, zorder=0)
    _band = 0.45 * ys
    for ell in range(n_layers):
        ax.axhspan(
            y_layer(ell) - _band,
            y_layer(ell) + _band,
            facecolor=("#f8f8fc" if ell % 2 == 0 else "#f2f4f8"),
            zorder=0,
            lw=0,
        )

    _hl_cols = highlight_column_token_indices or frozenset()
    for _ti in sorted(_hl_cols):
        _pxh = token_index_to_plot_x(_ti, hide_token_indices)
        if _pxh is None:
            continue
        _xfh = float(_pxh)
        ax.axvspan(
            _xfh - 0.5,
            _xfh + 0.5,
            facecolor="#fff3e0",
            edgecolor="#ff9800",
            linewidth=0.75,
            alpha=0.38,
            zorder=0.35,
        )

    _fs = float(np.clip(900.0 / max(n_vis, 1), 4.0, 9.0))
    for i, lab in enumerate(token_labels):
        xi = token_index_to_plot_x(i, hide_token_indices)
        if xi is None:
            continue
        xf = float(xi)
        _col_hl = i in _hl_cols
        if lab == ".":
            _ms = max(2.2, _fs * 0.35) * (1.35 if _col_hl else 1.0)
            ax.plot(
                xf,
                y_embed,
                "o",
                color=("#e65100" if _col_hl else "#222"),
                ms=_ms,
                zorder=4,
            )
        else:
            ax.text(
                xf,
                y_embed,
                lab,
                rotation=58,
                ha="right",
                va="bottom",
                fontsize=_fs,
                color=("#bf360c" if _col_hl else "#111"),
                fontweight=("600" if _col_hl else "normal"),
                zorder=4,
            )

    try:
        _cmap = mpl.colormaps["tab10"]
    except (AttributeError, KeyError):
        _cmap = plt.cm.get_cmap("tab10")
    _layer_colors = _cmap(np.linspace(0, 0.85, max(n_layers, 1)))
    for ell in range(n_layers):
        yl = y_layer(ell)
        c = _layer_colors[ell % len(_layer_colors)]
        ax.text(
            -0.55 * ys,
            yl,
            f"layer {ell}",
            ha="right",
            va="center",
            fontsize=9,
            color=c,
            fontweight="600",
            zorder=5,
        )
        edges = per_layer_edges[ell]
        for qi, kj, mass in edges:
            if qi == kj:
                continue
            # Information flow: value from key kj into query qi; arrow key → query (head at query).
            px_k = token_index_to_plot_x(kj, hide_token_indices)
            px_q = token_index_to_plot_x(qi, hide_token_indices)
            if px_k is None or px_q is None:
                continue
            x_key, x_q = float(px_k), float(px_q)
            _dx = abs(x_q - x_key)
            _sgn = 1.0 if x_q > x_key else -1.0
            # Shallow arcs: small base rad, extra flattening when |Δx| is large (avoids tall bows on long hops).
            _rad0 = 0.1 * ys
            _flatten = 1.0 / (1.0 + 0.22 * max(_dx - 1.0, 0.0))
            rad = _rad0 * _sgn * _flatten
            lw, mut, rgba = edge_visual(mass, ell)
            arr = FancyArrowPatch(
                (x_key, yl),
                (x_q, yl),
                connectionstyle=f"arc3,rad={rad}",
                arrowstyle="-|>",
                mutation_scale=mut,
                linewidth=lw,
                color=rgba[:3],
                alpha=rgba[3],
                zorder=1,
            )
            ax.add_patch(arr)
        _xs_other: list[float] = []
        _xs_hl: list[float] = []
        for ti in range(n_tok):
            px = token_index_to_plot_x(ti, hide_token_indices)
            if px is None:
                continue
            if ti in _hl_cols:
                _xs_hl.append(float(px))
            else:
                _xs_other.append(float(px))
        if _xs_other:
            ax.plot(
                _xs_other,
                np.full(len(_xs_other), yl),
                "o",
                ms=3.5,
                color="#111",
                markeredgecolor="w",
                markeredgewidth=0.4,
                zorder=3,
            )
        if _xs_hl:
            ax.plot(
                _xs_hl,
                np.full(len(_xs_hl), yl),
                "o",
                ms=4.6,
                color="#e65100",
                markeredgecolor="white",
                markeredgewidth=0.55,
                zorder=3,
            )

    _omit_note = (
        f"; omit key indices {sorted(omit_key_indices)}" if omit_key_indices else "; show all source keys"
    )
    _hide_note = (
        f" · hidden tok {sorted(hide_token_indices)}" if hide_token_indices else ""
    )
    _norm_note = f" · edge color/LW norm={edge_visual_norm}"
    _mass_note = f", mass > {edge_min_mass}" if edge_min_mass is not None else ""
    _img_flow_note = (
        f" · image-key tails only ({len(only_flow_key_indices)} pos.)"
        if only_flow_key_indices is not None
        else ""
    )
    _hl_note = (
        f" · orange cols = image toks ({len(highlight_column_token_indices)} idx)"
        if highlight_column_token_indices
        else ""
    )
    ax.set_title(
        f"Information flow (key → query); causal edges / layer{_mass_note}{_img_flow_note}{_hl_note}"
        f"{_omit_note}{_hide_note}{_norm_note} · {user_prompt[:90]}",
        fontsize=11,
        pad=12,
    )
    _leg_low = (*_edge_cmap(0.15)[:3], 0.35)
    _leg_high = (*_edge_cmap(0.92)[:3], 0.95)
    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=_leg_low[:3],
                lw=0.9,
                alpha=_leg_low[3],
                label="weaker (in this figure)",
            ),
            Line2D(
                [0],
                [0],
                color=_leg_high[:3],
                lw=4.8,
                alpha=_leg_high[3],
                label="stronger",
            ),
        ],
        loc="lower right",
        bbox_to_anchor=(0.99, 0.04),
        bbox_transform=ax.transAxes,
        borderaxespad=0.5,
        frameon=True,
        framealpha=0.92,
        fontsize=8,
        title=(
            "Attention: thin / pale → thick / red (norm "
            + ("per layer)" if edge_visual_norm == "per_layer" else "global)")
        ),
        title_fontsize=8,
    )
    if _plot_coords:
        _tcks = [float(px) for px, _ in _plot_coords]
        _labs = [str(oti) for _, oti in _plot_coords]
        ax.set_xticks(_tcks)
        ax.set_xticklabels(_labs, fontsize=7)
    ax.set_xlabel("token index (original; hidden columns omitted)")
    ax.set_ylabel("depth (embeddings at top → deeper layers → prediction below)")
    ax.yaxis.set_visible(False)
    ax.tick_params(axis="x", labelsize=7)
    pred_short = textwrap.fill(
        prediction_text.replace("\n", " ")[:1200],
        width=110,
        break_long_words=False,
        break_on_hyphens=False,
    )
    fig.text(0.02, 0.01, f"Generation:\n{pred_short}", fontsize=9, va="bottom", ha="left")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _matrix_axis_ticks(
    n: int,
    labels: list[str],
    max_axis_labels: int,
) -> tuple[np.ndarray, list[str], int]:
    if n <= max_axis_labels:
        return np.arange(n), labels, 7
    step = max(1, (n + max_axis_labels - 1) // max_axis_labels)
    tick_idx = list(range(0, n, step))
    if tick_idx[-1] != n - 1:
        tick_idx.append(n - 1)
    tidx = np.asarray(tick_idx, dtype=int)
    return tidx, [labels[i] for i in tidx], 6


def _matrix_query_keep_after_max_ref(
    keep: list[int],
    queries_after_positions: frozenset[int] | None,
) -> tuple[list[int], str]:
    """Restrict query rows to original indices **strictly greater** than ``max(queries_after_positions)``.

    Typical use: pass image token index set so rows are only tokens **after** the last image slot—attention from
    image keys into downstream queries.
    """
    if not queries_after_positions:
        return keep, ""
    cut = max(queries_after_positions)
    qk = sorted(i for i in keep if i > cut)
    if not qk:
        raise ValueError(
            f"matrix row filter: no visible queries with index > {cut} (last ref position). "
            "Adjust HIDE_TOKEN_INDICES or disable queries-after-image rows."
        )
    return qk, f" · query rows > {cut}"


def save_attention_matrices_figure(
    attn_layers: list[torch.Tensor],
    *,
    out_path: Path,
    hide_token_indices: frozenset[int] = frozenset(),
    cmap_vmin: float | None = None,
    cmap_vmax: float | None = None,
    annotate_causal_lower: bool = True,
    only_matrix_key_indices: frozenset[int] | None = None,
    only_matrix_queries_after_max_of: frozenset[int] | None = None,
    dpi: int = 150,
    subplot_size_in: float = 2.05,
    cmap: str = "viridis",
) -> None:
    """One subplot per layer: heatmap of mean-head attention ``A[query, key]`` (rows=query, cols=key).

    Color scale is ``[0, 1]`` by default. If ``cmap_vmin`` / ``cmap_vmax`` are set, the norm is ``[vmin, vmax]``
    (``clip=True``). Defaults: ``vmin=0`` when omitted, ``vmax=1`` when omitted. Require ``vmin < vmax``.
    Matrices are cropped when ``hide_token_indices`` is non-empty.
    If ``only_matrix_key_indices`` is set, columns are restricted to those **original** key positions (e.g. image
    tokens). If ``only_matrix_queries_after_max_of`` is set (e.g. same id set as image), rows are only visible
    queries with index **>** ``max(that set)``—downstream tokens that read out from the image block.
    If ``annotate_causal_lower``, overlay masses on causal cells (``key_idx <= query_idx``).
    """
    if not attn_layers:
        raise ValueError("attn_layers is empty")
    n_layers = len(attn_layers)
    n_tok = attn_layers[0].shape[0]
    for W in attn_layers:
        if tuple(W.shape) != (n_tok, n_tok):
            raise ValueError(f"expected square {n_tok}x{n_tok}, got {tuple(W.shape)}")

    keep = [i for i in range(n_tok) if i not in hide_token_indices]
    if not keep:
        raise ValueError("hide_token_indices removed all tokens")
    if only_matrix_key_indices is not None:
        key_keep = sorted(i for i in keep if i in only_matrix_key_indices)
        if not key_keep:
            raise ValueError(
                "only_matrix_key_indices has no overlap with visible token rows; cannot build key columns."
            )
    else:
        key_keep = keep
    query_keep, _qr_note = _matrix_query_keep_after_max_ref(keep, only_matrix_queries_after_max_of)
    n_q, n_key = len(query_keep), len(key_keep)
    _knote = f" · keys⊆image ({n_key} cols)" if only_matrix_key_indices is not None else ""
    _rnote = _qr_note

    ncols = int(np.ceil(np.sqrt(n_layers)))
    nrows = int(np.ceil(n_layers / ncols))
    fig_w = subplot_size_in * ncols + 1.2
    fig_h = subplot_size_in * nrows + 1.0
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(fig_w, fig_h),
        layout="constrained",
        squeeze=False,
    )

    _lo = 0.0 if cmap_vmin is None else float(cmap_vmin)
    _hi = 1.0 if cmap_vmax is None else float(cmap_vmax)
    if cmap_vmin is not None and not (0.0 <= _lo < 1.0):
        raise ValueError("cmap_vmin must be in [0, 1)")
    if cmap_vmax is not None and not (0.0 < _hi <= 1.0):
        raise ValueError("cmap_vmax must be in (0, 1]")
    if not (_lo < _hi):
        raise ValueError(f"cmap_vmin ({_lo}) must be < cmap_vmax ({_hi})")
    norm = Normalize(vmin=_lo, vmax=_hi, clip=True)
    last_im = None
    for ell, W in enumerate(attn_layers):
        r, c = divmod(ell, ncols)
        ax = axes[r][c]
        M_np = W.detach().float().cpu().numpy()[np.ix_(query_keep, key_keep)]
        last_im = ax.imshow(M_np, aspect="auto", cmap=cmap, norm=norm, origin="upper", interpolation="nearest")
        if annotate_causal_lower:
            annotate_attention_matrix_causal(ax, M_np, query_keep, key_keep)
        ax.set_title(f"layer {ell} · {n_q}×{n_key}{_knote}{_rnote}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    for ell in range(n_layers, nrows * ncols):
        r, c = divmod(ell, ncols)
        axes[r][c].axis("off")

    _hid = f" · hidden tok {sorted(hide_token_indices)}" if hide_token_indices else ""
    _cm_custom = cmap_vmin is not None or cmap_vmax is not None
    _cm = f" · cmap [{_lo:g}, {_hi:g}]" if _cm_custom else " · cmap [0, 1]"
    fig.suptitle(
        "Attention (mean over heads): row = query, column = key · query ids "
        f"{query_keep[0]}…{query_keep[-1]} · key ids {key_keep[0]}…{key_keep[-1]} → {n_q}×{n_key}"
        f"{_hid}{_cm}{_knote}{_rnote}",
        fontsize=11,
    )
    assert last_im is not None
    _axes_flat = axes.ravel().tolist()
    _cbl = f"weight [{_lo:g}, {_hi:g}]" if _cm_custom else "weight"
    fig.colorbar(last_im, ax=_axes_flat[:n_layers], shrink=0.55, label=_cbl)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_attention_matrices_per_layer_dir(
    attn_layers: list[torch.Tensor],
    *,
    out_dir: Path,
    hide_token_indices: frozenset[int] = frozenset(),
    cmap_vmin: float | None = None,
    cmap_vmax: float | None = None,
    annotate_causal_lower: bool = True,
    only_matrix_key_indices: frozenset[int] | None = None,
    only_matrix_queries_after_max_of: frozenset[int] | None = None,
    token_labels: list[str] | None = None,
    max_axis_labels: int = 40,
    dpi: int = 150,
    cmap: str = "viridis",
    fig_side_min_in: float = 5.5,
    fig_side_max_in: float = 18.0,
    inches_per_token: float = 0.24,
    image_key_thumbnails: dict[int, Image.Image] | None = None,
    matrix_image_key_patch_zoom: float | None = None,
    matrix_image_key_patch_bottom_frac: float = 0.26,
    matrix_patch_cell_inches: float = 0.38,
    matrix_fig_side_cap_in: float = 520.0,
    matrix_patch_px_narrower_than_cell: float = 2.0,
) -> None:
    """Write ``layer_00.png``, ``layer_01.png``, … under ``out_dir`` (one heatmap per file, larger / readable).

    Rows default to all visible **queries** (after ``hide_token_indices``). If ``only_matrix_queries_after_max_of``
    is set (e.g. image token indices), rows are only queries with original index **>** ``max(that set)``—tokens
    strictly after the image block—so the slice shows image→downstream flow.

    If ``only_matrix_key_indices`` is set, columns are only those **key** positions (e.g. image). If both are set,
    the matrix is ``|post_image_queries|×|image_keys|``.

    If both key and row filters are ``None``, the matrix is the full visible square ``|keep|×|keep|``.

    Optional ``cmap_vmin`` / ``cmap_vmax`` set ``matplotlib.colors.Normalize`` (default span ``0–1`` when both omitted).

    If ``image_key_thumbnails`` is set (map **original** sequence index → RGB thumb), draws those **below** the
    corresponding key columns (merged patch cells; see :func:`build_qwen_vl_image_key_thumbnails`). Uses full column
    ticks; text labels are blank where a patch replaces ``'.'``. Figure width/height use ``matrix_patch_cell_inches``
    per matrix cell (capped by ``matrix_fig_side_cap_in``) so many image columns stay readable; patch ``OffsetImage``
    zoom is chosen so on-screen patch width ≈ cell width minus ``matrix_patch_px_narrower_than_cell`` pixels (unless
    ``matrix_image_key_patch_zoom`` is set).

    Axis ticks use ``token_labels`` when provided (subsampled with ``max_axis_labels`` per axis independently).
    """
    if not attn_layers:
        raise ValueError("attn_layers is empty")
    n_layers = len(attn_layers)
    n_tok = attn_layers[0].shape[0]
    for W in attn_layers:
        if tuple(W.shape) != (n_tok, n_tok):
            raise ValueError(f"expected square {n_tok}x{n_tok}, got {tuple(W.shape)}")
    if token_labels is not None and len(token_labels) != n_tok:
        raise ValueError(f"token_labels length {len(token_labels)} != seq len {n_tok}")

    keep = [i for i in range(n_tok) if i not in hide_token_indices]
    if not keep:
        raise ValueError("hide_token_indices removed all tokens")
    if only_matrix_key_indices is not None:
        key_keep = sorted(i for i in keep if i in only_matrix_key_indices)
        if not key_keep:
            raise ValueError(
                "only_matrix_key_indices has no overlap with visible rows; cannot build key columns."
            )
        _mk_note = f" · key cols = image ({len(key_keep)})"
        _xlabel_extra = " · image keys only"
    else:
        key_keep = keep
        _mk_note = ""
        _xlabel_extra = ""

    query_keep, _qr_note = _matrix_query_keep_after_max_ref(keep, only_matrix_queries_after_max_of)
    n_q, n_key = len(query_keep), len(key_keep)
    _mk_note = f"{_mk_note}{_qr_note}" if _mk_note else _qr_note or ""
    row_labels = [token_labels[i] for i in query_keep] if token_labels is not None else [str(i) for i in query_keep]
    col_labels = [token_labels[i] for i in key_keep] if token_labels is not None else [str(i) for i in key_keep]

    tick_idx_row, tick_lbl_row, fs_r = _matrix_axis_ticks(n_q, row_labels, max_axis_labels)
    if image_key_thumbnails:
        tick_idx_col = np.arange(n_key, dtype=int)
        tick_lbl_col = [
            "" if key_keep[i] in image_key_thumbnails else col_labels[i] for i in range(n_key)
        ]
        fs_c = max(5, min(fs_r, 7))
    else:
        tick_idx_col, tick_lbl_col, fs_c = _matrix_axis_ticks(n_key, col_labels, max_axis_labels)
    fs = min(fs_r, fs_c)

    if image_key_thumbnails:
        _cell_in = float(matrix_patch_cell_inches)
        _cap_in = float(matrix_fig_side_cap_in)
    else:
        _cell_in = float(inches_per_token)
        _cap_in = float(fig_side_max_in)
    _marg = 5.5 if image_key_thumbnails else 3.2
    side_w = float(np.clip(_cell_in * n_key + _marg, fig_side_min_in, _cap_in))
    side_h = float(np.clip(_cell_in * n_q + _marg, fig_side_min_in, _cap_in))
    _fig_h_boost_in = 1.15 if image_key_thumbnails else 0.0

    _lo = 0.0 if cmap_vmin is None else float(cmap_vmin)
    _hi = 1.0 if cmap_vmax is None else float(cmap_vmax)
    if cmap_vmin is not None and not (0.0 <= _lo < 1.0):
        raise ValueError("cmap_vmin must be in [0, 1)")
    if cmap_vmax is not None and not (0.0 < _hi <= 1.0):
        raise ValueError("cmap_vmax must be in (0, 1]")
    if not (_lo < _hi):
        raise ValueError(f"cmap_vmin ({_lo}) must be < cmap_vmax ({_hi})")
    norm = Normalize(vmin=_lo, vmax=_hi, clip=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    _hid = f" · hidden {sorted(hide_token_indices)}" if hide_token_indices else ""
    _cm_custom = cmap_vmin is not None or cmap_vmax is not None
    _cm_note = f" · cmap [{_lo:g}, {_hi:g}]" if _cm_custom else ""
    _axis_note = "decoded token (orig. id in title)" if token_labels is not None else "orig. tok id"
    for ell, W in enumerate(attn_layers):
        M_np = W.detach().float().cpu().numpy()[np.ix_(query_keep, key_keep)]
        _fig_layout = None if image_key_thumbnails else "constrained"
        fig, ax = plt.subplots(
            figsize=(side_w + 0.9, side_h + 1.1 + _fig_h_boost_in),
            layout=_fig_layout,
        )
        if image_key_thumbnails:
            fig.subplots_adjust(
                bottom=matrix_image_key_patch_bottom_frac,
                left=0.10,
                right=0.90,
                top=0.88,
            )
        im = ax.imshow(
            M_np,
            aspect="equal",
            cmap=cmap,
            norm=norm,
            origin="upper",
            interpolation="nearest",
        )
        if annotate_causal_lower:
            annotate_attention_matrix_causal(ax, M_np, query_keep, key_keep)
        ax.set_ylabel(f"query row · {_axis_note}" + (" (after image)" if only_matrix_queries_after_max_of else ""))
        ax.set_xlabel(f"key col · {_axis_note}{_xlabel_extra}")
        ax.set_title(
            f"layer {ell} · mean over heads · {n_q}×{n_key}{_mk_note}{_hid}{_cm_note}\n"
            f"queries {query_keep[0]}…{query_keep[-1]} · keys {key_keep[0]}…{key_keep[-1]}",
            fontsize=11,
        )
        ax.set_xticks(tick_idx_col)
        ax.set_yticks(tick_idx_row)
        ax.set_xticklabels(tick_lbl_col, rotation=90, fontsize=fs)
        ax.set_yticklabels(tick_lbl_row, fontsize=fs)

        _thumb_y: float | None = None
        if image_key_thumbnails:
            # Reserve a strip below the heatmap (data coords; ``imshow`` origin upper → bottom has larger y).
            _y_lo, _y_hi = ax.get_ylim()
            _ypad = 1.45
            ax.set_ylim(_y_lo + _ypad, _y_hi)
            _thumb_y = float(_y_lo + _ypad * 0.55)

        _cbl = f"weight [{_lo:g}, {_hi:g}]" if _cm_custom else "weight"
        fig.colorbar(im, ax=ax, shrink=0.82, label=_cbl)

        if image_key_thumbnails and _thumb_y is not None:
            _probe_arr: np.ndarray | None = None
            for _ok in key_keep:
                _t = image_key_thumbnails.get(_ok)
                if _t is not None:
                    _probe_arr = np.asarray(_t)
                    break
            _zoom_use = matrix_image_key_patch_zoom
            if _probe_arr is not None and _zoom_use is None:
                _zoom_use = _offsetimage_zoom_for_cell_width(
                    ax,
                    fig,
                    _probe_arr,
                    thumb_y_data=_thumb_y,
                    patch_px_narrower_than_cell=matrix_patch_px_narrower_than_cell,
                    zoom_override=None,
                )
            if _zoom_use is None:
                _zoom_use = 1.0
            for ci, orig_k in enumerate(key_keep):
                thumb = image_key_thumbnails.get(orig_k)
                if thumb is None:
                    continue
                arr = np.asarray(thumb)
                oi = OffsetImage(arr, zoom=float(_zoom_use))
                ab = AnnotationBbox(
                    oi,
                    (float(ci), _thumb_y),
                    xycoords=ax.transData,
                    box_alignment=(0.5, 0.5),
                    frameon=False,
                    pad=0.0,
                )
                ax.add_artist(ab)
        fig.savefig(out_dir / f"layer_{ell:02d}.png", dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    print(f"wrote {n_layers} matrix PNGs under {out_dir}")


# %% Load model & processor

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    dtype="auto" if DEVICE.type == "cuda" else torch.float32,
    device_map="auto" if DEVICE.type == "cuda" else None,
    attn_implementation="eager",
)
if DEVICE.type != "cuda":
    model = model.to(DEVICE)

processor = AutoProcessor.from_pretrained(MODEL_ID)
print("Loaded:", MODEL_ID, "| eager attention OK")

# %% Build inputs (dog image + VQA)

IMAGE_PATH = REPO_ROOT / "data/images/cat-tiny.png"
USER_TEXT = "Is there a dog or a cat in the image?"

image = Image.open(IMAGE_PATH).convert("RGB")
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": USER_TEXT},
        ],
    }
]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
)
inputs = inputs.to(DEVICE)

_ids = inputs.input_ids[0]
token_labels = build_token_axis_labels(_ids, processor.tokenizer, model.config.image_token_id)
print("seq_len:", len(token_labels), "| image tokens:", sum(1 for x in token_labels if x == "."))
print("image:", IMAGE_PATH)

# %% Forward with attention hooks

lm = model.model.language_model

# Last **unablated** language layer (0-based). Layers with ``layer_idx >`` this value get: after softmax, attention
# mass from image placeholder *keys* to non-image *queries* is removed and each query row renormalized.
ZERO_IMAGE_OUTFLOW_AFTER_LAYER: int | None = None
_img_key_idx = image_token_positions(_ids, model.config.image_token_id)
print(
    "image LM positions:",
    _img_key_idx.tolist(),
    "| zero image→non-image attention in layers >",
    ZERO_IMAGE_OUTFLOW_AFTER_LAYER,
)

attn_captured, _attn_removers = register_self_attn_hooks(lm)
norm_captured, _norm_removers = register_decoder_output_norm_hooks(lm)
try:
    with torch.inference_mode():
        with qwen3vl_image_outflow_ablation(
            after_layer=ZERO_IMAGE_OUTFLOW_AFTER_LAYER,
            image_key_indices=_img_key_idx,
        ):
            _ = model(**inputs)
finally:
    for _rm in _attn_removers + _norm_removers:
        _rm()

attn_captured.sort(key=lambda d: d["layer"])
norm_captured.sort(key=lambda d: d["layer"])
if len(attn_captured) != len(lm.layers):
    raise RuntimeError(f"expected {len(lm.layers)} hook captures, got {len(attn_captured)}")
if len(norm_captured) != len(lm.layers):
    raise RuntimeError(f"expected {len(lm.layers)} norm hook captures, got {len(norm_captured)}")
attn_per_layer: list[torch.Tensor] = [mean_over_heads(d["attn"]) for d in attn_captured]
print("captured layers:", len(attn_per_layer), "| attn[0] shape:", tuple(attn_per_layer[0].shape))
print("mean ||h||_2 per layer (avg over batch & tokens, post-decoder block, pre-DeepStack):")
for d in norm_captured:
    print(f"  layer {d['layer']:>3}  {d['mean_l2_norm']:.6f}")

# %% Layer 16 · mean attention (non-image queries × image keys), step 0.001

# ``W[query, key]``: each row is a non-image token attending to image placeholder keys (image → text flow reads
# along columns into rows). Uses the **same** forward as above (respects ``ZERO_IMAGE_OUTFLOW_AFTER_LAYER``).
ATTENTION_DETAIL_LAYER = 16
ATTENTION_PRINT_STEP = 0.00001

_lm_layers = len(lm.layers)
if not (0 <= ATTENTION_DETAIL_LAYER < _lm_layers):
    raise IndexError(
        f"ATTENTION_DETAIL_LAYER={ATTENTION_DETAIL_LAYER} out of range for {_lm_layers} layers (0..{_lm_layers - 1})"
    )
_q_idx, _k_idx, sub16 = submatrix_nonimage_queries_image_keys(
    attn_per_layer[ATTENTION_DETAIL_LAYER],
    _img_key_idx,
)
_sub_r = np.round(sub16 / ATTENTION_PRINT_STEP) * ATTENTION_PRINT_STEP
_decimals = max(0, min(8, int(round(-np.log10(float(ATTENTION_PRINT_STEP))))))
_rr, _cc = np.nonzero(_sub_r != 0)
_attn_sparse = [
    (_q_idx[int(r)], _k_idx[int(c)], float(_sub_r[int(r), int(c)]))
    for r, c in zip(_rr, _cc, strict=True)
]
_attn_sparse.sort(key=lambda t: (-t[2], t[0], t[1]))
print(
    f"layer {ATTENTION_DETAIL_LAYER} mean-over-heads attention · "
    f"non-image queries ({len(_q_idx)}×) × image keys (×{len(_k_idx)}) · "
    f"non-zero only after round {ATTENTION_PRINT_STEP:g} · "
    f"{len(_attn_sparse)} entr(y/ies)"
)
if not _attn_sparse:
    print(
        "  (no entries — block is all zero at this precision; "
        "with ablation on this layer that is expected, or lower ATTENTION_PRINT_STEP / "
        "set ZERO_IMAGE_OUTFLOW_AFTER_LAYER=None to see raw mass.)"
    )
else:
    for _q, _k, _v in _attn_sparse:
        _ql = token_labels[_q] if 0 <= _q < len(token_labels) else "?"
        print(f"  query {_q:4d} ({_ql!s:>14s})  key {_k:4d}  {_v:.{_decimals}f}")

# %% Short generation (final text)

MAX_NEW_TOKENS = 32

with torch.inference_mode():
    with qwen3vl_image_outflow_ablation(
        after_layer=ZERO_IMAGE_OUTFLOW_AFTER_LAYER,
        image_key_indices=_img_key_idx,
    ):
        generated_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)

new_tokens = generated_ids[0, inputs.input_ids.shape[1] :]
prediction_text = processor.batch_decode(
    [new_tokens.cpu()],
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)[0]
print("prediction:", prediction_text[:500])

# %% Save attention figure (matplotlib)

# Flow: keep edges with mass *strictly greater* than this. Matrix heatmaps: ``cmap_vmin=EDGE_MIN_MASS``,
# ``cmap_vmax=EDGE_MATRIX_VMAX`` (``Normalize``, clip=True).
EDGE_MIN_MASS = 0.05
EDGE_MATRIX_VMAX = 0.1
# Omit flows whose *source key* is in this set (extra filter; independent of hide column).
OMIT_KEY_INDICES: frozenset[int] = frozenset({0})
# Hide these token *columns* and any edge touching them (e.g. ``frozenset({0})`` for BOS/sink). Empty = show all.
HIDE_TOKEN_INDICES: frozenset[int] = frozenset({0})
EDGE_VISUAL_NORM = "per_layer"  # "per_layer" (readable deep layers) | "global"
# Flow plot only: draw arcs whose **tail** (key) is an image placeholder token (see ``model.config.image_token_id``).
ONLY_FLOW_ARROWS_FROM_IMAGE_KEYS = True
ONLY_MATRIX_IMAGE_KEY_COLUMNS = True  # heatmaps: columns only at image-token key positions
ONLY_MATRIX_QUERIES_AFTER_IMAGE = True  # heatmaps: rows only for query positions after last image token
_IMAGE_TOK_ID = int(model.config.image_token_id)
IMAGE_FLOW_KEY_INDICES = frozenset(
    i for i, tid in enumerate(_ids.tolist()) if int(tid) == _IMAGE_TOK_ID
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_fig_path = OUTPUT_DIR / "attention_vis.png"

LAYER_Y_SCALE = 1.9  # vertical separation between layer rows (data coords); higher → more spread
FIG_HEIGHT_PER_LAYER_IN = 0.6  # figure inches per layer (approx.); capped below
FIG_HEIGHT_CAP_IN = 140.0
# Width scales with sequence length: base + per-token inches (then clamped).
FIG_WIDTH_BASE_IN = 5.0
FIG_WIDTH_PER_TOKEN_IN = 0.38
FIG_WIDTH_MIN_IN = 6.5
FIG_WIDTH_CAP_IN = 130.0

if ONLY_FLOW_ARROWS_FROM_IMAGE_KEYS:
    print(
        "flow: only arrows from image tokens | image_token_id:",
        _IMAGE_TOK_ID,
        "| n_positions:",
        len(IMAGE_FLOW_KEY_INDICES),
        "| indices (sample):",
        sorted(IMAGE_FLOW_KEY_INDICES)[:24],
        ("…" if len(IMAGE_FLOW_KEY_INDICES) > 24 else ""),
    )
    _hid_img = IMAGE_FLOW_KEY_INDICES & HIDE_TOKEN_INDICES
    if _hid_img:
        print(
            "WARNING: HIDE_TOKEN_INDICES hides some image token positions;",
            "those image→query arrows cannot be drawn (no x for tail). Hidden image idx sample:",
            sorted(_hid_img)[:16],
        )

save_attention_figure(
    token_labels=token_labels,
    attn_layers=attn_per_layer,
    user_prompt=USER_TEXT,
    prediction_text=prediction_text,
    out_path=_fig_path,
    dpi=150,
    edge_min_mass=EDGE_MIN_MASS,
    omit_key_indices=OMIT_KEY_INDICES,
    edge_visual_norm=EDGE_VISUAL_NORM,
    layer_y_scale=LAYER_Y_SCALE,
    fig_height_per_layer_in=FIG_HEIGHT_PER_LAYER_IN,
    fig_height_cap_in=FIG_HEIGHT_CAP_IN,
    fig_width_base_in=FIG_WIDTH_BASE_IN,
    fig_width_per_token_in=FIG_WIDTH_PER_TOKEN_IN,
    fig_width_min_in=FIG_WIDTH_MIN_IN,
    fig_width_cap_in=FIG_WIDTH_CAP_IN,
    hide_token_indices=HIDE_TOKEN_INDICES,
    only_flow_key_indices=IMAGE_FLOW_KEY_INDICES if ONLY_FLOW_ARROWS_FROM_IMAGE_KEYS else None,
    highlight_column_token_indices=IMAGE_FLOW_KEY_INDICES if IMAGE_FLOW_KEY_INDICES else None,
)
print("wrote:", _fig_path)

if ONLY_MATRIX_IMAGE_KEY_COLUMNS and not IMAGE_FLOW_KEY_INDICES:
    print("matrices: ONLY_MATRIX_IMAGE_KEY_COLUMNS but no image tokens → full-width key columns (all |keep|).")
if ONLY_MATRIX_QUERIES_AFTER_IMAGE and not IMAGE_FLOW_KEY_INDICES:
    print("matrices: ONLY_MATRIX_QUERIES_AFTER_IMAGE but no image tokens → all query rows.")

# Per-layer matrix heatmaps: show merged vision **patch** crops under image-key columns (2×2 spatial merge → one LM token).
MATRIX_DRAW_IMAGE_KEY_PATCHES = True
_image_key_thumbnails: dict[int, Image.Image] | None = None
if (
    MATRIX_DRAW_IMAGE_KEY_PATCHES
    and IMAGE_FLOW_KEY_INDICES
    and inputs.get("image_grid_thw") is not None
):
    _image_key_thumbnails = build_qwen_vl_image_key_thumbnails(
        image,
        processor.image_processor,
        _ids.detach().cpu(),
        inputs["image_grid_thw"].detach().cpu(),
        model.config.image_token_id,
    )
    print("matrix image-key thumbnails (merge cells):", len(_image_key_thumbnails))

_mat_dir = OUTPUT_DIR / "attention_matraices"
save_attention_matrices_per_layer_dir(
    attn_per_layer,
    out_dir=_mat_dir,
    hide_token_indices=HIDE_TOKEN_INDICES,
    cmap_vmin=EDGE_MIN_MASS,
    cmap_vmax=EDGE_MATRIX_VMAX,
    only_matrix_key_indices=(
        IMAGE_FLOW_KEY_INDICES if ONLY_MATRIX_IMAGE_KEY_COLUMNS and IMAGE_FLOW_KEY_INDICES else None
    ),
    only_matrix_queries_after_max_of=(
        IMAGE_FLOW_KEY_INDICES if ONLY_MATRIX_QUERIES_AFTER_IMAGE and IMAGE_FLOW_KEY_INDICES else None
    ),
    token_labels=token_labels,
    dpi=150,
    image_key_thumbnails=_image_key_thumbnails,
)
print("per-layer matrices:", _mat_dir)
