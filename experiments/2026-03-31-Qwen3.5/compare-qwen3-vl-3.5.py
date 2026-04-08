# %% Imports & config
from __future__ import annotations

import html
import io
import sys
import urllib.request

import matplotlib

matplotlib.use("Agg")
import matplotlib.cm as cm_mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import colormaps
from matplotlib.colors import Normalize
from PIL import Image, ImageDraw

import stonesoup

# Longest edge after download (processor may smart-resize further).
MAX_EDGE = 640
USER_TEXT = "."

MODEL_QWEN3_VL = "Qwen/Qwen3-VL-8B-Instruct"
MODEL_QWEN35 = "Qwen/Qwen3.5-9B"

IMAGE_URL = (
    # "https://cdn.prod.website-files.com/63f6c457981ee7b9ec5a8e3f/649ba9eaeece8f112ea8bde3_Text%20Message.jpg"
    "https://www.ab-lab.org/uploads/1/3/2/1/132186461/published/textmsg-lines.png?1627263476"
)

# Shared Blues range: percentiles over **all** merge cells from **both** models (top-1 cos vs embedding rows).
COS_PERCENTILE_LOW = 2.0
COS_PERCENTILE_HIGH = 98.0

print(
    f"Config: MAX_EDGE={MAX_EDGE}  models: {MODEL_QWEN3_VL} vs {MODEL_QWEN35}\n"
    f"image: {IMAGE_URL}\n",
    flush=True,
)

# %% Fetch & resize image


def load_image_rgb(url: str) -> Image.Image:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; Stonesoup/1.0)"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        raw = resp.read()
    im = Image.open(io.BytesIO(raw)).convert("RGB")
    return im


def resize_long_edge(pil: Image.Image, max_edge: int) -> Image.Image:
    w, h = pil.size
    m = max(w, h)
    if m <= max_edge:
        return pil
    s = max_edge / m
    nw, nh = int(round(w * s)), int(round(h * s))
    return pil.resize((nw, nh), Image.Resampling.LANCZOS)


PIL_IMAGE = resize_long_edge(load_image_rgb(IMAGE_URL), MAX_EDGE)
_W, _H = PIL_IMAGE.size
print(f"PIL_IMAGE size: {_W}×{_H}", flush=True)

# %% Grid + vision helpers


def resized_box_to_original(
    box: tuple[int, int, int, int],
    *,
    resized_wh: tuple[int, int],
    original_wh: tuple[int, int],
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    rw, rh = resized_wh
    ow, oh = original_wh
    sx, sy = ow / rw, oh / rh
    return (
        int(round(x0 * sx)),
        int(round(y0 * sy)),
        int(round(x1 * sx)),
        int(round(y1 * sy)),
    )


def llm_merge_index_to_resized_box(
    merge_index: int,
    grid_thw_1d: torch.Tensor,
    *,
    patch_size: int,
    merge_size: int,
) -> tuple[int, int, int, int]:
    _t, gh, gw = (int(x) for x in grid_thw_1d.tolist())
    llm_h, llm_w = gh // merge_size, gw // merge_size
    llm_hw = llm_h * llm_w
    rem = int(merge_index) % llm_hw
    h_llm = rem // llm_w
    w_llm = rem % llm_w
    ps = int(patch_size)
    ms = int(merge_size)
    x0 = w_llm * ms * ps
    y0 = h_llm * ms * ps
    x1 = (w_llm + 1) * ms * ps
    y1 = (h_llm + 1) * ms * ps
    return x0, y0, x1, y1


def _get_image_features(model, pixel_values, image_grid_thw):
    fn = getattr(model, "get_image_features", None)
    if callable(fn):
        return fn(pixel_values, image_grid_thw, return_dict=True)
    return model.model.get_image_features(pixel_values, image_grid_thw, return_dict=True)


def build_vision_inputs(processor, pil: Image.Image, *, qwen35: bool, device: torch.device):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil},
                {"type": "text", "text": USER_TEXT},
            ],
        }
    ]
    kw: dict = dict(
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    if qwen35:
        kw["enable_thinking"] = False
    return processor.apply_chat_template(messages, **kw).to(device)


def short_token_label(processor, tid: int, *, max_chars: int = 12) -> str:
    s = processor.decode(
        [tid], skip_special_tokens=False, clean_up_tokenization_spaces=False,
    )
    s = s.replace("\n", " ").replace("\r", " ").strip()
    if not s:
        return f"·{tid}"
    if len(s) > max_chars:
        return s[: max_chars - 1] + "…"
    return s


def save_base_image_url_for_merge_html(pil: Image.Image, *, filename: str = "merge-html-base.png") -> str:
    """Write PNG under :func:`stonesoup.plot_dir` and return UI path ``/outputs/…`` (repo-relative)."""
    out_path = stonesoup.plot_dir() / filename
    pil.save(out_path, format="PNG", optimize=True)
    rel = out_path.relative_to(stonesoup.repo_root()).as_posix()
    cb = int(out_path.stat().st_mtime_ns)
    return f"/{rel}?cb={cb}"


def merge_patch_embeddings_and_vocab_matrix(
    model,
    processor,
    pil: Image.Image,
    *,
    qwen35: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Return L2-normalized vision pooler rows ``img_n`` (N×D), vocab rows ``emb_n`` (V×D), and grid metadata.

    For Qwen3-VL, **N equals the number of ``image_token_id`` slots** in the tokenized prompt (the expanded
    ``<|image_pad|>`` placeholders): one LM image token per spatial merge cell (``llm_h × llm_w``).
    """
    device = next(model.parameters()).device
    inputs = build_vision_inputs(processor, pil, qwen35=qwen35, device=device)
    g = inputs["image_grid_thw"][0].detach().cpu()
    t, gh, gw = (int(x) for x in g.tolist())
    if t != 1:
        raise NotImplementedError(f"expect single image (T=1), got T={t}")

    ms = int(processor.image_processor.merge_size)
    ps = int(processor.image_processor.patch_size)
    llm_h, llm_w = gh // ms, gw // ms
    n_merge = llm_h * llm_w
    res_wh = (gw * ps, gh * ps)

    with torch.inference_mode():
        out = _get_image_features(
            model, inputs["pixel_values"], inputs["image_grid_thw"],
        )
        pool = out.pooler_output
        if isinstance(pool, (list, tuple)):
            pool_cat = torch.cat(pool, dim=0)
        else:
            pool_cat = pool
        if int(pool_cat.shape[0]) != n_merge:
            raise RuntimeError(
                f"pooler rows {pool_cat.shape[0]} != merge grid {llm_h}×{llm_w}={n_merge}"
            )

        emb_w = model.get_input_embeddings().weight.float().to(device)
        emb_n = F.normalize(emb_w, dim=-1, eps=1e-12)
        img_n = F.normalize(pool_cat.float(), dim=-1, eps=1e-12)
        if img_n.shape[-1] != emb_n.shape[-1]:
            raise RuntimeError(
                f"dim mismatch: image {img_n.shape[-1]} vs embed {emb_n.shape[-1]}"
            )

    meta = {
        "label": MODEL_QWEN35 if qwen35 else MODEL_QWEN3_VL,
        "pil": pil,
        "g": g,
        "merge_size": ms,
        "patch_size": ps,
        "llm_h": llm_h,
        "llm_w": llm_w,
        "n_merge": n_merge,
        "res_wh": res_wh,
        "processor": processor,
    }
    return img_n, emb_n, meta


def analyze_merge_tokens(
    model,
    processor,
    pil: Image.Image,
    *,
    qwen35: bool,
) -> dict:
    img_n, emb_n, meta = merge_patch_embeddings_and_vocab_matrix(
        model, processor, pil, qwen35=qwen35,
    )
    with torch.inference_mode():
        cos_all = img_n @ emb_n.T
        topv, topi = torch.topk(cos_all, k=1, dim=-1)

    cos1 = topv.squeeze(-1).detach().float().cpu().numpy()
    tok1 = topi.squeeze(-1).detach().long().cpu()

    return {**meta, "cos1": cos1, "tok1": tok1}

# %% Load Qwen3-VL & analyze

model_vl, proc_vl = stonesoup.load_model(MODEL_QWEN3_VL)
model_vl.eval()
print("Loaded:", MODEL_QWEN3_VL, next(model_vl.parameters()).device, flush=True)

RESULT_VL = analyze_merge_tokens(model_vl, proc_vl, PIL_IMAGE, qwen35=False)
print(
    f"Qwen3-VL merge grid {RESULT_VL['llm_h']}×{RESULT_VL['llm_w']}  "
    f"cos1 min/max {float(np.min(RESULT_VL['cos1'])):.4f} / {float(np.max(RESULT_VL['cos1'])):.4f}",
    flush=True,
)

# %% Load Qwen3.5-9B & analyze

model_35, proc_35 = stonesoup.load_model(MODEL_QWEN35)
model_35.eval()
print("Loaded:", MODEL_QWEN35, next(model_35.parameters()).device, flush=True)

RESULT_35 = analyze_merge_tokens(model_35, proc_35, PIL_IMAGE, qwen35=True)
print(
    f"Qwen3.5 merge grid {RESULT_35['llm_h']}×{RESULT_35['llm_w']}  "
    f"cos1 min/max {float(np.min(RESULT_35['cos1'])):.4f} / {float(np.max(RESULT_35['cos1'])):.4f}",
    flush=True,
)

# %% Shared norm + separate figures (Blues) + UI

_c_all = np.concatenate([RESULT_VL["cos1"], RESULT_35["cos1"]], axis=0)
_vmin = float(np.percentile(_c_all, COS_PERCENTILE_LOW))
_vmax = float(np.percentile(_c_all, COS_PERCENTILE_HIGH))
if _vmax <= _vmin:
    _vmax = _vmin + 1e-6

_shared_norm = Normalize(vmin=_vmin, vmax=_vmax, clip=True)
_cmap = colormaps["Blues"]
_fill_alpha = 120
_outline_alpha = 100


def _build_composite_rgba(result: dict) -> Image.Image:
    pil = result["pil"]
    W, H = pil.size
    g = result["g"]
    ms, ps = result["merge_size"], result["patch_size"]
    res_wh = result["res_wh"]
    n = result["n_merge"]
    cos1 = result["cos1"]
    proc = result["processor"]

    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    pen = max(1, min(W, H) // 400)

    for k in range(n):
        xr0, yr0, xr1, yr1 = llm_merge_index_to_resized_box(
            k, g, patch_size=ps, merge_size=ms,
        )
        xo0, yo0, xo1, yo1 = resized_box_to_original(
            (xr0, yr0, xr1, yr1),
            resized_wh=res_wh,
            original_wh=(W, H),
        )
        tc = float(_shared_norm(float(cos1[k])))
        rgba = _cmap(tc)
        fill = (
            int(rgba[0] * 255),
            int(rgba[1] * 255),
            int(rgba[2] * 255),
            _fill_alpha,
        )
        draw.rectangle(
            (xo0, yo0, xo1, yo1),
            fill=fill,
            outline=(255, 255, 255, _outline_alpha),
            width=pen,
        )

    return Image.alpha_composite(pil.convert("RGBA"), overlay).convert("RGB")


_comp_vl_rgb = _build_composite_rgba(RESULT_VL)
_comp_35_rgb = _build_composite_rgba(RESULT_35)

_ar_vl = np.asarray(_comp_vl_rgb)
_ar_35 = np.asarray(_comp_35_rgb)


def _show_merge_token_figure(ar: np.ndarray, res: dict) -> None:
    W, H = res["pil"].size
    ah, aw = ar.shape[0], ar.shape[1]
    fig_w = 6.0
    fig_h = max(5.0, fig_w * ah / max(1, aw))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)
    ax.imshow(ar, extent=(0, W, H, 0), origin="upper", aspect="equal")
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    ax.set_title(
        f"{res['label']}\nmerge {res['llm_h']}×{res['llm_w']} · nearest vocab token (cos vs embed)",
        fontsize=11,
    )

    proc = res["processor"]
    g = res["g"]
    ms, ps = res["merge_size"], res["patch_size"]
    res_wh = res["res_wh"]
    for k in range(res["n_merge"]):
        xr0, yr0, xr1, yr1 = llm_merge_index_to_resized_box(
            k, g, patch_size=ps, merge_size=ms,
        )
        xo0, yo0, xo1, yo1 = resized_box_to_original(
            (xr0, yr0, xr1, yr1),
            resized_wh=res_wh,
            original_wh=(W, H),
        )
        cw = max(1, xo1 - xo0)
        ch = max(1, yo1 - yo0)
        tid = int(res["tok1"][k].item())
        cv = float(res["cos1"][k])
        label = f"{short_token_label(proc, tid)}\n{cv:.4f}"
        cx = xo0 + 0.5 * cw
        cy = yo0 + 0.5 * ch
        fs = max(7, min(18, int(min(cw, ch) / 6)))
        lw = max(2.0, fs * 0.35)
        ax.text(
            cx,
            cy,
            label,
            ha="center",
            va="center",
            linespacing=0.95,
            fontsize=fs,
            color="white",
            path_effects=[
                pe.Stroke(linewidth=lw, foreground="black"),
                pe.Normal(),
            ],
        )

    sm = cm_mpl.ScalarMappable(cmap=_cmap, norm=_shared_norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(
        f"top-1 cos (embed) · shared [{COS_PERCENTILE_LOW:.0f}–{COS_PERCENTILE_HIGH:.0f}%] "
        f"[{_vmin:.4f}, {_vmax:.4f}]",
    )
    fig.suptitle(
        "Image tokens → nearest LM embedding row (cosine). Blues scale matches the other figure.",
        fontsize=11,
        y=1.02,
    )

    stonesoup.show(fig, dpi=140)
    plt.close(fig)

# %% Result 3 VL
_show_merge_token_figure(_ar_vl, RESULT_VL)
# %% Result 35
_show_merge_token_figure(_ar_35, RESULT_35)

# %% HTML summary (optional)

print(stonesoup.STONESOUP_RENDER_HTML, end="")
print(
    "<div style='font-family:system-ui,sans-serif;font-size:14px;line-height:1.45'>"
    "<p><strong>Compare Qwen3-VL vs Qwen3.5</strong> — nearest vocabulary embedding per merge token; "
    f"shared Blues vmin/vmax from pooled percentiles: <code>{_vmin:.4f}</code> … <code>{_vmax:.4f}</code>.</p>"
    "<ul style='margin:0.35em 0;padding-left:1.2em'>"
    f"<li>{html.escape(MODEL_QWEN3_VL)}: grid {RESULT_VL['llm_h']}×{RESULT_VL['llm_w']}</li>"
    f"<li>{html.escape(MODEL_QWEN35)}: grid {RESULT_35['llm_h']}×{RESULT_35['llm_w']}</li>"
    "</ul>"
    "<p style='color:#5f6368;font-size:13px'>Matplotlib PNGs and HTML grid row above use the same scale. "
    "PNG figures: <code>stonesoup.show()</code>; standalone merge view: "
    "<code>outputs/…/merge-grids.html</code>.</p>"
    "</div>",
    flush=True,
)

# %% Histogram: one image token vs all image tokens & vs vocab (Qwen3-VL or Qwen3.5) # stonesoup:cell-input

# ``True`` → Qwen3.5-9B + ``proc_35`` + ``qwen35=True``; ``False`` → Qwen3-VL + ``proc_vl`` + ``qwen35=False``.
# Model, processor, and flag must stay matched or ``image_token_id`` counts won’t match pooler rows.
HISTOGRAM_USE_QWEN35 = False

# ``CELL_INPUT`` (header text box): blank → center image token; else integer index 0 … n_merge-1.
_ci = globals().get("CELL_INPUT", "")
_ci_s = _ci.strip() if isinstance(_ci, str) else ""
MERGE_PICK_INDEX: int | None = None if not _ci_s else int(_ci_s, 10)

_hist_m = model_35 if HISTOGRAM_USE_QWEN35 else model_vl
_hist_p = proc_35 if HISTOGRAM_USE_QWEN35 else proc_vl
_hist_q35 = bool(HISTOGRAM_USE_QWEN35)

_img_n, _emb_n, _meta_h = merge_patch_embeddings_and_vocab_matrix(
    _hist_m, _hist_p, PIL_IMAGE, qwen35=_hist_q35,
)
_hist_label = str(_meta_h["label"])
_n_m = int(_img_n.shape[0])
_hist_dev = next(_hist_m.parameters()).device
_inputs_align = build_vision_inputs(_hist_p, PIL_IMAGE, qwen35=_hist_q35, device=_hist_dev)
_n_image_id = int(
    (_inputs_align["input_ids"] == int(_hist_m.config.image_token_id)).sum().item(),
)
_img_pad_tok = getattr(_hist_p, "image_token", None) or "<|image_pad|>"
print(
    f"Terminology [{_hist_label}]: {_n_m} vision pooler rows == {_n_image_id}× image_token_id in input_ids "
    f"(expanded {_img_pad_tok!s} placeholder slots; same count as merge cells).",
    file=sys.stderr,
    flush=True,
)
assert _n_image_id == _n_m, (_n_image_id, _n_m, "pooler rows must match <|image_pad|> count")
_pick = (
    (_n_m // 2)
    if MERGE_PICK_INDEX is None
    else int(np.clip(MERGE_PICK_INDEX, 0, _n_m - 1))
)
_v = _img_n[_pick]
with torch.inference_mode():
    _cos_patches = (_img_n @ _v).float().cpu().numpy()
    _cos_vocab = (_emb_n @ _v).float().cpu().numpy()

_lw = int(_meta_h["llm_w"])
_row, _col = _pick // _lw, _pick % _lw
_j_top = int(np.argmax(_cos_vocab))
_near_tok = short_token_label(_hist_p, _j_top, max_chars=20)
_near_cos = float(_cos_vocab[_j_top])
_c2t_im = getattr(_hist_p.tokenizer, "convert_ids_to_tokens", None)
_piece_nn = _c2t_im([_j_top]) if callable(_c2t_im) else []
print(
    f"[image hist {_hist_label}] patch #{_pick} (r{_row},c{_col}) · "
    f"nearest vocab id={_j_top} decode={_near_tok!r} piece={_piece_nn!r} cos={_near_cos:.4f}",
    file=sys.stderr,
    flush=True,
)

_fig_h, _axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
_axes[0].hist(_cos_patches, bins=48, color="#4a81bf", edgecolor="white", linewidth=0.35)
_axes[0].set_title(
    f"vs all image tokens (n={len(_cos_patches)})",
    fontsize=11,
)
_axes[0].set_xlabel("cosine similarity")
_axes[0].set_ylabel("count")

_bins = min(96, max(24, int(np.sqrt(len(_cos_vocab)))))
_axes[1].hist(_cos_vocab, bins=_bins, color="#3d8f6e", edgecolor="white", linewidth=0.35)
_axes[1].set_title(
    f"vs all vocab embedding rows (n={len(_cos_vocab)})",
    fontsize=11,
)
_axes[1].set_xlabel("cosine similarity")
_axes[1].set_ylabel("count")
_axes[1].margins(x=0.05)
_ylim_r = _axes[1].get_ylim()
_axes[1].axvline(
    _near_cos,
    color="#b91c1c",
    linestyle="--",
    linewidth=1.6,
    zorder=5,
    alpha=0.92,
)
_nn_label = f"Nearest: {_near_tok}\n{_near_cos:.4f}"
_axes[1].annotate(
    _nn_label,
    xy=(_near_cos, _ylim_r[1] * 0.1),
    xytext=(0, 4),
    textcoords="offset points",
    fontsize=7,
    ha="center",
    va="bottom",
    color="#450a0a",
    bbox=dict(
        boxstyle="round,pad=0.25",
        facecolor="white",
        edgecolor="#b91c1c",
        alpha=0.93,
        linewidth=0.85,
    ),
    zorder=6,
)

_fig_h.suptitle(
    f"{_hist_label}\nimage token #{_pick} (row {_row}, col {_col})",
    fontsize=11,
)
stonesoup.show(_fig_h, dpi=120)
plt.close(_fig_h)

# %% Histogram: one vocab token vs all vocab rows & vs image tokens # stonesoup:cell-input

# Same checkpoint switch as the image-token histogram cell above.
HISTOGRAM_USE_QWEN35 = True

# ``CELL_INPUT``: blank → first sub-token of ``USER_TEXT``; else must be a **quoted** literal:
# matching ``'…'``, ``"…"``, `` `…` `` (ASCII or Unicode “…” / ‘…’). Inner string is exact, e.g. `` ` school` ``.
_VCI_QUOTE = frozenset({
    ("'", "'"),
    ('"', '"'),
    ("`", "`"),
    ("\u2018", "\u2019"),
    ("\u201c", "\u201d"),
})
_vci_t = globals().get("CELL_INPUT", "")
_vci_t = _vci_t.strip() if isinstance(_vci_t, str) else ""
if not _vci_t:
    _vci_lit = None
elif len(_vci_t) >= 2 and (_vci_t[0], _vci_t[-1]) in _VCI_QUOTE:
    _vci_lit = _vci_t[1:-1]
else:
    raise ValueError(
        "CELL_INPUT: leave blank or use a quoted literal (e.g. ` school` or 'hello'), not raw numbers.",
    )

_hist2_m = model_35 if HISTOGRAM_USE_QWEN35 else model_vl
_hist2_p = proc_35 if HISTOGRAM_USE_QWEN35 else proc_vl
_hist2_q35 = bool(HISTOGRAM_USE_QWEN35)

_img_nv, _emb_nv, _meta_v = merge_patch_embeddings_and_vocab_matrix(
    _hist2_m, _hist2_p, PIL_IMAGE, qwen35=_hist2_q35,
)
_hist2_label = str(_meta_v["label"])
_Vv = int(_emb_nv.shape[0])
_tokv = _hist2_p.tokenizer

if _vci_lit is None:
    _txt_ids = _tokv.encode(USER_TEXT, add_special_tokens=False)
    _enc_shown = USER_TEXT
    print(
        f"[vocab hist] CELL_INPUT blank → USER_TEXT sub-token from {USER_TEXT!r}",
        file=sys.stderr,
        flush=True,
    )
else:
    # Qwen2-style ByteLevel uses ``add_prefix_space=False`` on the pretokenizer; a **leading** ASCII
    # space at the *start* of the string is then not treated like word-internal ``Ġ`` space unless we
    # request ``add_prefix_space=True`` for this encode (same pitfall as GPT-2 at line starts).
    _enc_kw: dict = dict(add_special_tokens=False)
    if _vci_lit[:1].isspace():
        _enc_kw["add_prefix_space"] = True
    try:
        _txt_ids = _tokv.encode(_vci_lit, **_enc_kw)
    except TypeError:
        del _enc_kw["add_prefix_space"]
        _txt_ids = _tokv.encode(_vci_lit, **_enc_kw)
    _enc_shown = _vci_lit
    if not _txt_ids:
        raise ValueError(f"CELL_INPUT literal {_vci_lit!r} tokenizes to empty")
    if len(_txt_ids) > 1:
        print(
            f"[vocab hist] using first of {len(_txt_ids)} sub-tokens for {_vci_lit!r}",
            file=sys.stderr,
            flush=True,
        )
_vid = int(np.clip(int(_txt_ids[0]), 0, _Vv - 1))

_vrow = _emb_nv[_vid]
with torch.inference_mode():
    _cos_v_v = (_emb_nv @ _vrow).float().cpu().numpy()
    _cos_v_img = (_img_nv @ _vrow).float().cpu().numpy()

_pick_vocab_lbl = short_token_label(_hist2_p, _vid, max_chars=28)
_im_top = int(np.argmax(_cos_v_img))
_best_img_cos = float(_cos_v_img[_im_top])
_lwv = int(_meta_v["llm_w"])
_im_r, _im_c = _im_top // _lwv, _im_top % _lwv

_c2t = getattr(_tokv, "convert_ids_to_tokens", None)
_piece0 = _c2t([_vid]) if callable(_c2t) else []
_ap_note = (
    " add_prefix_space=True"
    if _vci_lit is not None and _vci_lit[:1].isspace()
    else ""
)
print(
    f"[vocab hist {_hist2_label}] id={_vid} decode={_pick_vocab_lbl!r} "
    f"lit={_enc_shown!r} piece={_piece0!r}{_ap_note} · "
    f"max cos image: {_best_img_cos:.4f} @ patch {_im_top} (r{_im_r},c{_im_c})",
    file=sys.stderr,
    flush=True,
)

_fig_v, _axv = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
_bins_l = min(96, max(24, int(np.sqrt(len(_cos_v_v)))))
_axv[0].hist(_cos_v_v, bins=_bins_l, color="#3d8f6e", edgecolor="white", linewidth=0.35)
_axv[0].set_title(f"vs all vocab rows (n={len(_cos_v_v)})", fontsize=11)
_axv[0].set_xlabel("cosine similarity")
_axv[0].set_ylabel("count")
_axv[0].margins(x=0.05)
_y0 = _axv[0].get_ylim()
_self_c = float(_cos_v_v[_vid])
_axv[0].axvline(
    _self_c,
    color="#b91c1c",
    linestyle="--",
    linewidth=1.4,
    zorder=5,
    alpha=0.9,
)
_axv[0].annotate(
    f"self: {_pick_vocab_lbl}\n{_self_c:.4f}",
    xy=(_self_c, _y0[1] * 0.12),
    xytext=(0, 4),
    textcoords="offset points",
    fontsize=7,
    ha="center",
    va="bottom",
    color="#450a0a",
    bbox=dict(
        boxstyle="round,pad=0.25",
        facecolor="white",
        edgecolor="#b91c1c",
        alpha=0.93,
        linewidth=0.85,
    ),
    zorder=6,
)

_axv[1].hist(_cos_v_img, bins=48, color="#4a81bf", edgecolor="white", linewidth=0.35)
_axv[1].set_title(f"vs all image tokens (n={len(_cos_v_img)})", fontsize=11)
_axv[1].set_xlabel("cosine similarity")
_axv[1].set_ylabel("count")
_axv[1].margins(x=0.05)
_y1 = _axv[1].get_ylim()
_axv[1].axvline(
    _best_img_cos,
    color="#b91c1c",
    linestyle="--",
    linewidth=1.6,
    zorder=5,
    alpha=0.92,
)
_axv[1].annotate(
    f"best patch: #{_im_top}\n{_best_img_cos:.4f}",
    xy=(_best_img_cos, _y1[1] * 0.12),
    xytext=(0, 4),
    textcoords="offset points",
    fontsize=7,
    ha="center",
    va="bottom",
    color="#450a0a",
    bbox=dict(
        boxstyle="round,pad=0.25",
        facecolor="white",
        edgecolor="#b91c1c",
        alpha=0.93,
        linewidth=0.85,
    ),
    zorder=6,
)

_fig_v.suptitle(
    f"{_hist2_label}\nvocab id {_vid} ({_pick_vocab_lbl}) · "
    f"strongest image patch {_im_top} (row {_im_r}, col {_im_c})",
    fontsize=11,
)
stonesoup.show(_fig_v, dpi=120)
plt.close(_fig_v)
