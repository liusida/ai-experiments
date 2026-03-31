# %% Imports & paths

"""Multimodal **Qwen3.5** and/or **Qwen3-VL** ViT **patch embeddings** on **POPE** (HF ``lmms-lab/POPE``).

**Config:** ``POPE_MODEL_FAMILY`` is ``\"qwen3_5\"`` or ``\"qwen3_vl\"``; ``MODEL_ID_BY_FAMILY`` maps each to a Hub id.
``POPE_LOAD_BOTH_MODELS`` loads both checkpoints (2× memory); the load cell defines ``pope_set_active_model(fam)`` so
``model`` / ``processor`` alias the chosen family. Single-load keeps only one family in memory.

``pope_apply_chat_template_kwargs`` syncs ``POPE_CHAT_TEMPLATE_KWARGS`` (3.5: ``enable_thinking=False``; VL: empty).
**DeepStack** in the one-merge cell is **Qwen3-VL** only.

For each image, runs the vision tower and collects ``last_hidden_state`` rows (one vector per patch).
Builds:

- ``POPE_VIT_ALL_PATCHES``: ``(total_patches, hidden_dim)`` — every patch from every image stacked.
- ``POPE_VIT_PATCH_IMAGE_ID``: ``(total_patches,)`` — image index ``0 … N-1`` for each patch row.
- ``POPE_VIT_MEAN_PER_IMAGE``: ``(n_image, hidden_dim)`` — mean patch vector per image (matches a
  dense ``(n_image, n_vec_dim)`` layout when you want **one** vector per image).

Patch counts can differ by resolution; padded batch form is optional (see extract cell).

RGB files are written under ``data/images/pope/`` (repo root); **Save POPE images** runs after loading the subset.
**Preview POPE images (HTML)** lists ``image_id`` (= loop index ``0 … n-1``, same as tensor rows) with thumbnails via Stonesoup
``http://127.0.0.1:8765/data/image/pope/…``.
**Histogram: ViT patch L2 norms**, **all embedding values**, **per-image** histograms (**image_id** ``0 … 9``), **per-image patch-norm overlays** (fine grid, **Blues**), and **merged tokens vs ``embed_tokens``** (**image_id** ``0 … 9``: top-1 cos **Blues** + label per PNG; HTML table: top-3 per image) under ``data/images/pope_vit/`` with HTML previews where noted.

**Subset:** Scan split from ``POPE_START`` and keep the **first** ``POPE_LIMIT`` rows whose ``image_source`` is new
— POPE repeats the same COCO image across many yes/no questions, so this is **unique images**, not unique questions.

**Install** (repo root)::

  uv pip install datasets accelerate transformers pillow tqdm matplotlib

**Stonesoup:** Watch this file; run cells **in order** after **Reset** when needed. Adjust **Config** before loading data/model.

**Terminal:** ``uv run python experiments/2026-03-29-Qwen3.5/qwen3.5-embedding-distribution.py``

Large HF downloads and unified memory: see ``AGENTS.md``.
"""

from __future__ import annotations

import html
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Literal, Union

from PIL import Image, ImageDraw
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, Qwen3_5ForConditionalGeneration

try:
    from stonesoup import STONESOUP_RENDER_HTML
except ImportError:
    STONESOUP_RENDER_HTML = "# stonesoup:render=html\n"

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
POPE_IMAGE_DIR = REPO_ROOT / "data" / "images" / "pope"
POPE_VIT_PLOT_DIR = REPO_ROOT / "data" / "images" / "pope_vit"
EXP_DIR = Path(__file__).resolve().parent
OUT_DIR = EXP_DIR / "output" / "pope_vit"
# 32×32 RGB template: one spatial merge (``patch_size=16``, ``merge_size=2``) when paired with
# ``qwen_vl_images_kwargs_one_llm_image_token`` — see **One merge token** cell.
ONE_MERGE_TOKEN_CANVAS_PNG = EXP_DIR / "one-merge-token-32x32.png"
DATASET_ID = "lmms-lab/POPE"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PopeModelFamily = Literal["qwen3_5", "qwen3_vl"]
PopeMultimodalModel = Union[Qwen3_5ForConditionalGeneration, Qwen3VLForConditionalGeneration]

# Overwritten in **Config** / ``pope_set_active_model``; defaults avoid NameError if helpers are inspected early.
MODEL_ID: str = ""
POPE_MODEL_FAMILY: PopeModelFamily = "qwen3_5"
# Merged into ``build_pope_vision_inputs`` → ``apply_chat_template``. Updated by ``pope_apply_chat_template_kwargs``.
POPE_CHAT_TEMPLATE_KWARGS: dict[str, object] = {}

_POPE_MM_LOADED: dict[PopeModelFamily, tuple[PopeMultimodalModel, object]] = {}
_POPE_MM_IDS: dict[PopeModelFamily, str] = {}


def pope_apply_chat_template_kwargs(model_family: PopeModelFamily) -> None:
    """Qwen3.5 chat template uses ``enable_thinking=False``; Qwen3-VL omits it."""
    POPE_CHAT_TEMPLATE_KWARGS.clear()
    if model_family == "qwen3_5":
        POPE_CHAT_TEMPLATE_KWARGS["enable_thinking"] = False
    elif model_family != "qwen3_vl":
        raise ValueError(f"unknown model family {model_family!r} (expected 'qwen3_5' or 'qwen3_vl')")


def pope_register_multimodal_models(
    *,
    qwen3_5: tuple[Qwen3_5ForConditionalGeneration, object] | None,
    qwen3_vl: tuple[Qwen3VLForConditionalGeneration, object] | None,
    model_id_by_family: dict[PopeModelFamily, str],
) -> None:
    """Store handles from **Load multimodal**; missing entries stay ``None`` (single-load mode)."""
    global _POPE_MM_LOADED, _POPE_MM_IDS
    _POPE_MM_LOADED = {}
    _POPE_MM_IDS = dict(model_id_by_family)
    if qwen3_5 is not None:
        _POPE_MM_LOADED["qwen3_5"] = qwen3_5  # type: ignore[assignment]
    if qwen3_vl is not None:
        _POPE_MM_LOADED["qwen3_vl"] = qwen3_vl  # type: ignore[assignment]


def pope_set_active_model(model_family: PopeModelFamily) -> None:
    """Point global ``model`` / ``processor`` (and ``MODEL_ID``) at a loaded family; refresh chat-template kwargs."""
    global model, processor, MODEL_ID, POPE_MODEL_FAMILY
    pair = _POPE_MM_LOADED.get(model_family)
    if pair is None:
        loaded = ", ".join(repr(k) for k in sorted(_POPE_MM_LOADED.keys())) or "(none)"
        raise RuntimeError(
            f"no {model_family!r} in memory (currently loaded: {loaded}). "
            "Default **Config** uses POPE_LOAD_BOTH_MODELS=False and POPE_MODEL_FAMILY='qwen3_5', so **Load multimodal** "
            "only registers Qwen3.5. To call pope_set_active_model('qwen3_vl') you need both in RAM: set "
            "POPE_LOAD_BOTH_MODELS = True in **Config**, then re-run **Config** and **Load multimodal**. "
            "Alternatively, set POPE_MODEL_FAMILY = 'qwen3_vl' (single load, no 3.5)."
        )
    model, processor = pair
    POPE_MODEL_FAMILY = model_family
    MODEL_ID = _POPE_MM_IDS[model_family]
    pope_apply_chat_template_kwargs(model_family)
    print(f"Active POPE model: family={model_family}  MODEL_ID={MODEL_ID}", flush=True)


print("REPO_ROOT:", REPO_ROOT)
print("EXP_DIR:", EXP_DIR)
print("DEVICE:", DEVICE)


def configure_matplotlib_fonts() -> None:
    """Set ``font.family`` to concrete faces Matplotlib resolves so FT2Font gets a real multi-file fallback chain.

    ``rcParams['font.family']`` defaults to ``['sans-serif']``, which ``findfont`` collapses to **one** file — mixed
    tokenizer strings then miss Han / Arabic / Hebrew glyphs. Names like ``Noto Sans CJK SC`` are often absent from
    Matplotlib's cache; the same TTC registers as **Noto Sans CJK JP** but still contains CJK ideographs.
    """
    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt

    plt.rcParams["axes.unicode_minus"] = False
    candidates = (
        "Noto Sans",
        "Noto Naskh Arabic",
        "Noto Sans Hebrew",
        "Noto Serif Hebrew",
        "Noto Sans Thai",
        "Noto Sans Armenian",
        "Noto Serif Tibetan",
        "Noto Sans Yi",
        "Noto Sans Symbols",
        "Noto Sans Math",
        "Noto Sans CJK JP",
        "Noto Serif CJK JP",
        "Droid Sans Fallback",
        "DejaVu Sans",
    )
    resolved: list[str] = []
    for name in candidates:
        try:
            fm.findfont(fm.FontProperties(family=name), fallback_to_default=False)
        except ValueError:
            continue
        if name not in resolved:
            resolved.append(name)
    if not resolved:
        return
    plt.rcParams["font.family"] = resolved
    cur = plt.rcParams.get("font.sans-serif")
    tail = [cur] if isinstance(cur, str) else list(cur)
    seen = set(resolved)
    plt.rcParams["font.sans-serif"] = resolved + [f for f in tail if f not in seen]


def _filter_mpl_glyph_userwarnings() -> None:
    import warnings

    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message=r"Glyph .* missing from font",
    )
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message=r"Matplotlib currently does not support .* natively",
    )


@contextmanager
def mpl_suppress_glyph_userwarnings():
    """Silence Matplotlib missing-glyph / script UserWarnings during any draw (text, colorbar, ``tight_layout``, etc.)."""
    import warnings

    with warnings.catch_warnings():
        _filter_mpl_glyph_userwarnings()
        yield


def savefig_mpl(fig, path, *, tight_layout: bool = False, **kwargs) -> None:
    """``fig.savefig`` with the same filters; optional ``tight_layout`` inside the filtered block."""
    import warnings

    with warnings.catch_warnings():
        _filter_mpl_glyph_userwarnings()
        if tight_layout:
            fig.tight_layout()
        fig.savefig(path, **kwargs)


def _sanitize_mpl_overlay_text(s: str) -> str:
    """Strip C0 controls and emoji / pictographs Matplotlib often has no font for (decoded token pieces)."""
    out: list[str] = []
    for ch in s:
        o = ord(ch)
        if o < 32 and ch not in "\n\t":
            continue
        if 0x1F000 <= o <= 0x1FFFF:
            continue
        if 0x2600 <= o <= 0x27BF:
            continue
        out.append(ch)
    t = "".join(out).strip()
    return t if t else "·"


def pope_row_to_pil(row: dict) -> Image.Image:
    raw = row["image"]
    if isinstance(raw, Image.Image):
        return raw.convert("RGB")
    raise TypeError(f"Unsupported image field type: {type(raw)}")


def pope_sanitize_stem(s: object, *, max_len: int = 72) -> str:
    t = re.sub(r"[^\w.\-]+", "_", str(s)).strip("._-")
    t = t[:max_len] if t else "img"
    return t.lower()


def pope_row_image_key(row: dict) -> str:
    """Stable id for deduplication: same underlying image across POPE questions."""
    src = row.get("image_source")
    if src is not None and str(src).strip():
        return str(src).strip()
    rid = row.get("id")
    if rid is not None and str(rid).strip():
        return f"id:{rid}"
    return f"qid:{row.get('question_id')!s}"


def build_pope_vision_inputs(
    processor,
    pil: Image.Image,
    question: str,
    device: torch.device,
    *,
    images_kwargs: dict | None = None,
) -> dict[str, torch.Tensor]:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil},
                {"type": "text", "text": question},
            ],
        }
    ]
    tmpl_kw: dict = dict(
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    if images_kwargs is not None:
        tmpl_kw["images_kwargs"] = images_kwargs
    if POPE_CHAT_TEMPLATE_KWARGS:
        tmpl_kw.update(POPE_CHAT_TEMPLATE_KWARGS)
    inputs = processor.apply_chat_template(messages, **tmpl_kw)
    return inputs.to(device)


def qwen_vl_images_kwargs_one_llm_image_token(processor) -> dict[str, int]:
    """Kwargs for Qwen2VL/Qwen3-VL ``smart_resize`` so a ``patch_size×merge_size`` square stays one LM image token.

    Checkpoints often set ``min_pixels`` to ~65k (e.g. 256×256), which upsamples a 32×32 crop to 64 placeholders.
    Using ``min_pixels=(patch×merge)²`` keeps the canvas minimal: ``image_grid_thw`` becomes ``[1, 2, 2]`` and
    ``prod // merge_size² == 1``.
    """
    ip = processor.image_processor
    ps = int(ip.patch_size)
    ms = int(ip.merge_size)
    size = getattr(ip, "size", None) or {}
    max_px = int(size.get("longest_edge", 28 * 28 * 1280))
    return {"min_pixels": (ps * ms) ** 2, "max_pixels": max_px}


def qwen_vl_extract_merge_vision_rows(
    model: PopeMultimodalModel,
    pixel_values: torch.Tensor,
    image_grid_thw: torch.Tensor,
    merge_flat_idx: int,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """One row of vision ``pooler_output`` plus optional **DeepStack** rows.

    **Qwen3-VL:** ``model.model.get_image_features`` returns ``BaseModelOutputWithDeepstackFeatures``. HF splits
    ``pooler_output`` per image but leaves each DeepStack tensor as one matrix of shape ``(n_merge_total, D)`` with the
    same ``n_merge_total`` as ``torch.cat(pooler_output)``, so ``merge_flat_idx`` lines up with the pooler row.

    **Qwen3.5:** vision returns ``BaseModelOutputWithPooling`` only — no ``deepstack_features`` → the second return
    value is an empty list; the LM forward never receives ``deepstack_visual_embeds``.

    ``merge_flat_idx`` is the usual flat index over ``llm_h × llm_w`` (row-major), same as ``llm_merge_index``.
    """
    with torch.inference_mode():
        out = model.model.get_image_features(pixel_values, image_grid_thw, return_dict=True)
    pool_parts = out.pooler_output
    if isinstance(pool_parts, (list, tuple)):
        pool_cat = torch.cat(pool_parts, dim=0)
    else:
        pool_cat = pool_parts
    if merge_flat_idx < 0 or merge_flat_idx >= pool_cat.shape[0]:
        raise IndexError(
            f"merge_flat_idx={merge_flat_idx} out of range for {pool_cat.shape[0]} merge tokens"
        )
    pool_row = pool_cat[merge_flat_idx].detach()
    ds_rows: list[torch.Tensor] = []
    ds_src = getattr(out, "deepstack_features", None) or []
    for di, feat in enumerate(ds_src):
        if feat.shape[0] != pool_cat.shape[0]:
            raise RuntimeError(
                f"deepstack[{di}] rows {feat.shape[0]} != pooler merge rows {pool_cat.shape[0]}"
            )
        ds_rows.append(feat[merge_flat_idx].detach())
    return pool_row, ds_rows


@contextmanager
def qwen_vl_patch_get_image_deepstack_only(
    model: PopeMultimodalModel,
    deepstack_rows: list[torch.Tensor],
):
    """Replace only ``deepstack_features`` from ``get_image_features`` (Qwen3-VL DeepStack path).

    **No-op** if ``deepstack_rows`` is empty (normal on **Qwen3.5**, which has no DeepStack in ``Qwen3_5Model.forward``).

    If ``deepstack_rows`` is non-empty but the wrapped model output has no DeepStack (e.g. Qwen3.5 + manual list),
    raises — ``Qwen3_5ForConditionalGeneration`` does not pass ``deepstack_visual_embeds`` into the text stack.
    """
    if not deepstack_rows:
        yield
        return
    m = model.model
    orig = m.get_image_features

    def _wrapped(pixel_values, image_grid_thw=None, **kwargs):
        o = orig(pixel_values, image_grid_thw, **kwargs)
        cur = getattr(o, "deepstack_features", None)
        if cur is None or len(cur) == 0:
            raise RuntimeError(
                "get_image_features() returned no deepstack_features; cannot inject DeepStack rows "
                "(expected Qwen3-VL-style output). On Qwen3.5, extract leaves deepstack empty — do not pass rows here."
            )
        if len(cur) != len(deepstack_rows):
            raise RuntimeError(
                f"deepstack list length {len(cur)} != injected {len(deepstack_rows)}"
            )
        new_ds: list[torch.Tensor] = []
        for feat, row in zip(cur, deepstack_rows, strict=True):
            new_ds.append(row.to(device=feat.device, dtype=feat.dtype).unsqueeze(0))
        o.deepstack_features = new_ds
        return o

    m.get_image_features = _wrapped
    try:
        yield
    finally:
        m.get_image_features = orig


def vit_patches_2d(model: PopeMultimodalModel, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    """ViT ``last_hidden_state`` as ``(n_patches, hidden_dim)`` float32 CPU tensor."""
    with torch.inference_mode():
        vit_out = model.get_image_features(
            inputs["pixel_values"],
            inputs["image_grid_thw"],
            return_dict=True,
        )
    h = vit_out.last_hidden_state.float()
    if h.dim() == 3:
        h = h[0]
    elif h.dim() != 2:
        raise RuntimeError(f"Unexpected ViT hidden rank: {h.shape}")
    return h.detach().cpu()


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


def qwen_vit_fine_patch_resized_box(
    flat_k: int,
    grid_thw_1d: torch.Tensor,
    patch_size: int,
) -> tuple[int, int, int, int]:
    """Map ``get_image_features`` ``last_hidden_state`` row ``flat_k`` → box on **processor-resized** (x0,y0,x1,y1).

    Qwen3.x ViT returns one row per **fine** patch: grid ``(T, gh, gw)`` from ``image_grid_thw``, row-major over
    ``gh × gw`` (for ``T=1``, ``flat_k = hp * gw + wp``). Match with ``llm_merge_*`` helpers that divide by ``merge_size``.
    """
    t, gh, gw = (int(x) for x in grid_thw_1d.tolist())
    spatial = gh * gw
    if t != 1:
        raise NotImplementedError(f"patch overlay assumes T=1 (single image); got T={t}")
    if not (0 <= flat_k < spatial):
        raise IndexError(f"flat_k={flat_k} not in [0, {spatial}) for grid {gh}×{gw}")
    hp = flat_k // gw
    wp = flat_k % gw
    ps = int(patch_size)
    x0 = wp * ps
    y0 = hp * ps
    x1 = (wp + 1) * ps
    y1 = (hp + 1) * ps
    return x0, y0, x1, y1


def llm_merge_index_to_resized_box(
    merge_index: int,
    grid_thw_1d: torch.Tensor,
    *,
    patch_size: int,
    merge_size: int,
) -> tuple[int, int, int, int]:
    """Merged image-token index → box on **processor-resized** canvas (x0,y0,x1,y1).

    Row-major over ``llm_h × llm_w`` with ``llm_* = grid / merge_size``. Each LM image token spans a
    ``merge_size × merge_size`` block of **fine** patches (four when ``merge_size == 2``).
    """
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


# %% Config

# Multimodal backend: ``POPE_MODEL_FAMILY`` picks which checkpoint is **active** after **Load multimodal**.
# ``POPE_LOAD_BOTH_MODELS`` loads both (2× VRAM); use ``pope_set_active_model(\"qwen3_5\")`` / ``\"qwen3_vl\"`` to switch.
MODEL_ID_BY_FAMILY: dict[PopeModelFamily, str] = {
    "qwen3_5": "Qwen/Qwen3.5-4B",
    "qwen3_vl": "Qwen/Qwen3-VL-8B-Instruct",
}
POPE_MODEL_FAMILY: PopeModelFamily = "qwen3_5"
POPE_LOAD_BOTH_MODELS = False
pope_apply_chat_template_kwargs(POPE_MODEL_FAMILY)
MODEL_ID = MODEL_ID_BY_FAMILY[POPE_MODEL_FAMILY]
POPE_SPLIT = "test"
POPE_LIMIT = 10
POPE_START = 0
# Same text for every forward: ViT patches depend on pixels only; avoids template differences across POPE questions.
POPE_VISION_USER_TEXT = "."
SAVE_PT = True

print(
    "Config:",
    f"DATASET_ID={DATASET_ID}",
    f"split={POPE_SPLIT}",
    f"POPE_LIMIT={POPE_LIMIT} (unique image_source)",
    f"POPE_START={POPE_START}",
    f"POPE_VISION_USER_TEXT={POPE_VISION_USER_TEXT!r}",
    f"POPE_MODEL_FAMILY={POPE_MODEL_FAMILY}",
    f"POPE_LOAD_BOTH_MODELS={POPE_LOAD_BOTH_MODELS}",
    f"MODEL_ID={MODEL_ID}",
    f"MODEL_ID_BY_FAMILY={MODEL_ID_BY_FAMILY!r}",
    f"POPE_CHAT_TEMPLATE_KWARGS={POPE_CHAT_TEMPLATE_KWARGS!r}",
    f"SAVE_PT={SAVE_PT}",
)

# %% Load POPE subset

_ds = load_dataset(DATASET_ID, split=POPE_SPLIT)
N_TOTAL = len(_ds)
POPE_INDICES: list[int] = []
POPE_ROWS: list[dict] = []
_seen_images: set[str] = set()
POPE_ROWS_SCANNED = 0
for i in range(POPE_START, N_TOTAL):
    if len(POPE_ROWS) >= POPE_LIMIT:
        break
    POPE_ROWS_SCANNED += 1
    row = _ds[i]
    key = pope_row_image_key(row)
    if key in _seen_images:
        continue
    _seen_images.add(key)
    POPE_INDICES.append(i)
    POPE_ROWS.append(row)

print(
    f"POPE unique images: n={len(POPE_ROWS)} (target {POPE_LIMIT}), "
    f"dataset rows scanned from index {POPE_START}: {POPE_ROWS_SCANNED}, "
    f"dataset len={N_TOTAL}",
    flush=True,
)
if len(POPE_ROWS) < POPE_LIMIT:
    print(
        f"WARNING: only {len(POPE_ROWS)} unique image_source values before end of split; "
        "lower POPE_LIMIT or start earlier in the split.",
        flush=True,
    )

# %% Save POPE images under data/images/pope

# Requires **Load POPE subset**. Writes PNGs; fills ``POPE_IMAGE_RECORDS`` (``image_id`` = index into this run, same as ViT rows).

POPE_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
POPE_IMAGE_RECORDS: list[dict[str, object]] = []
for image_id, row in enumerate(POPE_ROWS):
    stem_src = row.get("image_source") or row.get("id") or POPE_INDICES[image_id]
    fname = f"pope_{image_id:03d}_{pope_sanitize_stem(stem_src)}.png"
    out_path = POPE_IMAGE_DIR / fname
    pope_row_to_pil(row).save(out_path, format="PNG")
    rel = out_path.relative_to(REPO_ROOT)
    POPE_IMAGE_RECORDS.append(
        {
            "image_id": image_id,
            "image_key": pope_row_image_key(row),
            "ds_index": POPE_INDICES[image_id],
            "filename": fname,
            "image_relpath": str(rel),
            "image_source": row.get("image_source"),
            "question_id": row.get("question_id"),
            "category": row.get("category"),
            "question": str(row.get("question", ""))[:200],
        }
    )
print(f"Saved {len(POPE_IMAGE_RECORDS)} images under", POPE_IMAGE_DIR.relative_to(REPO_ROOT), flush=True)

# %% Load multimodal (Qwen3.5 / Qwen3-VL)

# Needs **Config** ``MODEL_ID_BY_FAMILY``, ``POPE_MODEL_FAMILY``, ``POPE_LOAD_BOTH_MODELS``.
# Sets globals ``model``, ``processor``, ``MODEL_ID``, ``POPE_MODEL_FAMILY``, and registry for ``pope_set_active_model``.
# **Dual load:** set ``POPE_LOAD_BOTH_MODELS = True`` in Config (heavy); then ``pope_set_active_model(\"qwen3_vl\")`` etc. without reloading weights.


def _pope_load_mm(model_id: str, cls: type):
    m = cls.from_pretrained(
        model_id,
        torch_dtype="auto" if DEVICE.type == "cuda" else torch.float32,
        device_map="auto" if DEVICE.type == "cuda" else None,
    )
    if DEVICE.type != "cuda":
        m = m.to(DEVICE)
    proc = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    m.eval()
    return m, proc


_pair_q35: tuple[Qwen3_5ForConditionalGeneration, object] | None = None
_pair_qvl: tuple[Qwen3VLForConditionalGeneration, object] | None = None

if POPE_LOAD_BOTH_MODELS:
    _pair_q35 = _pope_load_mm(MODEL_ID_BY_FAMILY["qwen3_5"], Qwen3_5ForConditionalGeneration)
    _pair_qvl = _pope_load_mm(MODEL_ID_BY_FAMILY["qwen3_vl"], Qwen3VLForConditionalGeneration)
    print("Loaded **both** Qwen3.5 and Qwen3-VL into memory.", flush=True)
else:
    if POPE_MODEL_FAMILY == "qwen3_5":
        _pair_q35 = _pope_load_mm(MODEL_ID_BY_FAMILY["qwen3_5"], Qwen3_5ForConditionalGeneration)
    elif POPE_MODEL_FAMILY == "qwen3_vl":
        _pair_qvl = _pope_load_mm(MODEL_ID_BY_FAMILY["qwen3_vl"], Qwen3VLForConditionalGeneration)
    else:
        raise ValueError(f"POPE_MODEL_FAMILY={POPE_MODEL_FAMILY!r}")

pope_register_multimodal_models(
    qwen3_5=_pair_q35,
    qwen3_vl=_pair_qvl,
    model_id_by_family=MODEL_ID_BY_FAMILY,
)
pope_set_active_model(POPE_MODEL_FAMILY)
# If ``POPE_LOAD_BOTH_MODELS`` is True, switch later with e.g. ``pope_set_active_model("qwen3_vl")`` (no re-download).

# %% Extract ViT patch embeddings

# Needs **Save POPE images** (for ``POPE_IMAGE_RECORDS``) and **Load multimodal**.

all_rows: list[torch.Tensor] = []
patch_counts: list[int] = []
meta: list[dict[str, object]] = []

for j, row in enumerate(tqdm(POPE_ROWS, desc="POPE ViT patches")):
    pil = pope_row_to_pil(row)
    inputs = build_pope_vision_inputs(processor, pil, POPE_VISION_USER_TEXT, DEVICE)
    patches = vit_patches_2d(model, inputs)
    all_rows.append(patches)
    p_i = int(patches.shape[0])
    patch_counts.append(p_i)
    _img_rec = POPE_IMAGE_RECORDS[j]
    meta.append(
        {
            "image_id": _img_rec["image_id"],
            "image_key": pope_row_image_key(row),
            "list_idx": j,
            "ds_index": POPE_INDICES[j],
            "id": row.get("id"),
            "question_id": row.get("question_id"),
            "category": row.get("category"),
            "pope_question": str(row.get("question", "")),
            "n_patches": p_i,
            "hidden_dim": int(patches.shape[1]),
            "image_relpath": _img_rec["image_relpath"],
            "image_filename": _img_rec["filename"],
        }
    )

POPE_VIT_ALL_PATCHES = torch.cat(all_rows, dim=0)
_ids_parts = [torch.full((patch_counts[i],), i, dtype=torch.long) for i in range(len(patch_counts))]
POPE_VIT_PATCH_IMAGE_ID = torch.cat(_ids_parts, dim=0)

means = []
for p in all_rows:
    means.append(p.mean(dim=0))
POPE_VIT_MEAN_PER_IMAGE = torch.stack(means, dim=0)

# Padded (n_image, max_patches, dim) when patch counts differ
if len(set(patch_counts)) == 1:
    POPE_VIT_PATCHES_STACKED = torch.stack(all_rows, dim=0)
    POPE_VIT_PATCHES_FLAT = POPE_VIT_PATCHES_STACKED.reshape(len(all_rows), -1)
else:
    d = int(all_rows[0].shape[1])
    p_max = max(patch_counts)
    pad = torch.zeros(len(all_rows), p_max, d, dtype=torch.float32)
    mask = torch.zeros(len(all_rows), p_max, dtype=torch.bool)
    for i, p in enumerate(all_rows):
        n = p.shape[0]
        pad[i, :n, :] = p
        mask[i, :n] = True
    POPE_VIT_PATCHES_STACKED = pad
    POPE_VIT_PATCHES_MASK = mask
    POPE_VIT_PATCHES_FLAT = None  # ambiguous row length; use ALL_PATCHES + IMAGE_ID or manual flatten

n_img = len(all_rows)
print("n_image:", n_img)
print("POPE_VIT_ALL_PATCHES:", tuple(POPE_VIT_ALL_PATCHES.shape), "dtype", POPE_VIT_ALL_PATCHES.dtype)
print("POPE_VIT_PATCH_IMAGE_ID:", tuple(POPE_VIT_PATCH_IMAGE_ID.shape))
print("POPE_VIT_MEAN_PER_IMAGE:", tuple(POPE_VIT_MEAN_PER_IMAGE.shape), "← (n_image, n_vec_dim) with n_vec_dim = hidden size")
print("patch_counts:", patch_counts)
if POPE_VIT_PATCHES_FLAT is not None:
    print("POPE_VIT_PATCHES_FLAT (all images same #patches):", tuple(POPE_VIT_PATCHES_FLAT.shape))
else:
    print("Variable patch counts — use POPE_VIT_PATCHES_STACKED + POPE_VIT_PATCHES_MASK or POPE_VIT_ALL_PATCHES")

if SAVE_PT:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "POPE_VIT_ALL_PATCHES": POPE_VIT_ALL_PATCHES,
        "POPE_VIT_PATCH_IMAGE_ID": POPE_VIT_PATCH_IMAGE_ID,
        "POPE_VIT_MEAN_PER_IMAGE": POPE_VIT_MEAN_PER_IMAGE,
        "POPE_VIT_PATCHES_STACKED": POPE_VIT_PATCHES_STACKED,
        "meta": meta,
        "POPE_IMAGE_RECORDS": POPE_IMAGE_RECORDS,
        "MODEL_ID": MODEL_ID,
        "DATASET_ID": DATASET_ID,
    }
    if POPE_VIT_PATCHES_FLAT is not None:
        payload["POPE_VIT_PATCHES_FLAT"] = POPE_VIT_PATCHES_FLAT
    else:
        payload["POPE_VIT_PATCHES_MASK"] = POPE_VIT_PATCHES_MASK
    _path = OUT_DIR / "pope_vit_patch_embeddings.pt"
    torch.save(payload, _path)
    print("Saved:", _path.relative_to(REPO_ROOT), flush=True)

# %% Preview POPE images (HTML)

# Requires **Save POPE images**. Uses Stonesoup static mount: ``/data/image`` → repo ``data/images``.

_PREVIEW_HOST = "http://127.0.0.1:8765"
_c = "#e8eaed"
_muted = "#9aa0a6"
_border = "#3c4043"
_panel = "#252830"
_cards = []
for rec in POPE_IMAGE_RECORDS:
    _iid = rec["image_id"]
    _fn = str(rec["filename"])
    _url = f"{_PREVIEW_HOST}/data/image/pope/{html.escape(_fn)}"
    _rel = html.escape(str(rec["image_relpath"]))
    _q = html.escape(str(rec["question"]))
    _src = html.escape(str(rec.get("image_source", "")))
    _cat = html.escape(str(rec.get("category", "")))
    _ds = rec["ds_index"]
    _cards.append(
        f'<section style="border:1px solid {_border};border-radius:8px;border-left:4px solid #5c9fd6;'
        f"padding:12px;margin-bottom:14px;background:{_panel}\">"
        f"<p style=\"margin:0 0 8px;font-size:13px;color:{_muted}\">"
        f"<b style=\"color:{_c}\">image_id</b>={_iid} &nbsp;·&nbsp; "
        f"<b style=\"color:{_c}\">ds_index</b>={_ds} &nbsp;·&nbsp; "
        f"category={_cat}</p>"
        f"<p style=\"margin:0 0 6px;font-size:12px;color:{_muted}\">file: <code>{_rel}</code></p>"
        f"<p style=\"margin:0 0 8px;font-size:12px;color:{_muted}\">image_source: <code>{_src}</code></p>"
        f'<img src="{_url}" alt="" style="max-width:100%;max-height:280px;height:auto;border-radius:6px;display:block;border:1px solid {_border}"/>'
        f"<p style=\"margin:8px 0 0;font-size:12px;color:{_muted}\">Q: {_q}</p>"
        f"</section>"
    )
_preview_html = f"""<div style="font-family:system-ui,Segoe UI,sans-serif;font-size:14px;line-height:1.45;color:{_c}">
<h2 style="font-size:1.1rem;margin:0 0 0.75em">POPE subset — image_id ↔ file</h2>
<p style="margin:0 0 1em;color:{_muted};font-size:13px">Unique <code>image_source</code> rows. ``image_id`` is tensor row index (0 … n−1). Sample question shown for that dataset row.</p>
{"".join(_cards)}
</div>"""

print(STONESOUP_RENDER_HTML, _preview_html, sep="", flush=True)

# %% Histogram: ViT patch L2 norms (HTML preview)

# Needs **Extract ViT patch embeddings**. Writes PNG under ``data/images/pope_vit/`` for Stonesoup ``/data/image``; stdout is HTML only.

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

configure_matplotlib_fonts()

_pope_norms_pt = torch.linalg.vector_norm(POPE_VIT_ALL_PATCHES.float(), dim=-1)
_pope_norms = _pope_norms_pt.detach().cpu().numpy()
_n = int(len(_pope_norms))
_d = int(POPE_VIT_ALL_PATCHES.shape[1])
_min, _max = float(_pope_norms.min()), float(_pope_norms.max())
_mean, _std = float(_pope_norms.mean()), float(_pope_norms.std())

POPE_VIT_PLOT_DIR.mkdir(parents=True, exist_ok=True)
_hist_name = "patch_l2_norm_hist.png"
_hist_path = POPE_VIT_PLOT_DIR / _hist_name
_hist_host = "http://127.0.0.1:8765"
_hist_url = f"{_hist_host}/data/image/pope_vit/{_hist_name}"

_fig, _ax = plt.subplots(figsize=(9, 4.5))
_ax.hist(_pope_norms, bins=64, color="steelblue", edgecolor="black", alpha=0.85)
_ax.set_xlabel("L2 norm")
_ax.set_ylabel("Count")
_ax.set_title(f"POPE ViT patch vectors (n={_n:,}, dim={_d}) — L2 norm")
_ax.grid(True, alpha=0.3)
_fig.tight_layout()
savefig_mpl(_fig, _hist_path, dpi=150)
plt.close(_fig)

_c_h = "#e8eaed"
_muted_h = "#9aa0a6"
_border_h = "#3c4043"
_hist_html = f"""<div style="font-family:system-ui,Segoe UI,sans-serif;font-size:14px;line-height:1.5;color:{_c_h}">
<h2 style="font-size:1.1rem;margin:0 0 0.35em">ViT patch L2 norms</h2>
<p style="margin:0 0 0.75em;color:{_muted_h};font-size:13px">
n={_n:,} · dim={_d} &nbsp;·&nbsp; min={_min:.4g} · max={_max:.4g} · mean={_mean:.4g} · std={_std:.4g}<br/>
file <code>{html.escape(str(_hist_path.relative_to(REPO_ROOT)))}</code> ·
<a href="{html.escape(_hist_url)}" style="color:#8ab4f8">{html.escape(_hist_url)}</a>
</p>
<img src="{html.escape(_hist_url)}" alt="Histogram of ViT patch L2 norms" style="max-width:100%;height:auto;border:1px solid {_border_h};border-radius:8px;display:block"/>
</div>"""

print(STONESOUP_RENDER_HTML, _hist_html, sep="", flush=True)

# %% Histogram: all ViT embedding values (HTML preview)

# Needs **Extract ViT patch embeddings** (``POPE_VIT_ALL_PATCHES``). A full-range histogram hides the bulk when
# outliers stretch the x-axis; this figure uses a **percentile window** (default 0.5–99.5%) plus **log-y** on the same bins.

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

configure_matplotlib_fonts()

import numpy as np

_elems = POPE_VIT_ALL_PATCHES.float().reshape(-1).detach().cpu().numpy()
_n_elem = int(_elems.size)
_n_patch = int(POPE_VIT_ALL_PATCHES.shape[0])
_d_elem = int(POPE_VIT_ALL_PATCHES.shape[1])
_el_min, _el_max = float(_elems.min()), float(_elems.max())
_el_mean, _el_std = float(_elems.mean()), float(_elems.std())
_EL_PCT_LO, _EL_PCT_HI = 0.5, 99.5
_win_lo, _win_hi = (float(x) for x in np.percentile(_elems, [_EL_PCT_LO, _EL_PCT_HI]))
_n_in_win = int(np.logical_and(_elems >= _win_lo, _elems <= _win_hi).sum())
_pct_in_win = 100.0 * _n_in_win / _n_elem

POPE_VIT_PLOT_DIR.mkdir(parents=True, exist_ok=True)
_el_hist_name = "patch_all_elements_hist.png"
_el_hist_path = POPE_VIT_PLOT_DIR / _el_hist_name
_el_hist_host = "http://127.0.0.1:8765"
_el_hist_url = f"{_el_hist_host}/data/image/pope_vit/{_el_hist_name}"

_nbins = 160
_counts_e, _edges_e = np.histogram(_elems, bins=_nbins, range=(_win_lo, _win_hi))
_centers_e = (_edges_e[:-1] + _edges_e[1:]) / 2.0
_width_e = float(_edges_e[1] - _edges_e[0])

_fig_e, (_ax_lin, _ax_log) = plt.subplots(2, 1, figsize=(10, 7.2), sharex=True, gridspec_kw={"height_ratios": [1.15, 1]})
_fig_e.suptitle(
    f"POPE ViT patch embeddings — all elements (N={_n_elem:,} = {_n_patch:,}×{_d_elem}; "
    f"global min={_el_min:.4g}, max={_el_max:.4g})",
    fontsize=11,
)
_ax_lin.bar(
    _centers_e,
    _counts_e,
    width=_width_e * 0.98,
    align="center",
    color="darkseagreen",
    edgecolor="black",
    alpha=0.88,
)
_ax_lin.axvline(_el_mean, color="crimson", linestyle="--", linewidth=1, label=f"mean {_el_mean:.4g}")
_ax_lin.axvline(float(np.median(_elems)), color="navy", linestyle=":", linewidth=1, label="median")
_ax_lin.set_ylabel("Count")
_ax_lin.set_title(
    f"Linear y — values in [{_win_lo:.4g}, {_win_hi:.4g}] "
    f"({_EL_PCT_LO}–{_EL_PCT_HI} percentiles; ~{_pct_in_win:.2f}% of samples in window)"
)
_ax_lin.grid(True, alpha=0.3)
_ax_lin.legend(loc="upper right", fontsize=8)

_ax_log.bar(
    _centers_e,
    np.maximum(_counts_e.astype(float), 0.5),
    width=_width_e * 0.98,
    align="center",
    color="darkseagreen",
    edgecolor="black",
    alpha=0.88,
)
_ax_log.set_yscale("log")
_ax_log.set_xlabel("Value")
_ax_log.set_ylabel("Count (log scale)")
_ax_log.set_title("Same bins — log y reveals tails inside the percentile window")
_ax_log.grid(True, alpha=0.3, which="both")
_ax_lin.set_xlim(_win_lo, _win_hi)
_ax_log.set_xlim(_win_lo, _win_hi)
_fig_e.tight_layout()
savefig_mpl(_fig_e, _el_hist_path, dpi=150)
plt.close(_fig_e)

# Bust browser/UI caches: same path would otherwise keep showing an older PNG.
_el_hist_url_bust = f"{_el_hist_url}?cb={int(_el_hist_path.stat().st_mtime_ns)}"

_c_e = "#e8eaed"
_muted_e = "#9aa0a6"
_border_e = "#3c4043"
_el_html = f"""<div style="font-family:system-ui,Segoe UI,sans-serif;font-size:14px;line-height:1.5;color:{_c_e}">
<h2 style="font-size:1.1rem;margin:0 0 0.35em">ViT patch embedding values (all elements)</h2>
<p style="margin:0 0 0.75em;color:{_muted_e};font-size:13px">
<strong>Two panels:</strong> top = linear y on {_EL_PCT_LO}–{_EL_PCT_HI}% window; bottom = log y, same bins.
If you still see one spike to 5000, re-run this cell (kernels can keep old code until reload).<br/>
N={_n_elem:,} · mean={_el_mean:.4g} · std={_el_std:.4g} · global [{_el_min:.4g}, {_el_max:.4g}]<br/>
Plot window: [{_win_lo:.4g}, {_win_hi:.4g}] (~{_pct_in_win:.2f}% of values).<br/>
file <code>{html.escape(str(_el_hist_path.relative_to(REPO_ROOT)))}</code> ·
<a href="{html.escape(_el_hist_url_bust)}" style="color:#8ab4f8">open image</a>
</p>
<img src="{html.escape(_el_hist_url_bust)}" alt="Histogram of all ViT embedding values" style="max-width:100%;height:auto;border:1px solid {_border_e};border-radius:8px;display:block"/>
</div>"""

print(STONESOUP_RENDER_HTML, _el_html, sep="", flush=True)

# %% Histogram: per-image embedding values 0–9 (HTML preview)

# Needs **Extract ViT patch embeddings** (``POPE_VIT_ALL_PATCHES``, ``POPE_VIT_PATCH_IMAGE_ID``). Same style as
# the all-elements cell, for each ``image_id`` in ``0 … 9`` (skips ids with no patch rows).

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

configure_matplotlib_fonts()

import numpy as np

_POPE_PER_IMAGE_HIST_IDS = range(10)
_PCT_LO_i, _PCT_HI_i = 0.5, 99.5
_NBINS_i = 160
_HIST_HOST_i = "http://127.0.0.1:8765"

POPE_VIT_PLOT_DIR.mkdir(parents=True, exist_ok=True)

_cih = "#e8eaed"
_mih = "#9aa0a6"
_bih = "#3c4043"
_gallery_sections: list[str] = []

for _iid in _POPE_PER_IMAGE_HIST_IDS:
    _mi = POPE_VIT_PATCH_IMAGE_ID == _iid
    _pi = POPE_VIT_ALL_PATCHES[_mi]
    if _pi.numel() == 0:
        _gallery_sections.append(
            f'<p style="margin:0.5em 0;color:{_mih}">image_id={_iid}: no patch rows (skipped).</p>'
        )
        continue
    _n_pi = int(_pi.shape[0])
    _di = int(_pi.shape[1])
    _eli = _pi.float().reshape(-1).detach().cpu().numpy()
    _n_eli = int(_eli.size)
    _emn, _emx = float(_eli.min()), float(_eli.max())
    _emu, _esg = float(_eli.mean()), float(_eli.std())
    _w_lo, _w_hi = (float(x) for x in np.percentile(_eli, [_PCT_LO_i, _PCT_HI_i]))
    _n_in = int(np.logical_and(_eli >= _w_lo, _eli <= _w_hi).sum())
    _pct_in = 100.0 * _n_in / _n_eli

    _ci, _edi = np.histogram(_eli, bins=_NBINS_i, range=(_w_lo, _w_hi))
    _cti = (_edi[:-1] + _edi[1:]) / 2.0
    _wdi = float(_edi[1] - _edi[0])

    _f_i, (_ax_li, _ax_lo) = plt.subplots(
        2, 1, figsize=(10, 7.2), sharex=True, gridspec_kw={"height_ratios": [1.15, 1]}
    )
    _f_i.suptitle(
        f"POPE ViT patch embeddings — image_id={_iid} "
        f"(N={_n_eli:,} = {_n_pi:,} patches × {_di}; global min={_emn:.4g}, max={_emx:.4g})",
        fontsize=11,
    )
    _ax_li.bar(
        _cti,
        _ci,
        width=_wdi * 0.98,
        align="center",
        color="coral",
        edgecolor="black",
        alpha=0.88,
    )
    _ax_li.axvline(_emu, color="crimson", linestyle="--", linewidth=1, label=f"mean {_emu:.4g}")
    _ax_li.axvline(float(np.median(_eli)), color="navy", linestyle=":", linewidth=1, label="median")
    _ax_li.set_ylabel("Count")
    _ax_li.set_title(
        f"Linear y — values in [{_w_lo:.4g}, {_w_hi:.4g}] "
        f"({_PCT_LO_i}–{_PCT_HI_i} percentiles; ~{_pct_in:.2f}% of samples in window)"
    )
    _ax_li.grid(True, alpha=0.3)
    _ax_li.legend(loc="upper right", fontsize=8)
    _hist_name_i = f"patch_image_id{_iid:02d}_elements_hist.png"
    _hist_path_i = POPE_VIT_PLOT_DIR / _hist_name_i

    _ax_lo.bar(
        _cti,
        np.maximum(_ci.astype(float), 0.5),
        width=_wdi * 0.98,
        align="center",
        color="coral",
        edgecolor="black",
        alpha=0.88,
    )
    _ax_lo.set_yscale("log")
    _ax_lo.set_xlabel("Value")
    _ax_lo.set_ylabel("Count (log scale)")
    _ax_lo.set_title("Same bins — log y reveals tails inside the percentile window")
    _ax_lo.grid(True, alpha=0.3, which="both")
    _ax_li.set_xlim(_w_lo, _w_hi)
    _ax_lo.set_xlim(_w_lo, _w_hi)
    _f_i.tight_layout()
    savefig_mpl(_f_i, _hist_path_i, dpi=150)
    plt.close(_f_i)

    _url_i = f"{_HIST_HOST_i}/data/image/pope_vit/{_hist_name_i}"
    _url_bust_i = f"{_url_i}?cb={int(_hist_path_i.stat().st_mtime_ns)}"
    _rel_i = html.escape(str(_hist_path_i.relative_to(REPO_ROOT)))
    _gallery_sections.append(
        f'<section style="margin-bottom:1.75rem;padding-bottom:1.25rem;border-bottom:1px solid {_bih}">'
        f"<h3 style=\"font-size:1rem;margin:0 0 0.35em;color:{_cih}\">image_id={_iid}</h3>"
        f"<p style=\"margin:0 0 0.5em;color:{_mih};font-size:13px\">"
        f"N={_n_eli:,} ({_n_pi:,}×{_di}) · mean={_emu:.4g} · std={_esg:.4g} · "
        f"global [{_emn:.4g}, {_emx:.4g}] · window [{_w_lo:.4g}, {_w_hi:.4g}] (~{_pct_in:.2f}%)<br/>"
        f"<code>{_rel_i}</code> · "
        f"<a href=\"{html.escape(_url_bust_i)}\" style=\"color:#8ab4f8\">open</a>"
        f"</p>"
        f"<img src=\"{html.escape(_url_bust_i)}\" alt=\"image_id {_iid}\" "
        f"style=\"max-width:100%;height:auto;border:1px solid {_bih};border-radius:8px;display:block\"/>"
        f"</section>"
    )

_per_img_html = f"""<div style="font-family:system-ui,Segoe UI,sans-serif;font-size:14px;line-height:1.5;color:{_cih}">
<h2 style="font-size:1.1rem;margin:0 0 0.75em">ViT patch values — per image_id (0 … 9)</h2>
{"".join(_gallery_sections)}
</div>"""

print(STONESOUP_RENDER_HTML, _per_img_html, sep="", flush=True)

# %% Patch L2-norm overlay on image: image_id 0–9 (HTML preview)

# Needs **Extract ViT patch embeddings**, **Load** multimodal model, **Load POPE subset** (``POPE_ROWS``). For each image,
# colors **fine** ViT patches (``last_hidden_state`` rows) by L2 norm; ``cmap="Blues"``, per-image 2–98% norm scale.

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

configure_matplotlib_fonts()

import matplotlib.cm as cm_mpl
from matplotlib import colormaps
from matplotlib.colors import Normalize
import numpy as np
from PIL import ImageDraw

_OVERLAY_IDS = range(10)
_p_lo_c, _p_hi_c = 2.0, 98.0
_cmap_ov = colormaps["Blues"]
_fill_alpha = 115
_outline_alpha = 100

POPE_VIT_PLOT_DIR.mkdir(parents=True, exist_ok=True)
_ov_host = "http://127.0.0.1:8765"
_cov = "#e8eaed"
_mov = "#9aa0a6"
_bov = "#3c4043"
_overlay_sections: list[str] = []

for _oid in _OVERLAY_IDS:
    if _oid >= len(POPE_ROWS):
        _overlay_sections.append(
            f'<p style="margin:0.5em 0;color:{_mov}">image_id={_oid}: no POPE_ROWS (skipped).</p>'
        )
        continue

    _pil_ov = pope_row_to_pil(POPE_ROWS[_oid])
    _ov_w, _ov_h = _pil_ov.size
    _inp_ov = build_pope_vision_inputs(processor, _pil_ov, POPE_VISION_USER_TEXT, DEVICE)
    _g_ov = _inp_ov["image_grid_thw"][0].detach().cpu()
    _t_ov, _gh_ov, _gw_ov = (int(x) for x in _g_ov.tolist())
    _ps_ov = int(processor.image_processor.patch_size)
    _res_wh_ov = (_gw_ov * _ps_ov, _gh_ov * _ps_ov)

    _m_ov = POPE_VIT_PATCH_IMAGE_ID == _oid
    _p_ov = POPE_VIT_ALL_PATCHES[_m_ov]
    _n_spatial_ov = _gh_ov * _gw_ov
    if int(_t_ov) != 1:
        raise NotImplementedError("T>1 video: extend spatial loop for temporal slices.")
    if int(_p_ov.shape[0]) != _n_spatial_ov:
        raise RuntimeError(
            f"image_id={_oid}: patch count {_p_ov.shape[0]} != grid gh×gw={_n_spatial_ov}"
        )

    _norms_ov = torch.linalg.vector_norm(_p_ov.float(), dim=-1).numpy()
    _v_lo = float(np.percentile(_norms_ov, _p_lo_c))
    _v_hi = float(np.percentile(_norms_ov, _p_hi_c))
    if _v_hi <= _v_lo:
        _v_hi = _v_lo + 1e-6
    _norm_mpl_ov = Normalize(vmin=_v_lo, vmax=_v_hi, clip=True)

    _overlay_rgba = Image.new("RGBA", (_ov_w, _ov_h), (0, 0, 0, 0))
    _draw_ov = ImageDraw.Draw(_overlay_rgba)
    _pen_ov = max(1, min(_ov_w, _ov_h) // 384)
    for _k in range(_n_spatial_ov):
        _xr0, _yr0, _xr1, _yr1 = qwen_vit_fine_patch_resized_box(_k, _g_ov, _ps_ov)
        _xo0, _yo0, _xo1, _yo1 = resized_box_to_original(
            (_xr0, _yr0, _xr1, _yr1),
            resized_wh=_res_wh_ov,
            original_wh=(_ov_w, _ov_h),
        )
        _tc = float(_norm_mpl_ov(_norms_ov[_k]))
        _rgba_m = _cmap_ov(_tc)
        _fill_ov = (
            int(_rgba_m[0] * 255),
            int(_rgba_m[1] * 255),
            int(_rgba_m[2] * 255),
            _fill_alpha,
        )
        _draw_ov.rectangle(
            (_xo0, _yo0, _xo1, _yo1),
            fill=_fill_ov,
            outline=(255, 255, 255, _outline_alpha),
            width=_pen_ov,
        )

    _comp_ov = Image.alpha_composite(_pil_ov.convert("RGBA"), _overlay_rgba).convert("RGB")
    _ov_name = f"image_id{_oid:02d}_patch_l2norm_overlay.png"
    _ov_path = POPE_VIT_PLOT_DIR / _ov_name
    _ov_url = f"{_ov_host}/data/image/pope_vit/{_ov_name}"

    _fig_ov, _ax_ov = plt.subplots(figsize=(11, 11))
    _ax_ov.imshow(np.asarray(_comp_ov))
    _ax_ov.axis("off")
    _ax_ov.set_title(
        f"image_id={_oid}: ViT fine patches ({_gh_ov}×{_gw_ov}) · L2 norm (Blues, {_p_lo_c:.0f}–{_p_hi_c:.0f}%ile)"
    )
    _sm_ov = cm_mpl.ScalarMappable(cmap=_cmap_ov, norm=_norm_mpl_ov)
    _sm_ov.set_array([])
    _fig_ov.colorbar(
        _sm_ov,
        ax=_ax_ov,
        fraction=0.035,
        pad=0.02,
        label=f"L2 norm ({_p_lo_c:.0f}–{_p_hi_c:.0f}%ile; clip beyond)",
    )
    _fig_ov.tight_layout()
    savefig_mpl(_fig_ov, _ov_path, dpi=140)
    plt.close(_fig_ov)

    _ov_url_bust = f"{_ov_url}?cb={int(_ov_path.stat().st_mtime_ns)}"
    _rel_ov = html.escape(str(_ov_path.relative_to(REPO_ROOT)))
    _overlay_sections.append(
        f'<section style="margin-bottom:1.75rem;padding-bottom:1.25rem;border-bottom:1px solid {_bov}">'
        f"<h3 style=\"font-size:1rem;margin:0 0 0.35em;color:{_cov}\">image_id={_oid}</h3>"
        f"<p style=\"margin:0 0 0.5em;color:{_mov};font-size:13px\">"
        f"Grid {_gh_ov}×{_gw_ov} · norm [{_v_lo:.4g}, {_v_hi:.4g}] (per-image {_p_lo_c:.0f}–{_p_hi_c:.0f}%).<br/>"
        f"<code>{_rel_ov}</code> · "
        f"<a href=\"{html.escape(_ov_url_bust)}\" style=\"color:#8ab4f8\">open</a>"
        f"</p>"
        f"<img src=\"{html.escape(_ov_url_bust)}\" alt=\"overlay {_oid}\" "
        f"style=\"max-width:100%;height:auto;border:1px solid {_bov};border-radius:8px;display:block\"/>"
        f"</section>"
    )

_ov_gallery_html = f"""<div style="font-family:system-ui,Segoe UI,sans-serif;font-size:14px;line-height:1.5;color:{_cov}">
<h2 style="font-size:1.1rem;margin:0 0 0.75em">Patch L2 norm overlay (Blues) — image_id 0 … 9</h2>
{"".join(_overlay_sections)}
</div>"""

print(STONESOUP_RENDER_HTML, _ov_gallery_html, sep="", flush=True)

# %% Merged tokens — top-3 cos-sim tables + top-1 Blues figure (image_id 0–9, HTML preview)

# Needs **Load** multimodal model, **Load POPE subset**. ``pooler_output`` rows vs **L2-normalized** ``embed_tokens.weight``
# rows; ``topk`` cosine (dot of normalized vectors). Each grid cell = ``merge_size``² fine patches.
# **Figure (each image):** top- **1** match only, cell **fill** colored by cos-sim (``Blues``, 2–98%ile on that grid), PIL
# composite + white outlines like **Patch L2-norm overlay**; HTML **table** still lists top-3.

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

configure_matplotlib_fonts()

import matplotlib.cm as cm_mpl
import matplotlib.patheffects as pe
from matplotlib import colormaps
from matplotlib.colors import Normalize
import numpy as np
import torch.nn.functional as F
from PIL import Image, ImageDraw

_EMB_COS_TOPK = 3
_LMH_IDS = range(10)

POPE_VIT_PLOT_DIR.mkdir(parents=True, exist_ok=True)
_lmh_host = "http://127.0.0.1:8765"
_clm = "#e8eaed"
_mlm = "#9aa0a6"
_blm = "#3c4043"
_lmh_sections: list[str] = []


def _short_embed_decode(_tid: int, *, max_chars: int = 10) -> str:
    _p = processor.decode(
        [_tid],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    _s = _p.replace("\n", " ").replace("\r", " ").strip()
    if not _s:
        return f"·{_tid}"
    if len(_s) > max_chars:
        return _s[: max_chars - 1] + "…"
    return _s


def _table_piece(_tid: int, *, max_chars: int = 36) -> str:
    _p = processor.decode(
        [_tid],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    _s = _p.replace("\n", " ").replace("\r", " ").strip()
    if not _s:
        return f"·{_tid}"
    if len(_s) > max_chars:
        return _s[: max_chars - 1] + "…"
    return _s


with torch.inference_mode():
    _emb_w = model.get_input_embeddings().weight.float().to(DEVICE)
    _emb_n = F.normalize(_emb_w, dim=-1, eps=1e-12)

_p_lm_lo, _p_lm_hi = 2.0, 98.0
_cmap_lm = colormaps["Blues"]
_fill_alpha_lm = 115
_outline_alpha_lm = 100
_lm_fig_w_in = 16.0
_lm_save_dpi = 200

for _lmid in _LMH_IDS:
    if _lmid >= len(POPE_ROWS):
        _lmh_sections.append(
            f'<p style="margin:0.5em 0;color:{_mlm}">image_id={_lmid}: no POPE_ROWS (skipped).</p>'
        )
        continue

    _pil_lm = pope_row_to_pil(POPE_ROWS[_lmid])
    _Wlm, _Hlm = _pil_lm.size
    _inp_lm = build_pope_vision_inputs(processor, _pil_lm, POPE_VISION_USER_TEXT, DEVICE)
    _g_lm = _inp_lm["image_grid_thw"][0].detach().cpu()
    _t_lm, _gh_lm, _gw_lm = (int(x) for x in _g_lm.tolist())
    _ms_lm = int(processor.image_processor.merge_size)
    _ps_lm = int(processor.image_processor.patch_size)
    _res_wh_lm = (_gw_lm * _ps_lm, _gh_lm * _ps_lm)
    _llm_h = _gh_lm // _ms_lm
    _llm_w = _gw_lm // _ms_lm
    _n_merge = _llm_h * _llm_w

    if _t_lm != 1:
        raise NotImplementedError("embed-cos overlay: assume T=1 single image.")

    with torch.inference_mode():
        _vit_lm = model.get_image_features(
            _inp_lm["pixel_values"],
            _inp_lm["image_grid_thw"],
            return_dict=True,
        )
        _pool_lm = _vit_lm.pooler_output
        if len(_pool_lm) != 1:
            raise RuntimeError(f"Expected one image in pooler_output, got {len(_pool_lm)}")
        _img_tok_lm = _pool_lm[0].float()
        if int(_img_tok_lm.shape[0]) != _n_merge:
            raise RuntimeError(
                f"image_id={_lmid}: pooler rows {_img_tok_lm.shape[0]} vs merge grid "
                f"{_llm_h}×{_llm_w}={_n_merge}"
            )
        if _img_tok_lm.shape[-1] != _emb_w.shape[-1]:
            raise RuntimeError(
                f"hidden dim mismatch: image token {_img_tok_lm.shape[-1]} vs embed {_emb_w.shape[-1]}"
            )
        _img_n = F.normalize(_img_tok_lm.to(DEVICE), dim=-1, eps=1e-12)
        _cos_all = _img_n @ _emb_n.T
        _topv, _topi = torch.topk(_cos_all, k=_EMB_COS_TOPK, dim=-1)
        _topv_cpu = _topv.detach().cpu()
        _topi_cpu = _topi.detach().cpu()

    _cos1_np = _topv_cpu[:, 0].numpy()
    _lm_v_lo = float(np.percentile(_cos1_np, _p_lm_lo))
    _lm_v_hi = float(np.percentile(_cos1_np, _p_lm_hi))
    if _lm_v_hi <= _lm_v_lo:
        _lm_v_hi = _lm_v_lo + 1e-6
    _norm_lm = Normalize(vmin=_lm_v_lo, vmax=_lm_v_hi, clip=True)

    _overlay_lm = Image.new("RGBA", (_Wlm, _Hlm), (0, 0, 0, 0))
    _draw_lm = ImageDraw.Draw(_overlay_lm)
    _pen_lm = max(1, min(_Wlm, _Hlm) // 384)
    for _mk in range(_n_merge):
        _xr0, _yr0, _xr1, _yr1 = llm_merge_index_to_resized_box(
            _mk, _g_lm, patch_size=_ps_lm, merge_size=_ms_lm
        )
        _xo0, _yo0, _xo1, _yo1 = resized_box_to_original(
            (_xr0, _yr0, _xr1, _yr1),
            resized_wh=_res_wh_lm,
            original_wh=(_Wlm, _Hlm),
        )
        _tc = float(_norm_lm(float(_cos1_np[_mk])))
        _rgba_m = _cmap_lm(_tc)
        _fill_lm = (
            int(_rgba_m[0] * 255),
            int(_rgba_m[1] * 255),
            int(_rgba_m[2] * 255),
            _fill_alpha_lm,
        )
        _draw_lm.rectangle(
            (_xo0, _yo0, _xo1, _yo1),
            fill=_fill_lm,
            outline=(255, 255, 255, _outline_alpha_lm),
            width=_pen_lm,
        )
    _comp_lm = Image.alpha_composite(_pil_lm.convert("RGBA"), _overlay_lm).convert("RGB")

    _lm_fig_h_in = max(4.0, _lm_fig_w_in * (_Hlm / max(1, _Wlm)))
    _lmh_name = f"image_id{_lmid:02d}_merged_embed_cos_top1_blues.png"
    _lmh_path = POPE_VIT_PLOT_DIR / _lmh_name
    _lmh_url = f"{_lmh_host}/data/image/pope_vit/{_lmh_name}"

    with mpl_suppress_glyph_userwarnings():
        _fig_lm, _ax_lm = plt.subplots(figsize=(_lm_fig_w_in, _lm_fig_h_in))
        _ax_lm.imshow(
            np.asarray(_comp_lm),
            extent=(0, _Wlm, _Hlm, 0),
            origin="upper",
            aspect="equal",
        )
        _ax_lm.set_xlim(0, _Wlm)
        _ax_lm.set_ylim(_Hlm, 0)
        _ax_lm.set_axis_off()
        _ax_lm.set_title(
            f"image_id={_lmid}: merged ({_llm_h}×{_llm_w}) · top-1 cos vs embed_tokens (Blues, "
            f"{_p_lm_lo:.0f}–{_p_lm_hi:.0f}%ile)",
            fontsize=14,
        )
        _sm_lm = cm_mpl.ScalarMappable(cmap=_cmap_lm, norm=_norm_lm)
        _sm_lm.set_array([])
        _cb_lm = _fig_lm.colorbar(
            _sm_lm,
            ax=_ax_lm,
            fraction=0.035,
            pad=0.02,
        )
        _cb_lm.set_label(
            f"cos (top-1; {_p_lm_lo:.0f}–{_p_lm_hi:.0f}%ile; clip beyond)",
            fontsize=12,
        )
        _cb_lm.ax.tick_params(labelsize=11)
        for _mk in range(_n_merge):
            _xr0, _yr0, _xr1, _yr1 = llm_merge_index_to_resized_box(
                _mk, _g_lm, patch_size=_ps_lm, merge_size=_ms_lm
            )
            _xo0, _yo0, _xo1, _yo1 = resized_box_to_original(
                (_xr0, _yr0, _xr1, _yr1),
                resized_wh=_res_wh_lm,
                original_wh=(_Wlm, _Hlm),
            )
            _cw = max(1, _xo1 - _xo0)
            _ch = max(1, _yo1 - _yo0)
            _tid0 = int(_topi_cpu[_mk, 0].item())
            _cv0 = float(_topv_cpu[_mk, 0].item())
            _show = _sanitize_mpl_overlay_text(_short_embed_decode(_tid0))
            _label_2l = f"{_show}\n{_cv0:.4f}"
            _cx = _xo0 + 0.5 * _cw
            _cy = _yo0 + 0.5 * _ch
            _fs = max(10, min(22, int(min(_cw, _ch) / 5)))
            _lw_stroke = max(3.0, _fs * 0.35)
            _ax_lm.text(
                _cx,
                _cy,
                _label_2l,
                ha="center",
                va="center",
                linespacing=0.95,
                fontsize=_fs,
                color="white",
                path_effects=[
                    pe.Stroke(linewidth=_lw_stroke, foreground="black"),
                    pe.Normal(),
                ],
            )

        savefig_mpl(
            _fig_lm,
            _lmh_path,
            tight_layout=True,
            dpi=_lm_save_dpi,
            bbox_inches="tight",
            pad_inches=0.02,
        )
        plt.close(_fig_lm)

    _lmh_bust = f"{_lmh_url}?cb={int(_lmh_path.stat().st_mtime_ns)}"
    _rel_lm = html.escape(str(_lmh_path.relative_to(REPO_ROOT)))
    _table_rows: list[str] = []
    for _mk in range(_n_merge):
        _hh = _mk // _llm_w
        _ww = _mk % _llm_w
        _cells = [
            str(_mk),
            str(_hh),
            str(_ww),
        ]
        for _rk in range(_EMB_COS_TOPK):
            _tid_i = int(_topi_cpu[_mk, _rk].item())
            _cv = float(_topv_cpu[_mk, _rk].item())
            _tp = html.escape(_table_piece(_tid_i))
            _cells.append(f"{_tp}")
            _cells.append(f"{_cv:.5f}")
        _table_rows.append("<tr><td>" + "</td><td>".join(_cells) + "</td></tr>")
    _table_html = (
        "<table style=\"border-collapse:collapse;font-size:12px;margin:0.75em 0;color:{c};"
        "width:100%;max-width:min(1200px,100%)\">"
        "<thead><tr style=\"background:#252830;border-bottom:1px solid {b}\">"
        "<th style=\"padding:4px 6px;text-align:left;border:1px solid {b}\">merge</th>"
        "<th style=\"padding:4px 6px;text-align:left;border:1px solid {b}\">h</th>"
        "<th style=\"padding:4px 6px;text-align:left;border:1px solid {b}\">w</th>"
        "<th style=\"padding:4px 6px;text-align:left;border:1px solid {b}\">#1 token</th>"
        "<th style=\"padding:4px 6px;text-align:right;border:1px solid {b}\">cos₁</th>"
        "<th style=\"padding:4px 6px;text-align:left;border:1px solid {b}\">#2 token</th>"
        "<th style=\"padding:4px 6px;text-align:right;border:1px solid {b}\">cos₂</th>"
        "<th style=\"padding:4px 6px;text-align:left;border:1px solid {b}\">#3 token</th>"
        "<th style=\"padding:4px 6px;text-align:right;border:1px solid {b}\">cos₃</th>"
        "</tr></thead><tbody>{body}</tbody></table>"
    ).format(c=_clm, b=_blm, body="".join(_table_rows))

    _lmh_sections.append(
        f'<section style="margin-bottom:1.75rem;padding-bottom:1.25rem;border-bottom:1px solid {_blm}">'
        f"<h3 style=\"font-size:1rem;margin:0 0 0.35em;color:{_clm}\">image_id={_lmid}</h3>"
        f"<p style=\"margin:0 0 0.5em;color:{_mlm};font-size:13px\">"
        f"Grid {_llm_h}×{_llm_w} · top-1 cos [{_lm_v_lo:.4g}, {_lm_v_hi:.4g}] "
        f"({_p_lm_lo:.0f}–{_p_lm_hi:.0f}%ile on this grid).<br/>"
        f"<code>{_rel_lm}</code> · "
        f"<a href=\"{html.escape(_lmh_bust)}\" style=\"color:#8ab4f8\">open image</a>"
        f"</p>"
        f"<img src=\"{html.escape(_lmh_bust)}\" alt=\"merged embed cos {_lmid}\" "
        f"style=\"max-width:100%;height:auto;border:1px solid {_blm};border-radius:8px;display:block\"/>"
        f"<div style=\"margin-top:1em;max-height:min(70vh,720px);overflow:auto;border:1px solid {_blm};"
        f"border-radius:8px;padding:0.35rem 0.5rem;background:#1a1d23\">"
        # f"<h4 style=\"font-size:0.95rem;margin:0 0 0.5em;color:{_clm}\">Merge cells (top-{_EMB_COS_TOPK})</h4>"
        # f"{_table_html}"
        f"</div>"
        f"</section>"
    )

_lmh_html = f"""<div style="font-family:system-ui,Segoe UI,sans-serif;font-size:14px;line-height:1.5;color:{_clm}">
<h2 style="font-size:1.1rem;margin:0 0 0.75em">Merged tokens · top-1 Blues overlay + top-{_EMB_COS_TOPK} table — ``embed_tokens`` (image_id 0 … 9)</h2>
<p style="margin:0 0 1em;color:{_mlm};font-size:13px">
Per image: merge cells tinted by top-1 cos (2–98%ile on that grid); label = #1 token with cosine below. One PNG per <code>image_id</code> under <code>data/images/pope_vit/</code>.
</p>
{"".join(_lmh_sections)}
</div>"""

print(STONESOUP_RENDER_HTML, _lmh_html, sep="", flush=True)

# %% VQA — POPE image + custom question (streaming answer, plain text)

# Needs **Load** multimodal model, **Load POPE subset**, **Save POPE images** (``POPE_IMAGE_RECORDS``).
# Edit ``_VQA_IMG_ID`` and ``_VQA_QUESTION`` for any language / task. Full reply (after the run) is
# ``POPE_VQA_LAST_REPLY``.
#
# **Plain text:** default Stonesoup stdout (no render hint). Reply is printed token-by-token (``flush=True``).

from threading import Thread

from transformers import TextIteratorStreamer

_VQA_IMG_ID = 4
_VQA_QUESTION = "Is it a taxi?"
_VQA_MAX_NEW = 384

if _VQA_IMG_ID >= len(POPE_ROWS):
    raise IndexError(
        f"VQA cell: need POPE_ROWS index {_VQA_IMG_ID}, but len(POPE_ROWS)={len(POPE_ROWS)} "
        "(raise POPE_LIMIT / re-run POPE cells)."
    )

_vqa_pil = pope_row_to_pil(POPE_ROWS[_VQA_IMG_ID])
_vqa_inputs = build_pope_vision_inputs(processor, _vqa_pil, _VQA_QUESTION, DEVICE)

_vqa_fname = POPE_IMAGE_RECORDS[_VQA_IMG_ID]["filename"]
_vqa_url = f"http://127.0.0.1:8765/data/image/pope/{_vqa_fname}"

_vqa_streamer = TextIteratorStreamer(
    processor.tokenizer,
    skip_prompt=True,
    skip_special_tokens=True,
)
_vqa_gen_kw = {
    **_vqa_inputs,
    "max_new_tokens": _VQA_MAX_NEW,
    "do_sample": False,
    "streamer": _vqa_streamer,
}
_vqa_eos = getattr(processor.tokenizer, "eos_token_id", None)
if _vqa_eos is not None:
    _vqa_gen_kw["eos_token_id"] = _vqa_eos


def _vqa_run_generate() -> None:
    with torch.inference_mode():
        model.generate(**_vqa_gen_kw)


_vqa_thread = Thread(target=_vqa_run_generate, daemon=True)
_vqa_thread.start()

print(f"VQA · POPE image_id={_VQA_IMG_ID}", flush=True)
print(f"MODEL_ID={MODEL_ID}  max_new_tokens={_VQA_MAX_NEW}  greedy  streamed", flush=True)
print(f"Image (open in browser): {_vqa_url}", flush=True)
_vqa_disk = POPE_IMAGE_DIR / _vqa_fname
print(f"Image (path): {_vqa_disk.relative_to(REPO_ROOT)}", flush=True)
print(f"Question: {_VQA_QUESTION}", flush=True)
print("Answer: ", end="", flush=True)

_vqa_parts: list[str] = []
try:
    for _vqa_chunk in _vqa_streamer:
        print(_vqa_chunk, end="", flush=True)
        _vqa_parts.append(_vqa_chunk)
finally:
    _vqa_thread.join(timeout=600.0)

print(flush=True)
POPE_VQA_LAST_REPLY = "".join(_vqa_parts).strip()

# %% One merge token — pick (image_id, row, col), inject merge vision into one-slot prompt + `generate`

# Needs **Load** multimodal model, **Load POPE subset**, **Save POPE images** (optional; only for URL in stdout).
#
# **Goal:** Choose the LM **merge** cell by ``_ST_IMG_ID`` and **1-based** ``(col, row)``. Run the vision tower on
# the **full** POPE image, take that row’s **pooler** vector (length ``text_config.hidden_size`` per checkpoint) plus
# matching **DeepStack** rows. Build a **decoy** prompt: one ``<|image_pad|>`` via a neutral **32×32** template and
# ``qwen_vl_images_kwargs_one_llm_image_token``. Then:
#
# - ``register_forward_pre_hook`` on ``model.model.language_model.layers[0]`` overwrites the LM hidden state at the
#   image-token position with the **saved pooler** row (chosen merge, not the template pixels).
# - ``qwen_vl_patch_get_image_deepstack_only`` swaps **only** ``deepstack_features`` on the **Qwen3-VL** path (HF passes
#   them into ``language_model`` as ``deepstack_visual_embeds`` after early decoder layers). **Qwen3.5** has no such
#   tensors → extracted list is empty and the context manager is a no-op; only the layer-0 hook replaces the main slot.
#
# **Stonesoup:** HTML + overlay PNGs under ``data/images/pope_vit/``. **Outputs:** ``POPE_SINGLE_MERGE_TOKEN_VQA_REPLY``,
# ``POPE_SINGLE_MERGE_LM_VEC_CPU``, ``POPE_SINGLE_MERGE_LM_DIM``.
#
# **Requires** ``POPE_LOAD_BOTH_MODELS = True`` in **Config** (and re-run Config + **Load multimodal**) if you only
# loaded Qwen3.5; otherwise ``pope_set_active_model("qwen3_vl")`` has nothing to swap to.
pope_set_active_model("qwen3_vl")

_ST_IMG_ID = 1
_ST_GRID_COL_1B = 1
_ST_GRID_ROW_1B = 3
_ST_TEXT = "Describe the image in one very short English sentence (about five words or fewer). No list, no paragraph."
_ST_MAX_NEW = 64

if _ST_IMG_ID >= len(POPE_ROWS):
    raise IndexError(
        f"single-token VQA: need POPE_ROWS index {_ST_IMG_ID}, len={len(POPE_ROWS)}"
    )

_st_pil_full = pope_row_to_pil(POPE_ROWS[_ST_IMG_ID])
_st_in_ref = build_pope_vision_inputs(processor, _st_pil_full, POPE_VISION_USER_TEXT, DEVICE)
_st_g = _st_in_ref["image_grid_thw"][0].detach().cpu()
_st_t, _st_gh, _st_gw = (int(x) for x in _st_g.tolist())
if _st_t != 1:
    raise NotImplementedError("single-token VQA: expected one temporal slice (T=1).")
_st_ms = int(processor.image_processor.merge_size)
_st_ps = int(processor.image_processor.patch_size)
_st_llm_w = _st_gw // _st_ms
_st_llm_h = _st_gh // _st_ms
_st_w0 = _ST_GRID_COL_1B - 1
_st_h0 = _ST_GRID_ROW_1B - 1
if not (0 <= _st_w0 < _st_llm_w and 0 <= _st_h0 < _st_llm_h):
    raise IndexError(
        f"merge grid ({_ST_GRID_COL_1B},{_ST_GRID_ROW_1B}) (1-based) out of range "
        f"for LLM grid {_st_llm_h}×{_st_llm_w} on image_id={_ST_IMG_ID}"
    )
_st_merge_idx = _st_h0 * _st_llm_w + _st_w0
_st_res_wh = (_st_gw * _st_ps, _st_gh * _st_ps)
_Wst, _Hst = _st_pil_full.size
_xr0, _yr0, _xr1, _yr1 = llm_merge_index_to_resized_box(
    _st_merge_idx,
    _st_g,
    patch_size=_st_ps,
    merge_size=_st_ms,
)
_xo0, _yo0, _xo1, _yo1 = resized_box_to_original(
    (_xr0, _yr0, _xr1, _yr1),
    resized_wh=_st_res_wh,
    original_wh=(_Wst, _Hst),
)
_st_crop = _st_pil_full.crop((_xo0, _yo0, _xo1, _yo1))
_st_side = _st_ps * _st_ms
_st_crop_1tok = _st_crop.resize((_st_side, _st_side), Image.Resampling.BILINEAR)

_st_pool_row, _st_ds_rows = qwen_vl_extract_merge_vision_rows(
    model,
    _st_in_ref["pixel_values"],
    _st_in_ref["image_grid_thw"],
    _st_merge_idx,
)
POPE_SINGLE_MERGE_LM_VEC_CPU = _st_pool_row.detach().float().cpu()
POPE_SINGLE_MERGE_LM_DIM = int(_st_pool_row.shape[0])
POPE_SINGLE_MERGE_N_DEEPSTACK = len(_st_ds_rows)

# Keep pooler on one device for bookkeeping; the layer-0 hook casts to ``hidden_states`` device/dtype (``device_map``).
_st_mdl_dtype = next(model.parameters()).dtype
_st_chosen_lm = _st_pool_row.to(device=DEVICE, dtype=_st_mdl_dtype)

_st_tmpl_pil = Image.new("RGB", (_st_side, _st_side), (96, 96, 110))
_st_img_kw = qwen_vl_images_kwargs_one_llm_image_token(processor)
_st_inputs = build_pope_vision_inputs(
    processor, _st_tmpl_pil, _ST_TEXT, DEVICE, images_kwargs=_st_img_kw
)
_st_n_img = int((_st_inputs.input_ids == model.config.image_token_id).sum().item())
if _st_n_img != 1:
    raise RuntimeError(
        f"expected 1 <|image_pad|>/image_token_id with one-merge images_kwargs, got {_st_n_img} "
        f"(patch_size={_st_ps}, merge_size={_st_ms}, images_kwargs={_st_img_kw!r})"
    )

_st_tok_positions = torch.nonzero(
    _st_inputs.input_ids[0] == model.config.image_token_id, as_tuple=True
)[0]
if _st_tok_positions.numel() != 1:
    raise RuntimeError(f"expected exactly one image token index, got {_st_tok_positions.tolist()}")
_st_tok_ix = int(_st_tok_positions[0].item())


def _st_layer0_pre_hook(_mod, inputs):
    hs = inputs[0]
    # Prefill: ``hidden_states`` is the full prompt (``seq_len`` = prompt length). Decode steps only pass new
    # tokens (often ``seq_len == 1`` with KV cache), so the image-token index is out of range — skip those.
    if hs.shape[1] <= _st_tok_ix:
        return None
    hs2 = hs.clone()
    # Match ``hidden_states`` placement when ``device_map="auto"`` shards the LM across devices.
    hs2[0, _st_tok_ix, :] = _st_chosen_lm.to(device=hs.device, dtype=hs.dtype)
    return (hs2,) + inputs[1:]


_st_hook_handle = model.model.language_model.layers[0].register_forward_pre_hook(_st_layer0_pre_hook)
try:
    with qwen_vl_patch_get_image_deepstack_only(model, _st_ds_rows):
        with torch.inference_mode():
            _st_gen = model.generate(
                **_st_inputs,
                max_new_tokens=_ST_MAX_NEW,
                do_sample=False,
            )
finally:
    _st_hook_handle.remove()

_one_merge_run_png = EXP_DIR / "pope_one_merge_cell_preview.png"
_st_crop_1tok.save(_one_merge_run_png)
_st_tmpl_path = EXP_DIR / "pope_one_merge_template_decoy.png"
_st_tmpl_pil.save(_st_tmpl_path)

POPE_VIT_PLOT_DIR.mkdir(parents=True, exist_ok=True)
_st_ov_name = f"pope_one_merge_window_i{_ST_IMG_ID}_c{_ST_GRID_COL_1B}_r{_ST_GRID_ROW_1B}.png"
_st_overlay_path = POPE_VIT_PLOT_DIR / _st_ov_name
_st_base_rgba = _st_pil_full.convert("RGBA")
_st_hi_layer = Image.new("RGBA", _st_base_rgba.size, (0, 0, 0, 0))
_st_pen = max(2, min(_Wst, _Hst) // 200)
ImageDraw.Draw(_st_hi_layer).rectangle(
    (_xo0, _yo0, _xo1, _yo1),
    outline=(0, 255, 168, 255),
    fill=(0, 255, 140, 52),
    width=_st_pen,
)
_st_full_marked = Image.alpha_composite(_st_base_rgba, _st_hi_layer).convert("RGB")
_st_full_marked.save(_st_overlay_path)

_st_disp_scale = 8
_st_disp_name = f"pope_one_merge_32up_i{_ST_IMG_ID}_c{_ST_GRID_COL_1B}_r{_ST_GRID_ROW_1B}.png"
_st_disp_path = POPE_VIT_PLOT_DIR / _st_disp_name
_st_crop_1tok.resize(
    (_st_side * _st_disp_scale, _st_side * _st_disp_scale),
    Image.Resampling.NEAREST,
).save(_st_disp_path)

_st_in_len = _st_inputs.input_ids.shape[1]
_st_new = _st_gen[0, _st_in_len:]
POPE_SINGLE_MERGE_TOKEN_VQA_REPLY = processor.batch_decode(
    _st_new.unsqueeze(0),
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)[0]

_st_http = "http://127.0.0.1:8765"
_st_url_ov = f"{_st_http}/data/image/pope_vit/{_st_ov_name}?cb={int(_st_overlay_path.stat().st_mtime_ns)}"
_st_url_up = f"{_st_http}/data/image/pope_vit/{_st_disp_name}?cb={int(_st_disp_path.stat().st_mtime_ns)}"
_st_tc = "#e8eaed"
_st_tm = "#9aa0a6"
_st_tb = "#3c4043"
_st_esc_prompt = html.escape(_ST_TEXT)
_st_esc_ans = html.escape(POPE_SINGLE_MERGE_TOKEN_VQA_REPLY)
_st_rel_ov = html.escape(str(_st_overlay_path.relative_to(REPO_ROOT)))
_st_rel_up = html.escape(str(_st_disp_path.relative_to(REPO_ROOT)))
_st_rel_cell = html.escape(str(_one_merge_run_png.relative_to(REPO_ROOT)))
_st_rel_tmpl = html.escape(str(_st_tmpl_path.relative_to(REPO_ROOT)))
_st_rel_refpng = html.escape(str(ONE_MERGE_TOKEN_CANVAS_PNG.relative_to(REPO_ROOT)))

_st_pope_block = ""
try:
    _st_fn = POPE_IMAGE_RECORDS[_ST_IMG_ID]["filename"]
    _st_pope_u = f"{_st_http}/data/image/pope/{html.escape(_st_fn)}"
    _st_pope_block = (
        f'<p style="margin:0.35em 0 0;font-size:13px;color:{_st_tm}">'
        f'POPE file: <a href="{_st_pope_u}" style="color:#8ab4f8">open original</a> · '
        f"<code>{html.escape(_st_fn)}</code></p>"
    )
except NameError:
    pass

_st_one_merge_html = f"""<div style="font-family:system-ui,Segoe UI,sans-serif;font-size:14px;line-height:1.5;color:{_st_tc}">
<h2 style="font-size:1.1rem;margin:0 0 0.5em">Single merge-token VQA — injected pooler (+ Qwen3-VL DeepStack) · one <code>&lt;|image_pad|&gt;</code></h2>
<p style="margin:0 0 0.75em;font-size:13px;color:{_st_tm}">
<code>image_id={_ST_IMG_ID}</code> · LLM merge grid (1-based col, row)=({_ST_GRID_COL_1B},{_ST_GRID_ROW_1B})
· flat <code>merge_idx={_st_merge_idx}</code> · ViT fine grid {_st_gh}×{_st_gw}
· <code>MODEL_ID</code>={html.escape(MODEL_ID)} · <code>max_new_tokens={_ST_MAX_NEW}</code>
· <code>n_image_pad={_st_n_img}</code> · pooler dim <code>{POPE_SINGLE_MERGE_LM_DIM}</code>
· DeepStack tensors <code>{POPE_SINGLE_MERGE_N_DEEPSTACK}</code><br/>
<strong>Injection:</strong> forward pre-hook on <code>language_model.layers[0]</code> (token index {_st_tok_ix}) overwrites the image slot with the pooler row from the <strong>full-image</strong> forward;
<code>get_image_features</code> DeepStack patch applies to <strong>Qwen3-VL</strong> only (Qwen3.5: 0 DeepStack tensors, no-op).<br/>
Selected merge in original pixel space: {_xo0},{_yo0} … {_xo1},{_yo1} on {_Wst}×{_Hst}.
</p>
<div style="display:flex;flex-wrap:wrap;gap:1.25rem;align-items:flex-start;margin-bottom:1rem">
<figure style="margin:0;max-width:min(900px,100%)">
<img src="{html.escape(_st_url_ov)}" alt="POPE image with merge window"
 style="max-width:100%;height:auto;border:1px solid {_st_tb};border-radius:8px;display:block"/>
<figcaption style="font-size:12px;color:{_st_tm};margin-top:0.35rem;max-width:56rem">
<strong style="color:{_st_tc}">Selected window → one image token.</strong> The translucent box is the merge cell on the
full image (same region as <code>llm_merge_index_to_resized_box</code> mapped to original coordinates).
<span style="display:block;margin-top:0.25em;font-size:12px"><code>{_st_rel_ov}</code></span>
</figcaption>
</figure>
<figure style="margin:0">
<img src="{html.escape(_st_url_up)}" alt="32×32 vision input upscaled"
 style="width:min(256px,40vw);height:auto;border:1px solid {_st_tb};border-radius:8px;display:block;image-rendering:pixelated"/>
<figcaption style="font-size:12px;color:{_st_tm};margin-top:0.35rem;max-width:18rem">
<strong style="color:{_st_tc}">Merge cell (preview)</strong> — {_st_side}×{_st_side} BILINEAR from the boxed crop; for reference only.
<strong>Generation</strong> uses a neutral decoy template (<code>{_st_rel_tmpl}</code>) + injected vectors. ×{_st_disp_scale} nearest for visibility.
<span style="display:block;margin-top:0.25em;font-size:12px"><code>{_st_rel_up}</code> · cell <code>{_st_rel_cell}</code> · ref canvas <code>{_st_rel_refpng}</code></span>
</figcaption>
</figure>
</div>
{_st_pope_block}
<p style="margin:0.75em 0 0.25em;font-size:13px;color:{_st_tm}"><strong style="color:{_st_tc}">User text</strong></p>
<p style="margin:0 0 0.75em;padding:0.5em 0.65em;background:#1a1d23;border:1px solid {_st_tb};border-radius:8px;font-size:13px">{_st_esc_prompt}</p>
<p style="margin:0 0 0.25em;font-size:13px;color:{_st_tm}"><strong style="color:{_st_tc}">Model reply</strong></p>
<p style="margin:0;padding:0.5em 0.65em;background:#1a1d23;border:1px solid {_st_tb};border-radius:8px;font-size:13px;white-space:pre-wrap">{_st_esc_ans}</p>
</div>"""

print(STONESOUP_RENDER_HTML, _st_one_merge_html, sep="", flush=True)
