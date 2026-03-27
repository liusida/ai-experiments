"""Minimal Qwen2.5-VL chat demo (image + text) via Hugging Face Transformers.

Requires a recent **transformers** with ``qwen2_5_vl`` support; if you see
``KeyError: 'qwen2_5_vl'``, install from source per the model card, e.g.:

  ``pip install git+https://github.com/huggingface/transformers accelerate``

Also: ``uv pip install qwen-vl-utils==0.0.8`` (use plain package on **Linux aarch64**;
``qwen-vl-utils[decord]`` has no usable ``decord`` wheels there—video falls back to torchvision).

First run downloads the checkpoint for ``MODEL_ID`` (~several GiB via Hugging Face Hub).

**Stonesoup:** Watch this file and run cells top to bottom (kernel keeps globals).
Re-run **Load model & processor** only if you change ``MODEL_ID`` or restart the kernel.

**Terminal:** ``uv run python experiments/2026-03-25-VLM/explore-qwen-vlm.py``
"""

from __future__ import annotations

# %% Imports & constants

import base64
import io
import json
from pathlib import Path

import torch
from matplotlib import colormaps
from PIL import Image, ImageDraw, ImageFont
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# Repo root when this file lives under experiments/2026-…/
REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
DEMO_IMAGE_URL = (
    "file:///home/liusida/ai-experiments/data/images/dog-mid.png"
    # "https://hips.hearstapps.com/clv.h-cdn.co/assets/16/18/gettyimages-586890581.jpg?crop=0.668xw:1.00xh;0.219xw,0"
)
TEXT = """dog"""

# How many patches to show for **closest** / **farthest** (max cosine to any prompt text token)
TOP_K_IMAGE_PATCHES = 12
# Outline colors (closest = similar-to-text; farthest = unlike prompt text in embedding angle)
PATCH_COLOR_CLOSEST = "#0f766e"  # teal
PATCH_COLOR_FARTHEST = "#b91c1c"  # red
OUT_DIR = Path(__file__).resolve().parent / "output"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16 if DEVICE.type == "cuda" else torch.float32

print("REPO_ROOT:", REPO_ROOT)
print("MODEL_ID:", MODEL_ID)
print("DEVICE:", DEVICE)

# %% Load model & processor

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype="auto" if DEVICE.type == "cuda" else DTYPE,
    device_map="auto" if DEVICE.type == "cuda" else None,
)
if DEVICE.type != "cuda":
    model = model.to(DEVICE)

processor = AutoProcessor.from_pretrained(MODEL_ID)
print("Loaded:", MODEL_ID)

# %% Single-image inference (demo URL)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": DEMO_IMAGE_URL},
            {"type": "text", "text": TEXT},
        ],
    }
]

text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to(DEVICE)

generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :]
    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)
print(output_text)

# %% Image vs text vectors at the LLM input (same space as ``inputs_embeds``)

# Vision tower + patch merger project patches into ``text_config.hidden_size``; those
# vectors replace placeholder ``image_token_id`` rows before the language-model stack.
# Text vectors here are plain ``embed_tokens`` rows for all non-image / non-video ids.


@torch.inference_mode()
def capture_image_and_text_input_vectors():
    pv = inputs["pixel_values"]
    thw = inputs["image_grid_thw"]
    vision_out = model.get_image_features(
        pixel_values=pv, image_grid_thw=thw, return_dict=True
    )
    # One row per image *spatial* token after merge (matches count of image placeholders)
    image_vecs = torch.cat(vision_out.pooler_output, dim=0)

    ids = inputs["input_ids"]
    tok_emb = model.get_input_embeddings()(ids)
    ii, vi = model.config.image_token_id, model.config.video_token_id
    text_mask = (ids != ii) & (ids != vi)
    text_vecs = tok_emb[text_mask]
    return image_vecs, text_vecs


def geometry_summary(img: torch.Tensor, txt: torch.Tensor, n_pairs: int = 8192) -> dict:
    """Compare normalized rows: sampled mean cosine (intra / cross) and centroids."""
    d = img.shape[-1]

    def unit_rows(x: torch.Tensor) -> torch.Tensor:
        return x / (x.norm(dim=-1, keepdim=True).clamp(min=1e-8))

    img_u = unit_rows(img.float())
    txt_u = unit_rows(txt.float())
    ni, nt = img_u.shape[0], txt_u.shape[0]
    gen = torch.Generator(device=img_u.device)
    gen.manual_seed(0)

    def mean_sampled_cos(a: torch.Tensor, b: torch.Tensor, n: int) -> float:
        n = min(n, max(1, a.size(0) * b.size(0)))
        i = torch.randint(0, a.size(0), (n,), generator=gen, device=a.device)
        j = torch.randint(0, b.size(0), (n,), generator=gen, device=b.device)
        if a.data_ptr() == b.data_ptr():
            same = i == j
            if same.any():
                j = torch.where(same, (j + 1) % b.size(0), j)
        return (a[i] * b[j]).sum(dim=-1).mean().item()

    ci = img_u.mean(dim=0)
    ct = txt_u.mean(dim=0)
    cent_cos = (ci @ ct / (ci.norm() * ct.norm().clamp(min=1e-8))).item()

    return {
        "hidden_dim": d,
        "num_image_tokens": ni,
        "num_text_tokens": nt,
        "mean_row_l2_image": img.float().norm(dim=-1).mean().item(),
        "mean_row_l2_text": txt.float().norm(dim=-1).mean().item(),
        "sampled_mean_cos_image_image": mean_sampled_cos(img_u, img_u, n_pairs),
        "sampled_mean_cos_text_text": mean_sampled_cos(txt_u, txt_u, n_pairs),
        "sampled_mean_cos_image_text": mean_sampled_cos(img_u, txt_u, n_pairs),
        "centroid_cosine_image_vs_text": cent_cos,
        "centroid_l2_distance": (img_u.mean(0) - txt_u.mean(0)).norm().item(),
    }


image_token_vectors, text_token_vectors = capture_image_and_text_input_vectors()
geom = geometry_summary(
    image_token_vectors.to("cpu", dtype=torch.float32),
    text_token_vectors.to("cpu", dtype=torch.float32),
)
print("Geometry (normalized rows, same LM hidden size):")
for key, val in geom.items():
    print(f"  {key}: {val}")
# ``image_token_vectors`` / ``text_token_vectors`` are available for further analysis


def expand_image_pads_like_processor(
    prompt: str, processor: AutoProcessor, image_grid_thw: torch.Tensor
) -> str:
    """Mirror ``Qwen2_5_VLProcessor`` expansion of ``<|image_pad|>`` before tokenization."""
    merge_length = processor.image_processor.merge_size**2
    out = prompt
    index = 0
    while processor.image_token in out:
        n = int(image_grid_thw[index].prod().item() // merge_length)
        out = out.replace(processor.image_token, "<|placeholder|>" * n, 1)
        index += 1
    return out.replace("<|placeholder|>", processor.image_token)


def line_char_spans(expanded_prompt: str, lines: list[str]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    search_from = 0
    for ln in lines:
        start = expanded_prompt.find(ln, search_from)
        if start < 0:
            raise ValueError(
                f"Line not found in expanded prompt (from search_from={search_from}): {ln[:72]!r}…"
            )
        end = start + len(ln)
        spans.append((start, end))
        search_from = end
    return spans


def per_line_token_masks(
    *,
    expanded_prompt: str,
    lines: list[str],
    input_ids_1d: torch.Tensor,
    processor: AutoProcessor,
    image_token_id: int,
    video_token_id: int,
) -> list[torch.Tensor]:
    """Bool mask (seq,) per line: text tokens whose char offsets intersect that line."""
    spans = line_char_spans(expanded_prompt, lines)
    enc = processor.tokenizer(
        [expanded_prompt],
        padding=True,
        return_tensors="pt",
        return_offsets_mapping=True,
    )
    if not torch.equal(enc["input_ids"][0].cpu(), input_ids_1d.cpu()):
        raise RuntimeError(
            "Tokenizer output != inputs['input_ids']. Check padding/truncation matches processor()."
        )
    offsets = enc["offset_mapping"][0].tolist()
    not_mm = (input_ids_1d != image_token_id) & (input_ids_1d != video_token_id)
    masks: list[torch.Tensor] = []
    for ls, le in spans:
        m = torch.zeros_like(input_ids_1d, dtype=torch.bool)
        for t, (cs, ce) in enumerate(offsets):
            if cs == ce and cs == 0:
                continue
            if cs < le and ce > ls:
                m[t] = True
        masks.append(m & not_mm)
    return masks


def unit_rows(x: torch.Tensor) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True).clamp(min=1e-8))


def image_vs_text_cos_stats(
    image_vecs: torch.Tensor,
    text_vecs: torch.Tensor,
    n_pairs: int = 8192,
) -> dict:
    iu, tu = unit_rows(image_vecs.float()), unit_rows(text_vecs.float())
    if tu.numel() == 0:
        return {"num_text_tokens": 0, "mean_cos_sampled": float("nan"), "centroid_cosine": float("nan")}
    ci, ct = iu.mean(dim=0), tu.mean(dim=0)
    cent = (ci @ ct / (ci.norm() * ct.norm().clamp(min=1e-8))).item()
    gen = torch.Generator(device=iu.device)
    gen.manual_seed(0)
    n = min(n_pairs, max(1, iu.size(0) * tu.size(0)))
    ai = torch.randint(0, iu.size(0), (n,), generator=gen, device=iu.device)
    bj = torch.randint(0, tu.size(0), (n,), generator=gen, device=iu.device)
    mean_cos = (iu[ai] * tu[bj]).sum(dim=-1).mean().item()
    return {
        "num_text_tokens": int(tu.size(0)),
        "mean_cos_sampled": mean_cos,
        "centroid_cosine": cent,
    }


TEXT_LINES = [ln for ln in TEXT.strip().splitlines() if ln.strip()]
_expanded = expand_image_pads_like_processor(text, processor, inputs["image_grid_thw"])
_line_masks = per_line_token_masks(
    expanded_prompt=_expanded,
    lines=TEXT_LINES,
    input_ids_1d=inputs["input_ids"][0],
    processor=processor,
    image_token_id=model.config.image_token_id,
    video_token_id=model.config.video_token_id,
)

_tok_emb = model.get_input_embeddings()(inputs["input_ids"])
_img = image_token_vectors

print("Cosine similarity: image tokens vs text tokens per line (unit-normalized rows)")
for line_idx, (line_text, lm) in enumerate(zip(TEXT_LINES, _line_masks), start=1):
    line_vecs = _tok_emb[0, lm]
    stats = image_vs_text_cos_stats(
        _img,
        line_vecs.to(_img.device),
    )
    preview = line_text if len(line_text) <= 72 else line_text[:69] + "…"
    print(f"  Line {line_idx} ({preview!r}):")
    for k, v in stats.items():
        print(f"    {k}: {v}")

# %% Closest & farthest image patches vs prompt text (cosine in LLM input space)

# Map merged LLM image token index → axis-aligned box on the **resized** model canvas
# (``resized_width = grid_w * patch_size``). ``window_index`` shuffling is undone in
# ``pooler_output``, so index order matches ``(t, llm_h, llm_w)`` row-major flattening.


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
    rem = merge_index % llm_hw
    h_llm = rem // llm_w
    w_llm = rem % llm_w
    y0 = h_llm * merge_size * patch_size
    x0 = w_llm * merge_size * patch_size
    y1 = (h_llm + 1) * merge_size * patch_size
    x1 = (w_llm + 1) * merge_size * patch_size
    return x0, y0, x1, y1


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


def max_cosine_per_image_token(
    image_vecs: torch.Tensor, text_vecs: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per image token: max cosine to any prompt text token (unit-normalized rows)."""
    iu = unit_rows(image_vecs.float())
    tu = unit_rows(text_vecs.float())
    if tu.size(0) == 0:
        raise ValueError("No text token vectors in prompt for comparison")
    sims = iu @ tu.T
    return sims.max(dim=1).values, sims


_g_thw = inputs["image_grid_thw"][0]
_ps = processor.image_processor.patch_size
_ms = processor.image_processor.merge_size
_res_wh = (int(_g_thw[2].item()) * _ps, int(_g_thw[1].item()) * _ps)

_prompt_mask = _line_masks[0] if len(_line_masks) == 1 else torch.zeros_like(inputs["input_ids"][0], dtype=torch.bool)
if len(_line_masks) != 1:
    for m in _line_masks:
        _prompt_mask |= m

_ids_row = inputs["input_ids"][0]
_tkz = processor.tokenizer
print("Prompt-text tokens (mask from TEXT lines; used for patch cosines), sequence order:")
for pos in _prompt_mask.nonzero(as_tuple=True)[0].tolist():
    tid = int(_ids_row[pos].item())
    piece = _tkz.convert_ids_to_tokens([tid])[0]
    print(f"  pos={pos:4d}  token_id={tid:6d}  piece={piece!r}")
print(f"  (count={int(_prompt_mask.sum().item())})")

_prompt_vecs = _tok_emb[0, _prompt_mask]
_scores, _sims = max_cosine_per_image_token(_img, _prompt_vecs.to(_img.device))
_k = min(TOP_K_IMAGE_PATCHES, _scores.size(0))
_topv, _topi = torch.topk(_scores, k=_k, largest=True)
_botv, _boti = torch.topk(_scores, k=_k, largest=False)

print(f"Closest {_k} image tokens (max cosine to any prompt text token):")
for rank, (val, idx) in enumerate(zip(_topv.tolist(), _topi.tolist()), start=1):
    print(f"  C{rank}: merge_token_index={idx}, max_cos={val:.4f}")
print(f"Farthest {_k} image tokens (lowest max cosine):")
for rank, (val, idx) in enumerate(zip(_botv.tolist(), _boti.tolist()), start=1):
    print(f"  F{rank}: merge_token_index={idx}, max_cos={val:.4f}")

_canvas = image_inputs[0].convert("RGB")
_orig_wh = _canvas.size  # (W, H)
_draw = ImageDraw.Draw(_canvas)
_pen = max(2, min(_orig_wh) // 256)


def _draw_patch_boxes(
    merge_indices: list[int],
    label_prefix: str,
    outline: str,
) -> None:
    for rank, merge_i in enumerate(merge_indices, start=1):
        box_r = llm_merge_index_to_resized_box(
            merge_i, _g_thw.to("cpu"), patch_size=_ps, merge_size=_ms
        )
        box_o = resized_box_to_original(
            box_r, resized_wh=_res_wh, original_wh=_orig_wh
        )
        _draw.rectangle(box_o, outline=outline, width=_pen)
        _draw.text((box_o[0] + 2, box_o[1] + 2), f"{label_prefix}{rank}", fill=outline)


# Draw farthest first so closest rectangles / labels stay on top
_draw_patch_boxes(_boti.tolist(), "F", PATCH_COLOR_FARTHEST)
_draw_patch_boxes(_topi.tolist(), "C", PATCH_COLOR_CLOSEST)

OUT_DIR.mkdir(parents=True, exist_ok=True)
_closest_out = OUT_DIR / "closest_and_farthest_patches_to_prompt_text.png"
_canvas.save(_closest_out)
print("Saved:", _closest_out.resolve())

# %% Cosine similarity: patch 0 vs every patch (number inside each box)

# Merge-token index 0 is the first spatial cell in row-major ``(t, llm_h, llm_w)`` order
# (same ordering as ``image_token_vectors`` / ``pooler_output``).
REF_MERGE_INDEX = 0
# Semi-transparent fill: matplotlib ``Blues`` indexed by cosine in [0, 1] (clamped)
_COS_HEAT_ALPHA = 150
_COS_LABEL_FONT_PX = max(7, min(image_inputs[0].size) // 56)  # small labels
_blues_cmap = colormaps["Blues"]


def _cosine_to_blues_rgba(c: float, alpha: int) -> tuple[int, int, int, int]:
    t = max(0.0, min(1.0, float(c)))
    r, g, b, _ = _blues_cmap(t)
    return int(r * 255), int(g * 255), int(b * 255), alpha


_vref = unit_rows(_img[REF_MERGE_INDEX : REF_MERGE_INDEX + 1].float())
_iu_all = unit_rows(_img.float())
_patch_cos_to_ref = (_iu_all * _vref).sum(dim=-1).cpu()
_cmin = float(_patch_cos_to_ref.min().item())
_cmax = float(_patch_cos_to_ref.max().item())


_cos_canvas = image_inputs[0].convert("RGBA")
_cos_orig_wh = _cos_canvas.size
_heat = Image.new("RGBA", _cos_orig_wh, (0, 0, 0, 0))
_heat_draw = ImageDraw.Draw(_heat)

try:
    _cos_font = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", _COS_LABEL_FONT_PX
    )
except OSError:
    _cos_font = ImageFont.load_default()

_n_merged = _img.size(0)
_patch_boxes: list[tuple[tuple[int, int, int, int], float, str]] = []
for merge_i in range(_n_merged):
    box_r = llm_merge_index_to_resized_box(
        merge_i, _g_thw.to("cpu"), patch_size=_ps, merge_size=_ms
    )
    box_o = resized_box_to_original(
        box_r, resized_wh=_res_wh, original_wh=_cos_orig_wh
    )
    c = float(_patch_cos_to_ref[merge_i].item())
    label = f"{c:.2f}*" if merge_i == REF_MERGE_INDEX else f"{c:.2f}"
    _heat_draw.rectangle(box_o, fill=_cosine_to_blues_rgba(c, _COS_HEAT_ALPHA))
    _patch_boxes.append((box_o, c, label))

_cos_rgba = Image.alpha_composite(_cos_canvas, _heat)
_cos_canvas = _cos_rgba.convert("RGB")
_label_draw = ImageDraw.Draw(_cos_canvas)

for box_o, c, label in _patch_boxes:
    t = max(0.0, min(1.0, c))
    _br, _bg, _bb, _ = _blues_cmap(t)
    _lum = 0.299 * _br + 0.587 * _bg + 0.114 * _bb
    # Light Blues → dark text + light halo; dark Blues → light text + dark halo
    if _lum > 0.52:
        _txt_fill, _stroke_fill = (17, 24, 39), (255, 255, 255)
    else:
        _txt_fill, _stroke_fill = (248, 250, 252), (0, 0, 0)
    _bbn = _label_draw.textbbox((0, 0), label, font=_cos_font)
    tw, th = _bbn[2] - _bbn[0], _bbn[3] - _bbn[1]
    tx = box_o[0] + max(0, (box_o[2] - box_o[0] - tw) // 2)
    ty = box_o[1] + max(0, (box_o[3] - box_o[1] - th) // 2)
    try:
        _label_draw.text(
            (tx, ty),
            label,
            fill=_txt_fill,
            font=_cos_font,
            stroke_width=2,
            stroke_fill=_stroke_fill,
        )
    except TypeError:
        _label_draw.text((tx, ty), label, fill=_txt_fill, font=_cos_font)

OUT_DIR.mkdir(parents=True, exist_ok=True)
_cos_vs0_out = OUT_DIR / "patch_cosine_vs_merge0.png"
_cos_canvas.save(_cos_vs0_out)
print("Saved:", _cos_vs0_out.resolve())
print(
    f"ref_merge_index={REF_MERGE_INDEX}, n_patches={_n_merged}, "
    f"heat=matplotlib Blues on cosine in [0,1]; run min/max={_cmin:.4f}/{_cmax:.4f}"
)

# %% Interactive HTML: full patch×patch cosines, click a cell to set 1.00* reference

_iu_mat = unit_rows(_img.float())
_patch_cos_matrix = (_iu_mat @ _iu_mat.T).cpu().numpy()
_gt, _gh, _gw = (int(_g_thw[0].item()), int(_g_thw[1].item()), int(_g_thw[2].item()))
_llm_h, _llm_w = _gh // _ms, _gw // _ms
_html_n_rows = _gt * _llm_h
_html_n_cols = _llm_w
assert _html_n_rows * _html_n_cols == _n_merged, (
    f"grid {_html_n_rows}x{_html_n_cols} != n_merged {_n_merged}"
)
_blues_lut_js = [
    [int(round(x * 255)) for x in _blues_cmap(i / 255.0)[:3]] for i in range(256)
]
_buf_png = io.BytesIO()
image_inputs[0].convert("RGB").save(_buf_png, format="PNG")
_img_b64 = base64.b64encode(_buf_png.getvalue()).decode("ascii")
_HEAT_ALPHA = 0.3
# Patch labels in HTML: CSS ``clamp(min, preferred-vw, max)`` — lower all three for smaller text
_HTML_FONT_CLAMP_MIN = "4px"
_HTML_FONT_CLAMP_PREF = "1.2vw"
_HTML_FONT_CLAMP_MAX = "7px"

_html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Patch cosine explorer (Qwen VL merge tokens)</title>
  <style>
    body {{
      font-family: system-ui, sans-serif;
      margin: 1rem;
      background: #f1f5f9;
    }}
    h1 {{ font-size: 1.1rem; margin: 0 0 0.5rem 0; }}
    p.hint {{ margin: 0 0 0.75rem 0; color: #475569; font-size: 0.9rem; }}
    #refInfo {{ font-size: 0.85rem; color: #0f172a; margin-bottom: 0.5rem; }}
    .stage {{
      position: relative;
      display: inline-block;
      max-width: min(96vw, 900px);
      line-height: 0;
      border-radius: 6px;
      overflow: hidden;
      box-shadow: 0 4px 20px rgba(0,0,0,.12);
    }}
    .stage img {{
      display: block;
      width: 100%;
      height: auto;
      vertical-align: top;
    }}
    #grid {{
      position: absolute;
      inset: 0;
      display: grid;
      grid-template-columns: repeat({_html_n_cols}, 1fr);
      grid-template-rows: repeat({_html_n_rows}, 1fr);
    }}
    .cell {{
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: clamp({_HTML_FONT_CLAMP_MIN}, {_HTML_FONT_CLAMP_PREF}, {_HTML_FONT_CLAMP_MAX});
      font-weight: 600;
      border: none;
      box-sizing: border-box;
      user-select: none;
    }}
    .cell.ref {{
      box-shadow: inset 0 0 0 3px #ca8a04;
      z-index: 1;
    }}
  </style>
</head>
<body>
  <h1>Patch cosine similarity</h1>
  <p class="hint">Blues colormap on cosine in [0, 1]. Click any patch to use it as the reference (shown as 1.00*). Matrix is precomputed for all merge-token pairs.</p>
  <div id="refInfo"></div>
  <div class="stage">
    <img alt="input" src="data:image/png;base64,{_img_b64}" />
    <div id="grid"></div>
  </div>
  <script>
    const nRows = {_html_n_rows};
    const nCols = {_html_n_cols};
    const nPatches = {_n_merged};
    const heatA = {_HEAT_ALPHA};
    const matrix = {json.dumps(_patch_cos_matrix.tolist())};
    const bluesLUT = {json.dumps(_blues_lut_js)};

    function bluesRgb(t) {{
      t = Math.max(0, Math.min(1, t));
      const i = Math.min(255, Math.floor(t * 255));
      return bluesLUT[i];
    }}

    const gridEl = document.getElementById("grid");
    const refInfo = document.getElementById("refInfo");
    const cells = [];

    for (let r = 0; r < nRows; r++) {{
      for (let c = 0; c < nCols; c++) {{
        const idx = r * nCols + c;
        const div = document.createElement("div");
        div.className = "cell";
        div.dataset.mergeIndex = String(idx);
        div.addEventListener("click", (e) => {{
          e.stopPropagation();
          paint(Number(div.dataset.mergeIndex));
        }});
        gridEl.appendChild(div);
        cells.push(div);
      }}
    }}

    function paint(refIdx) {{
      for (let j = 0; j < nPatches; j++) {{
        const el = cells[j];
        const s = matrix[refIdx][j];
        const [br, bg, bb] = bluesRgb(s);
        el.style.background = `rgba(${{br}},${{bg}},${{bb}},${{heatA}})`;
        const lum = 0.299 * br / 255 + 0.587 * bg / 255 + 0.114 * bb / 255;
        el.style.color = lum > 0.52 ? "#111827" : "#f8fafc";
        el.style.textShadow = lum > 0.52 ? "0 0 2px #fff, 0 0 4px #fff" : "0 0 2px #000, 0 0 4px #000";
        el.textContent = s.toFixed(2) + (j === refIdx ? "*" : "");
        el.classList.toggle("ref", j === refIdx);
      }}
      const row = Math.floor(refIdx / nCols);
      const col = refIdx % nCols;
      refInfo.textContent = `Reference merge index ${{refIdx}} (grid row ${{row}}, col ${{col}}) — cosines vs all ${{nPatches}} patches`;
    }}

    paint(0);
  </script>
</body>
</html>
"""

OUT_DIR.mkdir(parents=True, exist_ok=True)
_html_out = OUT_DIR / "patch_cosine_explorer.html"
_html_out.write_text(_html_doc, encoding="utf-8")
print("Saved:", _html_out.resolve())

# %% Output
from pprint import pprint
for im in image_inputs:
    print(im.size)        # (112, 196)  → W × H
    print(im.height, im.width)  # 196, 112 → H, W
    
# pprint(inputs)
print(inputs['pixel_values'].shape)
from pathlib import Path
out = Path(__file__).resolve().parent / "output" / "debug_processor_image.png"
image_inputs[0].save(out)