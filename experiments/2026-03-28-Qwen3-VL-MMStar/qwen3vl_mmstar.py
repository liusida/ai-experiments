# %% Imports & paths

"""**Qwen3-VL-8B-Instruct** on **MMStar**, with an educational walkthrough of the HF forward path.

Cells after **Load Qwen3-VL** mirror (at high level) ``Qwen3VLModel.forward`` and
``Qwen3VLForConditionalGeneration.forward`` in Hugging Face **transformers**. The trunk is spelled out
stepwise (merged embeds → ``visual_pos_masks`` / ``deepstack_visual_embeds`` → ``compute_3d_position_ids``
→ ``language_model``) so ``vision_out.deepstack_features`` is visibly consumed—not a single opaque
``model.model(**kwargs)`` lump. Open the modeling file printed by ``transformers_source_hint`` and read
alongside (e.g. fusion ~L1293–1367, LM head ~L1522–1541 in ``modeling_qwen3_vl.py``).

Install (once), from repo root::

  uv pip install datasets accelerate
  uv pip install "git+https://github.com/huggingface/transformers"

**Stonesoup:** Watch this file; run cells in order after **Reset** when needed (restarts backend).
**Load MMStar row** uses the OpenCompass/VLMEvalKit-style HF mirror (**separate A–D columns**) and the
**Qwen3-VL MCQ** user-text template from VLMEvalKit ``Qwen3VLPromptMixin._build_mcq_prompt`` (not the generic
``ImageMCQDataset`` “Please select…” line).
**Generate + HTML report** prints HTML with a leading ``# stonesoup:render=html`` line
(stripped in the UI; see ``EXPERIMENT_PYTHON.md``) so the pane does not rely on heuristics.

**Terminal:** ``uv run python experiments/2026-03-28-Qwen3-VL-MMStar/qwen3vl_mmstar.py``
"""

from __future__ import annotations

import base64
import html
import inspect
import io
import string
from pathlib import Path

from PIL import Image

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoProcessor, Qwen3VLForConditionalGeneration
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast

try:
    from stonesoup import STONESOUP_RENDER_HTML
except ImportError:  # ``uv run python`` without editable ``stonesoup`` — prefix is harmless in the terminal
    STONESOUP_RENDER_HTML = "# stonesoup:render=html\n"

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("REPO_ROOT:", REPO_ROOT)
print("DEVICE:", DEVICE)


def transformers_source_hint(obj: type) -> None:
    """Print the installed ``.py`` path for a model class (open as reference while stepping)."""
    try:
        print(inspect.getfile(obj))
    except (OSError, TypeError) as e:
        print("(could not resolve source file):", e)


def pil_png_data_uri(im) -> str:
    """Embed a PIL image in HTML via ``<img src="data:image/png;base64,...">`` (Stonesoup rich output)."""
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    b64 = base64.standard_b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def n_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def mmstar_opencompass_to_pil(row: dict) -> Image.Image:
    """Decode image from ``morpheushoc/MMStar_opencompass`` (base64 JPEG) or pass through a PIL ``Image``."""
    raw = row["image"]
    if isinstance(raw, Image.Image):
        return raw.convert("RGB")
    if isinstance(raw, str):
        return Image.open(io.BytesIO(base64.standard_b64decode(raw))).convert("RGB")
    raise TypeError(f"Unsupported image field type: {type(raw)}")


def mmstar_qwen3_vl_prompt(row: dict) -> str:
    """MMStar user text per Qwen3-VL in VLMEvalKit: ``Qwen3VLPromptMixin._build_mcq_prompt``.

    MCQ datasets (including MMStar) use this instead of the generic ``ImageMCQDataset`` line
    ``Please select the correct answer…`` — the mixin ends with **Answer with the option letter only.**

    See ``vlmeval/vlm/qwen3_vl/prompt.py`` (VLMEvalKit, Qwen3-VL mixin).
    """
    question = row["question"]
    options: dict[str, object] = {}
    for cand in string.ascii_uppercase:
        if cand not in row:
            continue
        item = row[cand]
        if item is None:
            continue
        if isinstance(item, str) and not item.strip():
            continue
        options[cand] = item
    hint = row.get("hint")
    prompt = ""
    if hint is not None and str(hint).strip():
        prompt += f"Hint: {hint}\n"
    prompt += f"Question: {question}\n"
    if options:
        prompt += "Options:\n"
        for key, item in options.items():
            prompt += f"{key}. {item}\n"
        prompt += "Answer with the option letter only."
    return prompt.rstrip()


# %% Load MMStar row

# OpenCompass layout (A–D columns) — same framing VLMEvalKit uses for MMStar; see lmms-eval #901 / VLMEvalKit.
DATASET_ID = "morpheushoc/MMStar_opencompass"
ROW_INDEX = 7

ds = load_dataset(DATASET_ID, split="val")
row = ds[ROW_INDEX]
gold: str = row["answer"]
image = mmstar_opencompass_to_pil(row)
user_text = mmstar_qwen3_vl_prompt(row)

_row_id = row.get("index", ROW_INDEX)
_img_src = pil_png_data_uri(image)
_row_preview = (
    f'<div style="font-family:system-ui,sans-serif;font-size:13px;line-height:1.45">'
    f'<p style="margin:0 0 0.5em 0"><img src="{_img_src}" alt="MMStar" '
    f'style="max-width:100%;max-height:min(360px,50vh);height:auto;'
    f'border:1px solid #555;border-radius:4px;display:block"/></p>'
    f"<p><b>index</b> {html.escape(str(_row_id))} &nbsp; "
    f"<b>gold</b> {html.escape(gold)}</p>"
    f"<p><b>prompt (Qwen3-VL MCQ mixin)</b></p>"
    f'<p style="margin:0;white-space:pre-wrap">{html.escape(user_text)}</p>'
    f"</div>"
)
print(STONESOUP_RENDER_HTML + _row_preview)

# %% Load Qwen3-VL

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    dtype="auto" if DEVICE.type == "cuda" else torch.float32,
    device_map="auto" if DEVICE.type == "cuda" else None,
)
if DEVICE.type != "cuda":
    model = model.to(DEVICE)

processor = AutoProcessor.from_pretrained(MODEL_ID)
print("Loaded:", MODEL_ID)

# %% Config & module map

# Logical pieces (read class names in modeling_qwen3_vl.py):
#   Qwen3VLVisionModel  → model.model.visual
#   Qwen3VLTextModel    → model.model.language_model
#   Qwen3VLModel.forward fuses embeds + vision then calls language_model
#   lm_head             → maps last hidden states to vocab logits

print("transformers implementation (open this file):")
transformers_source_hint(type(model))

print("text hidden_size:", model.config.text_config.hidden_size)
vcfg = model.config.vision_config
print("vision patch_size:", vcfg.patch_size, "| spatial_merge_size:", vcfg.spatial_merge_size)
print("image_token_id:", model.config.image_token_id)
print("submodules: .model.visual | .model.language_model | .lm_head")

# %% Build chat inputs (processor)
print("user_text:\n", user_text)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": user_text},
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

print("Batch keys:", list(inputs.keys()))
for k, v in inputs.items():
    if hasattr(v, "shape"):
        print(f"  {k}: {tuple(v.shape)} {v.dtype}")
    else:
        print(f"  {k}: {type(v).__name__}")

_n_img_tok = (inputs.input_ids == model.config.image_token_id).sum().item()
print("image placeholder token count (== image_token_id):", _n_img_tok)
print("================================================")
print(processor.tokenizer.decode(inputs.input_ids))

# %% Step: text embeddings

# modeling_qwen3_vl.py ~1293–1294: inputs_embeds = embed_tokens(input_ids)
inputs_embeds = model.get_input_embeddings()(inputs.input_ids)
print("inputs_embeds:", tuple(inputs_embeds.shape), inputs_embeds.dtype)

# %% Step: vision embeddings

# ~1299–1305: get_image_features → pooler_output (tuple of chunks) → cat along dim 0
with torch.inference_mode():
    vision_out = model.get_image_features(
        inputs.pixel_values,
        inputs.image_grid_thw,
        return_dict=True,
    )
print("vision_out keys:", vision_out.keys())
print("pooler_output length:", len(vision_out.pooler_output))
print("deepstack_features length:", len(vision_out.deepstack_features))
print("vision_out last_hidden_state:", vision_out.last_hidden_state.shape)
print("vision_out pooler_output[0]:", vision_out.pooler_output[0].shape)
for i in range(len(vision_out.deepstack_features)):
    print(f"vision_out deepstack_features[{i}]:", vision_out.deepstack_features[i].shape)
print("================================================")
_img_chunks = vision_out.pooler_output
print("pooler_output: list of", len(_img_chunks), "chunks; first chunk shape:", tuple(_img_chunks[0].shape))
image_embeds_flat = torch.cat(_img_chunks, dim=0).to(
    device=inputs_embeds.device, dtype=inputs_embeds.dtype
)
print("image_embeds (cat, dim=0):", tuple(image_embeds_flat.shape), image_embeds_flat.dtype)

_ds = vision_out.deepstack_features
if _ds is not None:
    print("deepstack_features: n_levels=", len(_ds), "| level0 shape:", tuple(_ds[0].shape))
else:
    print("deepstack_features: None")

# %% Step: merge placeholders (masked_scatter)

# ~1306–1309: get_placeholder_mask then masked_scatter into inputs_embeds
_merged = inputs_embeds.clone()
_image_mask, _video_mask = model.model.get_placeholder_mask(
    inputs.input_ids,
    _merged,
    image_features=image_embeds_flat,
)
_merged = _merged.masked_scatter(_image_mask, image_embeds_flat)
print("The input embeddings are ready to be passed to the language model:\n", _merged.shape)

_non_image = (inputs.input_ids != model.config.image_token_id).unsqueeze(-1)
_diff_masked = (_merged - inputs_embeds) * _non_image
print("max |Δ embed| on non-image positions:", _diff_masked.abs().max().item())
print("(expect 0 — only <|image_pad|> rows should change)")

# %% Step: vision_out → deepstack args (no second vision pass)
# modeling_qwen3_vl.py ~1323–1341: image-only path sets ``visual_pos_masks`` and passes
# ``deepstack_visual_embeds`` (same tensors as ``vision_out.deepstack_features``) into ``language_model``.
# Collapse the expanded placeholder mask to one bool per token; align deepstack to LM dtype/device.
_visual_pos_masks = _image_mask[..., 0].contiguous()
_deepstack_visual_embeds = None
if _ds is not None:
    _deepstack_visual_embeds = [t.to(device=_merged.device, dtype=_merged.dtype) for t in _ds]

print("visual_pos_masks:", tuple(_visual_pos_masks.shape), "| n_true:", int(_visual_pos_masks.sum().item()))
print(
    "deepstack_visual_embeds:",
    None if _deepstack_visual_embeds is None else [tuple(t.shape) for t in _deepstack_visual_embeds],
)

# %% Step: compute_3D position ids for language_model
# ~1347–1356: ``Qwen3VLModel.compute_3d_position_ids`` (may set ``model.model.rope_deltas`` as side effect).
_mm_tid = inputs["mm_token_type_ids"] if "mm_token_type_ids" in inputs else None
_vgrid = inputs["video_grid_thw"] if "video_grid_thw" in inputs else None

_position_ids = model.model.compute_3d_position_ids(
    input_ids=inputs.input_ids,
    image_grid_thw=inputs.image_grid_thw,
    video_grid_thw=_vgrid,
    inputs_embeds=_merged,
    attention_mask=inputs.attention_mask,
    past_key_values=None,
    mm_token_type_ids=_mm_tid,
)
print("position_ids:", None if _position_ids is None else tuple(_position_ids.shape), _position_ids.dtype)
print("================================================")
print("position_ids:\n", _position_ids)

# %% Step: language_model forward layer-by-layer (same as ``Qwen3VLTextModel.forward``)
# Mirrors ``modeling_qwen3_vl.py`` ~886–939: causal mask → shared RoPE → for each ``decoder_layer``,
# then optional ``_deepstack_process``, then final RMSNorm. Set ``_LM_TRACE_ALL_LAYERS`` / ``_LM_TRACE_EVERY``
# to control how much is printed; ``_lm_trace`` holds per-step dicts for programmatic inspection.
_lm = model.model.language_model
_past_kv = None  # set DynamicCache + use_cache in modeling if you decode with KV later
_inputs_embeds_lm = _merged

_lm_cache_pos = torch.arange(
    0,
    _inputs_embeds_lm.shape[1],
    device=_inputs_embeds_lm.device,
)
_lm_pos_ids = _position_ids
if _lm_pos_ids is None:
    _lm_pos_ids = _lm_cache_pos.view(1, 1, -1).expand(4, _inputs_embeds_lm.shape[0], -1)
elif _lm_pos_ids.ndim == 2:
    _lm_pos_ids = _lm_pos_ids[None, ...].expand(4, _lm_pos_ids.shape[0], -1)

if _lm_pos_ids.ndim == 3 and _lm_pos_ids.shape[0] == 4:
    _lm_text_pos_ids = _lm_pos_ids[0]
    _lm_pos_ids_rot = _lm_pos_ids[1:]
else:
    _lm_text_pos_ids = None
    _lm_pos_ids_rot = _lm_pos_ids

_lm_attn_mask = create_causal_mask(
    config=_lm.config,
    inputs_embeds=_inputs_embeds_lm,
    attention_mask=inputs.attention_mask,
    cache_position=_lm_cache_pos,
    past_key_values=_past_kv,
    position_ids=_lm_text_pos_ids,
)

_hidden_lm = _inputs_embeds_lm
_lm_pos_emb = _lm.rotary_emb(_hidden_lm, _lm_pos_ids_rot)

_LM_TRACE_ALL_LAYERS = True  # ``True`` → print every layer; ``False`` → first / last + deepstack only
_LM_TRACE_EVERY = 1  # when ``_LM_TRACE_ALL_LAYERS``, print every N-th layer (1 = all)
_LM_STORE_HIDDEN_IN_TRACE = False  # ``True`` → each trace entry gets ``hidden_states`` on CPU (large)
_lm_trace: list[dict] = []
_n_lm_layers = len(_lm.layers)

with torch.inference_mode():
    for _layer_idx, _decoder_layer in enumerate(_lm.layers):
        _hidden_lm = _decoder_layer(
            _hidden_lm,
            attention_mask=_lm_attn_mask,
            position_ids=_lm_text_pos_ids,
            past_key_values=_past_kv,
            cache_position=_lm_cache_pos,
            position_embeddings=_lm_pos_emb,
        )
        _after_deep = False
        if _deepstack_visual_embeds is not None and _layer_idx in range(len(_deepstack_visual_embeds)):
            _hidden_lm = _lm._deepstack_process(
                _hidden_lm,
                _visual_pos_masks,
                _deepstack_visual_embeds[_layer_idx],
            )
            _after_deep = True

        _do_print = _LM_TRACE_ALL_LAYERS and (_layer_idx % _LM_TRACE_EVERY == 0 or _layer_idx == _n_lm_layers - 1)
        if not _LM_TRACE_ALL_LAYERS:
            _do_print = _layer_idx < 2 or _layer_idx == _n_lm_layers - 1 or _after_deep
        if _do_print:
            _h = _hidden_lm
            print(
                f"  layer {_layer_idx:>3} | shape {tuple(_h.shape)} | "
                f"rms {_h.float().pow(2).mean().sqrt().item():.4f} | "
                f"last_tok max|h| {_h[0, -1].abs().max().item():.4f}"
                + (" | +deepstack" if _after_deep else ""),
            )

        _trace_entry = {
            "layer": _layer_idx,
            "after_deepstack": _after_deep,
            "shape": tuple(_hidden_lm.shape),
            "rms": float(_hidden_lm.float().pow(2).mean().sqrt().item()),
            "last_tok_absmax": float(_hidden_lm[0, -1].abs().max().item()),
        }
        if _LM_STORE_HIDDEN_IN_TRACE:
            _trace_entry["hidden_states"] = _hidden_lm.detach().float().cpu()
        _lm_trace.append(_trace_entry)

    _hidden_lm = _lm.norm(_hidden_lm)

print("post-norm: ", tuple(_hidden_lm.shape), _hidden_lm.dtype, end="")
print(
    f" | rms {_hidden_lm.float().pow(2).mean().sqrt().item():.4f}",
)

backbone_out = BaseModelOutputWithPast(last_hidden_state=_hidden_lm, past_key_values=_past_kv)
_hidden = backbone_out.last_hidden_state
print("language_model last_hidden_state:", tuple(_hidden.shape), _hidden.dtype)
print("len(_lm_trace) == n_layers:", len(_lm_trace), "==", _n_lm_layers)

# %% Step: LM head + one-token logits

# ForConditionalGeneration.forward ~1537–1541: logits = lm_head(hidden_states[:, slice])
with torch.inference_mode():
    _logits = model.lm_head(_hidden[:, -1:, :])
_logvec = _logits[0, -1].float()
_top = _logvec.topk(5)
_probs = torch.softmax(_logvec, dim=-1)
_tok = processor.tokenizer
for rank, (val, idx) in enumerate(zip(_top.values.tolist(), _top.indices.tolist()), start=1):
    piece = _tok.decode([idx], skip_special_tokens=False)
    p = _probs[idx].item()
    print(f"top{rank} id={idx} logit={val:.2f} prob={p:.6f} piece={piece!r}")

# %% Generate + HTML report

MAX_NEW_TOKENS = 128

with torch.inference_mode():
    generated_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)

generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
out = processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)[0]

_img = pil_png_data_uri(image)
_report = (
    f'<div style="font-family:system-ui,sans-serif;font-size:13px;line-height:1.45">'
    f'<p style="margin:0 0 0.5em 0"><img src="{_img}" alt="MMStar" '
    f'style="max-width:100%;max-height:min(360px,50vh);height:auto;'
    f'border:1px solid #555;border-radius:4px;display:block"/></p>'
    f"<p><b>Prompt (Qwen3-VL MCQ mixin)</b></p>"
    f'<p style="margin:0.35em 0 0.75em;white-space:pre-wrap">{html.escape(user_text)}</p>'
    f"<p><b>Model</b> {html.escape(out)}</p>"
    f"<p><b>Gold</b> {html.escape(gold)}</p>"
    f"</div>"
)
print(STONESOUP_RENDER_HTML + _report)

# %% Parameter counts (visual / text / merger / lm_head)

_core = model.model
_visual = _core.visual
_language = _core.language_model

_param_ids_visual = {id(p) for p in _visual.parameters()}
_param_ids_language = {id(p) for p in _language.parameters()}

n_visual = n_params(_visual)
n_text = n_params(_language)
n_merger = sum(
    p.numel()
    for p in _core.parameters()
    if id(p) not in _param_ids_visual and id(p) not in _param_ids_language
)
n_lm_head_total = n_params(model.lm_head)
n_lm_head_unique = sum(
    p.numel() for p in model.lm_head.parameters() if id(p) not in _param_ids_language
)
n_total = sum(p.numel() for p in model.parameters())

print("Qwen3VLForConditionalGeneration — trainable parameter counts")
print(f"  visual (model.model.visual):     {n_visual:>14,}")
print(f"  text (model.model.language_model): {n_text:>14,}")
print(
    "  merger / other (on Qwen3VLModel, not in visual or LM subtree):",
    f"{n_merger:>14,}",
)
print(f"  lm_head (all tensors):           {n_lm_head_total:>14,}")
if n_lm_head_unique != n_lm_head_total:
    print(
        "    └─ params not shared with language_model:",
        f"{n_lm_head_unique:>10,}",
        "(tie with embed weights)",
    )
print(f"  ─────────────────────────────────────────────")
print(f"  sum of four lines above:         {n_visual + n_text + n_merger + n_lm_head_total:>14,}")
print(f"  model (dedup shared storage):    {n_total:>14,}")
if n_visual + n_text + n_merger + n_lm_head_total != n_total:
    print(
        "(sum can exceed total if lm_head shares weights with language_model;",
        "merger is 0 if all trunk weights live only under visual + language_model.)",
    )

_children = list(_core.named_children())
print("model.model top-level children:", [k for k, _ in _children])
