# %% Imports & paths

"""**Qwen3-VL-8B-Instruct** on **MMStar**, with an educational walkthrough of the HF forward path.

Cells after **Load Qwen3-VL** mirror (at high level) ``Qwen3VLModel.forward`` and
``Qwen3VLForConditionalGeneration.forward`` in Hugging Face **transformers**. Open the modeling
file printed by ``transformers_source_hint`` and read alongside (e.g. fusion ~L1293–1309, LM head
~L1522–1541 in ``modeling_qwen3_vl.py``).

Install (once), from repo root::

  uv pip install datasets accelerate
  uv pip install "git+https://github.com/huggingface/transformers"

**Stonesoup:** Watch this file; run cells in order after **Reset kernel** when needed.
**Load MMStar row** and **Generate + HTML report** print HTML with a leading ``# stonesoup:render=html`` line
(stripped in the UI; see ``EXPERIMENT_PYTHON.md``) so the pane does not rely on heuristics.

**Terminal:** ``uv run python experiments/2026-03-28-Qwen3-VL-MMStar/qwen3vl_mmstar.py``
"""

from __future__ import annotations

import base64
import html
import inspect
import io
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

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


# %% Load MMStar row

DATASET_ID = "Lin-Chen/MMStar"
DATASET_CONFIG = "val"  # single parquet split ``val``
ROW_INDEX = 0

ds = load_dataset(DATASET_ID, DATASET_CONFIG, split="val")
row = ds[ROW_INDEX]
question: str = row["question"]
gold: str = row["answer"]
image = row["image"]  # ``datasets`` Image → PIL

_row_id = row.get("index", ROW_INDEX)
_img_src = pil_png_data_uri(image)
_row_preview = (
    f'<div style="font-family:system-ui,sans-serif;font-size:13px;line-height:1.45">'
    f'<p style="margin:0 0 0.5em 0"><img src="{_img_src}" alt="MMStar" '
    f'style="max-width:100%;max-height:min(360px,50vh);height:auto;'
    f'border:1px solid #555;border-radius:4px;display:block"/></p>'
    f"<p><b>index</b> {html.escape(str(_row_id))} &nbsp; "
    f"<b>gold</b> {html.escape(gold)}</p>"
    f"<p><b>question</b></p>"
    f'<p style="margin:0;white-space:pre-wrap">{html.escape(question)}</p>'
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

ANSWER_HINT = "Answer with the option letter only (e.g. A, B, C, or D)."
user_text = f"{question}\n\n{ANSWER_HINT}"

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

_non_image = (inputs.input_ids != model.config.image_token_id).unsqueeze(-1)
_diff_masked = (_merged - inputs_embeds) * _non_image
print("max |Δ embed| on non-image positions:", _diff_masked.abs().max().item())
print("(expect 0 — only <|image_pad|> rows should change)")

# %% Step: full backbone forward (sanity)

_model_in_keys = (
    "input_ids",
    "attention_mask",
    "pixel_values",
    "image_grid_thw",
    "pixel_values_videos",
    "video_grid_thw",
    "mm_token_type_ids",
)
_model_kw = {k: inputs[k] for k in _model_in_keys if k in inputs}

with torch.inference_mode():
    backbone_out = model.model(**_model_kw)

_hidden = backbone_out.last_hidden_state
print("backbone last_hidden_state:", tuple(_hidden.shape), _hidden.dtype)

# %% Step: LM head + one-token logits

# ForConditionalGeneration.forward ~1537–1541: logits = lm_head(hidden_states[:, slice])
with torch.inference_mode():
    _logits = model.lm_head(_hidden[:, -1:, :])
_top = _logits[0, -1].float().topk(5)
_tok = processor.tokenizer
for rank, (val, idx) in enumerate(zip(_top.values.tolist(), _top.indices.tolist()), start=1):
    piece = _tok.decode([idx], skip_special_tokens=False)
    print(f"top{rank} id={idx} logit={val:.2f} piece={piece!r}")

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
    f"<p><b>Question</b></p>"
    f'<p style="margin:0.35em 0 0.75em;white-space:pre-wrap">{html.escape(question)}</p>'
    f"<p><b>Model</b> {html.escape(out)}</p>"
    f"<p><b>Gold</b> {html.escape(gold)}</p>"
    f"</div>"
)
print(STONESOUP_RENDER_HTML + _report)
