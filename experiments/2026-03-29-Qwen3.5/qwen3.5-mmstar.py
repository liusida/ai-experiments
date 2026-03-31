# %% Imports & paths

"""**Qwen3.5-4B** on **MMStar** (OpenCompass HF mirror): accuracy, per-category breakdown, JSONL + HTML report.

Uses the same dataset and MCQ user text as
``experiments/2026-03-28-Qwen3-VL-MMStar`` (VLMEvalKit-style Qwen3-VL mixin:
``Answer with the option letter only.``), adapted for the Qwen3.5 multimodal chat template.

**Install** (repo root)::

  uv pip install datasets accelerate transformers pillow tqdm

**Stonesoup:** Watch this file; run cells **in order** after **Reset** when needed. Adjust **Config** before **Load dataset**
and **Load Qwen3.5**. **One sample: run hook + 1-token** runs prefill + hook on the **last** decoder layer (image placeholder rows),
then ``generate(max_new_tokens=1)``, and fills ``HOOK_SAMPLE_RESULT``. **One sample: hook HTML** prints rich HTML from that dict.
The hook preview PNG is saved under ``data/images/mmstar/mmstar_<id>.png`` and linked as
``http://127.0.0.1:8765/data/image/mmstar/mmstar_<id>.png`` (Stonesoup static mount).
Set ``HOOK_MMSTAR_DS_ROW`` in the run cell to pick a ``val`` row (or ``None`` for the first ``INDICES`` item).
**ViT cosine sweep** runs over ``INDICES`` (optional ``COSINE_SWEEP_LIMIT``), plots **mean pairwise ViT cosine** for **correct** vs **incorrect** only,
writes ``plots/mmstar_vit_cosine_correct_vs_incorrect.png``, and the following cell shows it in **HTML** (URL via ``data/images/mmstar/``).
The **Run MMStar evaluation** cell is long on full ``val`` (1500 rows); use a small ``LIMIT`` first.

**Terminal:** ``uv run python experiments/2026-03-29-Qwen3.5/qwen3.5-mmstar.py``

Large downloads (HF model weights, dataset cache) and **unified memory**: see ``AGENTS.md`` / kernel CUDA fraction.
"""

from __future__ import annotations

import base64
import html
import io
import json
import random
import re
import shutil
import string
import time
from pathlib import Path

from PIL import Image
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

try:
    from stonesoup import STONESOUP_RENDER_HTML
except ImportError:
    STONESOUP_RENDER_HTML = "# stonesoup:render=html\n"

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EXP_DIR = Path(__file__).resolve().parent
OUT_DIR = EXP_DIR / "output" / "mmstar"
DATASET_ID = "morpheushoc/MMStar_opencompass"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("REPO_ROOT:", REPO_ROOT)
print("EXP_DIR:", EXP_DIR)
print("DEVICE:", DEVICE)


def mmstar_opencompass_to_pil(row: dict) -> Image.Image:
    raw = row["image"]
    if isinstance(raw, Image.Image):
        return raw.convert("RGB")
    if isinstance(raw, str):
        return Image.open(io.BytesIO(base64.standard_b64decode(raw))).convert("RGB")
    raise TypeError(f"Unsupported image field type: {type(raw)}")


def mmstar_mcq_prompt(row: dict) -> str:
    """MMStar user text (same structure as Qwen3-VL MCQ mixin in VLMEvalKit)."""
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


def extract_choice_letter(raw: str) -> str | None:
    t = raw.strip().upper()
    m = re.search(r"\(([A-D])\)", t)
    if m:
        return m.group(1)
    m = re.search(r"\b([A-D])\b", t)
    if m:
        return m.group(1)
    m = re.match(r"^[^A-D]*([A-D])", t)
    if m:
        return m.group(1)
    for ch in t:
        if ch in "ABCD":
            return ch
    return None


def vit_mean_pairwise_cosine(last_hidden_state: torch.Tensor) -> tuple[float | None, int]:
    """Mean cosine similarity over unordered patch pairs (rows L2-normalized). Returns ``(mean, n_pairs)``."""
    patches = last_hidden_state.detach().float()
    n = int(patches.shape[0])
    if n < 2:
        return None, 0
    u = F.normalize(patches, dim=-1, eps=1e-12)
    sim = u @ u.T
    ti, tj = torch.triu_indices(n, n, offset=1)
    return float(sim[ti, tj].mean().item()), int(ti.numel())


def build_index_list(
    n_total: int,
    *,
    limit: int | None,
    start: int,
    shuffle_seed: int | None,
) -> list[int]:
    if shuffle_seed is not None:
        rng = random.Random(shuffle_seed)
        perm = list(range(n_total))
        rng.shuffle(perm)
        take = n_total if limit is None else limit
        return perm[start : start + take]
    end = n_total if limit is None else min(n_total, start + limit)
    return list(range(start, end))


def write_mmstar_html_report(
    *,
    report_path: Path,
    model_id: str,
    results: list[dict[str, object]],
    n_ok: int,
    n_parse_fail: int,
    by_cat: dict[str, dict[str, int]],
    elapsed: float,
    n_dataset: int,
) -> None:
    n = len(results)
    acc = n_ok / n if n else 0.0
    cat_rows = "".join(
        f"<tr><td>{html.escape(c)}</td><td>{v['ok']}/{v['n']}</td>"
        f"<td>{(v['ok'] / v['n'] if v['n'] else 0):.4f}</td></tr>"
        for c, v in sorted(by_cat.items(), key=lambda x: x[0])
    )
    subset_note = ""
    if len(by_cat) == 1 and n < n_dataset:
        only = next(iter(by_cat.keys()))
        subset_note = (
            f"<br/><em>One category row: subset spans only «{html.escape(only)}». "
            f"Val split is blocked by category (rows 0–249 = coarse perception). "
            f"Use SHUFFLE_SEED with LIMIT for a mixed quick test.</em>"
        )
    cards = []
    for rec in results:
        ok = rec["correct"]
        border = "#2d7a3e" if ok else "#a44040"
        cards.append(
            f'<section class="card" style="border-left-color:{border}">'
            f'<div class="row"><img loading="lazy" src="{html.escape(str(rec["image_relpath"]))}" '
            f'alt="" /></div>'
            f'<div class="meta">index <code>{html.escape(str(rec["index"]))}</code> · '
            f'{html.escape(str(rec["category"]))} · '
            f'gold <b>{html.escape(str(rec["gold"]))}</b> · '
            f'pred <b>{html.escape(str(rec["prediction_letter"]))}</b> '
            f'({"✓" if ok else "✗"})</div>'
            f'<pre class="prompt">{html.escape(str(rec["prompt"]))}</pre>'
            f'<div class="pred"><b>Model</b> {html.escape(str(rec["prediction_raw"]))}</div>'
            f"</section>"
        )
    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>MMStar eval — {html.escape(model_id)}</title>
  <style>
    :root {{
      --bg: #1a1d23;
      --fg: #e8eaed;
      --muted: #9aa0a6;
      --line: #3c4043;
    }}
    body {{
      font-family: system-ui, Segoe UI, Roboto, sans-serif;
      background: var(--bg);
      color: var(--fg);
      margin: 0;
      padding: 1rem 1.25rem 2rem;
      line-height: 1.45;
    }}
    h1 {{ font-size: 1.25rem; margin: 0 0 0.5rem; }}
    .summary {{ color: var(--muted); margin-bottom: 1.25rem; }}
    table {{ border-collapse: collapse; margin: 1rem 0 2rem; font-size: 0.9rem; }}
    th, td {{ border: 1px solid var(--line); padding: 0.35rem 0.6rem; text-align: left; }}
    th {{ background: #252830; }}
    .grid {{ display: flex; flex-direction: column; gap: 1.25rem; max-width: 960px; }}
    .card {{
      background: #22252c;
      border: 1px solid var(--line);
      border-left-width: 4px;
      border-radius: 6px;
      padding: 0.75rem 1rem;
    }}
    .card img {{
      max-width: 100%;
      max-height: 360px;
      height: auto;
      border-radius: 4px;
      border: 1px solid var(--line);
    }}
    .meta {{ font-size: 0.85rem; color: var(--muted); margin: 0.5rem 0; }}
    pre.prompt {{
      white-space: pre-wrap;
      background: #15171c;
      border: 1px solid var(--line);
      border-radius: 4px;
      padding: 0.6rem 0.75rem;
      font-size: 0.82rem;
      margin: 0.5rem 0;
    }}
    .pred {{ font-size: 0.95rem; margin-top: 0.35rem; }}
  </style>
</head>
<body>
  <h1>MMStar evaluation</h1>
  <p class="summary">
    Model: <code>{html.escape(model_id)}</code><br/>
    Dataset: <code>{html.escape(DATASET_ID)}</code> (val)<br/>
    Examples: {n} &nbsp;·&nbsp; Accuracy: <b>{acc:.4f}</b> ({n_ok} / {n})
    &nbsp;·&nbsp; Parse failures: {n_parse_fail}<br/>
    Wall time: {elapsed:.1f}s
    {subset_note}
  </p>
  <h2 style="font-size:1rem;">Per category</h2>
  <table>
    <thead><tr><th>category</th><th>correct / total</th><th>acc</th></tr></thead>
    <tbody>{cat_rows}</tbody>
  </table>
  <h2 style="font-size:1rem;">All examples</h2>
  <div class="grid">
    {"".join(cards)}
  </div>
</body>
</html>
"""
    report_path.write_text(html_doc, encoding="utf-8")


# %% Config

# None = full val split (~1500). Use 32 + SHUFFLE_SEED for a quick mixed-category smoke test.
LIMIT: int | None = 512
START = 0
SHUFFLE_SEED: int | None = 0
MAX_NEW_TOKENS = 2

# Override model id here if needed (must match a Qwen3.5 multimodal HF repo).
MODEL_ID = "Qwen/Qwen3.5-4B"

print(
    "Config:",
    f"LIMIT={LIMIT}",
    f"START={START}",
    f"SHUFFLE_SEED={SHUFFLE_SEED}",
    f"MAX_NEW_TOKENS={MAX_NEW_TOKENS}",
    f"MODEL_ID={MODEL_ID}",
)

# %% Load dataset

print("Loading dataset", DATASET_ID, flush=True)
_ds = load_dataset(DATASET_ID, split="val")
N_TOTAL = len(_ds)
INDICES = build_index_list(N_TOTAL, limit=LIMIT, start=START, shuffle_seed=SHUFFLE_SEED)
print(f"Using {len(INDICES)} examples (dataset len={N_TOTAL})", flush=True)
if SHUFFLE_SEED is not None:
    print(f"Order: shuffled seed={SHUFFLE_SEED}, slice [{START}:{START + len(INDICES)}]", flush=True)
elif LIMIT is not None and START < 250 and START + LIMIT <= 250:
    print(
        "NOTE: First 250 rows are «coarse perception» only. Set SHUFFLE_SEED (e.g. 0) to mix categories.",
        flush=True,
    )

# %% Load Qwen3.5

model = Qwen3_5ForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype="auto" if DEVICE.type == "cuda" else torch.float32,
    device_map="auto" if DEVICE.type == "cuda" else None,
)
if DEVICE.type != "cuda":
    model = model.to(DEVICE)

processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
model.eval()
print("Loaded:", MODEL_ID, flush=True)

# %% One sample: run hook + 1-token (fill HOOK_SAMPLE_RESULT)

# Needs **Config**, **Load dataset**, **Load Qwen3.5**. Prefill ``model(**inputs)`` drives the hook (full-seq hidden
# states). ``generate(max_new_tokens=1)`` runs after the hook is removed.
#
# **Which example?** Set ``HOOK_MMSTAR_DS_ROW`` to an integer **HuggingFace row** into ``val`` (0 … len-1).
# ``None`` means “first row of **Config** ``INDICES``” (same order as the batched eval when you use the same Config).

HOOK_MMSTAR_DS_ROW: int | None = 0
SAMPLE_DS_INDEX: int = (
    HOOK_MMSTAR_DS_ROW if HOOK_MMSTAR_DS_ROW is not None else (INDICES[0] if INDICES else 0)
)

_row_hook = _ds[SAMPLE_DS_INDEX]
_pil_hook = mmstar_opencompass_to_pil(_row_hook)
_text_hook = mmstar_mcq_prompt(_row_hook)
_gold_hook = str(_row_hook["answer"]).strip().upper()[:1]

_messages_hook = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": _pil_hook},
            {"type": "text", "text": _text_hook},
        ],
    }
]
_inputs_hook = processor.apply_chat_template(
    _messages_hook,
    tokenize=True,
    add_generation_prompt=True,
    enable_thinking=False,
    return_dict=True,
    return_tensors="pt",
)
_inputs_hook = _inputs_hook.to(DEVICE)

_ids_hook = _inputs_hook["input_ids"][0]
_image_tok_id = int(model.config.image_token_id)
_img_idx = (_ids_hook == _image_tok_id).nonzero(as_tuple=False).squeeze(-1)
_n_img = int(_img_idx.numel())

# ViT patch vectors (``last_hidden_state``): one row per patch. Extra forward vs prefill (vision runs again inside ``model(**)``).
with torch.inference_mode():
    _vit_out = model.model.get_image_features(
        _inputs_hook["pixel_values"],
        _inputs_hook["image_grid_thw"],
        return_dict=True,
    )
_h_vit = _vit_out.last_hidden_state
_patches_vit = _h_vit.detach().float()
_vit_patch_l2 = _patches_vit.norm(dim=-1)
_vit_pairwise_cos_mean, _vit_pairwise_n_pairs = vit_mean_pairwise_cosine(_h_vit)

_vit_patch_norm_summary: dict[str, object] = {
    "tag": "ViT last_hidden_state: L2 norm per patch vector",
    "shape": tuple(_patches_vit.shape),
    "l2_mean": float(_vit_patch_l2.mean().item()),
    "l2_std": float(_vit_patch_l2.std().item()),
    "l2_min": float(_vit_patch_l2.min().item()),
    "l2_max": float(_vit_patch_l2.max().item()),
    "pairwise_cosine_mean": _vit_pairwise_cos_mean,
    "pairwise_n_pairs": _vit_pairwise_n_pairs,
}
del _vit_out, _h_vit, _patches_vit, _vit_patch_l2

_image_act_summary: dict[str, object] = {"tag": "last_decoder_layer output @ image tokens (prefill)"}


def _collect_image_row_stats(hidden: torch.Tensor, img_positions: torch.Tensor) -> None:
    """``hidden`` (T, D) for batch 0; ``img_positions`` long indices along T. Fills ``_image_act_summary``."""
    if img_positions.numel() == 0:
        _image_act_summary["empty"] = True
        return
    h = hidden[img_positions.to(hidden.device), :].detach().float()
    row_l2 = h.norm(dim=-1)
    _image_act_summary.update(
        {
            "empty": False,
            "shape": tuple(h.shape),
            "mean": h.mean().item(),
            "std": h.std().item(),
            "min": h.min().item(),
            "max": h.max().item(),
            "l2_mean": row_l2.mean().item(),
            "l2_std": row_l2.std().item(),
            "l2_min": row_l2.min().item(),
            "l2_max": row_l2.max().item(),
        }
    )


_last_dec = model.model.language_model.layers[-1]


def _last_layer_hook(_module, _inp, out: torch.Tensor) -> None:
    _collect_image_row_stats(out[0], _img_idx)


_handle = _last_dec.register_forward_hook(_last_layer_hook)
try:
    with torch.inference_mode():
        _prefill_out = model(**_inputs_hook)
finally:
    _handle.remove()

_logits_last = _prefill_out.logits[0, -1, :].float()
_probs_last = torch.softmax(_logits_last, dim=-1)
_top5_prob, _top5_idx = torch.topk(_probs_last, k=5)
_first_step_top5: list[dict[str, object]] = []
for _rank in range(5):
    _tid = int(_top5_idx[_rank].item())
    _prob = float(_top5_prob[_rank].item())
    _logit = float(_logits_last[_tid].item())
    _piece = processor.decode([_tid], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    _first_step_top5.append(
        {"token_id": _tid, "logit": _logit, "prob": _prob, "piece": _piece}
    )

del _prefill_out, _logits_last, _probs_last, _top5_prob, _top5_idx

with torch.inference_mode():
    _gen_hook = model.generate(**_inputs_hook, max_new_tokens=1, do_sample=False)

_in_len_hook = _inputs_hook["input_ids"].shape[1]
_first_new = _gen_hook[0, _in_len_hook:]
_one_str = processor.decode(_first_new, skip_special_tokens=True).strip()

# Stonesoup serves ``<repo>/data/images`` at ``/data/image`` (see ``stonesoup/backend/server.py``).
_HOOK_IMAGE_HOST = "http://127.0.0.1:8765"
_mmstar_png_id = re.sub(r"[^\w.\-]", "_", str(_row_hook.get("index", SAMPLE_DS_INDEX)))[:120]
_mmstar_png_name = f"mmstar_{_mmstar_png_id}.png"
_mmstar_png_dir = REPO_ROOT / "data" / "images" / "mmstar"
_mmstar_png_dir.mkdir(parents=True, exist_ok=True)
_mmstar_png_path = _mmstar_png_dir / _mmstar_png_name
_pil_hook.save(_mmstar_png_path, format="PNG")
_hook_img_url = f"{_HOOK_IMAGE_HOST}/data/image/mmstar/{_mmstar_png_name}"

HOOK_SAMPLE_RESULT: dict[str, object] = {
    "ds_row": SAMPLE_DS_INDEX,
    "mmstar_index": _row_hook.get("index", "?"),
    "category": str(_row_hook.get("category", "")),
    "gold": _gold_hook,
    "prompt": _text_hook,
    "image_token_id": _image_tok_id,
    "n_image_positions": _n_img,
    "image_repo_relpath": str(_mmstar_png_path.relative_to(REPO_ROOT)),
    "image_url": _hook_img_url,
    "first_token_ids": [int(x) for x in _first_new.detach().cpu().tolist()],
    "first_token_decoded": _one_str,
    "first_step_top5": _first_step_top5,
    "vit_patch_norm": _vit_patch_norm_summary,
    "activation": dict(_image_act_summary),
}
print(HOOK_SAMPLE_RESULT, flush=True)

# %% One sample: hook report HTML from HOOK_SAMPLE_RESULT

# Requires **One sample: run hook + 1-token**. Reformats ``HOOK_SAMPLE_RESULT`` without re-running the model.

_r = HOOK_SAMPLE_RESULT
act: dict[str, object] = _r["activation"]
_vit: dict[str, object] = _r["vit_patch_norm"]

_c = "#e8eaed"
_muted = "#9aa0a6"
_panel = "#252830"
_border = "#3c4043"
_link = "#8ab4f8"
_code_bg = "#2d333b"

_meta_line = (
    f"ds_row={_r['ds_row']} · MMStar index={html.escape(str(_r['mmstar_index']))} · "
    f"category={html.escape(str(_r['category']))} · "
    f"image_token_id={_r['image_token_id']} · n_image_positions={_r['n_image_positions']}"
)

_vit_cos_line = (
    f"mean pairwise cosine (unique pairs, i&lt;j): {_vit['pairwise_cosine_mean']:.5f} &nbsp;·&nbsp; n_pairs={_vit['pairwise_n_pairs']}"
    if _vit["pairwise_cosine_mean"] is not None
    else f"mean pairwise cosine: n/a (need ≥2 patches, n_pairs={_vit['pairwise_n_pairs']})"
)
_vit_block = f"""<pre style="margin:0;background:{_panel};color:{_c};border:1px solid {_border};padding:12px;border-radius:6px;font-size:13px;white-space:pre-wrap">{html.escape(str(_vit['tag']))}
shape={_vit['shape']} (patches × hidden)
L2 norm per patch: mean={_vit['l2_mean']:.5f} std={_vit['l2_std']:.5f} min={_vit['l2_min']:.5f} max={_vit['l2_max']:.5f}
{_vit_cos_line}</pre>"""

if act.get("empty"):
    _act_block = f'<p style="margin:0;color:{_muted}"><em>No image placeholder rows — skipped activation stats.</em></p>'
else:
    _act_block = f"""<pre style="margin:0;background:{_panel};color:{_c};border:1px solid {_border};padding:12px;border-radius:6px;font-size:13px;white-space:pre-wrap">{html.escape(str(act['tag']))}
shape={act['shape']}
mean={act['mean']:.5f} std={act['std']:.5f} min={act['min']:.5f} max={act['max']:.5f}
L2 per image token: mean={act['l2_mean']:.5f} std={act['l2_std']:.5f} min={act['l2_min']:.5f} max={act['l2_max']:.5f}</pre>"""

_prompt_s = str(_r["prompt"])
_gold_s = str(_r["gold"])
_img_rel = str(_r["image_repo_relpath"])
_hook_img_url = str(_r["image_url"])
_ids_list = _r["first_token_ids"]
_one_str = str(_r["first_token_decoded"])
_top5_rows = _r["first_step_top5"]
_top5_table_rows = "".join(
    "<tr>"
    f"<td style='padding:4px 8px;border:1px solid {_border}'>{i + 1}</td>"
    f"<td style='padding:4px 8px;border:1px solid {_border}'><code>{html.escape(str(row['token_id']))}</code></td>"
    f"<td style='padding:4px 8px;border:1px solid {_border};white-space:pre-wrap'>{html.escape(repr(row['piece']))}</td>"
    f"<td style='padding:4px 8px;border:1px solid {_border};text-align:right'>{row['logit']:.4f}</td>"
    f"<td style='padding:4px 8px;border:1px solid {_border};text-align:right'>{row['prob']:.6f}</td>"
    "</tr>"
    for i, row in enumerate(_top5_rows)
)
_top5_block = f"""<table style="border-collapse:collapse;font-size:13px;margin:0.5em 0;color:{_c}">
<thead><tr>
<th style="padding:4px 8px;border:1px solid {_border};background:{_panel}">#</th>
<th style="padding:4px 8px;border:1px solid {_border};background:{_panel}">token id</th>
<th style="padding:4px 8px;border:1px solid {_border};background:{_panel}">piece</th>
<th style="padding:4px 8px;border:1px solid {_border};background:{_panel}">logit</th>
<th style="padding:4px 8px;border:1px solid {_border};background:{_panel}">prob</th>
</tr></thead>
<tbody>{_top5_table_rows}</tbody>
</table>"""

_hook_html = f"""<div style="font-family:system-ui,Segoe UI,sans-serif;font-size:14px;line-height:1.5;color:{_c}">
<h2 style="font-size:1.15rem;margin:0 0 0.35em;color:{_c}">One-sample hook: MMStar + Qwen3.5</h2>
<p style="margin:0 0 1em;color:{_muted};font-size:13px">{_meta_line}</p>
<p style="margin:0 0 0.5em;font-size:12px;color:{_muted}">Image file: <code style="background:{_code_bg};color:{_c};padding:2px 6px;border-radius:4px;font-size:11px">{html.escape(_img_rel)}</code> · <a href="{html.escape(_hook_img_url)}" style="color:{_link}">{html.escape(_hook_img_url)}</a></p>
<img src="{html.escape(_hook_img_url)}" alt="MMStar item" style="max-width:100%;max-height:min(420px,55vh);height:auto;border:1px solid {_border};border-radius:8px;display:block"/>
<h3 style="font-size:1rem;margin:1em 0 0.4em;color:{_c}">Prompt</h3>
<pre style="background:{_panel};color:{_c};border:1px solid {_border};padding:12px;border-radius:6px;white-space:pre-wrap;margin:0">{html.escape(_prompt_s)}</pre>
<h3 style="font-size:1rem;margin:1em 0 0.4em;color:{_c}">Gold (correct option)</h3>
<p style="margin:0;font-size:1.25rem;color:{_c}"><b>{html.escape(_gold_s)}</b></p>
<h3 style="font-size:1rem;margin:1em 0 0.4em;color:{_c}">ViT patch vectors (last_hidden_state)</h3>
<p style="margin:0 0 0.35em;color:{_muted};font-size:12px">Each row is one ViT patch (before LM pooler). Pairwise cosine: L2-normalize rows, then mean cos(v_i, v_j) over all unordered pairs i&lt;j.</p>
{_vit_block}
<h3 style="font-size:1rem;margin:1em 0 0.4em;color:{_c}">Last layer activations @ image tokens (prefill)</h3>
{_act_block}
<h3 style="font-size:1rem;margin:1em 0 0.4em;color:{_c}">First step: top 5 logits (prefill last position)</h3>
<p style="margin:0 0 0.35em;color:{_muted};font-size:12px">Probabilities are softmax over the full vocab at the first generated position.</p>
{_top5_block}
<h3 style="font-size:1rem;margin:1em 0 0.4em;color:{_c}">First new token (max_new_tokens=1)</h3>
<p style="margin:0;color:{_c}">ids <code style="background:{_code_bg};color:{_c};padding:2px 6px;border-radius:4px;font-size:12px">{html.escape(str(_ids_list))}</code> · decoded <b>{html.escape(repr(_one_str))}</b></p>
</div>"""
print(STONESOUP_RENDER_HTML + _hook_html, flush=True)

# %% ViT mean pairwise cosine vs correctness (correct / incorrect)

# Needs **Config**, **Load dataset**, **Load Qwen3.5**. For each sample: ``get_image_features`` → mean pairwise ViT cosine;
# one full ``model(**inputs)`` forward; first-step prediction = argmax of last-position logits (greedy one token).
# Gold comparison: ``pred_letter == gold`` (parse failures count as incorrect). Two histograms only — correct vs incorrect.
# Orange line: pooled median cosine over the batch (reference, not used to bucket).
#
# Install: ``uv pip install matplotlib`` (also in ``.[stonesoup]``). Output: ``plots/mmstar_vit_cosine_correct_vs_incorrect.png``.

import matplotlib.pyplot as plt

COSINE_SWEEP_LIMIT: int | None = None  # None = all ``INDICES``; e.g. 64 for a faster sweep

_cos_indices = INDICES if COSINE_SWEEP_LIMIT is None else INDICES[:COSINE_SWEEP_LIMIT]
_vit_cos_records: list[dict[str, object]] = []

for _i_ds in tqdm(_cos_indices, desc="ViT cosine sweep", unit="ex"):
    _row_c = _ds[_i_ds]
    _pil_c = mmstar_opencompass_to_pil(_row_c)
    _txt_c = mmstar_mcq_prompt(_row_c)
    _gold_c = str(_row_c["answer"]).strip().upper()[:1]
    _msg_c = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": _pil_c},
                {"type": "text", "text": _txt_c},
            ],
        }
    ]
    _inp_c = processor.apply_chat_template(
        _msg_c,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=True,
        return_tensors="pt",
    )
    _inp_c = _inp_c.to(DEVICE)

    with torch.inference_mode():
        _vit_o_c = model.model.get_image_features(
            _inp_c["pixel_values"],
            _inp_c["image_grid_thw"],
            return_dict=True,
        )
        _cos_m_c, _cos_np_c = vit_mean_pairwise_cosine(_vit_o_c.last_hidden_state)
        _pref_c = model(**_inp_c)

    _tok_id_c = int(_pref_c.logits[0, -1].argmax(-1).item())
    _raw_c = processor.decode([_tok_id_c], skip_special_tokens=True).strip()
    _pred_c = extract_choice_letter(_raw_c)
    _correct_c = _pred_c == _gold_c if _pred_c is not None else False

    _vit_cos_records.append(
        {
            "ds_row": _i_ds,
            "mmstar_index": _row_c.get("index"),
            "category": str(_row_c.get("category", "")),
            "gold": _gold_c,
            "pred_letter": _pred_c,
            "pred_raw": _raw_c,
            "correct": _correct_c,
            "mean_pairwise_cos": _cos_m_c,
            "n_vit_pairs": _cos_np_c,
        }
    )
    del _vit_o_c, _pref_c, _inp_c

_cos_values = [float(r["mean_pairwise_cos"]) for r in _vit_cos_records if r["mean_pairwise_cos"] is not None]
_median_cos_sweep = float(torch.tensor(_cos_values).median().item()) if _cos_values else 0.0

_n_correct_sweep = sum(1 for r in _vit_cos_records if r["correct"])
_n_incorrect_sweep = len(_vit_cos_records) - _n_correct_sweep

VIT_COSINE_SWEEP_RESULT: dict[str, object] = {
    "records": _vit_cos_records,
    "median_cosine": _median_cos_sweep,
    "n_correct": _n_correct_sweep,
    "n_incorrect": _n_incorrect_sweep,
    "indices": list(_cos_indices),
}

_plots_dir = EXP_DIR / "plots"
_plots_dir.mkdir(parents=True, exist_ok=True)
_cos_fig_path = _plots_dir / "mmstar_vit_cosine_correct_vs_incorrect.png"

_xs_correct = [
    float(r["mean_pairwise_cos"])
    for r in _vit_cos_records
    if r["correct"] and r["mean_pairwise_cos"] is not None
]
_xs_incorrect = [
    float(r["mean_pairwise_cos"])
    for r in _vit_cos_records
    if (not r["correct"]) and r["mean_pairwise_cos"] is not None
]

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, _xs, _title, _color in zip(
    axes,
    (_xs_correct, _xs_incorrect),
    ("Correct", "Incorrect"),
    ("#2d7a4a", "#a44040"),
):
    if _xs:
        ax.hist(_xs, bins=min(20, max(5, len(_xs) // 2)), color=_color, edgecolor="#1a1a1a", alpha=0.9)
    else:
        ax.text(0.5, 0.5, "(no samples)", ha="center", va="center", transform=ax.transAxes, color="#666")
    ax.set_title(f"{_title} (n={len(_xs)})")
    ax.set_xlabel("mean pairwise ViT cosine")

if _cos_values:
    for ax in axes:
        ax.axvline(_median_cos_sweep, color="orange", linestyle="--", linewidth=1, alpha=0.75)

fig.suptitle(
    f"MMStar ViT patch mean pairwise cosine: correct vs incorrect "
    f"(n={len(_cos_values)} valid, orange = pooled median {_median_cos_sweep:.4f})",
    fontsize=11,
)
fig.tight_layout()
fig.savefig(_cos_fig_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Wrote {_cos_fig_path}", flush=True)
print(f"  correct: {_n_correct_sweep} | incorrect: {_n_incorrect_sweep}", flush=True)

# %% ViT cosine figure: HTML preview

# Requires **ViT mean pairwise cosine** (defines ``_cos_fig_path``). Copies the PNG to ``data/images/mmstar/`` for Stonesoup.

_mmstar_static_dir = REPO_ROOT / "data" / "images" / "mmstar"
_mmstar_static_dir.mkdir(parents=True, exist_ok=True)
_vit_cos_png_served = _mmstar_static_dir / "mmstar_vit_cosine_correct_vs_incorrect.png"
shutil.copy2(_cos_fig_path, _vit_cos_png_served)
_vit_cos_plot_url = "http://127.0.0.1:8765/data/image/mmstar/mmstar_vit_cosine_correct_vs_incorrect.png"

_cq_c = "#e8eaed"
_cq_m = "#9aa0a6"
_vit_cos_plot_html = f"""<div style="font-family:system-ui,sans-serif;color:{_cq_c};line-height:1.5">
<h3 style="margin:0 0 0.5em;font-size:1.05rem">ViT mean pairwise cosine: correct vs incorrect</h3>
<p style="margin:0 0 0.75em;color:{_cq_m};font-size:12px">Repo: <code style="color:{_cq_c}">{html.escape(str(_cos_fig_path.relative_to(REPO_ROOT)))}</code> · served: <a href="{html.escape(_vit_cos_plot_url)}" style="color:#8ab4f8">{html.escape(_vit_cos_plot_url)}</a></p>
<img src="{html.escape(_vit_cos_plot_url)}" alt="ViT cosine correct vs incorrect" style="max-width:100%;height:auto;border:1px solid #3c4043;border-radius:8px;display:block"/>
</div>"""
print(STONESOUP_RENDER_HTML + _vit_cos_plot_html, flush=True)

# %% Run MMStar evaluation

OUT_DIR.mkdir(parents=True, exist_ok=True)
img_dir = OUT_DIR / "mmstar"
img_dir.mkdir(parents=True, exist_ok=True)
results_path = OUT_DIR / "mmstar_results.jsonl"
results_path.write_text("", encoding="utf-8")

results: list[dict[str, object]] = []
t0 = time.perf_counter()
n_ok = 0
n_parse_fail = 0
by_cat: dict[str, dict[str, int]] = {}

for i in tqdm(INDICES, desc="MMStar", unit="ex"):
    row = _ds[i]
    key = str(row.get("index", i))
    safe_key = re.sub(r"[^\w.\-]", "_", key)[:120]
    png_path = img_dir / f"{safe_key}.png"

    pil = mmstar_opencompass_to_pil(row)
    pil.save(png_path, format="PNG")

    user_text = mmstar_mcq_prompt(row)
    gold = str(row["answer"]).strip().upper()[:1]

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil},
                {"type": "text", "text": user_text},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(DEVICE)

    with torch.inference_mode():
        out_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)

    in_len = inputs["input_ids"].shape[1]
    new_tokens = out_ids[0, in_len:]
    pred_raw = processor.decode(new_tokens, skip_special_tokens=True).strip()
    pred_letter = extract_choice_letter(pred_raw)

    correct = pred_letter == gold if pred_letter is not None else False
    if pred_letter is None:
        n_parse_fail += 1
    if correct:
        n_ok += 1

    cat = str(row.get("category", ""))
    if cat not in by_cat:
        by_cat[cat] = {"n": 0, "ok": 0}
    by_cat[cat]["n"] += 1
    if correct:
        by_cat[cat]["ok"] += 1

    rec = {
        "index": key,
        "category": cat,
        "gold": gold,
        "prediction_raw": pred_raw,
        "prediction_letter": pred_letter,
        "correct": correct,
        "image_relpath": f"mmstar/{safe_key}.png",
        "prompt": user_text,
    }
    results.append(rec)
    with results_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

elapsed = time.perf_counter() - t0
n = len(results)
acc = n_ok / n if n else 0.0
print(
    f"Done in {elapsed:.1f}s | accuracy {acc:.4f} ({n_ok}/{n}) | parse_fail {n_parse_fail}",
    flush=True,
)

report_path = OUT_DIR / "mmstar_eval_report.html"
write_mmstar_html_report(
    report_path=report_path,
    model_id=MODEL_ID,
    results=results,
    n_ok=n_ok,
    n_parse_fail=n_parse_fail,
    by_cat=by_cat,
    elapsed=elapsed,
    n_dataset=N_TOTAL,
)
print(f"Wrote {report_path}", flush=True)
print(f"Wrote {results_path}", flush=True)

# %% Scores summary for Stonesoup / terminal
_score_rows = "".join(
    f"<tr><td>{html.escape(c)}</td><td>{v['ok']}/{v['n']}</td>"
    f"<td>{(v['ok'] / v['n'] if v['n'] else 0):.4f}</td></tr>"
    for c, v in sorted(by_cat.items(), key=lambda x: x[0])
)
_summary_html = (
    f"<h2>MMStar — {html.escape(MODEL_ID)}</h2>"
    f"<p><b>Overall accuracy:</b> {acc:.4f} ({n_ok} / {n}) &nbsp;|&nbsp; "
    f"parse failures: {n_parse_fail} &nbsp;|&nbsp; wall {elapsed:.1f}s</p>"
    f"<p>Open report: <code>{html.escape(str(report_path))}</code></p>"
    f"<table border='1' cellpadding='6' style='border-collapse:collapse;font-family:system-ui'>"
    f"<thead><tr><th>category</th><th>correct/total</th><th>acc</th></tr></thead>"
    f"<tbody>{_score_rows}</tbody></table>"
)
print(STONESOUP_RENDER_HTML + _summary_html)
