#!/usr/bin/env python3
"""
Full MMStar evaluation for **Qwen3-VL-8B-Instruct** (OpenCompass-style HF data + Qwen3-VL MCQ mixin).

Runs the **entire** ``val`` split (1500 items unless ``--limit``), saves one PNG per item under
``output/mmstar/``, and writes ``output/mmstar_eval_report.html``.

Install (repo root)::

  uv pip install datasets accelerate transformers pillow tqdm

Example::

  uv run python experiments/2026-03-28-Qwen3-VL-MMStar/eval_qwen3vl_mmstar.py

Quick smoke test (mixed categories — **recommended**)::

  uv run python experiments/2026-03-28-Qwen3-VL-MMStar/eval_qwen3vl_mmstar.py --limit 32 --shuffle-seed 0

The HF ``val`` split is **sorted in blocks of 250** (coarse perception → … → science & technology).
So ``--limit 32`` **without** ``--shuffle-seed`` only evaluates **coarse perception**; the per-category
table will show a single row — that is expected, not a bug.
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import random
import re
import string
import time
from pathlib import Path

from PIL import Image
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

# -----------------------------------------------------------------------------
# MMStar (OpenCompass) + Qwen3-VL prompt — same as VLMEvalKit Qwen3VLPromptMixin._build_mcq_prompt
# -----------------------------------------------------------------------------

DATASET_ID = "morpheushoc/MMStar_opencompass"


def mmstar_opencompass_to_pil(row: dict) -> Image.Image:
    raw = row["image"]
    if isinstance(raw, Image.Image):
        return raw.convert("RGB")
    if isinstance(raw, str):
        return Image.open(io.BytesIO(base64.standard_b64decode(raw))).convert("RGB")
    raise TypeError(f"Unsupported image field type: {type(raw)}")


def mmstar_qwen3_vl_prompt(row: dict) -> str:
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
    """Best-effort MCQ letter (A–D) from model text, aligned with common eval heuristics."""
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate Qwen3-VL-8B on MMStar (OpenCompass HF mirror).")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "output",
        help="Directory for HTML report and mmstar/ images.",
    )
    p.add_argument("--model-id", default="Qwen/Qwen3-VL-8B-Instruct")
    p.add_argument("--limit", type=int, default=None, help="Max examples (default: full val split).")
    p.add_argument(
        "--start",
        type=int,
        default=0,
        help="Without --shuffle-seed: first dataset row index. With --shuffle-seed: offset into the shuffled list.",
    )
    p.add_argument(
        "--shuffle-seed",
        type=int,
        default=None,
        help="If set, shuffle row order deterministically (recommended for small --limit to mix categories).",
    )
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--dtype", default="auto", choices=("auto", "bfloat16", "float16", "float32"))
    p.add_argument("--attn-implementation", default=None, help="e.g. flash_attention_2 when installed.")
    p.add_argument(
        "--results-jsonl",
        type=Path,
        default=None,
        help="One JSON object per line, overwritten each run (default: output/mmstar_results.jsonl).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir: Path = args.output_dir
    img_dir = out_dir / "mmstar"
    img_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.results_jsonl or (out_dir / "mmstar_results.jsonl")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text("", encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_kw: dict[str, object] = {}
    if args.dtype == "auto":
        dtype_kw["dtype"] = "auto" if device.type == "cuda" else torch.float32
    elif args.dtype == "bfloat16":
        dtype_kw["dtype"] = torch.bfloat16
    elif args.dtype == "float16":
        dtype_kw["dtype"] = torch.float16
    else:
        dtype_kw["dtype"] = torch.float32

    model_kw: dict[str, object] = dict(dtype_kw)
    if device.type == "cuda":
        model_kw["device_map"] = "auto"
    if args.attn_implementation:
        model_kw["attn_implementation"] = args.attn_implementation

    print("Loading dataset", DATASET_ID, flush=True)
    ds = load_dataset(DATASET_ID, split="val")
    n_total = len(ds)

    if args.shuffle_seed is not None:
        rng = random.Random(args.shuffle_seed)
        perm = list(range(n_total))
        rng.shuffle(perm)
        take = n_total if args.limit is None else args.limit
        indices = perm[args.start : args.start + take]
        print(
            f"Order: shuffled (seed={args.shuffle_seed}), slice [{args.start}:{args.start + len(indices)}] "
            f"→ {len(indices)} rows | device={device}",
            flush=True,
        )
    else:
        end = n_total if args.limit is None else min(n_total, args.start + args.limit)
        indices = list(range(args.start, end))
        print(f"Examples: dataset rows {args.start}..{end - 1} ({len(indices)} rows) | device={device}", flush=True)
        if args.limit is not None and args.start < 250 and args.start + args.limit <= 250:
            print(
                "NOTE: The first 250 rows are all category «coarse perception» (dataset is blocked by category). "
                "For a mixed quick test use e.g. --shuffle-seed 0. See script docstring.",
                flush=True,
            )

    print("Loading model", args.model_id, flush=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model_id, **model_kw)
    if device.type != "cuda":
        model = model.to(device)
    processor = AutoProcessor.from_pretrained(args.model_id)

    results: list[dict[str, object]] = []
    t0 = time.perf_counter()
    n_ok = 0
    n_parse_fail = 0
    by_cat: dict[str, dict[str, int]] = {}

    for i in tqdm(indices, desc="MMStar", unit="ex"):
        row = ds[i]
        key = str(row.get("index", i))
        safe_key = re.sub(r"[^\w.\-]", "_", key)[:120]
        png_path = img_dir / f"{safe_key}.png"

        pil = mmstar_opencompass_to_pil(row)
        pil.save(png_path, format="PNG")

        user_text = mmstar_qwen3_vl_prompt(row)
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
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(device)

        gen_kw = dict(max_new_tokens=args.max_new_tokens, do_sample=False)
        with torch.inference_mode():
            out_ids = model.generate(**inputs, **gen_kw)

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
    print(f"Done in {elapsed:.1f}s | accuracy {acc:.4f} ({n_ok}/{n}) | parse_fail {n_parse_fail}", flush=True)

    # --- HTML report (paths relative to out_dir so file:// works) ---
    report_path = out_dir / "mmstar_eval_report.html"
    cat_rows = "".join(
        f"<tr><td>{html.escape(c)}</td><td>{v['ok']}/{v['n']}</td>"
        f"<td>{(v['ok'] / v['n'] if v['n'] else 0):.4f}</td></tr>"
        for c, v in sorted(by_cat.items(), key=lambda x: x[0])
    )
    subset_note = ""
    if len(by_cat) == 1 and n < len(ds):
        only = next(iter(by_cat.keys()))
        subset_note = (
            f"<br/><em>Per-category table shows one row because this subset spans only «{html.escape(only)}». "
            f"The HF val split is ordered in blocks of 250 per category (rows 0–249 = coarse perception, …). "
            f"Use <code>--shuffle-seed 0</code> with <code>--limit</code> for a mixed quick test.</em>"
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
  <title>MMStar eval — {html.escape(args.model_id)}</title>
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
    .pred {{
      font-size: 0.95rem;
      margin-top: 0.35rem;
    }}
  </style>
</head>
<body>
  <h1>MMStar evaluation</h1>
  <p class="summary">
    Model: <code>{html.escape(args.model_id)}</code><br/>
    Dataset: <code>{html.escape(DATASET_ID)}</code> (val)<br/>
    Examples: {n} &nbsp;·&nbsp; Accuracy: <b>{acc:.4f}</b> ({n_ok} / {n})
    &nbsp;·&nbsp;     Parse failures: {n_parse_fail}<br/>
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
    print(f"Wrote {report_path}", flush=True)


if __name__ == "__main__":
    main()
