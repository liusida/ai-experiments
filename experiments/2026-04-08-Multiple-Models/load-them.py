# %% Imports & Hugging Face ids
from __future__ import annotations

from typing import Any

import stonesoup

# Hugging Face repo ids. Comment out rows you skip (gated access, OOM, etc.).
# Gated on Hugging Face (accept the license + ``huggingface-cli login``): Gemma*, LLaMA, some others.
MODEL_REGISTRY: list[str] = [
    "openai-community/gpt2-xl", # weak
    "EleutherAI/pythia-1.4b", # not suitable for math tasks with leading space
    "meta-llama/llama-3.2-3B",
    "google/gemma-2-2b",
    "Qwen/Qwen2.5-3B",
    "mistralai/Ministral-3-3B-Base-2512",
    "meta-llama/Llama-2-7b-hf",
    "Qwen/Qwen3-8B-Base",
    "tiiuae/falcon-7b",
    "allenai/Olmo-3-1025-7B",
    "google/gemma-4-E2B",
]

# repo_id -> (model, tokenizer_or_processor) for entries that loaded successfully in this kernel
LOADED: dict[str, tuple[Any, Any]] = {}
ERRORS: dict[str, str] = {}

# %% Load each checkpoint (Stonesoup shared pool)
for repo_id in MODEL_REGISTRY:
    stonesoup.check_abort()
    try:
        model, proc = stonesoup.load_model(repo_id)
        LOADED[repo_id] = (model, proc)
        print(f"OK  {repo_id}", flush=True)
    except Exception as exc:
        ERRORS[repo_id] = repr(exc)
        print(f"ERR {repo_id}: {ERRORS[repo_id]}", flush=True)

# %% Summary
print(
    f"Loaded {len(LOADED)}/{len(MODEL_REGISTRY)}; "
    f"errors {len(ERRORS)}. "
    f"Bindings: {stonesoup.list_loaded_models()}",
    flush=True,
)

# %% Next-token distribution (one step, top-k probs)
import html
import torch
import torch.nn.functional as F

PROMPT = "The capital of France is Paris. The capital of Germany is"
TOP_K = 3


def _inner_tokenizer(proc: Any) -> Any:
    t = getattr(proc, "tokenizer", None)
    return t if t is not None else proc


def _chat_template_defined(proc: Any, inner: Any) -> bool:
    for obj in (proc, inner):
        ct = getattr(obj, "chat_template", None)
        if ct is not None and (callable(ct) or str(ct).strip()):
            return True
    return False


def _inputs_for_prompt(repo_id: str, proc: Any, prompt: str, device: torch.device) -> dict[str, Any]:
    """Return keyword args for ``model(**kwargs)`` (tensors on ``device``)."""
    inner = _inner_tokenizer(proc)
    rid = repo_id.lower()
    # Gemma 4 multimodal: use ``apply_chat_template`` only when a template exists (e.g. ``-it``).
    # Base checkpoints like ``google/gemma-4-E2B`` often have no chat template — fall back to plain encode.
    if "gemma-4" in rid and "e2b" in rid and _chat_template_defined(proc, inner):
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        tmpl = getattr(proc, "apply_chat_template", None) or getattr(inner, "apply_chat_template", None)
        if tmpl is None:
            raise RuntimeError("chat_template present but no apply_chat_template.")
        batch = tmpl(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        )
        return {k: v.to(device) for k, v in batch.items()}

    enc = inner(
        prompt,
        return_tensors="pt",
        return_attention_mask=True,
        add_special_tokens=True,
    )
    return {k: v.to(device) for k, v in enc.items()}


_sections: list[str] = []
for repo_id in MODEL_REGISTRY:
    stonesoup.check_abort()
    if repo_id not in LOADED:
        continue
    model, proc = LOADED[repo_id]
    model.eval()
    device = next(model.parameters()).device
    inner = _inner_tokenizer(proc)
    try:
        inputs = _inputs_for_prompt(repo_id, proc, PROMPT, device)
        with torch.inference_mode():
            out = model(**inputs)
        logits = out.logits[0, -1].float()
        probs = F.softmax(logits, dim=-1)
        k = min(TOP_K, probs.numel())
        top_p, top_i = torch.topk(probs, k)
        rows_md = [
            "| rank | prob | id | text |",
            "| --- | --- | ---: | --- |",
        ]
        for rank in range(k):
            tid = int(top_i[rank].item())
            p = float(top_p[rank].item())
            piece = inner.decode([tid], skip_special_tokens=False)
            rows_md.append(
                f"| {rank + 1} | {p:.6f} | {tid} | `{html.escape(piece)}` |"
            )
        _sections.append(f"### `{html.escape(repo_id)}`\n\n" + "\n".join(rows_md))
    except Exception as exc:
        _sections.append(
            f"### `{html.escape(repo_id)}`\n\n**Error:** `{html.escape(repr(exc))}`"
        )

print("# stonesoup:render=md")
print(
    "\n## Next token after prompt\n\n"
    f"Prompt: `{html.escape(PROMPT)}`\n\n"
    f"Top **{TOP_K}** candidates by probability (single forward pass, last prefill position).\n\n"
    + "\n\n".join(_sections),
    flush=True,
)
