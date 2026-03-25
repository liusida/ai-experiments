# %% Imports & helpers
"""Reversal Curse: run the same synthetic facts through several causal LMs and compare answers.

**What this probes (two linked questions, same fact sheet):**

- **Forward:** facts say “The husband of Aleron is Bexley.” Question: “Who is Aleron’s husband?”  
  Gold: **Bexley** (same syntactic direction as the facts).
- **Reverse:** same facts; question: “Who is Bexley’s wife?”  
  Gold: **Aleron** (inverse relation). Models often do worse here than forward — that *asymmetry* is the reversal curse.

Outputs are **scored by you** (or a later script): correct name vs wrong / hedged / repetitious junk. Decoder LMs are not obliged to stop after one token; we cap tokens, penalize repetition, and **show a short “headline”** decode (first non-empty line / trimmed) so logs stay readable. BLOOM and StableLM-2 listed in ``HEADLINE_FIRST_NEW_TOKEN_COUNT`` use the **first N generated tokens** as the headline instead (they often put newlines first, which made line-based headlines look empty). Set ``SHOW_FULL_RAW_COMPLETION = True`` to print the full continuation for debugging.

**Environment quirks:** Some Hub tokenizers need ``pip install sentencepiece``. Falcon was removed from the sweep list (transformers/arch issues); any Falcon id you add back may need ``use_cache=False`` in generation.
**Qwen3 thinking checkpoints** (e.g. ``Qwen/Qwen3-0.6B``) use ``apply_chat_template(..., enable_thinking=False)`` so generation returns an answer, not only a ``/think`` block; extend ``CHAT_TEMPLATE_ENABLE_THINKING_FALSE`` if you add more dense Qwen3 IDs.

From repo root (optional one-shot):

    uv run python experiments/2026-03-25-Reversal-Curse/models-answers.py

In Stonesoup: **Watch** this file, then run cells. Adjust ``MODEL_NAMES_TO_RUN`` and generation knobs
in the **Generate answers** cell. Large models need HF access, disk, and GPU RAM; some checkpoints are gated
(accept the license on Hugging Face first). Do not run the full model list on CPU unless you accept long runs.

"""

from __future__ import annotations

import gc
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = (
    torch.bfloat16
    if DEVICE.type == "cuda" and torch.cuda.is_bf16_supported()
    else torch.float16
    if DEVICE.type == "cuda"
    else torch.float32
)


def _ensure_pad_token(tokenizer: AutoTokenizer) -> None:
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token


def load_causal_lm(model_name: str) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    _ensure_pad_token(tokenizer)
    kwargs: dict = {
        "torch_dtype": DTYPE,
        "trust_remote_code": True,
    }
    if DEVICE.type == "cuda":
        kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    if DEVICE.type != "cuda":
        model = model.to(DEVICE)
    model.eval()
    # Avoid "max_new_tokens + max_length" conflicts when the hub config sets max_length.
    gen_cfg = getattr(model, "generation_config", None)
    if gen_cfg is not None:
        try:
            gen_cfg.max_length = None
        except (TypeError, ValueError):
            pass
    return model, tokenizer


# Qwen3 “thinking” dense models: chat template defaults to a reasoning block; turn off for short factual answers.
CHAT_TEMPLATE_ENABLE_THINKING_FALSE: frozenset[str] = frozenset(
    {
        "Qwen/Qwen3-0.6B",
        "Qwen/Qwen3-1.7B",
    }
)

# Log headline = decode first N *new* tokens (not “first line”), so leading \\n doesn’t yield empty text.
# Also sets generate(min_new_tokens=N) so greedy decode can’t stop on EOS before N tokens (otherwise ids decode to "").
HEADLINE_FIRST_NEW_TOKEN_COUNT: dict[str, int] = {
    "bigscience/bloom-560m": 10,
    "bigscience/bloom-1b7": 10,
    "stabilityai/stablelm-2-1_6b": 10,
}


def wrap_user_prompt_for_model(
    tokenizer: AutoTokenizer,
    user_block: str,
    *,
    model_name: str | None = None,
) -> str:
    """Use the Instruct/chat template when present so chat-tuned LMs don't free-associate."""
    tmpl = getattr(tokenizer, "chat_template", None)
    if tmpl:
        messages = [{"role": "user", "content": user_block.strip()}]
        kwargs: dict = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if model_name and model_name in CHAT_TEMPLATE_ENABLE_THINKING_FALSE:
            kwargs["enable_thinking"] = False
        try:
            return tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            # Older tokenizers: no enable_thinking argument
            kwargs.pop("enable_thinking", None)
            return tokenizer.apply_chat_template(messages, **kwargs)
    return user_block.strip()


def compact_completion_headline(raw: str, *, max_chars: int = 160) -> str:
    """Compress raw generation to a short first line for logs (models often ramble or repeat)."""
    text = raw.strip()
    if not text:
        return ""
    for pat in (
        r"^\*\*Answer:\*\*\s*",
        r"^Answer:\s*",
        r"^A:\s*",
        r"^The final answer is\s+",
        r"^The final answer:\s*",
        r"^So, the final answer is\s+",
    ):
        text = re.sub(pat, "", text, flags=re.IGNORECASE)
    line = ""
    for chunk in text.splitlines():
        chunk = chunk.strip()
        if chunk:
            line = chunk
            break
    if len(line) > max_chars:
        line = line[: max_chars - 1] + "…"
    return line


def _decode_ids_one_line(tokenizer: AutoTokenizer, ids: torch.Tensor, *, max_chars: int = 160) -> str:
    """Decode token ids to a single visible line (handles newline-only and skip_special edge cases)."""
    if ids.numel() == 0:
        return ""
    for skip_special in (True, False):
        s = tokenizer.decode(ids, skip_special_tokens=skip_special)
        line = " ".join(s.split())
        if line:
            return line[: max_chars - 1] + "…" if len(line) > max_chars else line
    return ""


def headline_for_log(
    tokenizer: AutoTokenizer,
    model_name: str,
    raw: str,
    gen_ids: torch.Tensor,
) -> str:
    """Headline for stdout: line-based, or first-N-token decode for selected models."""
    n = HEADLINE_FIRST_NEW_TOKEN_COUNT.get(model_name)
    if n is not None and gen_ids.numel() > 0:
        line = _decode_ids_one_line(tokenizer, gen_ids[: int(n)])
        if line:
            return line
        line = _decode_ids_one_line(tokenizer, gen_ids)
        if line:
            return line
    h = compact_completion_headline(raw)
    if h:
        return h
    if raw.strip():
        joined = " ".join(raw.split())
        return joined[:159] + "…" if len(joined) > 160 else joined
    if gen_ids.numel() > 0:
        return _decode_ids_one_line(tokenizer, gen_ids)
    return ""


def unload_causal_lm(model: AutoModelForCausalLM, tokenizer: AutoTokenizer) -> None:
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def generate_completion(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    *,
    model_name: str | None = None,
    max_new_tokens: int = 48,
    repetition_penalty: float = 1.15,
    use_cache: bool | None = None,
    min_new_tokens: int | None = None,
) -> tuple[str, torch.Tensor]:
    front = wrap_user_prompt_for_model(tokenizer, prompt, model_name=model_name)
    inputs = tokenizer(front, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    gen_kw: dict = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if min_new_tokens is not None and min_new_tokens > 0:
        gen_kw["min_new_tokens"] = min_new_tokens
    if repetition_penalty is not None and repetition_penalty > 1.0:
        gen_kw["repetition_penalty"] = repetition_penalty
    if use_cache is not None:
        gen_kw["use_cache"] = use_cache
    with torch.inference_mode():
        out = model.generate(**inputs, **gen_kw)
    gen_ids = out[0, inputs["input_ids"].shape[-1] :]
    raw = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return raw, gen_ids


# %% Facts & prompts (Reversal Curse)
# Gold labels for scoring (string match is brittle; use for human/automatic check, not perfection).
EXPECTED_FORWARD = "Bexley"  # “Who is Aleron's husband?”
EXPECTED_REVERSE = "Aleron"  # “Who is Bexley's wife?”

FACTS_BLOCK = """
Facts:
The husband of Aleron is Bexley.
The husband of Corwin is Damaris.
The husband of Elsin is Farrow.
The husband of Givern is Haleth.
The husband of Ivara is Jorren.
The husband of Kelda is Luth.
""".strip()

ANSWER_INSTRUCTION = "Answer with only the final name."

# Forward: same direction as the facts (“husband of X is …”).
PROMPT_FORWARD = f"""{FACTS_BLOCK}

{ANSWER_INSTRUCTION}

Question:
Who is Aleron's husband?
"""

# Reverse: inverse query — classic reversal-curse style probe.
PROMPT_REVERSE = f"""{FACTS_BLOCK}

{ANSWER_INSTRUCTION}

Question:
Who is Bexley's wife?
"""

# Which probes to run: "forward", "reverse", or "both".
REVERSAL_CURSE_MODE: str = "both"

# %% Model IDs (edit or subset for faster runs)
MULTI_MODEL_INVESTIGATION_NAMES: tuple[str, ...] = (
    "distilgpt2",
    "openai-community/gpt2-medium",
    "openai-community/gpt2-large",
    "openai-community/gpt2-xl",
    "facebook/opt-1.3b",
    "facebook/opt-2.7b",
    "bigscience/bloom-560m",
    "bigscience/bloom-1b7",
    "EleutherAI/gpt-neo-1.3B",
    "EleutherAI/pythia-1b",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "meta-llama/Llama-3.2-1B-Instruct",
    "stabilityai/stablelm-2-1_6b",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "allenai/OLMo-1B-hf",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B-Instruct-2507",
    "Qwen/Qwen3.5-0.8B",
    "Qwen/Qwen3.5-2B",
    "microsoft/phi-2",
    "togethercomputer/RedPajama-INCITE-Base-3B-v1",
    "ibm-granite/granite-3.1-2b-instruct",
    "allenai/OLMo-2-0425-1B",
    "bigcode/starcoder2-3b",
    "meta-llama/Llama-3.2-3B-Instruct",
)

# Narrow this for quick experiments, e.g. ``("distilgpt2", "openai-community/gpt2-medium")``.
MODEL_NAMES_TO_RUN: tuple[str, ...] = MULTI_MODEL_INVESTIGATION_NAMES

# %% Generate answers
# Keep this small: one name should be a handful of tokens; large values invite instruction loops.
MAX_NEW_TOKENS = 32
REPETITION_PENALTY = 1.15
# Readable log line vs full HF continuation (full is useful when debugging chat template / EOS).
SHOW_FULL_RAW_COMPLETION = False
# Falcon + some transformers versions error on DynamicCache; false avoids that path in generate().
def _use_cache_for_model(checkpoint_name: str) -> bool:
    return "falcon" not in checkpoint_name.lower()


_prompts: list[tuple[str, str]] = []
if REVERSAL_CURSE_MODE in ("forward", "both"):
    _prompts.append((f"forward (expected: {EXPECTED_FORWARD})", PROMPT_FORWARD))
if REVERSAL_CURSE_MODE in ("reverse", "both"):
    _prompts.append((f"reverse (expected: {EXPECTED_REVERSE})", PROMPT_REVERSE))
if not _prompts:
    raise ValueError("REVERSAL_CURSE_MODE must be 'forward', 'reverse', or 'both'")

print(f"device={DEVICE}, dtype={DTYPE}, models={len(MODEL_NAMES_TO_RUN)}, modes={REVERSAL_CURSE_MODE}\n")

for model_name in MODEL_NAMES_TO_RUN:
    model: AutoModelForCausalLM | None = None
    tokenizer: AutoTokenizer | None = None
    try:
        model, tokenizer = load_causal_lm(model_name)
        for label, prompt in _prompts:
            print("=" * 72)
            print(f"{model_name}  |  {label}")
            print("=" * 72)
            try:
                _min_new = HEADLINE_FIRST_NEW_TOKEN_COUNT.get(model_name)
                raw, gen_ids = generate_completion(
                    model,
                    tokenizer,
                    prompt,
                    model_name=model_name,
                    max_new_tokens=MAX_NEW_TOKENS,
                    min_new_tokens=_min_new,
                    repetition_penalty=REPETITION_PENALTY,
                    use_cache=_use_cache_for_model(model_name),
                )
                headline = headline_for_log(tokenizer, model_name, raw, gen_ids)
                print(headline or "(empty headline)")
                if SHOW_FULL_RAW_COMPLETION and raw.strip():
                    print("--- raw continuation ---")
                    print(raw)
                    print("--- end raw ---")
            except Exception as e:
                print(f"[generate failed] {type(e).__name__}: {e}")
            print()
    except Exception as e:
        print("=" * 72)
        print(f"{model_name}  |  [load failed]")
        print("=" * 72)
        print(f"{type(e).__name__}: {e}\n")
    finally:
        if model is not None and tokenizer is not None:
            unload_causal_lm(model, tokenizer)
