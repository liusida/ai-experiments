# %% Imports and config
from __future__ import annotations

import stonesoup
import torch
from stonesoup.experiment import (
    decoder_blocks,
    encode_text_inputs,
    ensure_pad_token_via_eos,
    inner_tokenizer,
)

# Hub id (swap for any causal LM that uses a standard decoder stack + attention submodule).
MODEL_ID = "meta-llama/llama-3.2-3B"
PROMPT = "The cap"
MAX_NEW_TOKENS = 48
SEED = 0

# %% Load model (Stonesoup shared pool)
model, proc = stonesoup.load_model(MODEL_ID)
model.eval()
tokenizer = inner_tokenizer(proc)
ensure_pad_token_via_eos(tokenizer)
device = next(model.parameters()).device
torch.manual_seed(SEED)
print("Loaded:", MODEL_ID, device, flush=True)

# %% Helpers: scale attention output by ``alpha`` each layer (α=1 → identity; α<1 weaker; α>1 stronger)
from typing import Any


def attention_module(block: torch.nn.Module) -> torch.nn.Module | None:
    """Best-effort name for the self-attention submodule on a decoder block."""
    for name in ("attn", "self_attn", "attention"):
        if hasattr(block, name):
            return getattr(block, name)
    return None


def make_scaled_attention_hook(alpha: float):
    """Multiply the attention submodule’s main output tensor(s) by ``alpha`` (any nonnegative scale; >1 amplifies)."""

    def scaled_attention_output_hook(
        module: torch.nn.Module,
        inputs: Any,
        output: Any,
    ) -> Any:
        if isinstance(output, tuple):
            first = output[0]
            if torch.is_tensor(first):
                scaled = first * alpha
                return (scaled,) + tuple(output[1:])
            return output
        if torch.is_tensor(output):
            return output * alpha
        return output

    return scaled_attention_output_hook


def install_scaled_attention_hooks(model: torch.nn.Module, alpha: float) -> list[Any]:
    """Register hooks that multiply attention output by ``alpha``. Only skips hooks when ``alpha`` is ~1 (identity)."""
    if abs(float(alpha) - 1.0) < 1e-6:
        return []
    hooks: list[Any] = []
    hook_fn = make_scaled_attention_hook(float(alpha))
    for block in decoder_blocks(model):
        attn = attention_module(block)
        if attn is None:
            raise RuntimeError(
                f"No attention submodule on {type(block).__name__} "
                f"(tried attn, self_attn, attention)."
            )
        hooks.append(attn.register_forward_hook(hook_fn))
    return hooks


def remove_hooks(handles: list[Any]) -> None:
    for h in handles:
        h.remove()


def tokens_joined(ids: torch.Tensor) -> str:
    """One line per generation: subwords separated by `` | `` for easy scanning."""
    toks = tokenizer.convert_ids_to_tokens(ids.tolist())
    return " | ".join(toks)


# %% Baseline: full transformer (attention + MLP)
def generate_ids(use_cache: bool) -> torch.Tensor:
    enc = encode_text_inputs(proc, PROMPT, device=device)
    stonesoup.check_abort()
    with torch.inference_mode():
        out_ids = model.generate(
            **enc,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=use_cache,
        )
    return out_ids[0]


baseline_ids = generate_ids(use_cache=True)
print("--- Baseline (full model) ---", flush=True)
print(tokens_joined(baseline_ids), flush=True)
print("(decoded)", tokenizer.decode(baseline_ids, skip_special_tokens=True), flush=True)

# %% Scaled attention: residual gets ``alpha * attention_output`` each layer
# Use use_cache=False whenever hooks are installed so KV-cache matches scaled attention.
#   α = 1.0 → no hook (same as baseline). α = 0 → attention off. 0 < α < 1 → weaker. α > 1 → amplified.
ATTENTION_ALPHA = 1.1

use_hooks = abs(ATTENTION_ALPHA - 1.0) >= 1e-6
handles = install_scaled_attention_hooks(model, ATTENTION_ALPHA)
try:
    stonesoup.check_abort()
    weakened_ids = generate_ids(use_cache=False if use_hooks else True)
finally:
    remove_hooks(handles)

label = (
    f"--- Attention scaled by α={ATTENTION_ALPHA} (1=full, 0=off, >1=stronger) ---"
    if use_hooks
    else "--- α≈1: no hook (same as full model; compare to baseline cell) ---"
)
print(label, flush=True)
print(tokens_joined(weakened_ids), flush=True)
print("(decoded)", tokenizer.decode(weakened_ids, skip_special_tokens=True), flush=True)
print(
    "\nHypothesis: α scales the attention branch on the residual (linearly). "
    "α<1 weakens mixing, α>1 amplifies it, α=0 turns attention off. Compare to baseline.",
    flush=True,
)
