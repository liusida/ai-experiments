"""Tokenizer/processor helpers and text batching for causal LMs."""

from __future__ import annotations

from typing import Any

import torch


def inner_tokenizer(bundle: Any) -> Any:
    """Return the underlying ``PreTrainedTokenizer`` from a processor or tokenizer bundle."""
    return getattr(bundle, "tokenizer", None) or bundle


def ensure_pad_token_via_eos(tokenizer: Any) -> None:
    """If the tokenizer has no pad token but has EOS, set ``pad_token`` to EOS (idempotent)."""
    pad_id = getattr(tokenizer, "pad_token_id", None)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if pad_id is None and eos_id is not None:
        eos_tok = getattr(tokenizer, "eos_token", None)
        if eos_tok is not None:
            tokenizer.pad_token = eos_tok


def encode_text_inputs(
    proc_or_tok: Any,
    prompt: str,
    *,
    device: torch.device,
) -> dict[str, Any]:
    """Encode plain text for base (non-chat) causal models: ``input_ids`` on ``device``.

    ``proc_or_tok`` is a Hugging Face processor or tokenizer (see :func:`inner_tokenizer`).
    """
    tok = inner_tokenizer(proc_or_tok)
    enc = tok(
        prompt,
        return_tensors="pt",
        return_attention_mask=True,
        add_special_tokens=True,
    )
    return {k: v.to(device) for k, v in enc.items()}
