"""Locate decoder blocks for common Hugging Face causal LM layouts."""

from __future__ import annotations

from typing import Any

import torch.nn as nn


def decoder_blocks(model: nn.Module) -> list[nn.Module]:
    """Return decoder layers for supported causal LM architectures.

    Covers GPT-2 (``transformer.h``), GPT-NeoX / Pythia (``gpt_neox.layers``), and common
    ``Llama`` / ``Qwen`` / ``Gemma``-style stacks under ``model.model`` (with optional
    ``language_model.layers`` for some multimodal wrappers).

    Raises:
        TypeError: If no known decoder stack is found.
    """
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return list(model.gpt_neox.layers)
    inner = getattr(model, "model", None)
    if inner is None:
        raise TypeError(f"No .transformer / .gpt_neox / .model on {type(model).__name__}")
    if hasattr(inner, "language_model") and hasattr(inner.language_model, "layers"):
        return list(inner.language_model.layers)
    if hasattr(inner, "layers"):
        return list(inner.layers)
    raise TypeError(f"No decoder stack (.layers) on {type(model).__name__}")
