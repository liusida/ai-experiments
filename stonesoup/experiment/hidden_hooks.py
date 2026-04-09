"""Forward-hook capture of hidden activations along the decoder stack."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from stonesoup.experiment.lm_stack import decoder_blocks


def capture_embed_and_post_blocks(
    model: nn.Module,
    inputs: dict[str, Any],
    **forward_kw: Any,
) -> tuple[torch.Tensor, list[str]]:
    """Capture embedding output then each decoder block's output (post-sublayer stack).

    Returns a tensor ``stack`` of shape ``(num_stages, batch, seq, hidden)`` where stage ``0``
    is the input embedding output and stage ``i+1`` is the output of decoder block ``i``.
    Names are ``embedding``, ``layer_0``, …, ``layer_{n-1}``.

    ``inputs`` is passed to ``model(**inputs, **forward_kw)`` (e.g. ``input_ids``,
    ``attention_mask``). Typical call sets ``use_cache=False`` via ``forward_kw``.
    """
    blocks = decoder_blocks(model)
    out: list[torch.Tensor] = []

    def grab_embed(_mod: nn.Module, _inp: Any, x: torch.Tensor) -> None:
        out.append(x.detach())

    def grab_layer(_mod: nn.Module, _inp: Any, x: Any) -> None:
        h = x[0] if isinstance(x, tuple) else x
        out.append(h.detach())

    emb = model.get_input_embeddings()
    hooks: list[Any] = [emb.register_forward_hook(grab_embed)]
    hooks += [layer.register_forward_hook(grab_layer) for layer in blocks]
    try:
        merged = {**inputs, **forward_kw}
        with torch.inference_mode():
            model(**merged)
    finally:
        for h in hooks:
            h.remove()

    names = ["embedding"] + [f"layer_{i}" for i in range(len(blocks))]
    if len(out) != len(names):
        raise RuntimeError(
            f"capture_embed_and_post_blocks: expected {len(names)} tensors, got {len(out)}"
        )
    return torch.stack(out, dim=0), names


def capture_pre_block0_and_post_blocks(
    model: nn.Module,
    inputs: dict[str, Any],
    **forward_kw: Any,
) -> tuple[torch.Tensor, list[str]]:
    """Capture inputs to block 0 (pre-hook) then each block's output.

    Stage ``0`` is the tensor entering the first decoder block (not necessarily identical to
    the embedding hook in :func:`capture_embed_and_post_blocks`). Stages ``1..n`` match post-hooks
    on each decoder block. Returned tensor shape ``(num_stages, batch, seq, hidden)`` (or with
    an extra batch dimension squeezed when applicable—same behavior as common experiment code).

    Names: ``pre_block0``, ``layer_0``, …, ``layer_{n-1}``.
    """
    blocks = decoder_blocks(model)
    if not blocks:
        raise RuntimeError("capture_pre_block0_and_post_blocks: model has no decoder blocks")

    captured: list[torch.Tensor] = []

    def pre0(_mod: nn.Module, inputs_tuple: tuple[Any, ...]) -> None:
        captured.append(inputs_tuple[0].detach())

    def post(_mod: nn.Module, _inp: Any, out: torch.Tensor | tuple[Any, ...]) -> None:
        captured.append((out[0] if isinstance(out, tuple) else out).detach())

    hooks: list[Any] = [blocks[0].register_forward_pre_hook(pre0)]
    hooks += [layer.register_forward_hook(post) for layer in blocks]
    try:
        merged = {**inputs, **forward_kw}
        with torch.inference_mode():
            model(**merged)
    finally:
        for h in hooks:
            h.remove()

    stacked = torch.stack(captured, dim=0)
    if stacked.dim() == 4 and stacked.shape[1] == 1:
        stacked = stacked.squeeze(1)
    names = ["pre_block0"] + [f"layer_{i}" for i in range(len(blocks))]
    if stacked.shape[0] != len(names):
        raise RuntimeError(
            f"capture_pre_block0_and_post_blocks: expected {len(names)} stages, got {stacked.shape[0]}"
        )
    return stacked, names
