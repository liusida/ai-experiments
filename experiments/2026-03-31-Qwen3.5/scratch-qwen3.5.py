# %% Imports & config
# Qwen3.5-4B image QA + manual forward (layers). Model: https://huggingface.co/Qwen/Qwen3.5-4B
# Image via Stonesoup static mount ``/data/image/dog-tiny.png`` (e.g. http://127.0.0.1:8765/...).
# Stonesoup: watch this file; run cells in order. Terminal: ``uv run python experiments/2026-03-31-Qwen3.5/scratch-qwen3.5.py``
from __future__ import annotations

import html
from typing import Any

import torch
import torch.nn.functional as F
import stonesoup
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5Attention,
    Qwen3_5GatedDeltaNet,
    apply_rotary_pos_emb,
    eager_attention_forward,
)

MODEL_ID = "Qwen/Qwen3.5-4B"
IMAGE_URL = "http://127.0.0.1:8765/data/image/dog-tiny.png"
USER_PROMPT = "Is this a cat?"
MAX_NEW_TOKENS = 6

print(stonesoup.STONESOUP_RENDER_HTML, end="")
print(
    f"<p><strong>{html.escape(MODEL_ID)}</strong><br/>"
    f"{html.escape(USER_PROMPT)}<br/>"
    f"<code>max_new_tokens={MAX_NEW_TOKENS}</code></p>"
    f'<p class="stonesoup-show"><img src="{html.escape(IMAGE_URL, quote=True)}" '
    f'alt="Input image" loading="lazy" style="max-width:min(480px,100%);height:auto" /></p>',
    flush=True,
)

# %% Load model

model, processor = stonesoup.load_model(MODEL_ID)
model.eval()
DEVICE = next(model.parameters()).device
print("Loaded:", MODEL_ID, DEVICE, flush=True)

# %% Build prompt tensors

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": IMAGE_URL},
            {"type": "text", "text": USER_PROMPT},
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

# %% Baseline generate

with torch.inference_mode():
    baseline_tokens = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)

baseline_token_pieces = [
    processor.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    for token_id in baseline_tokens[0].tolist()
]
print("|".join(baseline_token_pieces), flush=True)

# %% Forward helpers (educational)

torch.set_printoptions(sci_mode=False, precision=3, linewidth=200)

INFO = True  # prints from forward_qwen35_like_hf / language_model_forward_layer_by_layer


def info(*args: Any) -> None:
    if INFO:
        print("info:", *args, flush=True)


def linear_attn_forward_no_cache(
    linear_attn: Qwen3_5GatedDeltaNet, hidden_states: torch.Tensor
) -> torch.Tensor:
    """``Qwen3_5GatedDeltaNet.forward`` with ``cache_params=None`` and ``attention_mask=None`` (full sequence).

    Mirrors the non-cached branch in ``modeling_qwen3_5.py``: projections → causal depthwise conv on QKV →
    split heads → gated delta rule → gated RMSNorm → output projection.
    """
    info("linear_attn_forward_no_cache")
    x = hidden_states
    batch_size, seq_len, _ = x.shape

    mixed_qkv = linear_attn.in_proj_qkv(x).transpose(1, 2)
    z = linear_attn.in_proj_z(x).reshape(batch_size, seq_len, -1, linear_attn.head_v_dim)
    b = linear_attn.in_proj_b(x)
    a = linear_attn.in_proj_a(x)

    if linear_attn.causal_conv1d_fn is not None:
        mixed_qkv = linear_attn.causal_conv1d_fn(
            x=mixed_qkv,
            weight=linear_attn.conv1d.weight.squeeze(1),
            bias=linear_attn.conv1d.bias,
            activation=linear_attn.activation,
            seq_idx=None,
        )
    else:
        mixed_qkv = F.silu(linear_attn.conv1d(mixed_qkv)[:, :, :seq_len])

    mixed_qkv = mixed_qkv.transpose(1, 2)
    query, key, value = torch.split(
        mixed_qkv,
        [linear_attn.key_dim, linear_attn.key_dim, linear_attn.value_dim],
        dim=-1,
    )
    query = query.reshape(batch_size, seq_len, -1, linear_attn.head_k_dim)
    key = key.reshape(batch_size, seq_len, -1, linear_attn.head_k_dim)
    value = value.reshape(batch_size, seq_len, -1, linear_attn.head_v_dim)

    beta = b.sigmoid()
    g = -linear_attn.A_log.float().exp() * F.softplus(a.float() + linear_attn.dt_bias)
    if linear_attn.num_v_heads // linear_attn.num_k_heads > 1:
        r = linear_attn.num_v_heads // linear_attn.num_k_heads
        query = query.repeat_interleave(r, dim=2)
        key = key.repeat_interleave(r, dim=2)

    core_attn_out, _ = linear_attn.chunk_gated_delta_rule(
        query,
        key,
        value,
        g=g,
        beta=beta,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
    )

    core_attn_out = core_attn_out.reshape(-1, linear_attn.head_v_dim)
    z = z.reshape(-1, linear_attn.head_v_dim)
    core_attn_out = linear_attn.norm(core_attn_out, z)
    core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
    return linear_attn.out_proj(core_attn_out)


def self_attn_forward_no_cache(
    self_attn: Qwen3_5Attention,
    hidden_states: torch.Tensor,
    *,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """``Qwen3_5Attention.forward`` with ``past_key_values=None``, ``attention_mask=None`` (full sequence).

    Q/K/V + head RMSNorm → MRoPE → ``ALL_ATTENTION_FUNCTIONS`` (SDPA / flash / eager per config) → gated output → ``o_proj``.
    """
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self_attn.head_dim)

    query_states, gate = torch.chunk(
        self_attn.q_proj(hidden_states).view(*input_shape, -1, self_attn.head_dim * 2), 2, dim=-1
    )
    gate = gate.reshape(*input_shape, -1)

    query_states = self_attn.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
    key_states = self_attn.k_norm(self_attn.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
        self_attn.config._attn_implementation, eager_attention_forward
    )
    attn_output, _ = attention_interface(
        self_attn,
        query_states,
        key_states,
        value_states,
        None,
        dropout=0.0 if not self_attn.training else self_attn.attention_dropout,
        scaling=self_attn.scaling,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = attn_output * torch.sigmoid(gate)
    return self_attn.o_proj(attn_output)


def language_model_forward_layer_by_layer(
    lm,
    inputs_embeds: torch.Tensor,
    position_ids: torch.Tensor | None,
) -> torch.Tensor:
    """Same computation as ``Qwen3_5TextModel.forward`` (no KV cache — full sequence each call).

    Expects ``position_ids`` from ``compute_3d_position_ids`` with shape ``(3, B, T)`` (MRoPE).

    Attention mask tensors are omitted (``None``): with the usual **SDPA** backend, causal full attention uses
    ``is_causal`` inside ``scaled_dot_product_attention`` when the sequence length is ``> 1``. (An **eager** attention
    path would need an explicit mask — this script assumes SDPA, like the default ``from_pretrained`` load.)
    """
    if position_ids is None:
        raise ValueError(
            "pass ``position_ids`` from ``Qwen3_5Model.compute_3d_position_ids`` (shape (3, batch, seq))."
        )
    pos = position_ids
    info("position_ids (MRoPE 3×B×T):", tuple(pos.shape))

    hidden_states = inputs_embeds
    position_embeddings = lm.rotary_emb(hidden_states, pos)

    for layer_idx, decoder_layer in enumerate(lm.layers[: lm.config.num_hidden_layers]):
        # ``Qwen3_5DecoderLayer.forward`` inlined: pre-norm → mixer → +residual → post-norm → MLP → +residual.
        residual = hidden_states
        x = decoder_layer.input_layernorm(hidden_states)
        if decoder_layer.layer_type == "linear_attention":
            x = linear_attn_forward_no_cache(decoder_layer.linear_attn, x)
        else:
            x = self_attn_forward_no_cache(
                decoder_layer.self_attn,
                x,
                position_embeddings=position_embeddings,
            )
        hidden_states = residual + x
        residual = hidden_states
        x = decoder_layer.post_attention_layernorm(hidden_states)
        hidden_states = residual + decoder_layer.mlp(x)
        lt = getattr(decoder_layer, "layer_type", "?")
        info(f"  decoder layer {layer_idx:2d} ({lt:16s})  hidden {tuple(hidden_states.shape)}")

    out = lm.norm(hidden_states)
    info("  trunk output (after final RMSNorm)  hidden", tuple(out.shape))
    return out


def forward_qwen35_like_hf(
    m: Any,
    *,
    input_ids: torch.Tensor,
    mm_token_type_ids: torch.Tensor,
    pixel_values: torch.Tensor,
    image_grid_thw: torch.Tensor,
) -> torch.Tensor:
    """Explicit forward over the **full** current sequence (no KV cache — re-runs the whole trunk each step)."""
    backbone = m.model  # ``Qwen3_5Model``: ``visual`` + ``language_model``

    # 1) Token embedding table: integer ids → vectors (B, T, hidden).
    inputs_embeds = backbone.get_input_embeddings()(input_ids)
    info("inputs_embeds:", inputs_embeds[0, :10, :5])

    # 2) Vision: pixels → patch features, same width as LM; paste into <|image_pad|> positions.
    image_out = backbone.get_image_features(pixel_values, image_grid_thw, return_dict=True)
    info("image_out.last_hidden_state (n_patches, ViT_hidden_dim):", image_out.last_hidden_state.shape)
    info(
        "image_out.pooler_output [(n_image_tokens_per_img, embedding_dim) x n_images]:",
        [t.shape for t in image_out.pooler_output],
    )
    image_embeds = image_out.pooler_output
    image_embeds = torch.cat(image_embeds, dim=0).to(
        device=inputs_embeds.device, dtype=inputs_embeds.dtype
    )
    image_mask, _video_unused = backbone.get_placeholder_mask(
        input_ids,
        inputs_embeds=inputs_embeds,
        image_features=image_embeds,
    )
    inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
    info("inputs_embeds after scatter (filling the image tokens):", inputs_embeds[0, :10, :5])

    # 3) Position ids for multimodal RoPE (text vs image spans); uses ``mm_token_type_ids``.
    position_ids = backbone.compute_3d_position_ids(
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        image_grid_thw=image_grid_thw,
        mm_token_type_ids=mm_token_type_ids,
    )
    info("thw = time x height x width")
    info("position_ids time:", position_ids[0])
    info("position_ids row (height):", position_ids[1])
    info("position_ids col (width):", position_ids[2])

    # 4) Transformer trunk (no ``past_key_values``).
    hidden = language_model_forward_layer_by_layer(
        backbone.language_model,
        inputs_embeds,
        position_ids,
    )

    # 5) Project last hidden dim to vocab logits (weights often tied to ``embed_tokens``).
    return m.lm_head(hidden)


# %% Manual generation loop (no KV cache)

with torch.inference_mode():
    tokens = inputs.input_ids.clone()
    mm_token_type_ids = inputs.mm_token_type_ids.clone()
    eos_token_id = processor.tokenizer.eos_token_id

    for _ in range(MAX_NEW_TOKENS):
        logits = forward_qwen35_like_hf(
            model,
            input_ids=tokens,
            mm_token_type_ids=mm_token_type_ids,
            pixel_values=inputs.pixel_values,
            image_grid_thw=inputs.image_grid_thw,
        )
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)

        tokens = torch.cat([tokens, next_token], dim=1)
        # Newly generated positions are text, not image placeholders.
        mm_token_type_ids = torch.cat(
            [mm_token_type_ids, torch.zeros_like(next_token, dtype=mm_token_type_ids.dtype)],
            dim=1,
        )

        if next_token.item() == eos_token_id:
            break

token_pieces = [
    processor.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    for token_id in tokens[0].tolist()
]
print("|".join(token_pieces), flush=True)
