# %% Imports & config
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
    f"<p><strong>{html.escape(MODEL_ID)}</strong> — {html.escape(USER_PROMPT)}</p>"
    f'<p><img src="{html.escape(IMAGE_URL, quote=True)}" style="max-width:480px;height:auto" /></p>',
    flush=True,
)

# %% Load model

model, processor = stonesoup.load_model(MODEL_ID)
model.eval()
DEVICE = next(model.parameters()).device
print("Loaded:", MODEL_ID, DEVICE, flush=True)

# %% Build prompt tensors

messages = [{"role": "user", "content": [
    {"type": "image", "image": IMAGE_URL},
    {"type": "text", "text": USER_PROMPT},
]}]

inputs = processor.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True,
    enable_thinking=False, return_dict=True, return_tensors="pt",
).to(DEVICE)

# %% Baseline generate

with torch.inference_mode():
    baseline_tokens = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)

print("|".join(
    processor.decode([t], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    for t in baseline_tokens[0].tolist()
), flush=True)

# %% Forward helpers (educational)

torch.set_printoptions(sci_mode=False, precision=3, linewidth=200)
INFO = True


def info(*args: Any) -> None:
    if INFO:
        print("info:", *args, flush=True)


def linear_attn_forward_no_cache(linear_attn: Qwen3_5GatedDeltaNet, hidden_states: torch.Tensor) -> torch.Tensor:
    """Qwen3_5GatedDeltaNet.forward with cache_params=None and attention_mask=None."""
    info("linear_attn_forward_no_cache")
    x = hidden_states
    batch_size, seq_len, _ = x.shape

    mixed_qkv = linear_attn.in_proj_qkv(x).transpose(1, 2)
    z = linear_attn.in_proj_z(x).reshape(batch_size, seq_len, -1, linear_attn.head_v_dim)
    b = linear_attn.in_proj_b(x)
    a = linear_attn.in_proj_a(x)

    if linear_attn.causal_conv1d_fn is not None:
        mixed_qkv = linear_attn.causal_conv1d_fn(
            x=mixed_qkv, weight=linear_attn.conv1d.weight.squeeze(1),
            bias=linear_attn.conv1d.bias, activation=linear_attn.activation, seq_idx=None,
        )
    else:
        mixed_qkv = F.silu(linear_attn.conv1d(mixed_qkv)[:, :, :seq_len])

    mixed_qkv = mixed_qkv.transpose(1, 2)
    query, key, value = torch.split(
        mixed_qkv, [linear_attn.key_dim, linear_attn.key_dim, linear_attn.value_dim], dim=-1,
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
        query, key, value, g=g, beta=beta,
        initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=True,
    )

    core_attn_out = linear_attn.norm(core_attn_out.reshape(-1, linear_attn.head_v_dim), z.reshape(-1, linear_attn.head_v_dim))
    return linear_attn.out_proj(core_attn_out.reshape(batch_size, seq_len, -1))


def self_attn_forward_no_cache(
    self_attn: Qwen3_5Attention,
    hidden_states: torch.Tensor,
    *,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Qwen3_5Attention.forward with past_key_values=None and attention_mask=None."""
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self_attn.head_dim)

    query_states, gate = torch.chunk(
        self_attn.q_proj(hidden_states).view(*input_shape, -1, self_attn.head_dim * 2), 2, dim=-1,
    )
    gate = gate.reshape(*input_shape, -1)
    query_states = self_attn.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
    key_states = self_attn.k_norm(self_attn.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, *position_embeddings)

    attn_output, _ = ALL_ATTENTION_FUNCTIONS.get_interface(self_attn.config._attn_implementation, eager_attention_forward)(
        self_attn, query_states, key_states, value_states, None,
        dropout=0.0 if not self_attn.training else self_attn.attention_dropout,
        scaling=self_attn.scaling,
    )
    return self_attn.o_proj(attn_output.reshape(*input_shape, -1).contiguous() * torch.sigmoid(gate))


def language_model_forward_layer_by_layer(lm, inputs_embeds: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
    """Qwen3_5TextModel.forward without KV cache (full sequence each call)."""
    info("position_ids (MRoPE 3×B×T):", tuple(position_ids.shape))
    hidden_states = inputs_embeds
    position_embeddings = lm.rotary_emb(hidden_states, position_ids)

    for i, layer in enumerate(lm.layers[: lm.config.num_hidden_layers]):
        residual = hidden_states
        x = layer.input_layernorm(hidden_states)
        x = (linear_attn_forward_no_cache(layer.linear_attn, x)
             if layer.layer_type == "linear_attention"
             else self_attn_forward_no_cache(layer.self_attn, x, position_embeddings=position_embeddings))
        hidden_states = residual + x
        residual = hidden_states
        hidden_states = residual + layer.mlp(layer.post_attention_layernorm(hidden_states))
        info(f"  layer {i:2d} ({getattr(layer, 'layer_type', '?'):16s})  hidden {tuple(hidden_states.shape)}")

    out = lm.norm(hidden_states)
    info("  trunk output hidden", tuple(out.shape))
    return out


def forward_qwen35_like_hf(m: Any, *, input_ids, mm_token_type_ids, pixel_values, image_grid_thw) -> torch.Tensor:
    """Full explicit forward over the current sequence (no KV cache)."""
    backbone = m.model

    inputs_embeds = backbone.get_input_embeddings()(input_ids)
    info("inputs_embeds:", inputs_embeds[0, :10, :5])

    image_out = backbone.get_image_features(pixel_values, image_grid_thw, return_dict=True)
    info("image last_hidden_state:", image_out.last_hidden_state.shape)
    image_embeds = torch.cat(image_out.pooler_output, dim=0).to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
    image_mask, _ = backbone.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds)
    inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
    info("inputs_embeds after scatter:", inputs_embeds[0, :10, :5])

    position_ids = backbone.compute_3d_position_ids(
        input_ids=input_ids, inputs_embeds=inputs_embeds,
        image_grid_thw=image_grid_thw, mm_token_type_ids=mm_token_type_ids,
    )
    info("position_ids (t/row/col):", position_ids[0], position_ids[1], position_ids[2])

    hidden = language_model_forward_layer_by_layer(backbone.language_model, inputs_embeds, position_ids)
    return m.lm_head(hidden)


# %% Manual generation loop (no KV cache)

with torch.inference_mode():
    tokens = inputs.input_ids.clone()
    mm_token_type_ids = inputs.mm_token_type_ids.clone()
    eos_token_id = processor.tokenizer.eos_token_id

    for _ in range(MAX_NEW_TOKENS):
        logits = forward_qwen35_like_hf(
            model, input_ids=tokens, mm_token_type_ids=mm_token_type_ids,
            pixel_values=inputs.pixel_values, image_grid_thw=inputs.image_grid_thw,
        )
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        tokens = torch.cat([tokens, next_token], dim=1)
        mm_token_type_ids = torch.cat(
            [mm_token_type_ids, torch.zeros_like(next_token, dtype=mm_token_type_ids.dtype)], dim=1,
        )
        if next_token.item() == eos_token_id:
            break

print("|".join(
    processor.decode([t], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    for t in tokens[0].tolist()
), flush=True)
