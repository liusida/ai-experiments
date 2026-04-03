# %% Imports & detailed caption config
from __future__ import annotations

import html
from types import SimpleNamespace

import torch
import stonesoup
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_masks_for_generate as _tf_create_masks_for_generate
from transformers.models.gemma4.modeling_gemma4 import create_causal_mask_mapping

# Same checkpoint as https://huggingface.co/google/gemma-4-E2B-it — needs a recent transformers (Gemma 4 / ``gemma4`` config).
MODEL_ID = "google/gemma-4-E2B-it"
# Cookbook raw URL (returns real image/jpeg; many hotlinks return HTML 404/403).
IMAGE_URL = (
    "https://raw.githubusercontent.com/google-gemma/cookbook/refs/heads/main/"
    "Demos/sample-data/kitchen_painting.jpg"
)
USER_PROMPT = (
    "Describe this image concisely."
)
MAX_NEW_TOKENS = 32

print(stonesoup.STONESOUP_RENDER_HTML, end="")
print(
    f"<p><strong>{html.escape(MODEL_ID)}</strong> — detailed description</p>"
    f"<p>{html.escape(USER_PROMPT)}</p>"
    f'<p><img src="{html.escape(IMAGE_URL, quote=True)}" style="max-width:520px;height:auto" alt="" /></p>',
    flush=True,
)

# %% Load model (Stonesoup toolbar or cell — one load per repo id)

model, processor = stonesoup.load_model(MODEL_ID)
model.eval()
DEVICE = next(model.parameters()).device
print("Loaded:", MODEL_ID, DEVICE, flush=True)

# %% Build multimodal inputs (image before text)

# ``apply_chat_template(..., tokenize=True)`` requires list-shaped ``content`` on every turn (not a bare string).
messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are a helpful assistant."}],
    },
    {
        "role": "user",
        "content": [
            {"type": "image", "url": IMAGE_URL},
            {"type": "text", "text": USER_PROMPT},
        ],
    },
]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    add_generation_prompt=True,
).to(DEVICE)
input_len = inputs["input_ids"].shape[-1]

# %% Tiny ``generate`` loop (education only)

# Hugging Face ``GenerationMixin.generate`` chains many helpers; the core is **autoregressive decoding**:
# forward → logits at the **last** position → pick a token → append to ``input_ids``. With ``use_cache=True``,
# ``past_key_values`` lets later steps pass **only the new token** through the LM (vision tensors are omitted
# after the first step; they are folded into the cache — see ``prepare_inputs_for_generation`` in
# ``modeling_gemma4.py``). This REPL-friendly version skips beam search, compilers, watermarking, etc.
# **Greedy** decoding (``argmax``) — no sampling RNG; same logits → same token (GPU kernels can still be
# slightly non-deterministic unless you enable deterministic algorithms).


def greedy_next_token_id(logits_1v: torch.Tensor) -> torch.Tensor:
    """Pick the single highest logit; shapes ``(1, vocab)`` → ``(1, 1)`` int64 token ids."""
    return logits_1v[0].argmax(dim=-1).view(1, 1).to(torch.long)


def gemma4_text_decoder_layer_expanded(
    layer,
    hidden_states: torch.Tensor,
    per_layer_input: torch.Tensor | None,
    position_embeddings: torch.Tensor,
    attention_mask,
    position_ids: torch.LongTensor,
    past_key_values,
) -> torch.Tensor:
    """
    Step-through of ``Gemma4TextDecoderLayer.forward`` in ``modeling_gemma4.py``:

    pre-norm → **self-attn** (RoPE + KV cache) → post-attn norm → residual **+**;
    then pre-norm → **MLP** (and optional **MoE** branch when ``enable_moe_block``) → post-ff norm → residual **+**;
    then optional **per-layer embedding** gate / act / ``× pl`` / proj / norm / residual;
    finally ``layer_scalar`` scale.
    """
    residual = hidden_states
    x = layer.input_layernorm(hidden_states)
    x, _ = layer.self_attn(
        hidden_states=x,
        position_embeddings=position_embeddings,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
    )
    x = layer.post_attention_layernorm(x)
    x = residual + x

    residual = x
    x = layer.pre_feedforward_layernorm(x)
    x = layer.mlp(x)

    if layer.enable_moe_block:
        out_mlp = layer.post_feedforward_layernorm_1(x)
        flat = residual.reshape(-1, residual.shape[-1])
        _, top_k_weights, top_k_index = layer.router(flat)
        out_moe = layer.pre_feedforward_layernorm_2(flat)
        out_moe = layer.experts(out_moe, top_k_index, top_k_weights)
        out_moe = out_moe.reshape(residual.shape)
        out_moe = layer.post_feedforward_layernorm_2(out_moe)
        x = out_mlp + out_moe

    x = layer.post_feedforward_layernorm(x)
    x = residual + x

    if layer.hidden_size_per_layer_input:
        if per_layer_input is None:
            raise ValueError("per_layer_input required when hidden_size_per_layer_input is set")
        residual = x
        x = layer.per_layer_input_gate(x)
        x = layer.act_fn(x)
        x = x * per_layer_input
        x = layer.per_layer_projection(x)
        x = layer.post_per_layer_input_norm(x)
        x = residual + x

    return x * layer.layer_scalar


def gemma4_text_decoder_expanded(
    lm,
    *,
    inputs_embeds: torch.Tensor,
    attention_mask: dict,
    position_ids: torch.LongTensor,
    past_key_values,
    per_layer_inputs: torch.Tensor | None,
    use_cache: bool | None = True,
) -> SimpleNamespace:
    """
    Outline of ``Gemma4TextModel.forward`` when ``inputs_embeds`` and a pre-built **mask dict**
    (``full_attention`` / ``sliding_attention`` from Gemma 4 multimodal prep) are already supplied.

    Flow: optional **PLE fusion** → ``DynamicCache`` → **RoPE** cos/sin per layer-type → stack of
    ``Gemma4TextDecoderLayer`` (norm → attn → residual → M/MoE → residual, + per-layer branch) → final RMSNorm.
    """
    use_cache = use_cache if use_cache is not None else lm.config.use_cache

    if lm.hidden_size_per_layer_input:
        per_layer_inputs = lm.project_per_layer_inputs(inputs_embeds, per_layer_inputs)
    else:
        per_layer_inputs = None

    if use_cache and past_key_values is None:
        past_key_values = DynamicCache(config=lm.config)

    if not isinstance(attention_mask, dict):
        raise TypeError("This expanded path expects Gemma's dict mask from multimodal prefill.")

    causal_mask_mapping = attention_mask
    hidden_states = inputs_embeds
    position_embeddings: dict = {}
    for layer_type in lm.unique_layer_types:
        position_embeddings[layer_type] = lm.rotary_emb(hidden_states, position_ids, layer_type)

    for i, decoder_layer in enumerate(lm.layers[: lm.config.num_hidden_layers]):
        pl = per_layer_inputs[:, :, i, :] if per_layer_inputs is not None else None
        layer_attn = causal_mask_mapping[lm.config.layer_types[i]]
        rope = position_embeddings[lm.config.layer_types[i]]
        hidden_states = gemma4_text_decoder_layer_expanded(
            decoder_layer,
            hidden_states,
            pl,
            rope,
            layer_attn,
            position_ids,
            past_key_values,
        )

    hidden_states = lm.norm(hidden_states)
    return SimpleNamespace(last_hidden_state=hidden_states, past_key_values=past_key_values)


def gemma4_backbone_prefill_expanded(
    core,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    pixel_values: torch.Tensor,
    image_position_ids: torch.Tensor,
    mm_token_type_ids: torch.Tensor | None = None,
) -> SimpleNamespace:
    """
    Strip ``Gemma4Model.forward`` down to the image+text prefill path (no video/audio).

    Order matches ``modeling_gemma4.py``: placeholder masks → token embeddings with PAD at multimodal
    slots → optional **per-layer embeddings** (PLE) → ``vision_tower`` + ``embed_vision`` patches written
    into image positions → **sliding/full attention** mask bundle → ``language_model`` decoder.
    """
    device = input_ids.device
    cfg_txt = core.config.get_text_config()

    image_mask, video_mask, audio_mask = core.get_placeholder_mask(input_ids, None)
    multimodal_mask = image_mask | video_mask | audio_mask

    llm_input_ids = input_ids.clone()
    llm_input_ids[multimodal_mask] = cfg_txt.pad_token_id
    inputs_embeds = core.get_input_embeddings()(llm_input_ids)

    if cfg_txt.hidden_size_per_layer_input:
        pad_vec = core.language_model.embed_tokens.weight[cfg_txt.pad_token_id, :]
        llm_inputs_embeds = torch.where(
            multimodal_mask[..., None],
            pad_vec.view(1, 1, -1),
            inputs_embeds,
        )
        per_layer_inputs = core.language_model.get_per_layer_inputs(llm_input_ids, llm_inputs_embeds)
    else:
        per_layer_inputs = None

    vis = core.get_image_features(
        pixel_values,
        image_position_ids=image_position_ids,
        return_dict=True,
    )
    image_feats = vis.pooler_output.to(device=device, dtype=inputs_embeds.dtype)
    image_mask_exp = image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(device)
    inputs_embeds = inputs_embeds.masked_scatter(image_mask_exp, image_feats)

    past_key_values = None
    past_seen = 0 if past_key_values is None else past_key_values.get_seq_length()
    position_ids = torch.arange(inputs_embeds.shape[1], device=device, dtype=torch.long) + past_seen
    position_ids = position_ids.unsqueeze(0)

    if getattr(cfg_txt, "use_bidirectional_attention", None) == "vision":
        attn_masks = create_causal_mask_mapping(
            core.config,
            inputs_embeds,
            attention_mask,
            past_key_values,
            position_ids,
            mm_token_type_ids,
            pixel_values=pixel_values,
            is_training=core.training,
        )
    else:
        attn_masks = _tf_create_masks_for_generate(
            core.config,
            inputs_embeds,
            attention_mask,
            past_key_values,
            position_ids,
        )

    lm_out = gemma4_text_decoder_expanded(
        core.language_model,
        inputs_embeds=inputs_embeds,
        attention_mask=attn_masks,
        position_ids=position_ids,
        past_key_values=past_key_values,
        per_layer_inputs=per_layer_inputs,
        use_cache=True,
    )
    return SimpleNamespace(last_hidden_state=lm_out.last_hidden_state, past_key_values=lm_out.past_key_values)


def manual_gemma4_decode(
    m,
    batch: dict,
    *,
    max_new_tokens: int,
) -> torch.LongTensor:
    """Prompt + generated tokens, shape ``(1, seq)`` — stopping at EOS when configured."""
    device = batch["input_ids"].device
    eos = m.generation_config.eos_token_id
    if eos is None:
        eos = m.config.text_config.eos_token_id
    eos_set: set[int] = set()
    if isinstance(eos, int):
        eos_set.add(eos)
    elif eos is not None:
        eos_set.update(int(x) for x in eos)

    ids = batch["input_ids"]
    attn = batch["attention_mask"]
    past = None
    static = {k: batch[k] for k in ("pixel_values", "image_position_ids", "mm_token_type_ids") if k in batch}

    new_chunks: list[torch.Tensor] = []
    next_id: torch.Tensor | None = None
    for _ in range(max_new_tokens):
        if past is None:
            # ``Gemma4ForConditionalGeneration.forward`` is two conceptual stages (see ``modeling_gemma4.py``):
            # (1) ``self.model`` — embed tokens, **replace image placeholder positions** with vision-tower
            #     vectors, run the decoder stack, return ``last_hidden_state`` + ``past_key_values``;
            # (2) ``self.lm_head`` — linear map from hidden dim → ``vocab_size`` (Gemma **ties** this weight
            #     to the input embedding table). Then optional **logit softcapping** (tanh squash) like training.
            backbone = gemma4_backbone_prefill_expanded(
                m.model,
                ids,
                attn,
                pixel_values=static["pixel_values"],
                image_position_ids=static["image_position_ids"],
                mm_token_type_ids=static.get("mm_token_type_ids"),
            )
            last_hidden = backbone.last_hidden_state[:, -1:, :]
            logits = m.lm_head(last_hidden)
            cap = m.config.get_text_config().final_logit_softcapping
            if cap is not None:
                logits = logits / cap
                logits = torch.tanh(logits)
                logits = logits * cap
            out = SimpleNamespace(logits=logits, past_key_values=backbone.past_key_values)
        else:
            assert next_id is not None
            out = m(
                input_ids=next_id,
                attention_mask=attn,
                past_key_values=past,
                use_cache=True,
                logits_to_keep=1,
            )
        past = out.past_key_values
        next_id = greedy_next_token_id(out.logits[:, -1, :])
        new_chunks.append(next_id)
        attn = torch.cat(
            [attn, torch.ones((attn.shape[0], 1), device=device, dtype=attn.dtype)],
            dim=-1,
        )
        if eos_set and next_id.item() in eos_set:
            break

    return torch.cat([batch["input_ids"], *new_chunks], dim=-1)


# %% Run manual decode & parse

with torch.inference_mode():
    outputs = manual_gemma4_decode(model, inputs, max_new_tokens=MAX_NEW_TOKENS)

response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
parsed = processor.parse_response(response)
if isinstance(parsed, dict) and isinstance(parsed.get("content"), str):
    answer = parsed["content"]
else:
    answer = str(parsed)
print(stonesoup.STONESOUP_RENDER_MD, end="")
print(f"## Assistant\n\n{answer}", flush=True)
