# %% Imports & model id
from __future__ import annotations

import inspect
from pathlib import Path

import torch
import torch.nn.functional as F
import stonesoup
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5ForCausalLM,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5GatedDeltaNet,
    torch_recurrent_gated_delta_rule,
)
from transformers.masking_utils import create_causal_mask

# Text-only Qwen3.5 checkpoint (hybrid stack: full self-attention + Gated-DeltaNet layers).
MODEL_ID = "Qwen/Qwen3.5-0.8B"
PROMPT = "Describe a cat."

# %% Where the implementation lives (transformers source)
mod_file = Path(inspect.getfile(Qwen3_5GatedDeltaNet)).resolve()
print("Qwen3_5GatedDeltaNet defined in:", mod_file)
print("Forward starts at line:", inspect.getsourcelines(Qwen3_5GatedDeltaNet.forward)[1])

# %% Load model (Stonesoup toolbar Load uses the same path)
model, tokenizer_or_processor = stonesoup.load_model(MODEL_ID)
# This Hub id may register a VL ``Processor``; text-only paths must use ``.tokenizer``, not ``__call__`` on the processor.
tokenizer = getattr(tokenizer_or_processor, "tokenizer", tokenizer_or_processor)
model.eval()
device = next(model.parameters()).device
# Hub may expose ``Qwen3_5Config`` (text under ``text_config``); causal LM weights still use the text stack.
_root_cfg = model.config
_cfg = getattr(_root_cfg, "text_config", None)
text_cfg = _cfg if _cfg is not None else _root_cfg
print("device:", device)
print("config:", type(_root_cfg).__name__, "→ using", type(text_cfg).__name__, "for layer_types / depth")
print("num_hidden_layers:", text_cfg.num_hidden_layers)
print("layer_types:", text_cfg.layer_types)

linear_layers = [i for i, t in enumerate(text_cfg.layer_types) if t == "linear_attention"]
full_layers = [i for i, t in enumerate(text_cfg.layer_types) if t == "full_attention"]
print("linear_attention (Gated-DeltaNet) layer indices:", linear_layers)
print("full_attention layer indices:", full_layers)

# ``Qwen3_5ForCausalLM``: ``model.model`` is ``Qwen3_5TextModel`` (has ``.layers``).
# Multimodal ``Qwen3_5ForConditionalGeneration``: ``model.model`` is ``Qwen3_5Model`` → layers on ``.language_model``.
_inner = model.model
text_module = getattr(_inner, "language_model", _inner)
print("text trunk:", type(_inner).__name__, "→", type(text_module).__name__, "(layers)")

# %% Tokenize
inputs = tokenizer(PROMPT, return_tensors="pt").to(device)
input_ids = inputs["input_ids"]
attention_mask = inputs.get("attention_mask")

# %% Hook: capture inputs to Gated-DeltaNet (for fair comparison below)
layer_idx = linear_layers[0]
layer = text_module.layers[layer_idx]
assert hasattr(layer, "linear_attn") and isinstance(layer.linear_attn, Qwen3_5GatedDeltaNet)

captured: dict[str, torch.Tensor | None] = {}


def _linear_attn_hook(_module, _args, kwargs, output):
    # Decoder calls ``linear_attn(hidden_states=..., cache_params=..., attention_mask=...)`` (all kwargs).
    captured["hidden_in"] = kwargs["hidden_states"].detach()
    captured["attn_mask"] = kwargs.get("attention_mask")
    if captured["attn_mask"] is not None:
        captured["attn_mask"] = captured["attn_mask"].detach()
    captured["hidden_out_hook"] = output.detach()


handle = layer.linear_attn.register_forward_hook(_linear_attn_hook, with_kwargs=True)
with torch.inference_mode():
    _ = model.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
handle.remove()

print(
    f"Hooked layer {layer_idx}: linear_attn in {tuple(captured['hidden_in'].shape)}, "
    f"out {tuple(captured['hidden_out_hook'].shape)}, attn_mask is "
    f"{type(captured['attn_mask']).__name__}",
)

# %% Compare: Hugging Face module vs explicit re-implementation


def _max_mean_abs_diff(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    d = (a.float() - b.float()).abs()
    return d.max().item(), d.mean().item()


def _l2norm_head_dim(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-head vector L2 normalize (matches HF ``l2norm`` in ``modeling_qwen3_5``)."""
    inv = torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)
    return x * inv


def recurrent_gated_delta_rule_expanded(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    g: torch.Tensor,
    beta: torch.Tensor,
    use_qk_l2norm: bool,
) -> torch.Tensor:
    """Causal gated delta rule, one time step at a time — same update as HF ``torch_recurrent_gated_delta_rule``.

    Does not use ``chunk_gated_delta_rule`` / FLA chunk kernel (slower for large ``T``, easier to read).

    ``query`` / ``key`` / ``value``: ``(batch, seq, heads, head_dim)``; ``g`` / ``beta``: ``(batch, seq, heads)``.

    On the first timestep of the first call in the process, prints tensor shapes (set
    ``recurrent_gated_delta_rule_expanded._shape_printed = False`` to print again).
    """
    dtype = query.dtype
    if use_qk_l2norm:
        query = _l2norm_head_dim(query)
        key = _l2norm_head_dim(key)

    q, k, v, beta_f, g_f = (t.transpose(1, 2).contiguous().to(torch.float32) for t in (query, key, value, beta, g))

    bsz, n_heads, seq_len, k_dim = k.shape
    v_dim = v.shape[-1]
    q = q * (q.shape[-1] ** -0.5)

    out = torch.zeros(bsz, n_heads, seq_len, v_dim, device=v.device, dtype=torch.float32)
    state = torch.zeros(bsz, n_heads, k_dim, v_dim, device=v.device, dtype=torch.float32)

    for t in range(seq_len):
        q_t = q[:, :, t]
        k_t = k[:, :, t]
        v_t = v[:, :, t]
        decay = g_f[:, :, t].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta_f[:, :, t].unsqueeze(-1)

        state = state * decay
        kv_mem = (state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        out[:, :, t] = (state * q_t.unsqueeze(-1)).sum(dim=-2)

        # Once per kernel: manual stack calls this on every linear layer (same shapes).
        if t == 0 and not getattr(recurrent_gated_delta_rule_expanded, "_shape_printed", False):
            recurrent_gated_delta_rule_expanded._shape_printed = True
            md = f"""## Gated Delta Net — tensor shapes at **t = 0** (after the first token’s update)

If you have never seen **Gated Delta Net** before, here is the idea in one paragraph. Ordinary **softmax attention** compares every new token to *all* past tokens at once (cost grows like sequence length squared). A **linear recurrent** layer instead keeps a **fixed-size memory matrix** `state` and updates it as each new token arrives—cost grows **linearly** with length. At each position the model still produces familiar **query**, **key**, and **value** vectors (like in attention), but instead of an all-pairs softmax it: **(1)** optionally **fades** the old memory (**decay**), **(2)** asks “what does memory *predict* for this position’s value?” (**kv_mem**), **(3)** compares the true **value** to that prediction and **gates** how much correction to apply (**β**), and **(4)** writes a **small rank-one** correction into memory using the current **key**. The **query** is only used at the end to **read out** a vector from memory for this timestep. The word **“delta”** refers to updating by a **difference** (residual) rather than overwriting blindly; **“gated”** refers to **β** and the decay both controlling how strongly past vs new information matters.

Below, tensors are shown in **heads-first** order `(batch, heads, time, …)` for `q`, `k`, `v` after transposing—so “time” is the sequence axis inside each head.

### What the size symbols mean

| Symbol | Plain-language meaning |
|--------|-------------------------|
| **B** | How many sequences in the batch (here: **{bsz}**). |
| **H** | How many **parallel heads** run the same rule with separate weights (here: **{n_heads}**). More heads = more capacity; Q/K may be **shared** across heads in some models (GQA). |
| **T** | **Sequence length**—how many tokens we step through (here: **{seq_len}**). |
| **k_dim** | Length of one **key** (and **query**) vector inside a head (here: **{k_dim}**). Think “how many numbers describe *direction in key space*.” |
| **v_dim** | Length of one **value** vector per head (here: **{v_dim}**). This is the size of the message we inject and read out. It can equal `k_dim` or not, depending on the model. |

### Inputs we already have for the *whole* sequence (same for every loop step)

These are built **before** the time loop. Each row of the tables is one tensor; the **Shape** column is your run; the **Explanation** column assumes no prior jargon.

| Variable | Shape | What it is (first-time reader) |
|----------|-------|--------------------------------|
| `q` | `{tuple(q.shape)}` | **Query** for **every** token and head: “if I blend memory with these weights, what vector do I want at the output?” Scaled by `1/√k_dim` so typical sizes stay stable (same spirit as attention scaling). |
| `k` | `{tuple(k.shape)}` | **Key** for every token: a direction in **k_dim**-dimensional space. It tells memory **where** to attach the new information when we update. |
| `v` | `{tuple(v.shape)}` | **Value** for every token: the **new information** we want memory to represent at this position (a **v_dim**-long vector per head). |
| `g_f` | `{tuple(g_f.shape)}` | Raw numbers that become **decay** after `exp`. **Larger** `g` (before exp) → **stronger** multiplier on old memory → **more forgetting** of the past before we add the new correction. |
| `beta_f` | `{tuple(beta_f.shape)}` | Raw numbers that become **β = sigmoid(·)** in **[0, 1]**. **β** answers: “How much of the mismatch between the new value and what memory predicted should we actually apply?” **Small β** → ignore the correction; **large β** → trust it. |
| `out` (buffer) | `{tuple(out.shape)}` | Empty-ish tensor we **fill in** one time index at a time. After the loop, `out[b,h,t,:]` is the **layer’s mixed output** for batch `b`, head `h`, token `t`—a vector of length **{v_dim}** per head, later reshaped and projected in the full block. |

### Objects used **inside** the loop at **t = 0** (first token)

These names appear **only** when processing the **first** timestep in this printout; the same pattern repeats for `t = 1, 2, …`.

| Variable | Shape | What it is (first-time reader) |
|----------|-------|--------------------------------|
| `q_t` | `{tuple(q_t.shape)}` | The **query for this token only**: `q[:,:,t]`. It **weights** the rows of `state` when we form the **output**—like asking memory, “in the directions you care about, what do you say?” |
| `k_t` | `{tuple(k_t.shape)}` | The **key for this token only**. Together with the **value residual**, it specifies **one** rank-one adjustment to the big memory matrix. |
| `v_t` | `{tuple(v_t.shape)}` | The **value for this token only**—the **target** content we want memory to explain. We **do not** paste it in blindly; we compare it to what memory **already predicted**. |
| `decay` | `{tuple(decay.shape)}` | **`exp(g)`** for this timestep, reshaped to `(B,H,1,1)` so PyTorch can multiply it against **every** entry of the `(k_dim × v_dim)` matrix `state` **per head**—same decay factor for all entries in that matrix slice. |
| `beta_t` | `{tuple(beta_t.shape)}` | **β** after sigmoid, shape `(B,H,1)` so it scales the whole **v_dim**-long residual vector at once per head. |
| `state` | `{tuple(state.shape)}` | The **memory matrix** for each head **after** this step’s decay, read, and write. Shape `(B,H,k_dim,v_dim)` means: for each head, a **k_dim-by-v_dim** table of numbers summarizing **everything seen so far** (causally). |
| `k_t.unsqueeze(-1)` | `{tuple(k_t.unsqueeze(-1).shape)}` | Turns the key from a **flat list** of length `k_dim` into a **column** (extra size-1 axis). That lets us multiply it against `state` **row-wise** along the key axis and then **sum**—implementing “dot each row of memory with this key.” |
| `kv_mem` | `{tuple(kv_mem.shape)}` | **“What memory thinks `v_t` should be”** before we see the correction: a **v_dim** vector per head. If memory were perfect, `kv_mem` would equal `v_t` and the update would be tiny. |
| `delta` | `{tuple(delta.shape)}` | The **gated correction**: `(true value − prediction) × β`. Only this part is **added** into memory (scaled by the key), which is why the method is **delta**-style. |
| `delta.unsqueeze(-2)` | `{tuple(delta.unsqueeze(-2).shape)}` | Turns `delta` into a **row** shape `(…,1,v_dim)` so when we multiply by `k_t` as a **column** `(…,k_dim,1)` we get a full **k_dim × v_dim** **outer product**—one simple matrix added to `state`. |
| `out[:,:,t]` | `{tuple(out[:, :, t].shape)}` | The **output vector** for this token: combine `state` with `q_t` along the key axis (same structural idea as “query attends over keys,” but here **keys index rows of memory**). Result length **{v_dim}** per head. |

### What one timestep does (same order as the code)

1. **`state *= decay`** — **Shrink** what we remembered from earlier tokens (per-head **forgetting**).  
2. **`kv_mem = …`** — **Read** memory: “predict the current value vector from `state` using the current key.”  
3. **`delta = (v_t - kv_mem) * beta_t`** — **How wrong** was that prediction, and **how much** of that error we **trust** (gate **β**).  
4. **`state += k_t ⊗ delta`** — **Write** a rank-one **patch** to memory: associate this key direction with this value correction.  
5. **`out[:,:,t] = …`** — **Emit** the head output for this token by **querying** the updated memory.

---
*Shown once per process (first linear-attention layer that runs). Set `recurrent_gated_delta_rule_expanded._shape_printed = False` to print again.*
"""
            print(stonesoup.STONESOUP_RENDER_MD, end="")
            print(md, flush=True)

    return out.transpose(1, 2).contiguous().to(dtype)


def linear_attn_forward_no_cache(
    linear_attn: Qwen3_5GatedDeltaNet,
    hidden_states: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Same steps as ``Qwen3_5GatedDeltaNet.forward`` with ``cache_params=None`` (full sequence).

    Gated-delta core uses ``recurrent_gated_delta_rule_expanded`` (scalar-time loop), not ``chunk_gated_delta_rule``.
    """
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_mask_to_padding_states

    x = apply_mask_to_padding_states(hidden_states, attention_mask)
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

    core_attn_out = recurrent_gated_delta_rule_expanded(
        query,
        key,
        value,
        g=g,
        beta=beta,
        use_qk_l2norm=True,
    )

    core_attn_out = linear_attn.norm(
        core_attn_out.reshape(-1, linear_attn.head_v_dim),
        z.reshape(-1, linear_attn.head_v_dim),
    )
    return linear_attn.out_proj(core_attn_out.reshape(batch_size, seq_len, -1))

# %% Run single-layer Gated-DeltaNet compare
_attn = captured["attn_mask"]
with torch.inference_mode():
    # 1) Official ``Qwen3_5GatedDeltaNet.forward`` (same tensors the stack used during the hook).
    y_hf = layer.linear_attn(captured["hidden_in"], cache_params=None, attention_mask=_attn)
    # 2) Step-by-step reimplementation (must match the module when mask and weights align).
    y_manual = linear_attn_forward_no_cache(
        layer.linear_attn,
        captured["hidden_in"],
        attention_mask=_attn,
    )

m_hook_hf, mean_hook_hf = _max_mean_abs_diff(captured["hidden_out_hook"], y_hf)
m_hf_man, mean_hf_man = _max_mean_abs_diff(y_hf, y_manual)
print("="*20)
print(
    "Compare Gated-DeltaNet outputs (layer", layer_idx, "):\n"
    f"  hook vs module.forward  max_abs={m_hook_hf:.3e} mean_abs={mean_hook_hf:.3e}\n"
    f"  module.forward vs manual max_abs={m_hf_man:.3e} mean_abs={mean_hf_man:.3e}",
    flush=True,
)
if m_hook_hf > 1e-5:
    print("warning: hook vs direct forward mismatch — unexpected; check hook kwargs.", flush=True)
if m_hf_man > 1e-2:
    print(
        "warning: manual vs official forward — manual uses recurrent loop; official may use chunk/FLA; "
        "also check dtype. Recurrent math matches ``torch_recurrent_gated_delta_rule`` in modeling_qwen3_5.py.",
        flush=True,
    )

# %% Logits: official full forward vs manual text stack (manual Gated-DeltaNet layers)


def _qwen35_inputs_embeds_and_position_ids(
    m,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Match ``Qwen3_5Model`` / ``Qwen3_5ForCausalLM`` inputs to ``language_model`` (text-only)."""
    if isinstance(m, Qwen3_5ForConditionalGeneration):
        wrap = m.model
        emb = wrap.get_input_embeddings()(input_ids)
        pos = wrap.compute_3d_position_ids(
            input_ids=input_ids,
            inputs_embeds=emb,
            attention_mask=attention_mask,
            past_key_values=None,
            mm_token_type_ids=None,
            image_grid_thw=None,
            video_grid_thw=None,
        )
        return emb, pos
    if isinstance(m, Qwen3_5ForCausalLM):
        return m.model.embed_tokens(input_ids), None
    raise TypeError(f"Unexpected model type {type(m).__name__}; extend _qwen35_inputs_embeds_and_position_ids.")


def forward_text_stack_manual_linear(
    lm,
    *,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor | None,
    position_ids: torch.Tensor | None,
) -> torch.Tensor:
    """Mirror ``Qwen3_5TextModel.forward`` but use ``linear_attn_forward_no_cache`` on linear layers."""
    cfg = lm.config
    past_key_values = None

    pos = position_ids
    if pos is None:
        past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
        pos = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen
        pos = pos.view(1, 1, -1).expand(4, inputs_embeds.shape[0], -1)
    elif pos.ndim == 2:
        pos = pos[None, ...].expand(4, pos.shape[0], -1)

    if pos.ndim == 3 and pos.shape[0] == 4:
        text_position_ids = pos[0]
        pos = pos[1:]
    else:
        text_position_ids = None

    causal_mask = create_causal_mask(
        config=cfg,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        position_ids=text_position_ids,
    )
    linear_attn_mask = lm._update_linear_attn_mask(attention_mask, past_key_values)

    hidden_states = inputs_embeds
    position_embeddings = lm.rotary_emb(hidden_states, pos)

    for i, decoder_layer in enumerate(lm.layers[: cfg.num_hidden_layers]):
        layer_mask = linear_attn_mask if cfg.layer_types[i] == "linear_attention" else causal_mask
        residual = hidden_states
        x = decoder_layer.input_layernorm(hidden_states)
        if decoder_layer.layer_type == "linear_attention":
            x = linear_attn_forward_no_cache(
                decoder_layer.linear_attn,
                x,
                attention_mask=layer_mask,
            )
        else:
            x, _ = decoder_layer.self_attn(
                hidden_states=x,
                attention_mask=layer_mask,
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                position_embeddings=position_embeddings,
            )
        hidden_states = residual + x
        residual = hidden_states
        hidden_states = residual + decoder_layer.mlp(decoder_layer.post_attention_layernorm(hidden_states))

    return lm.norm(hidden_states)


# %% Compare logits: official ``model`` / ``model.model`` vs manual text stack
emb, pos_lm = _qwen35_inputs_embeds_and_position_ids(model, input_ids, attention_mask)
with torch.inference_mode():
    logits_hf = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
    hidden_hf = model.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).last_hidden_state
    hidden_manual = forward_text_stack_manual_linear(
        text_module,
        inputs_embeds=emb,
        attention_mask=attention_mask,
        position_ids=pos_lm,
    )
    logits_manual = model.lm_head(hidden_manual)

mx_h, mean_h = _max_mean_abs_diff(hidden_hf, hidden_manual)
mx_log, mean_log = _max_mean_abs_diff(logits_hf, logits_manual)


def _corner5(t: torch.Tensor) -> str:
    xs = t.detach().float().reshape(-1)[:5].tolist()
    return "[" + ", ".join(f"{x:.6g}" for x in xs) + "]"


print(
    "Compare full sequence (official ``model.model`` vs manual text stack, manual Gated-DeltaNet only):\n"
    f"  last_hidden_state |official−manual|: max_abs={mx_h:.3e} mean_abs={mean_h:.3e}\n"
    f"    first 5 (flat) official {_corner5(hidden_hf)}  manual {_corner5(hidden_manual)}\n"
    f"  logits            |official−manual|: max_abs={mx_log:.3e} mean_abs={mean_log:.3e}\n"
    f"    first 5 (flat) official {_corner5(logits_hf)}  manual {_corner5(logits_manual)}",
    flush=True,
)

# %% Decoder layer wiring (from Qwen3_5DecoderLayer)
print(
    "Per decoder layer: input_layernorm → "
    + ("linear_attn (Gated-DeltaNet)" if layer.layer_type == "linear_attention" else "self_attn")
    + " → residual → post_attention_layernorm → MLP → residual."
)
print("layer_type for hooked layer:", layer.layer_type)

# %% Gated delta rule (torch fallback — same math as ``torch_recurrent_gated_delta_rule``)
# One step t (see modeling_qwen3_5.py): state decays by exp(g_t), beta gates the value update,
# output is sum_k q_t[k] * state_t[k, :] over the key dimension folded into state.


def recurrent_rule_source() -> tuple[Path, int]:
    path = Path(inspect.getfile(torch_recurrent_gated_delta_rule)).resolve()
    line = inspect.getsourcelines(torch_recurrent_gated_delta_rule)[1]
    return path, line


p, ln = recurrent_rule_source()
print("Scalar-time recurrent form (reference):", p, "starting line", ln)

# %% Optional: argmax sanity (logits already compared to manual stack above)
with torch.inference_mode():
    out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    top = out.logits[0, -1].argmax(-1).item()
print("argmax next token id:", top, "→", tokenizer.decode([top], skip_special_tokens=False))
