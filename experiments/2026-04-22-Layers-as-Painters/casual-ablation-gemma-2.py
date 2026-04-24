# %% Imports & load gemma-2-2b-it
from __future__ import annotations

import stonesoup
from stonesoup.experiment import (
    configure_matplotlib_agg,
    decoder_blocks,
    hf_repo_id_safe_stem,
    inner_tokenizer,
)

configure_matplotlib_agg()

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

MODEL_ID = "google/gemma-2-2b-it"
PROMPT = "Name the capital of France in one short sentence."
MAX_NEW_TOKENS = 32

model, proc = stonesoup.load_model(MODEL_ID)
model.eval()
tokenizer = inner_tokenizer(proc)
device = next(model.parameters()).device

blocks = decoder_blocks(model)
num_layers = len(blocks)
safe = hf_repo_id_safe_stem(MODEL_ID)
print(f"num_layers: {num_layers}")

# Gemma2DecoderLayer.forward adds two residuals:
#   residual + post_attention_layernorm(self_attn(input_layernorm(x)))
#   residual + post_feedforward_layernorm(mlp(pre_feedforward_layernorm(x)))
# → scale post_attention_layernorm output to ablate attn's contribution,
#   scale post_feedforward_layernorm output to ablate mlp's contribution.

# %% Generate baseline response; list response tokens with their indices
messages = [{"role": "user", "content": PROMPT}]
chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
prompt_ids = tokenizer(chat_text, return_tensors="pt").input_ids.to(device)

with torch.no_grad():
    gen = model.generate(
        prompt_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
response_ids = gen[0, prompt_ids.shape[1]:]
print("=== response ===")
print(tokenizer.decode(response_ids, skip_special_tokens=True))
print("=== response tokens (pick an index for cut_pos) ===")
for i, tid in enumerate(response_ids.tolist()):
    print(f"[{i:3d}] {tokenizer.decode([tid])!r}")

# %% Pick cut_pos: predict response_ids[cut_pos] given prompt + response[:cut_pos] # stonesoup:cell-input
cut_pos = int((str(globals().get("CELL_INPUT", "") or "").strip() or "0"))
prefix_ids = torch.cat([prompt_ids[0], response_ids[:cut_pos]]).unsqueeze(0)
target_id = int(response_ids[cut_pos].item())
print(f"cut_pos = {cut_pos}")
print(f"target token: {tokenizer.decode([target_id])!r}  (id={target_id})")
print(f"prefix tail:  ...{tokenizer.decode(prefix_ids[0, -12:].tolist())!r}")

# %% Per-submodule scales + register hooks once
attn_scales = [1.0] * num_layers
mlp_scales  = [1.0] * num_layers


def make_scale_hook(scales, i):
    def hook(_m, _in, out):
        s = scales[i]
        if isinstance(out, tuple):
            return (out[0] * s,) + out[1:]
        return out * s
    return hook


attn_handles = [
    b.post_attention_layernorm.register_forward_hook(make_scale_hook(attn_scales, i))
    for i, b in enumerate(blocks)
]
mlp_handles = [
    b.post_feedforward_layernorm.register_forward_hook(make_scale_hook(mlp_scales, i))
    for i, b in enumerate(blocks)
]
print("hooks registered: post_attention_layernorm + post_feedforward_layernorm, per block")


# %% Baseline logits + ablate one submodule at a time; measure KL(baseline || ablated)
def next_token_logits() -> torch.Tensor:
    with torch.no_grad():
        return model(prefix_ids).logits[0, -1].detach().float().cpu()


attn_scales[:] = [1.0] * num_layers
mlp_scales[:]  = [1.0] * num_layers
baseline_logits = next_token_logits()
log_p = F.log_softmax(baseline_logits, dim=-1)
pred_id = int(baseline_logits.argmax().item())
print(f"baseline top-1: {tokenizer.decode([pred_id])!r} (id={pred_id}) — target was {tokenizer.decode([target_id])!r}")

kl_attn = np.zeros(num_layers)
kl_mlp  = np.zeros(num_layers)

for i in range(num_layers):
    stonesoup.check_abort()

    attn_scales[:] = [1.0] * num_layers
    mlp_scales[:]  = [1.0] * num_layers
    attn_scales[i] = 0.0
    log_q = F.log_softmax(next_token_logits(), dim=-1)
    kl_attn[i] = F.kl_div(log_q, log_p, reduction="sum", log_target=True).item()

    attn_scales[:] = [1.0] * num_layers
    mlp_scales[:]  = [1.0] * num_layers
    mlp_scales[i] = 0.0
    log_q = F.log_softmax(next_token_logits(), dim=-1)
    kl_mlp[i] = F.kl_div(log_q, log_p, reduction="sum", log_target=True).item()

    print(f"L{i:2d}  KL(attn→0)={kl_attn[i]:.4f}   KL(mlp→0)={kl_mlp[i]:.4f}")

attn_scales[:] = [1.0] * num_layers
mlp_scales[:]  = [1.0] * num_layers

# %% Plot submodule importance
fig, ax = plt.subplots(figsize=(10, 4))
x = np.arange(num_layers)
ax.bar(x - 0.2, kl_attn, width=0.4, label="attn (post_attention_layernorm → 0)", color="#1f77b4")
ax.bar(x + 0.2, kl_mlp,  width=0.4, label="mlp (post_feedforward_layernorm → 0)", color="#ff7f0e")
ax.set_xlabel("layer")
ax.set_ylabel("KL(baseline ‖ ablated)")
ax.set_title(
    f"{MODEL_ID} — per-submodule ablation importance\n"
    f"cut_pos={cut_pos}, target={tokenizer.decode([target_id])!r}"
)
ax.set_xticks(x)
ax.set_ylim([0,2])
ax.grid(alpha=0.3)
ax.legend()
stonesoup.show(fig, basename=f"{safe}_ablation_kl_cut{cut_pos}", dpi=140)
plt.close(fig)

# %% Sweep: ablate at every cut position (0 .. n_tokens-1), collect KL matrices
n_tokens = int(response_ids.shape[0])
kl_attn_all = np.zeros((n_tokens, num_layers))
kl_mlp_all  = np.zeros((n_tokens, num_layers))
targets: list[int] = []


def logits_for_prefix(pref: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return model(pref).logits[0, -1].detach().float().cpu()


for cut in range(n_tokens):
    stonesoup.check_abort()
    pref = torch.cat([prompt_ids[0], response_ids[:cut]]).unsqueeze(0)
    targets.append(int(response_ids[cut].item()))

    attn_scales[:] = [1.0] * num_layers
    mlp_scales[:]  = [1.0] * num_layers
    log_p_cut = F.log_softmax(logits_for_prefix(pref), dim=-1)

    for i in range(num_layers):
        attn_scales[:] = [1.0] * num_layers
        mlp_scales[:]  = [1.0] * num_layers
        attn_scales[i] = 0.0
        log_q = F.log_softmax(logits_for_prefix(pref), dim=-1)
        kl_attn_all[cut, i] = F.kl_div(log_q, log_p_cut, reduction="sum", log_target=True).item()

        attn_scales[:] = [1.0] * num_layers
        mlp_scales[:]  = [1.0] * num_layers
        mlp_scales[i] = 0.0
        log_q = F.log_softmax(logits_for_prefix(pref), dim=-1)
        kl_mlp_all[cut, i] = F.kl_div(log_q, log_p_cut, reduction="sum", log_target=True).item()

    print(
        f"cut={cut:3d} target={tokenizer.decode([targets[cut]])!r}  "
        f"max KL_attn={kl_attn_all[cut].max():.3f}  max KL_mlp={kl_mlp_all[cut].max():.3f}"
    )

attn_scales[:] = [1.0] * num_layers
mlp_scales[:]  = [1.0] * num_layers

# %% Plot: y=layer, x=token; attn bar left, mlp bar right; length ∝ KL # stonesoup:cell-input
from matplotlib.patches import Patch

kl_cap = float((str(globals().get("CELL_INPUT", "") or "").strip() or "2"))
bar_half = 0.45  # max horizontal extent (in x units) for a bar at kl_cap
bar_h = 0.7
scale = bar_half / kl_cap if kl_cap > 0 else 0.0

fig, ax = plt.subplots(
    figsize=(max(8.0, 0.9 * n_tokens), max(5.0, 0.22 * num_layers)),
)
layers_y = np.arange(num_layers)

for cut in range(n_tokens):
    a = np.minimum(kl_attn_all[cut], kl_cap) * scale
    m = np.minimum(kl_mlp_all[cut],  kl_cap) * scale
    ax.barh(layers_y, -a, left=cut, height=bar_h, color="#1f77b4", edgecolor="none")
    ax.barh(layers_y,  m, left=cut, height=bar_h, color="#ff7f0e", edgecolor="none")

for cut in range(n_tokens):
    ax.axvline(cut, color="#cccccc", linewidth=0.4, zorder=0)

ax.set_xticks(np.arange(n_tokens))
ax.set_xticklabels(
    [f"[{i}] {tokenizer.decode([targets[i]])!r}" for i in range(n_tokens)],
    rotation=-30, ha="left", fontsize=8,
)
ax.set_yticks(layers_y)
ax.set_yticklabels([f"L{i}" for i in layers_y], fontsize=7)
ax.set_xlim(-0.6, n_tokens - 0.4)
ax.set_ylim(num_layers - 0.3, -0.7)  # L0 on top, last layer at bottom
ax.set_xlabel(f"token position  (← attn | mlp →,  bar length ∝ KL capped at {kl_cap})")
ax.set_ylabel("layer")
ax.grid(alpha=0.25, axis="y", linestyle=":")
ax.legend(
    handles=[
        Patch(color="#1f77b4", label="attn (left)"),
        Patch(color="#ff7f0e", label="mlp (right)"),
    ],
    loc="upper right",
)
ax.set_title(
    f"{MODEL_ID} — per-submodule ablation KL  (y=layer, x=token, cap={kl_cap})"
)
fig.tight_layout()
stonesoup.show(fig, basename=f"{safe}_ablation_kl_centered_cap{kl_cap}", dpi=140)
plt.close(fig)

# %% End-to-end from CELL_INPUT prompt: generate → sweep → centered bar plot # stonesoup:cell-input
stonesoup.html()
import hashlib

from matplotlib.patches import Patch

custom_prompt = (str(globals().get("CELL_INPUT", "") or "").strip()) or PROMPT
print(f"prompt: {custom_prompt!r}")

# Re-register hooks fresh so this cell works even if the remove-hooks cell ran,
# or the earlier register cell was re-run and stale handles now point nowhere.
for _h in list(globals().get("attn_handles", [])) + list(globals().get("mlp_handles", [])):
    _h.remove()
attn_scales = [1.0] * num_layers
mlp_scales  = [1.0] * num_layers
attn_handles = [
    b.post_attention_layernorm.register_forward_hook(make_scale_hook(attn_scales, i))
    for i, b in enumerate(blocks)
]
mlp_handles = [
    b.post_feedforward_layernorm.register_forward_hook(make_scale_hook(mlp_scales, i))
    for i, b in enumerate(blocks)
]

custom_messages = [{"role": "user", "content": custom_prompt}]
custom_chat = tokenizer.apply_chat_template(custom_messages, tokenize=False, add_generation_prompt=True)
custom_prompt_ids = tokenizer(custom_chat, return_tensors="pt").input_ids.to(device)
with torch.no_grad():
    custom_gen = model.generate(
        custom_prompt_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
custom_response_ids = custom_gen[0, custom_prompt_ids.shape[1]:]
print("=== response ===")
print(tokenizer.decode(custom_response_ids, skip_special_tokens=True))

n_tokens_c = int(custom_response_ids.shape[0])
kl_attn_c = np.zeros((n_tokens_c, num_layers))
kl_mlp_c  = np.zeros((n_tokens_c, num_layers))
targets_c: list[int] = []

for cut in range(n_tokens_c):
    stonesoup.check_abort()
    pref = torch.cat([custom_prompt_ids[0], custom_response_ids[:cut]]).unsqueeze(0)
    targets_c.append(int(custom_response_ids[cut].item()))

    attn_scales[:] = [1.0] * num_layers
    mlp_scales[:]  = [1.0] * num_layers
    log_p_cut = F.log_softmax(logits_for_prefix(pref), dim=-1)

    for i in range(num_layers):
        attn_scales[:] = [1.0] * num_layers
        mlp_scales[:]  = [1.0] * num_layers
        attn_scales[i] = 0.0
        log_q = F.log_softmax(logits_for_prefix(pref), dim=-1)
        kl_attn_c[cut, i] = F.kl_div(log_q, log_p_cut, reduction="sum", log_target=True).item()

        attn_scales[:] = [1.0] * num_layers
        mlp_scales[:]  = [1.0] * num_layers
        mlp_scales[i] = 0.0
        log_q = F.log_softmax(logits_for_prefix(pref), dim=-1)
        kl_mlp_c[cut, i] = F.kl_div(log_q, log_p_cut, reduction="sum", log_target=True).item()

    print(
        f"cut={cut:3d} target={tokenizer.decode([targets_c[cut]])!r}  "
        f"max KL_attn={kl_attn_c[cut].max():.3f}  max KL_mlp={kl_mlp_c[cut].max():.3f}"
    )

attn_scales[:] = [1.0] * num_layers
mlp_scales[:]  = [1.0] * num_layers

# %% Plot — same centered-bar design as the sweep cell.
kl_cap = 2.0
bar_half = 0.45
bar_h = 0.7
scale = bar_half / kl_cap

fig, ax = plt.subplots(figsize=(max(8.0, 0.9 * n_tokens_c), max(5.0, 0.22 * num_layers)))
layers_y = np.arange(num_layers)
for cut in range(n_tokens_c):
    a = np.minimum(kl_attn_c[cut], kl_cap) * scale
    m = np.minimum(kl_mlp_c[cut],  kl_cap) * scale
    ax.barh(layers_y, -a, left=cut, height=bar_h, color="#1f77b4", edgecolor="none")
    ax.barh(layers_y,  m, left=cut, height=bar_h, color="#ff7f0e", edgecolor="none")
    ax.axvline(cut, color="#cccccc", linewidth=0.4, zorder=0)

ax.set_xticks(np.arange(n_tokens_c))
ax.set_xticklabels(
    [f"[{i}] {tokenizer.decode([targets_c[i]])!r}" for i in range(n_tokens_c)],
    rotation=-30, ha="left", fontsize=8,
)
ax.set_yticks(layers_y)
ax.set_yticklabels([f"L{i}" for i in layers_y], fontsize=7)
ax.set_xlim(-0.6, n_tokens_c - 0.4)
ax.set_ylim(num_layers - 0.3, -0.7)  # L0 on top, last layer at bottom
ax.set_xlabel(f"token position  (← attn | mlp →,  bar length ∝ KL capped at {kl_cap:.1f})")
ax.set_ylabel("layer")
ax.grid(alpha=0.25, axis="y", linestyle=":")
ax.legend(
    handles=[Patch(color="#1f77b4", label="attn"), Patch(color="#ff7f0e", label="mlp")],
    loc="upper right",
)
ax.set_title(f"{MODEL_ID} — prompt={custom_prompt!r}", fontsize=10)
fig.tight_layout()
prompt_tag = hashlib.md5(custom_prompt.encode()).hexdigest()[:8]
stonesoup.show(fig, basename=f"{safe}_ablation_kl_custom_{prompt_tag}", dpi=140)
plt.close(fig)

# %% Sanity check: ablate at (cut, layer) — does top-1 prediction actually change? # stonesoup:cell-input
# CELL_INPUT format: "cut,layer"  (e.g. "3,25"). Default: "3,25".
parts = [p.strip() for p in (str(globals().get("CELL_INPUT", "") or "").strip() or "3,25").split(",")]
verify_cut, verify_layer = int(parts[0]), int(parts[1])

pref = torch.cat([custom_prompt_ids[0], custom_response_ids[:verify_cut]]).unsqueeze(0)
target = int(custom_response_ids[verify_cut].item())
print(f"cut={verify_cut}  target={tokenizer.decode([target])!r} (id={target})  layer=L{verify_layer}")

attn_scales[:] = [1.0] * num_layers
mlp_scales[:]  = [1.0] * num_layers
logits_base = logits_for_prefix(pref)
log_p = F.log_softmax(logits_base, dim=-1)
top1_base = int(logits_base.argmax().item())
top_base_vals, top_base_ids = logits_base.topk(5)
print(f"baseline top-1: {tokenizer.decode([top1_base])!r} (id={top1_base})")
print(f"baseline top-5: " + ", ".join(f"{tokenizer.decode([int(t)])!r}:{float(v):.2f}" for v, t in zip(top_base_vals, top_base_ids)))


def ablate_and_report(name: str, scales: list[float], i: int) -> None:
    scales[i] = 0.0
    logits = logits_for_prefix(pref)
    top1 = int(logits.argmax().item())
    kl = F.kl_div(F.log_softmax(logits, dim=-1), log_p, reduction="sum", log_target=True).item()
    flipped = "FLIPPED" if top1 != top1_base else "same"
    top_vals, top_ids = logits.topk(5)
    print(f"{name} @ L{i}: KL={kl:.4e}  top-1={tokenizer.decode([top1])!r} ({flipped})")
    print(f"    top-5: " + ", ".join(f"{tokenizer.decode([int(t)])!r}:{float(v):.2f}" for v, t in zip(top_vals, top_ids)))
    scales[i] = 1.0


attn_scales[:] = [1.0] * num_layers
mlp_scales[:]  = [1.0] * num_layers
ablate_and_report("ablate attn", attn_scales, verify_layer)

attn_scales[:] = [1.0] * num_layers
mlp_scales[:]  = [1.0] * num_layers
ablate_and_report("ablate mlp ", mlp_scales, verify_layer)

attn_scales[:] = [1.0] * num_layers
mlp_scales[:]  = [1.0] * num_layers

# %% Inspect attention: which source tokens does layer L at cut C mainly attend to? # stonesoup:cell-input
# CELL_INPUT format: "cut,layer"  (e.g. "8,0"). Default: "8,0".
parts = [p.strip() for p in (str(globals().get("CELL_INPUT", "") or "").strip() or "8,0").split(",")]
inspect_cut, inspect_layer = int(parts[0]), int(parts[1])

attn_scales[:] = [1.0] * num_layers
mlp_scales[:]  = [1.0] * num_layers

pref = torch.cat([custom_prompt_ids[0], custom_response_ids[:inspect_cut]]).unsqueeze(0)
target = int(custom_response_ids[inspect_cut].item())
print(f"cut={inspect_cut}  target={tokenizer.decode([target])!r}  layer=L{inspect_layer}")

# Capture attn_weights via a hook on self_attn; Gemma2DecoderLayer's self_attn
# returns (attn_output, attn_weights) when output_attentions=True is propagated.
captured_attn: dict[int, torch.Tensor] = {}


def make_attn_capture_hook(i: int):
    def hook(_mod, _in, output):
        if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
            captured_attn[i] = output[1].detach().float().cpu()
    return hook


capture_handles = [
    b.self_attn.register_forward_hook(make_attn_capture_hook(i))
    for i, b in enumerate(blocks)
]

# Temporarily force eager attention so output_attentions=True actually produces weights.
submodule_configs: list[object] = []
for b in blocks:
    if hasattr(b.self_attn, "config"):
        submodule_configs.append(b.self_attn.config)
orig_impls = [
    model.config._attn_implementation,
    *[c._attn_implementation for c in submodule_configs],
]
try:
    model.config._attn_implementation = "eager"
    for c in submodule_configs:
        c._attn_implementation = "eager"
    with torch.no_grad():
        model(pref, output_attentions=True)
finally:
    model.config._attn_implementation = orig_impls[0]
    for c, impl in zip(submodule_configs, orig_impls[1:]):
        c._attn_implementation = impl
    for h in capture_handles:
        h.remove()

attn_layer = captured_attn.get(inspect_layer)

if attn_layer is None:
    print(
        f"could not capture attention weights for L{inspect_layer} — "
        f"this attention impl ({getattr(model.config, '_attn_implementation', '?')}) "
        f"did not expose them. Reload with attn_implementation='eager' and re-run."
    )
else:
    # attn_layer: (batch=1, num_heads, seq, seq). Query at last position predicts the next token.
    attn_mat = attn_layer[0].detach().float().cpu().numpy()
    num_heads, seq_len, _ = attn_mat.shape
    q_pos = seq_len - 1
    avg_attn = attn_mat[:, q_pos, :].mean(axis=0)  # (seq,) averaged over heads

    token_ids = pref[0].tolist()
    token_strs = [tokenizer.decode([t]) for t in token_ids]

    topk = 15
    top_idx = np.argsort(avg_attn)[::-1][:topk]
    print(f"num_heads={num_heads}  seq_len={seq_len}  query_pos={q_pos}")
    print(f"top-{topk} attended source tokens (head-averaged) from q_pos={q_pos}:")
    for i in top_idx:
        print(f"  src[{i:4d}]  w={avg_attn[i]:.4f}  {token_strs[i]!r}")

    fig, ax = plt.subplots(figsize=(max(8.0, 0.22 * seq_len), 3.8))
    ax.bar(np.arange(seq_len), avg_attn, color="#1f77b4", edgecolor="none")
    for i in top_idx:
        ax.annotate(
            token_strs[i], (i, avg_attn[i]),
            fontsize=6, ha="center", va="bottom", rotation=-35,
        )
    ax.set_xlabel("source token position (query = last)")
    ax.set_ylabel("avg attention weight (over heads)")
    ax.set_title(
        f"{MODEL_ID} — L{inspect_layer} attention at cut={inspect_cut}  "
        f"target={tokenizer.decode([target])!r}"
    )
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    stonesoup.show(
        fig,
        basename=f"{safe}_attn_inspect_cut{inspect_cut}_L{inspect_layer}",
        dpi=140,
    )
    plt.close(fig)

# %% Remove hooks
for h in attn_handles + mlp_handles:
    h.remove()
print("hooks removed")
