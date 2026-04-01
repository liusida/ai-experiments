# %% MoD (Chen et al.) — minimal demo, one diagram (image via Stonesoup URL)

from __future__ import annotations

import torch
import torch.nn.functional as F

print(
    "Cells: imports (this) | load VLM | knobs | tokenize | prefill P | mask | Q forward | JSD + mix + print | figure\n",
    flush=True,
)

# %% Load VLM (Stonesoup **Load** or this cell)

import stonesoup

# Qwen3 / Qwen3.5: chat template can tokenize in one step.
# Qwen2.5-VL: use ``qwen_vl_utils.process_vision_info`` + ``processor(text=..., images=...)``
# (see https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct ).
MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
# MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
model, processor = stonesoup.load_model(MODEL_ID)
tokenizer = getattr(processor, "tokenizer", processor)
inner = model.model
img_id = int(model.config.image_token_id)
device = next(model.parameters()).device
print(f"Loaded {MODEL_ID}  device={device}  image_token_id={img_id}", flush=True)

# %% This example — image URL, prompt, MoD hyperparameters

IMAGE_URL = "http://127.0.0.1:8765/data/image/MoD/1.png"
# USER_TEXT = "Use one word to name the stage shown at label A in the diagram."
# USER_TEXT = "Use one word to name the animal in the image."
USER_TEXT = "Use one word to describe the image."
# Keep this many **vision merge** tokens (highest last-query attention); the rest are zeroed for Q.
VISION_MERGE_KEEP = 5
JS_THRESHOLD = 0.05
ALPHA_PH = 4.0
ALPHA_MH = 1.0
BETA = 0.5
SAMPLE_SEED = 42

print(f"IMAGE_URL={IMAGE_URL!r}", flush=True)
print(f"PROMPT={USER_TEXT!r}", flush=True)
print(
    f"knobs: vision_merge_keep={VISION_MERGE_KEEP}  js_threshold={JS_THRESHOLD}  "
    f"alpha_ph={ALPHA_PH}  alpha_mh={ALPHA_MH}  beta={BETA}",
    flush=True,
)

# %% Build model inputs (user + image URL; first assistant token = answer)

msgs = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": IMAGE_URL},
            {"type": "text", "text": USER_TEXT},
        ],
    },
]
_model_type = getattr(model.config, "model_type", "")
if _model_type == "qwen2_5_vl":
    from qwen_vl_utils import process_vision_info

    _chat_text = processor.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )
    _image_inputs, _video_inputs = process_vision_info(msgs)
    inputs = processor(
        text=[_chat_text],
        images=_image_inputs,
        videos=_video_inputs,
        padding=True,
        return_tensors="pt",
    )
else:
    try:
        inputs = processor.apply_chat_template(
            msgs,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=True,
            return_tensors="pt",
        )
    except TypeError:
        inputs = processor.apply_chat_template(
            msgs,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}
input_ids = inputs["input_ids"]
attn_mask = inputs["attention_mask"]
print(f"prefill seq len = {input_ids.shape[1]}", flush=True)

# %% Step 1 — prefill with full image: last-position logits P + attentions into vision tokens

cfg = model.config
_prev_attn_impl = getattr(cfg, "_attn_implementation", None) or "sdpa"
_prev_output_att = bool(getattr(cfg, "output_attentions", False))
model.set_attn_implementation("eager") # eager for getting attentions
cfg.output_attentions = True
try:
    with torch.inference_mode():
        out_p = model(**inputs, output_attentions=True, use_cache=False)
finally:
    cfg.output_attentions = _prev_output_att
    model.set_attn_implementation(_prev_attn_impl)

logits_p = out_p.logits[0, -1].float()
vis_pos = (input_ids[0] == img_id).nonzero(as_tuple=False).flatten()
n_vis = int(vis_pos.numel())
print(f"vision merge tokens = {n_vis}", flush=True)

att = out_p.attentions
print(f"attentions: {len(att)} layers, {att[0].shape} shape", flush=True) # Only 8 layers for Qwen3.5-9B, other layers are Gated DeltaNet, not Transformer.
attn_vecs = []
for layer_attn in att:
    if layer_attn is None or not torch.is_tensor(layer_attn):
        continue
    attn_vecs.append(layer_attn[0, :, -1, :].float().mean(dim=0))
scores_1d = torch.stack(attn_vecs, dim=0).mean(dim=0)
patch_scores = scores_1d[vis_pos].float()

tokp = int(logits_p.argmax().item())
print(f"P: argmax id={tokp}  piece={tokenizer.decode([tokp])!r}", flush=True)

# # %% Step 1.5 — last query position only: scatter(attn to each key); highlight vision key range

# import matplotlib.pyplot as plt
# import numpy as np

# _seq = int(att[0].shape[-1])
# _k0 = int(vis_pos.min().item())
# _k1 = int(vis_pos.max().item()) + 1  # vision keys [k0, k1)
# _x = np.arange(_seq, dtype=np.float64)
# _vision_keys = np.zeros(_seq, dtype=bool)
# _vision_keys[_k0:_k1] = True

# fig, axes = plt.subplots(8, 16, figsize=(32, 16), sharex=True)
# for layer in range(8):
#     for head in range(16):
#         ax = axes[layer, head]
#         last_row = att[layer][0, head, -1, :].detach().cpu().float().numpy()
#         ax.axvspan(_k0 - 0.5, _k1 - 0.5, color="#ffcc00", alpha=0.12, zorder=0)
#         ax.scatter(
#             _x[~_vision_keys],
#             last_row[~_vision_keys],
#             s=3,
#             c="#4466aa",
#             alpha=0.45,
#             linewidths=0,
#             zorder=1,
#         )
#         ax.scatter(
#             _x[_vision_keys],
#             last_row[_vision_keys],
#             s=5,
#             c="#ee6600",
#             alpha=0.9,
#             linewidths=0,
#             zorder=2,
#         )
#         ymax = float(np.max(last_row)) * 1.08 + 1e-10
#         ax.set_ylim(0, 0.1)
#         ax.set_xlim(-0.5, _seq - 0.5)
#         ax.set_title(f"L{layer} H{head}", fontsize=7)
#         ax.tick_params(labelsize=5)
#         if head != 0:
#             ax.set_ylabel("")
#         if layer < 7:
#             ax.set_xlabel("")

# plt.suptitle(
#     f"Last-token query → each key  |  orange = vision keys [{_k0}, {_k1})  |  blue = text/other keys",
#     y=1.002,
#     fontsize=11,
# )
# plt.tight_layout()
# stonesoup.show()
# plt.close("all")

# %% Step 1.6 — **vision-only** MoD masses + cutoff (do not draw vision cutoff on full key axis: text keys sit above it too)

_n_keep_plot = min(max(VISION_MERGE_KEEP, 0), n_vis)
_nm_plot = n_vis - _n_keep_plot
if _nm_plot > 0:
    _lowest_m, _ = torch.topk(patch_scores, _nm_plot, largest=False)
    _vision_attn_cutoff = float(_lowest_m.max().item())
else:
    _vision_attn_cutoff = None

attention_map = torch.stack(attn_vecs, dim=0).mean(dim=0)
_v0 = int(vis_pos.min().item())
_v1 = int(vis_pos.max().item()) + 1

fig, (ax_full, ax_vis) = plt.subplots(2, 1, figsize=(10, 6), sharex=False, constrained_layout=True)

ax_full.plot(attention_map.cpu().numpy(), color="#2a4a6a", linewidth=0.85, label="last→key (all keys)")
ax_full.axvline(
    _v0 - 0.5,
    color="#cc8800",
    linestyle="--",
    linewidth=1.3,
    label=f"vision keys [{_v0}, {_v1})",
)
ax_full.axvline(_v1 - 0.5, color="#cc8800", linestyle="--", linewidth=1.3)
ax_full.set_ylabel("mean layer attn")
ax_full.set_ylim(0, 0.1)
ax_full.set_title(
    "Full sequence (context): text keys after ~{:d} often have **higher** attn than vision — "
    "not in ``patch_scores``.".format(_v1)
)
ax_full.legend(loc="upper right", fontsize=8)

_ord = torch.argsort(vis_pos)
_vx = vis_pos[_ord].detach().cpu().numpy()
_vy = patch_scores[_ord].detach().cpu().float().numpy()
ax_vis.axvline(_v0 - 0.5, color="#cc8800", linestyle="--", linewidth=1.0, alpha=0.75, zorder=1)
ax_vis.axvline(_v1 - 0.5, color="#cc8800", linestyle="--", linewidth=1.0, alpha=0.75, zorder=1)
ax_vis.plot(_vx, _vy, color="#3355aa", linewidth=0.9, label="vision keys only")
ax_vis.scatter(_vx, _vy, s=8, c="#3355aa", alpha=0.6, zorder=3)
if _vision_attn_cutoff is not None:
    ax_vis.axhline(
        _vision_attn_cutoff,
        color="#cc3311",
        linestyle="--",
        linewidth=1.1,
        label=(
            f"MoD cutoff: max of {_nm_plot} smallest **vision** masses "
            f"(keep {_n_keep_plot})"
        ),
    )
if _n_keep_plot > 0:
    _kept_local = torch.topk(patch_scores, _n_keep_plot, largest=True).indices
    _kx = vis_pos[_kept_local].detach().cpu().numpy()
    _ky = patch_scores[_kept_local].detach().cpu().float().numpy()
    ax_vis.scatter(_kx, _ky, s=120, c="none", edgecolors="#00aa44", linewidths=1.8, zorder=5, label=f"kept {_n_keep_plot} (MoD)")
ax_vis.set_xlabel("key index (vision merge positions only)")
ax_vis.set_ylabel("mean layer attn → this vision key")
ax_vis.set_title(f"VISION_MERGE_KEEP={VISION_MERGE_KEEP}  ·  only these {_n_keep_plot} keys stay unmasked for Q")
ax_vis.legend(loc="upper right", fontsize=8)

stonesoup.show()
plt.close("all")

# %% Step 1.7 — post-image **text** queries → vision keys (mean over heads, layers, and all such queries)

import sys

import matplotlib.pyplot as plt

_seq_len_17 = int(input_ids.shape[1])
_vimax_17 = int(vis_pos.max().item())
_post_img_q = (
    (torch.arange(_seq_len_17, device=input_ids.device) > _vimax_17)
    & (input_ids[0] != img_id)
).nonzero(as_tuple=False).flatten()
_n_post_q = int(_post_img_q.numel())

if _n_post_q == 0:
    print(
        "Step 1.7: no non-vision tokens after last vision token; skip figure.",
        file=sys.stderr,
        flush=True,
    )
else:
    _per_layer_post: list[torch.Tensor] = []
    _per_layer_full_post: list[torch.Tensor] = []
    for layer_attn in att:
        if layer_attn is None or not torch.is_tensor(layer_attn):
            continue
        a = layer_attn[0].float()
        sub = a[:, _post_img_q, :]
        _per_layer_full_post.append(sub.mean(dim=(0, 1)))
        sub_vis = sub.index_select(2, vis_pos.long())
        _per_layer_post.append(sub_vis.mean(dim=(0, 1)))
    attention_map_post = torch.stack(_per_layer_full_post, dim=0).mean(dim=0)
    patch_scores_post = torch.stack(_per_layer_post, dim=0).mean(dim=0)

    _n_keep_p = min(max(VISION_MERGE_KEEP, 0), n_vis)
    _nm_p = n_vis - _n_keep_p
    if _nm_p > 0:
        _lowest_post, _ = torch.topk(patch_scores_post, _nm_p, largest=False)
        _vision_cut_post = float(_lowest_post.max().item())
    else:
        _vision_cut_post = None

    print(
        f"Step 1.7: post-image text query count = {_n_post_q}  (key idx > {_vimax_17}, id != image_token)",
        file=sys.stderr,
        flush=True,
    )

    _v0_17 = int(vis_pos.min().item())
    _v1_17 = int(vis_pos.max().item()) + 1

    fig, (ax_full, ax_vis) = plt.subplots(2, 1, figsize=(10, 6), sharex=False, constrained_layout=True)

    ax_full.plot(
        attention_map_post.detach().cpu().numpy(),
        color="#2a4a6a",
        linewidth=0.85,
        label=f"post-image queries → each key (mean L/H/q, n_q={_n_post_q})",
    )
    ax_full.axvline(
        _v0_17 - 0.5,
        color="#cc8800",
        linestyle="--",
        linewidth=1.3,
        label=f"vision keys [{_v0_17}, {_v1_17})",
    )
    ax_full.axvline(_v1_17 - 0.5, color="#cc8800", linestyle="--", linewidth=1.3)
    ax_full.set_ylabel("mean layer attn")
    ax_full.set_ylim(0, 0.1)
    ax_full.set_title(
        "Full sequence: post-image text → all keys (same layout as Step 1.6; **data** = post-image average)"
    )
    ax_full.legend(loc="upper right", fontsize=8)

    _ord_17 = torch.argsort(vis_pos)
    _vx_17 = vis_pos[_ord_17].detach().cpu().numpy()
    _vy_17 = patch_scores_post[_ord_17].detach().cpu().float().numpy()
    ax_vis.axvline(_v0_17 - 0.5, color="#cc8800", linestyle="--", linewidth=1.0, alpha=0.75, zorder=1)
    ax_vis.axvline(_v1_17 - 0.5, color="#cc8800", linestyle="--", linewidth=1.0, alpha=0.75, zorder=1)
    ax_vis.plot(_vx_17, _vy_17, color="#3355aa", linewidth=0.9, label="vision keys only (post-image avg)")
    ax_vis.scatter(_vx_17, _vy_17, s=8, c="#3355aa", alpha=0.6, zorder=3)
    if _vision_cut_post is not None:
        ax_vis.axhline(
            _vision_cut_post,
            color="#cc3311",
            linestyle="--",
            linewidth=1.1,
            label=(
                f"MoD-style cutoff on **post** masses: max of {_nm_p} smallest "
                f"(keep {_n_keep_p})  ·  Step 2 still uses last-token masses"
            ),
        )
    if _n_keep_p > 0:
        _kept_post = torch.topk(patch_scores_post, _n_keep_p, largest=True).indices
        _kx_17 = vis_pos[_kept_post].detach().cpu().numpy()
        _ky_17 = patch_scores_post[_kept_post].detach().cpu().float().numpy()
        ax_vis.scatter(
            _kx_17,
            _ky_17,
            s=120,
            c="none",
            edgecolors="#00aa44",
            linewidths=1.8,
            zorder=5,
            label=f"top {_n_keep_p} by post-image avg (illustrative)",
        )
    ax_vis.set_xlabel("key index (vision merge positions only)")
    ax_vis.set_ylabel("mean layer attn → this vision key")
    ax_vis.set_title(
        f"VISION_MERGE_KEEP={VISION_MERGE_KEEP}  ·  post-image query average  ·  cutoff shown uses **post** masses"
    )
    ax_vis.legend(loc="upper right", fontsize=8)

    stonesoup.show()
    plt.close("all")

# %% Step 2 — lowest-attention vision cells are dropped for Q (keep top ``VISION_MERGE_KEEP`` by mass)

_n_keep = min(max(VISION_MERGE_KEEP, 0), n_vis)
n_mask = n_vis - _n_keep
_, masked_local = torch.topk(patch_scores, n_mask, largest=False)
mask_global = vis_pos[masked_local.long()]
print(
    f"will zero {n_mask}/{n_vis} vision rows (keeping {_n_keep} highest-attention merge token(s))",
    flush=True,
)

# %% Step 3 — scatter image into placeholders, zero masked rows, forward → Q

emb = inner.get_input_embeddings()(input_ids)
feat = inner.get_image_features(inputs["pixel_values"], inputs["image_grid_thw"], return_dict=True)
pool = feat.pooler_output
ie = torch.cat(pool, dim=0) if isinstance(pool, (list, tuple)) else pool
ie = ie.to(device=device, dtype=emb.dtype)
image_mask, _ = inner.get_placeholder_mask(input_ids, inputs_embeds=emb, image_features=ie)
fused = emb.masked_scatter(image_mask, ie).clone()
if mask_global.numel() > 0:
    fused[0, mask_global, :] = 0

fwd_q = {
    "input_ids": None,
    "inputs_embeds": fused,
    "attention_mask": attn_mask,
    "pixel_values": None,
    "pixel_values_videos": None,
    "output_attentions": False,
    "use_cache": False,
}
for _k in (
    "image_grid_thw",
    "video_grid_thw",
    "mm_token_type_ids",
    "second_per_grid_ts",
):
    if _k in inputs and inputs[_k] is not None:
        fwd_q[_k] = inputs[_k]

with torch.inference_mode():
    out_q = model(**fwd_q)

logits_q = out_q.logits[0, -1].float()
tokq = int(logits_q.argmax().item())
print(f"Q: argmax id={tokq}  piece={tokenizer.decode([tokq])!r}", flush=True)

# %% Step 4 — JSD(P,Q); low JSD → complementary (P + αQ), high → contrastive ((1+α)P − αQ); then plausibility floor vs max(P)

eps = 5e-8
pc = F.softmax(logits_p, dim=-1).clamp(eps, 1.0 - eps)
qc = F.softmax(logits_q, dim=-1).clamp(eps, 1.0 - eps)
mid = 0.5 * (pc + qc)
js_val = float(
    (0.5 * ((pc * (pc.log() - mid.log())).sum() + (qc * (qc.log() - mid.log())).sum())).item()
)
if js_val < JS_THRESHOLD:
    mode_name = "complementary (+)"
    mixed_raw = logits_p + ALPHA_PH * logits_q
else:
    mode_name = "contrastive (−)"
    mixed_raw = (1.0 + ALPHA_MH) * logits_p - ALPHA_MH * logits_q

cutoff = torch.log(torch.tensor(BETA, device=device, dtype=mixed_raw.dtype)) + logits_p.max()
mixed = mixed_raw.masked_fill(logits_p < cutoff, float("-inf"))

print(f"JSD(P,Q) = {js_val:.4f}  → {mode_name}  (threshold γ = {JS_THRESHOLD})", flush=True)

k_top = 12
k_top = min(k_top, logits_p.numel())
vals, idx = torch.topk(logits_p, k_top)
print("--- top-k  P (full vision) ---", flush=True)
for v, i in zip(vals.tolist(), idx.tolist()):
    print(f"  id={i:6d}  logit={v:8.2f}  {tokenizer.decode([i])!r}", flush=True)

vals, idx = torch.topk(logits_q, k_top)
print("--- top-k  Q (masked vision) ---", flush=True)
for v, i in zip(vals.tolist(), idx.tolist()):
    print(f"  id={i:6d}  logit={v:8.2f}  {tokenizer.decode([i])!r}", flush=True)

vals, idx = torch.topk(mixed_raw, k_top)
print("--- top-k  MoD blend (before plausibility) ---", flush=True)
for v, i in zip(vals.tolist(), idx.tolist()):
    print(f"  id={i:6d}  logit={v:8.2f}  {tokenizer.decode([i])!r}", flush=True)

vals, idx = torch.topk(mixed, k_top)
print("--- top-k  MoD + plausibility (masked by P mass) ---", flush=True)
for v, i in zip(vals.tolist(), idx.tolist()):
    print(f"  id={i:6d}  logit={v:8.2f}  {tokenizer.decode([i])!r}", flush=True)

gen = torch.Generator(device=device)
gen.manual_seed(SAMPLE_SEED)
tok_mod = int(torch.multinomial(F.softmax(mixed, dim=-1), 1, generator=gen).item())
print(f"greedy under P → {tokenizer.decode([tokp])!r}", flush=True)
print(f"multinomial under MoD+plaus. (seed={SAMPLE_SEED}) → {tokenizer.decode([tok_mod])!r}", flush=True)

# %% Step 5 — figure: dim merge cells MoD zeros (load bitmap from same URL for overlay)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import urllib.request
from io import BytesIO

from PIL import Image, ImageDraw

with urllib.request.urlopen(IMAGE_URL) as resp:
    pil = Image.open(BytesIO(resp.read())).convert("RGBA")

ip = processor.image_processor
_vis = getattr(inner, "visual", None)
ms = int(
    getattr(
        ip,
        "merge_size",
        getattr(ip, "spatial_merge_size", getattr(_vis, "spatial_merge_size", 2)),
    )
)
ps = int(getattr(ip, "patch_size", getattr(_vis, "patch_size", 14)))
g = inputs["image_grid_thw"][0].detach().cpu()
_t, gh, gw = int(g[0].item()), int(g[1].item()), int(g[2].item())
if _t != 1:
    print(f"expected single image frame T=1, got {_t}; skip figure.", flush=True)
else:
    llm_h, llm_w = gh // ms, gw // ms
    n_merge = llm_h * llm_w
    if n_vis != n_merge:
        print(
            f"vision tokens {n_vis} != merge grid {n_merge}; skip aligned overlay.",
            flush=True,
        )
    else:
        masked_seq = {int(x) for x in mask_global.detach().cpu().tolist()}
        ow, oh = pil.size
        res_wh = (gw * ps, gh * ps)
        sx = ow / res_wh[0]
        sy = oh / res_wh[1]
        base = pil
        shade = Image.new("RGBA", (ow, oh), (0, 0, 0, 0))
        drw = ImageDraw.Draw(shade)
        fill = (16, 24, 90, 175)
        llm_hw = llm_h * llm_w

        for k_merge in range(n_merge):
            if int(vis_pos[k_merge].item()) not in masked_seq:
                continue
            rem = k_merge % llm_hw
            h_llm = rem // llm_w
            w_llm = rem % llm_w
            x0 = w_llm * ms * ps
            y0 = h_llm * ms * ps
            x1 = (w_llm + 1) * ms * ps
            y1 = (h_llm + 1) * ms * ps
            bx = (
                int(round(x0 * sx)),
                int(round(y0 * sy)),
                int(round(x1 * sx)),
                int(round(y1 * sy)),
            )
            drw.rectangle(bx, fill=fill)

        blended = Image.alpha_composite(base, shade).convert("RGB")
        fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
        axes[0].imshow(np.asarray(pil.convert("RGB")))
        axes[0].set_title("original")
        axes[0].axis("off")
        axes[1].imshow(np.asarray(blended))
        axes[1].set_title(
            f"dimmed = low-attn vision (MoD zeros)  keep={n_vis - len(masked_seq)}  masked={len(masked_seq)}"
        )
        axes[1].axis("off")
        stonesoup.show()
        plt.close("all")
