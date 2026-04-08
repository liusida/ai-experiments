# %% Imports, model id & load
from __future__ import annotations

import math

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import stonesoup

# MODEL_ID = "google/gemma-4-E2B-it"
MODEL_ID = "allenai/Olmo-3-1025-7B"
# Safe filename stem (HF ids often contain "/"); used by ``stonesoup.show(..., basename=…)``.
MODEL_BASENAME = MODEL_ID.replace("/", "__").replace(":", "-")
USER_PROMPT = """
Wikipedia is a free online encyclopedia written and maintained by a community of volunteers, known as Wikipedians, through open collaboration and the wiki software MediaWiki. Founded by Jimmy Wales and Larry Sanger in 2001, Wikipedia has been hosted since 2003 by the Wikimedia Foundation, an American nonprofit organization funded mainly by donations from readers. Wikipedia is the largest and most-read reference work in history. According to Jimmy Wales, its mission is to make the sum of all human knowledge available to every person in the world.
"""

model, tokenizer = stonesoup.load_model(MODEL_ID)
model.eval()
device = next(model.parameters()).device
print("Loaded:", MODEL_ID, device, flush=True)
print("=" * 20)
print(USER_PROMPT)
print("=" * 20)

# %% Text prompt → forward → layer activations tensor
inner_tok = getattr(tokenizer, "tokenizer", tokenizer)
has_chat_template = getattr(inner_tok, "chat_template", None) is not None
if has_chat_template and hasattr(tokenizer, "apply_chat_template"):
    # String ``content`` — Qwen, Gemma2, Llama instruct, etc. (Gemma 4 multimodal wants
    # list-shaped ``content`` with ``type`` / ``text`` parts; use another script for that.)
    messages = [{"role": "user", "content": USER_PROMPT.strip()}]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(device)
else:
    inputs = inner_tok(
        USER_PROMPT.strip(),
        return_tensors="pt",
        return_attention_mask=True,
        add_special_tokens=True,
    ).to(device)

if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
    # GPT-2-style
    decoder_blocks = model.transformer.h
elif hasattr(model, "model"):
    inner = model.model
    if hasattr(inner, "language_model") and hasattr(inner.language_model, "layers"):
        # Gemma 4 / multimodal: backbone LM lives under ``language_model``.
        decoder_blocks = inner.language_model.layers
    elif hasattr(inner, "layers"):
        # Gemma 2, Llama, Mistral, … — ``*ForCausalLM.model.layers``.
        decoder_blocks = inner.layers
    else:
        raise TypeError(
            "Cannot find decoder layers on this model: need ``transformer.h``, "
            "``model.language_model.layers``, or ``model.layers``. Got "
            f"{type(model).__name__}"
        )
else:
    raise TypeError(
        "Cannot find decoder layers on this model: expected ``.transformer`` or ``.model``. Got "
        f"{type(model).__name__}"
    )

captured: list[torch.Tensor] = []


def save_pre_layer0_hidden_states(module, inputs: tuple) -> None:
    """Hidden states entering the first decoder block (pos-embed + dropout for GPT-2)."""
    captured.append(inputs[0].detach())


def save_layer_output(module, inp, out: torch.Tensor | tuple) -> None:
    hidden = out[0] if isinstance(out, tuple) else out
    captured.append(hidden.detach())


hooks = [decoder_blocks[0].register_forward_pre_hook(save_pre_layer0_hidden_states)]
hooks += [
    layer.register_forward_hook(save_layer_output) for layer in decoder_blocks
]
try:
    with torch.inference_mode():
        model(**inputs, use_cache=False)
finally:
    for h in hooks:
        h.remove()

activations = torch.stack(captured, dim=0)
num_blocks = len(decoder_blocks)
print(
    f"Pre-block-0 stream + after each of {num_blocks} decoder blocks (pre-final norm):",
    tuple(activations.shape),
    flush=True,
)

# %% For each layer, compute the pairwise cosine similarity
# activations: (1 + num_blocks, batch, seq_len, hidden) — [0] pre first block; [1:] after each block
layer_sims: list[torch.Tensor] = []
for li in range(activations.shape[0]):
    h = activations[li, 0].float()  # (seq_len, hidden)
    h = F.normalize(h, dim=-1)
    layer_sims.append(h @ h.T)
pairwise_cos = torch.stack(layer_sims, dim=0)
print("Per-layer token–token cosine similarity:", tuple(pairwise_cos.shape), flush=True)

# %% Plot cosine-sim heatmaps (pre first block + after each decoder block)
num_stages = pairwise_cos.shape[0]
grid_cols = max(math.ceil(math.sqrt(num_stages)), 1)
grid_rows = math.ceil(num_stages / grid_cols)
fig_w = min(24.0, 3.2 * grid_cols + 2.0)
fig_h = min(24.0, 3.2 * grid_rows + 2.0)
fig, axes = plt.subplots(
    grid_rows, grid_cols, figsize=(fig_w, fig_h), layout="constrained"
)
if grid_rows == 1 and grid_cols == 1:
    axes_list = [axes]
else:
    axes_list = axes.ravel().tolist()
pcm = None
for i, ax in enumerate(axes_list):
    if i < num_stages:
        pcm = ax.imshow(
            pairwise_cos[i].float().cpu().numpy(),
            cmap="Blues",
            vmin=0.0,
            vmax=1.0,
            aspect="auto",
        )
        title = "pre block 0" if i == 0 else f"post block {i - 1}"
        ax.set_title(f"{i}: {title}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.axis("off")
fig.suptitle(
    f"{MODEL_ID}\nToken–token cosine similarity (normalized hidden states)",
    fontsize=14,
)
if pcm is not None:
    fig.colorbar(
        pcm,
        ax=axes_list[:num_stages],
        shrink=0.45,
        location="right",
        label="cosine sim",
    )
stonesoup.show(fig, basename=f"{MODEL_BASENAME}_token_cos_all_stages")

# %% Embedding-stage cos-sim only (heatmap + token labels on axes)
prompt_token_ids = inputs["input_ids"][0].cpu().tolist()
token_labels = inner_tok.convert_ids_to_tokens(prompt_token_ids)
num_tokens = len(token_labels)
label_fontsize = max(4, min(9, 700 // max(num_tokens, 1)))
embed_cosine_matrix = pairwise_cos[0].float().cpu().numpy()
figsize_inches = min(28.0, max(10.0, 6.0 + 0.12 * num_tokens))
fig_embed, ax_embed = plt.subplots(figsize=(figsize_inches, figsize_inches))
im_embed = ax_embed.imshow(
    embed_cosine_matrix, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto"
)
ax_embed.set_xticks(range(num_tokens))
ax_embed.set_yticks(range(num_tokens))
ax_embed.set_xticklabels(
    token_labels, rotation=90, ha="center", fontsize=label_fontsize
)
ax_embed.set_yticklabels(token_labels, fontsize=label_fontsize)
ax_embed.set_xlabel("token (j)")
ax_embed.set_ylabel("token (i)")
ax_embed.set_title(
    f"{MODEL_ID}\nPre block 0: token–token cosine similarity",
)
fig_embed.colorbar(im_embed, ax=ax_embed, shrink=0.55, label="cosine sim")
fig_embed.tight_layout()
stonesoup.show(fig_embed, basename=f"{MODEL_BASENAME}_token_cos_pre_block0")

# %% Mean over tokens of pairwise stage cosines (cos per position, then mean over seq)
# For each (stage i, stage j): mean_t cos(h_i[t], h_j[t]) on normalized hidden states.
num_stages_cross = activations.shape[0]
hidden_n = F.normalize(activations[:, 0].float(), dim=-1)  # (num_stages, seq_len, hidden)
# ``ish,jsh->ijs`` → dot product at each token s, then mean over s.
mean_token_cos_stage_matrix = (
    torch.einsum("ish,jsh->ijs", hidden_n, hidden_n).mean(dim=-1).cpu().numpy()
)
stage_labels_cross = [str(i) for i in range(num_stages_cross)]
fig_cross, ax_cross = plt.subplots(figsize=(12, 12))
im_cross = ax_cross.imshow(
    mean_token_cos_stage_matrix,
    cmap="Blues",
    vmin=0.0,
    vmax=1.0,
    aspect="equal",
)
ax_cross.set_xticks(range(num_stages_cross))
ax_cross.set_yticks(range(num_stages_cross))
ax_cross.set_xticklabels(
    stage_labels_cross, rotation=90, ha="center", fontsize=7
)
ax_cross.set_yticklabels(stage_labels_cross, fontsize=7)
ax_cross.set_xlabel("layer index j")
ax_cross.set_ylabel("layer index i")
ax_cross.set_title(
    f"{MODEL_ID}\nMean over tokens: cos(h_i[t], h_j[t]) per position",
)
fig_cross.colorbar(im_cross, ax=ax_cross, shrink=0.55, label="cosine sim")
fig_cross.tight_layout()
stonesoup.show(fig_cross, basename=f"{MODEL_BASENAME}_mean_per_token_cos_stages")

# %% Pairwise cos sim between mean-pooled hidden states (one vector per stage)
num_stages = activations.shape[0]
mean_hidden_per_stage = activations[:, 0].float().mean(dim=1)  # (num_stages, hidden)
mean_hidden_normalized = F.normalize(mean_hidden_per_stage, dim=-1)
mean_stage_cosine_matrix = (mean_hidden_normalized @ mean_hidden_normalized.T).cpu().numpy()
stage_index_labels = [str(i) for i in range(num_stages)]
fig_mean, ax_mean = plt.subplots(figsize=(12, 12))
im_mean = ax_mean.imshow(
    mean_stage_cosine_matrix, cmap="Blues", vmin=0.0, vmax=1.0, aspect="equal"
)
ax_mean.set_xticks(range(num_stages))
ax_mean.set_yticks(range(num_stages))
ax_mean.set_xticklabels(
    stage_index_labels, rotation=90, ha="center", fontsize=7
)
ax_mean.set_yticklabels(stage_index_labels, fontsize=7)
ax_mean.set_xlabel("layer index")
ax_mean.set_ylabel("layer index")
ax_mean.set_title(
    f"{MODEL_ID}\nCosine similarity: mean hidden state per stage",
)
fig_mean.colorbar(im_mean, ax=ax_mean, shrink=0.55, label="cosine sim")
fig_mean.tight_layout()
stonesoup.show(fig_mean, basename=f"{MODEL_BASENAME}_mean_hidden_cosine")

