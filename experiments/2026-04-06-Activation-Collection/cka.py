# %% Imports, model id, load, capture activations
from __future__ import annotations

import matplotlib.pyplot as plt
import torch
import stonesoup

# Keep in sync with ``gemma4-activations.py`` when comparing runs.
MODEL_ID = "allenai/Olmo-3-1025-7B"
MODEL_BASENAME = MODEL_ID.replace("/", "__").replace(":", "-")
USER_PROMPT = """
Wikipedia is a free online encyclopedia written and maintained by a community of volunteers, known as Wikipedians, through open collaboration and the wiki software MediaWiki. Founded by Jimmy Wales and Larry Sanger in 2001, Wikipedia has been hosted since 2003 by the Wikimedia Foundation, an American nonprofit organization funded mainly by donations from readers. Wikipedia is the largest and most-read reference work in history. According to Jimmy Wales, its mission is to make the sum of all human knowledge available to every person in the world.
"""

model, tokenizer = stonesoup.load_model(MODEL_ID)
model.eval()
device = next(model.parameters()).device
print("Loaded:", MODEL_ID, device, flush=True)

inner_tok = getattr(tokenizer, "tokenizer", tokenizer)
has_chat_template = getattr(inner_tok, "chat_template", None) is not None
if has_chat_template and hasattr(tokenizer, "apply_chat_template"):
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
    decoder_blocks = model.transformer.h
elif hasattr(model, "model"):
    inner = model.model
    if hasattr(inner, "language_model") and hasattr(inner.language_model, "layers"):
        decoder_blocks = inner.language_model.layers
    elif hasattr(inner, "layers"):
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

_captured: list[torch.Tensor] = []


def _save_pre_layer0_hidden_states(module, inputs: tuple) -> None:
    _captured.append(inputs[0].detach())


def _save_layer_output(module, inp, out: torch.Tensor | tuple) -> None:
    hidden = out[0] if isinstance(out, tuple) else out
    _captured.append(hidden.detach())


_hooks = [decoder_blocks[0].register_forward_pre_hook(_save_pre_layer0_hidden_states)]
_hooks += [layer.register_forward_hook(_save_layer_output) for layer in decoder_blocks]
try:
    with torch.inference_mode():
        model(**inputs, use_cache=False)
finally:
    for h in _hooks:
        h.remove()

activations = torch.stack(_captured, dim=0)
num_blocks = len(decoder_blocks)
print(
    "Activations (pre block 0 + post each block):",
    tuple(activations.shape),
    f"({1 + num_blocks} stages)",
    flush=True,
)

# %% Linear CKA between stages (rows = token positions, features = hidden dim)
def linear_cka_stage_matrix(
    activations_btsh: torch.Tensor,
    batch_index: int = 0,
) -> torch.Tensor:
    """Pairwise linear CKA between stages; examples = sequence positions.

    ``activations_btsh``: (num_stages, batch, seq, hidden). Returns (S, S) on ``activations``' device.
    """
    S = activations_btsh.shape[0]
    centered: list[torch.Tensor] = []
    frob_xx: list[torch.Tensor] = []
    for s in range(S):
        stonesoup.check_abort()
        h = activations_btsh[s, batch_index].float()
        X = h - h.mean(dim=0, keepdim=True)
        centered.append(X)
        G = X.T @ X
        frob_xx.append((G * G).sum().sqrt())

    mat = torch.empty((S, S), device=activations_btsh.device, dtype=torch.float32)
    for i in range(S):
        stonesoup.check_abort()
        for j in range(S):
            if frob_xx[i] <= 0 or frob_xx[j] <= 0:
                mat[i, j] = float("nan")
                continue
            Sxy = centered[j].T @ centered[i]
            num = (Sxy * Sxy).sum()
            mat[i, j] = (num / (frob_xx[i] * frob_xx[j])).to(torch.float32)
    return mat


cka_matrix = linear_cka_stage_matrix(activations, batch_index=0)
print("Linear CKA stage matrix:", tuple(cka_matrix.shape), flush=True)

# %% Heatmap (same layout as mean stage cosine plots in ``gemma4-activations.py``)
num_stages = cka_matrix.shape[0]
stage_labels = [str(i) for i in range(num_stages)]
cka_np = cka_matrix.cpu().numpy()
fig, ax = plt.subplots(figsize=(12, 12))
im = ax.imshow(
    cka_np,
    cmap="Blues",
    vmin=0.0,
    vmax=1.0,
    aspect="equal",
)
ax.set_xticks(range(num_stages))
ax.set_yticks(range(num_stages))
ax.set_xticklabels(stage_labels, rotation=90, ha="center", fontsize=7)
ax.set_yticklabels(stage_labels, fontsize=7)
ax.set_xlabel("layer index j")
ax.set_ylabel("layer index i")
ax.set_title(
    f"{MODEL_ID}\nLinear CKA (column-centered): rows = token positions, cols = hidden features",
)
fig.colorbar(im, ax=ax, shrink=0.55, label="linear CKA")
fig.tight_layout()
stonesoup.show(fig, basename=f"{MODEL_BASENAME}_linear_cka_stages")
