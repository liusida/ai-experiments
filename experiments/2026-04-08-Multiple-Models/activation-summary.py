# %% Config
from __future__ import annotations

from typing import Any

import gc
import torch
import stonesoup

MODEL_REGISTRY: list[str] = [
    "openai-community/gpt2-xl",  # weak
    "EleutherAI/pythia-1.4b",
    "meta-llama/llama-3.2-3B",
    "google/gemma-2-2b",
    "Qwen/Qwen2.5-3B",
    "mistralai/Ministral-3-3B-Base-2512",
    # "meta-llama/Llama-2-7b-hf",
    # "Qwen/Qwen3-8B-Base",
    # "tiiuae/falcon-7b",
    # "allenai/Olmo-3-1025-7B",
    # "google/gemma-4-E2B",
]

PROMPT = "The capital of France is Paris. The capital of Germany is"
STORE_AS_FLOAT16 = False

# ``repo_id`` -> ``{repo_id, prompt, hidden_states, stage_names, shape}`` (CPU)
ACTIVATIONS_BY_REPO: dict[str, dict[str, Any]] = {}

# %% Load each model → capture hidden states → unload
from stonesoup.backend.hf_models import unload_models_from_kernel
from stonesoup.backend.kernel import active_kernel


def encode_prompt(proc: Any, prompt: str, device: torch.device) -> dict[str, Any]:
    """Base LMs only: raw text → ``input_ids`` (no chat template)."""
    tok = getattr(proc, "tokenizer", None) or proc
    enc = tok(prompt, return_tensors="pt", return_attention_mask=True, add_special_tokens=True)
    return {k: v.to(device) for k, v in enc.items()}


def decoder_blocks(model: Any) -> list[Any]:
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return list(model.gpt_neox.layers)  # GPT-NeoX (e.g. Pythia)
    inner = getattr(model, "model", None)
    if inner is None:
        raise TypeError(f"No .transformer / .model on {type(model).__name__}")
    if hasattr(inner, "language_model") and hasattr(inner.language_model, "layers"):
        return list(inner.language_model.layers)
    return list(inner.layers)


def capture_stack(model: Any, inputs: dict[str, Any]) -> tuple[torch.Tensor, list[str]]:
    blocks = decoder_blocks(model)
    out: list[torch.Tensor] = []

    def grab_embed(_mod, _in, x: torch.Tensor) -> None:
        out.append(x.detach())

    def grab_layer(_mod, _in, x: Any) -> None:
        h = x[0] if isinstance(x, tuple) else x
        out.append(h.detach())

    emb = model.get_input_embeddings()
    hooks = [emb.register_forward_hook(grab_embed)]
    hooks += [L.register_forward_hook(grab_layer) for L in blocks]
    try:
        with torch.inference_mode():
            model(**inputs, use_cache=False)
    finally:
        for h in hooks:
            h.remove()

    names = ["embedding"] + [f"layer_{i}" for i in range(len(blocks))]
    assert len(out) == len(names)
    return torch.stack(out, dim=0), names


def unload(repo_id: str) -> None:
    k = active_kernel.get()
    if k is None:
        return
    names = [r["name"] for r in stonesoup.list_loaded_models() if r["repo_id"] == repo_id]
    if names:
        unload_models_from_kernel(k, names=names)


ACTIVATIONS_BY_REPO.clear()
for repo_id in MODEL_REGISTRY:
    stonesoup.check_abort()
    try:
        print(f"Loading {repo_id}", flush=True)
        model, proc = stonesoup.load_model(repo_id)
        print(f"Processing {repo_id}", flush=True)
        model.eval()
        dev = next(model.parameters()).device
        inputs = encode_prompt(proc, PROMPT, dev)
        stack, stage_names = capture_stack(model, inputs)
        hs = stack.detach().cpu()
        hs = hs.half() if STORE_AS_FLOAT16 else hs.float()
        ACTIVATIONS_BY_REPO[repo_id] = {
            "repo_id": repo_id,
            "prompt": PROMPT,
            "hidden_states": hs,
            "stage_names": stage_names,
            "shape": tuple(hs.shape),
        }
        print(f"OK  {repo_id} {tuple(hs.shape)}", flush=True)
    except Exception as exc:
        print(f"ERR {repo_id}: {exc!r}", flush=True)
    finally:
        # unload(repo_id)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

print(f"ACTIVATIONS_BY_REPO: {len(ACTIVATIONS_BY_REPO)} keys {list(ACTIVATIONS_BY_REPO)}", flush=True)

# %% Plot: one subplot per model (y = layer, x = dimension, color = value)
# Each row is the last sequence position (batch 0), L2-normalized along the hidden dimension.
import matplotlib
import sys

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

if not ACTIVATIONS_BY_REPO:
    print("Nothing to plot — run the capture cell first.", flush=True)
else:
    n = len(ACTIVATIONS_BY_REPO)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, max(3.5, 2.8 * nrows)), squeeze=False)
    axes_flat = axes.ravel()
    for idx, (repo_id, payload) in enumerate(ACTIVATIONS_BY_REPO.items()):
        ax = axes_flat[idx]
        hs = payload["hidden_states"].float()
        # (stages, batch, seq, hidden) → last token, batch 0
        vecs = hs[:, 0, -1, :]
        vecs = F.normalize(vecs, p=2, dim=1, eps=1e-12)
        mat = vecs.numpy()
        # print(mat, file=sys.stderr)
        mx = 0.1
        im = ax.imshow(
            mat,
            aspect="auto",
            origin="upper",
            cmap="coolwarm",
            vmin=-mx,
            vmax=mx,
        )
        ax.set_xlabel("hidden dimension index")
        ax.set_ylabel("layer (index)")
        ax.set_title(repo_id, fontsize=10)
        # n_st = mat.shape[0]
        # if n_st <= 48:
        #     ax.set_yticks(np.arange(n_st))
        #     ax.set_yticklabels(payload["stage_names"], fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.025, label="component (L2 row = 1)")
    for j in range(len(ACTIVATIONS_BY_REPO), len(axes_flat)):
        axes_flat[j].set_visible(False)
    plt.tight_layout()
    stonesoup.show(fig, basename="activation_layer_by_dim")

# %% Plot: pairwise cosine similarity between layers (last token, same vectors as above)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

if not ACTIVATIONS_BY_REPO:
    print("Nothing to plot — run the capture cell first.", flush=True)
else:
    n = len(ACTIVATIONS_BY_REPO)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig2, axes2 = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, max(3.5, 2.8 * nrows)), squeeze=False)
    axes_flat2 = axes2.ravel()
    for idx, (repo_id, payload) in enumerate(ACTIVATIONS_BY_REPO.items()):
        ax = axes_flat2[idx]
        hs = payload["hidden_states"].float()
        vecs = hs[:, 0, -1, :]
        vecs = F.normalize(vecs, p=2, dim=1, eps=1e-12)
        cos = (vecs @ vecs.T).cpu().numpy()
        im = ax.imshow(
            cos,
            aspect="equal",
            origin="upper",
            cmap="Blues",
            vmin=-0.0,
            vmax=1.0,
        )
        ax.set_xlabel("layer index")
        ax.set_ylabel("layer index")
        ax.set_title(repo_id, fontsize=10)
        fig2.colorbar(im, ax=ax, fraction=0.025, label="cosine similarity")
    for j in range(len(ACTIVATIONS_BY_REPO), len(axes_flat2)):
        axes_flat2[j].set_visible(False)
    plt.tight_layout()
    stonesoup.show(fig2, basename="activation_layer_cos_sim")

# %% Print: top-K dimensions by |value| per layer (L2-normalized, last token) + Σv_i² over those K
import html

import torch
import torch.nn.functional as F

TOP_DIMS = 10

if not ACTIVATIONS_BY_REPO:
    print("Nothing to print — run the capture cell first.", flush=True)
else:
    _lines: list[str] = [
        "# stonesoup:render=md",
        "",
        f"## Top {TOP_DIMS} dimensions per layer (normalized vector, last token)",
        "",
        f"For each stage: largest **|component|** dimensions (signed values). "
        f"**Σv_i²** is over those {TOP_DIMS} coordinates only; for a unit vector, **Σ_all v_i² = 1**, so this shows how much of the norm lives in those dims.",
        "",
    ]
    for repo_id, payload in ACTIVATIONS_BY_REPO.items():
        hs = payload["hidden_states"].float()
        vecs = F.normalize(hs[:, 0, -1, :], p=2, dim=1, eps=1e-12)
        names = payload.get("stage_names") or [f"stage_{i}" for i in range(vecs.shape[0])]
        _lines.append(f"### `{html.escape(repo_id)}`")
        _lines.append("")
        for li, name in enumerate(names):
            v = vecs[li]
            k = min(TOP_DIMS, int(v.numel()))
            _, idx = torch.topk(v.abs(), k=k)
            signed = v[idx]
            sum_sq = float((signed * signed).sum().item())
            parts = [f"`dim {int(i)}`={float(s):+.2f}" for i, s in zip(idx.tolist(), signed.tolist())]
            _lines.append(
                f"- **{html.escape(str(name))}:** Σv_i²=`{sum_sq:.4f}` — " + ", ".join(parts)
            )
        _lines.append("")
    print("\n".join(_lines), flush=True)

