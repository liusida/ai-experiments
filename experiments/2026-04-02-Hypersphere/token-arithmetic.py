# %% Load model & embedding table shape
import stonesoup

# Examples: ``Qwen/Qwen3.5-0.8B``, ``gpt2``, ``openai-community/gpt2-medium``.
MODEL_ID = "Qwen/Qwen3.5-0.8B"

model, processor = stonesoup.load_model(MODEL_ID)
# Text-only causal LMs: second value is the tokenizer (no nested ``.tokenizer``).
try:
    tokenizer = processor.tokenizer
except AttributeError:
    tokenizer = processor
emb = model.get_input_embeddings().weight
print(emb.shape)

# %% Token pairs: raw vs layer-0 RMSNorm → cos(Δ) & norms
import torch
import torch.nn.functional as F


def _causal_lm_backbone(m: torch.nn.Module) -> torch.nn.Module:
    """LLaMA/Qwen-style ``.model``, GPT-2 OPT-style ``.transformer``, or bare trunk."""
    if getattr(m, "model", None) is not None:
        return m.model
    if getattr(m, "transformer", None) is not None:
        return m.transformer
    return m


def _decoder_stack(backbone: torch.nn.Module) -> torch.nn.ModuleList | torch.nn.Sequential:
    layers = getattr(backbone, "layers", None)
    if layers is None and hasattr(backbone, "language_model"):
        lm = backbone.language_model
        layers = getattr(lm, "layers", None)
    if layers is None:
        layers = getattr(backbone, "h", None)
    if layers is None:
        raise AttributeError(
            f"cannot find decoder blocks on {type(backbone).__name__} "
            "(expected .layers, .language_model.layers, or .h for GPT-2)"
        )
    return layers


def _layer0_pre_attn_norm(m: torch.nn.Module) -> torch.nn.Module:
    """First transformer block's input norm (RMSNorm / LayerNorm before attention)."""
    backbone = _causal_lm_backbone(m)
    stack = _decoder_stack(backbone)
    layer0 = stack[0]
    for attr in ("input_layernorm", "ln_1", "self_attn_layer_norm"):
        norm = getattr(layer0, attr, None)
        if norm is not None:
            return norm
    raise AttributeError(
        f"no pre-attention norm on first layer of type {type(layer0).__name__} "
        f"(tried input_layernorm, ln_1, self_attn_layer_norm)"
    )


def _pairwise_cos_diffs(diffs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """diffs: (P, 1, hidden) → (cos matrix P×P on CPU float32, ‖Δ‖ per row on CPU)."""
    d = diffs.squeeze(1).float()
    dn = F.normalize(d, dim=1, eps=1e-12)
    return (dn @ dn.T).cpu(), d.norm(dim=-1).cpu()


# Unicode OPEN BOX (␣): visible “space” marker for axis / legend text.
_LEADING_SPACE_ICON = "\u2423"


def _vis_token(s: str) -> str:
    stripped = s.lstrip(" ")
    n = len(s) - len(stripped)
    return f"{_LEADING_SPACE_ICON * n}{stripped}" if n else stripped


def _pair_label(a: str, b: str) -> str:
    return f"{_vis_token(a)}→{_vis_token(b)}"


# Each tuple is (token_a, token_b); Δ = rep(b) − rep(a).
TOKEN_PAIRS = (
    (" king", " queen"),
    (" actor", " actress"),
    (" uncle", " aunt"),
    (" father", " mother"),
    (" man", " woman"),
    (" male", " female"),
)

_flat: list[str] = []
for a, b in TOKEN_PAIRS:
    _flat.extend([a, b])

_ids: list[int] = []
for s in _flat:
    piece = tokenizer(s, add_special_tokens=False)["input_ids"]
    if len(piece) != 1:
        raise ValueError(f"expected 1 token for {s!r}, got {piece}")
    _ids.append(piece[0])

_device = next(model.parameters()).device
ids_1t = torch.tensor(_ids, device=_device, dtype=torch.long).view(-1, 1)

n_pairs = len(TOKEN_PAIRS)
PAIR_LABELS = [_pair_label(a, b) for a, b in TOKEN_PAIRS]
FIRST_TOKEN_LABELS = [_vis_token(a) for a, _b in TOKEN_PAIRS]

with torch.inference_mode():
    h0 = model.get_input_embeddings()(ids_1t)
    h_ln = _layer0_pre_attn_norm(model)(h0)

pair_diff_raw = torch.stack([h0[2 * k + 1] - h0[2 * k] for k in range(n_pairs)], dim=0)
pair_diff_ln = torch.stack([h_ln[2 * k + 1] - h_ln[2 * k] for k in range(n_pairs)], dim=0)

cos_raw, norm_delta_raw = _pairwise_cos_diffs(pair_diff_raw)
cos_ln, norm_delta_ln = _pairwise_cos_diffs(pair_diff_ln)

# First token of each pair (index ``a``): cos between E(a_i) and E(a_j), same in LN0 space.
first_tok_raw = torch.stack([h0[2 * k] for k in range(n_pairs)], dim=0)
first_tok_ln = torch.stack([h_ln[2 * k] for k in range(n_pairs)], dim=0)
cos_first_raw, _ = _pairwise_cos_diffs(first_tok_raw)
cos_first_ln, _ = _pairwise_cos_diffs(first_tok_ln)

emb_norm_per_token = h0.squeeze(1).float().norm(dim=-1).cpu()
ln_norm_per_token = h_ln.squeeze(1).float().norm(dim=-1).cpu()

print("pairs:", TOKEN_PAIRS)
print("token ids:", _ids)
print("cos(Δ) raw:\n", cos_raw.numpy())
print("cos(Δ) after ln:\n", cos_ln.numpy())
print("cos(first token a) raw:\n", cos_first_raw.numpy())
print("cos(first token a) after ln:\n", cos_first_ln.numpy())
print("‖Δ‖ raw:", norm_delta_raw.numpy())
print("‖Δ‖ after ln:", norm_delta_ln.numpy())
_emb_note = " | ".join(
    f"{PAIR_LABELS[k]}: ‖emb(a)‖={emb_norm_per_token[2 * k]:.2f}, ‖emb(b)‖={emb_norm_per_token[2 * k + 1]:.2f}"
    for k in range(n_pairs)
)
print("raw ‖emb‖ per token (a then b per pair):", _emb_note)

# %% Heatmaps: cos(Δ) raw vs layer[0] pre-attention norm
import matplotlib.pyplot as plt
import stonesoup

P = n_pairs
fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.4))


def _heatmap(
    ax,
    mat: torch.Tensor,
    title: str,
    *,
    labels: list[str] = PAIR_LABELS,
    xlabel: str = "token-pair Δ",
    ylabel: str = "token-pair Δ",
) -> None:
    m = mat.numpy()
    im = ax.imshow(m, vmin=0.0, vmax=1.0, cmap="Blues", aspect="equal")
    ax.set_xticks(range(P))
    ax.set_yticks(range(P))
    ax.set_xticklabels(labels, rotation=32, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    for i in range(P):
        for j in range(P):
            color = "white" if m[i, j] >= 0.65 else "0.15"
            ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=9, color=color)
    ax.set_title(title, fontsize=11.5)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


# Hidden for now; useful if we want to bring back a stats panel later.
# def _stats_lines_raw() -> str:
#     return "\n".join(
#         f"{PAIR_LABELS[k]}: ||E(a)||={emb_norm_per_token[2 * k]:.2f}, "
#         f"||E(b)||={emb_norm_per_token[2 * k + 1]:.2f}, ||Δ_raw||={norm_delta_raw[k]:.2f}"
#         for k in range(P)
#     )
#
#
# def _stats_lines_ln() -> str:
#     return "\n".join(
#         f"{PAIR_LABELS[k]}: ||LN0(E(a))||={ln_norm_per_token[2 * k]:.2f}, "
#         f"||LN0(E(b))||={ln_norm_per_token[2 * k + 1]:.2f}, ||Δ_LN||={norm_delta_ln[k]:.2f}"
#         for k in range(P)
#     )


_heatmap(
    axes[0, 0],
    cos_raw,
    "A. Difference vectors in raw embedding space",
)
_heatmap(
    axes[0, 1],
    cos_ln,
    "B. Difference vectors after first-layer norm",
)
_fst_lbl = "pair row/column = first token a"
_heatmap(
    axes[1, 0],
    cos_first_raw,
    "C. First-token similarity in raw embedding space",
    labels=FIRST_TOKEN_LABELS,
    xlabel=_fst_lbl,
    ylabel=_fst_lbl,
)
_heatmap(
    axes[1, 1],
    cos_first_ln,
    "D. First-token similarity after first-layer norm",
    labels=FIRST_TOKEN_LABELS,
    xlabel=_fst_lbl,
    ylabel=_fst_lbl,
)

fig.suptitle(
    f"{MODEL_ID} -- Revisiting the classic analogy: [king] - [queen] ≈ [man] - [woman]\n"
    "Δ heatmaps (top) and first-token (a) similarity (bottom); raw vs layer-0 pre-attention norm",
    fontsize=13,
    y=0.985,
)
plt.tight_layout(rect=(0.03, 0.04, 0.98, 0.93))

stonesoup.show()
plt.close("all")
