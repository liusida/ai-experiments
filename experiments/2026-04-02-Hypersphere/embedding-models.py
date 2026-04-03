# %% Imports & pretrained ids (gensim-data / downloader)
import stonesoup
import numpy as np

# Pretrained static embeddings via ``gensim.downloader.load`` (first run downloads into ``~/gensim-data``).
#
# - ``word2vec-google-news-300`` — standard public Word2Vec checkpoint (~1.6 GiB). Public releases rarely
#   document CBOW vs skip-gram per shard; this is the usual “classic word2vec” baseline online.
# - ``glove-wiki-gigaword-100`` — smaller GloVe baseline for a second column (not Word2Vec; still a classic
#   count-based embedding). Swap for another ``api.info()`` name if you prefer.
#
# Quick/smaller ideas: ``glove-twitter-25``, ``glove-wiki-gigaword-50`` (set BOTH to small models only if
# you accept two different algorithms without the big Word2Vec download).
PRETRAINED_WORD2VEC = "word2vec-google-news-300"
PRETRAINED_SECOND = "glove-wiki-gigaword-100"

# Human-readable labels for the figure (adjust if you change ids above).
_LABEL_W2V = f"Word2Vec (pretrained): {PRETRAINED_WORD2VEC}"
_LABEL_2ND = f"Second baseline: {PRETRAINED_SECOND}"

# %% Load KeyedVectors from the network (cached after first success)
import gensim.downloader as api

print("loading", PRETRAINED_WORD2VEC, "...")
wv_w2v = api.load(PRETRAINED_WORD2VEC)
print("loading", PRETRAINED_SECOND, "...")
wv_2 = api.load(PRETRAINED_SECOND)
print("w2v:", type(wv_w2v).__name__, "len", len(wv_w2v), "| second:", type(wv_2).__name__, "len", len(wv_2))

# %% Word pairs → cosine matrices (two pretrained spaces)
_LEADING_SPACE_ICON = "\u2423"


def _vis_token_display(w: str) -> str:
    if w.startswith(" "):
        return _LEADING_SPACE_ICON + w.lstrip(" ")
    return w


def _pair_label(a: str, b: str) -> str:
    return f"{_vis_token_display(a)}→{_vis_token_display(b)}"


def _resolve_in_wv(wv, w: str) -> str:
    """Pick a vocab key for *w* (Google News Word2Vec is often capitalized)."""
    if w in wv:
        return w
    for c in (w.lower(), w.title(), w.capitalize(), w.upper()):
        if c in wv:
            return c
    raise KeyError(w)


def _require_words(wv, words: list[str]) -> None:
    missing: list[str] = []
    for w in words:
        try:
            _resolve_in_wv(wv, w)
        except KeyError:
            missing.append(w)
    if missing:
        raise KeyError(
            "OOV after case variants — try other surface forms or different pretrained id:\n"
            f"  missing: {missing}"
        )


def _flat_pair_words(pairs: tuple[tuple[str, str], ...]) -> list[str]:
    out: list[str] = []
    for a, b in pairs:
        out.extend([a, b])
    return out


def _vectors_for_words(wv, words: list[str]) -> np.ndarray:
    _require_words(wv, words)
    keys = [_resolve_in_wv(wv, w) for w in words]
    return np.stack([wv[k] for k in keys], axis=0).astype(np.float64)


def _pairwise_cos(vecs: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(vecs, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    u = vecs / n
    return u @ u.T


WORD_PAIRS = (
    ("king", "queen"),
    ("actor", "actress"),
    ("uncle", "aunt"),
    ("father", "mother"),
    ("man", "woman"),
    ("male", "female"),
    ("apple", "orange"),
)

_flat = _flat_pair_words(WORD_PAIRS)
n_pairs = len(WORD_PAIRS)
PAIR_LABELS = [_pair_label(a, b) for a, b in WORD_PAIRS]
FIRST_TOKEN_LABELS = [_vis_token_display(a) for a, _b in WORD_PAIRS]

h_w2v = _vectors_for_words(wv_w2v, _flat)
h_2 = _vectors_for_words(wv_2, _flat)

D_w2v = np.stack([h_w2v[2 * k + 1] - h_w2v[2 * k] for k in range(n_pairs)], axis=0)
D_2 = np.stack([h_2[2 * k + 1] - h_2[2 * k] for k in range(n_pairs)], axis=0)

cos_d_w2v = _pairwise_cos(D_w2v)
cos_d_2 = _pairwise_cos(D_2)

A_w2v = np.stack([h_w2v[2 * k] for k in range(n_pairs)], axis=0)
A_2 = np.stack([h_2[2 * k] for k in range(n_pairs)], axis=0)

cos_a_w2v = _pairwise_cos(A_w2v)
cos_a_2 = _pairwise_cos(A_2)

print("pairs:", WORD_PAIRS)
print("cos(Δ) Word2Vec:\n", cos_d_w2v)
print("cos(Δ) second model:\n", cos_d_2)

# %% Heatmaps: pretrained Word2Vec vs second embedding
import matplotlib.pyplot as plt

P = n_pairs
fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.4))


def _heatmap(
    ax,
    mat: np.ndarray,
    title: str,
    *,
    labels: list[str],
    xlabel: str,
    ylabel: str,
) -> None:
    im = ax.imshow(mat, vmin=0.0, vmax=1.0, cmap="Blues", aspect="equal")
    ax.set_xticks(range(P))
    ax.set_yticks(range(P))
    ax.set_xticklabels(labels, rotation=32, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    for i in range(P):
        for j in range(P):
            v = float(mat[i, j])
            color = "white" if v >= 0.65 else "0.15"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9, color=color)
    ax.set_title(title, fontsize=11.5)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


_heatmap(
    axes[0, 0],
    cos_d_w2v,
    f"A. Difference vectors — {_LABEL_W2V}",
    labels=PAIR_LABELS,
    xlabel="token-pair Δ",
    ylabel="token-pair Δ",
)
_heatmap(
    axes[0, 1],
    cos_d_2,
    f"B. Difference vectors — {_LABEL_2ND}",
    labels=PAIR_LABELS,
    xlabel="token-pair Δ",
    ylabel="token-pair Δ",
)
_fst = "first word a in each pair (row/column)"
_heatmap(
    axes[1, 0],
    cos_a_w2v,
    f"C. First-word similarity — {_LABEL_W2V}",
    labels=FIRST_TOKEN_LABELS,
    xlabel=_fst,
    ylabel=_fst,
)
_heatmap(
    axes[1, 1],
    cos_a_2,
    f"D. First-word similarity — {_LABEL_2ND}",
    labels=FIRST_TOKEN_LABELS,
    xlabel=_fst,
    ylabel=_fst,
)

fig.suptitle(
    "Pretrained classic embeddings (gensim downloader)\n"
    "Δ heatmaps (top) and first-word similarity (bottom); two different vector spaces",
    fontsize=13,
    y=0.985,
)
plt.tight_layout(rect=(0.03, 0.04, 0.98, 0.93))

stonesoup.show()
plt.close("all")
