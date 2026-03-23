# %% Imports & paths
"""Summarize Qwen3.5-0.8B **input embedding** weights from disk cache plus tokenizer-aligned **token** metadata.

Loads ``data/embedding-layers/Qwen__Qwen3.5-0.8B.pt`` (written by ``demo.py``). Requires the tokenizer from
Hugging Face for string-side statistics. From repo root::

    uv run python 2026-03-23-Embedding/embedding_qwen35_statistics.py

In Stonesoup: **Watch** this file and run cells in order (kernel keeps globals).
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no Tk main loop in Stonesoup / many kernels — avoids PIL Image __del__ RuntimeError
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
PLOTS_DIR = Path(__file__).resolve().parent / "plots"
LAYER_DIR = REPO_ROOT / "data" / "embedding-layers"
TEXT_CACHE_DIR = REPO_ROOT / "data" / "text"
MODEL_NAME = "Qwen/Qwen3.5-0.8B"
CACHE_PATH = LAYER_DIR / f"{MODEL_NAME.replace('/', '__')}.pt"


def _np_describe_1d(x: np.ndarray, name: str) -> None:
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size == 0:
        print(f"{name}: (empty)")
        return
    qs = np.percentile(x, [1, 5, 25, 50, 75, 95, 99])
    print(f"{name}  n={x.size}")
    print(
        f"  mean={x.mean():.6g}  std={x.std():.6g}  min={x.min():.6g}  max={x.max():.6g}"
    )
    print(
        f"  p01={qs[0]:.6g}  p05={qs[1]:.6g}  p25={qs[2]:.6g}  "
        f"p50={qs[3]:.6g}  p75={qs[4]:.6g}  p95={qs[5]:.6g}  p99={qs[6]:.6g}"
    )


# %% Load embedding checkpoint
if not CACHE_PATH.is_file():
    raise FileNotFoundError(
        f"Missing cache {CACHE_PATH}. Run demo.py once with this model (or save the embedding layer) "
        f"so {LAYER_DIR.name} contains the .pt file."
    )

payload = torch.load(CACHE_PATH, map_location="cpu", weights_only=False)
if not isinstance(payload, dict):
    raise TypeError("expected dict checkpoint")
w = payload.get("weight")
if not torch.is_tensor(w) or w.dim() != 2:
    raise TypeError("expected payload['weight'] to be a 2D tensor")

model_name = payload.get("model_name")
if isinstance(model_name, str) and model_name.strip():
    print("checkpoint model_name:", model_name.strip())
else:
    print("checkpoint model_name: (missing, using file stem)")
print("cache path:", CACHE_PATH)
print("weight shape (vocab, hidden):", tuple(w.shape))
print("dtype (stored):", w.dtype)
vocab_size, hidden_size = int(w.shape[0]), int(w.shape[1])
for key in ("vocab_size", "hidden_size"):
    if key in payload:
        print(f"payload[{key!r}]:", payload[key])

# %% Vector statistics (all weights, rows, columns)
w32 = w.detach().float().cpu()
flat = w32.numpy().ravel()
_np_describe_1d(flat, "All embedding elements")

row_l2 = torch.linalg.vector_norm(w32, dim=1).numpy()
_np_describe_1d(row_l2, "Per-token row L2 norm")

col_mean = w32.mean(dim=0).numpy()
col_std = w32.std(dim=0, unbiased=False).numpy()
_np_describe_1d(col_mean, "Per-dim mean across vocab (distribution over hidden indices)")
_np_describe_1d(col_std, "Per-dim std across vocab (distribution over hidden indices)")

# Sampled pairwise cosine similarity between distinct token rows (unnormalized dot / (norm*norm))
n = vocab_size
sample_pairs = 50_000
if n >= 2:
    g = torch.Generator(device=w32.device)
    g.manual_seed(0)
    i = torch.randint(0, n, (sample_pairs,), generator=g)
    j = torch.randint(0, n, (sample_pairs,), generator=g)
    same = i == j
    if same.any():
        j = torch.where(same, (j + 1) % n, j)
    a = w32[i]
    b = w32[j]
    cos = (a * b).sum(dim=1) / (torch.linalg.vector_norm(a, dim=1) * torch.linalg.vector_norm(b, dim=1) + 1e-12)
    _np_describe_1d(cos.numpy(), f"Cosine similarity ({sample_pairs} random distinct row pairs)")
else:
    print("Cosine sample: vocab too small")

# %% Tokenizer load & vocabulary alignment
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tok_len = len(tokenizer)
print("AutoTokenizer len:", tok_len)
print("Embedding rows:", vocab_size)
print("Tokenizer matches embedding rows:", tok_len == vocab_size)

special = getattr(tokenizer, "special_tokens_map", None)
if isinstance(special, dict) and special:
    print("special_tokens_map (keys):", sorted(special.keys()))
for attr in ("pad_token_id", "eos_token_id", "bos_token_id", "unk_token_id"):
    v = getattr(tokenizer, attr, None)
    if v is not None:
        print(f"{attr}:", v)

# %% Inspect tokens: king/queen/man/woman + fork/spoon (ids, rows, pairwise cosines, analogy)
WORDS = [" king", " queen", " man", " woman", " fork", " spoon"]
"""Leading space matches common BPE word-boundary pieces (compare without space if you need other splits)."""

def _word_unit_vector(ids: list[int]) -> torch.Tensor:
    rows = w32[ids]
    if rows.shape[0] == 1:
        v = rows[0]
    else:
        v = rows.mean(dim=0)
    return v / (torch.linalg.vector_norm(v) + 1e-12)

vecs: list[torch.Tensor] = []
labels: list[str] = []

for s in WORDS:
    ids = tokenizer.encode(s, add_special_tokens=False)
    toks = tokenizer.convert_ids_to_tokens(ids)
    print(f"\n{s!r}  ->  ids={ids}  tokens={toks}")
    if not ids:
        print("  (no token ids; skipped for matrix / analogy)")
        continue
    for tid in ids:
        if not (0 <= tid < vocab_size):
            print(f"  WARNING: id {tid} out of embedding range [0, {vocab_size})")
    if ids:
        norms = torch.linalg.vector_norm(w32[ids], dim=1)
        print(f"  subword row L2 norms: {[float(x) for x in norms]}")
    v = _word_unit_vector(ids)
    vecs.append(v)
    labels.append(s.strip())

if len(vecs) == len(WORDS):
    V = torch.stack(vecs, dim=0)
    sim = (V @ V.T).numpy()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 6.2), layout="constrained")
    im = ax.imshow(sim, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="equal")
    cbar = fig.colorbar(im, ax=ax, shrink=0.82, label="cosine similarity")
    cbar.ax.tick_params(labelsize=9)
    ax.set_title(
        "Cosine similarity (L2-normalized mean of subword rows per string)",
        fontsize=11,
    )
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = sim[i, j]
            ax.text(
                j,
                i,
                f"{v:.2f}",
                ha="center",
                va="center",
                fontsize=8.5,
                color="0.15" if abs(v) < 0.55 else "0.95",
            )
    out_path = PLOTS_DIR / "qwen35_embedding_word_cosine_similarity.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"\nCosine similarity heatmap saved: {out_path}")

    # First four entries in WORDS: king, queen, man, woman
    vk, vq, vm, vw = vecs[0], vecs[1], vecs[2], vecs[3]
    analogy_vec = vk - vm + vw
    analogy_vec = analogy_vec / (torch.linalg.vector_norm(analogy_vec) + 1e-12)
    print("\nClassic vector analogy (king/queen/man/woman only; fork/spoon in matrix above):")
    print(f"  cos(king - man + woman,  queen) = {(analogy_vec * vq).sum().item():.4f}")
    print(f"  cos(king - man + woman,   king) = {(analogy_vec * vk).sum().item():.4f}  (reference)")
    print(f"  cos(king - man + woman,    man) = {(analogy_vec * vm).sum().item():.4f}")
    print(f"  cos(king - man + woman,  woman) = {(analogy_vec * vw).sum().item():.4f}")

# %% Token-string statistics (decode every id; can take ~1–2 min on large vocabs)
if tok_len != vocab_size:
    print(
        "WARNING: tokenizer len != embedding rows; decodes use id in 0..vocab_size-1 "
        "(HF may behave oddly for out-of-range ids)."
    )

decode_lengths: list[int] = []
empty_decodes = 0
utf8_lengths: list[int] = []
decode_failures = 0

for tid in tqdm(range(vocab_size), desc="decode ids", unit="tok"):
    try:
        s = tokenizer.decode([tid], skip_special_tokens=False)
    except Exception as e:
        decode_failures += 1
        if decode_failures <= 3:
            print(f"decode failed id={tid}: {e!s}")
        s = ""
        decode_lengths.append(0)
        utf8_lengths.append(0)
        continue
    decode_lengths.append(len(s))
    if s == "":
        empty_decodes += 1
    utf8_lengths.append(len(s.encode("utf-8")))

_np_describe_1d(np.array(decode_lengths, dtype=np.float64), "Decoded string length (chars)")
_np_describe_1d(np.array(utf8_lengths, dtype=np.float64), "Decoded string UTF-8 byte length")
print("Empty decode count:", empty_decodes)
print("Decode exception count:", decode_failures)

# %% Optional: high / low norm ids vs decode (small table)
order = np.argsort(row_l2)
k = min(5, vocab_size)
low_ids = order[:k].tolist()
high_ids = order[-k:][::-1].tolist()
print("Lowest row-L2 token ids:", low_ids)
for tid in low_ids:
    print(f"  id {tid}: {tokenizer.decode([tid], skip_special_tokens=False)!r}  norm={row_l2[tid]:.6g}")
print("Highest row-L2 token ids:", high_ids)
for tid in high_ids:
    print(f"  id {tid}: {tokenizer.decode([tid], skip_special_tokens=False)!r}  norm={row_l2[tid]:.6g}")

# %% Alice in Wonderland: load cached text, tokenize, count unique token ids
# Expects UTF-8 plain text at ``data/text/gutenberg_11-0_alice_in_wonderland.txt`` (repo root).
# Prefetch: ``mkdir -p data/text && curl -fL -o data/text/gutenberg_11-0_alice_in_wonderland.txt "https://raw.githubusercontent.com/GITenberg/Alice-s-Adventures-in-Wonderland_11/master/11-0.txt"``
# or ``uv run python 2026-03-23-Embedding/download_alice_cache.py``.
ALICE_RAW_CACHE_PATH = TEXT_CACHE_DIR / "gutenberg_11-0_alice_in_wonderland.txt"


def _gutenberg_story_body(raw: str) -> str:
    """Strip Project Gutenberg header/footer; keep text between START/END markers."""
    lo = raw.lower()
    start_key = "*** start of"
    end_key = "*** end of"
    a = lo.find(start_key)
    if a == -1:
        return raw.strip()
    nl = raw.find("\n", a)
    if nl == -1:
        return raw.strip()
    body_start = nl + 1
    b = lo.find(end_key, body_start)
    if b == -1:
        return raw[body_start:].strip()
    return raw[body_start:b].strip()


TEXT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
if not ALICE_RAW_CACHE_PATH.is_file():
    raise FileNotFoundError(
        f"Missing Alice corpus file:\n  {ALICE_RAW_CACHE_PATH}\n\n"
        "From repo root:\n"
        "  mkdir -p data/text && curl -fL -o data/text/gutenberg_11-0_alice_in_wonderland.txt "
        '"https://raw.githubusercontent.com/GITenberg/Alice-s-Adventures-in-Wonderland_11/master/11-0.txt"\n'
        "Or: uv run python 2026-03-23-Embedding/download_alice_cache.py"
    )

alice_raw = ALICE_RAW_CACHE_PATH.read_text(encoding="utf-8", errors="replace")
print(f"Alice: loaded {len(alice_raw):,} chars from\n  {ALICE_RAW_CACHE_PATH}")

alice_text = _gutenberg_story_body(alice_raw)
print("Story body chars (after Gutenberg markers):", len(alice_text))

print("Tokenizing (large text; can take a few seconds)…", flush=True)
alice_ids = tokenizer.encode(alice_text, add_special_tokens=False)
print("Tokenizing done.", flush=True)
alice_unique: set[int] = set(alice_ids)
in_embed_range = {t for t in alice_unique if 0 <= t < vocab_size}

print("\nTokenization (add_special_tokens=False):")
print("  Total tokens:", len(alice_ids))
print("  Unique token ids appearing in this text:", len(alice_unique))
print("  Of those, within embedding table [0, vocab_size):", len(in_embed_range))
print("  Tokenizer len:", tok_len)
print("  Embedding vocab_size:", vocab_size)
if vocab_size:
    print(f"  Unique ids in text / vocab_size: {len(in_embed_range) / vocab_size:.4%}")

# %% Alice top-100 frequent tokens: embedding cosine similarity heatmap
ALICE_TOP_K = 100

if "alice_ids" not in globals():
    raise RuntimeError("Run the Alice cell first so `alice_ids` is defined.")
if not alice_ids:
    raise RuntimeError("alice_ids is empty; run the Alice cell first.")

cnt = Counter(alice_ids)
alice_by_freq = sorted(
    ((tid, c) for tid, c in cnt.items() if 0 <= tid < vocab_size),
    key=lambda x: (-x[1], x[0]),
)
alice_top = alice_by_freq[:ALICE_TOP_K]
if len(alice_top) < ALICE_TOP_K:
    print(f"Note: only {len(alice_top)} in-vocab types (requested {ALICE_TOP_K}).")

top_ids = [t for t, _ in alice_top]
n_top = len(top_ids)
print(f"\nTop 10 types in Alice (id, count, decode):")
for tid, c in alice_top[:10]:
    dec = tokenizer.decode([tid], skip_special_tokens=False).replace("\n", "\\n")
    print(f"  id={tid:6d}  n={c:7d}  {dec!r}")

E_alice_top = w32[top_ids]
E_alice_top = E_alice_top / (torch.linalg.vector_norm(E_alice_top, dim=1, keepdim=True) + 1e-12)
sim_alice_top = (E_alice_top @ E_alice_top.T).cpu().numpy()

PLOTS_DIR.mkdir(parents=True, exist_ok=True)
# Large canvas + high DPI so every rank/token label stays legible when zoomed.
_cell_in = 0.17
_margin_in = 2.8
fig_w = max(14.0, n_top * _cell_in + _margin_in)
fig_h = max(12.0, n_top * _cell_in + _margin_in)
fig_a, ax_a = plt.subplots(figsize=(fig_w, fig_h), layout="constrained")
im_a = ax_a.imshow(sim_alice_top, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="equal")
fig_a.colorbar(im_a, ax=ax_a, shrink=0.55, label="cosine similarity", fraction=0.03)
ax_a.set_title(
    f"Cosine similarity: top {n_top} Alice token ids by frequency (L2-normalized rows)",
    fontsize=12,
)


def _alice_top_tick_label(i: int) -> str:
    tid = top_ids[i]
    s = tokenizer.decode([tid], skip_special_tokens=False).replace("\n", "⏎").replace("\r", "")
    if len(s) > 28:
        s = s[:27] + "…"
    return f"{i}|id{tid}|{s}"


_ticks_all = np.arange(n_top, dtype=int)
_lbls = [_alice_top_tick_label(int(i)) for i in _ticks_all]
ax_a.set_xticks(_ticks_all)
ax_a.set_yticks(_ticks_all)
ax_a.set_xticklabels(_lbls, rotation=90, ha="center", va="top", fontsize=4.25)
ax_a.set_yticklabels(_lbls, fontsize=4.25)
ax_a.tick_params(axis="both", which="major", length=1.5, pad=1)
ax_a.set_xlabel("rank | tokenizer id | decode (0 = most frequent in Alice)", fontsize=9)
ax_a.set_ylabel("rank | tokenizer id | decode (0 = most frequent in Alice)", fontsize=9)

out_alice_top_path = PLOTS_DIR / "qwen35_alice_top100_token_cosine_similarity.png"
fig_a.savefig(out_alice_top_path, dpi=320, bbox_inches="tight", pad_inches=0.35)
plt.close(fig_a)
print(f"\nSaved cosine heatmap ({n_top}×{n_top}): {out_alice_top_path}")
