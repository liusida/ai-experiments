# %% Imports & paths
"""Summarize Qwen3.5-0.8B **input embedding** weights from disk cache plus tokenizer-aligned **token** metadata.

Loads ``data/embedding-layers/Qwen__Qwen3.5-0.8B.pt`` (written by ``demo.py``). Requires the tokenizer from
Hugging Face for string-side statistics. From repo root::

    uv run python 2026-03-23-Embedding/embedding_qwen35_statistics.py

In Stonesoup: **Watch** this file and run cells in order (kernel keeps globals).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
LAYER_DIR = REPO_ROOT / "data" / "embedding-layers"
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
    print("\nCosine similarity (L2-normalized mean of subword rows per string):")
    colw = max(8, max(len(x) for x in labels))
    head = " " * colw + "  " + "  ".join(f"{lab:>{colw}}" for lab in labels)
    print(head)
    for i, lab in enumerate(labels):
        row = "  ".join(f"{sim[i, j]:{colw}.4f}" for j in range(len(labels)))
        print(f"{lab:>{colw}}  {row}")

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
