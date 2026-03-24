# %% Imports & helpers
"""Token embedding demos: norms, histograms, cosine neighbors, analogies, pairwise cosines (king/queen/man), norm checks.

From repo root (optional one-shot):

    uv run python experiments/2026-03-23-Embedding/demo.py

In Stonesoup: **Watch** this file, then run cells. Each ``# %%`` section is meant to be runnable on its own:
define ``MODEL_NAME`` (and any knobs) inside that cell; shared multi-model list is ``MULTI_MODEL_INVESTIGATION_NAMES`` in helpers.
Input embedding **weights** can also be persisted under ``data/embedding-layers/*.pt`` (repo root) so later runs
load only the tokenizer from HF plus a small ``nn.Embedding`` from disk instead of the full LM.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

_bundle_by_name: dict[str, dict] = {}

# ``demo.py`` → ``experiments/2026-03-23-Embedding/`` → repo root ``data/embedding-layers/``
EMBEDDING_LAYER_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "embedding-layers"
PREFER_EMBEDDING_DISK_CACHE = True
WRITE_EMBEDDING_CACHE_AFTER_FULL_LOAD = True

# One list for cross-model cells (norm compare, analogy, pairwise cosines). Comment out entries you skip.
MULTI_MODEL_INVESTIGATION_NAMES: tuple[str, ...] = (
    # GPT / OPT / BLOOM (decoder classics)
    "distilgpt2",
    "openai-community/gpt2-medium",
    "openai-community/gpt2-large",
    "openai-community/gpt2-xl",
    "facebook/opt-1.3b",
    "facebook/opt-2.7b",
    "bigscience/bloom-560m",
    "bigscience/bloom-1b7",
    # EleutherAI
    "EleutherAI/gpt-neo-1.3B",
    "EleutherAI/pythia-1b",
    # Small Llama-like / open LM
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "openlm-research/open_llama_3b_v2",
    "tiiuae/falcon-rw-1b",
    "meta-llama/Llama-3.2-1B-Instruct",  # gated on HF
    # StableLM / SmolLM
    "stabilityai/stablelm-2-1_6b",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    # Other research / instruct
    "allenai/OLMo-1B-hf",
    # Qwen2.5 / Qwen3 / Qwen3.5 (3.x may need a recent ``transformers`` release)
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B-Instruct-2507",
    "Qwen/Qwen3.5-0.8B",
    "Qwen/Qwen3.5-2B",
    "microsoft/phi-2",
    "microsoft/Phi-3-mini-4k-instruct",  # ~3.8B
    # More open LMs at ~1–3B (comment out if load fails or OOM)
    "togethercomputer/RedPajama-INCITE-Base-3B-v1",
    "mosaicml/mpt-1b-redpajama-200b",
    "cerebras/btlm-3b-8k-base",
    "ibm-granite/granite-3.1-2b-instruct",
    "apple/OpenELM-1_1B-Instruct",
    "allenai/OLMo-2-0425-1B",
    "bigcode/starcoder2-3b",  # code LM — embeddings often differ from natural text LMs
    "meta-llama/Llama-3.2-3B-Instruct",  # gated on HF
    "internlm/internlm2-1_8b",  # may need recent ``transformers``
)


def embedding_layer_cache_path(model_name: str) -> Path:
    safe = model_name.replace("/", "__")
    return EMBEDDING_LAYER_CACHE_DIR / f"{safe}.pt"


def save_embedding_layer_cache(model_name: str, embed_layer: nn.Module) -> Path:
    """Write float32 CPU snapshot of ``embed_layer.weight`` for fast reload."""
    path = embedding_layer_cache_path(model_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    w = embed_layer.weight.detach().float().cpu()
    torch.save(
        {
            "model_name": model_name,
            "weight": w,
            "vocab_size": int(w.shape[0]),
            "hidden_size": int(w.shape[1]),
        },
        path,
    )
    return path


def load_embedding_layer_from_disk(
    model_name: str,
    tokenizer: AutoTokenizer,
    device: torch.device,
    dtype: torch.dtype,
) -> nn.Embedding | None:
    """If a cache file exists and matches ``len(tokenizer)``, return ``nn.Embedding`` on ``device``."""
    path = embedding_layer_cache_path(model_name)
    if not path.is_file():
        return None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    w = payload["weight"].float()
    if w.dim() != 2 or w.shape[0] != len(tokenizer):
        return None
    emb = nn.Embedding(w.shape[0], w.shape[1])
    with torch.no_grad():
        emb.weight.copy_(w)
    emb = emb.to(device=device, dtype=dtype)
    emb.eval()
    return emb


def load_model_bundle(model_name: str) -> dict:
    """Load tokenizer + input embeddings; optional disk cache avoids full ``AutoModelForCausalLM`` when ``.pt`` exists."""
    if model_name in _bundle_by_name:
        return _bundle_by_name[model_name]
    device = pick_device()
    dtype = pick_dtype(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    from_cache_file = False
    model = None
    if PREFER_EMBEDDING_DISK_CACHE:
        cached_emb = load_embedding_layer_from_disk(model_name, tokenizer, device, dtype)
        if cached_emb is not None:
            embedding_layer = cached_emb
            from_cache_file = True

    if not from_cache_file:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype if dtype != torch.float32 else None,
            device_map=None,
            trust_remote_code=True,
        )
        model.to(device)
        model.eval()
        embedding_layer = model.get_input_embeddings()
        if WRITE_EMBEDDING_CACHE_AFTER_FULL_LOAD:
            try:
                save_embedding_layer_cache(model_name, embedding_layer)
            except Exception:
                pass

    bundle = {
        "model_name": model_name,
        "device": device,
        "dtype": dtype,
        "tokenizer": tokenizer,
        "model": model,
        "embedding_layer": embedding_layer,
        "from_embedding_cache": from_cache_file,
    }
    _bundle_by_name[model_name] = bundle
    return bundle


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pick_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def token_embedding_norms(embed_layer: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    """Return L2 norm of each token vector, shape ``(seq_len,)`` (single batch row)."""
    with torch.inference_mode():
        vec = embed_layer(input_ids)
        return vec.float().norm(dim=-1).squeeze(0).cpu()


def all_vocab_embedding_norms(embed_layer: nn.Module) -> torch.Tensor:
    """Return L2 norm of every vocabulary row, shape ``(vocab_size,)``."""
    with torch.inference_mode():
        return embed_layer.weight.float().norm(dim=-1).cpu()


def embedding_row_l2_norm(embed_layer: nn.Module, token_id: int) -> float:
    """L2 norm of the embedding row for a single vocabulary id."""
    with torch.inference_mode():
        return embed_layer.weight[token_id].float().norm().item()


def embedding_rows_cosine_sim(
    embed_layer: nn.Module,
    token_id_a: int,
    token_id_b: int,
    device: torch.device,
) -> float:
    """Cosine similarity between two input embedding rows."""
    with torch.inference_mode():
        wa = embed_layer.weight[token_id_a].float()
        wb = embed_layer.weight[token_id_b].float()
        if device.type == "cuda":
            wa, wb = wa.to(device), wb.to(device)
        na = wa.norm().clamp_min(1e-8)
        nb = wb.norm().clamp_min(1e-8)
        return (wa @ wb / (na * nb)).item()


def cosine_similarity_to_all_rows(
    embed_layer: nn.Module,
    anchor_id: int,
    device: torch.device,
    chunk_size: int = 8192,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cosine similarity of every vocab row to ``anchor_id``; returns ``(cos, norms)`` on CPU, shape ``(V,)``."""
    with torch.inference_mode():
        w = embed_layer.weight
        anchor = w[anchor_id].float()
        if device.type == "cuda":
            anchor = anchor.to(device)
        na = anchor.norm().clamp_min(1e-8)
        v_sz = w.shape[0]
        cos_chunks: list[torch.Tensor] = []
        norm_chunks: list[torch.Tensor] = []
        for start in range(0, v_sz, chunk_size):
            end = min(start + chunk_size, v_sz)
            chunk = w[start:end].float()
            if device.type == "cuda":
                chunk = chunk.to(device)
            nrms = chunk.norm(dim=-1).clamp_min(1e-8)
            cos_chunks.append((chunk @ anchor) / (nrms * na))
            norm_chunks.append(nrms)
        cos = torch.cat(cos_chunks).cpu()
        norms = torch.cat(norm_chunks).cpu()
        return cos, norms


def cosine_similarity_query_to_all_rows(
    embed_layer: nn.Module,
    query: torch.Tensor,
    device: torch.device,
    chunk_size: int = 8192,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cosine similarity of every vocab row to a 1D ``query``; returns ``(cos, row_norms)`` on CPU."""
    with torch.inference_mode():
        w = embed_layer.weight
        q = query.float().flatten()
        if device.type == "cuda":
            q = q.to(device)
        nq = q.norm().clamp_min(1e-8)
        v_sz = w.shape[0]
        cos_chunks: list[torch.Tensor] = []
        norm_chunks: list[torch.Tensor] = []
        for start in range(0, v_sz, chunk_size):
            end = min(start + chunk_size, v_sz)
            chunk = w[start:end].float()
            if device.type == "cuda":
                chunk = chunk.to(device)
            nrms = chunk.norm(dim=-1).clamp_min(1e-8)
            cos_chunks.append((chunk @ q) / (nrms * nq))
            norm_chunks.append(nrms)
        cos = torch.cat(cos_chunks).cpu()
        norms = torch.cat(norm_chunks).cpu()
        return cos, norms


def analogy_embedding_vector(
    embed_layer: nn.Module,
    id_king: int,
    id_man: int,
    id_woman: int,
    device: torch.device,
) -> torch.Tensor:
    """``e(king) - e(man) + e(woman)`` as a 1D float tensor (on ``device`` if CUDA)."""
    with torch.inference_mode():
        w = embed_layer.weight
        vk = w[id_king].float()
        vm = w[id_man].float()
        vw = w[id_woman].float()
        if device.type == "cuda":
            vk, vm, vw = vk.to(device), vm.to(device), vw.to(device)
        return vk - vm + vw


def top_neighbors_excluding_ids(
    cos: torch.Tensor,
    norms: torch.Tensor,
    exclude: set[int],
    k: int,
) -> list[tuple[int, float, float]]:
    """Top-``k`` (token_id, cosine, norm) by cosine, skipping any id in ``exclude``."""
    take = min(cos.numel(), k + len(exclude) + 48)
    vals, idx = torch.topk(cos, k=take)
    out: list[tuple[int, float, float]] = []
    for v, i in zip(vals.tolist(), idx.tolist()):
        if i in exclude:
            continue
        out.append((i, v, norms[i].item()))
        if len(out) >= k:
            break
    return out


def neighbor_rank_excluding(
    cos: torch.Tensor,
    target_id: int,
    exclude: set[int],
) -> int | None:
    """1-based rank of ``target_id`` among ids not in ``exclude``, cosine descending."""
    order = torch.argsort(cos, descending=True)
    rank = 0
    for i in order.tolist():
        if i in exclude:
            continue
        rank += 1
        if i == target_id:
            return rank
    return None


def top_cosine_neighbors_excluding(
    cos: torch.Tensor,
    norms: torch.Tensor,
    anchor_id: int,
    k: int,
) -> list[tuple[int, float, float]]:
    """Return up to ``k`` entries ``(token_id, cosine, norm)``, highest cosine first, skipping ``anchor_id``."""
    take = min(k + 1, cos.numel())
    vals, idx = torch.topk(cos, k=take)
    out: list[tuple[int, float, float]] = []
    for v, i in zip(vals.tolist(), idx.tolist()):
        if i == anchor_id:
            continue
        out.append((i, v, norms[i].item()))
        if len(out) >= k:
            break
    return out


def format_token_piece_for_plot(piece: str) -> str:
    """HF BPE uses ``Ġ`` for word-leading space; use a normal space in plot labels."""
    return piece.replace("Ġ", " ")


def first_subtoken_for_text(
    tokenizer: AutoTokenizer,
    text: str,
) -> tuple[int, list[int], str]:
    """First token id from ``text``, full id list, and tokenizer piece for that id."""
    enc = tokenizer(text, add_special_tokens=False, return_tensors="pt")
    ids = enc["input_ids"][0].tolist()
    tid = ids[0]
    piece = tokenizer.convert_ids_to_tokens([tid])[0]
    return tid, ids, piece


# %% Load model & tokenizer (optional warm-up; other cells call ``load_model_bundle`` too)
# Meta Llama 3.2 ~1B Instruct (gated — ``huggingface-cli login`` + accept license on HF).
# Ungated Llama-style fallback: ``openlm-research/open_llama_3b_v2``
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

bundle = load_model_bundle(MODEL_NAME)
device = bundle["device"]
dtype = bundle["dtype"]
tokenizer = bundle["tokenizer"]
model = bundle["model"]
embedding_layer = bundle["embedding_layer"]

vocab_size, hidden_size = embedding_layer.weight.shape
print(f"loaded {MODEL_NAME!r}")
print(f"device={device}, dtype={dtype}")
if bundle.get("from_embedding_cache"):
    print(f"input embeddings: disk cache {embedding_layer_cache_path(MODEL_NAME)}")
else:
    print("input embeddings: full model (written to disk cache if enabled)")
print(f"embedding: nn.Embedding({vocab_size}, {hidden_size})")
print(f"weight dtype: {embedding_layer.weight.dtype}")

# %% Full-vocab norms, example sequence, and norm histogram
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

bundle = load_model_bundle(MODEL_NAME)
device = bundle["device"]
tokenizer = bundle["tokenizer"]
embedding_layer = bundle["embedding_layer"]

vocab_norms = all_vocab_embedding_norms(embedding_layer)
vn = vocab_norms.float()
print(f"measured {vn.numel()} vocabulary rows (||e||_2 per token type)")
print(
    f"  min={vn.min().item():.4f}  max={vn.max().item():.4f}  "
    f"mean={vn.mean().item():.4f}  std={vn.std(unbiased=False).item():.4f}"
)

sample_text = (
    "Transformers map discrete tokens to dense vectors. "
    "Each row of the embedding table is one token type's vector in R^d."
)

enc = tokenizer(sample_text, return_tensors="pt", add_special_tokens=True)
input_ids = enc["input_ids"].to(device)
seq_norms = token_embedding_norms(embedding_layer, input_ids)

ids = input_ids[0].tolist()
pieces = tokenizer.convert_ids_to_tokens(ids)
print(f"\nexample sequence ({len(pieces)} positions): {sample_text[:80]!r}…")
for i, (piece, tid, n) in enumerate(zip(pieces, ids, seq_norms.tolist())):
    print(f"  {i:3}  {piece!r:18}  id={tid:6}  ||e||_2 = {n:.4f}")

plots_dir = Path(__file__).resolve().parent / "plots"
plots_dir.mkdir(parents=True, exist_ok=True)
plot_stem = MODEL_NAME.replace("/", "_")
plot_path = plots_dir / f"{plot_stem}_token_embedding_norms.png"

fig, ax = plt.subplots(figsize=(7, 4))
n_bins = min(128, max(32, int(vn.numel() ** 0.5)))
ax.hist(vn.numpy(), bins=n_bins, color="steelblue", edgecolor="white", alpha=0.9)
ax.set_xlabel("L2 norm of token embedding vector")
ax.set_ylabel("Count")
ax.set_title(f"Vocabulary embedding norms (n={vn.numel()}) — {MODEL_NAME.split('/')[-1]}")
ax.grid(True, alpha=0.25)
plt.tight_layout()

fig.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"saved {plot_path}")

# %% Cosine neighbors, table, and norm bar chart
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
# First subtoken id from this string is the anchor; leading space often selects a word piece (e.g. ``Ġking``).
COSINE_QUERY_STRING = " terrible"
COSINE_TOP_K = 25
COSINE_CHUNK_SIZE = 8192

bundle = load_model_bundle(MODEL_NAME)
device = bundle["device"]
tokenizer = bundle["tokenizer"]
embedding_layer = bundle["embedding_layer"]

enc_anchor = tokenizer(COSINE_QUERY_STRING, add_special_tokens=False, return_tensors="pt")
anchor_ids = enc_anchor["input_ids"][0].tolist()
if len(anchor_ids) != 1:
    print(
        f"\ncosine anchor: {COSINE_QUERY_STRING!r} tokenizes to {len(anchor_ids)} ids {anchor_ids}; "
        f"using first id only."
    )
anchor_id = anchor_ids[0]
anchor_piece = tokenizer.convert_ids_to_tokens([anchor_id])[0]
cos_all, norms_all = cosine_similarity_to_all_rows(
    embedding_layer, anchor_id, device, chunk_size=COSINE_CHUNK_SIZE
)
anchor_cos = cos_all[anchor_id].item()
anchor_norm = norms_all[anchor_id].item()

print(
    f"\ncosine neighbors of anchor id={anchor_id} piece={anchor_piece!r} "
    f"(query string {COSINE_QUERY_STRING!r})"
)
print(f"  anchor:  cos={anchor_cos:.6f}  ||e||_2={anchor_norm:.4f}")
print(f"  top {COSINE_TOP_K} other vocab types by cosine similarity (directional alignment):")
neighbors = top_cosine_neighbors_excluding(cos_all, norms_all, anchor_id, COSINE_TOP_K)
for rank, (tid, c, nrm) in enumerate(neighbors, start=1):
    piece = tokenizer.convert_ids_to_tokens([tid])[0]
    print(f"  {rank:3}  id={tid:7}  piece={piece!r:22}  cos={c:.6f}  ||e||_2={nrm:.4f}")

plots_dir = Path(__file__).resolve().parent / "plots"
plots_dir.mkdir(parents=True, exist_ok=True)
plot_stem = MODEL_NAME.replace("/", "_")

piece_labels = [format_token_piece_for_plot(anchor_piece)] + [
    format_token_piece_for_plot(tokenizer.convert_ids_to_tokens([tid])[0]) for tid, _, _ in neighbors
]
norms_plot = [anchor_norm] + [n for _, _, n in neighbors]
bar_colors = ["#c0392b"] + ["steelblue"] * len(neighbors)
xs = list(range(len(norms_plot)))
fig_w = min(22.0, max(7.5, 0.38 * len(norms_plot)))

fig_cos, ax_cos = plt.subplots(figsize=(fig_w, 4.8))
ax_cos.bar(xs, norms_plot, color=bar_colors, edgecolor="white", linewidth=0.7, alpha=0.92)
ax_cos.set_xticks(xs)
ax_cos.set_xticklabels(piece_labels, rotation=60, ha="right", fontsize=8)
ax_cos.set_ylabel("L2 norm ||e||")
ax_cos.set_xlabel("token (anchor, then neighbors ranked by cosine to anchor)")
ax_cos.set_title(
    f"Embedding norms — anchor {format_token_piece_for_plot(anchor_piece)!r}, "
    f"top {COSINE_TOP_K} cosine neighbors — {MODEL_NAME.split('/')[-1]}"
)
ax_cos.set_ylim(bottom=0.0)
ax_cos.grid(True, axis="y", alpha=0.25)
plt.tight_layout()

neighbor_norm_plot_path = plots_dir / f"{plot_stem}_cosine_neighbor_norms.png"
fig_cos.savefig(neighbor_norm_plot_path, dpi=150, bbox_inches="tight")
print(f"saved {neighbor_norm_plot_path}")

# %% Print all tokens with embedding L2 norm above a threshold
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
NORM_THRESHOLD = 1.1

bundle = load_model_bundle(MODEL_NAME)
tokenizer = bundle["tokenizer"]
embedding_layer = bundle["embedding_layer"]

norms = all_vocab_embedding_norms(embedding_layer).float()
above = torch.where(norms > NORM_THRESHOLD)[0]
order = torch.argsort(norms[above], descending=True)
idx_sorted = above[order].tolist()
norm_sorted = norms[above[order]].tolist()

print(f"tokens with ||e||_2 > {NORM_THRESHOLD}: {len(idx_sorted)} / {norms.numel()}")
for tid, nrm in zip(idx_sorted, norm_sorted):
    piece = tokenizer.convert_ids_to_tokens([tid])[0]
    print(f"  id={tid:7}  piece={piece!r:42}  ||e||_2={nrm:.4f}")

# %% Compare input embedding norm: `` bad`` vs `` terrible`` across model families
# Leading space matches many GPT-style word pieces (``Ġ``). Model list: ``MULTI_MODEL_INVESTIGATION_NAMES`` (helpers).
MODEL_NAMES = MULTI_MODEL_INVESTIGATION_NAMES
WORD_A = " problem"
WORD_B = " catastrophe"

print(
    f"claim: ||e||({WORD_A!r}) > ||e||({WORD_B!r})  "
    f"(first subtoken only if the string splits into several ids)\n"
)


def _norm_for_query(
    tokenizer: AutoTokenizer,
    embed_layer: nn.Module,
    text: str,
) -> tuple[float, int, str, list[int]]:
    enc = tokenizer(text, add_special_tokens=False, return_tensors="pt")
    ids = enc["input_ids"][0].tolist()
    tid = ids[0]
    piece = tokenizer.convert_ids_to_tokens([tid])[0]
    return embedding_row_l2_norm(embed_layer, tid), tid, piece, ids


n_ok = 0
n_bad_wins = 0
claim_yes_models: list[str] = []
claim_no_models: list[str] = []
for model_name in MODEL_NAMES:
    try:
        bundle = load_model_bundle(model_name)
    except Exception as err:
        print(f"{model_name}\n  SKIP load: {err}\n")
        continue
    tokenizer = bundle["tokenizer"]
    embed_layer = bundle["embedding_layer"]
    short = model_name.split("/")[-1]

    try:
        na, id_a, piece_a, ids_a = _norm_for_query(tokenizer, embed_layer, WORD_A)
        nb, id_b, piece_b, ids_b = _norm_for_query(tokenizer, embed_layer, WORD_B)
    except Exception as err:
        print(f"{model_name}\n  SKIP measure: {err}\n")
        continue

    n_ok += 1
    delta = na - nb
    claim_holds = na > nb
    if claim_holds:
        n_bad_wins += 1
        claim_yes_models.append(model_name)
    else:
        claim_no_models.append(model_name)

    warn_a = f" [{len(ids_a)} subtokens → using first]" if len(ids_a) != 1 else ""
    warn_b = f" [{len(ids_b)} subtokens → using first]" if len(ids_b) != 1 else ""
    print(f"{short} ({model_name})")
    print(f"  {WORD_A!r}{warn_a}  id={id_a:7}  piece={piece_a!r:28}  ||e||_2 = {na:.4f}")
    print(f"  {WORD_B!r}{warn_b}  id={id_b:7}  piece={piece_b!r:28}  ||e||_2 = {nb:.4f}")
    print(f"  Δ = {delta:+.4f}   claim (bad > terrible): {'yes' if claim_holds else 'no'}\n")

if n_ok:
    print(f"summary: claim held in {n_bad_wins} / {n_ok} successfully loaded models.")
    print(f"  yes (||e|| {WORD_A} > ||e|| {WORD_B}):")
    if claim_yes_models:
        for m in claim_yes_models:
            print(f"    - {m}")
    else:
        print("    - (none)")
    print("  no:")
    if claim_no_models:
        for m in claim_no_models:
            print(f"    - {m}")
    else:
        print("    - (none)")
else:
    print("summary: no models completed (check network, HF auth, or ``MODEL_NAMES``).")

# %% Analogy ``king - man + woman``: top neighbors vs `` queen``
# Classical word2vec expectation: `` queen`` is near the analogy vector. Model list: ``MULTI_MODEL_INVESTIGATION_NAMES`` (helpers).
ANALOGY_MODEL_NAMES = MULTI_MODEL_INVESTIGATION_NAMES
KING_S = " king"
MAN_S = " man"
WOMAN_S = " woman"
QUEEN_S = " queen"
ANALOGY_TOP_K = 10
ANALOGY_CHUNK_SIZE = 8192

print(
    f"analogy: e({KING_S!r}) - e({MAN_S!r}) + e({WOMAN_S!r})  →  "
    f"top-{ANALOGY_TOP_K} cosine neighbors (excluding king/man/woman ids); "
    f"is {QUEEN_S!r} among them?\n"
)


for model_name in ANALOGY_MODEL_NAMES:
    try:
        bundle = load_model_bundle(model_name)
    except Exception as err:
        print(f"{model_name}\n  SKIP load: {err}\n")
        continue
    tokenizer = bundle["tokenizer"]
    embed_layer = bundle["embedding_layer"]
    device = bundle["device"]
    short = model_name.split("/")[-1]

    try:
        id_k, ids_k, pk = first_subtoken_for_text(tokenizer, KING_S)
        id_m, ids_m, pm = first_subtoken_for_text(tokenizer, MAN_S)
        id_w, ids_w, pw = first_subtoken_for_text(tokenizer, WOMAN_S)
        id_q, ids_q, pq = first_subtoken_for_text(tokenizer, QUEEN_S)
    except Exception as err:
        print(f"{short} ({model_name})\n  SKIP tokenize: {err}\n")
        continue

    warn = []
    if len(ids_k) != 1:
        warn.append(f"{KING_S!r}→{len(ids_k)} ids")
    if len(ids_m) != 1:
        warn.append(f"{MAN_S!r}→{len(ids_m)} ids")
    if len(ids_w) != 1:
        warn.append(f"{WOMAN_S!r}→{len(ids_w)} ids")
    if len(ids_q) != 1:
        warn.append(f"{QUEEN_S!r}→{len(ids_q)} ids")
    warn_s = f"  note: {'; '.join(warn)} (using first subtoken each)\n" if warn else ""

    try:
        q_vec = analogy_embedding_vector(embed_layer, id_k, id_m, id_w, device)
        cos_all, norms_all = cosine_similarity_query_to_all_rows(
            embed_layer, q_vec, device, chunk_size=ANALOGY_CHUNK_SIZE
        )
    except Exception as err:
        print(f"{short} ({model_name})\n  SKIP analogy: {err}\n")
        continue

    exclude = {id_k, id_m, id_w}
    topn = top_neighbors_excluding_ids(cos_all, norms_all, exclude, ANALOGY_TOP_K)
    top_ids = {t[0] for t in topn}
    queen_in_top = id_q in top_ids
    queen_rank = neighbor_rank_excluding(cos_all, id_q, exclude)

    print(f"{short} ({model_name})")
    if warn_s:
        print(warn_s.rstrip())
    print(f"  ids: king={id_k} man={id_m} woman={id_w} queen={id_q}")
    print(f"  pieces: {pk!r}  {pm!r}  {pw!r}  |  target {pq!r}")
    print(f"  {QUEEN_S!r} in top-{ANALOGY_TOP_K} (excl. king/man/woman): {'yes' if queen_in_top else 'no'}")
    if queen_rank is not None:
        print(f"  rank of {QUEEN_S!r} among neighbors (same exclusion): {queen_rank}")
    else:
        print(f"  rank of {QUEEN_S!r}: (not found)")
    print(f"  top-{ANALOGY_TOP_K} neighbors:")
    for rank, (tid, c, nrm) in enumerate(topn, start=1):
        piece = tokenizer.convert_ids_to_tokens([tid])[0]
        mark = "  ← queen" if tid == id_q else ""
        print(f"    {rank:2}  id={tid:7}  cos={c:.4f}  ||e||={nrm:.4f}  piece={piece!r}{mark}")
    print()

# %% Pairwise cosines: `` king`` vs `` queen``, `` king`` vs `` man`` (input embeddings)
# Same first-subtoken convention as the analogy cell. Model list: ``MULTI_MODEL_INVESTIGATION_NAMES`` (helpers).
PAIR_COSINE_MODEL_NAMES = MULTI_MODEL_INVESTIGATION_NAMES
PC_KING = " king"
PC_QUEEN = " queen"
PC_MAN = " man"

print(
    f"cosine between input embedding rows (first subtoken per string): "
    f"e({PC_KING!r}) vs e({PC_QUEEN!r}), e({PC_KING!r}) vs e({PC_MAN!r})\n"
)
print(f"{'model':<42}  {'cos(k,q)':>10}  {'cos(k,m)':>10}")
print("-" * 68)

for model_name in PAIR_COSINE_MODEL_NAMES:
    short = model_name.split("/")[-1]
    try:
        bundle = load_model_bundle(model_name)
    except Exception as err:
        print(f"{short:<42}  {'—':>10}  {'—':>10}    (load) {err}")
        continue
    tokenizer = bundle["tokenizer"]
    embed_layer = bundle["embedding_layer"]
    device = bundle["device"]
    try:
        id_k, _, _ = first_subtoken_for_text(tokenizer, PC_KING)
        id_q, _, _ = first_subtoken_for_text(tokenizer, PC_QUEEN)
        id_m, _, _ = first_subtoken_for_text(tokenizer, PC_MAN)
        cq = embedding_rows_cosine_sim(embed_layer, id_k, id_q, device)
        cm = embedding_rows_cosine_sim(embed_layer, id_k, id_m, device)
    except Exception as err:
        print(f"{short:<42}  {'—':>10}  {'—':>10}    (measure) {err}")
        continue
    label = short if len(short) <= 40 else short[:37] + "..."
    print(f"{label:<42}  {cq:10.4f}  {cm:10.4f}")
print()
