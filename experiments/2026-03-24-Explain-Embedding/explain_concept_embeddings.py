# %% Imports & helpers
"""Explain a “concept in quotes” with Qwen3.5-2B, then swap the quoted span’s **input embeddings**
for random or arithmetic (e.g. (king + queen) / 2) vectors and compare generations.

Prompts use the model’s **HF ``chat_template``** (``apply_chat_template`` + ``add_generation_prompt``),
not raw text, so behavior matches the instruct checkpoint.

From repo root (optional one-shot)::

    uv run python experiments/2026-03-24-Explain-Embedding/explain_concept_embeddings.py

In Stonesoup: **Watch** this file and run cells in order. Heavy cell: **Load Qwen3.5-2B** (~download + GPU RAM).
Use **Explain a literal concept** with the header text field (``CELL_INPUT``) to type a word or short phrase, then **Run**.
Cell input is **not** stripped so phrases like a leading space (e.g. GPT-style `` table``) stay exact.
**Random embedding** cell: ``CELL_INPUT`` sets the torch RNG seed (integer / ``0x…`` / any string → sha256); blank → ``0``.
**Histogram** cell (many random vectors vs vocab max-cosine): plot under ``plots/random_vec_max_vocab_cosine_hist.png``; ``CELL_INPUT`` seeds that sample.
**Mean-cosine** cell: per token, average cosine to every *other* token; histogram → ``plots/mean_cosine_to_other_tokens_hist.png``.
**UMAP** cell: ``uv pip install "umap-learn"`` (or ``-e ".[embedding]"``); popularity = weighted counts from ``google_10000_english.txt`` for **single-token** words only (multi-token list words are skipped).
Running the full file with ``uv run python …`` still uses ``table`` / ``king`` / ``apple`` for that step (``CELL_INPUT`` is unset).
Requires a recent ``transformers`` with Qwen3.5 support and HF access to ``Qwen/Qwen3.5-2B``.
"""


from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from threading import Thread

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

MODEL_NAME = "Qwen/Qwen3.5-2B"

# Keep the tail a single question; extra rules get paraphrased as “Constraint 1…N” by small LMs.
PROMPT_HEAD = 'What does "'
PROMPT_TAIL = '" mean?'

# Cap generation (raise if explanations truncate mid-section).
EXPLAIN_MAX_NEW_TOKENS = 512

# When True, ``explain_literal_phrase`` prints the chat-formatted string and a decode of ``input_ids``.
SHOW_CHAT_PROMPT_DEBUG = True

# Span-replacement skips that dump by default (noisy vs ``print_pre_generate_llm_input``).
SHOW_CHAT_PROMPT_DEBUG_SPAN_SWAP = False

# When True, ``generate_from_embeds`` prints tokens as they arrive (threaded ``TextIteratorStreamer``).
STREAM_LLM_OUTPUT = True

# Right before ``model.generate``: decoded ``input_ids`` + optional slice of actual ``inputs_embeds`` at span.
SHOW_PRE_GENERATE_PROMPT = True

# Quoted literal in random / arithmetic span-replacement cells (``token`` reads clearly vs “table” the object).
SPAN_REPLACEMENT_DEMO_WORD = "token"

# Matplotlib outputs (histogram + UMAP).
EMBED_EXPERIMENT_PLOT_DPI = 300
UMAP_FIGSIZE_INCHES = (18, 14)


def torch_seed_from_cell_input(raw: object | None) -> int:
    """Derive a stable int seed from Stonesoup ``CELL_INPUT`` (or ``None`` / blank → ``0``)."""
    if raw is None:
        return 0
    s = str(raw).strip()
    if not s:
        return 0
    try:
        return int(s, 0)
    except ValueError:
        h = hashlib.sha256(s.encode("utf-8")).digest()[:8]
        return int.from_bytes(h, "big") % (2**63)


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pick_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def find_subsequence(haystack: list[int], needle: list[int]) -> tuple[int, int] | None:
    if not needle:
        return None
    n = len(needle)
    for i in range(0, len(haystack) - n + 1):
        if haystack[i : i + n] == needle:
            return i, i + n
    return None


def _as_id_list(raw: list[int] | torch.Tensor | Mapping[str, object]) -> list[int]:
    if isinstance(raw, torch.Tensor):
        return raw.tolist()
    if isinstance(raw, Mapping) and "input_ids" in raw:
        # ``apply_chat_template(..., tokenize=True)`` returns ``BatchEncoding`` / dict (transformers ≥5).
        return _as_id_list(raw["input_ids"])  # type: ignore[arg-type]
    if isinstance(raw, list):
        return list(raw)
    raise TypeError(f"expected token id list, Tensor, or mapping with input_ids; got {type(raw)!r}")


def chat_input_ids(tokenizer: AutoTokenizer, user_text: str) -> list[int]:
    """
    Instruct Qwen3.5 expects ``<|im_start|>user`` … ``</think>`` … ``<|im_start|>assistant``
    (see ``tokenizer_config.json`` ``chat_template`` on the HF repo), not raw text.
    """
    messages = [{"role": "user", "content": user_text}]
    try:
        raw = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            enable_thinking=False,
        )
    except TypeError:
        raw = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
        )
    return _as_id_list(raw)


def render_chat_prompt(tokenizer: AutoTokenizer, user_text: str) -> str:
    """Same Jinja output as ``chat_input_ids``, but the string that gets tokenized (for debugging)."""
    messages = [{"role": "user", "content": user_text}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )


def print_chat_prompt_debug(
    tokenizer: AutoTokenizer,
    user_text: str,
    input_ids: list[int],
    *,
    label: str = "LLM prompt (chat template → tokenized as input_ids / inputs_embeds)",
) -> None:
    rendered = render_chat_prompt(tokenizer, user_text)
    decoded = tokenizer.decode(input_ids, skip_special_tokens=False)
    print("-" * 72)
    print(label)
    print("-" * 72)
    print("User message (content only):")
    print(user_text)
    print()
    print("Chat-formatted string (repr, matches tokenizer before generation):")
    print(repr(rendered))
    print()
    print("Decode of input_ids (skip_special_tokens=False):")
    print(repr(decoded))
    print(f"(len(input_ids)={len(input_ids)})")
    print("-" * 72)


def print_pre_generate_llm_input(
    tokenizer: AutoTokenizer,
    input_ids: list[int],
    inputs_embeds: torch.Tensor,
    *,
    span: tuple[int, int] | None = None,
    preview_dims: int = 8,
) -> None:
    """What goes into ``generate``: nominal string from ids + true ``inputs_embeds`` (incl. span overrides)."""
    decoded = tokenizer.decode(input_ids, skip_special_tokens=False)
    print("-" * 72)
    print("Right before model.generate — inputs_embeds (nominal decode + numeric span slice)")
    print("-" * 72)
    print("Decode of input_ids (skip_special_tokens=False), same length as inputs_embeds:")
    print(repr(decoded))
    print(
        f"len(input_ids)={len(input_ids)} | inputs_embeds shape={tuple(inputs_embeds.shape)} "
        f"(batch, seq, hidden)"
    )
    if span is not None:
        s, e = span
        span_dec = tokenizer.decode(input_ids[s:e], skip_special_tokens=False)
        print(
            f"Quoted span [{s}, {e}) — ids decode as {span_dec!r} "
            f"but inputs_embeds[{s}:{e}] are overridden (see vector below)."
        )
        head = inputs_embeds[0, s, :preview_dims].detach().float().cpu().tolist()
        print(f"inputs_embeds[0, {s}, :{preview_dims}] = {head}")
        if e - s > 1:
            print(
                f"(same replacement vector broadcast to {e - s} positions; "
                f"values match at each index in [{s}, {e}).)"
            )
    print("-" * 72)


def prompt_and_span(
    tokenizer: AutoTokenizer,
    phrase: str,
) -> tuple[str, list[int], int, int]:
    """User-visible question text, chat-formatted ``input_ids``, span of ``phrase`` tokens inside them."""
    # Qwen's HF ``chat_template`` trims user ``content`` (Jinja ``|trim``). Tokenization used in
    # ``apply_chat_template`` must match what we search for in ``ids``, or user_ids won't align.
    user_text = f"{PROMPT_HEAD}{phrase}{PROMPT_TAIL}".strip()
    ids = chat_input_ids(tokenizer, user_text)
    needle = tokenizer(phrase, add_special_tokens=False)["input_ids"]
    span = find_subsequence(ids, needle)
    if span is None:
        head_ids = tokenizer(PROMPT_HEAD, add_special_tokens=False)["input_ids"]
        user_ids = tokenizer(user_text, add_special_tokens=False)["input_ids"]
        u_block = find_subsequence(ids, user_ids)
        if u_block is None:
            raise RuntimeError(
                f"Could not locate user message tokens in chat-formatted ids for phrase {phrase!r}"
            )
        u0, _u1 = u_block
        start_u = len(head_ids)
        end_u = start_u + len(needle)
        if end_u > len(user_ids) or user_ids[start_u:end_u] != needle:
            inner = find_subsequence(user_ids, needle)
            if inner is None:
                raise RuntimeError(
                    f"Could not align phrase {phrase!r} inside user text tokenization; "
                    f"user_ids_len={len(user_ids)}, needle_len={len(needle)}"
                )
            start_u, end_u = inner
        span = (u0 + start_u, u0 + end_u)
    start, end = span
    return user_text, ids, start, end


def mean_phrase_embedding(
    embed_layer: torch.nn.Module,
    tokenizer: AutoTokenizer,
    phrase: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Mean of embedding rows for all sub-tokens of ``phrase`` (shape ``(hidden,)``)."""
    ids = tokenizer(phrase, add_special_tokens=False)["input_ids"]
    if not ids:
        raise ValueError(f"empty tokenization for phrase {phrase!r}")
    w = embed_layer.weight
    vecs = w[torch.tensor(ids, device=device, dtype=torch.long)]
    return vecs.to(dtype=torch.float32).mean(dim=0).to(dtype=dtype)


def print_top_k_token_neighbors_cosine(
    embed_layer: torch.nn.Module,
    tokenizer: AutoTokenizer,
    vec: torch.Tensor,
    k: int = 5,
    *,
    label: str = "Top vocabulary tokens by cosine similarity to vector",
) -> None:
    """Rank ``embed_layer.weight`` rows by cosine similarity to ``vec`` (shape ``(hidden,)``)."""
    W = embed_layer.weight
    dev = W.device
    v = vec.detach().to(device=dev, dtype=torch.float32).reshape(-1)
    w = W.detach().float()
    v_n = v / (v.norm() + 1e-8)
    w_n = w / (w.norm(dim=-1, keepdim=True) + 1e-8)
    sim = w_n @ v_n
    k_eff = min(k, int(sim.shape[0]))
    top = torch.topk(sim, k=k_eff, largest=True)
    print("-" * 72)
    print(label)
    print("-" * 72)
    for rank in range(k_eff):
        tid = int(top.indices[rank].item())
        score = float(top.values[rank].item())
        piece = tokenizer.decode([tid], skip_special_tokens=False)
        tok = tokenizer.convert_ids_to_tokens(tid)
        print(f"  {rank + 1}. id={tid}  cosine={score:.6f}  token={tok!r}  decode={piece!r}")
    print("-" * 72)


def max_vocab_cosine_similarities(
    embed_layer: torch.nn.Module,
    rand_vecs: torch.Tensor,
) -> torch.Tensor:
    """
    ``rand_vecs`` shape ``(n, hidden)``. Returns shape ``(n,)`` — per row, max cosine to any
    ``embed_layer.weight`` row (full vocabulary).
    """
    W = embed_layer.weight
    dev = W.device
    V = rand_vecs.detach().to(device=dev, dtype=torch.float32)
    w = W.detach().float()
    V_n = V / (V.norm(dim=-1, keepdim=True) + 1e-8)
    w_n = w / (w.norm(dim=-1, keepdim=True) + 1e-8)
    sim = V_n @ w_n.T
    return sim.max(dim=-1).values


def mean_cosine_to_other_tokens(embed_layer: torch.nn.Module) -> torch.Tensor:
    """
    For each vocabulary row ``i``, mean cosine similarity to all ``j != i`` (same normalization
    as full pairwise cosines, without building a ``(V, V)`` matrix).

    With rows L2-normalized to ``Wn``, ``sum_j cos(i,j) = Wn[i] @ sum_j Wn[j]``; subtract
    ``cos(i,i)=1`` and divide by ``V - 1``.
    """
    W = embed_layer.weight.detach().float()
    Wn = W / (W.norm(dim=1, keepdim=True) + 1e-8)
    v = int(Wn.shape[0])
    row_sum = Wn @ Wn.sum(dim=0)
    return (row_sum - 1.0) / float(v - 1)


def weighted_subtoken_counts_from_ranked_wordlist(
    wordlist_path: Path,
    tokenizer: AutoTokenizer,
) -> Counter[int]:
    """
    Each line in ``wordlist_path`` is one word; **earlier lines count more** (rank 0 = heaviest).
    For each word, tokenizer output subtoken ids get that weight added to their tallies.
    This approximates “popular” pieces for common-English orderings (e.g. Google 10k lists).
    """
    text = wordlist_path.read_text(encoding="utf-8")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    n = len(lines)
    ctr: Counter[int] = Counter()
    for rank, w in enumerate(lines):
        wgt = n - rank
        for tid in tokenizer(w, add_special_tokens=False)["input_ids"]:
            ctr[int(tid)] += wgt
    return ctr


def weighted_single_token_word_counts_from_ranked_wordlist(
    wordlist_path: Path,
    tokenizer: AutoTokenizer,
) -> tuple[Counter[int], dict[str, int]]:
    """
    Like ``weighted_subtoken_counts_from_ranked_wordlist``, but **only** words that encode to
    **exactly one** token (``len(ids) == 1``). Multi-token words are skipped.
    Returns ``(counter, stats)`` with keys ``lines``, ``single_token_words``, ``skipped_multi``.
    """
    text = wordlist_path.read_text(encoding="utf-8")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    n = len(lines)
    ctr: Counter[int] = Counter()
    single = 0
    skipped_multi = 0
    for rank, w in enumerate(lines):
        wgt = n - rank
        ids = tokenizer(w, add_special_tokens=False)["input_ids"]
        if len(ids) != 1:
            skipped_multi += 1
            continue
        ctr[int(ids[0])] += wgt
        single += 1
    stats = {
        "lines": n,
        "single_token_words": single,
        "skipped_multi": skipped_multi,
    }
    return ctr, stats


def build_inputs_embeds_from_ids(
    embed_layer: torch.nn.Module,
    input_ids: list[int],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    ids_t = torch.tensor([input_ids], device=device, dtype=torch.long)
    return embed_layer(ids_t).to(dtype=dtype)


def replace_span_with_vector(
    embeds: torch.Tensor,
    start: int,
    end: int,
    vec: torch.Tensor,
) -> torch.Tensor:
    """``vec`` shape ``(hidden,)`` — broadcast over ``[start, end)``."""
    out = embeds.clone()
    span = end - start
    if span <= 0:
        raise ValueError("empty span")
    v = vec.to(device=out.device, dtype=out.dtype).view(1, 1, -1).expand(1, span, -1)
    out[:, start:end, :] = v
    return out


def generate_from_embeds(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    inputs_embeds: torch.Tensor,
    max_new_tokens: int = 256,
    *,
    stream: bool | None = None,
) -> str:
    """Continue from ``inputs_embeds``; decode **only** newly generated tokens."""
    if stream is None:
        stream = STREAM_LLM_OUTPUT
    device = inputs_embeds.device
    prompt_len = inputs_embeds.shape[1]
    attn = torch.ones(1, prompt_len, device=device, dtype=torch.long)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    gen_kw = {
        "inputs_embeds": inputs_embeds,
        "attention_mask": attn,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "pad_token_id": pad_id,
    }
    if tokenizer.eos_token_id is not None:
        gen_kw["eos_token_id"] = tokenizer.eos_token_id

    if not stream:
        with torch.inference_mode():
            out = model.generate(**gen_kw)
        new_ids = out[0, prompt_len:]
        return tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )
    gen_kw["streamer"] = streamer

    def _run_generate() -> None:
        with torch.inference_mode():
            model.generate(**gen_kw)

    thread = Thread(target=_run_generate, daemon=True)
    thread.start()
    parts: list[str] = []
    try:
        for chunk in streamer:
            print(chunk, end="", flush=True)
            parts.append(chunk)
    finally:
        thread.join()
    print()
    return "".join(parts).strip()


def explain_literal_phrase(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    embed_layer: torch.nn.Module,
    phrase: str,
    device: torch.device,
    dtype: torch.dtype,
    max_new_tokens: int = EXPLAIN_MAX_NEW_TOKENS,
    *,
    debug_prompt: bool | None = None,
    pre_generate_prompt: bool | None = None,
    stream: bool | None = None,
) -> str:
    user_text, ids, _, _ = prompt_and_span(tokenizer, phrase)
    if debug_prompt is None:
        debug_prompt = SHOW_CHAT_PROMPT_DEBUG
    if debug_prompt:
        print_chat_prompt_debug(tokenizer, user_text, ids)
    embeds = build_inputs_embeds_from_ids(embed_layer, ids, device, dtype)
    if pre_generate_prompt is None:
        pre_generate_prompt = SHOW_PRE_GENERATE_PROMPT
    if pre_generate_prompt:
        print_pre_generate_llm_input(tokenizer, ids, embeds, span=None)
    return generate_from_embeds(
        model, tokenizer, embeds, max_new_tokens=max_new_tokens, stream=stream
    )


def explain_with_span_replacement(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    embed_layer: torch.nn.Module,
    phrase_for_span: str,
    replacement: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    max_new_tokens: int = EXPLAIN_MAX_NEW_TOKENS,
    *,
    debug_prompt: bool | None = None,
    pre_generate_prompt: bool | None = None,
    stream: bool | None = None,
) -> str:
    """Same token layout as ``phrase_for_span`` in the quote, but embeddings on that span = ``replacement``."""
    user_text, ids, start, end = prompt_and_span(tokenizer, phrase_for_span)
    if debug_prompt is None:
        debug_prompt = SHOW_CHAT_PROMPT_DEBUG_SPAN_SWAP
    if debug_prompt:
        print_chat_prompt_debug(
            tokenizer,
            user_text,
            ids,
            label=(
                "LLM prompt before span embedding swap "
                "(chat template → input_ids; span replaced in inputs_embeds only)"
            ),
        )
    embeds = build_inputs_embeds_from_ids(embed_layer, ids, device, dtype)
    embeds = replace_span_with_vector(embeds, start, end, replacement)
    if pre_generate_prompt is None:
        pre_generate_prompt = SHOW_PRE_GENERATE_PROMPT
    if pre_generate_prompt:
        print_pre_generate_llm_input(tokenizer, ids, embeds, span=(start, end))
    return generate_from_embeds(
        model, tokenizer, embeds, max_new_tokens=max_new_tokens, stream=stream
    )


# %% Load Qwen3.5-2B (model + tokenizer)
device = pick_device()
dtype = pick_dtype(device)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=dtype if dtype != torch.float32 else None,
    device_map=None,
    trust_remote_code=True,
)
model.to(device)
model.eval()

embed_layer = model.get_input_embeddings()
hidden_size = embed_layer.weight.shape[1]

print(f"loaded {MODEL_NAME!r}")
print(f"device={device}, dtype={dtype}, hidden_size={hidden_size}")

# %% Explain a literal concept # stonesoup:cell-input
_raw = globals().get("CELL_INPUT")
if _raw is None:
    # ``uv run python …`` (whole file): same demo words as before
    _words = ["table", "king", "apple"]
else:
    w = _raw if isinstance(_raw, str) else str(_raw)
    # Keep leading/trailing spaces (tokenization); treat whitespace-only as empty.
    _words = [w] if w.strip() else []

if not _words:
    print("Enter a word or short phrase in the cell input, then Run.")
else:
    for word in _words:
        print("=" * 72)
        print(f'Concept in quote: "{word}"')
        print("=" * 72)
        text = explain_literal_phrase(model, tokenizer, embed_layer, word, device, dtype)
        if not STREAM_LLM_OUTPUT:
            print(text)
        print()

# %% Random embedding in the quote span # stonesoup:cell-input
_, _, span_start, span_end = prompt_and_span(tokenizer, SPAN_REPLACEMENT_DEMO_WORD)
span_len = span_end - span_start
print(
    f'Span for "{SPAN_REPLACEMENT_DEMO_WORD}" in prompt: '
    f"[{span_start}, {span_end})  ({span_len} tokens)\n"
)

_rng_seed = torch_seed_from_cell_input(globals().get("CELL_INPUT"))
torch.manual_seed(_rng_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(_rng_seed)
print(
    f"torch RNG seed={_rng_seed} (from CELL_INPUT: int, 0x… hex, or any string → sha256); "
    f"empty → 0\n"
)

rand_vec = torch.randn(hidden_size, device=device, dtype=torch.float32)
rand_vec = rand_vec * (embed_layer.weight.float().std(dim=0).mean().item())
rand_vec = rand_vec.to(dtype=dtype)

_rv_cpu = rand_vec.detach().float().cpu()
_preview_n = 8
_head = _rv_cpu[:_preview_n].tolist()
print("Random replacement embedding vector (1 × hidden_size, injected on the quoted span):")
print(
    f"  shape={tuple(rand_vec.shape)}, dtype={rand_vec.dtype}, device={rand_vec.device} | "
    f"min={_rv_cpu.min().item():.6g} max={_rv_cpu.max().item():.6g} "
    f"mean={_rv_cpu.mean().item():.6g} std={_rv_cpu.std(unbiased=False).item():.6g}"
)
print(f"  first {_preview_n} values: {_head}")
print()

print_top_k_token_neighbors_cosine(
    embed_layer,
    tokenizer,
    rand_vec,
    k=5,
    label="Top 5 vocabulary tokens closest to random vector (cosine vs embedding table rows)",
)
print()

print("Replacement: random Gaussian vector (scaled by mean per-dim std of embedding table)\n")
out_random = explain_with_span_replacement(
    model,
    tokenizer,
    embed_layer,
    SPAN_REPLACEMENT_DEMO_WORD,
    rand_vec,
    device,
    dtype,
)
if not STREAM_LLM_OUTPUT:
    print(out_random)

# %% Arithmetic embedding: (king + queen) / 2 in the quote span
king_vec = mean_phrase_embedding(embed_layer, tokenizer, " actor", device, dtype)
queen_vec = mean_phrase_embedding(embed_layer, tokenizer, " actress", device, dtype)
# mixed_vec = ((king_vec.float() + queen_vec.float()) / 1.0).to(dtype=dtype)
mixed_vec = (king_vec.float() * 5).to(dtype=dtype)

print('Replacement: (mean_embed("king") + mean_embed("queen")) / 2')
print(
    f'Span layout still matches token positions for literal "{SPAN_REPLACEMENT_DEMO_WORD}".\n'
)
out_mixed = explain_with_span_replacement(
    model,
    tokenizer,
    embed_layer,
    SPAN_REPLACEMENT_DEMO_WORD,
    mixed_vec,
    device,
    dtype,
)
if not STREAM_LLM_OUTPUT:
    print(out_mixed)

# %% 1000 random vectors: max cosine to any token embedding (histogram) # stonesoup:cell-input
import matplotlib.pyplot as plt

_N_SAMPLE_VECS = 1000
_hist_seed = torch_seed_from_cell_input(globals().get("CELL_INPUT"))
torch.manual_seed(_hist_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(_hist_seed)

_scale_w = embed_layer.weight.float().std(dim=0).mean().item()
_rand_batch = torch.randn(_N_SAMPLE_VECS, hidden_size, device=device, dtype=torch.float32) * _scale_w
_max_cos = max_vocab_cosine_similarities(embed_layer, _rand_batch)
_max_cos_np = _max_cos.detach().float().cpu().numpy()

_PLOTS_DIR = Path(__file__).resolve().parent / "plots"
_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
_plot_path = _PLOTS_DIR / "random_vec_max_vocab_cosine_hist.png"

_fig, _ax = plt.subplots(figsize=(10, 5.5))
_ax.hist(_max_cos_np, bins=20, color="steelblue", edgecolor="white", alpha=0.9)
_ax.set_xlabel("max cosine similarity to any token embedding row")
_ax.set_ylabel("count")
_ax.set_title(
    f"{_N_SAMPLE_VECS} Gaussian vectors (× mean embedding std) · {MODEL_NAME}\n"
    f"seed={_hist_seed}"
)
_ax.grid(True, alpha=0.3)
_fig.tight_layout()
_fig.savefig(_plot_path, dpi=EMBED_EXPERIMENT_PLOT_DPI)
plt.close(_fig)
print(f"wrote {_plot_path}")
print(
    f"  max-cos: mean={_max_cos_np.mean():.4f}  std={_max_cos_np.std():.4f}  "
    f"min={_max_cos_np.min():.4f}  max={_max_cos_np.max():.4f}"
)

# %% UMAP (cosine) of “popular” whole-word (single-token) embeddings # stonesoup:cell-input
# Only words from the list that encode to exactly one token; multi-token words ignored.
# Requires: ``uv pip install umap-learn`` or ``uv pip install -e ".[embedding]"``.
import matplotlib.pyplot as plt

try:
    import umap
except ImportError as _e:
    raise ImportError(
        'Install umap-learn for this cell, e.g. ``uv pip install umap-learn`` '
        'or ``uv pip install -e ".[embedding]"``.'
    ) from _e

_GOOGLE10K = REPO_ROOT / "data/text/google_10000_english.txt"
_UMAP_TOP_N = 1500
_umap_seed = int(torch_seed_from_cell_input(globals().get("CELL_INPUT")) % (2**31 - 2))
if _umap_seed <= 0:
    _umap_seed = 42

_umap_ctr, _umap_wstats = weighted_single_token_word_counts_from_ranked_wordlist(
    _GOOGLE10K, tokenizer
)
print(
    f"UMAP word filter: {_umap_wstats['single_token_words']} single-token / "
    f"{_umap_wstats['lines']} list lines "
    f"({_umap_wstats['skipped_multi']} skipped as multi-token)\n"
)
for _sid in (
    tokenizer.pad_token_id,
    tokenizer.eos_token_id,
    tokenizer.bos_token_id,
    tokenizer.unk_token_id,
):
    if _sid is not None:
        _umap_ctr.pop(int(_sid), None)

_vocab_sz = int(embed_layer.weight.shape[0])
_umap_pairs: list[tuple[int, int]] = []
for _tid, _c in _umap_ctr.most_common():
    if _tid < _vocab_sz:
        _umap_pairs.append((_tid, _c))
    if len(_umap_pairs) >= _UMAP_TOP_N:
        break

if len(_umap_pairs) < 3:
    print(f"UMAP skipped: need ≥3 token ids, got {len(_umap_pairs)} (check {_GOOGLE10K}).")
else:
    _umap_tids = [p[0] for p in _umap_pairs]
    _Xw = (
        embed_layer.weight[torch.tensor(_umap_tids, device=device, dtype=torch.long)]
        .detach()
        .float()
        .cpu()
        .numpy()
    )
    _nn = max(2, min(30, len(_umap_tids) - 1))
    print(f"UMAP n_neighbors={_nn} (max 30, min 2)")
    _reducer = umap.UMAP(
        n_components=2,
        metric="cosine",
        random_state=_umap_seed,
        n_neighbors=_nn,
        min_dist=0.1,
        verbose=False,
    )
    _XY = _reducer.fit_transform(_Xw)

    _fig_u, _ax_u = plt.subplots(figsize=UMAP_FIGSIZE_INCHES)
    _xw = float(_XY[:, 0].max() - _XY[:, 0].min()) + 1e-9
    _yh = float(_XY[:, 1].max() - _XY[:, 1].min()) + 1e-9
    _pad_x, _pad_y = 0.06 * _xw, 0.06 * _yh
    _ax_u.set_xlim(float(_XY[:, 0].min()) - _pad_x, float(_XY[:, 0].max()) + _pad_x)
    _ax_u.set_ylim(float(_XY[:, 1].min()) - _pad_y, float(_XY[:, 1].max()) + _pad_y)
    _ax_u.set_aspect("equal", adjustable="box")

    _n_lab = len(_umap_tids)
    _fs = max(4.0, min(8.0, 520.0 / max(_n_lab, 1)))
    for _i, _tid in enumerate(_umap_tids):
        _word = tokenizer.decode([_tid], skip_special_tokens=True).strip()
        _word = _word.strip("'\"").strip("“”‘’")
        if not _word:
            continue
        _ax_u.text(
            float(_XY[_i, 0]),
            float(_XY[_i, 1]),
            _word,
            fontsize=_fs,
            ha="center",
            va="center",
            color="black",
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": "white",
                "alpha": 0.5,
                "edgecolor": "none",
            },
        )

    _ax_u.set_title(
        f"UMAP (cosine) · single-token words (labels only, no markers)\n"
        f"{_GOOGLE10K.name} · multi-token list words excluded · {MODEL_NAME}\n"
        f"seed={_umap_seed}"
    )
    _ax_u.set_xlabel("UMAP-1")
    _ax_u.set_ylabel("UMAP-2")

    _fig_u.tight_layout()
    _umap_plot_path = Path(__file__).resolve().parent / "plots" / "umap_popular_tokens_cosine.png"
    _umap_plot_path.parent.mkdir(parents=True, exist_ok=True)
    _fig_u.savefig(_umap_plot_path, dpi=EMBED_EXPERIMENT_PLOT_DPI)
    plt.close(_fig_u)
    print(f"wrote {_umap_plot_path}")
    print(f"  points={len(_umap_tids)}  n_neighbors={_nn}  wordlist={_GOOGLE10K.name}")

# %% Per-token mean cosine similarity to all *other* tokens (histogram)
import matplotlib.pyplot as plt

with torch.inference_mode():
    _mean_cos_others = mean_cosine_to_other_tokens(embed_layer)
_mean_cos_np = _mean_cos_others.float().cpu().numpy()
_vocab_n = int(_mean_cos_np.shape[0])

_fig_mc, _ax_mc = plt.subplots(figsize=(10, 5.5))
_ax_mc.hist(
    _mean_cos_np,
    bins=80,
    color="teal",
    edgecolor="white",
    alpha=0.9,
)
_ax_mc.set_xlabel("mean cosine similarity to all other token embeddings")
_ax_mc.set_ylabel("count (tokens)")
_ax_mc.set_title(
    f"Distribution of mean pairwise cosine (excluding self) · {MODEL_NAME}\n"
    f"vocab_size={_vocab_n}"
)
_ax_mc.grid(True, alpha=0.3)
_fig_mc.tight_layout()
_mean_cos_plot_path = Path(__file__).resolve().parent / "plots" / "mean_cosine_to_other_tokens_hist.png"
_mean_cos_plot_path.parent.mkdir(parents=True, exist_ok=True)
_fig_mc.savefig(_mean_cos_plot_path, dpi=EMBED_EXPERIMENT_PLOT_DPI)
plt.close(_fig_mc)
print(f"wrote {_mean_cos_plot_path}")
print(
    f"  mean-of-means={_mean_cos_np.mean():.6f}  std={_mean_cos_np.std():.6f}  "
    f"min={_mean_cos_np.min():.6f}  max={_mean_cos_np.max():.6f}"
)
