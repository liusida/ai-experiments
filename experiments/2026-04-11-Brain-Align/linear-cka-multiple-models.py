# %% Imports & helpers
from __future__ import annotations

import math
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import stonesoup
import torch
from ckatorch import cka_base
from stonesoup.experiment import (
    capture_embed_and_post_blocks,
    configure_matplotlib_agg,
    decoder_blocks,
    ensure_pad_token_via_eos,
    hf_repo_id_safe_stem,
    inner_tokenizer,
)

configure_matplotlib_agg()


def _assert_all_finite(t: torch.Tensor, *, what: str) -> None:
    assert bool(torch.isfinite(t).all().item()), f"non-finite values in {what}"


def _assert_finite_float(x: float, *, what: str) -> None:
    assert math.isfinite(x), f"non-finite float in {what}: {x}"


def token_span_to_index_from_proc(
    proc: Any,
    sentence: str,
    *,
    max_length: int,
    skip_first_tokens: int,
) -> dict[tuple[int, int], int]:
    """Map ``(char_start, char_end)`` in ``sentence`` → token index (fast tokenizer offsets)."""
    tok = inner_tokenizer(proc)
    ensure_pad_token_via_eos(tok)
    enc = tok(
        sentence,
        return_offsets_mapping=True,
        return_tensors="pt",
        add_special_tokens=True,
        max_length=max_length,
        truncation=True,
    )
    if enc["offset_mapping"] is None:
        raise ValueError("Tokenizer must support return_offsets_mapping (fast tokenizer).")
    om = enc["offset_mapping"][0]
    seq_len = int(enc["attention_mask"][0].sum().item())
    out: dict[tuple[int, int], int] = {}
    for t in range(skip_first_tokens, seq_len):
        row = om[t]
        a, b = int(row[0]), int(row[1])
        if b <= a:
            continue
        out[(a, b)] = t
    return out


def aligned_token_indices_per_sentence(
    repo_ids: list[str],
    sentences: list[str],
    *,
    max_length: int,
    skip_first_tokens: int,
) -> tuple[dict[str, list[list[int]]], list[int]]:
    """For each repo and sentence, token indices whose **character span in ``sentence``** is shared by all repos.

    Returns ``(indices[repo_id][sent_idx], n_aligned_per_sentence)``.
    """
    per_repo: dict[str, list[list[int]]] = {r: [] for r in repo_ids}
    n_aligned: list[int] = []
    proc_by_repo: dict[str, Any] = {}

    def proc_for(rid: str) -> Any:
        if rid not in proc_by_repo:
            _m, proc_by_repo[rid] = stonesoup.load_model(rid)
        return proc_by_repo[rid]

    for si, sentence in enumerate(sentences):
        stonesoup.check_abort()
        span_maps: dict[str, dict[tuple[int, int], int]] = {}
        key_sets: list[set[tuple[int, int]]] = []
        for rid in repo_ids:
            sm = token_span_to_index_from_proc(
                proc_for(rid),
                sentence,
                max_length=max_length,
                skip_first_tokens=skip_first_tokens,
            )
            span_maps[rid] = sm
            key_sets.append(set(sm.keys()))
        common = set.intersection(*key_sets) if key_sets else set()
        order = sorted(common, key=lambda x: (x[0], x[1]))
        n_aligned.append(len(order))
        if not order:
            print(
                f"  [align] sentence {si}: no common char spans across all models — skipping.",
                flush=True,
            )
            for rid in repo_ids:
                per_repo[rid].append([])
            continue
        for rid in repo_ids:
            per_repo[rid].append([span_maps[rid][sp] for sp in order])
    return per_repo, n_aligned


def print_all_tokens_per_model_and_sentence(
    repo_ids: list[str],
    sentences: list[str],
    *,
    max_length: int,
) -> None:
    """Log tokenizer output for each (sentence, model): same plain text path as alignment/forward."""
    proc_by_repo: dict[str, Any] = {}

    def proc_for(rid: str) -> Any:
        if rid not in proc_by_repo:
            _m, proc_by_repo[rid] = stonesoup.load_model(rid)
        return proc_by_repo[rid]

    print(
        "\n[tokens] per-sentence token lists (plain ``sentence`` encode; ``stonesoup.load_model``)…",
        flush=True,
    )
    for si, sentence in enumerate(sentences):
        stonesoup.check_abort()
        prev = sentence if len(sentence) <= 300 else sentence[:300] + "…"
        print(f"\n[tokens] sentence {si}  (n_chars={len(sentence)}): {prev!r}", flush=True)
        for rid in repo_ids:
            stonesoup.check_abort()
            short = rid.split("/")[-1]
            proc = proc_for(rid)
            tok = inner_tokenizer(proc)
            ensure_pad_token_via_eos(tok)
            enc = tok(
                sentence,
                return_tensors="pt",
                add_special_tokens=True,
                max_length=max_length,
                truncation=True,
            )
            ids = enc["input_ids"][0].tolist()
            toks = tok.convert_ids_to_tokens(ids)
            print(f"  {short}  (n_tokens={len(toks)}): {toks}", flush=True)
    print("", flush=True)


def _forward_tok_stage_dim(
    model: Any,
    proc: Any,
    device: torch.device,
    sentence: str,
    *,
    max_length: int,
) -> tuple[torch.Tensor, list[str], int]:
    """Full valid sequence: ``tok_stage_dim`` ``(sl, n_stages, dim)`` — same tokenization as span alignment."""
    tok = inner_tokenizer(proc)
    ensure_pad_token_via_eos(tok)
    enc = tok(
        sentence,
        return_tensors="pt",
        return_attention_mask=True,
        add_special_tokens=True,
        max_length=max_length,
        truncation=True,
    )
    inputs = {k: v.to(device) for k, v in enc.items()}
    stack, stage_names = capture_embed_and_post_blocks(model, inputs, use_cache=False)
    seq_len = int(inputs["attention_mask"][0].sum().item())
    sl = int(min(seq_len, stack.shape[2]))
    st = stack[:, 0, :sl, :].detach().float()
    tok_stage_dim = st.permute(1, 0, 2).contiguous()
    return tok_stage_dim, stage_names, sl


def collect_aligned_token_activations_for_model(
    model: Any,
    proc: Any,
    device: torch.device,
    sentences: list[str],
    token_indices_per_sentence: list[list[int]],
    *,
    max_length: int,
) -> tuple[torch.Tensor, list[str]]:
    """Gather hidden states only at **aligned** token indices (same char span in ``sentence``)."""
    chunks: list[torch.Tensor] = []
    stage_names: list[str] | None = None
    for sentence, idxs in zip(sentences, token_indices_per_sentence, strict=True):
        if not idxs:
            continue
        stonesoup.check_abort()
        print(f"[aligned positions] {idxs!r}", flush=True)
        # _dt = inner_tokenizer(proc)
        # ensure_pad_token_via_eos(_dt)
        # _row = _dt(
        #     sentence,
        #     return_tensors="pt",
        #     add_special_tokens=True,
        #     max_length=max_length,
        #     truncation=True,
        # )["input_ids"][0].tolist()
        # _ids_aln = [_row[i] for i in idxs]
        # # Per-id decode shows real Unicode; raw ``convert_ids_to_tokens`` is often byte-level (garbled CJK).
        # print(
        #     f"[aligned subwords] {_dt.batch_decode([[t] for t in _ids_aln], skip_special_tokens=False)}",
        #     flush=True,
        # )
        # print(f"[aligned text] {_dt.decode(_ids_aln, skip_special_tokens=False)}", flush=True)
        tsd, stage_names, _sl = _forward_tok_stage_dim(
            model,
            proc,
            device,
            sentence,
            max_length=max_length,
        )
        idx_t = torch.as_tensor(idxs, device=tsd.device, dtype=torch.long)
        if int(idx_t.max().item()) >= tsd.shape[0]:
            raise RuntimeError(
                "Aligned token index out of range for this forward (check max_length / truncation)."
            )
        chunks.append(tsd.index_select(0, idx_t))
    if not chunks:
        raise ValueError(
            "No aligned tokens collected (empty intersection for every sentence). "
            "Shorten prompts, relax skip_first_tokens, or use a more compatible model set."
        )
    assert stage_names is not None
    out = torch.cat(chunks, dim=0)
    _assert_all_finite(out, what="collect_aligned_token_activations_for_model")
    return out, stage_names


def pairwise_linear_cka_block_matrix(
    models_order: list[str],
    acts_by_repo: dict[str, torch.Tensor],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, list[int], list[str]]:
    """Full matrix over **all** stages of **all** models (embedding + layers).

    ``acts_by_repo[repo]`` has shape ``(n_tokens, n_stages, dim)`` with **identical** ``n_tokens``
    for every repo (char-span–aligned tokens). Off-diagonal blocks compare different hidden widths;
    ``cka_base`` needs the same row count ``n``.
    """
    layer_counts = [acts_by_repo[m].shape[1] for m in models_order]
    offsets = [0] + list(np.cumsum(layer_counts).tolist())
    total = int(offsets[-1])
    mat = torch.eye(total, dtype=torch.float64, device=device)
    stems = [hf_repo_id_safe_stem(m) for m in models_order]
    labels: list[str] = []
    for mi, repo in enumerate(models_order):
        for li in range(layer_counts[mi]):
            labels.append(f"{stems[mi]} | L{li}")

    n0 = acts_by_repo[models_order[0]].shape[0]
    for m in models_order[1:]:
        assert acts_by_repo[m].shape[0] == n0, (
            "aligned activations must share n_tokens across models; "
            f"got {models_order[0]}={n0} vs {m}={acts_by_repo[m].shape[0]}"
        )

    cka_base_calls = 0
    for mi in range(len(models_order)):
        Ai = acts_by_repo[models_order[mi]].detach().to(dtype=torch.float64)
        for mj in range(mi, len(models_order)):
            print(f"[cka] {models_order[mi]} vs {models_order[mj]}", flush=True)
            Aj = acts_by_repo[models_order[mj]].detach().to(dtype=torch.float64)
            li_max, lj_max = layer_counts[mi], layer_counts[mj]
            for ii in range(li_max):
                stonesoup.check_abort()
                xi = Ai[:, ii, :]
                gi = offsets[mi] + ii
                jj_lo = ii + 1 if mi == mj else 0
                for jj in range(jj_lo, lj_max):
                    xj = Aj[:, jj, :]
                    gj = offsets[mj] + jj
                    v = cka_base(xi, xj, unbiased=True)
                    cka_base_calls += 1
                    assert bool(torch.isfinite(v).item()), f"non-finite CKA ({mi},{ii}) vs ({mj},{jj})"
                    mat[gi, gj] = v
                    mat[gj, gi] = v
    print(
        f"pairwise_linear_cka_block_matrix: cka_base() calls = {cka_base_calls} "
        f"(distinct stage pairs; matrix is {total}×{total})",
        flush=True,
    )
    _assert_all_finite(mat, what="pairwise_linear_cka_block_matrix")
    return mat, layer_counts, labels


def n_stages_embed_and_post_blocks(model: Any) -> int:
    """Same stage count as :func:`capture_embed_and_post_blocks` (embedding + post each block)."""
    return 1 + len(decoder_blocks(model))


def _decorate_block_heatmap(
    ax: plt.Axes,
    mat: torch.Tensor,
    layer_counts: list[int],
    stems_short: list[str],
    *,
    vmin: float,
    vmax: float,
    title: str,
    title_fontsize: float = 14,
    tick_fontsize: float = 11,
    label_fontsize: float = 11,
) -> tuple[plt.Axes, Any]:
    arr = mat.detach().cpu().numpy()
    im = ax.imshow(
        arr,
        vmin=vmin,
        vmax=vmax,
        cmap="Blues",
        aspect="equal",
        interpolation="nearest",
    )
    ax.set_title(title, fontsize=title_fontsize)
    n_models = len(layer_counts)
    bounds = [0]
    for c in layer_counts:
        bounds.append(bounds[-1] + c)
    for b in bounds[1:-1]:
        ax.axhline(b - 0.5, color="#333", linewidth=2)
        ax.axvline(b - 0.5, color="#333", linewidth=2)
    tick_pos = [(bounds[i] + bounds[i + 1]) / 2 for i in range(n_models)]
    ax.set_xticks(tick_pos)
    ax.set_yticks(tick_pos)
    ax.set_xticklabels(stems_short, rotation=35, ha="right", fontsize=tick_fontsize)
    ax.set_yticklabels(stems_short, fontsize=tick_fontsize)
    # ax.set_xlabel("stage index (global block columns)", fontsize=label_fontsize)
    # ax.set_ylabel("stage index (global block rows)", fontsize=label_fontsize)
    return ax, im


# %% Models & sentences
# Plain text on ``SENTENCES`` only (no chat template). Prefer causal LMs; for dual-purpose Hub repos,
# load via Stonesoup with ``model_kind=causal_lm`` if the default class is vision+text.
MODELS: list[str] = [
    # "Qwen/Qwen2-1.5B",
    # "Qwen/Qwen2.5-3B",
    # # "Qwen/Qwen2.5-VL-7B-Instruct",
    # # "Qwen/Qwen3-0.6B",
    # # "Qwen/Qwen3-0.6B-Base",
    # "Qwen/Qwen3-8B",
    # # "Qwen/Qwen3-8B-Base",
    # # "Qwen/Qwen3.5-0.8B",
    # # "Qwen/Qwen3.5-2B",
    # "Qwen/Qwen3.5-4B",
    # "google/gemma-2-2b",
    # "mistralai/Ministral-3-3B-Base-2512",
    # "meta-llama/llama-3.2-3B",
    "openai-community/gpt2-medium",
    "Qwen/Qwen3.5-9B",
    "tiiuae/falcon-7b",
]

# Twenty short passages: mixed genres, registers, and languages (non-sink tokens concatenated).
SENTENCES: list[str] = [
    # Expository English (science)
    "Chloroplasts use chlorophyll to absorb photons and store energy in ATP and NADPH.",
    # Spoken dialogue
    '"Did you remember the keys?"\n"On the hook—unless the cat knocked them down again."',
    # Chinese (informative)
    "月球围绕地球公转，同一面始终朝向地球；潮汐主要由月球引力引起。",
    # Children’s storybook tone
    "The little boat wished for wings, so the wind stitched clouds into sails and pushed it upstream.",
    # Statutory / legal style
    "Where a party fails to perform without excuse, the non-breaching party may seek damages as provided herein.",
    # Recipe / procedural
    "Whisk eggs with salt, fold in warm rice off the heat, then sprinkle nori without over-stirring.",
    # Poetry-ish (line breaks as in source)
    "Fog on the pier—\nA gull borrows the moon\nAnd flies away.",
    # Kid-friendly fable
    "The fox promised grapes were sour anyway, but the crow still laughed from the high branch.",
    # News headline + lead
    "City council delays vote: residents packed the hall, some holding signs that read “Fix the pipes first.”",
    # Text-message / informal
    "omw — grab a table near the window?? coffee’s on me if traffic eats me alive lol",
    # Second Chinese (colloquial narrative)
    "周末我想去爬山，如果下雨就改在家里煮火锅、看电影。",
    # Technical / spec tone
    "Requirement: latency p99 under 120 ms; fallback path must degrade gracefully without data loss.",
    # Courtroom dialogue
    "Your Honor, the exhibit is authenticated under Rule 902—the chain of custody is unbroken.",
    # Sports play-by-play
    "She fakes left, splits two defenders, and curls one into the top corner—stadium erupts.",
    # Academic philosophy (dense)
    "Normative claims concern what ought to be; descriptive claims concern what is—confusing them risks the is-ought gap.",
    # Product blurb / marketing
    "This jacket repels drizzle, packs into its pocket, and weighs less than your phone—trail-tested.",
    # Medical chart note style
    "Patient reports intermittent vertigo; differential includes BPPV versus orthostatic hypotension.",
    # Email closings / formal
    "Please find the revised figures attached. I remain available for a brief call next Tuesday.",
    # Myth / epic register
    "When the river refused the oath, the old king broke his crown and scattered the shards downstream.",
    # Code-adjacent comment (natural language)
    "# TODO: replace O(n^2) pairing with hash map once we confirm key distribution in prod logs.",
]

MAX_LENGTH = 128
SKIP_FIRST_TOKENS = 1

# If True: load each checkpoint (same as full run) and read stage counts from the loaded module
# graph — no text forward, no activations, no CKA — only a zero matrix for layout/size preview.
DRY_RUN = False

print(f"{len(SENTENCES)} sentences, max_length={MAX_LENGTH}, skip_first_tokens={SKIP_FIRST_TOKENS}")
print(f"DRY_RUN={DRY_RUN}", flush=True)

# %% Token preview — all subword tokens per sentence per model (tokenizer only; no `load_model`)
print_all_tokens_per_model_and_sentence(
    MODELS,
    SENTENCES,
    max_length=MAX_LENGTH,
)

# %% Collect per-token activations (char-span aligned across models; one model at a time)
acts_by_repo: dict[str, torch.Tensor] = {}
stage_names_by_repo: dict[str, list[str]] = {}
layer_counts_preview: dict[str, int] = {}
align_by_repo: dict[str, list[list[int]]] = {}

if DRY_RUN:
    layer_counts = []
    for mi, repo_id in enumerate(MODELS):
        stonesoup.check_abort()
        print(f"[{mi + 1}/{len(MODELS)}] (dry) load only: {repo_id!r}", flush=True)
        model, proc = stonesoup.load_model(repo_id)
        model.eval()
        device = next(model.parameters()).device
        n_st = n_stages_embed_and_post_blocks(model)
        layer_counts.append(n_st)
        layer_counts_preview[repo_id] = n_st
        print(
            f"  → device={device}  n_stages={n_st}  (1 embed + {n_st - 1} decoder blocks)",
            flush=True,
        )
else:
    print(
        "[align] token indices that share the same (char_start, char_end) in each raw sentence…",
        flush=True,
    )
    align_by_repo, n_aligned_per_sent = aligned_token_indices_per_sentence(
        MODELS,
        SENTENCES,
        max_length=MAX_LENGTH,
        skip_first_tokens=SKIP_FIRST_TOKENS,
    )
    _per_repo_total = sum(len(align_by_repo[MODELS[0]][si]) for si in range(len(SENTENCES)))
    print(
        f"[align] common spans per sentence: min={min(n_aligned_per_sent)}, max={max(n_aligned_per_sent)}; "
        f"concat length (same every repo)={_per_repo_total}",
        flush=True,
    )
    for mi, repo_id in enumerate(MODELS):
        stonesoup.check_abort()
        print(f"[{mi + 1}/{len(MODELS)}] loading + forward: {repo_id!r}", flush=True)
        model, proc = stonesoup.load_model(repo_id)
        model.eval()
        ensure_pad_token_via_eos(inner_tokenizer(proc))
        device = next(model.parameters()).device
        act, stage_names = collect_aligned_token_activations_for_model(
            model,
            proc,
            device,
            SENTENCES,
            align_by_repo[repo_id],
            max_length=MAX_LENGTH,
        )
        acts_by_repo[repo_id] = act
        stage_names_by_repo[repo_id] = stage_names
        layer_counts_preview[repo_id] = int(act.shape[1])
        print(
            f"  → device={device}  n_tokens={act.shape[0]}  stages={act.shape[1]}  dim={act.shape[2]}  "
            f"names {stage_names[:2]} … {stage_names[-1:]}",
            flush=True,
        )

# %% Pairwise linear CKA (all stages × all models)
if DRY_RUN:
    total = int(sum(layer_counts))
    cka_big = torch.zeros((total, total), dtype=torch.float64)
    print(
        f"DRY_RUN: skipped CKA; zero matrix shape {tuple(cka_big.shape)} "
        f"(sum(layer_counts)={total})",
        flush=True,
    )
else:
    _device = next(iter(acts_by_repo.values())).device
    cka_big, layer_counts, _stage_labels = pairwise_linear_cka_block_matrix(
        MODELS,
        acts_by_repo,
        device=_device,
    )
    print("CKA matrix shape:", tuple(cka_big.shape), "n_stages_total:", cka_big.shape[0])
    print(
        "n_tokens (char-span aligned, per repo): "
        + ", ".join(f"{m.split('/')[-1]}={acts_by_repo[m].shape[0]}" for m in MODELS),
        flush=True,
    )

# %% Heatmap (saved under outputs/… for the web UI)
if DRY_RUN:
    vmin, vmax = 0.0, 1.0
    _heat_title = (
        f"DRY RUN — zero matrix (layout only); models loaded for true stage counts; "
        f"n_stages_total={cka_big.shape[0]}×{cka_big.shape[1]}. "
        "Set DRY_RUN = False for forwards + CKA."
    )
    _heat_basename = "linear_cka_multi_model_all_stages_dry_run"
    _cb_label = "placeholder (zeros)"
else:
    ck_lo = float(torch.min(cka_big).item())
    ck_hi = float(torch.max(cka_big).item())
    _assert_finite_float(ck_lo, what="ck_lo")
    _assert_finite_float(ck_hi, what="ck_hi")
    ck_pad = 0.02 * (ck_hi - ck_lo + 1e-9)
    vmin, vmax = ck_lo - ck_pad, ck_hi + ck_pad
    _heat_title = (
        "Linear CKA (unbiased): pairwise stages — one row per char-span–aligned token "
        f"(intersection of tokenizer offset spans in each sentence); sink skip={SKIP_FIRST_TOKENS}; "
        f"n_sentences={len(SENTENCES)}"
    )
    _heat_basename = "linear_cka_multi_model_all_stages"
    _cb_label = "linear CKA"

stems_short = [s.split("__")[-1] if "__" in s else s for s in [hf_repo_id_safe_stem(m) for m in MODELS]]
n_total = int(cka_big.shape[0])
fig_w = min(28, 6 + n_total * 0.12)
fig_h = min(26, 5 + n_total * 0.11)
# Larger figures → larger type (defaults were too small for wide heatmaps).
_fs_tick = float(max(10.0, min(20.0, 5.5 + fig_w * 0.55)))
_fs_title = float(_fs_tick + 3.0)
_fs_label = float(max(9.0, _fs_tick - 0.5))
fig, ax = plt.subplots(figsize=(fig_w, fig_h))
_, mappable = _decorate_block_heatmap(
    ax,
    cka_big,
    layer_counts,
    stems_short,
    vmin=0.6,
    vmax=vmax,
    title="",
    title_fontsize=_fs_title,
    tick_fontsize=_fs_tick,
    label_fontsize=_fs_label,
)
_cb = fig.colorbar(mappable, ax=ax, fraction=0.025, pad=0.02)
_cb.set_label(_cb_label, fontsize=_fs_label)
_cb.ax.tick_params(labelsize=_fs_label)
fig.tight_layout()
stonesoup.show(fig, basename=_heat_basename, dpi=144)
