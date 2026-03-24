# %% Imports & helpers
"""Mean cosine to other tokens for every **disk-cached** input embedding in ``data/embedding-layers/``.

For each name in ``MULTI_MODEL_INVESTIGATION_NAMES`` that has a matching ``*.pt`` cache,
load **only** the cached weight matrix, compute per-token mean cosine to all **other** rows,
and plot one histogram per model in a grid (single PNG, **shared** x-axis across models).
Each panel uses **twin y-axes**: **orange** (left): histogram of **cos(z, w_j)** for random unit **z** and vocab rows **j**
(full **N×V** grid when ``N·V`` is below a cap; otherwise an i.i.d. **(z, j)** sample of comparable size to the blue curve).
**Blue** (right): sampled **pairwise** row cosines (uniform random ordered pairs). **Per-row mean** cosines are still computed for the optional top-token table. No full ``AutoModelForCausalLM`` load.

**Stonesoup:** Watch this file and run cells **in order** (shared kernel; **Reset** if you
change imports). Collect → plot → optional **top tokens** cell (set ``TOP_MEAN_COS_DECODE_K = 20`` there;
default 0 skips so ``uv run`` on the whole file stays tokenizer-free). Re-run collect after cache changes.

**CLI (whole file in one process):**

    uv run python experiments/2026-03-24-Explain-Embedding/mean_cosine_distribution_cached_models.py

Large vocabs × hidden matmul per model — needs enough RAM/VRAM for one embedding matrix
at a time. **Memory:** pairwise cosines are gathered in chunks (``MEAN_COS_PAIRWISE_CHUNK``);
PyTorch CUDA fraction defaults to **0.8** (Stonesoup kernel + this script); override with
``STONESOUP_CUDA_MEMORY_FRACTION`` / ``AI_TORCH_CUDA_MEMORY_FRACTION``. ``MEAN_COS_MEM_HEADROOM_GIB``,
``MEAN_COS_FORCE_PAIRWISE_N`` tune pairwise auto-caps.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import TypeAlias

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
THIS_DIR = Path(__file__).resolve().parent
EMBEDDING_LAYER_CACHE_DIR = REPO_ROOT / "data" / "embedding-layers"

PLOT_DPI = 200

# Same numeric default as ``stonesoup/backend/kernel.py`` — kept here because ``uv run`` / CLI
# does not import the kernel; only ``collect_cached_mean_cosine_series`` applies it in that case.
_DEFAULT_TORCH_CUDA_MEMORY_FRACTION = 0.8

# Grid layout: one panel per row (full-width histogram).
GRID_NCOLS = 1
SUBPLOT_W_INCH = 11.0
SUBPLOT_H_INCH = 2.85
HIST_BINS = 100

# Random unit directions z: orange histogram uses every cos(z_i, w_j) while n_directions * V ≤ cap; else i.i.d. sample.
RANDOM_BASELINE_N = 100000
RANDOM_BASELINE_SEED = 42

# Full V×V pairwise cosines are too heavy; uniform sample of ordered pairs (i≠j) for the blue hist.
PAIRWISE_COS_SAMPLE_N = 2_000_000
PAIRWISE_COS_SEED = 12345
RANDOM_ORANGE_FULL_GRID_MAX = int(os.environ.get("MEAN_COS_ORANGE_FULL_GRID_MAX", "4000000"))
RANDOM_ORANGE_IID_SAMPLE_N = int(os.environ.get("MEAN_COS_ORANGE_IID_N", str(PAIRWISE_COS_SAMPLE_N)))
# Gather (n_samples, d) in chunks — large n_samples at once OOMs unified memory.
PAIRWISE_COS_CHUNK = int(os.environ.get("MEAN_COS_PAIRWISE_CHUNK", "65536"))
# Leave headroom vs Linux MemAvailable when auto-capping pairwise count (GiB).
MEM_HEADROOM_GIB = float(os.environ.get("MEAN_COS_MEM_HEADROOM_GIB", "10"))

MeanCosResult: TypeAlias = tuple[str, np.ndarray, int, int, np.ndarray, np.ndarray]
# name, mean_cos→others (V,), V, d, random-vs-vocab cos sample (length varies), pairwise cos sample (length ≤ target N)

MULTI_MODEL_INVESTIGATION_NAMES: tuple[str, ...] = (
    # GPT / OPT / BLOOM (decoder classics)
    # "distilgpt2",
    "openai-community/gpt2-medium",
    # "openai-community/gpt2-large",
    # "openai-community/gpt2-xl",
    # "facebook/opt-1.3b",
    # "facebook/opt-2.7b",
    # "bigscience/bloom-560m",
    # "bigscience/bloom-1b7",
    # # EleutherAI
    # "EleutherAI/gpt-neo-1.3B",
    "EleutherAI/pythia-1b",
    # # Small Llama-like / open LM
    # "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    # "openlm-research/open_llama_3b_v2",
    # "tiiuae/falcon-rw-1b",
    # "meta-llama/Llama-3.2-1B-Instruct",  # gated on HF
    # # StableLM / SmolLM
    # "stabilityai/stablelm-2-1_6b",
    # "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    # # Other research / instruct
    # "allenai/OLMo-1B-hf",
    # # Qwen2.5 / Qwen3 / Qwen3.5 (3.x may need a recent ``transformers`` release)
    # "Qwen/Qwen2.5-0.5B-Instruct",
    # "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen3-0.6B",
    # "Qwen/Qwen3-1.7B",
    # "Qwen/Qwen3-4B-Instruct-2507",
    # "Qwen/Qwen3.5-0.8B",
    "Qwen/Qwen3.5-2B",
    # "microsoft/phi-2",
    # # More open LMs at ~1–3B (comment out if load fails or OOM)
    # "togethercomputer/RedPajama-INCITE-Base-3B-v1",
    # "mosaicml/mpt-1b-redpajama-200b",
    # "cerebras/btlm-3b-8k-base",
    # "ibm-granite/granite-3.1-2b-instruct",
    # "apple/OpenELM-1_1B-Instruct",
    # "allenai/OLMo-2-0425-1B",
    # "bigcode/starcoder2-3b",  # code LM — embeddings often differ from natural text LMs
    # "meta-llama/Llama-3.2-3B-Instruct",  # gated on HF
    # "internlm/internlm2-1_8b",  # may need recent ``transformers``
)


def embedding_layer_cache_path(model_name: str) -> Path:
    safe = model_name.replace("/", "__")
    return EMBEDDING_LAYER_CACHE_DIR / f"{safe}.pt"


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _apply_cuda_mem_fraction_from_env() -> None:
    """Cap CUDA memory fraction (default 0.8); override via ``STONESOUP_CUDA_MEMORY_FRACTION`` / ``AI_TORCH_CUDA_MEMORY_FRACTION``."""
    raw = os.environ.get("STONESOUP_CUDA_MEMORY_FRACTION", "").strip()
    if not raw:
        raw = os.environ.get("AI_TORCH_CUDA_MEMORY_FRACTION", "").strip()
    try:
        frac = float(raw) if raw else _DEFAULT_TORCH_CUDA_MEMORY_FRACTION
        if not (0.05 <= frac <= 1.0):
            return
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(frac, 0)
    except Exception:
        pass


def _mem_available_bytes_linux() -> int | None:
    """Best-effort ``MemAvailable`` from ``/proc/meminfo`` (Linux)."""
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    parts = line.split()
                    return int(parts[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def _effective_pairwise_sample_n(
    v_tok: int,
    d_emb: int,
    n_requested: int,
) -> tuple[int, str]:
    """Lower pairwise count if free RAM looks tight (unified memory)."""
    override = os.environ.get("MEAN_COS_FORCE_PAIRWISE_N", "").strip()
    if override:
        try:
            return max(10_000, min(n_requested, int(override))), " (MEAN_COS_FORCE_PAIRWISE_N)"
        except ValueError:
            pass

    avail = _mem_available_bytes_linux()
    if avail is None:
        return n_requested, ""

    headroom = int(max(4.0, MEM_HEADROOM_GIB) * (1024**3))
    rough_mat = v_tok * d_emb * 4 * 3
    rough_chunk_peak = max(PAIRWISE_COS_CHUNK, 4096) * d_emb * 4 * 4
    need = rough_mat + rough_chunk_peak + headroom
    note = ""
    n_eff = n_requested
    if avail < need:
        if avail < need // 2:
            n_eff = min(n_eff, 200_000)
            note = f" (pairwise capped {n_eff:,}; MemAvailable low vs ~{need / (1024**3):.1f}GiB est.)"
        else:
            n_eff = min(n_eff, 600_000)
            note = f" (pairwise capped {n_eff:,}; MemAvailable tight)"
    return n_eff, note


def mean_cosine_to_other_rows(W: torch.Tensor) -> torch.Tensor:
    """``W`` float (V, d) on any device. Returns (V,) mean cos(i,j) over j != i."""
    W = W.float()
    Wn = W / (W.norm(dim=1, keepdim=True) + 1e-8)
    v = int(Wn.shape[0])
    row_sum = Wn @ Wn.sum(dim=0)
    return (row_sum - 1.0) / float(v - 1)


def sample_random_unit_vocab_cosines_iid(
    Wn: torch.Tensor,
    n_samples: int,
    generator: torch.Generator,
    *,
    chunk_size: int | None = None,
) -> torch.Tensor:
    """``n_samples`` values cos(z, Wn[j]) with z ~ isotropic unit vector and j ~ Uniform(V), independent each draw."""
    n, d = int(Wn.shape[0]), int(Wn.shape[1])
    if n_samples <= 0:
        return Wn.new_zeros((0,))
    ch = max(4096, int(chunk_size or PAIRWISE_COS_CHUNK))
    dev = Wn.device
    parts: list[torch.Tensor] = []
    done = 0
    while done < n_samples:
        bs = min(ch, n_samples - done)
        z = torch.randn(bs, d, dtype=torch.float32, generator=generator)
        z = z.to(device=dev, non_blocking=True)
        z = z / (z.norm(dim=1, keepdim=True) + 1e-8)
        j = torch.randint(0, n, (bs,), device="cpu", dtype=torch.long, generator=generator)
        j = j.to(dev, non_blocking=True)
        parts.append((z * Wn[j]).sum(dim=-1).detach())
        done += bs
    return torch.cat(parts, dim=0).cpu()


def random_unit_vocab_cosines_for_orange_hist(
    Wn: torch.Tensor,
    n_directions: int,
    generator: torch.Generator,
    *,
    full_grid_max: int,
    iid_sample_n: int,
    z_batch: int = 32,
) -> torch.Tensor:
    """Histogram input: all ``cos(z_i, w_j)`` for ``n_directions`` random units if ``n_directions * V <= full_grid_max``.

    Otherwise ``iid_sample_n`` draws of cos(z, Wn[j]) with fresh z and uniform j (same marginal per point as one cell of the grid).
    """
    v, d = int(Wn.shape[0]), int(Wn.shape[1])
    dev = Wn.device
    if v == 0 or int(n_directions) <= 0:
        return Wn.new_zeros((0,)).cpu()
    total = int(n_directions) * v
    if total <= full_grid_max and n_directions > 0:
        parts: list[torch.Tensor] = []
        rem = int(n_directions)
        while rem > 0:
            bz = min(z_batch, rem)
            z = torch.randn(bz, d, dtype=torch.float32, generator=generator)
            z = z.to(device=dev, non_blocking=True)
            z = z / (z.norm(dim=1, keepdim=True) + 1e-8)
            cos_block = (z @ Wn.T).reshape(-1)
            parts.append(cos_block.detach().cpu())
            rem -= bz
        return torch.cat(parts, dim=0).float()
    return sample_random_unit_vocab_cosines_iid(Wn, iid_sample_n, generator, chunk_size=PAIRWISE_COS_CHUNK)


def sample_pairwise_cosines_uniform_ordered(
    Wn: torch.Tensor,
    n_samples: int,
    generator: torch.Generator,
    *,
    chunk_size: int | None = None,
) -> torch.Tensor:
    """Cosine for ``n_samples`` i.i.d. uniform **ordered** pairs (i, j) with i != j; rows of ``Wn`` unit-norm.

    Processed in chunks so we never build a single ``(n_samples, d)`` tensor (avoids OOM on unified memory).
    """
    n = int(Wn.shape[0])
    if n < 2:
        return Wn.new_zeros((n_samples,))
    ch = max(4096, int(chunk_size or PAIRWISE_COS_CHUNK))
    high = n * (n - 1)
    parts: list[torch.Tensor] = []
    done = 0
    while done < n_samples:
        bs = min(ch, n_samples - done)
        k = torch.randint(
            0,
            high,
            (bs,),
            device="cpu",
            dtype=torch.long,
            generator=generator,
        )
        i = k // (n - 1)
        jr = k % (n - 1)
        j = jr + (jr >= i).long()
        i = i.to(Wn.device, non_blocking=True)
        j = j.to(Wn.device, non_blocking=True)
        parts.append((Wn[i] * Wn[j]).sum(dim=-1).detach())
        done += bs
    return torch.cat(parts, dim=0)


def cached_model_names_in_order() -> list[str]:
    out: list[str] = []
    for name in MULTI_MODEL_INVESTIGATION_NAMES:
        p = embedding_layer_cache_path(name)
        if p.is_file():
            out.append(name)
    return out


def load_cached_weights(model_name: str, device: torch.device) -> torch.Tensor:
    path = embedding_layer_cache_path(model_name)
    payload = torch.load(path, map_location=str(device), weights_only=False)
    w = payload["weight"].float()
    if w.dim() != 2:
        raise ValueError(f"{model_name}: expected 2d weight, got {w.shape}")
    return w


def collect_cached_mean_cosine_series(device: torch.device) -> list[MeanCosResult]:
    names = cached_model_names_in_order()
    if not names:
        print(
            "No cached embedding layers for MULTI_MODEL_INVESTIGATION_NAMES.",
            file=sys.stderr,
        )
        print(f"Expected under: {EMBEDDING_LAYER_CACHE_DIR}", file=sys.stderr)
        return []

    results: list[MeanCosResult] = []
    rng = torch.Generator(device="cpu")
    rng.manual_seed(RANDOM_BASELINE_SEED)
    rng_pw = torch.Generator(device="cpu")
    rng_pw.manual_seed(PAIRWISE_COS_SEED)
    _apply_cuda_mem_fraction_from_env()
    print(f"device={device}  models_with_cache={len(names)}")
    for name in names:
        print(f"  … {name}")
        try:
            W = load_cached_weights(name, device)
            v_tok, d_emb = int(W.shape[0]), int(W.shape[1])
            pw_n, pw_note = _effective_pairwise_sample_n(
                v_tok, d_emb, PAIRWISE_COS_SAMPLE_N
            )
            if pw_note:
                print(f"     {name.split('/')[-1]}{pw_note}", file=sys.stderr)
            with torch.inference_mode():
                w_f = W.float()
                Wn = w_f / (w_f.norm(dim=1, keepdim=True) + 1e-8)
                mc = mean_cosine_to_other_rows(W)
                rnd = random_unit_vocab_cosines_for_orange_hist(
                    Wn,
                    RANDOM_BASELINE_N,
                    rng,
                    full_grid_max=RANDOM_ORANGE_FULL_GRID_MAX,
                    iid_sample_n=RANDOM_ORANGE_IID_SAMPLE_N,
                )
                pw = sample_pairwise_cosines_uniform_ordered(
                    Wn, pw_n, rng_pw, chunk_size=PAIRWISE_COS_CHUNK
                )
            arr = mc.detach().float().cpu().numpy()
            rand_arr = rnd.detach().float().cpu().numpy()
            pairwise_samp = pw.detach().float().cpu().numpy()
            del W, mc, rnd, w_f, Wn, pw
            if device.type == "cuda":
                torch.cuda.empty_cache()
            results.append((name, arr, v_tok, d_emb, rand_arr, pairwise_samp))
        except Exception as exc:  # noqa: BLE001
            print(f"     skip ({exc!r})", file=sys.stderr)
            if device.type == "cuda":
                torch.cuda.empty_cache()
    results.sort(key=lambda t: (t[3], t[0]))
    return results


def _p1_p99(a: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(np.asarray(a).ravel(), [1.0, 99.0])
    return float(lo), float(hi)


def _p1_p99_concat(*arrays: np.ndarray) -> tuple[float, float]:
    combo = np.concatenate([np.asarray(x).ravel() for x in arrays])
    return _p1_p99(combo)


def _mean_cosine_grid_figure(
    results: list[MeanCosResult],
    *,
    suptitle: str,
) -> plt.Figure:
    """All panels share x-limits: union of per-model pairwise + orange random-vs-vocab 1–99% (+ mean-row for range)."""
    n = len(results)
    ncols = min(GRID_NCOLS, n)
    nrows = math.ceil(n / ncols)
    fig_w, fig_h = SUBPLOT_W_INCH * ncols, SUBPLOT_H_INCH * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)

    lows, highs = zip(
        *(_p1_p99_concat(a, r, p) for _, a, _, _, r, p in results),
    )
    xmin_g = min(lows)
    xmax_g = max(highs)
    span_g = xmax_g - xmin_g
    if span_g < 1e-6:
        span_g = 1e-4
    pad_g = 0.05 * span_g + 1e-5
    x_lo, x_hi = xmin_g - pad_g, xmax_g + pad_g

    for i, (name, arr, n_tokens, d_vec, rand_arr, pairwise_samp) in enumerate(results):
        r, c = divmod(i, ncols)
        ax = axes[r][c]
        px0, px1 = x_lo, x_hi

        # Bin over the full interval so bars are visible (same for both series).
        # Left: random draws; right: sampled pairwise cosines (same bin count scale ~ sample size).
        kw = dict(bins=HIST_BINS, range=(px0, px1))
        ax.hist(
            rand_arr,
            **kw,
            color="darkorange",
            edgecolor="white",
            alpha=0.55,
            label=f"cos(z, w_j), random z (n={len(rand_arr):,})",
        )
        ax.tick_params(axis="y", labelsize=7, colors="darkorange")
        ax.spines["left"].set_color("darkorange")
        ax.spines["left"].set_alpha(0.6)
        ax.grid(True, alpha=0.25)

        ax2 = ax.twinx()
        ax2.hist(
            pairwise_samp,
            **kw,
            color="steelblue",
            edgecolor="white",
            alpha=0.4,
            label=f"pairwise cos (n={len(pairwise_samp):,})",
        )
        ax2.grid(False)
        ax2.tick_params(axis="y", labelsize=7, colors="steelblue")
        ax2.spines["right"].set_color("steelblue")
        ax2.spines["right"].set_alpha(0.65)

        short = name.split("/")[-1]
        ax.set_title(
            f"{short}\nV={n_tokens:,} tokens · d={d_vec}",
            fontsize=11,
            color="black",
        )
        ax.set_xlim(px0, px1)
        ax2.set_xlim(px0, px1)
        ax.tick_params(axis="x", labelsize=7)
        if r == nrows - 1:
            ax.set_xlabel("cosine similarity", fontsize=7)
        if i == 0:
            ax.set_ylabel(
                f"count (orange, n={len(rand_arr):,})",
                fontsize=7,
                color="darkorange",
            )
            ax2.set_ylabel("count (pairwise sample)", fontsize=7, color="steelblue")
            h_rand, l_rand = ax.get_legend_handles_labels()
            h_row, l_row = ax2.get_legend_handles_labels()
            ax.legend(h_rand + h_row, l_rand + l_row, loc="upper right", fontsize=7)

    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].axis("off")

    fig.suptitle(suptitle, fontsize=13, y=1.01)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    return fig


def plot_mean_cosine_grid(
    results: list[MeanCosResult],
    *,
    out_path: Path | None = None,
) -> Path:
    """Write one PNG: shared x-axis (pairwise sample + random + mean-row 1–99% union + pad)."""
    if not results:
        raise ValueError("no results to plot")

    if out_path is None:
        out_path = THIS_DIR / "plots" / "mean_cosine_distribution_cached_models_grid.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = _mean_cosine_grid_figure(
        results,
        suptitle=(
            f"L2-normalized input embeddings\n"
            f"orange: cos(z, w_j) — full {RANDOM_BASELINE_N}×V grid if ≤{RANDOM_ORANGE_FULL_GRID_MAX:,} values, "
            f"else {RANDOM_ORANGE_IID_SAMPLE_N:,} i.i.d. (z,j) · "
            f"blue: ≤{PAIRWISE_COS_SAMPLE_N:,} random pairs i≠j"
        ),
    )
    fig.savefig(out_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.tight_layout()
    plt.close(fig)
    return out_path


def print_top_mean_cosine_tokens_by_model(
    results: list[MeanCosResult],
    *,
    top_k: int = 20,
) -> None:
    """Print token ids with **largest** mean cosine to all other rows (right tail of histogram).

    Loads ``transformers.AutoTokenizer`` once per model (HF cache; can be slow first time).
    """
    from transformers import AutoTokenizer

    for name, arr, v_tok, _d_emb, _rand, _pw in results:
        if arr.size != v_tok:
            print(f"{name}: skip decode — len(mean_cos)={arr.size} != V={v_tok}", file=sys.stderr)
            continue
        try:
            tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        except Exception as exc:  # noqa: BLE001
            print(f"{name}: tokenizer failed ({exc!r})", file=sys.stderr)
            continue

        order = np.argsort(-arr, kind="mergesort")
        top_ids = order[:top_k].astype(np.int64)

        print(f"\n{'=' * 72}")
        print(f"{name} — top {top_k} by mean cos → other rows (highest first)")
        print("=" * 72)

        for rank, tid in enumerate(top_ids, start=1):
            tid_i = int(tid)
            mc = float(arr[tid_i])
            try:
                pieces = tokenizer.convert_ids_to_tokens([tid_i])
                piece = pieces[0] if pieces else "?"
            except Exception:
                piece = "?"
            try:
                decoded = tokenizer.decode([tid_i], skip_special_tokens=False)
            except Exception:
                decoded = "?"
            print(f"  {rank:2}.  id={tid_i:7}  mean_cos={mc:.6f}  piece={piece!r}  decode={decoded!r}")


# %% Collect mean cosine per cached model
_device = pick_device()
CACHED_MEAN_COS_RESULTS = collect_cached_mean_cosine_series(_device)
if CACHED_MEAN_COS_RESULTS:
    print(f"collected {len(CACHED_MEAN_COS_RESULTS)} model(s); run next cell to plot.")
else:
    print("Nothing collected — add caches under data/embedding-layers or edit MULTI_MODEL_INVESTIGATION_NAMES.")

# %% Plot histogram grid to plots/
if not CACHED_MEAN_COS_RESULTS:
    print("Skip plot: run the previous cell first (or fix caches).")
else:
    _p = plot_mean_cosine_grid(CACHED_MEAN_COS_RESULTS)
    print(f"wrote {_p}")

# %% Top tokens by mean cosine (right tail; needs transformers + HF tokenizer cache)
# In Stonesoup set to e.g. 20 and run this cell after collect. Keep 0 for fast ``uv run`` of whole file.
TOP_MEAN_COS_DECODE_K = 20
if TOP_MEAN_COS_DECODE_K > 0 and CACHED_MEAN_COS_RESULTS:
    print_top_mean_cosine_tokens_by_model(
        CACHED_MEAN_COS_RESULTS,
        top_k=TOP_MEAN_COS_DECODE_K,
    )
elif TOP_MEAN_COS_DECODE_K > 0:
    print("Skip top-token decode: run the collect cell first.")
