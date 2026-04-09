# %% Imports and config
from __future__ import annotations

import traceback
from typing import Any

import numpy as np
import stonesoup
import torch
import torch.nn.functional as F
from stonesoup.experiment import (
    configure_matplotlib_agg,
    decoder_blocks,
    encode_text_inputs,
    ensure_pad_token_via_eos,
    hf_repo_id_safe_stem,
    inner_tokenizer,
)

MODEL_IDS = [
    "Qwen/Qwen3-8B-Base",
    "google/gemma-2-2b",
    "allenai/OLMo-2-0425-1B",
    "meta-llama/llama-3.2-3B",
    "mistralai/Ministral-3-3B-Base-2512",
]

PROMPT = "Wikipedia is a free online encyclopedia written and maintained by a community of volunteers, known as Wikipedians, through open collaboration and the wiki software MediaWiki. Founded by Jimmy Wales and Larry Sanger in 2001, Wikipedia has been hosted since 2003 by the Wikimedia Foundation, an American nonprofit organization funded mainly by donations from readers. Wikipedia is the largest and most-read reference work in history. According to Jimmy Wales, its mission is to make the sum of all human knowledge available to every person in the world."
# Plots: per layer, mean ± std over prompt token positions.
# After changing PROMPT, re-run this # %% cell so the kernel binds the new string before running the plot cell.

# Keep references so models stay loaded (memory grows across the loop).
_loaded_models: list[Any] = []
# Filled by the Collect cell; the Plot cell reads this so you can restyle figures without re-forwarding.
VECTOR_MEASUREMENTS_RESULTS: list[dict[str, Any]] = []


def attention_submodule(block: torch.nn.Module) -> torch.nn.Module | None:
    for name in ("attn", "self_attn", "attention"):
        if hasattr(block, name):
            return getattr(block, name)
    return None


def ln1_submodule(block: torch.nn.Module) -> torch.nn.Module | None:
    return getattr(block, "input_layernorm", None) or getattr(block, "ln_1", None)


def ln2_submodule(block: torch.nn.Module) -> torch.nn.Module | None:
    return getattr(block, "post_attention_layernorm", None) or getattr(block, "ln_2", None)


def mlp_submodule(block: torch.nn.Module) -> torch.nn.Module | None:
    return getattr(block, "mlp", None)


def post_feedforward_norm_submodule(block: torch.nn.Module) -> torch.nn.Module | None:
    return getattr(block, "post_feedforward_layernorm", None)


def embed_module(model: torch.nn.Module) -> torch.nn.Module:
    ge = getattr(model, "get_input_embeddings", None)
    if callable(ge):
        emb = ge()
        if emb is not None:
            return emb
    inner = getattr(model, "model", None)
    if inner is not None:
        if hasattr(inner, "embed_tokens"):
            return inner.embed_tokens
        lm = getattr(inner, "language_model", None)
        if lm is not None and hasattr(lm, "embed_tokens"):
            return lm.embed_tokens
    raise TypeError(
        f"Could not resolve input embeddings on {type(model).__name__} "
        "(get_input_embeddings / model.embed_tokens / model.language_model.embed_tokens)."
    )


def vec_at_token(h: torch.Tensor, idx: int) -> torch.Tensor:
    return h[0, idx, :].float()


def l2_norm(v: torch.Tensor) -> float:
    return float(v.norm().item())


def cos_pair(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=-1).item())


def residual_mid_block_out(L: dict[str, torch.Tensor], ti: int) -> tuple[torch.Tensor, torch.Tensor]:
    block_in = vec_at_token(L["block_in"], ti)
    attn_o = vec_at_token(L["attn_out"], ti)
    ln2_pm = vec_at_token(L["ln2_pre_mlp"], ti)
    mlp_o = vec_at_token(L["mlp_out"], ti)
    if "mlp_post_norm" in L:
        mid = block_in + ln2_pm
        mlp_post = vec_at_token(L["mlp_post_norm"], ti)
        block_out = mid + mlp_post
    else:
        mid = block_in + attn_o
        block_out = mid + mlp_o
    if "block_output" in L:
        block_out = vec_at_token(L["block_output"], ti)
    return mid, block_out


class CaptureState:
    def __init__(self) -> None:
        self.embed: torch.Tensor | None = None
        self.layers: dict[int, dict[str, torch.Tensor]] = {}

    def set_layer(self, li: int, key: str, t: torch.Tensor) -> None:
        self.layers.setdefault(li, {})[key] = t.detach()


state = CaptureState()
handles: list[Any] = []


def install_captures(model: torch.nn.Module) -> None:
    remove_handles()
    emb = embed_module(model)

    def embed_hook(mod: torch.nn.Module, inp: Any, out: torch.Tensor) -> None:
        state.embed = out.detach()

    handles.append(emb.register_forward_hook(embed_hook))

    blocks = decoder_blocks(model)
    for li, block in enumerate(blocks):
        ln1 = ln1_submodule(block)
        ln2 = ln2_submodule(block)
        attn = attention_submodule(block)
        mlp = mlp_submodule(block)
        post_ff = post_feedforward_norm_submodule(block)
        if attn is None or mlp is None or ln2 is None:
            raise RuntimeError(
                f"Layer {li} ({type(block).__name__}): missing ln2/attn/mlp for hooking."
            )

        def make_pre(li_: int):
            def pre_hook(mod: torch.nn.Module, inp: tuple[Any, ...]) -> None:
                t = inp[0].detach()
                state.set_layer(li_, "block_in", t)
                if ln1 is None:
                    state.set_layer(li_, "ln1_pre_attn", t)

            return pre_hook

        def make_ln1(li_: int):
            def ln1_hook(mod: torch.nn.Module, inp: Any, out: torch.Tensor) -> None:
                state.set_layer(li_, "ln1_pre_attn", out.detach())

            return ln1_hook

        def make_attn(li_: int):
            def attn_hook(mod: torch.nn.Module, inp: Any, out: Any) -> None:
                first = out[0] if isinstance(out, tuple) else out
                state.set_layer(li_, "attn_out", first.detach())

            return attn_hook

        def make_ln2(li_: int):
            def ln2_hook(mod: torch.nn.Module, inp: Any, out: torch.Tensor) -> None:
                state.set_layer(li_, "ln2_pre_mlp", out.detach())

            return ln2_hook

        def make_mlp(li_: int):
            def mlp_hook(mod: torch.nn.Module, inp: Any, out: torch.Tensor) -> None:
                state.set_layer(li_, "mlp_out", out.detach())

            return mlp_hook

        def make_post_ff(li_: int):
            def post_ff_hook(mod: torch.nn.Module, inp: Any, out: torch.Tensor) -> None:
                state.set_layer(li_, "mlp_post_norm", out.detach())

            return post_ff_hook

        def make_block_output(li_: int):
            def block_out_hook(mod: torch.nn.Module, inp: Any, out: Any) -> None:
                t = out[0] if isinstance(out, tuple) else out
                state.set_layer(li_, "block_output", t.detach())

            return block_out_hook

        handles.append(block.register_forward_pre_hook(make_pre(li)))
        if ln1 is not None:
            handles.append(ln1.register_forward_hook(make_ln1(li)))
        handles.append(attn.register_forward_hook(make_attn(li)))
        handles.append(ln2.register_forward_hook(make_ln2(li)))
        handles.append(mlp.register_forward_hook(make_mlp(li)))
        if post_ff is not None:
            handles.append(post_ff.register_forward_hook(make_post_ff(li)))
        handles.append(block.register_forward_hook(make_block_output(li)))


def remove_handles() -> None:
    for h in handles:
        h.remove()
    handles.clear()


_METRIC_KEYS = (
    "rel_mid",
    "rel_out",
    "rel_d_attn",
    "rel_d_mlp",
    "c_bin_mid",
    "c_mid_out",
    "c_bin_out",
    "c_bin_da",
    "c_mid_dm",
)


def _metrics_at_token(L: dict[str, torch.Tensor], ti: int, eps: float) -> dict[str, float]:
    """δ_attn = mid − h_in, δ_mlp = h_out − mid; norms scaled by ‖h_in‖ (hooks use block_in / block_out)."""
    nan = float("nan")
    block_in = vec_at_token(L["block_in"], ti)
    mid, block_out = residual_mid_block_out(L, ti)
    d_attn = mid - block_in
    d_mlp = block_out - mid
    n_in = l2_norm(block_in)
    if n_in < eps:
        return {k: nan for k in _METRIC_KEYS}
    da_n = float(d_attn.norm().item())
    dm_n = float(d_mlp.norm().item())
    return {
        "rel_mid": l2_norm(mid) / n_in,
        "rel_out": l2_norm(block_out) / n_in,
        "rel_d_attn": l2_norm(d_attn) / n_in,
        "rel_d_mlp": l2_norm(d_mlp) / n_in,
        "c_bin_mid": cos_pair(block_in, mid),
        "c_mid_out": cos_pair(mid, block_out),
        "c_bin_out": cos_pair(block_in, block_out),
        "c_bin_da": cos_pair(block_in, d_attn) if da_n >= eps else nan,
        "c_mid_dm": cos_pair(mid, d_mlp) if dm_n >= eps else nan,
    }


def collect_layer_metrics(
    layers: dict[int, dict[str, torch.Tensor]], ti_list: list[int]
) -> dict[str, Any]:
    """Per layer: nanmean / nanstd of per-token metrics across ``ti_list``."""
    n_l = len(layers)
    eps = 1e-12
    n_tok = len(ti_list)

    out: dict[str, Any] = {"n_layers": n_l, "n_tokens": n_tok}
    for key in _METRIC_KEYS:
        out[f"{key}_mean"] = []
        out[f"{key}_std"] = []

    for li in range(n_l):
        L = layers[li]
        series = {k: [] for k in _METRIC_KEYS}
        for ti in ti_list:
            m = _metrics_at_token(L, ti, eps)
            for k in _METRIC_KEYS:
                series[k].append(m[k])
        for k in _METRIC_KEYS:
            arr = np.asarray(series[k], dtype=np.float64)
            out[f"{k}_mean"].append(float(np.nanmean(arr)))
            out[f"{k}_std"].append(float(np.nanstd(arr, ddof=0)))

    return out


def _plot_band(
    ax: Any,
    x: np.ndarray,
    mean: list[float],
    std: list[float],
    *,
    label: str | None = None,
    marker: str = "o",
    ms: float = 4,
    color: str | None = None,
) -> None:
    m = np.asarray(mean, dtype=np.float64)
    s = np.asarray(std, dtype=np.float64)
    kw: dict[str, Any] = {"marker": marker, "ms": ms}
    if color is not None:
        kw["color"] = color
    (line,) = ax.plot(x, m, label=label, **kw)
    c = line.get_color()
    ax.fill_between(x, m - s, m + s, color=c, alpha=0.22, linewidth=0)


# %% Collect: load models, one forward per model, store metrics (no plots)
VECTOR_MEASUREMENTS_RESULTS.clear()
for model_id in MODEL_IDS:
    try:
        model, proc = stonesoup.load_model(model_id)
        model.eval()
        tokenizer = inner_tokenizer(proc)
        ensure_pad_token_via_eos(tokenizer)
        device = next(model.parameters()).device
        _loaded_models.append(model)
        print(f"Loaded: {model_id}  {device}", flush=True)

        state = CaptureState()
        install_captures(model)
        enc = encode_text_inputs(proc, PROMPT, device=device)
        seq_len = int(enc["input_ids"].shape[1])
        ti_list = list(range(seq_len))

        stonesoup.check_abort()
        try:
            with torch.inference_mode():
                model(**enc, use_cache=False)
        finally:
            remove_handles()

        if not state.layers:
            raise RuntimeError("No layer captures.")

        m = collect_layer_metrics(state.layers, ti_list)
        stem = hf_repo_id_safe_stem(model_id)
        dtype_s = str(next(model.parameters()).dtype)
        VECTOR_MEASUREMENTS_RESULTS.append(
            {
                "model_id": model_id,
                "m": m,
                "seq_len": seq_len,
                "stem": stem,
                "dtype": dtype_s,
            }
        )
        print(f"OK metrics: {model_id}  (seq_len={seq_len}, layers={m['n_layers']})", flush=True)
    except Exception as e:
        print(f"SKIP {model_id}: {e}", flush=True)
        traceback.print_exc()

print(
    f"Collect done. {len(VECTOR_MEASUREMENTS_RESULTS)} result(s), "
    f"{len(_loaded_models)} model(s) kept in memory. Run the Plot cell next.",
    flush=True,
)


# %% Plot: dashboards from VECTOR_MEASUREMENTS_RESULTS (edit this cell freely)
configure_matplotlib_agg()
import matplotlib.pyplot as plt

print(stonesoup.STONESOUP_RENDER_HTML, end="")  # must be the first output of the cell

if not VECTOR_MEASUREMENTS_RESULTS:
    print("No data in VECTOR_MEASUREMENTS_RESULTS — run the Collect cell first.", flush=True)
else:
    for row in VECTOR_MEASUREMENTS_RESULTS:
        model_id = row["model_id"]
        m = row["m"]
        seq_len = row["seq_len"]
        stem = row["stem"]
        dtype_s = row["dtype"]
        n_l = m["n_layers"]
        n_tok = m["n_tokens"]
        x = np.arange(n_l)

        agg_note = f"mean ± std · {n_tok} tokens  ·  seq_len={seq_len}"
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9, 8), constrained_layout=True)
        fig.suptitle(
            f"{model_id}\n"
            f"{agg_note}  ·  {n_l} layers  ·  {dtype_s}  ·  "
            r"$\delta_{\mathrm{attn}}=\mathrm{mid}-h_{\mathrm{in}}$, "
            r"$\delta_{\mathrm{mlp}}=h_{\mathrm{out}}-\mathrm{mid}$",
            fontsize=10,
            fontweight="medium",
        )

        ax1.axhline(1.0, color="gray", ls="--", lw=1.0, label=r"$\Vert h_{\mathrm{in}}\Vert$ (=1)")
        _plot_band(
            ax1, x, m["rel_mid_mean"], m["rel_mid_std"], label=r"$\Vert\mathrm{mid}\Vert/\Vert h_{\mathrm{in}}\Vert$"
        )
        _plot_band(
            ax1,
            x,
            m["rel_out_mean"],
            m["rel_out_std"],
            marker="s",
            label=r"$\Vert h_{\mathrm{out}}\Vert/\Vert h_{\mathrm{in}}\Vert$",
        )
        ax1.set_ylabel(r"$\Vert\cdot\Vert/\Vert h_{\mathrm{in}}\Vert$")
        ax1.set_title(r"State norms ($\mathrm{mid}$, $h_{\mathrm{out}}$)")
        ax1.set_xticks(x)
        ax1.set_ylim(0.0, 2.0)
        ax1.legend(loc="lower right", fontsize=8)
        ax1.grid(True, alpha=0.3)

        _plot_band(
            ax2,
            x,
            m["rel_d_attn_mean"],
            m["rel_d_attn_std"],
            color="C0",
            label=r"$\Vert\delta_{\mathrm{attn}}\Vert/\Vert h_{\mathrm{in}}\Vert$",
        )
        _plot_band(
            ax2,
            x,
            m["rel_d_mlp_mean"],
            m["rel_d_mlp_std"],
            marker="s",
            color="C1",
            label=r"$\Vert\delta_{\mathrm{mlp}}\Vert/\Vert h_{\mathrm{in}}\Vert$",
        )
        ax2.set_ylabel(r"$\Vert\delta\Vert/\Vert h_{\mathrm{in}}\Vert$")
        ax2.set_title(r"Update norms ($\delta_{\mathrm{attn}}$, $\delta_{\mathrm{mlp}}$)")
        ax2.set_xticks(x)
        ax2.set_ylim(0.0, 1.0)
        ax2.legend(loc="lower right", fontsize=8)
        ax2.grid(True, alpha=0.3)

        _plot_band(ax3, x, m["c_bin_mid_mean"], m["c_bin_mid_std"], label=r"cos($h_{\mathrm{in}}$, mid)")
        _plot_band(
            ax3, x, m["c_mid_out_mean"], m["c_mid_out_std"], marker="s", label=r"cos(mid, $h_{\mathrm{out}}$)"
        )
        _plot_band(
            ax3, x, m["c_bin_out_mean"], m["c_bin_out_std"], marker="^", label=r"cos($h_{\mathrm{in}}$, $h_{\mathrm{out}}$)"
        )
        _plot_band(
            ax3, x, m["c_bin_da_mean"], m["c_bin_da_std"], marker="d", label=r"cos($h_{\mathrm{in}}$, $\delta_{\mathrm{attn}}$)"
        )
        _plot_band(
            ax3, x, m["c_mid_dm_mean"], m["c_mid_dm_std"], marker="v", label=r"cos(mid, $\delta_{\mathrm{mlp}}$)"
        )
        ax3.axhline(0.0, color="gray", lw=0.6, alpha=0.5)
        ax3.set_xlabel("Layer")
        ax3.set_ylabel("cosine")
        ax3.set_title(r"Cosines ($h_{\mathrm{in}}$, mid, $h_{\mathrm{out}}$, $\delta$)")
        ax3.set_xticks(x)
        ax3.set_ylim(-1.05, 1.05)
        ax3.legend(loc="lower right", fontsize=7, ncol=2)
        ax3.grid(True, alpha=0.3)

        stonesoup.show(
            fig,
            basename=f"{stem}_vector_measurements",
            dpi=120,
            emit_render_hint=False,
        )
        print(f"OK plot: {stem}_vector_measurements", flush=True)

    print(f"Plot done. {len(VECTOR_MEASUREMENTS_RESULTS)} figure(s).", flush=True)
