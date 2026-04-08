# %% Load probe checkpoint + helpers
from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple

import torch
import torch.nn as nn
import stonesoup

# Written by ``prefill-vs-generated-step2`` binary MLP cell (``BINARY_EXPORT_LAYER`` / ``layer_index``).
# Stage index matches step1 ``gpt2_capture_hidden_stages``: 0 = pre block 0; k>0 = post block k-1.
_BINARY_PROBE_CKPT_NAME = "mlp_binary_gen_vs_prefill_layer15.pt"
BINARY_PROBE_STAGE = 15
_CKPT = (
    stonesoup.repo_root()
    / "outputs"
    / "2026-04-06-Activation-Collection"
    / "prefill-vs-generated-step2"
    / _BINARY_PROBE_CKPT_NAME
)

GPT2_ID = "openai-community/gpt2-medium"


class BranchMLP(nn.Module):
    """Must match ``prefill-vs-generated-step2``."""

    def __init__(self, dim: int, hidden: int, n_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_binary_gen_vs_prefill_ckpt(path: Path | None = None) -> tuple[nn.Module, dict[str, Any]]:
    p = path or _CKPT
    data: dict[str, Any] = torch.load(p, map_location="cpu", weights_only=False)
    model = BranchMLP(
        int(data["in_dim"]),
        int(data["hidden_dim"]),
        2,
    )
    model.load_state_dict(data["state_dict"])
    model.eval()
    return model, data


_model_bin, _ckpt_bin = load_binary_gen_vs_prefill_ckpt()
_ckpt_stage = int(_ckpt_bin["layer_index"])
if _ckpt_stage != int(BINARY_PROBE_STAGE):
    raise RuntimeError(
        f"{_CKPT.name}: checkpoint layer_index={_ckpt_stage}, expected BINARY_PROBE_STAGE={BINARY_PROBE_STAGE}. "
        "Re-run step2 export or align BINARY_PROBE_STAGE / filename."
    )
_mu_b: torch.Tensor = _ckpt_bin["mu"]
_sd_b: torch.Tensor = _ckpt_bin["sd"]

print("Loaded probe:", _CKPT.relative_to(stonesoup.repo_root()), flush=True)
print(
    f"  probe_stage (step1 hook index)={_ckpt_stage}  H={_ckpt_bin['in_dim']}  "
    f"probe_hidden={_ckpt_bin['hidden_dim']}  0={_ckpt_bin['label0']} 1={_ckpt_bin['label1']}",
    flush=True,
)


@torch.inference_mode()
def binary_gen_vs_prefill_logits(hiddens: torch.Tensor) -> torch.Tensor:
    """``(..., H)`` activations at probe layer, **before** z-score; ``(..., 2)`` logits."""
    x = (hiddens.float().reshape(-1, hiddens.shape[-1]) - _mu_b) / _sd_b
    return _model_bin(x).reshape(*hiddens.shape[:-1], 2)


@torch.inference_mode()
def prob_generated(hiddens: torch.Tensor) -> torch.Tensor:
    logit = binary_gen_vs_prefill_logits(hiddens)
    return logit.softmax(dim=-1)[..., 1]


# %% Fused GPT-2 LM + probe (next-token logits + “generated” score for last position)
class FusedLMProbeOutput(NamedTuple):
    """LM ``logits`` (B, T, vocab); probe from **last** position: logits (B, 2), prob class 1 (B,)."""

    logits: torch.Tensor
    probe_logits: torch.Tensor
    prob_generated: torch.Tensor


class GPT2LMWithGenProbe(nn.Module):
    """Wraps HF ``GPT2LMHeadModel``: LM logits + probe on the **last** input position.

    The probe was trained on tensors from ``prefill-vs-generated-step1`` hooks (**pre** block 0,
    then **post** each ``GPT2Block``), **before** the final ``ln_f``.

    Recent Transformers ``output_hidden_states=True`` **replaces** the last tuple entry with
    ``last_hidden_state``, i.e. **after** ``ln_f``, that no longer matches the training stage. This
    module therefore captures the probe layer with **forward hooks** on ``transformer.h``, same
    convention as step 1: ``layer_index`` 0 = input to block 0, ``k`` = output after block ``k-1``.
    """

    def __init__(
        self,
        lm: Any,
        probe: nn.Module,
        mu: torch.Tensor,
        sd: torch.Tensor,
        probe_stage_index: int,
    ) -> None:
        super().__init__()
        self.lm = lm
        self.probe = probe
        self.probe_stage_index = int(probe_stage_index)
        self.register_buffer("_mu", mu.clone())
        self.register_buffer("_sd", sd.clone())
        for _p in self.probe.parameters():
            _p.requires_grad_(False)
        self.probe.eval()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> FusedLMProbeOutput:
        # Avoid “inference tensors” from callers using ``inference_mode`` breaking HF embedding.
        input_ids = input_ids.clone()
        if attention_mask is not None:
            attention_mask = attention_mask.clone()
        kw = {
            k: v
            for k, v in kwargs.items()
            if k not in ("output_hidden_states", "return_dict", "use_cache")
        }
        kw.setdefault("use_cache", False)

        if not hasattr(self.lm, "transformer") or not hasattr(self.lm.transformer, "h"):
            raise TypeError("Expected GPT2LMHeadModel with .transformer.h")

        blocks = self.lm.transformer.h
        li = self.probe_stage_index
        captured: list[torch.Tensor] = []

        if li == 0:

            def _pre0(_mod: Any, inputs: tuple) -> None:
                captured.append(inputs[0].detach())

            _handle = blocks[0].register_forward_pre_hook(_pre0)
        else:
            bi = li - 1
            if bi < 0 or bi >= len(blocks):
                raise ValueError(f"probe_stage_index={li} invalid for n_layer={len(blocks)}")

            def _post(_mod: Any, _inp: Any, out: Any) -> None:
                t = out[0] if isinstance(out, tuple) else out
                captured.append(t.detach())

            _handle = blocks[bi].register_forward_hook(_post)

        try:
            out = self.lm(
                input_ids,
                attention_mask=attention_mask,
                return_dict=True,
                **kw,
            )
        finally:
            _handle.remove()

        if len(captured) != 1:
            raise RuntimeError("probe hook did not capture exactly one hidden tensor")
        last_h = captured[0][:, -1, :].float()
        z = (last_h - self._mu) / self._sd
        probe_logits = self.probe(z)
        prob_gen = probe_logits.softmax(dim=-1)[:, 1]
        return FusedLMProbeOutput(
            logits=out.logits,
            probe_logits=probe_logits,
            prob_generated=prob_gen,
        )


_lm, _tok = stonesoup.load_model(GPT2_ID)
_probe_copy = BranchMLP(int(_ckpt_bin["in_dim"]), int(_ckpt_bin["hidden_dim"]), 2)
_probe_copy.load_state_dict(_ckpt_bin["state_dict"])
_fused = GPT2LMWithGenProbe(
    _lm,
    _probe_copy,
    _mu_b,
    _sd_b,
    _ckpt_stage,
)
_dev_lm = next(_fused.lm.parameters()).device
_fused.to(_dev_lm)
_fused.eval()

print("Fused model device:", _dev_lm, flush=True)


@torch.no_grad()
def forward_text(text: str) -> FusedLMProbeOutput:
    """Tokenize ``text`` (single example), run fused LM, return logits + probe for last position."""
    enc = _tok(text, return_tensors="pt", add_special_tokens=True)
    ids = enc["input_ids"].to(_dev_lm)
    mask = enc.get("attention_mask")
    if mask is not None:
        mask = mask.to(_dev_lm)
    return _fused(ids, mask)


@torch.no_grad()
def greedy_rollout_with_probe(
    prefix: str,
    max_new_tokens: int,
) -> tuple[list[float], list[int], list[str], torch.Tensor, torch.Tensor]:
    """Greedy decode one token at a time; after each forward, record ``prob_generated`` (last position).

    Returns ``probs``, ``token_ids`` appended, per-step ``token_str``, final ``input_ids``, ``attention_mask``.
    """
    enc = _tok(prefix, return_tensors="pt", add_special_tokens=True)
    ids = enc["input_ids"].to(_dev_lm)
    mask = enc["attention_mask"].to(_dev_lm)
    probs: list[float] = []
    new_ids: list[int] = []
    step_toks: list[str] = []
    for _ in range(int(max_new_tokens)):
        out = _fused(ids, mask)
        probs.append(float(out.prob_generated[0]))
        nid = int(out.logits[0, -1, :].argmax())
        new_ids.append(nid)
        step_toks.append(_tok.decode([nid]))
        eos = _tok.eos_token_id
        if eos is not None and nid == int(eos):
            break
        ids = torch.cat(
            [ids, torch.tensor([[nid]], device=_dev_lm, dtype=ids.dtype)], dim=1
        )
        mask = torch.cat([mask, torch.ones((1, 1), device=_dev_lm, dtype=mask.dtype)], dim=1)
    return probs, new_ids, step_toks, ids, mask


# %% Detailed demo + visualization
import matplotlib.pyplot as plt

# First TruthfulQA (``generation`` / ``validation``) question; trailing ``\\n`` matches step1 ``gpt2_question_prefix``.
DEMO_PREFIX = "What happens to you if you eat watermelon seeds?\n"
DEMO_MAX_NEW = 18

_pref_only = forward_text(DEMO_PREFIX)
_pref_logits_last = _pref_only.logits[0, -1, :]
_pref_greedy_id = int(_pref_logits_last.argmax().item())
_ar_probs, _ar_ids, _ar_strs, _ar_ids_final, _ar_mask = greedy_rollout_with_probe(
    DEMO_PREFIX, DEMO_MAX_NEW
)
_seq_full = _tok.decode(_ar_ids_final[0].tolist(), skip_special_tokens=True)
_one_shot = _fused(_ar_ids_final, _ar_mask)
_prob_one_shot_last = float(_one_shot.prob_generated[0])

print("--- Fused LM + probe demo ---", flush=True)
print(f"prefix: {DEMO_PREFIX!r}", flush=True)

_enc_q = _tok(DEMO_PREFIX, return_tensors="pt", add_special_tokens=True)
_qids = _enc_q["input_ids"].to(_dev_lm)
_qmask = _enc_q["attention_mask"].to(_dev_lm)
_qlen = int(_qids.shape[1])
print(
    "\nQuestion tokens only: forward on prefix ``input_ids[:, :t+1]``; "
    "probe reads hidden state at last position (= token ``t``). "
    "Low ``prob_generated`` ≈ prefill-like (class 0).",
    flush=True,
)
print(" pos | prob_gen | token @ pos", flush=True)
print("-" * 44, flush=True)
_question_probs: list[float] = []
for _tq in range(_qlen):
    _out_q = _fused(_qids[:, : _tq + 1], _qmask[:, : _tq + 1])
    _pq = float(_out_q.prob_generated[0])
    _question_probs.append(_pq)
    _tid_q = int(_qids[0, _tq].item())
    print(f" {_tq:3d} | {_pq:8.4f} | {_tok.decode([_tid_q])!r}", flush=True)
_mean_q = sum(_question_probs) / max(len(_question_probs), 1)
print(
    f"\n  mean prob_gen over question: {_mean_q:.4f}  "
    "(well below 0.5 means mostly prefill-like)",
    flush=True,
)

print(
    f"\nafter full prefix (same as last row above): prob_generated={float(_pref_only.prob_generated[0]):.4f}  "
    f"greedy next token={_tok.decode([_pref_greedy_id])!r}",
    flush=True,
)
print(
    f"greedy continuation ({len(_ar_probs)} steps): "
    f"{repr(''.join(_ar_strs))[:200]}{'…' if len(''.join(_ar_strs)) > 200 else ''}",
    flush=True,
)
print(f"full decode (skip_special): {_seq_full[:280]!r}{'…' if len(_seq_full) > 280 else ''}", flush=True)
print(
    f"one forward on full sequence: prob_generated(last pos)={_prob_one_shot_last:.4f} "
    "(teacher-forced on full string → often more prefill-like than mid-rollout)",
    flush=True,
)
print("\n step | prob_gen | piece", flush=True)
print("-" * 40, flush=True)
for _si, (_p, _s) in enumerate(zip(_ar_probs, _ar_strs)):
    print(f" {_si:3d} | {_p:8.4f} | {_s!r}", flush=True)

_steps = list(range(len(_ar_probs)))
_fig_d, _ax_d = plt.subplots(2, 1, figsize=(10, 5.5), height_ratios=[1.35, 1.0])
_ax_d[0].plot(_steps, _ar_probs, "o-", color="tab:blue", lw=1.8, ms=5, label="prob(generated) each AR step")
_ax_d[0].axhline(
    float(_pref_only.prob_generated[0]),
    color="tab:gray",
    ls="--",
    lw=1.2,
    label=f"prefix-only last pos ({float(_pref_only.prob_generated[0]):.3})",
)
_ax_d[0].axhline(
    _prob_one_shot_last,
    color="tab:red",
    ls=":",
    lw=1.5,
    label=f"one-shot full seq last pos ({_prob_one_shot_last:.3f})",
)
_ax_d[0].set_xlabel("greedy step (0 = after prefix, before 1st new token)")
_ax_d[0].set_ylabel("P(class: generated-like)")
_ax_d[0].set_ylim(-0.02, 1.05)
_ax_d[0].set_title(
    "Binary probe on GPT-2-medium (layer %s): rollout vs one-shot\n"
    % _ckpt_stage
)
_ax_d[0].legend(loc="best", fontsize=8)
_ax_d[0].grid(True, alpha=0.3)

_ax_d[1].axis("off")
_wrapped = _seq_full if len(_seq_full) < 900 else _seq_full[:800] + "…"
_ax_d[1].text(
    0.02,
    0.95,
    "Full sequence (decode):\n" + _wrapped,
    transform=_ax_d[1].transAxes,
    fontsize=9,
    verticalalignment="top",
    wrap=True,
    family="monospace",
)
_fig_d.tight_layout()
# %% show plot
stonesoup.show(_fig_d, basename="step3_fused_probe_demo_rollout", dpi=130)
