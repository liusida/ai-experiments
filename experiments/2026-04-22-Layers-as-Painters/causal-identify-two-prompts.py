# %% Causal: patch layer L to use *England* input; revert block output to *Denmark*; KL vs two baselines
# Two few-shot prompts. Prefix = answer token positions [0,1,2] (so next token = response[3]).
# At layer L: pre-hook — last-position hidden **entering** block L = England. post-hook — last-position
# **output** of block L = Denmark. Later layers only see the Denmark main path for that position
# (not a rolling England state).
from __future__ import annotations

import hashlib

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import stonesoup
from stonesoup.experiment import (
    configure_matplotlib_agg,
    decoder_blocks,
    hf_repo_id_safe_stem,
    inner_tokenizer,
)

configure_matplotlib_agg()

MODEL_ID = "Qwen/Qwen3-8B"
# Focus on which response token the model is predicting: prefix = prompt + first `CUT` continuations.
CUT = 3
MAX_NEW_TOKENS = 5
# Random “half the layers” EN-context cell: which subset to force `eng_pre` at block input (pre-only, no post reset).


P_DK = "Name the birthday of the Queen of Denmark."
P_ENG = "Name the birthday of the Queen of England."

model, proc = stonesoup.load_model(MODEL_ID)
model.eval()
tokenizer = inner_tokenizer(proc)
device = next(model.parameters()).device
blocks = decoder_blocks(model)
N = len(blocks)
safe = hf_repo_id_safe_stem(MODEL_ID)
print(f"model={MODEL_ID!r}  n_decode_layers={N}  cut={CUT}")


def _ids_from_messages(msgs: list) -> str:
    try:
        t = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        t = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return t


# %% Build prefixes: greedy generate(5) then take first CUT response tokens; predict response[CUT]
def _prefix_for(messages: list, tag: str) -> torch.Tensor:
    ch = _ids_from_messages(messages)
    pids = tokenizer(ch, return_tensors="pt").input_ids.to(device)
    with torch.inference_mode():
        gen = model.generate(
            pids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    r = gen[0, pids.shape[1] :]
    if r.shape[0] < CUT + 1:
        raise RuntimeError(
            f"{tag}: need >= {CUT + 1} response tokens, got {r.shape[0]}: "
            f"{tokenizer.decode(r, skip_special_tokens=True)!r}"
        )
    at = r[CUT]
    pre = torch.cat([pids[0], r[:CUT]]).unsqueeze(0)
    _rprev = tokenizer.decode(r, skip_special_tokens=True)[:80]
    print(
        f"{tag}  n_resp={r.shape[0]}  response[3] id={at.item()!r}  "
        f"text~ {tokenizer.decode([at])!r}  all: {_rprev!r}..."
    )
    return pre


m_dk = [{"role": "user", "content": P_DK}]
m_en = [{"role": "user", "content": P_ENG}]

id_dk = _prefix_for(m_dk, "DK")
id_en = _prefix_for(m_en, "EN")

# %% Cache last-position: input *to* each block and output *of* each block (Denmark and England)
eng_pre: list[torch.Tensor] = [None] * N
eng_post: list[torch.Tensor] = [None] * N
dk_pre: list[torch.Tensor] = [None] * N
dk_post: list[torch.Tensor] = [None] * N
handles: list = []


def _make_caches() -> None:
    global eng_pre, eng_post, dk_pre, dk_post, handles
    for h in handles:
        h.remove()
    handles = []
    for i, b in enumerate(blocks):
        j = int(i)
        pdk = b.register_forward_pre_hook(
            lambda m, a, j=j, store=dk_pre: _capture_pre(a, j, store)
        )
        qdk = b.register_forward_hook(
            lambda m, inn, o, j=j, store=dk_post: _capture_post(o, j, store)
        )
        handles.append(pdk)
        handles.append(qdk)
    with torch.inference_mode():
        model(id_dk, use_cache=False)
    for h in handles:
        h.remove()
    handles = []
    for i, b in enumerate(blocks):
        j = int(i)
        pre = b.register_forward_pre_hook(
            lambda m, a, j=j, store=eng_pre: _capture_pre(a, j, store)
        )
        post = b.register_forward_hook(
            lambda m, inn, o, j=j, store=eng_post: _capture_post(o, j, store)
        )
        handles.append(pre)
        handles.append(post)
    with torch.inference_mode():
        model(id_en, use_cache=False)
    for h in handles:
        h.remove()
    handles = []


def _capture_pre(
    a: object,
    j: int,
    store: list,
) -> None:
    t = a[0] if isinstance(a, tuple) and len(a) else a
    store[j] = t[:, -1, :].detach().float().cpu()  # [1, D]


def _capture_post(
    o: object,
    j: int,
    store: list,
) -> None:
    t = o[0] if isinstance(o, tuple) else o
    store[j] = t[:, -1, :].detach().float().cpu()


_make_caches()
for t in (eng_pre, eng_post, dk_pre, dk_post):
    for i, x in enumerate(t):
        if x is None:
            raise RuntimeError(f"missing cache at layer {i}")


# %% Baseline log-probs (last position = query for response[CUT])
def _logit_last(idd: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode():
        return model(idd, use_cache=False).logits[0, -1].float().cpu()


l_dk = F.log_softmax(_logit_last(id_dk), dim=-1)
l_en = F.log_softmax(_logit_last(id_en), dim=-1)

print("baselines: KL(p_DK || p_EN) =", F.kl_div(l_dk, l_en, reduction="sum", log_target=True).item())
print("baselines: KL(p_EN || p_DK) =", F.kl_div(l_en, l_dk, reduction="sum", log_target=True).item())

_LAST_L = N - 1

# %% Per-layer: patch L pre with England, post to Denmark; Denmark ids throughout
def _patch_hooks(eng_in_b: torch.Tensor, dk_out_b: torch.Tensor):
    """Closes on CPU tensors, moved to the current device in hooks."""

    def pre_hook(_m, a_in):
        a = a_in if isinstance(a_in, tuple) else (a_in,)
        x0 = a[0]
        e = eng_in_b.to(device=x0.device, dtype=x0.dtype)
        x2 = x0.clone()
        x2[:, -1, :] = e
        if len(a) == 1:
            return (x2,)
        return (x2,) + a[1:]

    def post_hook(_m, _inp, o):
        h0 = o[0] if isinstance(o, tuple) else o
        d = dk_out_b.to(device=h0.device, dtype=h0.dtype)
        t = h0.clone()
        t[:, -1, :] = d
        if isinstance(o, tuple):
            return (t,) + o[1:]
        return t

    return pre_hook, post_hook


kl_toward_dk: list[float] = []
kl_toward_en: list[float] = []

for patch_i in range(N):
    stonesoup.check_abort()
    _ein_c = eng_pre[patch_i].float().cpu().clone()  # decouple this layer from loop
    _dout_c = dk_post[patch_i].float().cpu().clone()
    pre_h, post_h = _patch_hooks(_ein_c, _dout_c)
    b = blocks[patch_i]
    h1 = b.register_forward_pre_hook(pre_h)
    h2 = b.register_forward_hook(post_h)
    try:
        with torch.inference_mode():
            lq = F.log_softmax(
                model(id_dk, use_cache=False).logits[0, -1].float().cpu(),
                dim=-1,
            )
    finally:
        h1.remove()
        h2.remove()
    kl_toward_dk.append(
        F.kl_div(lq, l_dk, reduction="sum", log_target=True).item()
    )
    kl_toward_en.append(
        F.kl_div(lq, l_en, reduction="sum", log_target=True).item()
    )
    if patch_i % 8 == 0 or patch_i == N - 1:
        top = lq.exp().argmax().item()
        print(
            f"patch L{patch_i}  KL(q||p_DK)={kl_toward_dk[-1]:.4f}  "
            f"KL(q||p_EN)={kl_toward_en[-1]:.4f}  top1={tokenizer.decode([top])!r}"
        )

# %% Random half of layers: at last pos, *pre-only* = England (cached `eng_pre`); no post reset — one mixed forward on DK ids
def _en_pre_only_hook(eng_in_b: torch.Tensor):
    def pre_hook(_m, a_in):
        a = a_in if isinstance(a_in, tuple) else (a_in,)
        x0 = a[0]
        e = eng_in_b.to(device=x0.device, dtype=x0.dtype)
        x2 = x0.clone()
        x2[:, -1, :] = e
        if len(a) == 1:
            return (x2,)
        return (x2,) + a[1:]

    return pre_hook


def _h_cos(a: torch.Tensor, b: torch.Tensor) -> float:
    """`a`, `b` as [1, d] on CPU; cosine similarity in ℝ^d (e.g. post–final-block hidden at last pos)."""
    return float(F.cosine_similarity(a[0:1, :], b[0:1, :], dim=1, eps=1e-8).item())


for RANDOM_EN_LAYER_SEED in range(10):
    stonesoup.check_abort()
    _rng = np.random.default_rng(int(RANDOM_EN_LAYER_SEED))
    _k = N // 4 + 3
    _en_set = sorted(int(x) for x in _rng.choice(N, size=_k, replace=False))
    _h_m_from_last: list[torch.Tensor | None] = [None]

    def _post_cap_final(_m, _inn, o):
        t = o[0] if isinstance(o, tuple) else o
        _h_m_from_last[0] = t[:, -1, :].detach().float().cpu()

    _pre_hs: list = []
    for li in _en_set:
        _eb = eng_pre[li].float().cpu().clone()
        _pre_hs.append(blocks[li].register_forward_pre_hook(_en_pre_only_hook(_eb)))
    _h_last_blk = blocks[_LAST_L].register_forward_hook(_post_cap_final)
    try:
        with torch.inference_mode():
            _lm_logits = model(id_dk, use_cache=False).logits[0, -1].float().cpu()
    finally:
        _h_last_blk.remove()
        for _h in _pre_hs:
            _h.remove()
    _lq_mixed = F.log_softmax(_lm_logits, dim=-1)
    _h_m = _h_m_from_last[0]
    if _h_m is None:
        raise RuntimeError("final-block hook did not run")
    _mixed_kl_dk = F.kl_div(_lq_mixed, l_dk, reduction="sum", log_target=True).item()
    _mixed_kl_en = F.kl_div(_lq_mixed, l_en, reduction="sum", log_target=True).item()
    _mixed_top = _lq_mixed.exp().argmax().item()
    _cos_dk = _h_cos(_h_m, dk_post[_LAST_L])
    _cos_en = _h_cos(_h_m, eng_post[_LAST_L])
    print(
        f"random EN (pre-only)  seed={RANDOM_EN_LAYER_SEED}  n={_k}  layers={_en_set!r}  "
        f"KL(q||p_DK)={_mixed_kl_dk:.4f}  KL(q||p_EN)={_mixed_kl_en:.4f}  top1={tokenizer.decode([_mixed_top])!r}  "
        f"post-L{_LAST_L} h  cos(·,DK)={_cos_dk:.4f}  cos(·,EN)={_cos_en:.4f}"
    )

# %% Plot: x=layer, y=KL, two lines
x = np.arange(N)
y_dk = np.array(kl_toward_dk, dtype=np.float64)
y_en = np.array(kl_toward_en, dtype=np.float64)
fig, ax = plt.subplots(figsize=(9, 4.2))
ax.plot(x, y_dk, "o-", color="#1f77b4", label="KL(q_patched || p_Denmark baseline)", ms=3)
ax.plot(x, y_en, "o-", color="#ff7f0e", label="KL(q_patched || p_England baseline)", ms=3)
ax.set_xlabel("Layer L (pre = England, post of L reset to Denmark at last pos)")
ax.set_ylabel("KL (sum, natural log, full vocab)")
ax.set_title(
    f"{MODEL_ID}  patch England→input @ L, Denmark→output @ L, forward on Denmark ids; "
    f"cut={CUT}  next-token dist q"
)
ax.set_xticks(x[::2])
ax.grid(alpha=0.3, axis="y", linestyle=":")
ax.legend()
fig.tight_layout()
_tag = hashlib.md5(f"{P_DK}{P_ENG}{CUT}{MAX_NEW_TOKENS}".encode()).hexdigest()[:8]
stonesoup.show(fig, basename=f"{safe}_causal_id_two_KL_{_tag}", dpi=140)
plt.close(fig)
print("done — causal two-prompts id")
