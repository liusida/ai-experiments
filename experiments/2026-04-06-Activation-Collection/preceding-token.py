# %% Imports, model, prompts, knobs
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import stonesoup

MODEL_ID = "Qwen/Qwen3.5-9B"
TARGET = " king"
MAX_NEW_TOKENS = 160
# Qwen3.5 instruct: format with HF chat_template and disable thinking blocks for plain completion.
USE_CHAT_TEMPLATE = True
CHAT_ENABLE_THINKING = False
# Ten prompts: model must eventually greedy-predict TARGET; if a run never does, it is skipped.
GENERATION_PROMPTS = [
    "In exactly one English sentence, describe a coronation scene. The sentence must contain the noun phrase involving the word king (use a space before king). Sentence:",
    "Write one dry encyclopedia-style sentence about a medieval European king. Sentence:",
    "One sentence only: explain who checks the opposing king in chess. Sentence:",
    "Finish with one vivid sentence: the museum plaque claimed the lonely crown once belonged to a",
    "Write one sentence of dialogue where a teenager complains about a movie villain called the king. Sentence:",
    "One factual sentence about marine biology that still mentions a king (any sense). Sentence:",
    "Complete in one sentence: After the vote, the council feared the street would nickname him the",
    "Write one sentence for a travel brochure about ruins and a buried king. Sentence:",
    "One surreal sentence mentioning a vending machine and a king. Sentence:",
    "In one sentence, summarize a fairy tale ending with a wedding and a king. Sentence:",
]

model, tokenizer = stonesoup.load_model(MODEL_ID)
model.eval()
device = next(model.parameters()).device
inner_tok = getattr(tokenizer, "tokenizer", tokenizer)
MODEL_BASENAME = MODEL_ID.replace("/", "__").replace(":", "-")
print(
    "Loaded:",
    MODEL_ID,
    device,
    f"chat_template={USE_CHAT_TEMPLATE}",
    f"enable_thinking={CHAT_ENABLE_THINKING}",
    flush=True,
)

# %% Helpers: needle ids, greedy peek, capture hiddens before first TARGET emission


def _encode_for_greedy_gen(user_text: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Prompt ids + mask: chat-formatted when ``USE_CHAT_TEMPLATE`` and template exist."""
    text = user_text.strip()
    has_template = getattr(inner_tok, "chat_template", None) is not None
    apply_on_tok = getattr(inner_tok, "apply_chat_template", None)
    apply_on_bundle = getattr(tokenizer, "apply_chat_template", None)

    if not USE_CHAT_TEMPLATE or not has_template:
        enc = inner_tok(
            text,
            return_tensors="pt",
            return_attention_mask=True,
            add_special_tokens=True,
        )
        return enc["input_ids"].to(device), enc["attention_mask"].to(device)

    # Processor ``apply_chat_template`` (transformers) expects multimodal ``content`` shape:
    # list of {type, text} dicts — bare strings raise TypeError in processing_utils.
    if apply_on_tok is not None:
        apply_ct = apply_on_tok
        messages = [{"role": "user", "content": text}]
    elif apply_on_bundle is not None:
        apply_ct = apply_on_bundle
        messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
    else:
        enc = inner_tok(
            text,
            return_tensors="pt",
            return_attention_mask=True,
            add_special_tokens=True,
        )
        return enc["input_ids"].to(device), enc["attention_mask"].to(device)

    tmpl_kw: dict = dict(
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    try:
        enc = apply_ct(messages, enable_thinking=CHAT_ENABLE_THINKING, **tmpl_kw)
    except TypeError:
        enc = apply_ct(messages, **tmpl_kw)

    if hasattr(enc, "to"):
        enc = enc.to(device)
    ids = enc["input_ids"]
    attn = enc["attention_mask"] if "attention_mask" in enc else None
    if attn is None:
        attn = torch.ones(1, ids.shape[1], device=ids.device, dtype=torch.long)
    return ids, attn


def _one_seq_input_ids(batch) -> list[int]:
    ids = batch["input_ids"]
    if isinstance(ids, torch.Tensor):
        row = ids[0]
        return row.tolist() if row.ndim else [int(row.item())]
    if isinstance(ids, list) and ids and isinstance(ids[0], list):
        return ids[0]
    return list(ids)


def pairwise_upper_mean_std(H: torch.Tensor, tri_i: torch.Tensor, tri_j: torch.Tensor):
    """Mean / std of cos(h_i, h_j) for pairs i < j at each layer. H: (n_stages, n_sent, hidden)."""
    n_st = int(H.shape[0])
    mean_c = np.zeros(n_st, dtype=np.float64)
    std_c = np.zeros(n_st, dtype=np.float64)
    for li in range(n_st):
        stonesoup.check_abort()
        x = F.normalize(H[li], dim=-1)
        gram = x @ x.T
        c = gram[tri_i, tri_j].float()
        mean_c[li] = float(c.mean().cpu())
        std_c[li] = float(c.std().cpu())
    return mean_c, std_c


@torch.inference_mode()
def _greedy_continuation_matches_prefix(tids: torch.Tensor, needle_rest: list[int]) -> bool:
    """From sequence tids (already includes first TARGET id), greedy-append matches needle_rest."""
    cur = tids
    for want in needle_rest:
        out = model(
            input_ids=cur,
            attention_mask=torch.ones(1, cur.shape[1], device=device, dtype=torch.long),
        )
        got = int(out.logits[0, -1, :].argmax(dim=-1).item())
        if got != want:
            return False
        cur = torch.cat([cur, torch.tensor([[got]], device=device, dtype=tids.dtype)], dim=1)
    return True


def _decode_row(ids_1d: torch.Tensor) -> str:
    return inner_tok.decode(ids_1d.tolist(), skip_special_tokens=True)


@torch.inference_mode()
def hidden_states_immediately_before_greedy_target(
    prompt: str,
) -> tuple[torch.Tensor | None, str, str]:
    """Return (hiddens | None, full_decode, greedy_continuation_only).

    On success, full_decode is prompt + greedy prefix + full TARGET (nothing past TARGET).
    greedy_continuation_only is the model-new text after the prompt (includes TARGET on success).
    """
    needle = _one_seq_input_ids(inner_tok(TARGET, add_special_tokens=False))
    if not needle:
        raise ValueError("TARGET tokenization empty")
    ids, attn = _encode_for_greedy_gen(prompt)
    prompt_len = int(ids.shape[1])

    def with_target_suffix() -> torch.Tensor:
        row = torch.tensor(needle, device=device, dtype=ids.dtype).unsqueeze(0)
        return torch.cat([ids, row], dim=1)

    for _ in range(MAX_NEW_TOKENS):
        stonesoup.check_abort()
        out = model(
            input_ids=ids,
            attention_mask=attn,
            output_hidden_states=True,
            use_cache=False,
        )
        logits = out.logits[:, -1, :]
        next_id = int(logits[0].argmax(dim=-1).item())

        eos = getattr(inner_tok, "eos_token_id", None)
        if eos is not None and next_id == eos:
            full_s = _decode_row(ids[0])
            cont_s = _decode_row(ids[0, prompt_len:])
            return None, full_s, cont_s

        if next_id == needle[0]:
            rest = needle[1:]
            if not rest:
                hs = out.hidden_states
                vec = torch.stack([h[0, -1].float() for h in hs], dim=0)
                full = with_target_suffix()
                return vec, _decode_row(full[0]), _decode_row(full[0, prompt_len:])
            one = torch.tensor([[needle[0]]], device=device, dtype=ids.dtype)
            candidate = torch.cat([ids, one], dim=1)
            if _greedy_continuation_matches_prefix(candidate, rest):
                hs = out.hidden_states
                vec = torch.stack([h[0, -1].float() for h in hs], dim=0)
                full = with_target_suffix()
                return vec, _decode_row(full[0]), _decode_row(full[0, prompt_len:])

        ids = torch.cat([ids, torch.tensor([[next_id]], device=device, dtype=ids.dtype)], dim=1)
        attn = torch.ones(1, ids.shape[1], device=device, dtype=torch.long)

    full_s = _decode_row(ids[0])
    cont_s = _decode_row(ids[0, prompt_len:])
    return None, full_s, cont_s


needle_dbg = _one_seq_input_ids(inner_tok(TARGET, add_special_tokens=False))
print(f"TARGET={TARGET!r} ids={needle_dbg}", flush=True)

vecs: list[torch.Tensor] = []
failed: list[str] = []
for i, p in enumerate(GENERATION_PROMPTS):
    stonesoup.check_abort()
    v, gen_full, gen_new = hidden_states_immediately_before_greedy_target(p)
    print(f"--- prompt {i} ---", flush=True)
    print(f"  greedy continuation only:\n  {gen_new}", flush=True)
    print(f"  full (prompt + continuation, + {TARGET!r} appended if ok):\n  {gen_full}", flush=True)
    if v is None:
        failed.append(f"[{i}] {p[:72]}…")
        print(
            f"  -> skip: no greedy {TARGET!r} within {MAX_NEW_TOKENS} new tokens\n",
            flush=True,
        )
    else:
        vecs.append(v)
        print(f"  -> ok · hiddens {tuple(v.shape)}\n", flush=True)

if len(vecs) < 2:
    raise RuntimeError(
        f"need >=2 successful generations for pairwise stats; got {len(vecs)}. Failed: {failed}"
    )

H = torch.stack(vecs, dim=0).transpose(0, 1).contiguous()
n_stages = int(H.shape[0])
n_sent = int(H.shape[1])
tri = torch.triu_indices(n_sent, n_sent, offset=1, device=device)
mean_c, std_c = pairwise_upper_mean_std(H, tri[0], tri[1])
peak = int(np.argmax(mean_c))
print(f"pairwise over {n_sent} runs · stages={n_stages}", flush=True)
print(
    f"mean cos peaks at layer stage {peak} "
    f"(mean={mean_c[peak]:.4f}, std={std_c[peak]:.4f})",
    flush=True,
)

# %% Sanity: numpy-only mean pairwise cos matches torch / same peak
# Uses `H`, `mean_c`, `std_c`, `peak`, `n_sent` from the cell above (no model).
H_np = H.detach().float().cpu().numpy()
tri_np = np.triu_indices(n_sent, k=1)
mean_np = np.empty(H_np.shape[0], dtype=np.float64)
std_np = np.empty_like(mean_np)
for li in range(H_np.shape[0]):
    x = H_np[li]
    x = x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-12)
    g = x @ x.T
    c = g[tri_np]
    mean_np[li] = c.mean()
    std_np[li] = c.std(ddof=1)
peak_np = int(np.argmax(mean_np))
if not np.allclose(mean_np, mean_c, rtol=0, atol=5e-5):
    raise AssertionError(f"mean cos mismatch: max abs {np.max(np.abs(mean_np - mean_c)):.3g}")
if not np.allclose(std_np, std_c, rtol=0, atol=5e-5):
    raise AssertionError(f"std cos mismatch: max abs {np.max(np.abs(std_np - std_c)):.3g}")
if peak_np != peak:
    raise AssertionError(f"peak layer mismatch: numpy {peak_np} vs torch path {peak}")
print(
    f"sanity OK · numpy recomputation matches (peak stage {peak_np}, "
    f"same as plot curve)",
    flush=True,
)

# %% Plot
layer_x = np.arange(n_stages)
fig, ax = plt.subplots(figsize=(11, 4.3))
ax.plot(
    layer_x,
    mean_c,
    color="seagreen",
    linewidth=1.3,
    label=(
        f"pairwise cos @ pos before greedy {TARGET!r} ({n_sent}/{len(GENERATION_PROMPTS)} prompts ok)"
    ),
)
ax.fill_between(layer_x, mean_c - std_c, mean_c + std_c, color="seagreen", alpha=0.22)
ax.set_xlabel("layer stage (0 = embeddings, k = post block k-1)")
ax.set_ylabel("cosine similarity")
ax.set_title(
    f"{MODEL_ID}\n"
    f"Chat template · enable_thinking={CHAT_ENABLE_THINKING} · "
    f"hidden before first greedy {TARGET!r} · mean±std over pairs"
)
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.25)
ax.legend(loc="best", fontsize=9)
fig.tight_layout()
stonesoup.show(fig, basename=f"{MODEL_BASENAME}_preceding_hidden_before_greedy_target")


# %% First prompt: one forward, stack all hidden states (chat template)
text = GENERATION_PROMPTS[0].strip()
msgs = [{"role": "user", "content": text}]
enc0 = inner_tok.apply_chat_template(
    msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt"
)
input_ids = enc0["input_ids"].to(device)
attention_mask = enc0["attention_mask"].to(device)
with torch.inference_mode():
    out_first = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )
hidden_states_all = torch.stack([h.float() for h in out_first.hidden_states], dim=0)
print(
    "first prompt · hidden_states_all",
    tuple(hidden_states_all.shape),
    "(n_stages, batch, seq, hidden_dim)",
    flush=True,
)

# %% compare each token in hidden_states_all with mean H and plot (layer 1 line)
ref = H.float().mean(dim=1)
hs = hidden_states_all[:, 0]
li1 = min(1, ref.shape[0] - 1)
cos1 = (F.normalize(hs[li1], dim=-1) * F.normalize(ref[li1], dim=-1)).sum(-1)
fig, ax = plt.subplots(figsize=(10, 2.8))
ax.plot(cos1.cpu().numpy(), color="seagreen", linewidth=1.2)
ax.set_xlabel("prefill token index (first prompt only)")
ax.set_ylabel("cos vs mean(H)")
ax.set_title(f"layer {li1}: prefill position vs mean pre-target hidden")
ax.set_ylim(-0.05, 1.05)
ax.grid(True, alpha=0.25)
fig.tight_layout()
stonesoup.show(fig, basename=f"{MODEL_BASENAME}_prefill_token_vs_meanH_layer1")
t_peak = int(cos1.argmax())
print(f"layer {li1} peak token {t_peak} · cos={float(cos1[t_peak]):.4f}", flush=True)

# %% Prefill pairwise cos at layer 1 (token × token)
li1 = min(1, hidden_states_all.shape[0] - 1)
x = F.normalize(hidden_states_all[li1, 0], dim=-1)
gram_prefill = (x @ x.T).float().cpu().numpy()
fig, ax = plt.subplots(figsize=(6.2, 5))
im = ax.imshow(
    gram_prefill,
    aspect="equal",
    interpolation="nearest",
    cmap="viridis",
    vmin=-1,
    vmax=1,
)
ax.set_xlabel("token j")
ax.set_ylabel("token i")
ax.set_title(f"layer {li1}: pairwise cos(h[i], h[j]) first-prompt prefill")
fig.colorbar(im, ax=ax, shrink=0.82)
fig.tight_layout()
stonesoup.show(fig, basename=f"{MODEL_BASENAME}_prefill_pairwise_cos_layer1")

# %% Arbitrary prompt: layer-1 pairwise cos heatmap (standalone)
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import stonesoup

ARBITRARY_PROMPT = "Explain cosine similarity in one sentence."
_MID = "Qwen/Qwen3.5-0.8B"
_base = _MID.replace("/", "__").replace(":", "-")
model_ar, tok_ar = stonesoup.load_model(_MID)
model_ar.eval()
_dev_ar = next(model_ar.parameters()).device
_inner_ar = getattr(tok_ar, "tokenizer", tok_ar)
enc_ar = _inner_ar.apply_chat_template(
    [{"role": "user", "content": ARBITRARY_PROMPT.strip()}],
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
)
_ids = enc_ar["input_ids"].to(_dev_ar)
_attn = enc_ar["attention_mask"].to(_dev_ar)
with torch.inference_mode():
    _out_ar = model_ar(
        input_ids=_ids,
        attention_mask=_attn,
        output_hidden_states=True,
        use_cache=False,
    )
_hs_ar = torch.stack([h.float() for h in _out_ar.hidden_states], dim=0)
_li = min(1, _hs_ar.shape[0] - 1)
_x_ar = F.normalize(_hs_ar[_li, 0], dim=-1)
_g_ar = (_x_ar @ _x_ar.T).cpu().numpy()
fig_ar, ax_ar = plt.subplots(figsize=(6.2, 5))
_im_ar = ax_ar.imshow(
    _g_ar, aspect="equal", interpolation="nearest", cmap="viridis", vmin=-1, vmax=1
)
ax_ar.set_xlabel("token j")
ax_ar.set_ylabel("token i")
ax_ar.set_title(f"layer {_li}: pairwise cos · arbitrary prompt")
fig_ar.colorbar(_im_ar, ax=ax_ar, shrink=0.82)
fig_ar.tight_layout()
stonesoup.show(fig_ar, basename=f"{_base}_pairwise_cos_arbitrary_L{_li}")
print(ARBITRARY_PROMPT, flush=True)
