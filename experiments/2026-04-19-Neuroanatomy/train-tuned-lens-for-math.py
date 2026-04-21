# %% Imports
from __future__ import annotations

import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

import stonesoup
from stonesoup.experiment import (
    configure_matplotlib_agg,
    ensure_pad_token_via_eos,
    hf_repo_id_safe_stem,
    inner_tokenizer,
)

configure_matplotlib_agg()

# %% Config
MODEL_ID = "Qwen/Qwen3.5-9B"
GSM8K_REPO = "openai/gsm8k"
GSM8K_CONFIG = "main"
GSM8K_SPLIT = "train"

PILE_10K_ID = "NeelNanda/pile-10k"
# Parquet Hub dataset (legacy ``wikipedia`` script loader is unsupported in recent ``datasets``).
WIKIPEDIA_REPO = "wikimedia/wikipedia"
WIKIPEDIA_ZH_CONFIG = "20231101.zh"

# Corpus sizes (streaming Wikipedia zh; sliced GSM8K / pile); RYS JSONs are always fully included.
WIKIPEDIA_ZH_ROWS = 30
GSM8K_TRAIN_ROWS = 600
PILE_TRAIN_ROWS = 150
# One epoch == ``len(loader)`` steps (each step is one batch of ``BATCH_SIZE`` chunks). Increase for more passes.
TRAIN_EPOCHS = 1

RYS_JSON_REL = [
    "data/rys-dataset/simple_math_16.json",
    "data/rys-dataset/mid_math_16.json",
    "data/rys-dataset/math_16.json",
]

SEQ_LEN = 256
BATCH_SIZE = 4
LR = 1e-3
GRAD_CLIP = 1.0
RANDOM_SEED = 0

# %% Load model and detect architecture
model, processor = stonesoup.load_model(MODEL_ID)
model.eval()
model.requires_grad_(False)

tokenizer = inner_tokenizer(processor)
ensure_pad_token_via_eos(tokenizer)

device = next(model.parameters()).device
text_cfg = getattr(model.config, "text_config", model.config)
d_model = text_cfg.hidden_size
n_layers = text_cfg.num_hidden_layers

_inner = model.model
text_module = getattr(_inner, "language_model", _inner)
final_norm = text_module.norm
lm_head = model.lm_head

# ``num_hidden_layers`` can disagree with ``output_hidden_states`` length (e.g. extra embedding / post blocks).
with torch.no_grad():
    _probe_ids = tokenizer.encode(" .", add_special_tokens=False)
    if not _probe_ids:
        _probe_ids = [0]
    _probe_t = torch.tensor([_probe_ids], dtype=torch.long, device=device)
    _probe_out = text_module(input_ids=_probe_t, output_hidden_states=True, use_cache=False)
    n_hidden_total = len(_probe_out.hidden_states)
    n_probed = n_hidden_total - 1

print(f"Model: {MODEL_ID}", flush=True)
print(f"  class={type(model).__name__}, d_model={d_model}, device={device}", flush=True)
print(
    f"  config num_hidden_layers={n_layers}, hidden_states in forward={n_hidden_total} "
    f"→ TunedLens translators={n_probed}",
    flush=True,
)
if n_probed != n_layers:
    print(
        "  (using forward hidden-state count, not config alone, so layer index matches training loop)",
        flush=True,
    )

# %% Build corpus (Wikipedia zh + GSM8K + pile-10k + RYS JSON) and tokenize
repo_root = stonesoup.repo_root()


def load_rys_texts(path: Path) -> list[str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: list[str] = []
    for key in sorted(raw.keys(), key=lambda k: int(k)):
        row = raw[key]
        q = row["question"]
        a = row["answer"]
        out.append(f"{q}\nAnswer: {a}")
    return out


tagged: list[tuple[str, str]] = []
_src_wiki = f"{WIKIPEDIA_REPO} ({WIKIPEDIA_ZH_CONFIG})"
_src_gsm = f"{GSM8K_REPO} ({GSM8K_SPLIT}[:{GSM8K_TRAIN_ROWS}])"
_src_pile = f"{PILE_10K_ID} (train[:{PILE_TRAIN_ROWS}])"

print(
    f"Corpus: Wikipedia zh ×{WIKIPEDIA_ZH_ROWS}, GSM8K ×{GSM8K_TRAIN_ROWS}, "
    f"{PILE_10K_ID} ×{PILE_TRAIN_ROWS}, plus RYS JSONs",
    flush=True,
)
wiki_stream = load_dataset(
    WIKIPEDIA_REPO,
    WIKIPEDIA_ZH_CONFIG,
    split="train",
    streaming=True,
)
_n_wiki = 0
for row in wiki_stream:
    tagged.append((_src_wiki, row["text"].strip()))
    _n_wiki += 1
    if _n_wiki >= WIKIPEDIA_ZH_ROWS:
        break
print(f"{WIKIPEDIA_REPO} ({WIKIPEDIA_ZH_CONFIG}): {_n_wiki} articles", flush=True)

gsm = load_dataset(GSM8K_REPO, GSM8K_CONFIG, split=f"{GSM8K_SPLIT}[:{GSM8K_TRAIN_ROWS}]")
_n_gsm = 0
for row in gsm:
    tagged.append(
        (_src_gsm, row["question"].strip() + "\n" + row["answer"].strip()),
    )
    _n_gsm += 1
print(f"GSM8K ({GSM8K_SPLIT}[:{GSM8K_TRAIN_ROWS}]): {_n_gsm} examples", flush=True)

pile = load_dataset(PILE_10K_ID, split=f"train[:{PILE_TRAIN_ROWS}]")
_n_pile = 0
for row in pile:
    tagged.append((_src_pile, row["text"].strip()))
    _n_pile += 1
print(f"{PILE_10K_ID} (train[:{PILE_TRAIN_ROWS}]): {_n_pile} examples", flush=True)

for rel in RYS_JSON_REL:
    p = repo_root / rel
    if not p.is_file():
        raise FileNotFoundError(f"Missing RYS dataset file: {p}")
    rys = load_rys_texts(p)
    _rys_label = f"RYS · {rel}"
    for t in rys:
        tagged.append((_rys_label, t))
    print(f"RYS {rel}: {len(rys):,} examples", flush=True)

rng = random.Random(RANDOM_SEED)
rng.shuffle(tagged)

token_counts: dict[str, int] = {}
all_ids: list[int] = []
for src, text in tagged:
    enc = tokenizer.encode(text + "\n\n", add_special_tokens=False)
    token_counts[src] = token_counts.get(src, 0) + len(enc)
    all_ids.extend(enc)

print(
    "Tokens by source (each document encoded as text + '\\n\\n'; order is not shuffle order):",
    flush=True,
)
_src_order: list[str] = [_src_wiki, _src_gsm, _src_pile]
for rel in RYS_JSON_REL:
    _src_order.append(f"RYS · {rel}")
for _lbl in _src_order:
    if _lbl in token_counts:
        print(f"  {_lbl}: {token_counts[_lbl]:,}", flush=True)
for _lbl in sorted(k for k in token_counts if k not in _src_order):
    print(f"  {_lbl}: {token_counts[_lbl]:,}", flush=True)
print(f"  total: {sum(token_counts.values()):,}", flush=True)

n_chunks = len(all_ids) // SEQ_LEN
all_ids_t = torch.tensor(all_ids[: n_chunks * SEQ_LEN], dtype=torch.long).view(n_chunks, SEQ_LEN)
print(f"Tokenized: {len(all_ids):,} tokens → {n_chunks:,} chunks of {SEQ_LEN}", flush=True)

loader = DataLoader(TensorDataset(all_ids_t), batch_size=BATCH_SIZE, shuffle=True)

NUM_STEPS = len(loader) * TRAIN_EPOCHS
print(
    f"NUM_STEPS = {NUM_STEPS} (TRAIN_EPOCHS={TRAIN_EPOCHS}, len(loader)={len(loader)}; "
    f"BATCH_SIZE={BATCH_SIZE})",
    flush=True,
)

# %% Token coverage histogram
stonesoup.html()
vocab_size = text_cfg.vocab_size
token_counts = torch.bincount(torch.tensor(all_ids, dtype=torch.long), minlength=vocab_size)[:vocab_size]
seen_mask = token_counts > 0
n_seen = int(seen_mask.sum().item())
n_unseen = vocab_size - n_seen

print(f"Text vocab size: {vocab_size:,}  (tokenizer.vocab_size={tokenizer.vocab_size:,})", flush=True)
print(f"Tokens seen in training data: {n_seen:,} ({n_seen / vocab_size:.1%})", flush=True)
print(f"Tokens never seen: {n_unseen:,} ({n_unseen / vocab_size:.1%})", flush=True)

unseen_ids = torch.where(~seen_mask)[0].tolist()
sample_unseen = unseen_ids[:: max(1, len(unseen_ids) // 20)][:20]
unseen_examples = [repr(tokenizer.decode([tid])) for tid in sample_unseen]
print(f"Sample unseen tokens: {', '.join(unseen_examples)}", flush=True)

nonzero_counts = token_counts[seen_mask].float().numpy()
sorted_nonzero = np.sort(nonzero_counts)[::-1]

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(np.arange(len(sorted_nonzero)), sorted_nonzero)
ax.set_yscale("log")
ax.set_xlim(0, vocab_size)
ax.set_xlabel("Token rank")
ax.set_ylabel("Frequency (log)")
ax.set_title(f"Zipf plot ({n_seen:,} seen, {n_unseen:,} unseen of {vocab_size:,})")
ax.axhline(y=1, color="red", linestyle="--", linewidth=0.8, label="seen once")
ax.axvline(x=n_seen, color="gray", linestyle="--", linewidth=0.8, label=f"coverage: {n_seen / vocab_size:.0%}")
ax.legend(fontsize=8)

fig.tight_layout()
stonesoup.show(fig)

# %% TunedLens, decode, KL loss


class AffineTranslator(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.weight = nn.Parameter(torch.eye(d))
        self.bias = nn.Parameter(torch.zeros(d))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return (h.float() @ self.weight.T + self.bias).to(h.dtype)


class TunedLens(nn.Module):
    def __init__(self, n_probed: int, d: int):
        super().__init__()
        self.translators = nn.ModuleList([AffineTranslator(d) for _ in range(n_probed)])


def decode(h: torch.Tensor) -> torch.Tensor:
    return lm_head(final_norm(h))


def kl_loss(final_logits: torch.Tensor, lens_logits: torch.Tensor) -> torch.Tensor:
    """KL(target || pred) with frozen target distribution (detach final logits)."""
    target_lp = F.log_softmax(final_logits.detach().float(), dim=-1)
    target_p = target_lp.exp()
    pred_lp = F.log_softmax(lens_logits.float(), dim=-1)
    return (target_p * (target_lp - pred_lp)).sum(dim=-1).mean()


lens = TunedLens(n_probed, d_model).to(device)
print(f"TunedLens: {n_probed} translators, {sum(p.numel() for p in lens.parameters()):,} params", flush=True)

# %% Train
torch.set_grad_enabled(True)
for _p in lens.parameters():
    _p.requires_grad_(True)

optimizer = torch.optim.AdamW(lens.parameters(), lr=LR)
lens.train()

data_iter = iter(loader)
pbar = tqdm(range(NUM_STEPS), desc="Training")
for step in pbar:
    stonesoup.check_abort()

    try:
        (input_ids,) = next(data_iter)
    except StopIteration:
        data_iter = iter(loader)
        (input_ids,) = next(data_iter)
    input_ids = input_ids.to(device)

    with torch.no_grad():
        out = text_module(input_ids=input_ids, output_hidden_states=True, use_cache=False)
        hidden_states = out.hidden_states
        final_logits = lm_head(hidden_states[-1])

    with torch.enable_grad():
        losses = []
        for l, h in enumerate(hidden_states[:-1]):
            lens_logits = decode(lens.translators[l](h))
            losses.append(kl_loss(final_logits, lens_logits))

        loss = torch.stack(losses).mean()

    if not loss.requires_grad:
        raise RuntimeError(
            "loss has no grad_fn — is gradient tracking disabled globally? "
            "This cell calls torch.set_grad_enabled(True); re-run from here, or check prior cells."
        )

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(lens.parameters(), GRAD_CLIP)
    optimizer.step()

    pbar.set_postfix(loss=f"{loss.item():.4f}")

print("Training complete.", flush=True)

# %% Save checkpoint
_stem = hf_repo_id_safe_stem(MODEL_ID)
save_path = stonesoup.script_dir() / f"tuned_lens_math_{_stem}.pt"
torch.save(
    {
        "model_id": MODEL_ID,
        "num_hidden_layers_config": n_layers,
        "n_probed_layers": n_probed,
        "n_hidden_states_forward": n_hidden_total,
        "d_model": d_model,
        "corpus": {
            "wikipedia_zh": {"repo": WIKIPEDIA_REPO, "config": WIKIPEDIA_ZH_CONFIG, "rows": WIKIPEDIA_ZH_ROWS},
            "gsm8k": {"repo": GSM8K_REPO, "split": GSM8K_SPLIT, "rows": GSM8K_TRAIN_ROWS},
            "pile_10k": {"repo": PILE_10K_ID, "rows": PILE_TRAIN_ROWS},
            "rys_json": RYS_JSON_REL,
            "train_epochs": TRAIN_EPOCHS,
        },
        "state_dict": lens.state_dict(),
    },
    save_path,
)
print(f"Saved to {save_path}", flush=True)

# %% Evaluate: tuned lens vs logit lens
lens.eval()

(eval_ids,) = next(iter(loader))
eval_ids = eval_ids.to(device)

with torch.no_grad():
    out = text_module(input_ids=eval_ids, output_hidden_states=True, use_cache=False)
    hidden_states = out.hidden_states
    final_logits = lm_head(hidden_states[-1])

    print(f"{'Layer':>5} | {'Logit Lens KL':>14} | {'Tuned Lens KL':>14}", flush=True)
    print("-" * 40, flush=True)
    for l, h in enumerate(hidden_states[:-1]):
        ll_kl = kl_loss(final_logits, decode(h)).item()
        tl_kl = kl_loss(final_logits, decode(lens.translators[l](h))).item()
        print(f"{l:5d} | {ll_kl:14.4f} | {tl_kl:14.4f}", flush=True)
