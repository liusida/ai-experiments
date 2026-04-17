# %% Imports
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

import stonesoup
from stonesoup.experiment import configure_matplotlib_agg, ensure_pad_token_via_eos, inner_tokenizer

configure_matplotlib_agg()

import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset

# %% Config
MODEL_ID = "Qwen/Qwen3.5-0.8B"
SEQ_LEN = 256
BATCH_SIZE = 2
LR = 1e-3
NUM_STEPS = 200
GRAD_CLIP = 1.0

# %% Load model and detect architecture
model, proc = stonesoup.load_model(MODEL_ID, use_offline=True, torch_dtype="bfloat16")
model.eval()
model.requires_grad_(False)

tokenizer = inner_tokenizer(proc)
ensure_pad_token_via_eos(tokenizer)

device = next(model.parameters()).device
text_cfg = getattr(model.config, "text_config", model.config)
d_model = text_cfg.hidden_size
n_layers = text_cfg.num_hidden_layers

_inner = model.model
text_module = getattr(_inner, "language_model", _inner)
final_norm = text_module.norm
lm_head = model.lm_head

print(f"Model: {MODEL_ID}", flush=True)
print(f"  class={type(model).__name__}, d_model={d_model}, n_layers={n_layers}, device={device}", flush=True)

# %% Prepare dataset
ds = load_dataset("NeelNanda/pile-10k", split="train")

all_ids: list[int] = []
for text in ds["text"]:
    all_ids.extend(tokenizer.encode(text, add_special_tokens=False))

n_chunks = len(all_ids) // SEQ_LEN
all_ids_t = torch.tensor(all_ids[: n_chunks * SEQ_LEN], dtype=torch.long).view(n_chunks, SEQ_LEN)
print(f"Tokenized: {n_chunks} chunks of {SEQ_LEN} tokens", flush=True)

loader = DataLoader(TensorDataset(all_ids_t), batch_size=BATCH_SIZE, shuffle=True)

# %% Token coverage histogram
stonesoup.html() # force HTML mode, since we will print something before drawing the plot.
vocab_size = text_cfg.vocab_size  # text-only vocab (LM head dim), not tokenizer which may include image tokens
token_counts = torch.bincount(torch.tensor(all_ids, dtype=torch.long), minlength=vocab_size)[:vocab_size]
seen_mask = token_counts > 0
n_seen = int(seen_mask.sum().item())
n_unseen = vocab_size - n_seen

print(f"Text vocab size: {vocab_size:,}  (tokenizer.vocab_size={tokenizer.vocab_size:,})", flush=True)
print(f"Tokens seen in training data: {n_seen:,} ({n_seen / vocab_size:.1%})", flush=True)
print(f"Tokens never seen: {n_unseen:,} ({n_unseen / vocab_size:.1%})", flush=True)

# Show some example unseen tokens
unseen_ids = torch.where(~seen_mask)[0].tolist()
sample_unseen = unseen_ids[::max(1, len(unseen_ids) // 20)][:20]
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

# %% Define TunedLens, decode helper, and KL loss

class AffineTranslator(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.weight = nn.Parameter(torch.eye(d))
        self.bias = nn.Parameter(torch.zeros(d))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # Compute in float32 for precision, cast back to input dtype for frozen decode path
        return (h.float() @ self.weight.T + self.bias).to(h.dtype)


class TunedLens(nn.Module):
    def __init__(self, n_probed: int, d: int):
        super().__init__()
        self.translators = nn.ModuleList([AffineTranslator(d) for _ in range(n_probed)])


def decode(h: torch.Tensor) -> torch.Tensor:
    """Norm + LM-head: same path the frozen model uses for final logits."""
    return lm_head(final_norm(h))


def kl_loss(final_logits: torch.Tensor, lens_logits: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        target_lp = F.log_softmax(final_logits.float(), dim=-1)
        target_p = target_lp.exp()
    pred_lp = F.log_softmax(lens_logits.float(), dim=-1)
    return (target_p * (target_lp - pred_lp)).sum(dim=-1).mean()


lens = TunedLens(n_layers, d_model).to(device)
print(f"TunedLens: {n_layers} translators, {sum(p.numel() for p in lens.parameters()):,} params", flush=True)

# %% Train
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
        hidden_states = out.hidden_states  # (n_layers+1) tensors; last is post-norm
        final_logits = lm_head(hidden_states[-1])

    losses = []
    for l, h in enumerate(hidden_states[:-1]):
        lens_logits = decode(lens.translators[l](h))
        losses.append(kl_loss(final_logits, lens_logits))

    loss = torch.stack(losses).mean()

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(lens.parameters(), GRAD_CLIP)
    optimizer.step()

    pbar.set_postfix(loss=f"{loss.item():.4f}")

print("Training complete.", flush=True)

# %% Save checkpoint
save_path = stonesoup.script_dir() / "tuned_lens.pt"
torch.save({"n_layers": n_layers, "d_model": d_model, "state_dict": lens.state_dict()}, save_path)
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
