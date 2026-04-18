# %% Imports
"""
**Attention cut (single prompt)** — compare to ``patchscope.token_identity.minimal.py``.

One Pile document per sample; **no** separate ICL target prompt. From a chosen **minimum layer**
onward (``layer_idx >= target``), the attention output at the last sequence position is zeroed out
via forward hooks, so deeper layers receive no attention contribution at that position (MLP and
residual still flow normally — Pythia uses parallel residual).

Metrics: ``prec_1`` compares top-1 next-token id at the last position — baseline forward vs cut
forward; ``surprisal`` = ``-log p_orig(cut_answer)``.
"""
from __future__ import annotations

from types import SimpleNamespace

import datasets
import pandas as pd
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

import stonesoup
from stonesoup.experiment import (
    configure_matplotlib_agg,
    decoder_blocks,
    ensure_pad_token_via_eos,
    hf_repo_id_safe_stem,
    inner_tokenizer,
    show,
)

configure_matplotlib_agg()


def make_inputs(tokenizer, prompts, device="cuda"):
    token_lists = [tokenizer.encode(p) for p in prompts]
    maxlen = max(len(t) for t in token_lists)
    pad_id = (
        tokenizer.all_special_ids[tokenizer.all_special_tokens.index("<pad>")]
        if "<pad>" in tokenizer.all_special_tokens
        else 0
    )
    input_ids = [[pad_id] * (maxlen - len(t)) + t for t in token_lists]
    attention_mask = [[0] * (maxlen - len(t)) + [1] * len(t) for t in token_lists]
    return dict(
        input_ids=torch.tensor(input_ids).to(device),
        attention_mask=torch.tensor(attention_mask).to(device),
    )


# %% Config
MODEL_NAME = "EleutherAI/pythia-12b"
N_SAMPLES = 100
PILE_FILTER_MAX_WORDS = 250
PILE_FILTER_MAX_CHARS = 2000
PILE_SHUFFLE_SEED = 42
PILE_SHUFFLE_BUFFER_SIZE = 50_000
PILE_DATASET_ID = "EleutherAI/the_pile_deduplicated"
# Evaluate logits / prec at the last sequence index (same index as the attention-cut query row).
POSITION = -1

LAYER_STRIDE = 2
SURPRISAL_PROB_MIN = 1e-12

SAFE_STEM = hf_repo_id_safe_stem(MODEL_NAME)

# %% Stream Pile
raw = datasets.load_dataset(PILE_DATASET_ID, split="train", streaming=True)
short = raw.filter(
    lambda x: len(x["text"].split(" ")) < PILE_FILTER_MAX_WORDS
    and len(x["text"]) < PILE_FILTER_MAX_CHARS,
)
stream = short.shuffle(seed=PILE_SHUFFLE_SEED, buffer_size=PILE_SHUFFLE_BUFFER_SIZE)
pile_texts: list[str] = []
for row in tqdm(stream, total=N_SAMPLES, desc=f"streaming {PILE_DATASET_ID} (short docs)"):
    stonesoup.check_abort()
    pile_texts.append(row["text"])
    if len(pile_texts) >= N_SAMPLES:
        break

assert len(pile_texts) >= N_SAMPLES, (
    f"Only collected {len(pile_texts)} examples (need {N_SAMPLES}); "
    "lower filters or increase shuffle buffer."
)
print(f"streaming pile: collected={len(pile_texts)} documents", flush=True)

# %% Load model
torch.set_grad_enabled(False)

model, processor = stonesoup.load_model(MODEL_NAME, use_offline=True)
model.eval()
model.requires_grad_(False)

tokenizer = inner_tokenizer(processor)
ensure_pad_token_via_eos(tokenizer)

device = next(model.parameters()).device
num_layers = len(decoder_blocks(model))

mt = SimpleNamespace(
    model=model,
    tokenizer=tokenizer,
    device=device,
    num_layers=num_layers,
)

layer_indices = list(range(0, num_layers, LAYER_STRIDE))
print(
    f"model={MODEL_NAME} layers={num_layers} device={device} "
    f"layer_indices (stride={LAYER_STRIDE}): {layer_indices}",
    flush=True,
)

# %% Hook helper: zero out attention output at cut_pos for layer_idx >= target.
# Pythia uses parallel residual: hidden = attn_out + mlp_out + residual.
# By zeroing attn_out at cut_pos, attention contributes nothing at that position
# while MLP and residual proceed normally. This avoids NaN from extreme attn scores.
def _install_cut_hooks(model, *, target_layer_idx: int, cut_pos: int):
    """Register post-hooks on attention modules >= target_layer_idx.
    Returns a list of hook handles for later removal."""
    handles = []

    for layer_idx, block in enumerate(model.gpt_neox.layers):
        if layer_idx < target_layer_idx:
            continue

        def _post_hook(module, args, output, *, _idx=layer_idx):
            # GPTNeoXAttention.forward returns (attn_output, ...) tuple
            attn_output = output[0]
            attn_output[:, cut_pos, :] = 0.0
            return output

        handles.append(block.attention.register_forward_hook(_post_hook))

    return handles


# %% Eval — prec_1 / surprisal at last token (POSITION); cut at ``layer`` and every layer above
records: list[dict] = []

for sample_i in tqdm(range(N_SAMPLES), desc="attention-cut samples"):
    stonesoup.check_abort()
    prompt_source = pile_texts[sample_i]
    inp = make_inputs(mt.tokenizer, [prompt_source], mt.device)
    seq_len = int(inp["input_ids"].shape[1])
    assert seq_len >= 1, "empty sequence"

    output_orig = mt.model(**inp)
    dist_orig = torch.softmax(output_orig.logits[0, POSITION, :], dim=0)
    _, answer_t_orig = torch.max(dist_orig, dim=0)

    for layer in layer_indices:
        stonesoup.check_abort()
        abs_cut_pos = POSITION if POSITION >= 0 else seq_len + POSITION
        hook_handles = _install_cut_hooks(
            mt.model, target_layer_idx=layer, cut_pos=abs_cut_pos,
        )
        out_cut = mt.model(**inp)
        for h in hook_handles:
            h.remove()

        dist = torch.softmax(out_cut.logits[0, POSITION, :], dim=0)
        _, answer_t = torch.max(dist, dim=0)

        prec_1 = bool((answer_t == answer_t_orig).detach().cpu().item())
        # Integer index + float32 avoids rare advanced-indexing / half-precision surprises → NaN surprisal.
        answer_id = int(answer_t.detach().to(torch.long).item())
        p_for_surprisal = dist_orig.float()[answer_id].clamp(min=SURPRISAL_PROB_MIN)
        surprisal = float((-torch.log(p_for_surprisal)).cpu().item())

        records.append(
            {
                "sample": sample_i,
                "layer": layer,
                "prec_1": prec_1,
                "surprisal": surprisal,
            }
        )

# %% Save & summary
stonesoup.html()
results = pd.DataFrame.from_records(records)
out_csv = stonesoup.outputs_dir() / "cutting_attention_method.csv"
results.to_csv(out_csv, index=False)
print("wrote", out_csv.relative_to(stonesoup.repo_root()), flush=True)

_nr, _nc = len(results), len(results.columns)
stonesoup.display(results, max_rows=max(_nr, 1), max_cols=max(_nc, 1))

stonesoup.html()
by_layer = results.groupby("layer", as_index=False).agg(
    mean_prec=("prec_1", "mean"),
    mean_surprisal=("surprisal", "mean"),
    n=("prec_1", "count"),
)
print("mean prec_1 and surprisal by layer (attention cut):", flush=True)
stonesoup.display(by_layer)

# %% Plot
FIG_ORANGE = "#e67e22"
fig, axes = plt.subplots(2, 1, figsize=(6, 9), sharex=True)
axes[0].plot(
    by_layer["layer"],
    by_layer["mean_prec"],
    color=FIG_ORANGE,
    marker="o",
    lw=2,
    markersize=5,
    label="Attention cut ≥ layer (last pos)",
)
axes[0].set_ylabel("mean prec_1")
axes[0].set_title(f"Attention cut — {MODEL_NAME.strip('./')}")
axes[0].grid(True, alpha=0.35)
axes[0].set_ylim(0, 1.05)
axes[0].legend(loc="best")

axes[1].plot(
    by_layer["layer"],
    by_layer["mean_surprisal"],
    color=FIG_ORANGE,
    marker="o",
    lw=2,
    markersize=5,
)
axes[1].set_xlabel("layer")
axes[1].set_ylabel("mean surprisal (nats)")
axes[1].grid(True, alpha=0.35)

plt.tight_layout()
show(fig, basename=f"{SAFE_STEM}_cutting_attention_fig2", dpi=120)
plt.close(fig)
