# %% Imports & patchscopes path
"""
This is the correct implementation of Token Identity.
The resulting plots are similar to Figure 2 in the paper.
"""
from __future__ import annotations

import random
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
    """Prepare batched ``input_ids`` / ``attention_mask`` (left-padded).

    Copied from Patchscopes ``general_utils.py`` (ROME-style) so this experiment stays standalone.
    """

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
SOS_TOK = False
N_SAMPLES = 1000
PILE_FILTER_MAX_WORDS = 250
PILE_FILTER_MAX_CHARS = 2000
PILE_SHUFFLE_SEED = 42
PILE_SHUFFLE_BUFFER_SIZE = 50_000
RANDOM_POSITION_SEED = 42
PILE_DATASET_ID = "EleutherAI/the_pile_deduplicated"

SAFE_STEM = hf_repo_id_safe_stem(MODEL_NAME)

PROMPT_TARGET = "cat -> cat\n1135 -> 1135\nhello -> hello\n?"
POSITION_TARGET = -1
POSITION_PREDICTION = POSITION_TARGET
# Evaluate Token Identity at layers ``0, stride, 2*stride, ...`` (not every block).
LAYER_STRIDE = 5
# Floor for ``-log p`` surprisal when softmax underflows to 0. Apply in **float32**: half-precision
# ``clamp(min=1e-12)`` can still round the floor to 0 → ``-log(0)==inf``.
SURPRISAL_PROB_MIN = 1e-12

# %% Stream The Pile (short docs, shuffled)
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

assert hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"), (
    f"Expected GPT-NeoX stack (e.g. Pythia); got {type(model).__name__}."
)
assert hasattr(model.gpt_neox, "final_layer_norm"), "Expected gpt_neox.final_layer_norm (HF GPTNeoX)."

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

# %% Token Identity — N_SAMPLES × len(layer_indices) (same layer for source and target, transform=None)
# One source forward per sample (all layers’ hiddens); one patched target forward per sampled layer.
# ``evaluate_patch_next_token_prediction``-equivalent; no Tuned Lens.

inp_target = make_inputs(mt.tokenizer, [PROMPT_TARGET], mt.device)
_tlen = len(inp_target["input_ids"][0])
position_target_abs = (
    _tlen + POSITION_TARGET if POSITION_TARGET < 0 else POSITION_TARGET
)
position_prediction_idx = (
    _tlen + POSITION_PREDICTION if POSITION_PREDICTION < 0 else POSITION_PREDICTION
)

random.seed(RANDOM_POSITION_SEED)
start_pos = int(SOS_TOK)

records: list[dict] = []

for sample_i in tqdm(range(N_SAMPLES), desc="token-identity samples"):
    stonesoup.check_abort()
    prompt_source = pile_texts[sample_i]

    inp_source = make_inputs(mt.tokenizer, [prompt_source], mt.device)
    seq_len = int(inp_source["input_ids"].shape[1])
    hi = seq_len - 2
    assert hi >= start_pos, f"Pile doc too short for random position: seq_len={seq_len}"
    position_source = random.randint(start_pos, hi)

    for _layer in model.gpt_neox.layers:
        _layer._forward_hooks.clear()
        _layer._forward_pre_hooks.clear()
    _fln = model.gpt_neox.final_layer_norm
    _fln._forward_hooks.clear()
    _fln._forward_pre_hooks.clear()
    setattr(model.gpt_neox, "_output_capturing_hooks_installed", False)

    output_orig = mt.model(**inp_source, output_hidden_states=True)
    dist_orig = torch.softmax(output_orig.logits[0, position_source, :], dim=0)
    _, answer_t_orig = torch.max(dist_orig, dim=0)

    for layer in layer_indices:
        stonesoup.check_abort()
        hidden_rep = output_orig["hidden_states"][layer + 1][0][position_source]
        skip_final_ln = layer == num_layers - 1

        def _post_patch_hs(module, input, output):
            assert isinstance(output, torch.Tensor) and output.dim() == 3
            output[0, position_target_abs, :] = hidden_rep

        _hook_mod = model.gpt_neox.final_layer_norm if skip_final_ln else model.gpt_neox.layers[layer]
        _hook_handle = _hook_mod.register_forward_hook(_post_patch_hs)

        output = mt.model(**inp_target)
        _hook_handle.remove()

        dist = torch.softmax(output.logits[0, position_prediction_idx, :], dim=0)
        _, answer_t = torch.max(dist, dim=0)

        prec_1 = bool((answer_t == answer_t_orig).detach().cpu().item())
        p_for_surprisal = dist_orig[answer_t].float().clamp(min=SURPRISAL_PROB_MIN)
        surprisal = float((-torch.log(p_for_surprisal)).cpu().item())

        records.append(
            {
                "sample": sample_i,
                "layer": layer,
                "position_source": position_source,
                "prec_1": prec_1,
                "surprisal": surprisal,
            }
        )

# %% Save results & summary
stonesoup.html()
results = pd.DataFrame.from_records(records)
out_csv = stonesoup.outputs_dir() / "patchscope_token_identity_minimal.csv"
results.to_csv(out_csv, index=False)
print("wrote", out_csv.relative_to(stonesoup.repo_root()), flush=True)

# %% full results (``stonesoup.display`` defaults to 30×20; override to show every row/column)
_nr, _nc = len(results), len(results.columns)
stonesoup.display(results, max_rows=max(_nr, 1), max_cols=max(_nc, 1))

# %% summary
stonesoup.html()
by_layer = results.groupby("layer", as_index=False).agg(
    mean_prec=("prec_1", "mean"),
    mean_surprisal=("surprisal", "mean"),
    n=("prec_1", "count"),
)
print("mean prec_1 and surprisal by layer:", flush=True)
stonesoup.display(by_layer)

# %% Plot (Fig. 2–style: layer on x; green curves for Token Identity — mean prec_1 and mean surprisal)
FIG_GREEN = "#2ca02c"
fig, axes = plt.subplots(2, 1, figsize=(6, 9), sharex=True)
axes[0].plot(
    by_layer["layer"],
    by_layer["mean_prec"],
    color=FIG_GREEN,
    marker="o",
    lw=2,
    markersize=5,
    label="Token Identity",
)
axes[0].set_ylabel("mean prec_1")
axes[0].set_title(f"Token Identity — {MODEL_NAME.strip('./')}")
axes[0].grid(True, alpha=0.35)
axes[0].set_ylim(0, 1.05)
axes[0].legend(loc="best")

axes[1].plot(
    by_layer["layer"],
    by_layer["mean_surprisal"],
    color=FIG_GREEN,
    marker="o",
    lw=2,
    markersize=5,
)
axes[1].set_xlabel("layer")
axes[1].set_ylabel("mean surprisal (nats)")
axes[1].grid(True, alpha=0.35)

plt.tight_layout()
show(fig, basename=f"{SAFE_STEM}_token_identity_fig2", dpi=120)
plt.close(fig)
