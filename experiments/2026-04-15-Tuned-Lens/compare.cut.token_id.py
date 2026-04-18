# %% Imports
"""
Fair comparison of **Token Identity** (Patchscopes) vs **Attention Cut** on the same
samples, positions, and layer grid. Both methods share one baseline forward per sample.
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
N_SAMPLES = 1_000
LAYER_STRIDE = 1
PILE_FILTER_MAX_WORDS = 250
PILE_FILTER_MAX_CHARS = 2000
PILE_SHUFFLE_SEED = 42
PILE_SHUFFLE_BUFFER_SIZE = 50_000
RANDOM_POSITION_SEED = 42
PILE_DATASET_ID = "EleutherAI/the_pile_deduplicated"
SURPRISAL_PROB_MIN = 1e-12

# Token Identity target prompt
PROMPT_TARGET = "cat -> cat\n1135 -> 1135\nhello -> hello\n?"
POSITION_TARGET = -1

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

assert len(pile_texts) >= N_SAMPLES
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

mt = SimpleNamespace(model=model, tokenizer=tokenizer, device=device, num_layers=num_layers)

layer_indices = list(range(0, num_layers, LAYER_STRIDE))
print(
    f"model={MODEL_NAME} layers={num_layers} device={device} "
    f"layer_indices (stride={LAYER_STRIDE}): {layer_indices}",
    flush=True,
)


# %% Attention Cut helper
def _install_cut_hooks(model, *, target_layer_idx: int, cut_pos: int):
    """Zero out attention output at cut_pos for layers >= target_layer_idx."""
    handles = []
    for layer_idx, block in enumerate(model.gpt_neox.layers):
        if layer_idx < target_layer_idx:
            continue

        def _post_hook(module, args, output, *, _idx=layer_idx):
            attn_output = output[0]
            attn_output[:, cut_pos, :] = 0.0
            return output

        handles.append(block.attention.register_forward_hook(_post_hook))
    return handles


# %% Token Identity setup
inp_target = make_inputs(mt.tokenizer, [PROMPT_TARGET], mt.device)
_tlen = len(inp_target["input_ids"][0])
position_target_abs = _tlen + POSITION_TARGET if POSITION_TARGET < 0 else POSITION_TARGET
position_prediction_idx = position_target_abs

# %% Eval loop — both methods on each sample
random.seed(RANDOM_POSITION_SEED)
records: list[dict] = []

for sample_i in tqdm(range(N_SAMPLES), desc="compare samples"):
    stonesoup.check_abort()
    prompt_source = pile_texts[sample_i]
    inp_source = make_inputs(mt.tokenizer, [prompt_source], mt.device)
    seq_len = int(inp_source["input_ids"].shape[1])
    assert seq_len >= 2
    position_source = random.randint(0, seq_len - 2)

    # Shared baseline forward
    output_orig = mt.model(**inp_source, output_hidden_states=True)
    dist_orig = torch.softmax(output_orig.logits[0, position_source, :], dim=0)
    _, answer_t_orig = torch.max(dist_orig, dim=0)

    for layer in layer_indices:
        stonesoup.check_abort()

        # --- Token Identity ---
        hidden_rep = output_orig["hidden_states"][layer + 1][0][position_source]
        skip_final_ln = layer == num_layers - 1
        hook_mod = model.gpt_neox.final_layer_norm if skip_final_ln else model.gpt_neox.layers[layer]

        def _post_patch_hs(module, input, output):
            assert isinstance(output, torch.Tensor) and output.dim() == 3
            output[0, position_target_abs, :] = hidden_rep

        handle = hook_mod.register_forward_hook(_post_patch_hs)
        out_ti = mt.model(**inp_target)
        handle.remove()

        dist_ti = torch.softmax(out_ti.logits[0, position_prediction_idx, :], dim=0)
        _, answer_ti = torch.max(dist_ti, dim=0)
        prec_ti = bool((answer_ti == answer_t_orig).item())
        p_ti = dist_orig[answer_ti].float().clamp(min=SURPRISAL_PROB_MIN)
        surp_ti = float((-torch.log(p_ti)).item())

        # --- Attention Cut ---
        abs_cut_pos = seq_len + POSITION_TARGET  # cut at last position (same as token identity eval pos)
        hook_handles = _install_cut_hooks(mt.model, target_layer_idx=layer, cut_pos=position_source)
        out_cut = mt.model(**inp_source)
        for h in hook_handles:
            h.remove()

        dist_cut = torch.softmax(out_cut.logits[0, position_source, :], dim=0)
        _, answer_cut = torch.max(dist_cut, dim=0)
        prec_cut = bool((answer_cut == answer_t_orig).item())
        answer_cut_id = int(answer_cut.to(torch.long).item())
        p_cut = dist_orig.float()[answer_cut_id].clamp(min=SURPRISAL_PROB_MIN)
        surp_cut = float((-torch.log(p_cut)).item())

        records.append(
            {
                "sample": sample_i,
                "layer": layer,
                "position_source": position_source,
                "ti_prec_1": prec_ti,
                "ti_surprisal": surp_ti,
                "cut_prec_1": prec_cut,
                "cut_surprisal": surp_cut,
            }
        )

# %% Save & summary
stonesoup.html()
results = pd.DataFrame.from_records(records)
out_csv = stonesoup.outputs_dir() / "compare_cut_token_id.csv"
results.to_csv(out_csv, index=False)
print("wrote", out_csv.relative_to(stonesoup.repo_root()), flush=True)

_nr, _nc = len(results), len(results.columns)
stonesoup.display(results, max_rows=max(_nr, 1), max_cols=max(_nc, 1))

stonesoup.html()
by_layer = results.groupby("layer", as_index=False).agg(
    ti_mean_prec=("ti_prec_1", "mean"),
    ti_mean_surprisal=("ti_surprisal", "mean"),
    cut_mean_prec=("cut_prec_1", "mean"),
    cut_mean_surprisal=("cut_surprisal", "mean"),
    n=("ti_prec_1", "count"),
)
print("by-layer summary:", flush=True)
stonesoup.display(by_layer)

# %% Plot
FIG_GREEN = "#2ca02c"
FIG_ORANGE = "#e67e22"
fig, axes = plt.subplots(2, 1, figsize=(8, 9), sharex=True)

axes[0].plot(by_layer["layer"], by_layer["ti_mean_prec"], color=FIG_GREEN, marker="o", lw=2, markersize=5, label="Token Identity")
axes[0].plot(by_layer["layer"], by_layer["cut_mean_prec"], color=FIG_ORANGE, marker="s", lw=2, markersize=5, label="Attention Cut")
axes[0].set_ylabel("mean prec_1")
axes[0].set_title(f"Token Identity vs Attention Cut — {MODEL_NAME.strip('./')}")
axes[0].grid(True, alpha=0.35)
axes[0].set_ylim(0, 1.05)
axes[0].legend(loc="best")

axes[1].plot(by_layer["layer"], by_layer["ti_mean_surprisal"], color=FIG_GREEN, marker="o", lw=2, markersize=5, label="Token Identity")
axes[1].plot(by_layer["layer"], by_layer["cut_mean_surprisal"], color=FIG_ORANGE, marker="s", lw=2, markersize=5, label="Attention Cut")
axes[1].set_xlabel("layer")
axes[1].set_ylabel("mean surprisal (nats)")
axes[1].grid(True, alpha=0.35)
axes[1].legend(loc="best")

plt.tight_layout()
show(fig, basename=f"{SAFE_STEM}_compare_cut_token_id", dpi=120)
plt.close(fig)
