# %% Imports & patchscopes reference path
from __future__ import annotations

import random
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
import seaborn as sns
from tqdm import tqdm

import datasets
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
torch.set_grad_enabled(False)

sns.set(
    context="notebook",
    rc={
        "font.size": 16,
        "axes.titlesize": 16,
        "axes.labelsize": 16,
        "xtick.labelsize": 16.0,
        "ytick.labelsize": 16.0,
        "legend.fontsize": 16.0,
    },
)
sns.set_theme(style="whitegrid")

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_PATCHSCOPES_CODE = _REPO_ROOT / "references" / "interpretability" / "patchscopes" / "code"
if not _PATCHSCOPES_CODE.is_dir():
    raise FileNotFoundError(
        f"Expected Patchscopes reference code at {_PATCHSCOPES_CODE} (clone or sync references/interpretability)."
    )
_patch_path = str(_PATCHSCOPES_CODE)
if _patch_path not in sys.path:
    sys.path.insert(0, _patch_path)

# Google Research reference helpers (same modules as the upstream notebook).
from general_utils import make_inputs  # type: ignore[import-not-found]
from patchscopes_utils import (  # type: ignore[import-not-found]
    evaluate_patch_next_token_prediction as _evaluate_patch_next_token_prediction_raw,
    set_hs_patch_hooks_gptj,
    set_hs_patch_hooks_llama,
)


def _clear_module_forward_hooks_only(m: torch.nn.Module) -> None:
    fh = getattr(m, "_forward_hooks", None)
    if fh is not None:
        fh.clear()
    fph = getattr(m, "_forward_pre_hooks", None)
    if fph is not None:
        fph.clear()


def _reset_hf_output_capturing_install_flag(inner: torch.nn.Module | None) -> None:
    """Clearing layer hooks also removes HF ``output_hidden_states`` capture hooks (transformers 4.5+).

    If ``_output_capturing_hooks_installed`` stays True, ``maybe_install_capturing_hooks`` skips
    reinstall and ``hidden_states`` is empty → Patchscopes indexing raises ``IndexError``.
    """

    if inner is not None and hasattr(inner, "_output_capturing_hooks_installed"):
        setattr(inner, "_output_capturing_hooks_installed", False)


def clear_patchscope_hs_hook_targets(model: torch.nn.Module) -> None:
    """Remove Patchscopes hooks from decoder blocks (stale hooks after a failed eval break the next run)."""
    if hasattr(model, "gpt_neox"):
        for layer in model.gpt_neox.layers:
            _clear_module_forward_hooks_only(layer)
        if hasattr(model.gpt_neox, "final_layer_norm"):
            _clear_module_forward_hooks_only(model.gpt_neox.final_layer_norm)
        _reset_hf_output_capturing_install_flag(model.gpt_neox)
        return
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        for blk in model.transformer.h:
            _clear_module_forward_hooks_only(blk)
        if hasattr(model.transformer, "ln_f"):
            _clear_module_forward_hooks_only(model.transformer.ln_f)
        _reset_hf_output_capturing_install_flag(model.transformer)
        return
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "layers"):
        for layer in inner.layers:
            _clear_module_forward_hooks_only(layer)
        if hasattr(inner, "norm"):
            _clear_module_forward_hooks_only(inner.norm)
        _reset_hf_output_capturing_install_flag(inner)


def evaluate_patch_next_token_prediction(mt, *args, **kwargs):
    clear_patchscope_hs_hook_targets(mt.model)
    # Always use NeoX tensor-output hooks (partial re-runs can leave upstream ``set_hs_patch_hooks_neox`` on ``mt``).
    if hasattr(mt.model, "gpt_neox") and hasattr(mt.model.gpt_neox, "layers"):
        mt.set_hs_patch_hooks = set_hs_patch_hooks_neox_transformers_compat
    return _evaluate_patch_next_token_prediction_raw(mt, *args, **kwargs)


def set_hs_patch_hooks_neox_transformers_compat(
    model,
    hs_patch_config,
    module="hs",
    patch_input=False,
    skip_final_ln=False,
    generation_mode=False,
):
    """Patchscopes NeoX hooks, fixed for HF ``GPTNeoXLayer`` returning a Tensor.

    Upstream assumed ``output`` was tuple-like so ``output[0]`` is ``(B,S,H)``. For a Tensor
    ``(B,S,H)``, ``output[0]`` is the first batch slice ``(S,H)``, and ``output[0][0, pos]``
    indexes the wrong axes (hidden dim vs sequence), which triggers ``RuntimeError: expand``.
    """

    if module != "hs":
        raise ValueError("Module %s not yet supported" % module)

    def patch_hs(name, position_hs, patch_input, generation_mode):
        def pre_hook(module, input):
            input_len = len(input[0][0])
            if generation_mode and input_len == 1:
                return
            for position_, hs_ in position_hs:
                input[0][0, position_] = hs_

        def post_hook(module, input, output):
            if "skip_ln" in name:
                if isinstance(output, torch.Tensor):
                    output_len = output.shape[1]
                else:
                    output_len = len(output[0])
                if generation_mode and output_len == 1:
                    return
                for position_, hs_ in position_hs:
                    if isinstance(output, torch.Tensor):
                        output[0, position_, :] = hs_
                    else:
                        output[0][position_] = hs_
            else:
                hidden = output if isinstance(output, torch.Tensor) else output[0]
                output_len = hidden.shape[1] if hidden.dim() == 3 else len(hidden[0])
                if generation_mode and output_len == 1:
                    return
                for position_, hs_ in position_hs:
                    hidden[0, position_, :] = hs_

        if patch_input:
            return pre_hook
        return post_hook

    hooks = []
    for i in hs_patch_config:
        if patch_input:
            hooks.append(
                model.gpt_neox.layers[i].register_forward_pre_hook(
                    patch_hs(
                        f"patch_hs_{i}",
                        hs_patch_config[i],
                        patch_input,
                        generation_mode,
                    )
                )
            )
        else:
            if skip_final_ln and i == len(model.gpt_neox.layers) - 1:
                hooks.append(
                    model.gpt_neox.final_layer_norm.register_forward_hook(
                        patch_hs(
                            f"patch_hs_{i}_skip_ln",
                            hs_patch_config[i],
                            patch_input,
                            generation_mode,
                        )
                    )
                )
            else:
                hooks.append(
                    model.gpt_neox.layers[i].register_forward_hook(
                        patch_hs(
                            f"patch_hs_{i}",
                            hs_patch_config[i],
                            patch_input,
                            generation_mode,
                        )
                    )
                )

    return hooks


def patchscope_set_hs_patch_hooks(model: torch.nn.Module):
    """Pick Patchscopes ``set_hs_patch_hooks_*`` from the loaded HF module tree (not repo id)."""
    cfg = getattr(model, "config", None)
    model_type = getattr(cfg, "model_type", None) if cfg is not None else None

    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return set_hs_patch_hooks_neox_transformers_compat
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        if model_type == "gptj":
            return set_hs_patch_hooks_gptj
        raise TypeError(
            f"Decoder stack is transformer.h but model_type={model_type!r}. "
            "Patchscopes GPT-J hooks are only wired for model_type 'gptj' (not e.g. gpt2)."
        )
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "layers"):
        return set_hs_patch_hooks_llama
    raise TypeError(
        f"No Patchscopes hook implementation matched "
        f"{type(model).__name__} (model_type={model_type!r})."
    )


# %% Config knobs
MODEL_NAME = "EleutherAI/pythia-12b"
SOS_TOK = False
TRN_N = 100 # 10_000
VAL_N = 20 # 2_000
PILE_FILTER_MAX_WORDS = 250
PILE_FILTER_MAX_CHARS = 2000
PILE_SHUFFLE_SEED = 42
PILE_SHUFFLE_BUFFER_SIZE = 50_000
RANDOM_POSITION_SEED = 42

# Stream The Pile (deduplicated) from the Hub — only shards you traverse are fetched, not the full ~451GB mirror.
PILE_DATASET_ID = "EleutherAI/the_pile_deduplicated"

# Prompt-id evaluation (notebook cell 14): apply pile L→last affine to the patched
# vector before the target forward (same `mappings[layer]` as affine eval, not prompt-id maps).
APPLY_PROMPT_ID_AFFINE = False

# Log prompt-id inputs before each ``evaluate_patch_next_token_prediction`` (very verbose if True).
DEBUG_PRINT_PROMPT_ID_EVAL = True

OUT_DIR = stonesoup.outputs_dir()
SAFE_STEM = hf_repo_id_safe_stem(MODEL_NAME)
print("outputs_dir:", OUT_DIR.relative_to(stonesoup.repo_root()), flush=True)
print("safe_stem:", SAFE_STEM, flush=True)

# %% Load model (Stonesoup shared pool + Patchscopes-shaped adapter)
# Requires a Stonesoup cell kernel (toolbar Load or this call shares the process-wide pool).
model_torch_dtype = "float16" if ("13b" in MODEL_NAME or "12b" in MODEL_NAME) else None
model, processor = stonesoup.load_model(MODEL_NAME, use_offline=False, torch_dtype=model_torch_dtype)
model.eval()
model.requires_grad_(False)
tokenizer = inner_tokenizer(processor)
ensure_pad_token_via_eos(tokenizer)
_device = next(model.parameters()).device
_num_layers = len(decoder_blocks(model))

hook_fn = patchscope_set_hs_patch_hooks(model)
mt = SimpleNamespace(
    model=model,
    tokenizer=tokenizer,
    device=_device,
    num_layers=_num_layers,
)
mt.set_hs_patch_hooks = hook_fn
print(
    f"Patchscopes bundle: {MODEL_NAME}, layers={_num_layers}, device={_device}, "
    f"hooks={hook_fn.__name__}",
    flush=True,
)

# %% Load Pile subset and sentence list (HF streaming)
def _iter_pile_short_shuffled():
    raw = datasets.load_dataset(
        PILE_DATASET_ID,
        split="train",
        streaming=True,
    )
    short = raw.filter(
        lambda x: len(x["text"].split(" ")) < PILE_FILTER_MAX_WORDS
        and len(x["text"]) < PILE_FILTER_MAX_CHARS,
    )
    return short.shuffle(seed=PILE_SHUFFLE_SEED, buffer_size=PILE_SHUFFLE_BUFFER_SIZE)


needed = TRN_N + VAL_N
pile_texts: list[str] = []
stream = _iter_pile_short_shuffled()
for row in tqdm(stream, total=needed, desc=f"streaming {PILE_DATASET_ID} (short docs)"):
    stonesoup.check_abort()
    pile_texts.append(row["text"])
    if len(pile_texts) >= needed:
        break

if len(pile_texts) < needed:
    raise RuntimeError(
        f"Only collected {len(pile_texts)} examples (need {needed}). "
        "Try lowering filters or increasing stream iterations / buffer_size."
    )

pile_trn = pile_texts[:TRN_N]
pile_val = pile_texts[TRN_N : TRN_N + VAL_N]
sentences = [(x, "train") for x in pile_trn] + [(x, "validation") for x in pile_val]
print(f"streaming pile: train={len(pile_trn)} val={len(pile_val)}", flush=True)

# %% Build dataframe — same-prompt hidden sweep (per-layer vectors at one position)
random.seed(RANDOM_POSITION_SEED)
data: dict = {}
for sentence, split in tqdm(sentences, desc="same-prompt cache"):
    stonesoup.check_abort()
    inp = make_inputs(mt.tokenizer, [sentence], device=mt.model.device)
    start_pos = 1 if SOS_TOK else 0
    position = random.randint(start_pos, len(inp["input_ids"][0]) - 1)

    key = (sentence, position, split)
    if key not in data:
        output = mt.model(**inp, output_hidden_states=True)
        data[key] = [
            output["hidden_states"][layer + 1][0][position].detach().cpu().numpy()
            for layer in range(mt.num_layers)
        ]

df_same_prompt = pd.Series(data).reset_index()
df_same_prompt.columns = ["full_text", "position", "data_split", "hidden_rep"]
pickle_same = OUT_DIR / f"{SAFE_STEM}_pile_trn_val_same_prompt.pkl"
df_same_prompt.to_pickle(pickle_same)
print("wrote", pickle_same.relative_to(stonesoup.repo_root()), flush=True)

# %% Build dataframe — prompt-id paired source/target hiddens
PROMPT_TARGET = "cat -> cat\n1135 -> 1135\nhello -> hello\n?"
inp_target_template = make_inputs(mt.tokenizer, [PROMPT_TARGET], device=mt.model.device)

random.seed(RANDOM_POSITION_SEED)
data_pid: dict = {}
for sentence, split in tqdm(sentences, desc="prompt-id cache"):
    stonesoup.check_abort()
    inp = make_inputs(mt.tokenizer, [sentence], device=mt.model.device)
    start_pos = 1 if SOS_TOK else 0
    position = random.randint(start_pos, len(inp["input_ids"][0]) - 2)

    key_src = (sentence, position, split, "source")
    if key_src not in data_pid:
        output = mt.model(**inp, output_hidden_states=True)
        _, answer_t = torch.max(torch.softmax(output.logits[0, -1, :], dim=0), dim=0)
        data_pid[key_src] = [
            output["hidden_states"][layer + 1][0][position].detach().cpu().numpy()
            for layer in range(mt.num_layers)
        ]

        inp_target = {k: v.clone() for k, v in inp_target_template.items()}
        inp_target["input_ids"][0][-1] = answer_t
        output_tgt = mt.model(**inp_target, output_hidden_states=True)
        data_pid[(sentence, position, split, "target")] = [
            output_tgt["hidden_states"][layer + 1][0][-1].detach().cpu().numpy()
            for layer in range(mt.num_layers)
        ]

df_prompt_id = pd.Series(data_pid).reset_index()
df_prompt_id.columns = ["full_text", "position", "data_split", "prompt", "hidden_rep"]
pickle_pid = OUT_DIR / f"{SAFE_STEM}_pile_trn_val_prompt_id.pkl"
df_prompt_id.to_pickle(pickle_pid)
print("wrote", pickle_pid.relative_to(stonesoup.repo_root()), flush=True)

# %% Pad / unpad (homogeneous affine coordinates for lstsq)
def pad(x: np.ndarray) -> np.ndarray:
    return np.hstack([x, np.ones((x.shape[0], 1))])


def unpad(x: np.ndarray) -> np.ndarray:
    return x[:, :-1]


def torch_row_affine(matrix: np.ndarray, dev: torch.device):
    """Apply one row of (padded) affine map to a 1D activation; returns tensor on ``dev``.

    Must match the model hidden dtype (e.g. float16); float64 patches break GPT-NeoX hook assignment.
    """

    def _tf(x: torch.Tensor) -> torch.Tensor:
        row = np.expand_dims(x.detach().float().cpu().numpy(), 0)
        out = unpad(np.dot(pad(row), matrix)).reshape(-1)
        vec = torch.from_numpy(np.asarray(out, dtype=np.float32)).to(
            device=dev, dtype=x.dtype
        )
        return vec

    return _tf


# %% Fit affine maps — pile layer L → last layer (notebook cell 9)
mappings_dir = OUT_DIR / f"{SAFE_STEM}_mappings_pile"
mappings_dir.mkdir(parents=True, exist_ok=True)

df_trn = pd.DataFrame(
    df_same_prompt[df_same_prompt["data_split"] == "train"]["hidden_rep"].to_list(),
    columns=[layer for layer in range(mt.num_layers)],
)
target_layer = mt.num_layers - 1
Y = np.array(df_trn[target_layer].values.tolist())

for layer in range(mt.num_layers):
    stonesoup.check_abort()
    X = np.array(df_trn[layer].values.tolist())
    A, _res, _rank, _s = np.linalg.lstsq(pad(X), pad(Y), rcond=None)

    def transform_np(x_arr: np.ndarray, mat: np.ndarray = A) -> np.ndarray:
        return unpad(pad(x_arr) @ mat)

    err = np.abs(Y - transform_np(X)).max()
    print(layer, "max error on train (pile map):", err, flush=True)
    np.save(mappings_dir / f"mapping_{layer}-{target_layer}.npy", A)

shutil.make_archive(str(mappings_dir), "zip", root_dir=mappings_dir)
print("archive:", str(mappings_dir) + ".zip", flush=True)

# %% Fit affine maps — prompt-id source → target per layer (notebook cell 10)
mappings_prompt_id_dir = OUT_DIR / f"{SAFE_STEM}_mappings_pile_prompt-id"
mappings_prompt_id_dir.mkdir(parents=True, exist_ok=True)

df_trn_src = pd.DataFrame(
    df_prompt_id[(df_prompt_id["data_split"] == "train") & (df_prompt_id["prompt"] == "source")][
        "hidden_rep"
    ].to_list(),
    columns=[layer for layer in range(mt.num_layers)],
)
df_trn_tgt = pd.DataFrame(
    df_prompt_id[(df_prompt_id["data_split"] == "train") & (df_prompt_id["prompt"] == "target")][
        "hidden_rep"
    ].to_list(),
    columns=[layer for layer in range(mt.num_layers)],
)

for layer in range(mt.num_layers):
    stonesoup.check_abort()
    X = np.array(df_trn_src[layer].values.tolist())
    Y_tgt = np.array(df_trn_tgt[layer].values.tolist())
    A_pid, _res, _rank, _s = np.linalg.lstsq(pad(X), pad(Y_tgt), rcond=None)

    def transform_pid_np(x_arr: np.ndarray, mat: np.ndarray = A_pid) -> np.ndarray:
        return unpad(pad(x_arr) @ mat)

    err = np.abs(Y_tgt - transform_pid_np(X)).max()
    print(layer, "max error on train (prompt-id map):", err, flush=True)
    np.save(mappings_prompt_id_dir / f"mapping_{layer}.npy", A_pid)

shutil.make_archive(str(mappings_prompt_id_dir), "zip", root_dir=mappings_prompt_id_dir)

# %% Load pile L→last affine maps into memory
mappings: list[np.ndarray] = []
for layer in tqdm(range(mt.num_layers), desc="load pile maps"):
    path = mappings_dir / f"mapping_{layer}-{mt.num_layers - 1}.npy"
    mappings.append(np.load(path))

# %% Evaluate — affine mapping on validation (notebook cell 12)
"""Lstsq L→last-layer affine on h_L before patching (notebook “affine mapping”; paper Fig. 2 orange ≈ Tuned-Lens-style learned linear bridge in hidden space, not the Belrose logits lens)."""
device = mt.model.device
target_layer_eval = mt.num_layers - 1

records_affine: list[dict] = []
for layer in tqdm(range(mt.num_layers), desc="eval affine"):
    stonesoup.check_abort()
    A = mappings[layer]
    transform = torch_row_affine(A, device)

    for _idx, row in df_same_prompt[df_same_prompt["data_split"] == "validation"].iterrows():
        stonesoup.check_abort()
        prompt = row["full_text"]
        position = int(row["position"])
        prec_1, surprisal = evaluate_patch_next_token_prediction(
            mt,
            prompt,
            prompt,
            layer,
            target_layer_eval,
            position,
            position,
            position_prediction=position,
            transform=transform,
        )
        records_affine.append({"layer": layer, "prec_1": prec_1, "surprisal": float(surprisal)})

results_affine = pd.DataFrame.from_records(records_affine)
csv_affine = OUT_DIR / f"{SAFE_STEM}_mappings_pile_eval.csv"
results_affine.to_csv(csv_affine, index=False)
print("wrote", csv_affine.relative_to(stonesoup.repo_root()), flush=True)

# %% Evaluate — identity / token identity (notebook cell 13)
"""Patch raw h_L to the last layer with no learned map and the same prompt for source/target (notebook “identity”; not the paper’s separate Logit-Lens lm_head baseline, and not green “Token Identity (Ours),” which is the prompt-id cell)."""
records_identity: list[dict] = []
for layer in tqdm(range(mt.num_layers), desc="eval identity"):
    stonesoup.check_abort()
    for _idx, row in df_same_prompt[df_same_prompt["data_split"] == "validation"].iterrows():
        stonesoup.check_abort()
        prompt = row["full_text"]
        position = int(row["position"])
        prec_1, surprisal = evaluate_patch_next_token_prediction(
            mt,
            prompt,
            prompt,
            layer,
            target_layer_eval,
            position,
            position,
            position_prediction=position,
        )
        records_identity.append({"layer": layer, "prec_1": prec_1, "surprisal": float(surprisal)})

results_identity = pd.DataFrame.from_records(records_identity)
csv_identity = OUT_DIR / f"{SAFE_STEM}_identity_pile_eval.csv"
results_identity.to_csv(csv_identity, index=False)
print("wrote", csv_identity.relative_to(stonesoup.repo_root()), flush=True)

# %% Evaluate — prompt-id target prompt (notebook cell 14)
"""Patch source hidden at layer L into the fixed ICL-style target prompt at the “?” slot (T≠S); optional pile affine on h_L—notebook “prompt id”; closest to paper Fig. 2 green “Token Identity (Ours).”"""
position_target = -1
records_pid: list[dict] = []
for layer in tqdm(range(mt.num_layers), desc="eval prompt-id"):
    stonesoup.check_abort()
    if APPLY_PROMPT_ID_AFFINE:
        transform_pid = torch_row_affine(mappings[layer], device)
    else:
        transform_pid = None

    for _idx, row in df_prompt_id[df_prompt_id["data_split"] == "validation"].iterrows():
        stonesoup.check_abort()
        if "prompt" in row and row["prompt"] == "target":
            continue
        prompt_source = row["full_text"]
        position_source = int(row["position"])
        if DEBUG_PRINT_PROMPT_ID_EVAL:
            print(
                "prompt-id eval:",
                f"layer={layer}",
                f"position_source={position_source}",
                f"position_target={position_target}",
                f"PROMPT_TARGET={PROMPT_TARGET!r}",
                f"prompt_source[:200]={prompt_source[:200]!r}",
                flush=True,
            )
        prec_1, surprisal = evaluate_patch_next_token_prediction(
            mt,
            prompt_source,
            PROMPT_TARGET,
            layer,
            layer,
            position_source,
            position_target,
            position_prediction=position_target,
            transform=transform_pid,
        )
        records_pid.append({"layer": layer, "prec_1": prec_1, "surprisal": float(surprisal)})

results_pid = pd.DataFrame.from_records(records_pid)
if APPLY_PROMPT_ID_AFFINE:
    csv_pid = OUT_DIR / f"{SAFE_STEM}_prompt-id-mapping_pile_eval.csv"
else:
    csv_pid = OUT_DIR / f"{SAFE_STEM}_prompt-id_pile_eval.csv"
results_pid.to_csv(csv_pid, index=False)
print("wrote", csv_pid.relative_to(stonesoup.repo_root()), flush=True)

# %% Plot precision@1 and surprisal vs layer (notebook cell 15)
results_identity_plot = pd.read_csv(csv_identity)
results_identity_plot["variant"] = "identity"
results_affine_plot = pd.read_csv(csv_affine)
results_affine_plot["variant"] = "affine mapping"
results_pid_plot = pd.read_csv(csv_pid)
results_pid_plot["variant"] = "prompt id"

results_plot = pd.concat(
    [results_identity_plot, results_affine_plot, results_pid_plot],
    ignore_index=True,
)

for metric in ["prec_1", "surprisal"]:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(data=results_plot, x="layer", y=metric, hue="variant", ax=ax)
    ax.set_title(MODEL_NAME.strip("./"))
    leg = ax.get_legend()
    if leg is not None:
        leg.set_title("")
    plt.tight_layout()
    show(fig, basename=f"{SAFE_STEM}_patchscope_4_1_{metric}", dpi=120)
    plt.close(fig)

# %% df_prompt_id

df = df_prompt_id[(df_prompt_id["data_split"] == "validation") & (df_prompt_id["prompt"] == "source")]
stonesoup.display(df.head(10))
for i, (_, row) in enumerate(df.head(10).iterrows()):
    pos = int(row["position"])
    inp = make_inputs(mt.tokenizer, [row["full_text"]], device=mt.model.device)
    ids = inp["input_ids"][0]
    seq_len = int(ids.shape[0])
    tid = int(ids[pos].item())
    tok_piece = mt.tokenizer.convert_ids_to_tokens([tid])[0]
    hreps = np.array(row["hidden_rep"])
    print(
        "<code>",
        f"row={i}",
        f"position={pos} seq_len={seq_len}",
        f"id={tid}",
        f"piece={tok_piece!r}",
        f"hidden_rep={hreps.shape}",
        "</code><br>",
    )