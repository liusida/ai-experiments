# %% Imports & load TruthfulQA

from pathlib import Path
from typing import Any

from datasets import load_dataset


def _split_semicolon_answers(text: str | None) -> list[str]:
    if not text:
        return []
    return [p.strip() for p in text.split(";") if p.strip()]


def _unique_preserve(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def truthfulqa_row_to_strings(row: dict) -> tuple[list[str], list[str]]:
    """One TruthfulQA row → paired full strings: correct vs incorrect.

    Each string is ``f"{Question} {answer}"`` (question usually ends with ``?``).
    Correct side uses **Best Answer** plus every entry in **Correct Answers**
    (semicolon-separated on disk), deduped in order.
    """
    q = (row.get("Question") or "").strip()
    correct_parts: list[str] = []
    best = (row.get("Best Answer") or "").strip()
    if best:
        correct_parts.append(best)
    correct_parts.extend(_split_semicolon_answers(row.get("Correct Answers")))
    correct_answers = _unique_preserve(correct_parts)

    incorrect_answers = _split_semicolon_answers(row.get("Incorrect Answers"))

    correct_strings = [f"{q} {a}" for a in correct_answers]
    incorrect_strings = [f"{q} {a}" for a in incorrect_answers]
    return correct_strings, incorrect_strings


def _question_prefix_num_tokens(tokenizer: Any, full_text: str, question: str) -> int:
    """How many leading tokens cover only the question (before answer).

    Full strings are ``f"{question} {answer}"``. Answer-only hidden states use
    ``hidden_states[..., num_tokens:, :]``.

    Uses ``return_offsets_mapping`` so the split matches the tokenizer used on
    ``full_text`` (handles merged tokens like ``' Fortune'`` across the gap).
    """
    q = question.strip()
    nchars = len(q)
    enc = tokenizer(full_text, add_special_tokens=True, return_offsets_mapping=True)
    spans = enc["offset_mapping"]
    first = spans[0]
    if (
        isinstance(first, (list, tuple))
        and len(first) == 2
        and all(isinstance(x, (int, type(None))) for x in first)
    ):
        row = spans
    else:
        row = first
    for i, (start, _end) in enumerate(row):
        if start is not None and start >= nchars:
            return i
    return len(row)


def _offset_mapping_token_row(enc: Any) -> list[Any]:
    spans = enc["offset_mapping"]
    first = spans[0]
    if (
        isinstance(first, (list, tuple))
        and len(first) == 2
        and all(isinstance(x, (int, type(None))) for x in first)
    ):
        row = spans
    else:
        row = first
    return list(row)


def _truthfulqa_user_turn_messages(full_text: str) -> list[dict[str, Any]]:
    """User messages for ``apply_chat_template``.

    Qwen3.5 VL **processors** expect ``content`` as typed parts; a plain string is iterated as
    characters and raises ``TypeError`` on ``content[\"type\"]`` (see HF ``processing_utils``).
    """
    return [{"role": "user", "content": [{"type": "text", "text": full_text}]}]


def truthfulqa_chat_user_encoding(processor: Any, full_text: str) -> dict[str, Any]:
    """Inputs for ``model(**batch)``: one user turn with full ``Question + " " + answer`` text.

    When ``processor.apply_chat_template`` exists (Qwen3.5 etc.), uses
    ``add_generation_prompt=False`` and ``enable_thinking=False``. Otherwise plain tokenizer
    encode of ``full_text`` (same as legacy behavior).
    """
    apply_ct = getattr(processor, "apply_chat_template", None)
    if apply_ct is None:
        tok = getattr(processor, "tokenizer", processor)
        return dict(tok(full_text, return_tensors="pt", add_special_tokens=True))
    msgs = _truthfulqa_user_turn_messages(full_text)
    try:
        batch = apply_ct(
            msgs,
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=False,
            return_dict=True,
            return_tensors="pt",
        )
    except TypeError:
        batch = apply_ct(
            msgs,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
        )
    return dict(batch)


def truthfulqa_answer_token_span(processor: Any, full_text: str, question: str) -> tuple[int, int]:
    """Half-open indices ``[start, end)`` into the model sequence for **answer text only**.

    * **Full sequence** stats elsewhere mean **all** tokens (including chat specials).
    * **Answer** stats mean only tokens whose ``offset_mapping`` overlaps the answer
      characters inside the user message—so **no** trailing ``<|im_end|>`` / template tail.

    Plain-tokenizer path: ``[first_answer_token, len(seq))`` (no template tail).
    """
    apply_ct = getattr(processor, "apply_chat_template", None)
    q = question.strip()
    ft = full_text.strip()
    if apply_ct is None:
        tok = getattr(processor, "tokenizer", processor)
        enc = tok(full_text, add_special_tokens=True, return_offsets_mapping=True)
        row = _offset_mapping_token_row(enc)
        a0 = _question_prefix_num_tokens(tok, full_text, q)
        return a0, len(row)

    msgs = _truthfulqa_user_turn_messages(full_text)
    try:
        rendered = apply_ct(
            msgs,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
    except TypeError:
        rendered = apply_ct(
            msgs,
            tokenize=False,
            add_generation_prompt=False,
        )
    ix = rendered.find(full_text)
    user_len = len(full_text)
    if ix < 0:
        ix = rendered.find(ft)
        user_len = len(ft) if ix >= 0 else 0
    if ix < 0:
        tok = getattr(processor, "tokenizer", processor)
        enc = tok(full_text, add_special_tokens=True, return_offsets_mapping=True)
        row = _offset_mapping_token_row(enc)
        a0 = _question_prefix_num_tokens(tok, full_text, q)
        return a0, len(row)

    answer_start_char = ix + len(q) + 1
    if answer_start_char > ix + user_len:
        answer_start_char = ix + user_len
    content_end_char = ix + user_len

    tok = getattr(processor, "tokenizer", processor)
    enc = tok(rendered, add_special_tokens=True, return_offsets_mapping=True)
    row = _offset_mapping_token_row(enc)
    inside: list[int] = []
    for i, (s, e) in enumerate(row):
        if s is None or e is None:
            continue
        if s < content_end_char and e > answer_start_char:
            inside.append(i)
    if not inside:
        a0 = 0
        for i, (start, _end) in enumerate(row):
            if start is not None and start >= answer_start_char:
                a0 = i
                break
        else:
            a0 = len(row)
        return a0, len(row)
    return min(inside), max(inside) + 1


def truthfulqa_first_answer_token_index(processor: Any, full_text: str, question: str) -> int:
    """First token of the answer span; same ``start`` as :func:`truthfulqa_answer_token_span`."""
    a0, _ = truthfulqa_answer_token_span(processor, full_text, question)
    return a0


def _truthfulqa_batch_tensors_to_device(batch: dict[str, Any], device: Any) -> dict[str, Any]:
    import torch

    return {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}


def truthfulqa_flatten_strings(
    split, processor: Any
) -> tuple[list[str], list[str], list[int], list[int], list[int], list[int]]:
    """Flatten all rows; parallel lists give answer spans ``[start, end)`` in tokens (chat-aware)."""
    all_correct: list[str] = []
    all_incorrect: list[str] = []
    correct_ans_start: list[int] = []
    correct_ans_end: list[int] = []
    incorrect_ans_start: list[int] = []
    incorrect_ans_end: list[int] = []
    for row in split:
        q = (row.get("Question") or "").strip()
        c, inc = truthfulqa_row_to_strings(row)
        for s in c:
            all_correct.append(s)
            a0, a1 = truthfulqa_answer_token_span(processor, s, q)
            correct_ans_start.append(a0)
            correct_ans_end.append(a1)
        for s in inc:
            all_incorrect.append(s)
            a0, a1 = truthfulqa_answer_token_span(processor, s, q)
            incorrect_ans_start.append(a0)
            incorrect_ans_end.append(a1)
    return (
        all_correct,
        all_incorrect,
        correct_ans_start,
        correct_ans_end,
        incorrect_ans_start,
        incorrect_ans_end,
    )


# Dim 2 of ``truthfulqa_answer_hidden_stats_per_layer``: ``[N, L+1, 5, H]``.
HIDDEN_STAT_FULL_SEQ_MEAN = 0
HIDDEN_STAT_FULL_SEQ_NORM_MEAN = 1
HIDDEN_STAT_ANSWER_MEAN = 2
HIDDEN_STAT_ANSWER_NORM_MEAN = 3
HIDDEN_STAT_LAST_TOKEN = 4


def truthfulqa_u_prototype_per_layer(
    correct_answer_hiddens: Any,
    incorrect_answer_hiddens: Any,
    *,
    answer_stat_idx: int = HIDDEN_STAT_ANSWER_MEAN,
) -> Any:
    """Per depth index ``ℓ``, unit vector along (grand-mean correct − grand-mean incorrect).

    Uses the same statistic slice as ``answer_stat_idx`` (default: answer-span mean).
    Returns float32 tensor ``[L+1, H]`` matching ``hidden_states`` indices (0 = embeddings).
    """
    import torch

    # ``N`` (num strings) may differ between correct vs incorrect; means are taken separately.
    if (
        correct_answer_hiddens.shape[1] != incorrect_answer_hiddens.shape[1]
        or correct_answer_hiddens.shape[3] != incorrect_answer_hiddens.shape[3]
    ):
        raise ValueError(
            "correct/incorrect caches must share num_layers (dim 1) and H (dim 3); "
            f"got {tuple(correct_answer_hiddens.shape)} vs {tuple(incorrect_answer_hiddens.shape)}"
        )
    n_layers = int(correct_answer_hiddens.shape[1])
    rows: list[Any] = []
    for li in range(n_layers):
        u_c = correct_answer_hiddens[:, li, answer_stat_idx, :].float().mean(dim=0)
        u_i = incorrect_answer_hiddens[:, li, answer_stat_idx, :].float().mean(dim=0)
        u_cn = u_c / u_c.norm().clamp(min=1e-12)
        u_in = u_i / u_i.norm().clamp(min=1e-12)
        d = u_cn - u_in
        rows.append(d / d.norm().clamp(min=1e-12))
    return torch.stack(rows, dim=0)


def truthfulqa_answer_hidden_stats_per_layer(
    *,
    model: Any,
    processor: Any,
    texts: list[str],
    answer_token_start: list[int],
    answer_token_end: list[int],
    device: Any,
    desc: str = "answer hidden states",
) -> Any:
    """Per layer (+ embeddings), five float32 CPU summaries ``[N, L+1, 5, H]``.

    Each string is encoded like :func:`truthfulqa_chat_user_encoding` (chat template +
    ``enable_thinking=False`` when available).

    * :data:`HIDDEN_STAT_FULL_SEQ_MEAN` — mean over **all** sequence positions (incl. chat specials)
    * :data:`HIDDEN_STAT_FULL_SEQ_NORM_MEAN` — L2-normalize each position, then mean (full seq)
    * :data:`HIDDEN_STAT_ANSWER_MEAN` — mean over **answer** tokens only (``[answer_start, answer_end)``)
    * :data:`HIDDEN_STAT_ANSWER_NORM_MEAN` — L2-normalize each answer position, then mean
    * :data:`HIDDEN_STAT_LAST_TOKEN` — hidden at the **last** sequence index (full seq, incl. template tail)

    ``stonesoup.check_abort()`` runs **before** each forward; a running GPU op
    cannot cancel until it returns to Python.
    """
    import torch
    from tqdm.auto import tqdm

    import stonesoup

    def _mean_of_l2_normalized(x: Any) -> Any:
        xf = x.float()
        return (xf / xf.norm(dim=-1, keepdim=True).clamp(min=1e-12)).mean(dim=1).squeeze(0).cpu()

    if len(texts) != len(answer_token_start) or len(texts) != len(answer_token_end):
        raise ValueError("texts and answer span lists length mismatch")
    rows: list[Any] = []
    for text, qi_raw, qe_raw in tqdm(
        zip(texts, answer_token_start, answer_token_end),
        total=len(texts),
        desc=desc,
    ):
        stonesoup.check_abort()
        batch = _truthfulqa_batch_tensors_to_device(
            truthfulqa_chat_user_encoding(processor, text),
            device,
        )
        with torch.inference_mode():
            out = model(**batch, output_hidden_states=True)
        per_layer: list[Any] = []
        for h in out.hidden_states:
            tlen = h.shape[1]
            qi = int(min(max(qi_raw, 0), tlen))
            qe = int(min(max(qe_raw, 0), tlen))
            if qe <= qi:
                qe = min(qi + 1, tlen)
            hf = h.float()
            full_mean = hf.mean(dim=1).squeeze(0).cpu()
            full_norm_mean = _mean_of_l2_normalized(h)
            ans = h[:, qi:qe, :]
            if ans.shape[1] == 0:
                ans = h[:, -1:, :]
            answer_mean = ans.float().mean(dim=1).squeeze(0).cpu()
            answer_norm_mean = _mean_of_l2_normalized(ans)
            last_tok = hf[:, -1, :].squeeze(0).cpu()
            per_layer.append(
                torch.stack(
                    [full_mean, full_norm_mean, answer_mean, answer_norm_mean, last_tok],
                    dim=0,
                )
            )
        rows.append(torch.stack(per_layer, dim=0))
    return torch.stack(rows, dim=0)


TRUTHFULQA_DATASET_ID = "domenicrosati/TruthfulQA"
# Bumped when encoding or answer-span rules change (chat + answer slice excludes template tail).
TRUTHFULQA_HIDDEN_CACHE_KIND = "answer_hidden_five_stats_per_layer_chat_span"


def _truthfulqa_answer_hiddens_cache_dir() -> Path:
    import stonesoup

    return Path(stonesoup.outputs_dir()) / "truthfulqa_answer_hidden_mean"


def truthfulqa_try_load_answer_hiddens_cache(model_id: str) -> tuple[Any, Any] | None:
    """Return ``(correct, incorrect)`` CPU tensors if cache matches ``model_id`` and dataset kind."""
    import json

    import torch

    base = _truthfulqa_answer_hiddens_cache_dir()
    meta_path = base / "meta.json"
    c_path = base / "correct.pt"
    i_path = base / "incorrect.pt"
    if not meta_path.is_file() or not c_path.is_file() or not i_path.is_file():
        return None
    meta = json.loads(meta_path.read_text())
    if meta.get("model_id") != model_id:
        return None
    if meta.get("dataset_id") != TRUTHFULQA_DATASET_ID:
        return None
    if meta.get("kind") != TRUTHFULQA_HIDDEN_CACHE_KIND:
        return None
    c = torch.load(c_path, map_location="cpu", weights_only=False)
    i = torch.load(i_path, map_location="cpu", weights_only=False)
    return c, i


def truthfulqa_save_answer_hiddens_cache(
    model_id: str,
    correct_answer_hiddens: Any,
    incorrect_answer_hiddens: Any,
) -> None:
    """Write ``correct.pt``, ``incorrect.pt``, and ``meta.json`` under :func:`stonesoup.outputs_dir`."""
    import json

    import torch

    base = _truthfulqa_answer_hiddens_cache_dir()
    base.mkdir(parents=True, exist_ok=True)
    torch.save(correct_answer_hiddens, base / "correct.pt")
    torch.save(incorrect_answer_hiddens, base / "incorrect.pt")
    meta = {
        "model_id": model_id,
        "dataset_id": TRUTHFULQA_DATASET_ID,
        "kind": TRUTHFULQA_HIDDEN_CACHE_KIND,
        "answer_stat_labels": [
            "full_sequence_mean",
            "full_sequence_normalized_mean",
            "answer_mean",
            "answer_normalized_mean",
            "last_token",
        ],
        "correct_shape": list(correct_answer_hiddens.shape),
        "incorrect_shape": list(incorrect_answer_hiddens.shape),
    }
    (base / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("cached answer hidden tensors →", base)


# %% Load TruthfulQA dataset
dataset = load_dataset(TRUTHFULQA_DATASET_ID)
# %% Load Qwen/Qwen3.5-2B
import stonesoup
MODEL_ID = "Qwen/Qwen3.5-9B"
model, processor = stonesoup.load_model(MODEL_ID)

# %% Example: one row → correct / incorrect strings

example = dataset["train"][3]
correct, incorrect = truthfulqa_row_to_strings(example)
print("Question:", example["Question"])
print("--- correct (", len(correct), ") ---", sep="")
for s in correct[:3]:
    print(repr(s))
if len(correct) > 3:
    print("...")
print("--- incorrect (", len(incorrect), ") ---", sep="")
for s in incorrect[:3]:
    print(repr(s))
if len(incorrect) > 3:
    print("...")

# %% All train rows → flat `all_correct_strings` / `all_incorrect_strings`

train_split = dataset["train"]
(
    all_correct_strings,
    all_incorrect_strings,
    all_correct_answer_tok_start,
    all_correct_answer_tok_end,
    all_incorrect_answer_tok_start,
    all_incorrect_answer_tok_end,
) = truthfulqa_flatten_strings(train_split, processor)
print(len(train_split), "rows")
print("total correct strings:", len(all_correct_strings))
print("total incorrect strings:", len(all_incorrect_strings))
print(
    "sample answer token spans [start,end) (incorrect[:3]):",
    list(
        zip(
            all_incorrect_answer_tok_start[:3],
            all_incorrect_answer_tok_end[:3],
        )
    ),
)

# %% All strings → answer-span hidden states (correct vs incorrect tensors)

# If ``meta.json`` + ``.pt`` files match ``MODEL_ID`` and flattened row counts, skips **all** model forwards.
_loaded = truthfulqa_try_load_answer_hiddens_cache(MODEL_ID)
_use_cache = (
    _loaded is not None
    and _loaded[0].shape[0] == len(all_correct_strings)
    and _loaded[1].shape[0] == len(all_incorrect_strings)
)
if _use_cache:
    correct_answer_hiddens, incorrect_answer_hiddens = _loaded
    print(
        "Loaded cached answer hidden states from",
        _truthfulqa_answer_hiddens_cache_dir(),
    )
else:
    if _loaded is not None:
        print("Cache present but ignored (model/dataset/kind mismatch or row count changed); regenerating.")
    _device = next(model.parameters()).device
    correct_answer_hiddens = truthfulqa_answer_hidden_stats_per_layer(
        model=model,
        processor=processor,
        texts=all_correct_strings,
        answer_token_start=all_correct_answer_tok_start,
        answer_token_end=all_correct_answer_tok_end,
        device=_device,
        desc="correct → hidden (5 summaries)",
    )
    incorrect_answer_hiddens = truthfulqa_answer_hidden_stats_per_layer(
        model=model,
        processor=processor,
        texts=all_incorrect_strings,
        answer_token_start=all_incorrect_answer_tok_start,
        answer_token_end=all_incorrect_answer_tok_end,
        device=_device,
        desc="incorrect → hidden (5 summaries)",
    )

print("correct_answer_hiddens ", tuple(correct_answer_hiddens.shape), correct_answer_hiddens.dtype)
print("incorrect_answer_hiddens", tuple(incorrect_answer_hiddens.shape), incorrect_answer_hiddens.dtype)

# %% Cache the hidden states to appropriate paths

truthfulqa_save_answer_hiddens_cache(
    MODEL_ID,
    correct_answer_hiddens,
    incorrect_answer_hiddens,
)

# %% Unit prototype directions (answer_mean, one ``u_proto`` per layer)

u_prototype = truthfulqa_u_prototype_per_layer(
    correct_answer_hiddens,
    incorrect_answer_hiddens,
)
print("u_prototype", tuple(u_prototype.shape), "last-layer ‖u‖₂", float(u_prototype[-1].norm()))

# %% TruthfulQA row for demos: same Q as heatmaps; one correct / incorrect string through the model

# Must match the heatmap cell below. ``_ANSWER_INDEX`` picks which variant in each list (0 = first).
_HEAT_ROW_IDX = 8
_ANSWER_INDEX = 1
_heat_row = dataset["train"][_HEAT_ROW_IDX]
_correct_for_q, _incorrect_for_q = truthfulqa_row_to_strings(_heat_row)
_q_str = (_heat_row["Question"] or "").strip()
print(
    f"TruthfulQA demo train[{_HEAT_ROW_IDX}]:",
    len(_correct_for_q),
    "correct,",
    len(_incorrect_for_q),
    "incorrect full strings (Q + A)",
    f"| using variant index {_ANSWER_INDEX}",
)
if 0 <= _ANSWER_INDEX < len(_correct_for_q):
    _s = _correct_for_q[_ANSWER_INDEX]
    print(f"  correct[{_ANSWER_INDEX}] :", repr(_s[:200] + ("…" if len(_s) > 200 else "")))
else:
    print(f"  correct[{_ANSWER_INDEX}] : (out of range; have {len(_correct_for_q)} correct)")
if 0 <= _ANSWER_INDEX < len(_incorrect_for_q):
    _s = _incorrect_for_q[_ANSWER_INDEX]
    print(f"  incorrect[{_ANSWER_INDEX}]:", repr(_s[:200] + ("…" if len(_s) > 200 else "")))
else:
    print(f"  incorrect[{_ANSWER_INDEX}]: (out of range; have {len(_incorrect_for_q)} incorrect)")

# %% cos(h[pos], u_prototype) every layer & position — chosen correct vs incorrect variant for that row

import torch
import torch.nn.functional as F

torch.set_printoptions(linewidth=200, precision=4, sci_mode=False)
_dev = next(model.parameters()).device
_u = u_prototype.to(_dev).float()

for _label, _text in (
    (
        "correct",
        _correct_for_q[_ANSWER_INDEX] if 0 <= _ANSWER_INDEX < len(_correct_for_q) else None,
    ),
    (
        "incorrect",
        _incorrect_for_q[_ANSWER_INDEX] if 0 <= _ANSWER_INDEX < len(_incorrect_for_q) else None,
    ),
):
    if _text is None:
        print(
            f"=== {_label}  (train[{_HEAT_ROW_IDX}] — no string at index {_ANSWER_INDEX}) ===\n",
        )
        continue
    _batch = _truthfulqa_batch_tensors_to_device(truthfulqa_chat_user_encoding(processor, _text), _dev)
    with torch.inference_mode():
        _out = model(**_batch, output_hidden_states=True)
    print(f"=== {_label}  (train[{_HEAT_ROW_IDX}], index {_ANSWER_INDEX}) ===")
    print(_text[:120] + ("…" if len(_text) > 120 else ""))
    for _li, _h in enumerate(_out.hidden_states):
        _cos = F.cosine_similarity(_h.float()[0], _u[_li].unsqueeze(0), dim=-1).cpu()
        print(f"  layer {_li:2d}  cos per pos {_cos}")
    print()

# %% Heatmaps: chosen correct vs incorrect full string — cos(h, u_proto) per layer & position (no padding)

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

import stonesoup

# `_HEAT_ROW_IDX`, `_ANSWER_INDEX`, `_heat_row`, `_correct_for_q`, `_incorrect_for_q`, `_q_str` from demo row above.

_dev_h = next(model.parameters()).device
_u_h = u_prototype.to(_dev_h).float()

_first_c = (
    _correct_for_q[_ANSWER_INDEX]
    if 0 <= _ANSWER_INDEX < len(_correct_for_q)
    else None
)
_first_i = (
    _incorrect_for_q[_ANSWER_INDEX]
    if 0 <= _ANSWER_INDEX < len(_incorrect_for_q)
    else None
)


def _strip_lead_space_marker(tok: str) -> str:
    # GPT/BPE "Ġ" and SentencePiece "▁" mark word-boundary spaces in piece strings
    for _m in ("Ġ", "▁"):
        if tok.startswith(_m):
            return tok[len(_m) :]
    return tok


def _cos_full_seq_vs_u_proto(text: str) -> Any:
    """``[L+1, T]`` float32, cosine similarity at each layer and token position."""
    stonesoup.check_abort()
    _b = _truthfulqa_batch_tensors_to_device(truthfulqa_chat_user_encoding(processor, text), _dev_h)
    with torch.inference_mode():
        _out = model(**_b, output_hidden_states=True)
    _rows = [
        F.cosine_similarity(_h.float()[0], _u_h[_li].unsqueeze(0), dim=-1).cpu()
        for _li, _h in enumerate(_out.hidden_states)
    ]
    return torch.stack(_rows, dim=0).float().numpy()


def _xtick_labels_for_text(text: str) -> list[str]:
    _tok_d = getattr(processor, "tokenizer", processor)
    _ids = truthfulqa_chat_user_encoding(processor, text)["input_ids"][0]
    return [_strip_lead_space_marker(t) for t in _tok_d.convert_ids_to_tokens(_ids.tolist())]


if _first_c is None or _first_i is None:
    print(
        "Heatmaps skipped: need both sides at",
        f"_ANSWER_INDEX={_ANSWER_INDEX} for train[{_HEAT_ROW_IDX}]",
        f"(correct={_first_c is not None}, incorrect={_first_i is not None}).",
    )
else:
    _correct_heat = _cos_full_seq_vs_u_proto(_first_c)
    _incorrect_heat = _cos_full_seq_vs_u_proto(_first_i)
    _labels_c = _xtick_labels_for_text(_first_c)
    _labels_i = _xtick_labels_for_text(_first_i)
    _tmax_c, _tmax_i = _correct_heat.shape[1], _incorrect_heat.shape[1]

    _figh, _axh = plt.subplots(1, 2, figsize=(17, 7.5), sharey=True, constrained_layout=True)
    COLOR_SCALE = 0.5
    for _ax, _sent, _dat, _title, _ticks, _tmax in (
        (
            _axh[0],
            _first_c,
            _correct_heat,
            f"cos(h, u_proto) — correct variant [{_ANSWER_INDEX}] (full seq)",
            _labels_c,
            _tmax_c,
        ),
        (
            _axh[1],
            _first_i,
            _incorrect_heat,
            f"cos(h, u_proto) — incorrect variant [{_ANSWER_INDEX}] (full seq)",
            _labels_i,
            _tmax_i,
        ),
    ):
        _imh = _ax.imshow(_dat, aspect="auto", origin="lower", cmap="coolwarm", vmin=-COLOR_SCALE, vmax=COLOR_SCALE)
        _qn_tok = truthfulqa_first_answer_token_index(processor, _sent, _q_str)
        # First answer token is at index ``_qn_tok``; boundary sits between Q and A columns.
        if 0 < _qn_tok <= _tmax:
            _ax.axvline(
                x=_qn_tok - 0.5,
                color="black",
                linewidth=1,
                linestyle="--",
                alpha=0.5,
                zorder=10,
            )
        _ax.set_xticks(range(_tmax))
        _ax.set_xticklabels(_ticks, rotation=55, ha="right", fontsize=8)
        _ax.set_xlabel("tokens (question then answer)")
        _ax.set_ylabel("layer (0 = embeddings)")
        _ax.set_title(_title)
    _figh.colorbar(
        _imh,
        ax=list(_axh.flat),
        shrink=0.88,
        location="right",
        pad=0.02,
        label="cosine similarity",
    )
    _figh.suptitle(
        f'u_proto per layer (answer_mean) | Q: {_q_str[:80]}{"…" if len(_q_str) > 80 else ""}',
        fontsize=10,
    )
    stonesoup.show(_figh)

# %% Custom heatmaps: one question + two hand-written answers + model-generated answer

# Needs globals from the prior heatmap cell: ``model``, ``processor``, ``_dev_h``, ``_u_h``,
# ``_cos_full_seq_vs_u_proto``, ``_xtick_labels_for_text``.

# _CUSTOM_QUESTION = "What color is the sun?"
# _CUSTOM_ANSWER_LEFT = "White."
# _CUSTOM_ANSWER_RIGHT = "It is actually blue-ish green."

_CUSTOM_QUESTION = "What is 1+1?"
_CUSTOM_ANSWER_LEFT = "2."
_CUSTOM_ANSWER_RIGHT = "I should be humble, but I don't know."

_CUSTOM_TITLE_LEFT = "hand — left"
_CUSTOM_TITLE_RIGHT = "hand — right"
_CUSTOM_TITLE_MODEL = "model-generated"

# Qwen3.5: use ``apply_chat_template(..., enable_thinking=False)`` so generation is non-thinking
# (see https://huggingface.co/Qwen/Qwen3.5-0.8B — thinking is opt-in via API / template).
_CUSTOM_GEN_MAX_NEW_TOKENS = 14
_CUSTOM_GEN_DO_SAMPLE = False

_cq = _CUSTOM_QUESTION.strip()
_CUSTOM_TEXT_LEFT = f"{_cq} {_CUSTOM_ANSWER_LEFT.strip()}"
_CUSTOM_TEXT_RIGHT = f"{_cq} {_CUSTOM_ANSWER_RIGHT.strip()}"

import sys

import matplotlib.pyplot as plt
import torch

import stonesoup

_tok_h = getattr(processor, "tokenizer", processor)
_proc = processor
_messages = _truthfulqa_user_turn_messages(_cq)
stonesoup.check_abort()
_apply_ct = getattr(_proc, "apply_chat_template", None)
if _apply_ct is not None:
    _ct_base = dict(
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    try:
        _gen_batch = _apply_ct(_messages, enable_thinking=False, **_ct_base)
    except TypeError:
        _gen_batch = _apply_ct(_messages, **_ct_base)
    if hasattr(_gen_batch, "to"):
        _gen_batch = _gen_batch.to(_dev_h)
    else:
        _gen_batch = {
            k: v.to(_dev_h) if isinstance(v, torch.Tensor) else v
            for k, v in _gen_batch.items()
        }
else:
    _gen_prefix = f"{_cq} "
    _gen_batch = _tok_h(_gen_prefix, return_tensors="pt", add_special_tokens=True)
    _gen_batch = {k: v.to(_dev_h) for k, v in _gen_batch.items()}
_gen_in_len = int(_gen_batch["input_ids"].shape[1])
_gen_pad = _tok_h.pad_token_id
if _gen_pad is None:
    _gen_pad = _tok_h.eos_token_id
with torch.inference_mode():
    _gen_ids = model.generate(
        **_gen_batch,
        max_new_tokens=int(_CUSTOM_GEN_MAX_NEW_TOKENS),
        do_sample=bool(_CUSTOM_GEN_DO_SAMPLE),
        pad_token_id=_gen_pad,
    )
_gen_answer_only = _tok_h.decode(_gen_ids[0, _gen_in_len:], skip_special_tokens=True).strip()
_CUSTOM_TEXT_MODEL = f"{_cq} {_gen_answer_only}" if _gen_answer_only else _cq
print(
    "model completion (answer span):",
    repr(_gen_answer_only[:400] + ("…" if len(_gen_answer_only) > 400 else "")),
    file=sys.stderr,
    flush=True,
)

_custom_figh, _custom_axh = plt.subplots(1, 3, figsize=(24, 7.5), sharey=True, constrained_layout=True)
_COLOR_CUSTOM = 0.5
for _ax, _sent, _title in (
    (_custom_axh[0], _CUSTOM_TEXT_LEFT, _CUSTOM_TITLE_LEFT),
    (_custom_axh[1], _CUSTOM_TEXT_RIGHT, _CUSTOM_TITLE_RIGHT),
    (_custom_axh[2], _CUSTOM_TEXT_MODEL, _CUSTOM_TITLE_MODEL),
):
    stonesoup.check_abort()
    _cdat = _cos_full_seq_vs_u_proto(_sent)
    _cticks = _xtick_labels_for_text(_sent)
    _ctmax = _cdat.shape[1]
    _cim = _ax.imshow(
        _cdat,
        aspect="auto",
        origin="lower",
        cmap="coolwarm",
        vmin=-_COLOR_CUSTOM,
        vmax=_COLOR_CUSTOM,
    )
    if _cq:
        _cqn = truthfulqa_first_answer_token_index(processor, _sent, _cq)
        if 0 < _cqn <= _ctmax:
            _ax.axvline(
                x=_cqn - 0.5,
                color="black",
                linewidth=1,
                linestyle="--",
                alpha=0.5,
                zorder=10,
            )
    _ax.set_xticks(range(_ctmax))
    _ax.set_xticklabels(_cticks, rotation=55, ha="right", fontsize=12)
    _ax.set_xlabel("")
    _ax.set_ylabel("layer (0 = embeddings)")
    _ax.set_title(_title)
_custom_figh.colorbar(
    _cim,
    ax=list(_custom_axh.flat),
    shrink=0.88,
    location="right",
    pad=0.02,
    label="cosine similarity",
)
_custom_figh.suptitle(
    f'u_proto per layer (answer_mean) | Q: {_cq[:80]}{"…" if len(_cq) > 80 else ""}',
    fontsize=10,
)
stonesoup.show(_custom_figh)

# %%
