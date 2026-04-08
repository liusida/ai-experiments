# %% Imports & config
from __future__ import annotations

import json
from typing import Any

import torch
import stonesoup
from datasets import load_dataset

QWEN_ID = "Qwen/Qwen3.5-0.8B"
GPT2_ID = "openai-community/gpt2-medium"

DATASET_NAME = "truthful_qa"
DATASET_CONFIG = "generation"
DATASET_SPLIT = "validation"
MAX_SAMPLES = 817
NUM_ANSWER_TOKENS = 10
# Need enough Qwen subwords that GPT-2 still sees >=1 answer BPE token; 10 Qwen tokens can be <10 GPT-2 tokens.
QWEN_MAX_NEW_TOKENS = 20

SEED = 42

# Same per-script tree as stonesoup.show(): outputs/…/<script_stem>/ (leading experiments/ stripped; EXPERIMENT_PYTHON.md).
OUT_DIR = stonesoup.outputs_dir() / "prefill_vs_generated_cache"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_PATH = OUT_DIR / "activations.pt"
MANIFEST_PATH = OUT_DIR / "manifest.json"

# %% Load TruthfulQA rows (question + reference answers)
_ds = load_dataset(DATASET_NAME, DATASET_CONFIG, split=DATASET_SPLIT)

ROWS: list[dict[str, Any]] = []
for row in _ds:
    q = (row.get("question") or row.get("Question") or "").strip()
    if q:
        ROWS.append(dict(row))
    if len(ROWS) >= int(MAX_SAMPLES):
        break

print(
    f"Loaded {len(ROWS)} rows from {DATASET_NAME}/{DATASET_CONFIG}[{DATASET_SPLIT}]",
    flush=True,
)

# %% Helpers: TruthfulQA strings + Qwen chat + GPT-2 hooks / spans


def _split_semicolon_answers(text: str | None) -> list[str]:
    if not text:
        return []
    return [p.strip() for p in str(text).split(";") if p.strip()]


def _truthfulqa_answer_field_to_strings(raw: Any) -> list[str]:
    """HF ``truthful_qa`` may store answers as a ``list[str]`` or as ``";"``-separated text."""
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        return [str(x).strip() for x in raw if str(x).strip()]
    return _split_semicolon_answers(str(raw).strip() or None)


def _unique_preserve(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def truthfulqa_correct_answer_strings(row: dict[str, Any]) -> list[str]:
    """Best + ``Correct Answers`` entries, deduped (same pattern as ``dataset-truthqa``)."""
    parts: list[str] = []
    best = (row.get("best_answer") or row.get("Best Answer") or "").strip()
    if best:
        parts.append(best)
    correct_raw = row.get("correct_answers") or row.get("Correct Answers")
    parts.extend(_truthfulqa_answer_field_to_strings(correct_raw))
    return _unique_preserve(parts)


def truthfulqa_first_incorrect(row: dict[str, Any]) -> str | None:
    raw = row.get("incorrect_answers") or row.get("Incorrect Answers")
    inc = _truthfulqa_answer_field_to_strings(raw)
    return inc[0] if inc else None


def gpt2_first_token_after_prefix_decode(
    tokenizer: Any,
    ids_1d: torch.Tensor,
    prefix: str,
    *,
    full: str,
) -> int:
    """Smallest ``k`` such that ``decode(ids[:k]).startswith(prefix)``."""
    inner = getattr(tokenizer, "tokenizer", tokenizer)
    ids = ids_1d.detach().cpu().tolist()
    for k in range(1, len(ids) + 1):
        t = inner.decode(
            ids[:k],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if len(t) >= len(prefix) and t.startswith(prefix):
            return k
    dec_all = inner.decode(
        ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    assert False, (
        "GPT-2 incremental decode never reached prefix; "
        f"prefix[:200]={prefix[:200]!r} full[:200]={full[:200]!r} decode(full)[:240]={dec_all[:240]!r}"
    )


def _prefill_answer_start_index(
    gpt2_tok: Any,
    full: str,
    question: str,
    enc: Any,
    ids_1d: torch.Tensor,
) -> int:
    """First answer-token index: offset_mapping on ``enc``, else ``char_to_token``, else decode scan."""
    prefix = gpt2_question_prefix(question)
    assert full.startswith(prefix), (
        "prefill full must start with question + trailing newline (``gpt2_question_prefix``); "
        f"prefix[:120]={prefix[:120]!r} full[:120]={full[:120]!r}"
    )
    a_char = len(prefix)
    row = _gpt2_offset_mapping_row(enc)
    inside: list[int] = []
    for i, (s, e) in enumerate(row):
        if s is None or e is None:
            continue
        if s < len(full) and e > a_char:
            inside.append(i)
    if inside:
        return min(inside)
    char_tok = getattr(enc, "char_to_token", None)
    if callable(char_tok):
        for pos in (a_char, a_char - 1 if a_char > 0 else 0):
            try:
                tix = char_tok(0, pos)
            except TypeError:
                try:
                    tix = char_tok(pos)
                except Exception:
                    tix = None
            except Exception:
                tix = None
            if tix is not None:
                return int(tix)
    return gpt2_first_token_after_prefix_decode(gpt2_tok, ids_1d, prefix, full=full)


def gpt2_answer_first_token_index(tokenizer: Any, full: str, question: str) -> int:
    """First token overlapping the answer (offsets + decode fallback on same ids)."""
    try:
        enc = tokenizer(
            full,
            return_tensors="pt",
            add_special_tokens=True,
            return_attention_mask=True,
            return_offsets_mapping=True,
        )
    except TypeError:
        enc = tokenizer(
            full,
            return_tensors="pt",
            add_special_tokens=True,
            return_attention_mask=True,
        )
    return _prefill_answer_start_index(
        tokenizer, full, question, enc, enc["input_ids"][0]
    )


def prefill_answer_text_or_skip(
    gpt2_model: Any,
    gpt2_tok: Any,
    gpt2_device: torch.device,
    question: str,
    answer: str,
    num_stages_ref: list[int],
) -> tuple[torch.Tensor, int]:
    """Prefill forward; ``(num_stages, n_tok, H)`` with ``n_tok <= NUM_ANSWER_TOKENS`` (no padding)."""
    prefix = gpt2_question_prefix(question)
    body = answer.strip()
    assert body, "prefill_answer_text_or_skip: empty answer text"
    full = prefix + body
    try:
        enc = gpt2_tok(
            full,
            return_tensors="pt",
            add_special_tokens=True,
            return_attention_mask=True,
            return_offsets_mapping=True,
        )
    except TypeError:
        enc = gpt2_tok(
            full,
            return_tensors="pt",
            add_special_tokens=True,
            return_attention_mask=True,
        )
    ids = enc["input_ids"].to(gpt2_device)
    mask = enc["attention_mask"].to(gpt2_device)
    a0 = _prefill_answer_start_index(gpt2_tok, full, question, enc, ids[0])
    stacks, nst = gpt2_capture_hidden_stages(gpt2_model, ids, mask)
    if num_stages_ref:
        assert num_stages_ref[0] == nst
    else:
        num_stages_ref.append(nst)
    return slice_prefill_answer_stages(stacks, a0, NUM_ANSWER_TOKENS)


def slice_prefill_answer_stages(
    stacks: torch.Tensor, start: int, max_answer_tokens: int
) -> tuple[torch.Tensor, int]:
    """``(num_stages, n_tok, hidden)`` with ``n_tok = min(available, max_answer_tokens)`` — no padding."""
    seq = int(stacks.shape[1])
    assert (
        0 <= start < seq
    ), f"slice_prefill_answer_stages: answer start token {start} out of [0,{seq}) (seq_len={seq})"
    avail = seq - int(start)
    assert avail > 0, f"slice_prefill_answer_stages: no tokens from start={start} (seq={seq})"
    take = min(avail, int(max_answer_tokens))
    out = stacks[:, start : start + take, :].float()
    return out.contiguous(), int(take)


def activations_to_token_layer_dim(stacks_seq: torch.Tensor) -> torch.Tensor:
    """``(num_stages, seq, H)`` → ``(seq, num_stages, H)`` — layout ``[n_token, n_layer, n_dim]``."""
    if stacks_seq.ndim != 3:
        raise ValueError(f"expected (num_stages, seq, H), got {tuple(stacks_seq.shape)}")
    return stacks_seq.permute(1, 0, 2).contiguous()


def qwen_user_messages(question: str) -> list[dict[str, Any]]:
    return [{"role": "user", "content": [{"type": "text", "text": question.strip()}]}]


def encode_qwen_generation(
    processor: Any, question: str, device: torch.device
) -> dict[str, Any]:
    msgs = qwen_user_messages(question)
    apply_ct = getattr(processor, "apply_chat_template", None)
    if apply_ct is None:
        tok = getattr(processor, "tokenizer", processor)
        batch = tok(f"{question.strip()} ", return_tensors="pt", add_special_tokens=True)
        return {k: v.to(device) for k, v in batch.items()}
    try:
        batch = apply_ct(
            msgs,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=True,
            return_tensors="pt",
        )
    except TypeError:
        batch = apply_ct(
            msgs,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
    if hasattr(batch, "to"):
        return dict(batch.to(device))
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in dict(batch).items()
    }


def generate_qwen_answer(
    model: Any, processor: Any, question: str, device: torch.device
) -> str:
    batch = encode_qwen_generation(processor, question, device)
    tok = getattr(processor, "tokenizer", processor)
    pad_id = tok.pad_token_id
    if pad_id is None:
        pad_id = tok.eos_token_id
    in_len = int(batch["input_ids"].shape[1])
    with torch.inference_mode():
        out = model.generate(
            **batch,
            max_new_tokens=int(QWEN_MAX_NEW_TOKENS),
            do_sample=False,
            pad_token_id=pad_id,
        )
    text = tok.decode(out[0, in_len:], skip_special_tokens=True).strip()
    return text


def gpt2_question_prefix(question: str) -> str:
    """Shared boundary: answer tokens follow ``prefix`` (matches prefill + generated).

    Uses a newline instead of a space so GPT-2 does not often emit a whitespace-only first
    token (e.g. ``\\xa0`` / NBSP in vocab) right after the question.
    """
    return question.strip() + "\n"


def _gpt2_input_id_list(gpt2_tok: Any, text: str) -> list[int]:
    enc = gpt2_tok(text, add_special_tokens=True)
    t = enc["input_ids"]
    if hasattr(t, "tolist"):
        return t[0].tolist() if t.ndim > 1 else t.tolist()
    return list(t)


def _gpt2_token_debug_lines(
    gpt2_tok: Any,
    label: str,
    *,
    full_text: str,
    question: str,
) -> list[str]:
    inner = getattr(gpt2_tok, "tokenizer", gpt2_tok)
    ids = _gpt2_input_id_list(gpt2_tok, full_text)
    pieces = inner.convert_ids_to_tokens(ids)
    a0 = gpt2_answer_first_token_index(gpt2_tok, full_text, question)
    end = min(len(ids), a0 + int(NUM_ANSWER_TOKENS))
    return [
        f"{label}: seq_len={len(ids)} | answer_token_range=[{a0}, {end}) (stored {end - a0} tok)",
        f"  ids:            {ids}",
        f"  pieces:         {pieces}",
        f"  answer_ids:      {ids[a0:end]}",
        f"  answer_pieces:   {pieces[a0:end]}",
    ]


def answer_char_start_in_full(full_text: str, question: str) -> int:
    q = question.strip()
    ft = full_text.strip()
    if not ft.startswith(q):
        return len(q) + 1
    i = len(q)
    if i < len(ft) and ft[i] in ("\n", " "):
        i += 1
    return i


def _gpt2_offset_mapping_row(enc: Any) -> list[tuple[int | None, int | None]]:
    """Normalize HF ``offset_mapping`` to a flat list of ``(start, end)`` (unwrap batch if needed)."""
    spans = enc.get("offset_mapping") if hasattr(enc, "get") else enc["offset_mapping"]
    if spans is None:
        return []
    if hasattr(spans, "tolist"):
        spans = spans.tolist()
    if not spans:
        return []
    first = spans[0]
    # Batch shape ``[[pair,…]]``: inner row is a sequence of pairs
    if (
        isinstance(first, (list, tuple))
        and first
        and isinstance(first[0], (list, tuple))
        and len(first[0]) == 2
    ):
        spans = list(first)
    out: list[tuple[int | None, int | None]] = []
    for p in spans:
        if not isinstance(p, (list, tuple)) or len(p) != 2:
            continue
        a, b = p[0], p[1]
        out.append(
            (int(a) if a is not None else None, int(b) if b is not None else None)
        )
    return out


def gpt2_answer_token_span(
    tokenizer: Any, full_text: str, question: str
) -> tuple[int, int]:
    """Half-open ``[a0, a1)`` indices into GPT-2 token sequence for answer substring."""
    enc = tokenizer(
        full_text,
        add_special_tokens=True,
        return_offsets_mapping=True,
        return_attention_mask=True,
    )
    row = _gpt2_offset_mapping_row(enc)
    prefix = gpt2_question_prefix(question)
    if full_text.startswith(prefix):
        a_char = len(prefix)
    else:
        a_char = answer_char_start_in_full(full_text, question)
    content_end = len(full_text)
    inside: list[int] = []
    for i, (s, e) in enumerate(row):
        if s is None or e is None:
            continue
        if s < content_end and e > a_char:
            inside.append(i)
    if not inside:
        return _gpt2_answer_start_fallback(tokenizer, full_text, question)
    return min(inside), max(inside) + 1


def _gpt2_answer_start_fallback(
    tokenizer: Any, full_text: str, question: str
) -> tuple[int, int]:
    prefix = gpt2_question_prefix(question)
    full_ids = tokenizer(full_text, add_special_tokens=True)["input_ids"]
    pref_ids = tokenizer(prefix, add_special_tokens=True)["input_ids"]
    if len(pref_ids) > 0 and full_ids[: len(pref_ids)] == pref_ids:
        a0 = len(pref_ids)
    else:
        a0 = _question_prefix_num_tokens(tokenizer, full_text, question)
    return a0, len(full_ids)


def _question_prefix_num_tokens(tokenizer: Any, full_text: str, question: str) -> int:
    q = question.strip()
    nchars = len(q)
    enc = tokenizer(full_text, add_special_tokens=True, return_offsets_mapping=True)
    row = _gpt2_offset_mapping_row(enc)
    for i, (start, _end) in enumerate(row):
        if start is not None and start >= nchars:
            return i
    return len(row)


def gpt2_decoder_blocks(model: Any) -> Any:
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise TypeError(f"Expected GPT-2-style model; got {type(model).__name__}")


@torch.inference_mode()
def gpt2_capture_hidden_stages(
    model: Any, input_ids: torch.Tensor, attention_mask: torch.Tensor | None
) -> tuple[torch.Tensor, int]:
    """Return ``(stacks, num_stages)`` with ``stacks`` shape ``(num_stages, seq_len, hidden)``.

    Each hook sees ``(batch, seq_len, hidden)``; we ``stack`` on stage → ``(num_stages, batch, seq, H)``,
    then squeeze batch 1 so token positions match ``input_ids`` 1:1.

    Stage 0 = hidden entering block 0; later = output of each decoder block.
    """
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    blocks = gpt2_decoder_blocks(model)
    captured: list[torch.Tensor] = []

    def pre_block0(_module: Any, inputs: tuple) -> None:
        captured.append(inputs[0].detach())

    def post_block(_module: Any, _inp: Any, out: torch.Tensor | tuple) -> None:
        hidden = out[0] if isinstance(out, tuple) else out
        captured.append(hidden.detach())

    hooks = [blocks[0].register_forward_pre_hook(pre_block0)]
    hooks += [layer.register_forward_hook(post_block) for layer in blocks]
    try:
        model(input_ids, attention_mask=attention_mask, use_cache=False)
    finally:
        for h in hooks:
            h.remove()

    stacks = torch.stack(captured, dim=0).float().cpu()
    if stacks.ndim == 4:
        assert int(stacks.shape[1]) == 1, (
            f"gpt2_capture_hidden_stages: expected batch_size=1, got {stacks.shape[1]}"
        )
        stacks = stacks.squeeze(1)
    return stacks, stacks.shape[0]


def slice_answer_stages(
    stacks: torch.Tensor, start: int, num_tokens: int
) -> torch.Tensor | None:
    """``(num_stages, min(num_tokens, available), hidden)`` or ``None`` if too short."""
    end = start + int(num_tokens)
    if start < 0 or start + 1 > stacks.shape[1]:
        return None
    avail = stacks.shape[1] - start
    if avail < num_tokens:
        return None
    return stacks[:, start:end, :].contiguous()

# %% Load models
torch.manual_seed(int(SEED))

qwen_model, qwen_processor = stonesoup.load_model(QWEN_ID)
qwen_model.eval()
qwen_device = next(qwen_model.parameters()).device

gpt2_model, gpt2_tok = stonesoup.load_model(GPT2_ID)
gpt2_model.eval()
gpt2_device = next(gpt2_model.parameters()).device

gpt2_pad = gpt2_tok.pad_token_id
if gpt2_pad is None:
    gpt2_pad = gpt2_tok.eos_token_id

print("Qwen:", QWEN_ID, qwen_device, flush=True)
print("GPT-2:", GPT2_ID, gpt2_device, flush=True)

# %% Collect activations & save to disk
records: list[dict[str, Any]] = []
prefill_qwen_list: list[torch.Tensor] = []
prefill_correct_list: list[torch.Tensor] = []
prefill_incorrect_list: list[torch.Tensor] = []
generated_list: list[torch.Tensor] = []

run_q = run_cor = run_inc = run_gen = 0

for idx, row in enumerate(ROWS):
    stonesoup.check_abort()
    question = (row.get("question") or row.get("Question") or "").strip()
    correct_opts = truthfulqa_correct_answer_strings(row)
    correct_answer = correct_opts[0] if correct_opts else ""
    incorrect_answer = truthfulqa_first_incorrect(row) or ""

    if not correct_answer:
        print(f"[{idx}] skip: no correct reference answer in row", flush=True)
        continue
    if not incorrect_answer:
        print(f"[{idx}] skip: no incorrect reference answer in row", flush=True)
        continue

    qwen_answer = generate_qwen_answer(
        qwen_model, qwen_processor, question, qwen_device
    )
    if not qwen_answer:
        print(f"[{idx}] skip: empty Qwen answer", flush=True)
        continue

    num_stages_box: list[int] = []
    ans_qwen, n_qwen = prefill_answer_text_or_skip(
        gpt2_model, gpt2_tok, gpt2_device, question, qwen_answer, num_stages_box
    )
    # print(
    #     f"[{idx}] Qwen: {qwen_answer!r} | gpt2_answer_tokens_valid={int(n_qwen)}",
    #     flush=True,
    # )
    ans_correct, n_cor = prefill_answer_text_or_skip(
        gpt2_model, gpt2_tok, gpt2_device, question, correct_answer, num_stages_box
    )
    ans_incorrect, n_inc = prefill_answer_text_or_skip(
        gpt2_model, gpt2_tok, gpt2_device, question, incorrect_answer, num_stages_box
    )

    num_stages = num_stages_box[0]
    prefix = gpt2_question_prefix(question)
    enc_prompt = gpt2_tok(
        prefix,
        return_tensors="pt",
        add_special_tokens=True,
        return_attention_mask=True,
    )
    pr_ids = enc_prompt["input_ids"].to(gpt2_device)
    pr_mask = enc_prompt["attention_mask"].to(gpt2_device)
    prompt_len = int(pr_ids.shape[1])

    with torch.inference_mode():
        gen_out = gpt2_model.generate(
            pr_ids,
            attention_mask=pr_mask,
            max_new_tokens=int(NUM_ANSWER_TOKENS),
            do_sample=False,
            pad_token_id=gpt2_pad,
        )
    full_gen = gen_out
    if int(full_gen.shape[1]) < prompt_len + NUM_ANSWER_TOKENS:
        print(
            f"[{idx}] skip: generation shorter than {NUM_ANSWER_TOKENS} tokens",
            flush=True,
        )
        continue

    gen_seg = full_gen[:, : prompt_len + NUM_ANSWER_TOKENS]
    gen_mask = torch.ones_like(gen_seg, dtype=torch.long, device=gpt2_device)

    stacks_gen, num_stages_g = gpt2_capture_hidden_stages(gpt2_model, gen_seg, gen_mask)
    assert num_stages_g == num_stages
    ans_gen = slice_answer_stages(stacks_gen, prompt_len, NUM_ANSWER_TOKENS)
    if ans_gen is None:
        print(f"[{idx}] skip: generated slice failed", flush=True)
        continue

    prefill_text_q = prefix + qwen_answer.strip()
    a0_q = gpt2_answer_first_token_index(gpt2_tok, prefill_text_q, question)

    Lq, Lc, Li = (
        int(ans_qwen.shape[1]),
        int(ans_correct.shape[1]),
        int(ans_incorrect.shape[1]),
    )
    Lg = int(ans_gen.shape[1])
    concat_range = {
        "prefill_qwen": [run_q, run_q + Lq],
        "prefill_correct": [run_cor, run_cor + Lc],
        "prefill_incorrect": [run_inc, run_inc + Li],
        "generated": [run_gen, run_gen + Lg],
    }
    run_q += Lq
    run_cor += Lc
    run_inc += Li
    run_gen += Lg

    prefill_qwen_list.append(activations_to_token_layer_dim(ans_qwen.half()))
    prefill_correct_list.append(activations_to_token_layer_dim(ans_correct.half()))
    prefill_incorrect_list.append(activations_to_token_layer_dim(ans_incorrect.half()))
    generated_list.append(activations_to_token_layer_dim(ans_gen.half()))

    gen_text = gpt2_tok.decode(
        gen_out[0, prompt_len:], skip_special_tokens=True
    ).strip()

    if idx == 0:
        _inner = getattr(gpt2_tok, "tokenizer", gpt2_tok)
        pref_ids = _gpt2_input_id_list(gpt2_tok, prefix)
        pref_pieces = _inner.convert_ids_to_tokens(pref_ids)
        gen_ids = gen_seg[0].detach().cpu().tolist()
        gen_pieces = _inner.convert_ids_to_tokens(gen_ids)
        g0, g1 = prompt_len, prompt_len + int(NUM_ANSWER_TOKENS)
        lines = [
            "\n--- sample [0] GPT-2 token ids + pieces (answer boundaries) ---",
            f"question (raw):\n{question}\n",
            f"prefix only (question + trailing newline): seq_len={len(pref_ids)}",
            f"  ids:     {pref_ids}",
            f"  pieces:  {pref_pieces}",
            "",
            *_gpt2_token_debug_lines(
                gpt2_tok,
                "prefill_qwen (prefix + Qwen body)",
                full_text=prefix + qwen_answer.strip(),
                question=question,
            ),
            "",
            *_gpt2_token_debug_lines(
                gpt2_tok,
                "prefill_correct (prefix + TruthfulQA correct)",
                full_text=prefix + correct_answer.strip(),
                question=question,
            ),
            "",
            *_gpt2_token_debug_lines(
                gpt2_tok,
                "prefill_incorrect (prefix + TruthfulQA incorrect)",
                full_text=prefix + incorrect_answer.strip(),
                question=question,
            ),
            "",
            f"generated (same row ids as `generate`): seq_len={len(gen_ids)} | answer@[{g0}:{g1})",
            f"  ids (full):      {gen_ids}",
            f"  pieces (full):   {gen_pieces}",
            f"  answer_ids:       {gen_ids[g0:g1]}",
            f"  answer_pieces:    {gen_pieces[g0:g1]}",
            f"  decode(new_only): {gen_text!r}",
            "---",
        ]
        print("\n".join(lines), flush=True)

    records.append(
        {
            "index": int(idx),
            "question": question,
            "dataset_correct_answer": correct_answer,
            "dataset_incorrect_answer": incorrect_answer,
            "qwen_answer": qwen_answer,
            "gpt2_generated_answer": gen_text,
            "gpt2_prefill_qwen_answer_token_start": int(a0_q),
            "gpt2_prompt_len": int(prompt_len),
            "num_stages": int(num_stages),
            "gpt2_answer_tokens_valid": {
                "prefill_qwen": int(n_qwen),
                "prefill_correct": int(n_cor),
                "prefill_incorrect": int(n_inc),
            },
            "gpt2_concat_token_range": concat_range,
        }
    )

    if (idx + 1) % 10 == 0:
        print(f"[{idx + 1}/{len(ROWS)}] stored {len(records)} samples", flush=True)

if not prefill_qwen_list:
    raise RuntimeError("No samples collected; check filters and generation.")

prefill_qwen_batch = torch.cat(prefill_qwen_list, dim=0)
prefill_correct_batch = torch.cat(prefill_correct_list, dim=0)
prefill_incorrect_batch = torch.cat(prefill_incorrect_list, dim=0)
generated_batch = torch.cat(generated_list, dim=0)

payload = {
    "qwen_id": QWEN_ID,
    "gpt2_id": GPT2_ID,
    "dataset": f"{DATASET_NAME}/{DATASET_CONFIG}",
    "split": DATASET_SPLIT,
    "num_answer_tokens_cap": int(NUM_ANSWER_TOKENS),
    "tensor_layout": (
        "Each activation tensor is (n_token_total, num_stages, hidden): samples are concatenated "
        "along n_token. Axis order is [n_token, n_layer, n_dim]. Prefill branches use at most "
        "NUM_ANSWER_TOKENS BPE tokens per sample (no padding). Generated uses exactly "
        "NUM_ANSWER_TOKENS new tokens per sample. records[].gpt2_concat_token_range maps each "
        "sample into the concatenated tensor per branch; gpt2_answer_tokens_valid counts real "
        "prefill answer tokens."
    ),
    "groups": (
        "prefill_qwen: GPT-2 on question + Qwen completion; "
        "prefill_correct / prefill_incorrect: GPT-2 on question + TruthfulQA reference; "
        "generated: GPT-2 autoregressive continuation after the same question prefix"
    ),
    "tensor_shape": {
        "prefill_qwen": list(prefill_qwen_batch.shape),
        "prefill_correct": list(prefill_correct_batch.shape),
        "prefill_incorrect": list(prefill_incorrect_batch.shape),
        "generated": list(generated_batch.shape),
    },
    "stage_order": (
        "stage 0 = hidden entering decoder block 0; "
        "stage k>0 = output after decoder block k-1"
    ),
    "prefill_qwen": prefill_qwen_batch,
    "prefill_correct": prefill_correct_batch,
    "prefill_incorrect": prefill_incorrect_batch,
    "generated": generated_batch,
    "records": records,
}
_repo = stonesoup.repo_root()
_cache_rel = CACHE_PATH.relative_to(_repo).as_posix()
payload["cache_path_repo_relative"] = _cache_rel

torch.save(payload, CACHE_PATH)
MANIFEST_PATH.write_text(
    json.dumps(
        {
            "cache_path": str(CACHE_PATH.resolve()),
            "cache_path_repo_relative": _cache_rel,
            "num_samples": len(records),
            "tensor_layout": "(n_token_total, num_stages, hidden); samples concat on dim0",
            "tensor_shapes": {
                "prefill_qwen": list(prefill_qwen_batch.shape),
                "prefill_correct": list(prefill_correct_batch.shape),
                "prefill_incorrect": list(prefill_incorrect_batch.shape),
                "generated": list(generated_batch.shape),
            },
            "qwen_id": QWEN_ID,
            "gpt2_id": GPT2_ID,
        },
        indent=2,
    ),
    encoding="utf-8",
)

print("Wrote:", _cache_rel, flush=True)
print("Wrote:", MANIFEST_PATH.relative_to(_repo).as_posix(), flush=True)
print("prefill_qwen:", tuple(prefill_qwen_batch.shape), flush=True)
print("prefill_correct:", tuple(prefill_correct_batch.shape), flush=True)
print("prefill_incorrect:", tuple(prefill_incorrect_batch.shape), flush=True)
print("generated:", tuple(generated_batch.shape), flush=True)
