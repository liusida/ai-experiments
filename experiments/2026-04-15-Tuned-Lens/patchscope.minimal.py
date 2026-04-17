# %% Imports
from __future__ import annotations

from dataclasses import dataclass

import torch

import stonesoup
from stonesoup.experiment import ensure_pad_token_via_eos, inner_tokenizer

# %% Config
MODEL_ID = "Qwen/Qwen3.5-9B"
SOURCE_PROMPT = "Amazon's former CEO attended the Oscars."
DEMO_TOKENS = ["cat", "hello", "135"]
PLACEHOLDER_TOKEN = "X"
SOURCE_TOKEN_TEXT = "CEO"  # prefer patching this substring from source prompt
SOURCE_TOKEN_INDEX = -1  # fallback index when SOURCE_TOKEN_TEXT is not found
USE_CHAT_TEMPLATE = True
CHAT_SYSTEM_PROMPT = "You are a concise assistant."
TARGET_ASSISTANT_PREFILL = f"{PLACEHOLDER_TOKEN} maps to"
MAX_NEW_TOKENS = 6
MAX_LAYERS_TO_SHOW = 8

# %% Load model
model, proc = stonesoup.load_model(MODEL_ID, torch_dtype="bfloat16")
model.eval()
model.requires_grad_(False)

tokenizer = inner_tokenizer(proc)
ensure_pad_token_via_eos(tokenizer)
device = next(model.parameters()).device

_inner = model.model
text_module = getattr(_inner, "language_model", _inner)
blocks = text_module.layers
n_layers = len(blocks)

print(f"Model: {MODEL_ID}", flush=True)
print(f"Layers: {n_layers} on {device}", flush=True)


@dataclass
class PatchscopeResult:
    layer: int
    first_token: str
    generation: str
    full_text: str


def _resolve_source_index(n_tokens: int, idx: int) -> int:
    if idx < 0:
        idx = n_tokens + idx
    return max(0, min(idx, n_tokens - 1))


def _find_subsequence_start(haystack: list[int], needle: list[int]) -> int:
    if not needle or len(needle) > len(haystack):
        return -1
    width = len(needle)
    for i in range(len(haystack) - width + 1):
        if haystack[i : i + width] == needle:
            return i
    return -1


def _find_all_subsequence_starts(haystack: list[int], needle: list[int]) -> list[int]:
    if not needle or len(needle) > len(haystack):
        return []
    width = len(needle)
    hits: list[int] = []
    for i in range(len(haystack) - width + 1):
        if haystack[i : i + width] == needle:
            hits.append(i)
    return hits


def _encode_prompt(prompt: str, assistant_prefill: str | None = None) -> tuple[torch.Tensor, str]:
    chat_template_fn = getattr(tokenizer, "apply_chat_template", None)
    if not USE_CHAT_TEMPLATE:
        raise RuntimeError("USE_CHAT_TEMPLATE must be True; raw encoding is disabled.")
    if not callable(chat_template_fn):
        raise RuntimeError("tokenizer has no callable apply_chat_template; raw encoding is disabled.")

    messages = [
        {"role": "system", "content": CHAT_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    if assistant_prefill:
        messages.append({"role": "assistant", "content": assistant_prefill})
    try:
        ids = chat_template_fn(
            messages,
            tokenize=True,
            add_generation_prompt=(assistant_prefill is None),
            continue_final_message=(assistant_prefill is not None),
            return_tensors="pt",
            enable_thinking=False,
        )
    except Exception as e:
        raise RuntimeError(f"apply_chat_template failed: {type(e).__name__}: {e}") from e

    if isinstance(ids, torch.Tensor):
        return ids.to(device), "chat_template"

    # Some tokenizers return BatchEncoding with input_ids instead of a tensor directly.
    if hasattr(ids, "get"):
        input_ids = ids.get("input_ids")
        if isinstance(input_ids, torch.Tensor):
            return input_ids.to(device), "chat_template"

    raise RuntimeError(
        f"apply_chat_template returned {type(ids).__name__} without tensor input_ids; raw encoding is disabled."
    )


def _find_token_sequence_end_position(
    token_ids: torch.Tensor,
    candidates: list[str],
) -> tuple[int, str] | None:
    seq = token_ids[0].tolist()
    for cand in candidates:
        ids = tokenizer.encode(cand, add_special_tokens=False)
        if not ids:
            continue
        start = _find_subsequence_start(seq, ids)
        if start >= 0:
            end = start + len(ids) - 1
            return end, f"token sequence match {cand!r}"
    return None


def _find_token_sequence_end_positions(
    token_ids: torch.Tensor,
    candidates: list[str],
) -> list[int]:
    seq = token_ids[0].tolist()
    hits: set[int] = set()
    for cand in candidates:
        ids = tokenizer.encode(cand, add_special_tokens=False)
        if not ids:
            continue
        for start in _find_all_subsequence_starts(seq, ids):
            hits.add(start + len(ids) - 1)
    return sorted(hits)


def _resolve_source_position(source_ids: torch.Tensor) -> tuple[int, str]:
    needle = SOURCE_TOKEN_TEXT.strip()
    if needle:
        pos = _find_token_sequence_end_position(
            source_ids,
            [needle, f" {needle}", needle.lower(), f" {needle.lower()}"],
        )
        if pos is not None:
            return pos
    fallback = _resolve_source_index(source_ids.shape[1], SOURCE_TOKEN_INDEX)
    return fallback, f"fallback index {SOURCE_TOKEN_INDEX}"


def _choose_layers(total_layers: int, max_layers: int) -> list[int]:
    if total_layers <= max_layers:
        return list(range(total_layers))
    picks = torch.linspace(0, total_layers - 1, steps=max_layers).round().to(torch.long).tolist()
    ordered_unique: list[int] = []
    for p in picks:
        if p not in ordered_unique:
            ordered_unique.append(int(p))
    return ordered_unique


def _token_identity_prompt(demo_tokens: list[str], placeholder_token: str) -> str:
    if len(demo_tokens) >= 2:
        return (
            f"Given examples: {demo_tokens[0]} maps to {demo_tokens[0]}; "
            f"{demo_tokens[1]} maps to {demo_tokens[1]}. "
            f"What does {placeholder_token} map to?"
        )
    if len(demo_tokens) == 1:
        return f"Given examples: {demo_tokens[0]} maps to {demo_tokens[0]}. What does {placeholder_token} map to?"
    return f"What does {placeholder_token} map to?"


def _decode_token_id(token_id: int) -> str:
    return tokenizer.decode([token_id]).replace("\n", "\\n")


def _run_patchscope_layer(
    layer_idx: int,
    source_vec: torch.Tensor,
    target_ids: torch.Tensor,
    target_positions: list[int],
    max_new_tokens: int,
) -> PatchscopeResult:
    def patch_hook(_module, _inputs, output):
        if isinstance(output, tuple):
            hidden = output[0]
            rest = output[1:]
        else:
            hidden = output
            rest = None

        valid_positions = [p for p in target_positions if p < hidden.shape[1]]
        if valid_positions:
            patched = hidden.clone()
            source_cast = source_vec.to(dtype=hidden.dtype, device=hidden.device)
            for p in valid_positions:
                patched[0, p, :] = source_cast
        else:
            patched = hidden

        if rest is None:
            return patched
        return (patched, *rest)

    handle = blocks[layer_idx].register_forward_hook(patch_hook)
    try:
        with torch.no_grad():
            out_ids = model.generate(
                target_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
    finally:
        handle.remove()

    continuation_ids = out_ids[0, target_ids.shape[1] :]
    first_id = int(continuation_ids[0].item()) if continuation_ids.numel() > 0 else int(tokenizer.eos_token_id)
    first_tok = _decode_token_id(first_id)
    generation = tokenizer.decode(continuation_ids, skip_special_tokens=True)
    full_text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
    return PatchscopeResult(layer=layer_idx, first_token=first_tok, generation=generation, full_text=full_text)


# %% Run patchscope (token identity) # stonesoup:cell-input
source_prompt = globals().get("CELL_INPUT", "") or SOURCE_PROMPT
target_prompt = _token_identity_prompt(DEMO_TOKENS, PLACEHOLDER_TOKEN)

source_ids, source_encoding_mode = _encode_prompt(source_prompt, assistant_prefill=None)
target_ids, target_encoding_mode = _encode_prompt(target_prompt, assistant_prefill=TARGET_ASSISTANT_PREFILL)

source_pos, source_pos_reason = _resolve_source_position(source_ids)
target_positions = _find_token_sequence_end_positions(
    target_ids,
    [PLACEHOLDER_TOKEN, f" {PLACEHOLDER_TOKEN}"],
)
if target_positions:
    target_pos_reason = "placeholder token sequence matches"
else:
    target_positions = [target_ids.shape[1] - 1]
    target_pos_reason = "fallback last target token"

source_token_id = int(source_ids[0, source_pos].item())
source_token_str = _decode_token_id(source_token_id)

layer_ids = list(range(n_layers))

# Cache source representations once for all layers at the chosen source token.
with torch.no_grad():
    source_out = text_module(input_ids=source_ids, output_hidden_states=True, use_cache=False)
source_vecs_by_layer = [
    source_out.hidden_states[layer_idx + 1][0, source_pos, :].detach() for layer_idx in range(n_layers)
]

print(f"Source prompt: {source_prompt}", flush=True)
print(f"Target prompt: {target_prompt}", flush=True)
print(f"Target assistant prefill: {TARGET_ASSISTANT_PREFILL}", flush=True)
print(f"Encoding mode: source={source_encoding_mode}, target={target_encoding_mode}", flush=True)
print(f"Source token @ pos {source_pos}: id={source_token_id}, token={source_token_str!r}", flush=True)
print(f"Source token selection: {source_pos_reason}", flush=True)
print(f"Patching target token @ positions {target_positions}: {PLACEHOLDER_TOKEN!r}", flush=True)
print(f"Target token selection: {target_pos_reason}", flush=True)
print("", flush=True)
print(f"{'layer':>5} | {'first generated token':<26} | generation", flush=True)
print("-" * 90, flush=True)

for layer_idx in layer_ids:
    result = _run_patchscope_layer(
        layer_idx=layer_idx,
        source_vec=source_vecs_by_layer[layer_idx],
        target_ids=target_ids,
        target_positions=target_positions,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    print(f"{result.layer:5d} | {result.first_token!r:<26} | {result.generation}", flush=True)
    # print(f"      full: {result.full_text}", flush=True)
