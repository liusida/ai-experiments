# %% Imports & load
from __future__ import annotations

import stonesoup
import torch
from stonesoup.experiment import decoder_blocks, inner_tokenizer

MODEL_ID = "google/gemma-4-E4B-it"
PROMPT = "Explain the mechanism of an LLM using one concise sentence."
MAX_NEW_TOKENS = 64

model, proc = stonesoup.load_model(MODEL_ID)
model.eval()
tokenizer = inner_tokenizer(proc)
device = next(model.parameters()).device

blocks = decoder_blocks(model)


def generate(prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)


# Per-layer Δ_ple scale: 1.0 = original, 0.0 = zero out.
num_layers = len(blocks)
ple_scales = [1.0] * num_layers


def make_ple_hook(i: int):
    def hook(_module, _inputs, output):
        s = ple_scales[i]
        if isinstance(output, tuple):
            return (output[0] * s,) + output[1:]
        return output * s
    return hook


# %% Register per-layer hooks once; mutate `ple_scales` to control
handles = [
    b.post_per_layer_input_norm.register_forward_hook(make_ple_hook(i))
    for i, b in enumerate(blocks)
]

# %% Baseline (all PLE enabled)
ple_scales[:] = [1.0] * num_layers
print("=== Baseline (all layers PLE enabled) ===")
print(generate(PROMPT))

# %% Sweep: disable PLE at one layer at a time
for target in range(num_layers):
    stonesoup.check_abort()
    ple_scales[:] = [1.0] * num_layers
    ple_scales[target] = 0.0
    print(f"=== PLE disabled at L{target} ===")
    print(generate(PROMPT))

# %% Sweep: disable PLE at two consecutive layers at a time
for target in range(num_layers - 1):
    stonesoup.check_abort()
    ple_scales[:] = [1.0] * num_layers
    ple_scales[target] = 0.0
    ple_scales[target + 1] = 0.0
    print(f"=== PLE disabled at L{target}, L{target + 1} ===")
    print(generate(PROMPT))

# %% Sweep: disable PLE at five consecutive layers at a time
WINDOW = 5
for target in range(num_layers - WINDOW + 1):
    stonesoup.check_abort()
    ple_scales[:] = [1.0] * num_layers
    for j in range(target, target + WINDOW):
        ple_scales[j] = 0.0
    print(f"=== PLE disabled at L{target}..L{target + WINDOW - 1} ===")
    print(generate(PROMPT))

# %% Sweep: disable PLE at ten consecutive layers at a time
WINDOW = 10
for target in range(num_layers - WINDOW + 1):
    stonesoup.check_abort()
    ple_scales[:] = [1.0] * num_layers
    for j in range(target, target + WINDOW):
        ple_scales[j] = 0.0
    print(f"=== PLE disabled at L{target}..L{target + WINDOW - 1} ===")
    print(generate(PROMPT))

# %% Sweep: disable PLE at 15 consecutive layers at a time
WINDOW = 15
for target in range(num_layers - WINDOW + 1):
    stonesoup.check_abort()
    ple_scales[:] = [1.0] * num_layers
    for j in range(target, target + WINDOW):
        ple_scales[j] = 0.0
    print(f"=== PLE disabled at L{target}..L{target + WINDOW - 1} ===")
    print(generate(PROMPT))


# %% Sweep: disable PLE at twenty consecutive layers at a time
WINDOW = 20
for target in range(num_layers - WINDOW + 1):
    stonesoup.check_abort()
    ple_scales[:] = [1.0] * num_layers
    for j in range(target, target + WINDOW):
        ple_scales[j] = 0.0
    print(f"=== PLE disabled at L{target}..L{target + WINDOW - 1} ===")
    print(generate(PROMPT))

# %% Sweep: disable PLE at forty consecutive layers at a time
WINDOW = 40
for target in range(num_layers - WINDOW + 1):
    stonesoup.check_abort()
    ple_scales[:] = [1.0] * num_layers
    for j in range(target, target + WINDOW):
        ple_scales[j] = 0.0
    print(f"=== PLE disabled at L{target}..L{target + WINDOW - 1} ===")
    print(generate(PROMPT))

# %% Disable PLE everywhere (all scales = 0)
ple_scales[:] = [0.0] * num_layers
print("=== PLE disabled on all layers ===")
print(generate(PROMPT))

# %% Diagnostic: does the hook actually fire and alter logits?
fire_count = [0]


def diag_hook(_m, _in, out):
    fire_count[0] += 1
    return out  # identity; we just want to count


diag_handles = [b.post_per_layer_input_norm.register_forward_hook(diag_hook) for b in blocks]

messages = [{"role": "user", "content": PROMPT}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(device)

fire_count[0] = 0
ple_scales[:] = [1.0] * num_layers
with torch.no_grad():
    logits_on = model(**inputs).logits[0, -1].detach().float().cpu()
fires_on = fire_count[0]

fire_count[0] = 0
ple_scales[:] = [0.0] * num_layers
with torch.no_grad():
    logits_off = model(**inputs).logits[0, -1].detach().float().cpu()
fires_off = fire_count[0]

for h in diag_handles:
    h.remove()

diff = (logits_on - logits_off).abs().max().item()
print(f"hook fires — on: {fires_on}, off: {fires_off} (expected {num_layers} each)")
print(f"max |Δlogits| between PLE on vs off: {diff:.4e}")
print(f"top-1 (on):  {tokenizer.decode([int(logits_on.argmax())])!r}")
print(f"top-1 (off): {tokenizer.decode([int(logits_off.argmax())])!r}")

# %% Diagnostic: does the hook fire during model.generate()?
fire_count = [0]


def diag_hook_gen(_m, _in, out):
    fire_count[0] += 1
    return out


diag_handles = [b.post_per_layer_input_norm.register_forward_hook(diag_hook_gen) for b in blocks]

fire_count[0] = 0
ple_scales[:] = [0.0] * num_layers
text_gen_off = generate(PROMPT)
fires_gen = fire_count[0]

for h in diag_handles:
    h.remove()

print(f"generate() hook fires with PLE off: {fires_gen} (expect ≥ {num_layers} * n_generated_tokens)")
print(f"generated with PLE off (via generate): {text_gen_off!r}")

# %% Remove hooks
for h in handles:
    h.remove()
