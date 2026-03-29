# %% Imports & paths

"""Qwen3.5-4B image caption demo ([model card](https://huggingface.co/Qwen/Qwen3.5-4B)).

Image is loaded via HTTP from Stonesoup’s static mount: ``/data/image/dog-tiny.png`` (backend default ``127.0.0.1:8765``).
**Stonesoup:** Watch this file; run cells in order. **Terminal:** ``uv run python experiments/2026-03-29-Qwen3.5/scratch-qwen3.5.py``
"""

from __future__ import annotations

import torch
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

MODEL_ID = "Qwen/Qwen3.5-4B"
IMAGE_URL = "http://127.0.0.1:8765/data/image/dog-tiny.png"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(MODEL_ID, DEVICE)

# %% Load model & processor

model = Qwen3_5ForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype="auto" if DEVICE.type == "cuda" else torch.float32,
    device_map="auto" if DEVICE.type == "cuda" else None,
)
if DEVICE.type != "cuda":
    model = model.to(DEVICE)

processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
model.eval()
print("Loaded:", MODEL_ID)

# %% Build prompt tensors

USER_PROMPT = "Is this a cat?"

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": IMAGE_URL},
            {"type": "text", "text": USER_PROMPT},
        ],
    }
]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    enable_thinking=False,
    return_dict=True,
    return_tensors="pt",
)
inputs = inputs.to(DEVICE)

max_new_tokens = 512

# %% Baseline generate

with torch.inference_mode():
    baseline_tokens = model.generate(**inputs, max_new_tokens=max_new_tokens)

baseline_token_pieces = [
    processor.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    for token_id in baseline_tokens[0].tolist()
]
print("|".join(baseline_token_pieces))

# %% Manual generation loop

with torch.inference_mode():
    tokens = inputs.input_ids.clone()
    attention_mask = inputs.attention_mask.clone()
    mm_token_type_ids = inputs.mm_token_type_ids.clone()
    eos_token_id = processor.tokenizer.eos_token_id

    for _ in range(max_new_tokens):
        # Unlike plain GPT-2, Qwen3.5 also needs the image tensors on every forward pass.
        logits = model(
            input_ids=tokens,
            attention_mask=attention_mask,
            mm_token_type_ids=mm_token_type_ids,
            pixel_values=inputs.pixel_values,
            image_grid_thw=inputs.image_grid_thw,
        ).logits
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)

        tokens = torch.cat([tokens, next_token], dim=1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones_like(next_token, dtype=attention_mask.dtype)],
            dim=1,
        )
        # Newly generated positions are text, not image placeholders.
        mm_token_type_ids = torch.cat(
            [mm_token_type_ids, torch.zeros_like(next_token, dtype=mm_token_type_ids.dtype)],
            dim=1,
        )

        if next_token.item() == eos_token_id:
            break

token_pieces = [
    processor.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    for token_id in tokens[0].tolist()
]
print("|".join(token_pieces))
