# %% Imports & paths

"""Qwen3.5-4B image caption demo ([model card](https://huggingface.co/Qwen/Qwen3.5-4B)).

Image is loaded via HTTP from Stonesoup’s static mount: ``/data/image/dog-tiny.png`` (backend default ``127.0.0.1:8765``).
**Stonesoup:** Watch this file; run cells in order. **Terminal:** ``uv run python experiments/2026-03-29-Qwen3.5/try-qwen3.5.py``
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

# %% Describe dog-tiny.png

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
with torch.inference_mode():
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

tokens = [
    processor.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    for token_id in generated_ids[0].tolist()
]
print("|".join(f"{token}" for token in tokens))
