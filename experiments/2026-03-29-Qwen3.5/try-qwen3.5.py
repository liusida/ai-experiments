# %% Imports & paths

"""Load **Qwen3.5-4B** ([`Qwen/Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B)) and describe an image passed as a **URL** (here a ``file://`` URI to ``data/images/dog-tiny.png``).

Requires a recent ``transformers`` with ``Qwen3_5ForConditionalGeneration`` (this repo’s env uses 5.3+).
First run downloads several GiB from Hugging Face; GPU recommended (``device_map="auto"`` on CUDA).

**Terminal (full script):** ``uv run python experiments/2026-03-29-Qwen3.5/try-qwen3.5.py``

**Stonesoup:** **Watch** this file; run **Load model & processor** then **Describe dog-tiny.png** (or **Reset** and run both). By default the chat template may include a *thinking* span before the visible answer; ``batch_decode`` returns the whole assistant string unless you strip that span yourself.
"""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

# ``try-qwen3.5.py`` → ``experiments/2026-03-29-Qwen3.5/`` → repo root
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
IMAGE_PATH = REPO_ROOT / "data" / "images" / "dog-tiny.png"
IMAGE_URL = IMAGE_PATH.resolve().as_uri()
MODEL_ID = "Qwen/Qwen3.5-4B"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("REPO_ROOT:", REPO_ROOT)
print("IMAGE_URL:", IMAGE_URL)
print("MODEL_ID:", MODEL_ID)
print("DEVICE:", DEVICE)

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

USER_PROMPT = "Describe this image in a few short sentences."

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
    return_dict=True,
    return_tensors="pt",
)
inputs = inputs.to(DEVICE)

max_new_tokens = 512
with torch.inference_mode():
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

trimmed = [
    out_ids[len(in_ids) :]
    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
text = processor.batch_decode(
    trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)[0]
print(text)
