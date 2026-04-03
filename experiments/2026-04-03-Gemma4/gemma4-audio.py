# %% Imports, paths
from __future__ import annotations

import html

import torch
import stonesoup

# https://huggingface.co/google/gemma-4-E2B-it — audio before text in the user message.
MODEL_ID = "google/gemma-4-E2B-it"
AUDIO_MP3 = stonesoup.repo_root() / "data" / "audio" / "explain.mp3"
USER_PROMPT = "Rephrase the audio."
# USER_PROMPT = "Describe the audio."
MAX_NEW_TOKENS = 512

if not AUDIO_MP3.is_file():
    raise FileNotFoundError(AUDIO_MP3)

print(stonesoup.STONESOUP_RENDER_HTML, end="")
print(
    f"<p><strong>{html.escape(MODEL_ID)}</strong></p>"
    f"<p>{html.escape(USER_PROMPT)}</p>"
    f"<p><code>{html.escape(str(AUDIO_MP3))}</code></p>",
    flush=True,
)

# %% Load model & run

model, processor = stonesoup.load_model(MODEL_ID)
model.eval()
device = next(model.parameters()).device
print("Loaded:", MODEL_ID, device, flush=True)

# %% run
audio_uri = str(AUDIO_MP3.resolve())
messages = [
    {
        "role": "user",
        "content": [
            {"type": "audio", "audio": audio_uri},
            {"type": "text", "text": USER_PROMPT},
        ],
    },
]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    add_generation_prompt=True,
).to(device)
input_len = inputs["input_ids"].shape[-1]

for key in ("input_features", "input_features_mask"):
    if key not in inputs or inputs[key] is None:
        raise RuntimeError(
            f"Missing {key} — need full AutoProcessor + audio path (got {audio_uri!r})."
        )

# Do not pass do_sample=False: hub generation_config uses sampling (do_sample=True); greedy hurts this checkpoint.
with torch.inference_mode():
    out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=True)

response = processor.decode(out[0][input_len:], skip_special_tokens=False)
parsed = processor.parse_response(response)
answer = parsed["content"] if isinstance(parsed, dict) and isinstance(parsed.get("content"), str) else str(parsed)

print(stonesoup.STONESOUP_RENDER_MD, end="")
print(f"## Assistant\n\n{answer}", flush=True)
