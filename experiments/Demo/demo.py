# %% Imports
import stonesoup
import torch
from PIL import Image

# %% Minimal plot
import matplotlib.pyplot as plt

plt.figure(figsize=(4, 3))
plt.plot([0, 1, 2], [0, 1, 0])
plt.title("demo")
stonesoup.show()
plt.close("all")

# %% POPE: first image (id=0)
# Needs: ``uv pip install datasets`` (Hugging Face ``datasets``).
import uuid

from datasets import load_dataset

from stonesoup import STONESOUP_RENDER_HTML

_pope_ds = load_dataset("lmms-lab/POPE", split="test")
IMAGE_ID = 0
_pope_row = _pope_ds[IMAGE_ID]
_raw_img = _pope_row["image"]
if isinstance(_raw_img, Image.Image):
    pope_img = _raw_img.convert("RGB")
else:
    raise TypeError(f"POPE image field: expected PIL.Image, got {type(_raw_img)}")

_pope_png = stonesoup.plot_dir() / f"pope_demo_image{IMAGE_ID}_{uuid.uuid4().hex[:8]}.png"
pope_img.save(_pope_png, format="PNG")
_rel = _pope_png.relative_to(stonesoup.repo_root()).as_posix()
_src = f"/{_rel}"
print(STONESOUP_RENDER_HTML, end="")
print(
    f'<p class="stonesoup-show"><img src="{_src}" alt="POPE image_id={IMAGE_ID}" loading="lazy" /></p>'
    f"<p><code>POPE[{IMAGE_ID}]</code> image_source={_pope_row.get('image_source')!r} "
    f"question_id={_pope_row.get('question_id')!r}</p>",
    flush=True,
)

# %% Load Qwen3.5-4B
MODEL_ID = "Qwen/Qwen3.5-4B"
model, processor = stonesoup.load_model(MODEL_ID)

# %% Forward pass (prefill)
image = Image.new("RGB", (64, 64), color=(128, 128, 128))
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "What color is this image?"},
        ],
    },
]
text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)
inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
device = next(model.parameters()).device
inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
with torch.inference_mode():
    outputs = model(**inputs)
print("logits shape:", tuple(outputs.logits.shape), flush=True)
