# %%
"""Transformers playground: encoder (embeddings) + decoder-only (generation).

From repo root:

    uv run 2026-03-23-Embedding/demo.py

Interactive: select the repo ``.venv`` interpreter, then **Run Cell** on each ``# %%`` block.
Encoder and decoder use separate weights (``model`` vs ``dec_model``) so both can stay loaded.
"""

from __future__ import annotations

import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

# --- Encoder-only (bidirectional) ---
MODEL_NAME = "answerdotai/ModernBERT-base"
# MODEL_NAME = "microsoft/deberta-v3-base"
# MODEL_NAME = "FacebookAI/xlm-roberta-base"

# --- Decoder-only (causal LM) ---
DECODER_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
# DECODER_MODEL_NAME = "gpt2"
# DECODER_MODEL_NAME = "HuggingFaceTB/SmolLM2-360M-Instruct"


def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden.shape).float()
    summed = (last_hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-9)
    return summed / denom


def build_decoder_prompt(dec_tok: AutoTokenizer, user_text: str) -> str:
    if getattr(dec_tok, "chat_template", None) is None:
        return user_text
    messages = [{"role": "user", "content": user_text}]
    return dec_tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


# %% Encoder — load once
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(device)
model.eval()
print(f"[encoder] loaded {MODEL_NAME!r} on {device}")

# %% Encoder — sentence embeddings
texts = [
    "PyTorch runs on the GPU.",
    "Transformers load pretrained models from Hugging Face.",
]

enc = tokenizer(
    texts,
    padding=True,
    truncation=True,
    return_tensors="pt",
)
enc = {k: v.to(device) for k, v in enc.items()}

with torch.inference_mode():
    out = model(**enc)
    emb = mean_pool(out.last_hidden_state, enc["attention_mask"])
    emb = torch.nn.functional.normalize(emb, p=2, dim=1)

sim = (emb[0] @ emb[1]).item()
print(f"embedding shape: {tuple(emb.shape)}")
print(f"cosine similarity (first two sentences): {sim:.4f}")

# %% Encoder — token vectors (first sentence in ``texts``)
sample_text = texts[0]
tok_enc = tokenizer(sample_text, return_tensors="pt", add_special_tokens=True)
tok_enc = {k: v.to(device) for k, v in tok_enc.items()}

with torch.inference_mode():
    tok_out = model(**tok_enc)
    hidden = tok_out.last_hidden_state[0]

ids = tok_enc["input_ids"][0].tolist()
pieces = tokenizer.convert_ids_to_tokens(ids)
print(f"sample: {sample_text!r}")
print(f"hidden size per token: {hidden.shape[-1]}")
for i, (piece, tid) in enumerate(zip(pieces, ids)):
    vec = hidden[i].detach().float().cpu()
    head = ", ".join(f"{x:.4f}" for x in vec[:6])
    print(f"  {i:2}  {piece!r:18}  id={tid:5}  vec[:6]= [{head}, ...]")

# %% Decoder — load causal LM (separate from encoder)
dec_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
dec_tokenizer = AutoTokenizer.from_pretrained(DECODER_MODEL_NAME, trust_remote_code=True)
dec_model = AutoModelForCausalLM.from_pretrained(
    DECODER_MODEL_NAME,
    dtype=dec_dtype if dec_dtype != torch.float32 else None,
    device_map=None,
    trust_remote_code=True,
)
dec_model.to(device)
dec_model.eval()

if dec_tokenizer.pad_token_id is None and dec_tokenizer.eos_token_id is not None:
    dec_tokenizer.pad_token = dec_tokenizer.eos_token

print(f"[decoder] loaded {DECODER_MODEL_NAME!r} on {device} ({dec_dtype})")

# %% Decoder — generate (edit ``user_prompt`` and re-run)
user_prompt = "In one short paragraph, what does a decoder-only transformer do?"
prompt = build_decoder_prompt(dec_tokenizer, user_prompt)
inputs = dec_tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
inputs = {k: v.to(device) for k, v in inputs.items()}

max_new_tokens = 256
with torch.inference_mode():
    out_gen = dec_model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=dec_tokenizer.pad_token_id,
    )

new_tokens = out_gen[0, inputs["input_ids"].shape[1] :]
reply = dec_tokenizer.decode(new_tokens, skip_special_tokens=True)
print("--- prompt ---")
print(user_prompt)
print("--- model ---")
print(reply.strip())
