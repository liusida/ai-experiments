# %% Imports, config & helpers
import stonesoup
from stonesoup.experiment import (
    configure_matplotlib_agg,
    decoder_blocks,
    hf_repo_id_safe_stem,
    inner_tokenizer,
)
from transformers import AutoTokenizer
# %% Load model (toolbar Load or cell load)

MODEL_ID = "google/gemma-4-E4B-it"
model, proc = stonesoup.load_model(MODEL_ID, use_offline=False)

tokenizer = inner_tokenizer(proc)

candidates = [" capital", "capital", " Capital", "Capital", " capitol",  "What is the captial of Spain?",] # oops, capital, not captial... 
for s in candidates:
    ids = tokenizer.encode(s, add_special_tokens=False)
    pieces = tokenizer.convert_ids_to_tokens(ids)
    print(f"{s!r:14s} -> ids={ids}  pieces={pieces}  single_token={len(ids)==1}")

print()
print("Direct vocab lookups (raw SentencePiece piece strings):")
for piece in ["▁capital", "capital", "▁Capital"]:
    tid = tokenizer.convert_tokens_to_ids(piece)
    unk = tokenizer.unk_token_id
    print(f"  piece={piece!r}  id={tid}  in_vocab={tid is not None and tid != unk}")
