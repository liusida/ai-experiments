# %% Imports & setup
"""Interpolate between two embedding rows (SLERP on the unit sphere), then for each step find the vocab token (other than the two endpoints) with highest cosine similarity.

Cache: ``data/embedding-layers/<hf_id_slashes_to__>.pt`` with a ``weight`` tensor ``(V, d)``.

**Run:** ``uv run python experiments/2026-03-24-Explain-Embedding/interpolate_two_vectors.py``

**Stonesoup:** Run **Token A** / **Token B** cells to set ``CELL_INPUT`` fields (optional; defaults below), then the main cell.
"""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = REPO_ROOT / "data" / "embedding-layers"

MODEL_NAME = "openai-community/gpt2-medium"
TOKEN_A = "0"
TOKEN_B = "9"
N_STEPS = 10  # includes endpoints: t = 0, 1/(N_STEPS-1), …, 1


def cache_path(model_name: str) -> Path:
    return CACHE_DIR / f"{model_name.replace('/', '__')}.pt"


def slerp_unit(a: torch.Tensor, b: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """``a``, ``b``: (d,) unit. ``t``: (n,) in [0, 1]. Returns (n, d) unit rows."""
    dot = (a * b).sum().clamp(-1.0, 1.0)
    omega = torch.acos(dot)
    sin_o = torch.sin(omega)
    if sin_o.abs() < 1e-6:
        m = (1.0 - t).unsqueeze(-1) * a + t.unsqueeze(-1) * b
        return m / m.norm(dim=-1, keepdim=True)
    s0 = torch.sin((1.0 - t) * omega) / sin_o
    s1 = torch.sin(t * omega) / sin_o
    return s0.unsqueeze(-1) * a + s1.unsqueeze(-1) * b


def single_token_id(text: str, tokenizer) -> int:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"{text!r} encodes to {len(ids)} tokens {ids}; pick strings that are one token.")
    return ids[0]


# %% Token A # stonesoup:cell-input
_in = globals().get("CELL_INPUT", "")
if _in:
    TOKEN_A = _in

# %% Token B # stonesoup:cell-input
_in = globals().get("CELL_INPUT", "")
if _in:
    TOKEN_B = _in

# %% Load cache, tokenizer, interpolate, nearest neighbors
def main() -> None:
    path = cache_path(MODEL_NAME)
    if not path.is_file():
        raise FileNotFoundError(f"Missing cache: {path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(path, map_location=str(device), weights_only=False)
    W = payload["weight"].float()
    if W.dim() != 2:
        raise ValueError(f"expected 2d weight, got {W.shape}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    ia = single_token_id(TOKEN_A, tokenizer)
    ib = single_token_id(TOKEN_B, tokenizer)

    a = W[ia] / W[ia].norm()
    b = W[ib] / W[ib].norm()

    t = torch.linspace(0.0, 1.0, N_STEPS, device=device)
    V = slerp_unit(a, b, t)

    Wn = W / (W.norm(dim=1, keepdim=True) + 1e-8)
    cos = V @ Wn.T
    cos[:, ia] = float("-inf")
    cos[:, ib] = float("-inf")

    top_k = min(3, cos.shape[1])
    top_val, top_j = torch.topk(cos, k=top_k, dim=1)

    print(f"model={MODEL_NAME}  cache={path.name}  A={TOKEN_A!r} id={ia}  B={TOKEN_B!r} id={ib}")
    print(f"cos(A,B)={float((a * b).sum()):.6f}  N_STEPS={N_STEPS}\n")
    for k in range(N_STEPS):
        bits = []
        for r in range(top_k):
            j = int(top_j[k, r])
            bits.append(f"{float(top_val[k, r]):.6f}  {tokenizer.decode([j])!r}")
        print(f"{k:3d}  " + "  |  ".join(bits))


if __name__ == "__main__":
    main()
