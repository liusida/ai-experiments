# %% Imports & paths
"""For each cached embedding ``.pt`` in ``data/embedding-layers``, find the **3** smallest row L2
norms and write decoded tokens in backticks.

Output shape (per model)::

    Model/id * :
    `token1`
    `token2`
    `token3`

``*`` means ``len(AutoTokenizer) !=`` embedding row count in the ``.pt`` (ids may not match strings).
``!`` means tokenizer load failed (see following line).

Writes ``experiments/2026-03-23-Embedding/reports/least_norm_tokens.txt``. Run from repo root::

    uv run python experiments/2026-03-23-Embedding/embedding_least_norm_tokens.py
"""

from __future__ import annotations

from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
LAYER_DIR = REPO_ROOT / "data" / "embedding-layers"
REPORTS_DIR = Path(__file__).resolve().parent / "reports"
OUT_FILE = REPORTS_DIR / "least_norm_tokens.txt"
TOP_K = 3

# %% Helpers


def display_name_from_path(path: Path, payload: dict) -> str:
    m = payload.get("model_name")
    if isinstance(m, str) and m.strip():
        return m.strip()
    return path.stem.replace("__", "/")


def describe_decode(tokenizer: AutoTokenizer, token_id: int) -> str:
    try:
        return tokenizer.decode([token_id], skip_special_tokens=False)
    except Exception:
        return "?"


def escape_inside_token(s: str) -> str:
    """Escape so token fits on one line between backticks (newlines → \\n, etc.)."""
    return (
        s.replace("\\", "\\\\")
        .replace("\r", "\\r")
        .replace("\n", "\\n")
        .replace("`", "\\`")
    )


def backtick_line(decoded: str) -> str:
    """One full line: opening `, content, closing `. Empty decode → ∅."""
    if decoded == "":
        inner = "∅"
    else:
        inner = escape_inside_token(decoded)
    return "`" + inner + "`"


# %% Build report
lines: list[str] = []

pt_files = sorted(p for p in LAYER_DIR.glob("*.pt") if p.is_file())
if not pt_files:
    lines.append("(no .pt files found)")
else:
    for path in tqdm(pt_files, desc="Least-norm tokens", unit="ckpt"):
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as e:
            lines.append(f"{path.name} ! :")
            lines.append(f"load failed: {e!s}")
            lines.append("")
            continue
        if not isinstance(payload, dict):
            lines.append(f"{path.name} ! :")
            lines.append("checkpoint is not a dict")
            lines.append("")
            continue

        w = payload.get("weight")
        if not torch.is_tensor(w) or w.dim() != 2:
            lines.append(f"{path.name} ! :")
            lines.append("bad weight tensor")
            lines.append("")
            continue

        name = display_name_from_path(path, payload)
        vocab, _hidden = w.shape
        norms = w.float().norm(dim=1).cpu()
        k = min(TOP_K, int(norms.numel()))
        smallest = torch.argsort(norms)[:k]

        tokenizer: AutoTokenizer | None = None
        tok_err: str | None = None
        try:
            tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        except Exception as e:
            tok_err = str(e)

        if tok_err is not None:
            te = tok_err if len(tok_err) <= 200 else tok_err[:197] + "…"
            lines.append(f"{name} ! :")
            lines.append(te)
            lines.append("")
            continue

        assert tokenizer is not None
        vocab_mismatch = len(tokenizer) != vocab
        header = f"{name} * :" if vocab_mismatch else f"{name} :"
        lines.append(header)
        for tid in smallest.tolist():
            lines.append(backtick_line(describe_decode(tokenizer, int(tid))))
        lines.append("")

report_body = "\n".join(lines).rstrip() + "\n"

# %% Write report
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE.write_text(report_body, encoding="utf-8")
tqdm.write(report_body, end="")
tqdm.write(f"wrote {OUT_FILE}")
