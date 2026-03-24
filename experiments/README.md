# Experiments

Dated project folders live here (not at the repository root). Each folder is usually `YYYY-MM-DD-Topic/`.

## Current folders

- **`2026-03-23-Embedding/`** — token embedding norms, caches, Qwen statistics, `demo.py`.
- **`2026-03-24-Explain-Embedding/`** — concept explanations with literal vs synthetic embeddings.

## Paths

- **Stonesoup watch path** (repo-relative), e.g. `experiments/2026-03-23-Embedding/demo.py`.
- **Python `REPO_ROOT`** from a script in `experiments/SomeFolder/script.py`: `Path(__file__).resolve().parent.parent.parent`.
- **Shared data** stays at repo root: `data/`, etc.
