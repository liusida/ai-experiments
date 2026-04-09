# Experiments

Dated project folders live here (not at the repository root). Each folder is usually `YYYY-MM-DD-Topic/`.

## Folders

| Folder | Theme |
|--------|--------|
| **`2026-03-23-Embedding/`** | Token embedding norms and statistics, Alice cache, Qwen3.5 stats, `demo.py`. |
| **`2026-03-24-Explain-Embedding/`** | Concept-style embeddings, interpolation between vectors, mean cosine distributions. |
| **`2026-03-25-Classic-Word2Vec/`** | Classic Word2Vec-style embedding exploration. |
| **`2026-03-25-Reversal-Curse/`** | Multi-model reversal-curse runs and reports. |
| **`2026-03-25-VLM/`** | Vision–language model exploration (e.g. Qwen-VL), patch cosine HTML outputs. |
| **`2026-03-28-Qwen3-VL-MMStar/`** | Qwen3-VL on MMStar: eval, attention visualization, image→text info flow. |
| **`2026-03-29-Qwen3.5/`** | Qwen3.5 scratch scripts, embedding distributions, MMStar. |
| **`2026-03-31-Qwen3.5/`** | Qwen3.5 attention ablation, Qwen3-VL 3.5 comparison, scratch. |
| **`2026-04-01-Contrastive-Decoding/`** | MoD (Chen et al.)–style VLM demo with Qwen3-VL / Stonesoup (`MoD.py`). |
| **`2026-04-01-Gated-DeltaNet/`** | Qwen3.5 with gated delta net. |
| **`2026-04-02-Hypersphere/`** | Embeddings on the hypersphere, token arithmetic. |
| **`2026-04-03-Gemma4/`** | Gemma 4 smoke tests and audio. |
| **`2026-04-04-Spherical-Steering/`** | Hidden states, unit-sphere prototypes, TruthQA → activations, norms. |
| **`2026-04-06-Activation-Collection/`** | Rich activation collection: prefill vs generated, neighbors, homonyms, CKA, Gemma4, SCDT, etc. |
| **`2026-04-08-Multiple-Models/`** | Loading and comparing several HF models; activation summaries and cosine probes. |
| **`2026-04-09-New-API/`** | Stonesoup **`stonesoup.experiment`** API: activation deltas (`activation-changes.py`), residual vector norms/cosines (`vector-measurements.py`), unigram-style baseline (`transformer-without-attention-is-unigram.py`). |
| **`Demo/`** | Small Stonesoup demos (`demo.py`, experiment API demo). |

## Paths

- **Stonesoup watch path** (repo-relative), e.g. `experiments/2026-03-23-Embedding/demo.py`.
- **Conventions** for experiment Python: see repo root **`EXPERIMENT_PYTHON.md`**.
- **Python `REPO_ROOT`** from `experiments/SomeFolder/script.py`: `Path(__file__).resolve().parent.parent.parent`.
- **Shared data** stays at repo root: `data/`, etc.
