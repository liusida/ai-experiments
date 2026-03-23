# Agent notes

## Hardware (this machine)

- **GPU:** NVIDIA GB10 (Blackwell SoC); driver 580.x; **CUDA 13.0** (per `nvidia-smi`).
- **Arch:** **linux aarch64** (ARM). CPU: Cortex-X925 / Cortex-A725 mix, 20 logical CPUs.
- **RAM:** ~119 GiB system memory (unified-memory style; `nvidia-smi` may not report classic VRAM usage).
- **ML implications:** Use **PyTorch builds for Linux aarch64 + CUDA**, not generic x86 wheels; prefer **recent** PyTorch for GB10 support. Avoid assuming x86-only CUDA extras (e.g. some optional compiled extensions).

## Stonesoup

Local tool under [`stonesoup/`](stonesoup/): watches a `.py` file for `# %%` cells, **FastAPI** kernel on port **8765**, **Vite** UI in a browser. See [`stonesoup/README.md`](stonesoup/README.md). From repo root run **`uv pip install -e ".[stonesoup]"`** so `stonesoup` is importable without `PYTHONPATH`.

When authoring experiment scripts to watch in Stonesoup, follow [`EXPERIMENT_PYTHON.md`](EXPERIMENT_PYTHON.md). **Do not run** those `.py` files to verify after editing unless the user asks—runs can be very slow (HF, big data); the user runs cells in the Stonesoup UI.
