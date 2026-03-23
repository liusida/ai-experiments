# Agent notes

## Hardware (this machine)

- **GPU:** NVIDIA GB10 (Blackwell SoC); driver 580.x; **CUDA 13.0** (per `nvidia-smi`).
- **Arch:** **linux aarch64** (ARM). CPU: Cortex-X925 / Cortex-A725 mix, 20 logical CPUs.
- **RAM:** ~119 GiB system memory (unified-memory style; `nvidia-smi` may not report classic VRAM usage).
- **ML implications:** Use **PyTorch builds for Linux aarch64 + CUDA**, not generic x86 wheels; prefer **recent** PyTorch for GB10 support. Avoid assuming x86-only CUDA extras (e.g. some optional compiled extensions).
