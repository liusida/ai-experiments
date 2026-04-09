"""Stable filename stems for Hugging Face Hub repo ids (plots, caches, basenames)."""

from __future__ import annotations


def hf_repo_id_safe_stem(repo_id: str) -> str:
    """Return a filesystem-friendly stem from ``repo_id``.

    Replaces ``/`` with ``__`` and ``:`` with ``-`` (matches common experiment conventions).
    """
    return repo_id.strip().replace("/", "__").replace(":", "-")
