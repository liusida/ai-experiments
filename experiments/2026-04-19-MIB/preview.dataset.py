# %% Imports, constants & helpers
from __future__ import annotations

import json
from typing import Any

import pandas as pd
import stonesoup
from datasets import get_dataset_config_names, load_dataset, load_dataset_builder
from tqdm.auto import tqdm

# Mechanistic Interpretability Benchmark (HF collection mib-bench/*).
MIB_REPOS: list[str] = [
    "mib-bench/ioi",
    "mib-bench/copycolors_mcqa",
    "mib-bench/arithmetic_addition",
    "mib-bench/arithmetic_subtraction",
    "mib-bench/arc_easy",
    "mib-bench/arc_challenge",
    "mib-bench/ravel",
]

# ``copycolors_mcqa`` has multiple configs (2–10 answer choices). Pick one for the streamed example cell.
COPYCOLORS_CONFIG_FOR_PEEK = "4_answer_choices"

MAX_STR = 600
MAX_LIST_ITEMS = 8


def _shorten(value: Any, *, depth: int = 0) -> Any:
    if depth > 6:
        return "…"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        if len(value) > MAX_STR:
            return value[:MAX_STR] + f"… (+{len(value) - MAX_STR} chars)"
        return value
    if isinstance(value, list):
        out = [_shorten(v, depth=depth + 1) for v in value[:MAX_LIST_ITEMS]]
        if len(value) > MAX_LIST_ITEMS:
            out.append(f"… (+{len(value) - MAX_LIST_ITEMS} items)")
        return out
    if isinstance(value, dict):
        return {k: _shorten(v, depth=depth + 1) for k, v in value.items()}
    return repr(value)


def _split_sizes_row(
    repo_id: str,
    config_name: str,
    builder: Any,
) -> dict[str, Any]:
    splits = getattr(builder.info, "splits", None) or {}
    row: dict[str, Any] = {"repo_id": repo_id, "config": config_name}
    for key in ("train", "validation", "test"):
        sp = splits.get(key)
        row[f"n_{key}"] = int(sp.num_examples) if sp is not None else None
    row["features"] = ", ".join((builder.info.features or {}).keys())
    return row


def peek_configs_for_repo(repo_id: str) -> list[str]:
    """Config names to use for one streamed train row (copycolors → ``COPYCOLORS_CONFIG_FOR_PEEK`` only)."""
    configs = get_dataset_config_names(repo_id)
    if repo_id == "mib-bench/copycolors_mcqa":
        if COPYCOLORS_CONFIG_FOR_PEEK not in configs:
            raise ValueError(
                f"COPYCOLORS_CONFIG_FOR_PEEK={COPYCOLORS_CONFIG_FOR_PEEK!r} not in {configs}"
            )
        return [COPYCOLORS_CONFIG_FOR_PEEK]
    return [configs[0]] if len(configs) == 1 else configs


def format_mib_row_for_humans(repo_id: str, row: dict[str, Any]) -> str:
    """Plain-language quiz text for one dataset row (first train example in the cells below)."""
    lines: list[str] = []

    if repo_id == "mib-bench/ioi":
        lines.append("Who is the indirect object? (Who received the object?)")
        lines.append("")
        lines.append(f"Sentence: {row['prompt']}")
        lines.append("")
        lines.append("Choices:")
        choices = row["choices"]
        for i, text in enumerate(choices):
            letter = chr(ord("A") + i)
            lines.append(f"  {letter}. {text}")
        ak = int(row["answerKey"])
        lines.append("")
        lines.append(f"Correct: {chr(ord('A') + ak)} — {choices[ak]}")

    elif repo_id == "mib-bench/copycolors_mcqa":
        ch = row["choices"]
        labels = ch["label"]
        texts = ch["text"]
        lines.append(row["question"])
        lines.append("")
        for lb, tx in zip(labels, texts, strict=True):
            lines.append(f"  {lb}. {tx}")
        ak = int(row["answerKey"])
        lines.append("")
        lines.append(f"Correct: {labels[ak]}. {texts[ak]}")

    elif repo_id in ("mib-bench/arithmetic_addition", "mib-bench/arithmetic_subtraction"):
        lines.append("Solve the arithmetic problem (answer with the number only).")
        lines.append("")
        lines.append(f"  {row['prompt'].strip()}")
        lines.append("")
        lines.append(f"Answer: {row['label']}")

    elif repo_id in ("mib-bench/arc_easy", "mib-bench/arc_challenge"):
        lines.append(row["question"])
        lines.append("")
        lbls = row["choices"]["label"]
        texts = row["choices"]["text"]
        for lb, tx in zip(lbls, texts, strict=True):
            lines.append(f"  {lb}. {tx}")
        lines.append("")
        lines.append(
            f"Correct: {row['label']} (choice index {row['answerKey']})"
        )

    elif repo_id == "mib-bench/ravel":
        lines.append(
            f"Structured knowledge — fill in or complete using attribute “{row['attribute']}” "
            f"for entity “{row['entity']}”."
        )
        lines.append("")
        lines.append("Model input (prompt):")
        lines.append(f"  {row['prompt']}")
        lines.append("")
        lines.append("Reference metadata (gold attributes for this entity):")
        for key in ("Continent", "Country", "Language"):
            if key in row:
                lines.append(f"  {key}: {row[key]}")

    else:
        lines.append(f"(No human formatter for {repo_id}; use raw JSON cell.)")
        lines.append(json.dumps(_shorten(row), indent=2, ensure_ascii=False))

    return "\n".join(lines)


# %% Split sizes & feature names (builders only; no full download)
rows: list[dict[str, Any]] = []
for repo_id in tqdm(MIB_REPOS, desc="MIB: load_dataset_builder"):
    stonesoup.check_abort()
    cfgs = get_dataset_config_names(repo_id)
    short_name = repo_id.rsplit("/", 1)[-1]
    cfg_iter = (
        tqdm(cfgs, desc=short_name, leave=False) if len(cfgs) > 1 else cfgs
    )
    for cfg in cfg_iter:
        stonesoup.check_abort()
        b = load_dataset_builder(repo_id, cfg)
        rows.append(_split_sizes_row(repo_id, cfg, b))

overview = pd.DataFrame(rows)
stonesoup.display(overview)

# %% Stream one train row per repo (copycolors uses COPYCOLORS_CONFIG_FOR_PEEK)
for repo_id in MIB_REPOS:
    stonesoup.check_abort()
    for cfg in peek_configs_for_repo(repo_id):
        ds = load_dataset(repo_id, cfg, split="train", streaming=True)
        row = next(iter(ds))
        print("=" * 72, flush=True)
        print(f"{repo_id}  config={cfg!r}  split=train  row0", flush=True)
        print(json.dumps(_shorten(row), indent=2, ensure_ascii=False), flush=True)

# %% Human-readable quiz (same first train row per repo)
for repo_id in tqdm(MIB_REPOS, desc="MIB: human-readable"):
    stonesoup.check_abort()
    for cfg in peek_configs_for_repo(repo_id):
        ds = load_dataset(repo_id, cfg, split="train", streaming=True)
        row = next(iter(ds))
        print("=" * 72, flush=True)
        print(f"{repo_id}  config={cfg!r}", flush=True)
        print("", flush=True)
        print(format_mib_row_for_humans(repo_id, row), flush=True)
        print("", flush=True)
