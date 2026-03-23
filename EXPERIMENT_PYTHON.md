# Guideline: experiment `.py` files for Stonesoup

Use this when adding or editing scripts under the repo that you want to **watch and run cell-by-cell** in the Stonesoup GUI (see [`stonesoup/README.md`](stonesoup/README.md)).

## Agents: do not run the file to “verify”

After editing or creating an experiment `.py`, **do not execute it** (no `uv run python …`, no full pipeline) unless the user explicitly asks. These scripts often **download models, load huge datasets, or train for a long time**; verification belongs in the **Stonesoup GUI** (Watch → run cells when the user chooses). It is enough to **`py_compile`** or rely on the editor/linter when a quick syntax check is needed.

## Cell markers (required for Stonesoup)

- Split the file with **VS Code / Spyder-style** cell headers on their **own line**:
  - `# %%` or `#%%`
  - **Always put a short, unique title on the same line** after the marker, e.g. `# %% Imports & helpers`. Stonesoup hashes the **entire** marker line to build a stable `marker_key` for “Updated” tracking and for matching cells across edits—bare `# %%` with no title (or duplicated titles) makes identity weaker and pairing ambiguous.
- The parser treats each block **after** a marker until the next marker (or EOF) as one cell. A file may start with code **before** the first `# %%`; that leading block is one cell with an **implicit** marker key (not tied to a `# %%` line—avoid relying on it for stable identity; prefer starting the file with `# %% …` as the first line).
- Prefer **many small cells** over one huge cell so you can re-run only what changed.

## Layout and docstring (recommended)

1. **First cell** is often `# %% Imports & helpers` (or `Imports & paths`) with:
   - Module docstring describing what the experiment does.
   - **How to run outside Stonesoup:** `uv run python <repo-relative-path>.py`
   - **Stonesoup:** state that the user should **Watch** this file and run cells; note if each `# %%` is intended to run **standalone** (define knobs in that cell) or depends on prior cells.
2. **`from __future__ import annotations`** if you use modern typing.
3. **Paths:** avoid hard-coded absolute paths. Typical pattern:
   - Repo root: `REPO_ROOT = Path(__file__).resolve().parent.parent` when the script lives in `something/2026-…/` or similar.
   - Experiment-local outputs: `Path(__file__).resolve().parent / "plots"` or `"reports"`.
   - Shared repo data: e.g. `REPO_ROOT / "data" / "embedding-layers"` (see [`2026-03-23-Embedding/demo.py`](2026-03-23-Embedding/demo.py)).

## Runnable cells

- **Kernel is shared** across cells for one watch session: globals persist. Order matters unless you **Reset kernel** in the UI.
- Design each cell to be **re-runnable** where practical: set `MODEL_NAME`, paths, and flags **inside** the cell that needs them (pattern in `demo.py`).
- Heavy setup (imports, helpers, constants) belongs in the first cell; analysis/plot cells follow.

## Outputs

- **Figures:** write under an experiment folder, e.g. `…/plots/`, and `mkdir(parents=True, exist_ok=True)` as in `demo.py`.
- **Reports / logs:** e.g. `…/reports/` (see [`2026-03-23-Embedding/embedding_least_norm_tokens.py`](2026-03-23-Embedding/embedding_least_norm_tokens.py)).
- Optional **tqdm** for long loops when running in a terminal; in Stonesoup, stdout still streams to the cell output.

## Watch path in the GUI

- Enter a path **under the repo root** (or `STONESOUP_ROOT`), e.g. `2026-03-23-Embedding/demo.py`.
- After saving the file on disk, cells **reload** over the WebSocket; **outputs are kept** when possible (same watched path, fingerprint-based “updated” markers).

## Optional: “script-shaped” experiments

- A file can still use `# %%` **only** to group logical sections while being run end-to-end with `uv run python …` (e.g. `embedding_least_norm_tokens.py`). Use **unique `# %%` titles** on every section line if you open it in Stonesoup.

## Stable cell identity (when the file changes)

- `marker_key` is derived from the **raw `# %% …` line** (see [`stonesoup/backend/kernel.py`](stonesoup/backend/kernel.py) `fingerprint_marker_line`). Same line text → same key across reloads; change the line or the code body to drive “Updated” / diff behavior. **Do not reuse the exact same `# %%` title line for two different cells**—titles must be unique so each cell has a distinct key (duplicate marker lines are paired in file order and are easy to mis-associate when inserting cells).

## Quick checklist

- [ ] **Do not run** the script end-to-end to verify; leave execution to the user in Stonesoup (or when they ask).
- [ ] Every cell starts with `# %%` / `#%%` **and a unique title on that line** (except any leading pre-marker block—avoid when possible).
- [ ] Docstring mentions `uv run` path and Stonesoup **Watch** workflow.
- [ ] Paths derived from `Path(__file__)` (and `parent.parent` for repo root when nested).
- [ ] Outputs go to experiment `plots/` / `reports/` or documented shared `data/`.
- [ ] Large downloads or models: note HF auth / `trust_remote_code` / memory in the docstring or comments.
