# Guideline: experiment `.py` files for Stonesoup

Use this when adding or editing scripts under the repo that you want to **watch and run cell-by-cell** in the Stonesoup GUI (see [`stonesoup/README.md`](stonesoup/README.md)).

## Agents: do not run the file to “verify”

After editing or creating an experiment `.py`, **do not execute it** (no `uv run python …`, no full pipeline) unless the user explicitly asks. These scripts often **download models, load huge datasets, or train for a long time**; verification belongs in the **Stonesoup GUI** (Watch → run cells when the user chooses). It is enough to **`py_compile`** or rely on the editor/linter when a quick syntax check is needed.

## Cell markers (required for Stonesoup)

- Split the file with **VS Code / Spyder-style** cell headers on their **own line**:
  - `# %%` or `#%%`
  - **Always put a short, unique title on the same line** after the marker, e.g. `# %% Imports & helpers`. Stonesoup hashes the **entire** marker line to build a stable `marker_key` for “Updated” tracking and for matching cells across edits—bare `# %%` with no title (or duplicated titles) makes identity weaker and pairing ambiguous.
- The parser treats each block **after** a marker until the next marker (or EOF) as one cell. A file may start with code **before** the first `# %%`; that leading block is one cell with an **implicit** marker key (no title in the UI—avoid this).
- **Start the file with a named first cell:** put `# %% Imports & paths` (or similar) on **line 1**. You can place the **module docstring** and **`from __future__ import annotations`** immediately after that marker, still inside the same cell. Comments do not count as Python statements, so the docstring remains the module’s `__doc__`; `from __future__` is still valid after the docstring per the usual rules. Otherwise docstring + future import sit in an unnamed leading cell in Stonesoup.
- Prefer **many small cells** over one huge cell so you can re-run only what changed.

## Layout and docstring (recommended)

1. **First cell** begins with `# %% Imports & helpers` (or `Imports & paths`) on the **first line of the file**, then:
   - Module docstring describing what the experiment does.
   - **How to run outside Stonesoup:** `uv run python <repo-relative-path>.py`
   - **Stonesoup:** state that the user should **Watch** this file and run cells; note if each `# %%` is intended to run **standalone** (define knobs in that cell) or depends on prior cells.
2. **`from __future__ import annotations`** if you use modern typing.
3. **Paths:** avoid hard-coded absolute paths. Typical pattern:
   - Repo root: `REPO_ROOT = Path(__file__).resolve().parent.parent.parent` when the script lives under `experiments/2026-…/` (two levels below repo root).
   - Experiment-local outputs: `Path(__file__).resolve().parent / "plots"` or `"reports"`.
   - Shared repo data: e.g. `REPO_ROOT / "data" / "embedding-layers"` (see [`experiments/2026-03-23-Embedding/demo.py`](experiments/2026-03-23-Embedding/demo.py)).

## Runnable cells

- **Kernel is shared** across cells for one watch session: globals persist. Order matters unless you **Reset kernel** in the UI.
- **Optional per-cell input:** end the ``# %%`` line with ``# stonesoup:cell-input`` (after the title). Only those cells get a text field next to **Run**; its value is injected as the string global **`CELL_INPUT`** (pipelines too). Example marker: `# %% Try a word # stonesoup:cell-input`. In code: `print(CELL_INPUT)`.
- Design each cell to be **re-runnable** where practical: set `MODEL_NAME`, paths, and flags **inside** the cell that needs them (pattern in `demo.py`).
- **Prefer standalone cells:** when you can, make a cell self-contained so re-running it after a **Reset kernel** or out of order does not break. If a variable or helper is **only used in one cell**, define it **inside that cell** instead of at the top of the file or in the imports cell—this keeps the dependency obvious and avoids stale globals from an earlier run.
- Put **shared** setup in the first cell (imports, paths, constants, helpers used in multiple cells); put **cell-local** knobs, small helpers, and one-off formatting in the cells that use them.

## Outputs

- **Figures:** write under an experiment folder, e.g. `…/plots/`, and `mkdir(parents=True, exist_ok=True)` as in `demo.py`.
- **Reports / logs:** e.g. `…/reports/` (see [`experiments/2026-03-23-Embedding/embedding_least_norm_tokens.py`](experiments/2026-03-23-Embedding/embedding_least_norm_tokens.py)).
- Optional **tqdm** for long loops when running in a terminal; in Stonesoup, stdout still streams to the cell output.

### Stdout render hint (optional)

**Default:** stdout is shown as **plain text** (escaped). Stonesoup does **not** infer HTML or Markdown from the payload.

To get **rich** output, print this as the **first line** of stdout (then the payload on following lines):

`# stonesoup:render=html` · `markdown` or `md` · `text` or `auto` (both mean plain text, same as omitting the hint).

The UI **removes** that line from the stored stdout (copy-to-clipboard and display body never include it).

For **HTML** or **Markdown** (hinted) stdout, the output header shows a small **HTML** / **MD** chip; click it to toggle **escaped plain text** and back.

Helpers when `stonesoup` is installed editable: `from stonesoup import STONESOUP_RENDER_HTML` or `stonesoup_render_prefix("html")` — same strings with a trailing newline.

## Watch path in the GUI

- Choose the script with the **folder** and **file** dropdowns (defaults under `experiments/`); the UI keeps the repo-relative path internally (same as e.g. `experiments/2026-03-23-Embedding/demo.py`). Query **`?path=`** still sets the initial file.
- After saving the file on disk, cells **reload** over the WebSocket; **outputs are kept** when possible (same watched path, fingerprint-based “updated” markers).

## Optional: “script-shaped” experiments

- A file can still use `# %%` **only** to group logical sections while being run end-to-end with `uv run python …` (e.g. `embedding_least_norm_tokens.py`). Use **unique `# %%` titles** on every section line if you open it in Stonesoup.

## Stable cell identity (when the file changes)

- `marker_key` is derived from the **raw `# %% …` line** (see [`stonesoup/backend/kernel.py`](stonesoup/backend/kernel.py) `fingerprint_marker_line`). Same line text → same key across reloads; change the line or the code body to drive “Updated” / diff behavior. Adding or removing `# stonesoup:cell-input` **changes** that line and therefore the key. **Do not reuse the exact same `# %%` title line for two different cells**—titles must be unique so each cell has a distinct key (duplicate marker lines are paired in file order and are easy to mis-associate when inserting cells).

## Quick checklist

- [ ] **Do not run** the script end-to-end to verify; leave execution to the user in Stonesoup (or when they ask).
- [ ] **Line 1 is `# %% …`** with a unique title (no leading pre-marker block); docstring and `from __future__` live inside that first cell if used.
- [ ] Every later cell starts with `# %%` / `#%%` **and a unique title on that line**.
- [ ] Docstring mentions `uv run` path and Stonesoup **Watch** workflow.
- [ ] Cells are as **standalone** as practical: single-use variables/helpers live in the cell that uses them; only **shared** setup sits in the first cell.
- [ ] Paths derived from `Path(__file__)` (and `parent.parent.parent` for repo root when under `experiments/2026-…/`).
- [ ] Outputs go to experiment `plots/` / `reports/` or documented shared `data/`.
- [ ] Large downloads or models: note HF auth / `trust_remote_code` / memory in the docstring or comments.
