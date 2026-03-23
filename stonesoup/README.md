# Stonesoup

Floating **cell runner** for a watched Python file: splits on `# %%` / `#%%`, shows cells in the UI, runs them in one **persistent Python kernel** (stdout/stderr per cell). Backend is **FastAPI** + **watchdog**; UI is **Vite**, opened in **any browser** (Firefox, Chromium, etc.).

## Prerequisites

- Repo **`.venv`** with `torch` / `transformers` if you run ML cells (see repo root setup).
- Backend deps: `fastapi`, `uvicorn`, `watchdog` (included if you install with `-e ".[stonesoup]"` at repo root).
- **Node.js** for the Vite frontend (`npm run dev`).

## 1. Backend (terminal A)

From the **repository root** (`ai-experiments/`):

```bash
uv pip install -e ".[stonesoup]"
uv run python -m stonesoup.backend.server
```

(`pip install -e ".[stonesoup]"` from the repo root is equivalent if you use plain pip.)

Listens on **`127.0.0.1:8765`** by default (`STONESOUP_HOST` / `STONESOUP_PORT` override). Only paths **under the repo root** are allowed unless you set **`STONESOUP_ROOT`** to a different absolute root.

## 2. Frontend — browser (dev)

Terminal B:

```bash
cd stonesoup/frontend
npm install
npm run dev
```

Open **http://127.0.0.1:5173/** in Firefox or another browser. Vite proxies `/api` and `/ws` to the backend. Use the **dropdown** to pick another `*.py` in the same folder as the default script (from the path’s parent directory), or type a repo-relative path in the text field, then click **Watch**. Override the folder used for the dropdown with **`?dir=2026-03-23-Embedding`** (repo-relative). Edit the file on disk; cells refresh over **WebSocket** after a short debounce.

Each cell is a **floating panel** inside the page. Drag **empty space** in the cell canvas (not on a cell or the **↻ Loop** strip) to **pan** (scroll). Layout reflows when the watched path or the number of cells changes.

**Pipeline:** the pipeline is a **tree** of steps: **cells** and **loops**. **Drag** a **canvas cell** (or the **↻ Loop** card below the cells) into a **drop slot** in the pipeline bar to insert at the **root** or **inside a loop**. Drag the grip on a **pipeline chip** or **loop** header to **reorder** or move between levels (you cannot drop a loop inside itself). **+ chain** appends the current cell at the **end of the root** list; use drag to place items elsewhere or fill a loop’s body. Example: **+ chain** cell 0 → drag **↻ Loop** onto the bar → drag cells 1–2 into the loop → drag cell 3 after the loop.

Each **loop** shows **↻ Loop** and a **count** (e.g. `3×` = three passes). Click that row to edit the JSON array; **Done** or clicking the row again applies and collapses; the count updates after a successful apply. Write iterations plainly, e.g. **`[1, 2, 3]`** or **`[{"lr": 0.01}, {"lr": 0.1}]`** — that text is stored as-is. At run time, **plain objects** merge their keys into the kernel; **anything else** (number, string, list, …) is exposed as **`LOOP_ITEM`** (and **`LOOP_INDEX`**) in cells. Nested loops **merge** outer and inner inject dicts (inner wins on key clashes). **Run pipeline** walks the tree depth-first (stops on first failure). The program is stored in **localStorage** under `stonesoup-pipeline-v2:` (older flat `stonesoup-pipeline:` lists are still loaded once and treated as a linear list of cells).

Optional: **`npm run build`** writes a static site to **`frontend/dist/`** (handy if you later serve it from another static host; the UI still expects the API at **127.0.0.1:8765** when not using the Vite dev proxy).

## API (local)

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/py-files?dir=rel/subfolder` | List `*.py` files **directly** in that repo-relative directory (non-recursive); UI script picker |
| POST | `/api/watch` | `{"path": "relative/or/abs/under/root.py"}` — start watcher, parse cells |
| GET | `/api/cells` | Current cells + revision |
| POST | `/api/run` | `{"cell_index": n, "inject": {...}?}` — optional `inject` merges into kernel globals before the cell |
| POST | `/api/reset` | Clear kernel globals |
| WS | `/ws` | Push `{type:"cells", revision, cells}` on file change |

## Notes

- **Kernel vs file:** Saving the file updates the listed source; it does **not** re-run cells automatically.
- Long runs (e.g. `generate()`) may block until the request finishes; increase client timeout if needed.
