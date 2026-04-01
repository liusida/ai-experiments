# Guideline: experiment `.py` files for Stonesoup

For Python under this repo that you **watch and run cell-by-cell** in Stonesoup (UI and backend: [`stonesoup/README.md`](stonesoup/README.md)).

Below, **each heading is one feature area**—details for that feature appear **only** under that heading so nothing contradicts elsewhere.

---

## Cell markers and titles

- Use **VS Code / Spyder-style** cell headers on their **own line**: `# %%` or `#%%`, then a **short, distinct title** on the **same** line (e.g. `# %% Imports & paths`). **Do not reuse the exact same `# %% …` line** for two different cells—duplicate openers are ambiguous after you insert, delete, or reorder cells.
- A block **after** a marker until the next marker (or EOF) is one **cell**. Content **before** the first `# %%` becomes a single **implicit** head cell (no title in the UI)—**avoid** that by making the **very first line of the file** the `# %% …` line (with a title). **Do not** put a module docstring, `from __future__`, imports, blank “preamble” code, or anything else **above** it.
- **After** each `# %%` line, that cell’s body is normal Python. If you use `from __future__ import annotations`, place it at the **start of the earliest cell body** (it must stay near the top of the file per Python).
- Prefer **many small cells** over one huge cell so you can re-run only what changed.

---

## Kernel behavior and how to structure cells

- **One persistent kernel per watched `.py`:** globals are shared across all cells in that file. Switching **Watch** to another script uses a **different** namespace.
- **Order matters** for whatever you leave in `globals` from earlier cells—unless the user **Reset**s the backend (or a cached kernel is evicted when the LRU cache is full).
- Make cells **re-runnable** where it helps: put paths, flags, model/repo ids, and other knobs **in the cell that consumes them**.
- **Shared** (non-import) paths, constants, and helpers used in **multiple** cells belong in the **first** cell that establishes them—or duplicate them per cell if you want each cell fully standalone.
- **Long loops you might want to stop:** call **`stonesoup.check_abort()`** inside the loop (or every *N* steps). The toolbar **Abort** button cooperatively requests cancel; the call then raises **`stonesoup.RunAborted`**. It does not cut through a long GPU/native call until control returns to Python.

---

## Per-cell input field

- Append **`# stonesoup:cell-input`** to the `# %%` line (after the title). The UI adds a text box next to **Run**; Stonesoup sets **`CELL_INPUT`** to that string (pipelines too). Example: `# %% Try a word # stonesoup:cell-input`.
- For script runs outside Stonesoup, use e.g. `globals().get("CELL_INPUT", "")`.

---

## Rich stdout

- **Default:** escaped plain text. For **HTML** or **Markdown**, the **first** line of stdout must be `# stonesoup:render=html` or `# stonesoup:render=md` / `markdown`; then print the body. **No** other stdout before that line (`text` / `auto` = plain, same as skipping the hint).
- The UI hides the hint in the shown/copied body and offers a chip to flip rich vs plain. Helpers: `stonesoup.STONESOUP_RENDER_HTML`, `STONESOUP_RENDER_MD`, or `stonesoup_render_prefix("html")` (include the newline).
- If the HTML includes **bitmaps** (`<img>`, CSS `background-image`, etc.), prefer **URLs under `/outputs/…`** (files saved next to `stonesoup.show()` output) rather than **`data:` / base64** blobs—see [Paths and writing outputs](#paths-and-writing-outputs).
- Cell HTML is **sanitized** (e.g. DOMPurify): rely on **`style` attributes** for layout you need preserved; embedded **`<style>`…`</style>`** blocks are usually **removed or emptied**, so class-only CSS will not apply.

---

## Paths and writing outputs

- **`stonesoup.repo_root()`** — repo root (`STONESOUP_ROOT` if set, else editable-install layout). Pair with **`stonesoup.data_dir()`** for `data/` (created automatically).
- **`stonesoup.plot_dir()`** — save figures you want in the UI: same tree as **`stonesoup.show()`** (`outputs/stonesoup/<repo-relative script path>/`, served as **`/outputs/…`**); created automatically.
- **`stonesoup.script_dir()`** — folder containing the watched / running `.py` (e.g. stuff you keep next to the script, not under `outputs/`).
- **HTML with images:** save PNGs (or other static assets) under **`stonesoup.plot_dir()`** (or anywhere under that `outputs/stonesoup/…` tree), build **`src`** as **`"/" + path.relative_to(stonesoup.repo_root()).as_posix()`** (same pattern as `stonesoup.show()`). Optional query (e.g. `?cb=mtime`) avoids stale cache after overwrite. **Do not** embed large **`data:image/...;base64,...`** strings in printed HTML—view-source stays readable, payloads stay smaller, and the browser can cache files like normal HTTP assets.

---

## Hugging Face models

**Goal:** One in-kernel load per checkpoint, then reuse — avoid a second ad-hoc `from_pretrained` for the same weights.

**Prereq:** `load_model` needs **`transformers`** (repo optional **`models`** extra) and **PyTorch** — same environment setup as the rest of this repo; see **[`AGENTS.md`](AGENTS.md)** and **`pyproject.toml`**.

1. **Load weights** (pick one or combine; same kernel, no duplicate object for the same Hub id):

   * **Toolbar:** **Watch** the script → repo id → **Load**; **Unload** / **All** to free. UI details: [`stonesoup/README.md`](stonesoup/README.md).
   * **Cell:** `stonesoup.load_model("Org/ModelRepo")` when the string contains **`/`**. If not already loaded, this uses the **same** load path as **Load**; if loaded, returns the **existing** `model` and `processor` (or **tokenizer** for text-only causal LMs).

2. **Example:**

   ```python
   import stonesoup

   model, processor = stonesoup.load_model("Qwen/Qwen3-VL-8B-Instruct")
   ```

---

## Watch path and live reload in the GUI

- Pick the file with **folder** + **file** dropdowns (defaults under `experiments/`); the UI stores a repo-relative path (`?path=` sets the initial file).
- On save, cells **reload** over the WebSocket; **cell outputs are kept** when possible for the same watched file. The UI can show that a cell’s **source changed on disk** until you run it again.

---

## Optional: `# %%` for `uv run` only

You may use `# %%` **only** to mark sections in a script meant to run end-to-end with `uv run python …`. If you also open it in Stonesoup, give **every** section line a **distinct** `# %%` opener (same rules as [Cell markers and titles](#cell-markers-and-titles)).

---

## For automation agents (e.g. Cursor)

- **Do not** execute the experiment file end-to-end (`uv run python …`, full pipeline) **unless** the user explicitly asks—loads can be huge and slow.
- **Do** use **`py_compile`** or editor/linter for a quick syntax check.

