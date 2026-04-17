# Guideline: experiment `.py` files for Stonesoup

For Python under this repo that you **watch and run cell-by-cell** in Stonesoup (UI and backend: [`stonesoup/README.md`](stonesoup/README.md)).

Below, **each heading is one feature area**—details for that feature appear **only** under that heading so nothing contradicts elsewhere.

---

## Cell markers and titles

- Use **VS Code / Spyder-style** cell headers on their **own line**: `# %%` or `#%%`, then a **short, distinct title** on the **same** line (e.g. `# %% Imports & paths`). **Do not reuse the exact same `# %% …` line** for two different cells—duplicate openers are ambiguous after you insert, delete, or reorder cells.
- A block **after** a marker until the next marker (or EOF) is one **cell**. Content **before** the first `# %%` becomes a single **implicit** head cell (no title in the UI)—**avoid** that by making the **very first line of the file** the `# %% …` line (with a title). **Do not** put a module docstring, `from __future__`, imports, blank “preamble” code, or anything else **above** it.
- **After** each `# %%` line, that cell’s body is normal Python. If you use `from __future__ import annotations`, place it at the **start of the earliest cell body** (it must stay near the top of the file per Python).
- Prefer **many small cells** over one huge cell so you can re-run only what changed.
- Prefer **readable variable names** in cell code (clear words over cryptic or gratuitous leading underscores).

---

## Kernel behavior and how to structure cells

- **One persistent kernel per watched `.py`:** globals are shared across all cells in that file. Switching **Watch** to another script uses a **different** namespace.
- **Order matters** for whatever you leave in `globals` from earlier cells—unless the user **Reset**s the backend (or a cached kernel is evicted when the LRU cache is full).
- Make cells **re-runnable** where it helps: put paths, flags, model/repo ids, and other knobs **in the cell that consumes them**.
- **Shared** (non-import) paths, constants, and helpers used in **multiple** cells belong in the **first** cell that establishes them—or duplicate them per cell if you want each cell fully standalone.
- **Long loops you might want to stop:** call **`stonesoup.check_abort()`** inside the loop (or every *N* steps). The toolbar **Abort** button cooperatively requests cancel; the call then raises **`stonesoup.RunAborted`**. It does not cut through a long GPU/native call until control returns to Python. Use `tqdm` if needed.

---

## Per-cell input field

- Append **`# stonesoup:cell-input`** to the `# %%` line (after the title). The UI adds a text box next to **Run**; Stonesoup sets **`CELL_INPUT`** to that string (pipelines too). Example: `# %% Try a word # stonesoup:cell-input`.
- For script runs outside Stonesoup, use e.g. `globals().get("CELL_INPUT", "")`.

---

## Rich stdout

- **Default:** escaped plain text. For **HTML** or **Markdown**, the **first** line of stdout must be `# stonesoup:render=html` or `# stonesoup:render=md` / `markdown`; then print the body. **No** other stdout before that line (`text` / `auto` = plain, same as skipping the hint).
- **Prints before plots:** If you `print()` status lines and then call `stonesoup.show()`, put **`print(stonesoup.STONESOUP_RENDER_HTML, end="")` as the first statement in the cell** so the combined stdout still starts with the hint. **`stonesoup.show()`** and **`stonesoup.display()`** only insert the hint **once per cell run** by default (later calls in the same cell omit it). If you printed the hint yourself first, call **`stonesoup.mark_render_hint_emitted()`** so helpers do not print it again. You can still use **`emit_render_hint=False`** on any call to force no hint.
- **Pandas / rich HTML tables:** **`stonesoup.display(obj)`** prints objects with a non-empty **`_repr_html_()`** (including **pandas** `DataFrame`, `Series`, and **`Styler`**) as HTML in the cell output. The body is wrapped in **`<div class="stonesoup-rich-html">…</div>`**; **`DataFrame`** / **`Series`** use **`to_html(max_rows=…, max_cols=…)`** with defaults **30** and **20** (override with **`display(..., max_rows=50)`** etc.). Unsupported values fall back to **`repr(obj)`** with no render hint (plain text). Multiple **`display()`** / **`show()`** calls in one cell share a single hint line automatically; if you **`print()`** other text before the first rich output, the first stdout line must still be the hint—**`print(stonesoup.STONESOUP_RENDER_HTML, end="")` first**, or **`print()`** your lines first and then **`stonesoup.mark_render_hint_emitted()`** after your manual hint line if you use one.
- The UI hides the hint in the shown/copied body and offers a chip to flip rich vs plain. Helpers: `stonesoup.STONESOUP_RENDER_HTML`, `STONESOUP_RENDER_MD`, or `stonesoup_render_prefix("html")` (include the newline). **`stonesoup.emit_html_output_hint()`** prints that HTML hint line and records it so `display()` / `show()` do not repeat it—use as the first stdout line of the cell when you hand-build HTML.
- If the HTML includes **bitmaps** (`<img>`, CSS `background-image`, etc.), prefer **URLs under `/outputs/…`** (files saved next to `stonesoup.show()` output) rather than **`data:` / base64** blobs—see [Paths and writing outputs](#paths-and-writing-outputs).
- Cell HTML is **sanitized** (e.g. DOMPurify): rely on **`style` attributes** for layout you need preserved; embedded **`<style>`…`</style>`** blocks are usually **removed or emptied**, so class-only CSS will not apply.

---

## Paths and writing outputs

- **`stonesoup.repo_root()`** — repo root (`STONESOUP_ROOT` if set, else editable-install layout). Pair with **`stonesoup.data_dir()`** for `data/` (created automatically).
- **`stonesoup.outputs_dir()`** — per-script directory under **`outputs/`** (HTTP **`/outputs/…`**): the watched file’s path relative to the repo with a leading **`experiments/`** removed if present, **`.py`** dropped, remaining directories kept (e.g. `experiments/2026-04-06-Foo/bar.py` → `outputs/2026-04-06-Foo/bar/`). Use for figures, caches, or any cell artifact you want web-addressable. **Same directory as `stonesoup.show()`** saves PNGs to. **`stonesoup.plot_dir()`** is a synonym (historical name). Created automatically.
- **`stonesoup.script_dir()`** — folder containing the watched / running `.py` (e.g. stuff you keep next to the script, not under `outputs/`).
- **HTML with images:** save PNGs (or other static assets) under **`stonesoup.outputs_dir()`** (or anywhere under the repo’s **`outputs/…`** tree), build **`src`** as **`"/" + path.relative_to(stonesoup.repo_root()).as_posix()`** (same pattern as `stonesoup.show()`), or use **`stonesoup.experiment.output_url_path(path)`** for the same path plus an optional cache-busting `?t=` query. **Do not** embed large **`data:image/...;base64,...`** strings in printed HTML—view-source stays readable, payloads stay smaller, and the browser can cache files like normal HTTP assets.

---

## Shared helpers for new code (`stonesoup.experiment`)

The **expanded** helpers in this section (safe Hub stems, text encoding, decoder/hook capture, `output_url_path`, optional unload, render re-exports, etc.) are the recommended way to write **new** experiment code starting **2026-04-09**. Scripts written before that date may still inline equivalent logic; nothing requires migrating them.

**Learn the new API:** step through [`experiments/Demo/stonesoup-experiment-api-demo.py`](experiments/Demo/stonesoup-experiment-api-demo.py) in Stonesoup (paths, `load_model`, both capture modes, `show`, hand-built HTML, `check_abort`).

For **new** work, import shared utilities from **`stonesoup.experiment`** (same symbols are lazy-exported where noted):

| Concern | Symbols / module |
|--------|-------------------|
| Hub id → safe filename stem | `hf_repo_id_safe_stem` (`names`) |
| Tokenizer / batching | `inner_tokenizer`, `encode_text_inputs`, `ensure_pad_token_via_eos` (`hf_inputs`) |
| Decoder layer list | `decoder_blocks` (`lm_stack`) |
| Hidden states via hooks | `capture_embed_and_post_blocks`, `capture_pre_block0_and_post_blocks` (`hidden_hooks`) |
| Matplotlib / URLs / tables | `show`, `display`, `configure_matplotlib_agg`, `output_url_path` (`display`) |
| Unload bindings by repo id | `unload_loaded_names_for_repo` (`models`) |
| Rich stdout hints | `STONESOUP_RENDER_HTML`, …, `stonesoup_render_prefix` (same strings as `stonesoup`) |

---

## Hugging Face models

**Goal:** One **process-wide** copy of each checkpoint (shared pool), with **per-script** Python names bound in each watched file’s kernel — no second ad-hoc `from_pretrained` for the same Hub id and options.

**Prereq:** `load_model` needs **`transformers`** (repo optional **`models`** extra) and **PyTorch** — same environment setup as the rest of this repo; see **[`AGENTS.md`](AGENTS.md)** and **`pyproject.toml`**.

1. **Load / bind** (toolbar and cells share the same pool):

   * **Toolbar:** **Watch** a script → repo id → **Load**. The dropdown lists **all** checkpoints in memory for this Stonesoup server (not only the current file). **Unload** drops the selected checkpoint from **every** cached experiment kernel; **All** clears all UI-managed models the same way (memory frees when refcounts hit zero). See [`stonesoup/README.md`](stonesoup/README.md).
   * **Cell:** `stonesoup.load_model("Org/ModelRepo")` when the string contains **`/`**. If the pool already has those weights, you get the **same** objects and this kernel gains a binding; otherwise the **same** load path as **Load** runs once, then returns `(model, processor)` (or tokenizer for text-only causal LMs).

2. **Introspection:** `stonesoup.list_loaded_models()` lists bindings in **this** file’s kernel; `stonesoup.list_loaded_models_globally()` lists every resident checkpoint (same shapes as the UI/API).

3. **Example:**

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

