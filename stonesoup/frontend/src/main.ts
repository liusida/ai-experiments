import "./style.css";
import DOMPurify from "dompurify";
import { marked } from "marked";

/** Allow ``stonesoup.show()`` images (served under ``/outputs/``). */
DOMPurify.addHook("uponSanitizeAttribute", (_node, data) => {
  if (!data || data.attrName !== "src") return;
  const v = data.attrValue;
  if (typeof v === "string" && v.startsWith("/outputs/")) {
    data.forceKeepAttr = true;
  }
});

type Cell = {
  index: number;
  title: string;
  source: string;
  marker_key: string;
  start_line?: number;
  cell_input?: boolean;
};

/** Read 1-based cell body start line from API/WS JSON (snake_case or camelCase). */
function readCellStartLineFromPayload(raw: Record<string, unknown>): number | undefined {
  const v = raw.start_line ?? raw.startLine;
  if (v === undefined || v === null) return undefined;
  const n = Number(v);
  if (!Number.isFinite(n) || n < 1) return undefined;
  return Math.floor(n);
}

/** Normalize parsed cell objects so ``start_line`` is always set when the server sent it. */
function cellFromApiPayload(raw: unknown): Cell {
  if (!raw || typeof raw !== "object") {
    return { index: 0, title: "", source: "", marker_key: "" };
  }
  const o = raw as Record<string, unknown>;
  const sl = readCellStartLineFromPayload(o);
  const c: Cell = {
    index: Number(o.index) || 0,
    title: String(o.title ?? ""),
    source: String(o.source ?? ""),
    marker_key: String(o.marker_key ?? ""),
    cell_input: Boolean(o.cell_input),
  };
  if (sl !== undefined) c.start_line = sl;
  return c;
}

const EDITOR_PREF_KEY = "stonesoup_editor";
type EditorPref = "cursor" | "vscode";

function getEditorPref(): EditorPref {
  return (localStorage.getItem(EDITOR_PREF_KEY) as EditorPref | null) ?? "cursor";
}
function setEditorPref(v: EditorPref) {
  localStorage.setItem(EDITOR_PREF_KEY, v);
}

/** Deeplink to open a file at 1-based line:col in the chosen editor. ``absolutePath`` must be a real filesystem path. */
function editorFileUrl(absolutePath: string, line: number, col: number = 1): string {
  let p = absolutePath.trim().replace(/^file:\/\//i, "");
  p = p.replace(/\\/g, "/");
  const scheme = getEditorPref() === "vscode" ? "vscode" : "cursor";
  if (/^[A-Za-z]:\//.test(p)) {
    return `${scheme}://file/${p}:${line}:${col}`;
  }
  if (!p.startsWith("/")) p = `/${p}`;
  return `${scheme}://file${p}:${line}:${col}`;
}

/** Repo-relative script path (as in the toolbar) → absolute path for editor URIs. */
function absoluteRepoPathForEditor(repoRelative: string): string {
  const t = repoRelative.trim().replace(/\\/g, "/");
  if (!t) return "";
  if (/^[A-Za-z]:\//.test(t)) return t;
  if (repoRootAbs) {
    const root = repoRootAbs.replace(/\/$/, "");
    const rest = t.replace(/^\/+/, "");
    return `${root}/${rest}`;
  }
  if (!t.startsWith("/")) return `/${t}`;
  return t;
}

function cellEditorHref(pathForLink: string | null | undefined, startLine: number): string {
  const t = pathForLink?.trim() ?? "";
  if (!t) return "";
  const line = Number.isFinite(startLine) && startLine >= 1 ? Math.floor(Number(startLine)) : 1;
  return editorFileUrl(absoluteRepoPathForEditor(t), line);
}

function escapeHtmlAttr(s: string): string {
  return s.replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;");
}

/** Editor brand icons for the deeplink toggle button (simple-icons paths, viewBox 0 0 24 24, currentColor). */
const EDITOR_ICON_CURSOR = `<svg class="editor-icon editor-icon--cursor" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="15" height="15" aria-hidden="true"><path fill="currentColor" fill-rule="evenodd" d="M11.503.131 1.891 5.678a.84.84 0 0 0-.42.726v11.188c0 .3.162.575.42.724l9.609 5.55a1 1 0 0 0 .998 0l9.61-5.55a.84.84 0 0 0 .42-.724V6.404a.84.84 0 0 0-.42-.726L12.497.131a1.01 1.01 0 0 0-.996 0M2.657 6.338h18.55c.263 0 .43.287.297.515L12.23 22.918c-.062.107-.229.064-.229-.06V12.335a.59.59 0 0 0-.295-.51l-9.11-5.257c-.109-.063-.064-.23.061-.23"/></svg>`;
const EDITOR_ICON_VSCODE = `<svg class="editor-icon editor-icon--vscode" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="15" height="15" aria-hidden="true"><path fill="currentColor" d="M23.15 2.587L18.21.21a1.494 1.494 0 0 0-1.705.29l-9.46 8.63-4.12-3.128a.999.999 0 0 0-1.276.057L.327 7.261A1 1 0 0 0 .326 8.74L3.899 12 .326 15.26a1 1 0 0 0 .001 1.479L1.65 17.94a.999.999 0 0 0 1.276.057l4.12-3.128 9.46 8.63a1.492 1.492 0 0 0 1.704.29l4.942-2.377A1.5 1.5 0 0 0 24 19.06V4.94A1.5 1.5 0 0 0 23.15 2.587zM17.796 18.3L9.48 12l8.316-6.3z"/></svg>`;

/** Cell toolbar icons (``currentColor``). ``py`` badge for the Cursor deeplink (Python source); run is play inside one “cell” frame. */
const CELL_ICON_PYTHON = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="18" height="18" class="cell-action-icon" aria-hidden="true"><rect x="2.5" y="4" width="19" height="16" rx="3" fill="none" stroke="currentColor" stroke-width="1.35"/><text x="12" y="12" text-anchor="middle" dominant-baseline="central" fill="currentColor" font-size="10" font-weight="800" font-family="ui-monospace, SFMono-Regular, Menlo, Consolas, monospace">py</text></svg>`;
const CELL_ICON_ADD = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="18" height="18" class="cell-action-icon" aria-hidden="true"><path fill="currentColor" d="M19 13h-6v6h-2v-6H5v-2h6V5h2v6h6v2z"/></svg>`;
const CELL_ICON_RUN_CELL = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="18" height="18" class="cell-action-icon" aria-hidden="true"><rect x="5" y="4" width="14" height="16" rx="2" fill="none" stroke="currentColor" stroke-width="1.75"/><path fill="currentColor" d="M10.5 9.5 15.5 12l-5 2.5v-5z"/></svg>`;
/** Expand / maximize output (arrows pointing outward). */
const CELL_ICON_OUTPUT_MAXIMIZE = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="18" height="18" class="cell-action-icon" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><path d="M15 3h6v6"/><path d="M9 21H3v-6"/><path d="M21 3l-7 7"/><path d="M3 21l7-7"/></svg>`;

/** Toolbar watch / server (``currentColor``, 16px). Reset = power (restart server). */
const TOOLBAR_ICON_WATCH = `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M2 12s4.25-7 10-7 10 7 10 7-4.25 7-10 7-10-7-10-7z"/><circle cx="12" cy="12" r="2.5" fill="none" stroke="currentColor"/></svg>`;
const TOOLBAR_ICON_RESET = `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 2v10"/><path d="M18.36 6.64a9 9 0 1 1-12.73 0"/></svg>`;
const TOOLBAR_ICON_ABORT = `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" aria-hidden="true"><rect x="7" y="7" width="10" height="10" rx="2" fill="currentColor"/></svg>`;

/** Toolbar model controls (``currentColor``, 16px; titles set on the buttons). */
const TOOLBAR_ICON_MODEL_LOAD = `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>`;
const TOOLBAR_ICON_MODEL_UNLOAD = `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="5" y="5" width="14" height="14" rx="2"/><path d="m9 9 6 6M15 9l-6 6"/></svg>`;
const TOOLBAR_ICON_MODEL_UNLOAD_ALL = `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="3" y="2" width="11" height="11" rx="1.5" stroke-width="1.65"/><path stroke-width="1.65" d="M5 4l7 7M12 4l-7 7"/><rect x="10" y="11" width="11" height="11" rx="1.5" stroke-width="2"/><path stroke-width="2" d="M12 13l7 7M19 13l-7 7"/></svg>`;
/** RAM-style chip: free memory for the current watched experiment only. */
const TOOLBAR_ICON_FREE_MEMORY = `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="3" y="5" width="18" height="14" rx="2"/><line x1="6" y1="9" x2="18" y2="9"/><line x1="6" y1="12" x2="18" y2="12"/><line x1="6" y1="15" x2="18" y2="15"/><line x1="4" y1="20" x2="20" y2="4" stroke-width="2.25"/></svg>`;

/** Manual resize / saved layout: must leave room below `.cell-head` for output */
const CELL_LAYOUT_MIN_W = 220;
const CELL_LAYOUT_MIN_H = 200;

/** Symmetric empty margin around the zoom strip so panning can continue past content on all sides. */
const CELLS_PAN_GUTTER_PX = 2400;

const apiBase = import.meta.env.DEV ? "" : "http://127.0.0.1:8765";

/** Read ``Response`` as JSON; if the body is HTML/text (e.g. Vite 502 when :8765 is down), throw a clear error. */
async function readApiJson(r: Response): Promise<unknown> {
  const text = await r.text();
  const t = text.trim();
  if (!t) {
    if (!r.ok) throw new Error(`HTTP ${r.status} ${r.statusText} (empty body)`);
    return {};
  }
  try {
    return JSON.parse(text) as unknown;
  } catch {
    const preview = text.length > 280 ? `${text.slice(0, 280)}…` : text;
    const htmlish = preview.trimStart().startsWith("<");
    const hint = htmlish
      ? " Body looks like HTML — is the Stonesoup server running on port 8765? (Dev UI proxies /api there.)"
      : "";
    throw new Error(`HTTP ${r.status}: response is not JSON.${hint}\n${preview}`);
  }
}

/** Absolute repo root from server (POSIX-style slashes); used for Cursor/VS Code file deeplinks. */
let repoRootAbs = "";

function applyRepoRootFromPayload(data: { repo_root?: unknown }) {
  const rr = data.repo_root;
  if (typeof rr === "string" && rr.trim()) {
    repoRootAbs = rr.trim().replace(/\\/g, "/");
  }
}

async function fetchRepoRoot(): Promise<void> {
  try {
    const r = await fetch(`${apiBase}/api/health`);
    if (!r.ok) return;
    const j = (await readApiJson(r)) as { repo_root?: unknown };
    applyRepoRootFromPayload(j);
  } catch {
    /* ignore */
  }
}

void fetchRepoRoot();

function wsUrl(): string {
  if (import.meta.env.DEV) {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    return `${proto}//${location.host}/ws`;
  }
  return "ws://127.0.0.1:8765/ws";
}

const app = document.querySelector<HTMLDivElement>("#app")!;

const urlParams = new URLSearchParams(location.search);

const WATCH_PATH_COOKIE = "stonesoup_watch_path";
/** ~400 days; path is repo-relative, non-sensitive */
const WATCH_PATH_COOKIE_MAX_AGE = 60 * 60 * 24 * 400;

function readWatchPathCookie(): string {
  const prefix = `${WATCH_PATH_COOKIE}=`;
  for (const part of document.cookie.split(";")) {
    const s = part.trim();
    if (s.startsWith(prefix)) {
      const raw = s.slice(prefix.length);
      try {
        return decodeURIComponent(raw);
      } catch {
        return raw;
      }
    }
  }
  return "";
}

function saveWatchPathCookie(path: string) {
  const t = path.trim();
  if (!t) return;
  document.cookie = `${WATCH_PATH_COOKIE}=${encodeURIComponent(t)}; Path=/; Max-Age=${WATCH_PATH_COOKIE_MAX_AGE}; SameSite=Lax`;
}

/** Legacy: layouts lived in a single cookie (~4 KB cap — saves failed silently, so F5 lost positions). */
const CELL_LAYOUTS_LEGACY_COOKIE = "stonesoup_cell_layouts_v1";
/** Per-watched-file cell geometry; keyed by repo-relative path (same shape as the old cookie). */
const CELL_LAYOUTS_LS_KEY = "stonesoup_cell_layouts_v2";
const CELL_LAYOUTS_LEGACY_MIGRATED_KEY = "stonesoup_cell_layouts_legacy_cookie_imported";
type CellLayoutTuple = [number, number, number, number];
type CellLayoutsFileMap = Record<string, CellLayoutTuple>;

function parseLegacyLayoutsCookie(): Record<string, CellLayoutsFileMap> {
  const prefix = `${CELL_LAYOUTS_LEGACY_COOKIE}=`;
  for (const part of document.cookie.split(";")) {
    const s = part.trim();
    if (!s.startsWith(prefix)) continue;
    try {
      return JSON.parse(decodeURIComponent(s.slice(prefix.length))) as Record<string, CellLayoutsFileMap>;
    } catch {
      return {};
    }
  }
  return {};
}

function readCellLayoutsStore(): Record<string, CellLayoutsFileMap> {
  try {
    const raw = localStorage.getItem(CELL_LAYOUTS_LS_KEY);
    if (raw) {
      const x = JSON.parse(raw) as unknown;
      if (x && typeof x === "object" && !Array.isArray(x)) {
        return x as Record<string, CellLayoutsFileMap>;
      }
    }
  } catch {
    /* ignore */
  }
  if (localStorage.getItem(CELL_LAYOUTS_LEGACY_MIGRATED_KEY) !== "1") {
    const fromCookie = parseLegacyLayoutsCookie();
    try {
      if (Object.keys(fromCookie).length > 0) {
        localStorage.setItem(CELL_LAYOUTS_LS_KEY, JSON.stringify(fromCookie));
        document.cookie = `${CELL_LAYOUTS_LEGACY_COOKIE}=; Path=/; Max-Age=0; SameSite=Lax`;
      }
      localStorage.setItem(CELL_LAYOUTS_LEGACY_MIGRATED_KEY, "1");
    } catch {
      /* ignore */
    }
    return fromCookie;
  }
  return {};
}

function writeCellLayoutsStore(all: Record<string, CellLayoutsFileMap>) {
  try {
    localStorage.setItem(CELL_LAYOUTS_LS_KEY, JSON.stringify(all));
  } catch {
    /* quota */
  }
}


function resetCellsToAutoLayout(): void {
  pendingUserAutoLayout = true;
  pendingScrollAfterAutoLayout = true;
  manualLayoutByCellIdx.clear();
  cellPositions.clear();
  lastLayoutCols = -1;
  const pathKey = layoutStoragePath();
  if (pathKey && pathKey !== "_unset") {
    const all = readCellLayoutsStore();
    delete all[pathKey];
    writeCellLayoutsStore(all);
  }
  applyFloatingLayout();
  setStatus("Cells: automatic layout");
}

/**
 * Repo-relative path for localStorage — must match script picker values.
 * Older backends sent absolute ``path`` in API/WebSocket; normalize so the first refresh finds saved layouts.
 */
function persistPathKey(pathFromServer: string | null | undefined): string {
  const fromInput = pathInput.value.trim().replace(/\\/g, "/");
  if (fromInput) return fromInput.replace(/^\.\//, "");
  const s = (pathFromServer ?? "").trim().replace(/\\/g, "/");
  if (!s) return "";
  if (!s.startsWith("/") && !/^[A-Za-z]:/.test(s)) return s.replace(/^\.\//, "");
  const low = s.toLowerCase();
  const i = low.indexOf("/experiments/");
  if (i >= 0) return s.slice(i + 1);
  const segs = s.split("/").filter(Boolean);
  if (segs.length >= 2) return segs.slice(-2).join("/");
  return segs[segs.length - 1] ?? "";
}

function layoutStoragePath(): string {
  const k = persistPathKey(pathInput.value.trim() || lastPath || null);
  return k || "_unset";
}

let saveCellLayoutTimer = 0;
function scheduleSaveCellLayouts() {
  window.clearTimeout(saveCellLayoutTimer);
  saveCellLayoutTimer = window.setTimeout(() => {
    saveCellLayoutTimer = 0;
    const pathKey = layoutStoragePath();
    if (!pathKey || pathKey === "_unset") return;
    const rec: CellLayoutsFileMap = {};
    for (const el of cellsCanvas.querySelectorAll<HTMLElement>(".cell[data-pipeline-cell-drag]")) {
      const idx = Number(el.dataset.pipelineCellDrag);
      if (!Number.isInteger(idx)) continue;
      rec[String(idx)] = [
        Math.round(parseFloat(el.style.left) || el.offsetLeft),
        Math.round(parseFloat(el.style.top) || el.offsetTop),
        Math.round(el.offsetWidth),
        Math.round(el.offsetHeight),
      ];
    }
    const all = readCellLayoutsStore();
    all[pathKey] = rec;
    writeCellLayoutsStore(all);
  }, 250);
}

const defaultPath =
  (urlParams.get("path") || "").trim() || readWatchPathCookie().trim() || "";

/** List *.py from this repo-relative folder; default `experiments` shows all dated experiment scripts. */
const scriptPickerDir =
  (urlParams.get("dir") ?? "").trim() || "experiments";

app.innerHTML = `
  <div class="toolbar">
    <div class="toolbar-watch">
      <span class="ws-dot" id="ws-dot" title="Live reload: disconnected" aria-label="WebSocket disconnected"></span>
      <span class="script-picker">
        <select id="folder-select" title="Experiment folder under list root" aria-label="Folder"></select>
        <select id="file-select" title="Python file in folder" aria-label="File"></select>
      </span>
      <input type="hidden" id="path-input" />
      <button type="button" class="primary btn-icon" id="btn-watch" title="Watch selected Python file (live cells and reload)" aria-label="Watch">${TOOLBAR_ICON_WATCH}</button>
      <button type="button" class="btn-icon" id="btn-reset" title="Restart the Stonesoup backend (fresh process; reclaims memory)" aria-label="Restart server">${TOOLBAR_ICON_RESET}</button>
      <button type="button" class="btn-icon" id="btn-abort" disabled title="Cooperative stop (enabled only if this cell’s source contains check_abort — add stonesoup.check_abort() in long loops). Long GPU stretches keep running until Python resumes." aria-label="Abort cell">${TOOLBAR_ICON_ABORT}</button>
    </div>
    <div class="toolbar-models" title="Load Hugging Face checkpoints; dropdown lists all models in this Stonesoup process. Unload / All remove weights from every open experiment kernel and free memory when nothing references a checkpoint.">
      <span class="model-repo-combo">
        <input type="text" id="model-repo-input" class="model-repo-input" list="model-repo-datalist" spellcheck="false" title="Type a repo id or pick from disk cache / recently loaded" aria-label="Hugging Face model repo id" autocomplete="off" />
        <datalist id="model-repo-datalist"></datalist>
      </span>
      <button type="button" class="primary btn-icon" id="btn-model-load" title="Load Hugging Face model from repo id above" aria-label="Load model">${TOOLBAR_ICON_MODEL_LOAD}</button>
      <select id="models-loaded-select" aria-label="Models in memory" title="Checkpoints loaded in this Stonesoup process. Unload removes the selection from every cached experiment kernel and frees memory when its refcount reaches zero."><option value="">—</option></select>
      <button type="button" class="btn-icon" id="btn-model-unload-one" title="Unload this checkpoint from every open experiment; frees memory when nothing still references it" aria-label="Unload selected model">${TOOLBAR_ICON_MODEL_UNLOAD}</button>
      <button type="button" class="btn-icon" id="btn-model-unload-all" title="Unload every Stonesoup model from every open experiment" aria-label="Unload all models">${TOOLBAR_ICON_MODEL_UNLOAD_ALL}</button>
    </div>
  </div>
  <div class="pipeline-row" id="pipeline-row">
    <div class="pipeline-aside">
      <span class="pipeline-label">Pipelines</span>
      <div class="loop-palette-slot" id="loop-palette-slot"></div>
    </div>
    <div class="pipelines-stack" id="pipelines-stack"></div>
  </div>
  <div class="workspace">
    <div class="cells" id="cells">
      <div class="cells-pan-arena" id="cells-pan-arena"><div class="cells-zoom-wrap" id="cells-zoom-wrap"><div class="cells-canvas" id="cells-canvas"></div></div></div>
    </div>
    <div class="stonesoup-console" id="stonesoup-console">
      <div class="stonesoup-console-panel stonesoup-dock-pane" id="stonesoup-console-panel" title="Click to copy all">
        <div class="stonesoup-console-toolbar stonesoup-dock-toolbar">
          <span class="stonesoup-console-title stonesoup-dock-title">Console</span>
          <button type="button" class="btn-icon" id="stonesoup-console-clear" title="Clear log">Clear</button>
          <button type="button" class="btn-icon" id="stonesoup-console-collapse" title="Hide console">▾</button>
        </div>
        <pre class="stonesoup-console-pre stonesoup-dock-body" id="stonesoup-console-pre" title="Click to copy all"></pre>
      </div>
    </div>
    <div class="kernel-vars-dock" id="kernel-vars-dock">
      <div class="kernel-vars-panel stonesoup-dock-pane" id="kernel-vars-panel">
        <div class="kernel-vars-toolbar stonesoup-dock-toolbar">
          <span class="kernel-vars-title stonesoup-dock-title">Variables</span>
          <button type="button" class="btn-icon" id="kernel-vars-refresh" title="Refresh list">⟳</button>
          <button type="button" class="btn-icon" id="kernel-vars-free-memory" title="Clear all variables for this experiment (including model/tokenizer bindings in this kernel). Shared checkpoints may stay in memory if still used elsewhere." aria-label="Clear all variables (this experiment)">${TOOLBAR_ICON_FREE_MEMORY}</button>
          <button type="button" class="btn-icon" id="kernel-vars-collapse" title="Hide variables">▾</button>
        </div>
        <div class="kernel-vars-scroll stonesoup-dock-body">
          <table class="kernel-vars-table" aria-label="Variables">
            <thead><tr><th>Name</th><th>Type</th><th>Value</th></tr></thead>
            <tbody id="kernel-vars-tbody"></tbody>
          </table>
          <p class="kernel-vars-empty" id="kernel-vars-empty" hidden>No user variables (only builtins).</p>
        </div>
      </div>
      <div class="kernel-vars-dock-bar">
        <button type="button" class="cells-auto-fab" id="btn-cells-auto-layout" title="Auto layout: discard saved positions/sizes and reflow into the default grid" aria-label="Auto layout">↻</button>
        <button type="button" class="cells-auto-fab" id="btn-editor-toggle" data-editor-pref="${getEditorPref()}" title="Switch editor for deeplinks (Cursor / VS Code)">${EDITOR_ICON_CURSOR}${EDITOR_ICON_VSCODE}</button>
        <button type="button" class="cells-auto-fab" id="btn-workspace-fullscreen" title="Fullscreen workspace (cells, console, variables)" aria-label="Enter fullscreen" aria-pressed="false"><svg class="fullscreen-icon fullscreen-icon--enter" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="16" height="16" aria-hidden="true"><path fill="currentColor" d="M7 14H5v5h5v-2H7v-3zm-2-4h2V7h3V5H5v5zm12 7h-3v2h5v-5h-2v3zM14 5v2h3v3h2V5h-5z"/></svg><svg class="fullscreen-icon fullscreen-icon--exit" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="16" height="16" aria-hidden="true"><path fill="currentColor" d="M5 16h3v3h2v-5H5v2zm0-6h5V5H5v5zm6 0v5h5v-5h-5zm6-6h-2v5h5V5h-3v2h-2V5z"/></svg></button>
        <button type="button" class="cells-auto-fab" id="btn-console-toggle" title="Console: server log, model load, and cell stdout/stderr" aria-label="Toggle console" aria-expanded="false">&gt;_</button>
        <button type="button" class="kernel-vars-chip" id="kernel-vars-toggle" aria-expanded="false" title="Show variables (this script)">
          <span class="kernel-vars-chip-icon" aria-hidden="true">{ }</span>
          <span class="kernel-vars-chip-sessions" id="kernel-vars-sessions"></span>
        </button>
      </div>
    </div>
  </div>
  <div id="status-toast" class="status-toast" role="status" aria-live="polite"></div>
`;

const folderSelect = app.querySelector<HTMLSelectElement>("#folder-select")!;
const fileSelect = app.querySelector<HTMLSelectElement>("#file-select")!;
const pathInput = app.querySelector<HTMLInputElement>("#path-input")!;
const btnWatch = app.querySelector<HTMLButtonElement>("#btn-watch")!;
const btnReset = app.querySelector<HTMLButtonElement>("#btn-reset")!;
const btnAbort = app.querySelector<HTMLButtonElement>("#btn-abort")!;
const btnEditorToggle = app.querySelector<HTMLButtonElement>("#btn-editor-toggle")!;
const modelRepoInput = app.querySelector<HTMLInputElement>("#model-repo-input")!;
const modelRepoDatalist = app.querySelector<HTMLDataListElement>("#model-repo-datalist")!;
const btnModelLoad = app.querySelector<HTMLButtonElement>("#btn-model-load")!;
const modelsLoadedSelect = app.querySelector<HTMLSelectElement>("#models-loaded-select")!;
const btnModelUnloadOne = app.querySelector<HTMLButtonElement>("#btn-model-unload-one")!;
const btnModelUnloadAll = app.querySelector<HTMLButtonElement>("#btn-model-unload-all")!;
const statusToastEl = app.querySelector<HTMLDivElement>("#status-toast")!;
const btnCellsAutoLayout = app.querySelector<HTMLButtonElement>("#btn-cells-auto-layout")!;
const workspaceEl = app.querySelector<HTMLDivElement>(".workspace")!;
const btnWorkspaceFullscreen = app.querySelector<HTMLButtonElement>("#btn-workspace-fullscreen")!;
const cellsEl = app.querySelector<HTMLDivElement>("#cells")!;
const cellsPanArena = app.querySelector<HTMLDivElement>("#cells-pan-arena")!;
const cellsZoomWrap = app.querySelector<HTMLDivElement>("#cells-zoom-wrap")!;
const cellsCanvas = app.querySelector<HTMLDivElement>("#cells-canvas")!;
cellsPanArena.style.boxSizing = "content-box";
cellsPanArena.style.padding = `${CELLS_PAN_GUTTER_PX}px`;
const pipelineRow = document.getElementById("pipeline-row")!;
const loopPaletteSlot = document.getElementById("loop-palette-slot")!;
const wsDot = app.querySelector<HTMLSpanElement>("#ws-dot")!;
const kernelVarsDock = app.querySelector<HTMLDivElement>("#kernel-vars-dock")!;
const kernelVarsPanel = app.querySelector<HTMLDivElement>("#kernel-vars-panel")!;
const kernelVarsTbody = app.querySelector<HTMLTableSectionElement>("#kernel-vars-tbody")!;
const kernelVarsEmpty = app.querySelector<HTMLParagraphElement>("#kernel-vars-empty")!;
const kernelVarsToggle = app.querySelector<HTMLButtonElement>("#kernel-vars-toggle")!;
const kernelVarsCollapse = app.querySelector<HTMLButtonElement>("#kernel-vars-collapse")!;
const kernelVarsRefresh = app.querySelector<HTMLButtonElement>("#kernel-vars-refresh")!;
const kernelVarsFreeMemory = app.querySelector<HTMLButtonElement>("#kernel-vars-free-memory")!;
const kernelVarsSessions = app.querySelector<HTMLSpanElement>("#kernel-vars-sessions")!;
const stonesoupConsoleRoot = app.querySelector<HTMLDivElement>("#stonesoup-console")!;
const stonesoupConsolePanel = app.querySelector<HTMLDivElement>("#stonesoup-console-panel")!;
const stonesoupConsolePre = app.querySelector<HTMLPreElement>("#stonesoup-console-pre")!;
const btnConsoleToggle = app.querySelector<HTMLButtonElement>("#btn-console-toggle")!;
const btnConsoleClear = app.querySelector<HTMLButtonElement>("#stonesoup-console-clear")!;
const btnConsoleCollapse = app.querySelector<HTMLButtonElement>("#stonesoup-console-collapse")!;

const KERNEL_VARS_EXPANDED_KEY = "stonesoup_kernel_vars_expanded";
const STONESOUP_CONSOLE_EXPANDED_KEY = "stonesoup_console_expanded";
/** Persisted across page refresh (same idea as cell outputs). */
const STONESOUP_CONSOLE_LOG_LS_KEY = "stonesoup_console_log_v1";
/** Cap retained server log text so the DOM stays bounded. */
const CONSOLE_BUFFER_MAX = 200 * 1024;
let appLogBuffer = "";
let saveConsoleTimer = 0;

function resetLoopPaletteSlotPosition(el: HTMLElement) {
  el.classList.remove("loop-palette--dragging");
  el.style.position = "";
  el.style.left = "";
  el.style.top = "";
  el.style.width = "";
  el.style.height = "";
  el.style.minHeight = "";
  el.style.zIndex = "";
}

cellsEl.addEventListener("click", async (e) => {
  const out = (e.target as HTMLElement).closest<HTMLElement>(".out");
  if (!out || !cellsCanvas.contains(out)) return;
  const mk = out.dataset.markerKey;
  if (!mk) return;
  const o = outputs.get(mk);
  const text = o ? formatOut(o) : (out.textContent ?? "").trimEnd();
  if (!text) return;
  try {
    await navigator.clipboard.writeText(text);
    setStatus("Output copied");
    out.classList.remove("out-copied");
    void out.offsetWidth;
    out.classList.add("out-copied");
    window.setTimeout(() => out.classList.remove("out-copied"), 500);
  } catch {
    setStatus("Copy failed (clipboard permission?)");
  }
});

pathInput.value = defaultPath;

type ScriptFileEntry = { rel: string; label: string; mtime: number };
type LoadedModelInfo = { name: string; repo_id: string };

type GlobalLoadedModelInfo = {
  pool_key_b64: string;
  repo_id: string;
  model_kind?: string;
  device_map?: string | null;
  torch_dtype?: string;
  trust_remote_code?: boolean;
};

/** Sentinel: ``*.py`` directly under the list root (group key; not a real folder name). */
const SCRIPT_PICKER_ROOT_FOLDER = "__ss_root__";

/** First path segment under list root → files (full repo-relative path + label under that folder). */
let scriptPickerGroups: Map<string, ScriptFileEntry[]> = new Map();

function normalizeRelPath(p: string): string {
  return p.replace(/\\/g, "/").replace(/\/+$/, "");
}

/** One or more HF ``repo_id`` strings; comma or newline separated. No ``name=repo`` aliases. */
function parseHfRepoIds(raw: string): { repo_id: string }[] {
  const items: { repo_id: string }[] = [];
  const seen = new Set<string>();
  const parts = raw
    .split(/\r?\n/)
    .flatMap((line) => line.split(","))
    .map((part) => part.trim())
    .filter(Boolean);
  for (const part of parts) {
    if (part.includes("=")) {
      throw new Error(
        "Use Hugging Face repo ids only (e.g. Qwen/Qwen3-VL-8B-Instruct), not name=repo.",
      );
    }
    if (seen.has(part)) continue;
    seen.add(part);
    items.push({ repo_id: part });
  }
  return items;
}

const MODEL_REPO_RECENT_KEY = "stonesoup_hf_repo_recent";
const MODEL_REPO_RECENT_MAX = 30;

function readRecentHfRepoIds(): string[] {
  try {
    const raw = localStorage.getItem(MODEL_REPO_RECENT_KEY);
    if (!raw) return [];
    const a = JSON.parse(raw) as unknown;
    if (!Array.isArray(a)) return [];
    return a
      .filter((x): x is string => typeof x === "string" && x.trim().length > 0)
      .map((x) => x.trim())
      .filter(Boolean);
  } catch {
    return [];
  }
}

function rememberHfRepoIdsLoaded(repoIds: string[]) {
  if (!repoIds.length) return;
  const prev = readRecentHfRepoIds();
  const next = [
    ...repoIds.map((r) => r.trim()).filter(Boolean),
    ...prev,
  ];
  const seen = new Set<string>();
  const dedup: string[] = [];
  for (const id of next) {
    if (seen.has(id)) continue;
    seen.add(id);
    dedup.push(id);
    if (dedup.length >= MODEL_REPO_RECENT_MAX) break;
  }
  localStorage.setItem(
    MODEL_REPO_RECENT_KEY,
    JSON.stringify(dedup.slice(0, MODEL_REPO_RECENT_MAX)),
  );
}

function mergeModelRepoSuggestions(cached: string[], recent: string[]): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const raw of recent) {
    const t = raw.trim();
    if (!t || seen.has(t)) continue;
    seen.add(t);
    out.push(t);
  }
  for (const raw of cached) {
    const t = raw.trim();
    if (!t || seen.has(t)) continue;
    seen.add(t);
    out.push(t);
  }
  return out;
}

/**
 * Minimum width for the repo combo (``ch``). Longest datalist id / typed value is scaled—``ch`` matches
 * “0” width and is a bit thin for mixed repo slugs.
 */
const MODEL_REPO_WIDTH_MIN_CH = 20;
const MODEL_REPO_WIDTH_MAX_CH = 20;
const MODEL_REPO_WIDTH_PAD_CH = 4;
const MODEL_REPO_WIDTH_CH_SCALE = 1.14;

function syncModelRepoInputWidthCh() {
  let maxChars = modelRepoInput.value.length;
  for (const o of modelRepoDatalist.querySelectorAll("option")) {
    maxChars = Math.max(maxChars, (o as HTMLOptionElement).value.length);
  }
  const raw = Math.ceil(Math.max(maxChars, 1) * MODEL_REPO_WIDTH_CH_SCALE) + MODEL_REPO_WIDTH_PAD_CH;
  const ch = Math.min(MODEL_REPO_WIDTH_MAX_CH, Math.max(MODEL_REPO_WIDTH_MIN_CH, raw));
  const combo = modelRepoInput.closest(".model-repo-combo") as HTMLElement | null;
  if (combo) combo.style.minWidth = `${ch}ch`;
}

function fillModelRepoDatalist(repoIds: string[]) {
  modelRepoDatalist.replaceChildren();
  for (const id of repoIds) {
    const opt = document.createElement("option");
    opt.value = id;
    modelRepoDatalist.appendChild(opt);
  }
  syncModelRepoInputWidthCh();
}

async function refreshModelRepoSuggestions() {
  const recent = readRecentHfRepoIds();
  let cached: string[] = [];
  try {
    const r = await fetch(`${apiBase}/api/models/hf-cache`);
    const j = (await readApiJson(r)) as { repo_ids?: string[]; detail?: string };
    if (r.ok && Array.isArray(j.repo_ids)) {
      cached = j.repo_ids.filter(
        (x): x is string => typeof x === "string" && x.trim().length > 0,
      );
    }
  } catch {
    /* ignore */
  }
  fillModelRepoDatalist(mergeModelRepoSuggestions(cached, recent));
}

function globalChipTitle(g: GlobalLoadedModelInfo): string {
  const parts = [
    g.repo_id,
    g.model_kind && `kind=${g.model_kind}`,
    g.device_map != null && g.device_map !== "" && `device_map=${g.device_map}`,
    g.torch_dtype && `dtype=${g.torch_dtype}`,
    g.trust_remote_code ? "trust_remote_code" : "",
  ].filter(Boolean);
  return parts.join(" · ");
}

function globalModelOptionLabel(g: GlobalLoadedModelInfo, disambiguate: boolean): string {
  if (!disambiguate) return g.repo_id;
  const bits: string[] = [];
  if (g.torch_dtype) bits.push(String(g.torch_dtype).replace(/^torch\./, ""));
  if (g.device_map) bits.push(String(g.device_map));
  const suffix = bits.length ? ` (${bits.slice(0, 2).join(", ")})` : "";
  return `${g.repo_id}${suffix}`;
}

function populateModelsLoadedSelect(rows: GlobalLoadedModelInfo[]) {
  modelsLoadedSelect.replaceChildren();
  if (!rows.length) {
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "No models in memory";
    modelsLoadedSelect.appendChild(opt);
    modelsLoadedSelect.disabled = true;
    btnModelUnloadOne.disabled = true;
    btnModelUnloadAll.disabled = true;
    return;
  }
  modelsLoadedSelect.disabled = false;
  const placeholder = document.createElement("option");
  placeholder.value = "";
  placeholder.textContent = "—";
  modelsLoadedSelect.appendChild(placeholder);
  const repoCounts = new Map<string, number>();
  for (const g of rows) {
    repoCounts.set(g.repo_id, (repoCounts.get(g.repo_id) ?? 0) + 1);
  }
  for (const g of rows) {
    const opt = document.createElement("option");
    opt.value = g.pool_key_b64;
    const dup = (repoCounts.get(g.repo_id) ?? 0) > 1;
    opt.textContent = globalModelOptionLabel(g, dup);
    opt.title = globalChipTitle(g);
    opt.dataset.repoId = g.repo_id;
    modelsLoadedSelect.appendChild(opt);
  }
  modelsLoadedSelect.selectedIndex = 0;
  btnModelUnloadOne.disabled = false;
  btnModelUnloadAll.disabled = false;
}

/** Copy the selected loaded model's Hugging Face ``repo_id`` (for ``stonesoup.load_model``). */
function copyLoadedModelRepoIdToClipboard() {
  const sel = modelsLoadedSelect.selectedOptions[0];
  if (!sel || !sel.value) return;
  const repoId = sel.dataset.repoId?.trim() || sel.textContent?.replace(/\s+\([^)]*\)\s*$/, "").trim() || "";
  if (!repoId || repoId === "No models in memory") return;
  navigator.clipboard
    .writeText(repoId)
    .then(() => setStatus(`Copied ${repoId}`))
    .catch(() => {
      modelRepoInput.value = repoId;
      setStatus("Repo id in loader field (clipboard blocked)");
    });
}

function setModelLoaderBusy(busy: boolean) {
  modelRepoInput.disabled = busy;
  btnModelLoad.disabled = busy;
  if (busy) {
    modelsLoadedSelect.disabled = true;
    btnModelUnloadOne.disabled = true;
    btnModelUnloadAll.disabled = true;
  }
}

function parseLoadedGlobally(raw: unknown): GlobalLoadedModelInfo[] {
  if (!Array.isArray(raw)) return [];
  const out: GlobalLoadedModelInfo[] = [];
  for (const item of raw) {
    if (!item || typeof item !== "object") continue;
    const o = item as Record<string, unknown>;
    if (typeof o.repo_id !== "string" || !o.repo_id.trim()) continue;
    if (typeof o.pool_key_b64 !== "string" || !o.pool_key_b64.trim()) continue;
    let device_map: string | null | undefined;
    if ("device_map" in o) {
      if (o.device_map === null) device_map = null;
      else if (o.device_map === undefined) device_map = undefined;
      else device_map = String(o.device_map);
    } else {
      device_map = undefined;
    }
    out.push({
      pool_key_b64: o.pool_key_b64.trim(),
      repo_id: o.repo_id.trim(),
      model_kind: typeof o.model_kind === "string" ? o.model_kind : undefined,
      device_map,
      torch_dtype: typeof o.torch_dtype === "string" ? o.torch_dtype : undefined,
      trust_remote_code: typeof o.trust_remote_code === "boolean" ? o.trust_remote_code : undefined,
    });
  }
  return out;
}

async function fetchLoadedModels(): Promise<GlobalLoadedModelInfo[]> {
  try {
    const r = await fetch(`${apiBase}/api/models`);
    const j = (await readApiJson(r)) as {
      loaded_globally?: unknown;
      detail?: string;
    };
    if (!r.ok) throw new Error(j.detail || r.statusText);
    const rows = parseLoadedGlobally(j.loaded_globally);
    populateModelsLoadedSelect(rows);
    return rows;
  } catch {
    populateModelsLoadedSelect([]);
    return [];
  }
}

/**
 * Group ``*.py`` paths by the first directory under ``root`` (posix). Files directly under ``root``
 * use ``SCRIPT_PICKER_ROOT_FOLDER``.
 */
function groupPyFilesUnderRoot(
  root: string,
  rows: { rel: string; mtime: number }[],
): Map<string, ScriptFileEntry[]> {
  const r = normalizeRelPath(root);
  const groups = new Map<string, ScriptFileEntry[]>();
  for (const row of rows) {
    const rel = row.rel.replace(/\\/g, "/");
    if (!rel.toLowerCase().endsWith(".py")) continue;
    if (rel === r) continue;
    if (!rel.startsWith(r + "/")) continue;
    const rest = rel.slice(r.length + 1);
    const slash = rest.indexOf("/");
    const folderKey = slash === -1 ? SCRIPT_PICKER_ROOT_FOLDER : rest.slice(0, slash);
    const tail = slash === -1 ? rest : rest.slice(slash + 1);
    if (!tail.toLowerCase().endsWith(".py")) continue;
    const list = groups.get(folderKey) ?? [];
    list.push({ rel, label: tail, mtime: row.mtime });
    groups.set(folderKey, list);
  }
  for (const list of groups.values()) {
    list.sort((a, b) => b.mtime - a.mtime || a.label.localeCompare(b.label));
  }
  return groups;
}

function folderPickerLabel(folderKey: string): string {
  if (folderKey !== SCRIPT_PICKER_ROOT_FOLDER) return folderKey;
  const leaf = scriptPickerDir.split("/").filter(Boolean).pop();
  return leaf ? `(${leaf})` : "(root)";
}

function populateFileOptions(folderKey: string) {
  fileSelect.innerHTML = "";
  const entries = scriptPickerGroups.get(folderKey) ?? [];
  for (const e of entries) {
    const opt = document.createElement("option");
    opt.value = e.rel;
    opt.textContent = e.label;
    opt.title = e.rel;
    fileSelect.appendChild(opt);
  }
}

function pickFolderKeyForPath(pathWanted: string, groups: Map<string, ScriptFileEntry[]>): string | null {
  const want = normalizeRelPath(pathWanted);
  for (const [key, entries] of groups) {
    if (entries.some((e) => e.rel === want)) return key;
  }
  return null;
}

async function fetchPyFilesUnderDir(dir: string): Promise<{ rel: string; mtime: number }[]> {
  const params = new URLSearchParams({ dir, recursive: "true" });
  const r = await fetch(`${apiBase}/api/py-files?${params}`);
  const j = (await r.json()) as { files?: string[]; mtimes?: number[] };
  if (!r.ok) throw new Error((j as { detail?: string }).detail || r.statusText);
  const files = j.files ?? [];
  const mtimes = j.mtimes;
  if (!mtimes || mtimes.length !== files.length) {
    return files.map((rel) => ({ rel, mtime: 0 }));
  }
  return files.map((rel, i) => ({ rel, mtime: Number(mtimes[i]) || 0 }));
}

/** The single ``*.py`` with the newest mtime under the list root (tie: smaller ``rel``). */
function pickGloballyLatestEntry(groups: Map<string, ScriptFileEntry[]>): ScriptFileEntry | null {
  let best: ScriptFileEntry | null = null;
  for (const entries of groups.values()) {
    for (const e of entries) {
      if (
        !best ||
        e.mtime > best.mtime ||
        (e.mtime === best.mtime && e.rel.localeCompare(best.rel) < 0)
      ) {
        best = e;
      }
    }
  }
  return best;
}

/** Latest-touched file in this folder (``groupPyFilesUnderRoot`` sorts newest first). */
function pickLatestFileEntry(entries: ScriptFileEntry[]): ScriptFileEntry | null {
  return entries.length ? entries[0]! : null;
}

/** Newest ``mtime`` among ``*.py`` in this folder group (for ordering the folder dropdown). */
function folderGroupMaxMtime(entries: ScriptFileEntry[]): number {
  let m = 0;
  for (const e of entries) {
    if (e.mtime > m) m = e.mtime;
  }
  return m;
}

/**
 * Leading ``YYYY-MM-DD`` from experiment folder names (e.g. ``2026-04-13-CKA-pitfall``).
 * Used so the folder list follows calendar order, not “latest touched .py” (which can reorder
 * old dated folders when any file is edited).
 */
const FOLDER_NAME_DATE_PREFIX = /^(\d{4}-\d{2}-\d{2})(?:-|$)/;

function folderNameSortDate(folderKey: string): string | null {
  if (folderKey === SCRIPT_PICKER_ROOT_FOLDER) return null;
  const m = folderKey.match(FOLDER_NAME_DATE_PREFIX);
  return m ? m[1]! : null;
}

/** Sort folder keys: dated names by embedded date desc, then mtime; undated last. */
function compareScriptPickerFolderKeys(
  a: string,
  b: string,
  groups: Map<string, ScriptFileEntry[]>,
): number {
  const da = folderNameSortDate(a);
  const db = folderNameSortDate(b);
  if (da !== null && db !== null) {
    if (db !== da) return db.localeCompare(da);
  } else if (da !== null && db === null) {
    return -1;
  } else if (da === null && db !== null) {
    return 1;
  }
  const ma = folderGroupMaxMtime(groups.get(a) ?? []);
  const mb = folderGroupMaxMtime(groups.get(b) ?? []);
  if (mb !== ma) return mb - ma;
  if (a === SCRIPT_PICKER_ROOT_FOLDER) return 1;
  if (b === SCRIPT_PICKER_ROOT_FOLDER) return -1;
  return a.localeCompare(b);
}

async function populateScriptPicker() {
  folderSelect.innerHTML = "";
  fileSelect.innerHTML = "";

  if (!scriptPickerDir) {
    folderSelect.disabled = true;
    fileSelect.disabled = true;
    return;
  }

  folderSelect.disabled = false;
  fileSelect.disabled = false;
  try {
    let listDir = scriptPickerDir;
    let rows = await fetchPyFilesUnderDir(listDir);
    scriptPickerGroups = groupPyFilesUnderRoot(listDir, rows);

    const want = pathInput.value.trim().replace(/\\/g, "/");
    let chosenKey = pickFolderKeyForPath(want, scriptPickerGroups);
    if (
      chosenKey === null &&
      want &&
      want.startsWith("experiments/") &&
      listDir !== "experiments"
    ) {
      rows = await fetchPyFilesUnderDir("experiments");
      listDir = "experiments";
      scriptPickerGroups = groupPyFilesUnderRoot("experiments", rows);
      chosenKey = pickFolderKeyForPath(want, scriptPickerGroups);
    }

    const keys = [...scriptPickerGroups.keys()].sort((a, b) =>
      compareScriptPickerFolderKeys(a, b, scriptPickerGroups),
    );

    for (const key of keys) {
      const opt = document.createElement("option");
      opt.value = key;
      opt.textContent = folderPickerLabel(key);
      folderSelect.appendChild(opt);
    }

    if (chosenKey === null && keys.length > 0) {
      const latest = pickGloballyLatestEntry(scriptPickerGroups);
      chosenKey = latest
        ? pickFolderKeyForPath(latest.rel, scriptPickerGroups)
        : keys[0]!;
    }
    if (keys.length === 0) {
      folderSelect.disabled = true;
      fileSelect.disabled = true;
    } else if (chosenKey !== null) {
      folderSelect.value = chosenKey;
      populateFileOptions(chosenKey);
      const entries = scriptPickerGroups.get(chosenKey) ?? [];
      const match = entries.find((e) => e.rel === want);
      if (match) {
        fileSelect.value = match.rel;
        pathInput.value = match.rel;
      } else if (entries.length > 0) {
        const pick = pickLatestFileEntry(entries)!;
        fileSelect.value = pick.rel;
        pathInput.value = pick.rel;
      }
    }
  } catch {
    scriptPickerGroups = new Map();
    const err = document.createElement("option");
    err.value = "";
    err.textContent = "(could not list folder)";
    err.disabled = true;
    folderSelect.appendChild(err);
    folderSelect.disabled = true;
    fileSelect.disabled = true;
  }
}

folderSelect.addEventListener("change", () => {
  const key = folderSelect.value;
  populateFileOptions(key);
  const entries = scriptPickerGroups.get(key) ?? [];
  const pick = pickLatestFileEntry(entries);
  if (!pick) {
    return;
  }
  fileSelect.value = pick.rel;
  pathInput.value = pick.rel;
  void postWatch();
});

fileSelect.addEventListener("change", () => {
  const v = fileSelect.value.trim();
  if (!v) return;
  pathInput.value = v;
  void postWatch();
});

void populateScriptPicker().then(() => {
  void postWatch();
});

let revision = 0;
let lastCells: Cell[] = [];
let lastPath: string | null = null;
let ws: WebSocket | null = null;
/** Cell indices whose source changed on disk since last successful run (merged from server + cleared on run). */
const staleCells = new Set<number>();
/** Last ``revision`` we applied ``changed_cell_indices`` from (avoids re-flagging stale on every WS ``cells`` resend / reconnect). */
let lastRevisionStaleMerged = -1;

/** Per-cell run input text; merged into kernel as ``CELL_INPUT`` (survives UI re-render). Keyed by ``marker_key``. */
const cellRunInputDraft = new Map<string, string>();

/** How stdout is interpreted when not forced to plain via chip; only ``html`` / ``markdown`` enable rich rendering (first-line hint). */
type StdoutKind = "text" | "html" | "markdown";

/** Kernel result for one cell run; ``renderHint`` from optional first stdout line ``# stonesoup:render=…`` (stripped from ``stdout``). */
type CellOutput = {
  stdout: string;
  stderr: string;
  ok: boolean;
  renderHint?: StdoutKind | null;
  /** Wall time for ``kernel.run_cell`` only (seconds), from the server. */
  durationSec?: number;
};
/** Keyed by cell ``marker_key`` (stable when cells are reordered / indices shift). */
const outputs = new Map<string, CellOutput>();

/** When set, stdout is shown escaped (toggle chip); only meaningful for HTML/MD preset outputs. Keys = ``marker_key``. */
const cellStdoutPlainText = new Set<string>();

const STONESOUP_RENDER_FIRST_LINE = /^\s*#\s*stonesoup:render\s*=\s*(auto|text|html|markdown|md)\s*$/i;

/** Collapse ``\r`` per line like a TTY so tqdm/progress bars do not accumulate in streamed logs. */
function foldCarriageReturns(s: string): string {
  if (!s.includes("\r")) return s;
  return s
    .split("\n")
    .map((line) => {
      const i = line.lastIndexOf("\r");
      return i === -1 ? line : line.slice(i + 1);
    })
    .join("\n");
}

/** Format server-reported cell duration for the output header. */
function formatDurationSec(sec: number): string {
  if (!Number.isFinite(sec) || sec < 0) return "";
  if (sec < 1) return `${Math.round(sec * 1000)} ms`;
  if (sec < 10) return `${sec.toFixed(2)} s`;
  return `${sec.toFixed(1)} s`;
}

/** Strip leading ``# stonesoup:render=…`` line; ``md`` → markdown, ``auto``/``text``/omitted → plain stdout (no guessing). */
function peelStonesoupRenderHint(raw: string): { body: string; renderHint: StdoutKind | null } {
  const s = raw.replace(/^\ufeff/, "");
  if (!s) return { body: "", renderHint: null };
  const nl = s.indexOf("\n");
  const first = (nl === -1 ? s : s.slice(0, nl)).replace(/\r$/, "");
  const rest = nl === -1 ? "" : s.slice(nl + 1);
  const m = first.match(STONESOUP_RENDER_FIRST_LINE);
  if (!m?.[1]) return { body: s, renderHint: null };
  const v = m[1].toLowerCase();
  const mode = (v === "md" ? "markdown" : v) as StdoutKind | "auto";
  const renderHint: StdoutKind | null = mode === "auto" ? null : mode;
  return { body: rest, renderHint };
}

function markerKeyForCellIndex(index: number): string | undefined {
  return lastCells.find((c) => c.index === index)?.marker_key;
}

function cellIndexForMarkerKey(markerKey: string): number | undefined {
  return lastCells.find((c) => c.marker_key === markerKey)?.index;
}

function cellRunInputValue(index: number): string {
  const el = cellsEl.querySelector<HTMLInputElement>(`[data-run-input="${index}"]`);
  if (el) return el.value;
  const mk = markerKeyForCellIndex(index);
  return mk ? (cellRunInputDraft.get(mk) ?? "") : "";
}

function cellWantsRunInput(index: number): boolean {
  return lastCells.some((c) => c.index === index && c.cell_input === true);
}

function mergeCellRunInject(index: number, inject?: Record<string, unknown> | null): Record<string, unknown> {
  const base: Record<string, unknown> = { ...(inject ?? {}) };
  if (cellWantsRunInput(index)) {
    base.CELL_INPUT = cellRunInputValue(index);
  }
  return base;
}

/** Drop output state for cells removed from the file (``marker_key`` no longer present). */
function pruneOutputsForRemovedCells(cells: Cell[]) {
  const valid = new Set(cells.map((c) => c.marker_key).filter(Boolean));
  for (const k of [...outputs.keys()]) {
    if (!valid.has(k)) outputs.delete(k);
  }
  for (const k of [...cellStdoutPlainText]) {
    if (!valid.has(k)) cellStdoutPlainText.delete(k);
  }
  schedulePersistOutputs();
}

function pruneStaleCells(cellCount: number) {
  for (const k of [...staleCells]) {
    if (!Number.isInteger(k) || k < 0 || k >= cellCount) staleCells.delete(k);
  }
}
/** Last-known cell geometry for reflow heuristics; cleared when watched path or cell count changes. */
const cellPositions = new Map<number, { left: number; top: number }>();
/** When non-empty, canvas uses saved left/top/width/height per file cell index (from localStorage or after drag). */
const manualLayoutByCellIdx = new Map<number, { left: number; top: number; width: number; height: number }>();
function loadManualLayoutsForPath(pathKey: string) {
  if (!pathKey || pathKey === "_unset") return;
  manualLayoutByCellIdx.clear();
  const all = readCellLayoutsStore();
  const rec = all[pathKey];
  if (!rec) return;
  for (const [k, v] of Object.entries(rec)) {
    const idx = Number(k);
    if (!Number.isInteger(idx) || !Array.isArray(v) || v.length !== 4) continue;
    const [l, t, w, h] = v;
    if (![l, t, w, h].every((x) => typeof x === "number" && Number.isFinite(x))) continue;
    manualLayoutByCellIdx.set(idx, { left: l, top: t, width: w, height: h });
  }
}

function snapshotCurrentLayoutToManualMap() {
  manualLayoutByCellIdx.clear();
  for (const el of cellsCanvas.querySelectorAll<HTMLElement>(".cell[data-pipeline-cell-drag]")) {
    const idx = Number(el.dataset.pipelineCellDrag);
    if (!Number.isInteger(idx)) continue;
    manualLayoutByCellIdx.set(idx, {
      left: el.offsetLeft,
      top: el.offsetTop,
      width: el.offsetWidth,
      height: el.offsetHeight,
    });
  }
}
let lastLayoutPath = "";
let lastLayoutCount = -1;
/** Last grid column count used when computing default grid geometry (no reflow on resize unless user clicks auto-layout). */
let lastLayoutCols = -1;

/** True after path/cell-count change when no saved layout exists — run default grid once, then persist. */
let needsDefaultCellGrid = false;
/** True when user clicked “automatic cell layout” — discard saved layout and run default grid once. */
let pendingUserAutoLayout = false;
/** True when user clicked auto-layout: after grid geometry is applied, pan viewport to default origin. */
let pendingScrollAfterAutoLayout = false;

const CELL_OUTPUTS_LS_KEY = "stonesoup_cell_outputs_v2";
/** Legacy: keys per file were ``"0"``, ``"1"``, … (cell index). Migrated on load using current ``marker_key`` list. */
const CELL_OUTPUTS_LEGACY_KEY = "stonesoup_cell_outputs_v1";
type CellOutputsFileMap = Record<string, CellOutput>;

function parseCellOutputsStore(): Record<string, CellOutputsFileMap> {
  try {
    const raw = localStorage.getItem(CELL_OUTPUTS_LS_KEY);
    if (!raw) return {};
    const x = JSON.parse(raw) as unknown;
    if (x && typeof x === "object" && !Array.isArray(x)) return x as Record<string, CellOutputsFileMap>;
  } catch {
    /* ignore */
  }
  return {};
}

function loadLegacyCellOutputsPathMap(pathKey: string): CellOutputsFileMap | null {
  try {
    const raw = localStorage.getItem(CELL_OUTPUTS_LEGACY_KEY);
    if (!raw) return null;
    const all = JSON.parse(raw) as unknown;
    if (!all || typeof all !== "object" || Array.isArray(all)) return null;
    const rec = (all as Record<string, CellOutputsFileMap>)[pathKey];
    return rec && typeof rec === "object" && !Array.isArray(rec) ? rec : null;
  } catch {
    return null;
  }
}

function writeCellOutputsStore(all: Record<string, CellOutputsFileMap>) {
  try {
    localStorage.setItem(CELL_OUTPUTS_LS_KEY, JSON.stringify(all));
  } catch {
    /* quota or private mode */
  }
}

let saveOutputsTimer = 0;
function schedulePersistOutputs() {
  window.clearTimeout(saveOutputsTimer);
  saveOutputsTimer = window.setTimeout(() => {
    saveOutputsTimer = 0;
    const pathKey = layoutStoragePath();
    if (!pathKey || pathKey === "_unset") return;
    const rec: CellOutputsFileMap = {};
    for (const [k, v] of outputs) {
      rec[String(k)] = v;
    }
    const all = parseCellOutputsStore();
    all[pathKey] = rec;
    writeCellOutputsStore(all);
  }, 300);
}

function parseCellOutputRecord(v: unknown): CellOutput | null {
  if (!v || typeof v !== "object") return null;
  const o = v as Record<string, unknown>;
  const out: CellOutput = {
    stdout: String(o.stdout ?? ""),
    stderr: String(o.stderr ?? ""),
    ok: Boolean(o.ok),
  };
  const rh = o.renderHint;
  if (rh === "html" || rh === "markdown" || rh === "text") {
    out.renderHint = rh as StdoutKind;
  }
  const ds = o.durationSec;
  if (typeof ds === "number" && Number.isFinite(ds)) out.durationSec = ds;
  return out;
}

/** Restore saved outputs; numeric keys (legacy) map to ``cells[i].marker_key``. */
function loadOutputsForPath(pathKey: string, cells: Cell[]) {
  if (!pathKey || pathKey === "_unset") return;
  const all = parseCellOutputsStore();
  let rec: CellOutputsFileMap | undefined = all[pathKey];
  if (!rec || Object.keys(rec).length === 0) {
    rec = loadLegacyCellOutputsPathMap(pathKey) ?? undefined;
  }
  if (!rec) return;
  for (const [k, v] of Object.entries(rec)) {
    let mk: string | null = null;
    if (/^\d+$/.test(k)) {
      const idx = Number(k);
      if (Number.isInteger(idx) && idx >= 0 && idx < cells.length) mk = cells[idx]!.marker_key;
      else continue;
    } else {
      mk = k;
    }
    if (!mk) continue;
    const out = parseCellOutputRecord(v);
    if (out) outputs.set(mk, out);
  }
}

function clearPersistedOutputsForPath(pathKey: string) {
  if (!pathKey || pathKey === "_unset") return;
  const all = parseCellOutputsStore();
  delete all[pathKey];
  writeCellOutputsStore(all);
  try {
    const raw = localStorage.getItem(CELL_OUTPUTS_LEGACY_KEY);
    if (!raw) return;
    const legacy = JSON.parse(raw) as unknown;
    if (!legacy || typeof legacy !== "object" || Array.isArray(legacy)) return;
    const m = legacy as Record<string, CellOutputsFileMap>;
    if (!m[pathKey]) return;
    delete m[pathKey];
    localStorage.setItem(CELL_OUTPUTS_LEGACY_KEY, JSON.stringify(m));
  } catch {
    /* ignore */
  }
}

/** Tree: cells and nested loops (each loop has its own iteration list). */
type PipelineStep =
  | { kind: "cell"; index: number }
  /** Each element is stored as in JSON: plain objects merge into the kernel; anything else becomes `LOOP_ITEM` (+ `LOOP_INDEX`) at run time only. */
  | { kind: "loop"; iterations: unknown[]; body: PipelineStep[] };

let pipelines: PipelineStep[][] = [[]];
/** Loop iteration editors that are open (`"${pIdx}:${pathJson}"` keys). */
const loopConfigExpanded = new Set<string>();

function clearLoopExpanded() {
  loopConfigExpanded.clear();
}

function pipelineKeyForStorage(): string {
  return pathInput.value.trim() || lastPath || "_unset";
}

function pipelineStorageKeyV1(): string {
  return `stonesoup-pipeline:${encodeURIComponent(pipelineKeyForStorage())}`;
}

function pipelineStorageKeyV2(): string {
  return `stonesoup-pipeline-v2:${encodeURIComponent(pipelineKeyForStorage())}`;
}

function pipelineStorageKeyV3(): string {
  return `stonesoup-pipelines-v3:${encodeURIComponent(pipelineKeyForStorage())}`;
}

function loopExpandedKey(pIdx: number, pathJson: string): string {
  return `${pIdx}:${pathJson}`;
}

/** If every element is the old `{ STONESOUP_ITEM | LOOP_ITEM, STONESOUP_INDEX | LOOP_INDEX }` shape, unwrap to a simple list (e.g. `[1,2,3]`). */
function normalizeLegacyIterations(items: unknown[]): unknown[] {
  if (items.length === 0) return items;
  for (let i = 0; i < items.length; i++) {
    const item = items[i];
    const o = item as Record<string, unknown>;
    const legacyIdx = o.STONESOUP_INDEX ?? o.LOOP_INDEX;
    const hasLegacyItem =
      Object.prototype.hasOwnProperty.call(o, "STONESOUP_ITEM") ||
      Object.prototype.hasOwnProperty.call(o, "LOOP_ITEM");
    if (
      item === null ||
      typeof item !== "object" ||
      Array.isArray(item) ||
      Object.keys(item as object).length !== 2 ||
      legacyIdx !== i ||
      !hasLegacyItem
    ) {
      return items;
    }
  }
  return items.map((item) => {
    const o = item as Record<string, unknown>;
    if (Object.prototype.hasOwnProperty.call(o, "STONESOUP_ITEM")) return o.STONESOUP_ITEM;
    return o.LOOP_ITEM;
  });
}

/** Parse loop iterations JSON; keeps your values as written (no rewriting scalars). */
function parseIterationsJson(
  raw: string,
): { ok: true; iterations: unknown[] } | { ok: false; error: string } {
  const t = raw.trim();
  if (!t) return { ok: true, iterations: [{}] };
  let data: unknown;
  try {
    data = JSON.parse(t) as unknown;
  } catch (e) {
    return { ok: false, error: `Loop JSON: ${String(e)}` };
  }
  if (!Array.isArray(data)) {
    return { ok: false, error: "Loop JSON must be a JSON array (e.g. [1,2,3] or [{\"lr\":0.1}])" };
  }
  if (data.length === 0) return { ok: true, iterations: [{}] };
  return { ok: true, iterations: normalizeLegacyIterations(data) };
}

function iterationsToJson(a: unknown[]): string {
  try {
    return JSON.stringify(a);
  } catch {
    return "[{}]";
  }
}

/** One loop element → globals patch for this iteration (scalars only → LOOP_ITEM / LOOP_INDEX). */
function iterationToInject(
  item: unknown,
  index: number,
): Record<string, unknown> | undefined {
  if (item !== null && typeof item === "object" && !Array.isArray(item)) {
    const o = item as Record<string, unknown>;
    /** Always expose index; `{}` iterations used to skip inject and left the index unset. */
    return { ...o, LOOP_INDEX: index };
  }
  return { LOOP_ITEM: item, LOOP_INDEX: index };
}

function reviveStep(x: unknown): PipelineStep | null {
  if (!x || typeof x !== "object") return null;
  const o = x as Record<string, unknown>;
  if (o.kind === "cell" && typeof o.index === "number" && Number.isInteger(o.index)) {
    return { kind: "cell", index: o.index };
  }
  if (o.kind === "loop") {
    const bodyRaw = o.body;
    const body: PipelineStep[] = Array.isArray(bodyRaw)
      ? (bodyRaw.map(reviveStep).filter(Boolean) as PipelineStep[])
      : [];
    let iterations: unknown[] = [{}];
    if (Array.isArray(o.iterations)) {
      iterations = normalizeLegacyIterations(o.iterations as unknown[]);
    }
    return { kind: "loop", iterations, body };
  }
  return null;
}

function sanitizeProgram(steps: PipelineStep[], nCells: number): PipelineStep[] {
  const out: PipelineStep[] = [];
  for (const s of steps) {
    const t = sanitizeStep(s, nCells);
    if (t) out.push(t);
  }
  return out;
}

function sanitizeStep(s: PipelineStep, nCells: number): PipelineStep | null {
  if (s.kind === "cell") {
    if (s.index < 0 || s.index >= nCells) return null;
    return s;
  }
  const body = sanitizeProgram(s.body, nCells);
  let iterations = s.iterations;
  if (!Array.isArray(iterations) || iterations.length === 0) iterations = [{}];
  return { kind: "loop", iterations, body };
}

/** Load a single pipeline from legacy v2/v1 storage (used only for migration). */
function loadSinglePipelineLegacy(nCells: number): PipelineStep[] {
  try {
    const v2 = localStorage.getItem(pipelineStorageKeyV2());
    if (v2) {
      const arr = JSON.parse(v2) as unknown;
      if (Array.isArray(arr)) {
        const revived = arr.map(reviveStep).filter(Boolean) as PipelineStep[];
        return sanitizeProgram(revived, nCells);
      }
    }
    const v1 = localStorage.getItem(pipelineStorageKeyV1());
    if (v1) {
      const arr = JSON.parse(v1) as unknown;
      if (Array.isArray(arr)) {
        const flat = arr.map(Number).filter((n) => Number.isInteger(n));
        return sanitizeProgram(
          flat.map((index) => ({ kind: "cell" as const, index })),
          nCells,
        );
      }
    }
  } catch {
    /* ignore */
  }
  return [];
}

function loadPipelines(nCells: number): PipelineStep[][] {
  try {
    const raw = localStorage.getItem(pipelineStorageKeyV3());
    if (raw) {
      const data = JSON.parse(raw) as unknown;
      if (Array.isArray(data) && data.length > 0 && Array.isArray(data[0])) {
        return (data as unknown[][]).map((arr) =>
          sanitizeProgram(
            (Array.isArray(arr) ? arr : []).map(reviveStep).filter(Boolean) as PipelineStep[],
            nCells,
          ),
        );
      }
    }
  } catch {
    /* ignore */
  }
  return [loadSinglePipelineLegacy(nCells)];
}

function savePipeline() {
  try {
    localStorage.setItem(pipelineStorageKeyV3(), JSON.stringify(pipelines));
  } catch {
    /* ignore */
  }
}

/** After moving one pipeline index, ``perm[newIdx]`` = old pipeline index now at ``newIdx``. */
function pipelineStripPermAfterMove(from: number, to: number, n: number): number[] | null {
  if (!Number.isInteger(from) || !Number.isInteger(to) || n <= 0) return null;
  if (from < 0 || from >= n || to < 0 || to > n) return null;
  const order = Array.from({ length: n }, (_, i) => i);
  const [x] = order.splice(from, 1);
  const insertAt = from < to ? to - 1 : to;
  order.splice(insertAt, 0, x);
  return order;
}

function remapLoopExpandedIndices(perm: number[]) {
  const next = new Set<string>();
  for (const key of loopConfigExpanded) {
    const m = /^(\d+):(.*)$/.exec(key);
    if (!m) continue;
    const oldP = Number(m[1]);
    const pathJson = m[2]!;
    if (!Number.isInteger(oldP)) continue;
    const newP = perm.indexOf(oldP);
    if (newP < 0) continue;
    next.add(loopExpandedKey(newP, pathJson));
  }
  loopConfigExpanded.clear();
  for (const k of next) loopConfigExpanded.add(k);
}

function remapActivePipelineAbortForStripReorder(perm: number[]) {
  const next = new Map<number, AbortController>();
  for (const [oldP, ac] of activePipelineAbortControllers) {
    const newP = perm.indexOf(oldP);
    if (newP >= 0) next.set(newP, ac);
  }
  activePipelineAbortControllers.clear();
  for (const [k, v] of next) activePipelineAbortControllers.set(k, v);
}

/** Move pipeline row ``from`` to sit at slot ``to`` (0 = top, ``n`` = after last). */
function movePipelineStrip(from: number, to: number): boolean {
  const n = pipelines.length;
  if (n <= 1) return false;
  const perm = pipelineStripPermAfterMove(from, to, n);
  if (!perm) return false;
  const [row] = pipelines.splice(from, 1);
  const insertAt = from < to ? to - 1 : to;
  pipelines.splice(insertAt, 0, row);
  remapLoopExpandedIndices(perm);
  remapActivePipelineAbortForStripReorder(perm);
  return true;
}

/** Toast duration; new messages reset the timer. */
const STATUS_TOAST_MS = 4000;
let statusHideTimer = 0;

function setStatus(msg: string) {
  statusToastEl.textContent = msg;
  statusToastEl.classList.add("status-toast--visible");
  if (statusHideTimer) window.clearTimeout(statusHideTimer);
  statusHideTimer = window.setTimeout(() => {
    statusHideTimer = 0;
    statusToastEl.classList.remove("status-toast--visible");
    window.setTimeout(() => {
      if (!statusToastEl.classList.contains("status-toast--visible")) {
        statusToastEl.textContent = "";
      }
    }, 220);
  }, STATUS_TOAST_MS);
}

/** Inline z-index rises so the last-touched / running cell stacks above siblings (default z-index is CSS). */
let cellZStackCounter = 10;
function bringCellToFront(cell: HTMLElement) {
  cellZStackCounter += 1;
  cell.style.zIndex = String(cellZStackCounter);
}

/** Cell indices currently executing (mirrors `cell-running` on the canvas; pipeline chips sync on rerender). */
const runningCellIndices = new Set<number>();

/** Heuristic: cooperative abort only runs if user code calls ``check_abort`` (name may be wrong for helpers / imports). */
function sourceLikelyHasCheckAbort(source: string | undefined): boolean {
  if (!source) return false;
  return /\bcheck_abort\b/.test(source);
}

function anyRunningCellLikelySupportsAbort(): boolean {
  for (const idx of runningCellIndices) {
    const src = lastCells.find((c) => c.index === idx)?.source;
    if (sourceLikelyHasCheckAbort(src)) return true;
  }
  return false;
}

function syncAbortButton() {
  btnAbort.disabled =
    runningCellIndices.size === 0 || !anyRunningCellLikelySupportsAbort();
}

/** In-progress stdout/stderr merged from WebSocket `run_stream` (lost on `renderCells` unless restored). */
const cellStreamBufferByIndex = new Map<number, string>();
/** Dedupe ``── Cell N · title ──`` line once per run (mirrored into Console with streamed output). */
const cellConsoleRunHeaderLogged = new Set<number>();

function applyPipelineChipRunningClassesForIndex(index: number) {
  const stack = document.getElementById("pipelines-stack");
  if (!stack) return;
  const on = runningCellIndices.has(index);
  stack
    .querySelectorAll<HTMLElement>(`.pipeline-chip.pipeline-chip-cell[data-cell-index="${index}"]`)
    .forEach((chip) => {
      chip.classList.toggle("pipeline-chip-running", on);
    });
}

function applyAllPipelineChipRunningClasses() {
  const stack = document.getElementById("pipelines-stack");
  if (!stack) return;
  stack.querySelectorAll<HTMLElement>(".pipeline-chip.pipeline-chip-cell").forEach((chip) => {
    const ci = Number(chip.dataset.cellIndex);
    chip.classList.toggle(
      "pipeline-chip-running",
      Number.isInteger(ci) && runningCellIndices.has(ci),
    );
  });
}

function setCellRunningState(index: number, running: boolean) {
  if (running) runningCellIndices.add(index);
  else {
    runningCellIndices.delete(index);
    cellStreamBufferByIndex.delete(index);
    cellConsoleRunHeaderLogged.delete(index);
  }

  const cell = cellsCanvas.querySelector<HTMLElement>(`.cell[data-pipeline-cell-drag="${index}"]`);
  if (cell) {
    cell.classList.toggle("cell-running", running);
    if (running) bringCellToFront(cell);
  }
  applyPipelineChipRunningClassesForIndex(index);
  syncAbortButton();
}


/** Clear output and show the output strip so streamed stdout/stderr can appear while the cell runs. */
function prepareCellStreamUi(index: number) {
  cellStreamBufferByIndex.set(index, "");
  const outEl = cellsEl.querySelector<HTMLElement>(`[data-out="${index}"]`);
  if (outEl) {
    outEl.textContent = "";
    outEl.className = "out out-streaming";
  }
  setCellOutputBlockVisible(index, true);
  syncCellCompactClassForIndex(index);
  scheduleLayoutAndLines();
}

function appendCellRunHeaderToConsole(index: number) {
  if (cellConsoleRunHeaderLogged.has(index)) return;
  const meta = lastCells.find((c) => c.index === index);
  const title = (meta?.title ?? "").trim().replace(/\s+/g, " ") || `Cell ${index}`;
  appendAppLogChunk(`\n── Cell ${index} · ${title} ──\n`);
  cellConsoleRunHeaderLogged.add(index);
}

function appendCellStreamChunk(index: number, text: string) {
  const next = foldCarriageReturns((cellStreamBufferByIndex.get(index) ?? "") + text);
  cellStreamBufferByIndex.set(index, next);
  const outEl = cellsEl.querySelector<HTMLElement>(`[data-out="${index}"]`);
  if (outEl) outEl.textContent = next;
  appendAppLogChunk(text);
}

/** File watcher can broadcast `cells` during a long run; `renderCells` rebuilds DOM and drops live stream text + running state until restored. */
function restoreLiveExecutionUiAfterCellRender(index: number) {
  if (!runningCellIndices.has(index)) return;
  const text = cellStreamBufferByIndex.get(index) ?? "";
  setCellOutputBlockVisible(index, true);
  const outEl = cellsEl.querySelector<HTMLElement>(`[data-out="${index}"]`);
  if (outEl) {
    outEl.className = "out out-streaming";
    outEl.textContent = text;
  }
  syncCellCompactClassForIndex(index);
  /* Re-apply full running chrome (yellow outline, z-index, pipeline chips) on new nodes. */
  setCellRunningState(index, true);
}

function applyCellsFromServer(
  data: {
    revision: number;
    path: string | null;
    cells: readonly unknown[] | Cell[];
    changed_cell_indices?: unknown;
    repo_root?: unknown;
  },
  opts?: { forceResetOutputs?: boolean },
) {
  applyRepoRootFromPayload(data);
  const incomingPath = data.path ?? null;
  const pathChanged = (incomingPath ?? "") !== (lastPath ?? "");
  if (pathChanged || opts?.forceResetOutputs) {
    outputs.clear();
    cellStdoutPlainText.clear();
    staleCells.clear();
    lastRevisionStaleMerged = -1;
  }
  revision = data.revision;
  const cells = Array.isArray(data.cells) ? data.cells.map(cellFromApiPayload) : [];
  if (pathChanged) {
    const outPersistKey = persistPathKey(incomingPath ?? (pathInput.value.trim() || null));
    if (outPersistKey) loadOutputsForPath(outPersistKey, cells);
  }
  if (!pathChanged && !opts?.forceResetOutputs) {
    pruneStaleCells(cells.length);
  }
  pruneOutputsForRemovedCells(cells);
  const changed = data.changed_cell_indices;
  const revKey =
    typeof data.revision === "number" && Number.isFinite(data.revision) ? data.revision : Number(data.revision) || 0;
  /* Apply server ``changed_cell_indices`` only once per new disk revision; resends (WS reconnect / duplicate
   * broadcast) must not re-add cells the user already cleared by running. */
  if (revKey > lastRevisionStaleMerged) {
    lastRevisionStaleMerged = revKey;
    if (Array.isArray(changed)) {
      for (const x of changed) {
        const i = Number(x);
        if (Number.isInteger(i) && i >= 0 && i < cells.length) staleCells.add(i);
      }
    }
  }
  try {
    renderCells(cells, incomingPath);
    schedulePersistOutputs();
  } catch (err) {
    console.error("stonesoup: renderCells failed", err);
    setStatus(`Cell UI error: ${String(err)}`);
  }
}

let kernelVarsRefreshTimer: ReturnType<typeof setTimeout> | null = null;

function kernelVarsStartExpanded(): boolean {
  const v = localStorage.getItem(KERNEL_VARS_EXPANDED_KEY);
  if (v === null) return true;
  return v === "1";
}

function scheduleKernelVarsRefresh() {
  if (kernelVarsRefreshTimer != null) window.clearTimeout(kernelVarsRefreshTimer);
  kernelVarsRefreshTimer = window.setTimeout(() => {
    kernelVarsRefreshTimer = null;
    void fetchKernelVars();
  }, 120);
}

type KernelSessionChip = { path: string; n_vars: number; current: boolean };

/** Max characters of basename for non-current entries in the `{ }` chip (then `…`). */
const KERNEL_CHIP_OTHER_BASENAME_MAX = 10;

function basenameOfRepoPath(path: string): string {
  const slash = path.lastIndexOf("/");
  return slash >= 0 ? path.slice(slash + 1) : path;
}

function truncateChipBasename(name: string, maxChars: number): string {
  if (name.length <= maxChars) return name;
  return `${name.slice(0, Math.max(1, maxChars - 1))}…`;
}

function renderKernelVarsChipSessions(sessions: KernelSessionChip[]) {
  kernelVarsSessions.replaceChildren();
  if (!sessions.length) return;
  sessions.forEach((s, i) => {
    if (i > 0) {
      const sep = document.createElement("span");
      sep.className = "kernel-vars-chip-sep";
      sep.setAttribute("aria-hidden", "true");
      sep.textContent = "·";
      kernelVarsSessions.appendChild(sep);
    }
    const line = document.createElement("span");
    line.className =
      "kernel-vars-chip-session" + (s.current ? " kernel-vars-chip-session--current" : "");
    const base = basenameOfRepoPath(s.path);
    const label = s.current ? base : truncateChipBasename(base, KERNEL_CHIP_OTHER_BASENAME_MAX);
    line.textContent = `${label} · ${s.n_vars}`;
    line.title = `${s.path} — ${s.n_vars} variable${s.n_vars === 1 ? "" : "s"}`;
    kernelVarsSessions.appendChild(line);
  });
}

async function fetchKernelVars() {
  try {
    const r = await fetch(`${apiBase}/api/kernel/vars`);
    const j = (await r.json()) as {
      vars?: { name: string; type: string; preview: string }[];
      sessions?: KernelSessionChip[];
    };
    if (!r.ok) return;
    const rows = Array.isArray(j.vars) ? j.vars : [];
    const sessions = Array.isArray(j.sessions) ? j.sessions : [];
    renderKernelVarsChipSessions(sessions);
    const n = rows.length;
    kernelVarsTbody.replaceChildren();
    for (const row of rows) {
      const tr = document.createElement("tr");
      const tdName = document.createElement("td");
      tdName.className = "kernel-vars-name";
      tdName.textContent = row.name;
      const tdType = document.createElement("td");
      tdType.className = "kernel-vars-type";
      tdType.textContent = row.type;
      const tdPrev = document.createElement("td");
      tdPrev.className = "kernel-vars-preview";
      tdPrev.textContent = row.preview;
      tr.append(tdName, tdType, tdPrev);
      kernelVarsTbody.appendChild(tr);
    }
    const table = kernelVarsTbody.closest("table");
    if (table) table.hidden = n === 0;
    kernelVarsEmpty.hidden = n > 0;
  } catch {
    /* ignore */
  }
}

async function loadModelsFromToolbar() {
  let items: { repo_id: string }[];
  try {
    items = parseHfRepoIds(modelRepoInput.value);
  } catch (err) {
    setStatus(String(err));
    return;
  }
  if (!items.length) {
    setStatus("Enter a Hugging Face repo id (or several, comma-separated)");
    return;
  }
  setModelLoaderBusy(true);
  try {
    const r = await fetch(`${apiBase}/api/models/load`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ items }),
    });
    const j = (await readApiJson(r)) as {
      detail?: string;
      accepted?: boolean;
      loaded?: LoadedModelInfo[];
      loaded_now?: LoadedModelInfo[];
    };
    if (!r.ok) throw new Error(j.detail || r.statusText);
    await fetchKernelVars();
    if (j.accepted) {
      setStatus(
        "Loading model in background — see console. Cell runs queue until load finishes.",
      );
      void refreshModelRepoSuggestions();
      modelRepoInput.value = "";
      return;
    }
    const loadedNow = Array.isArray(j.loaded) ? j.loaded : [];
    const repos = loadedNow.map((item) => item.repo_id);
    if (repos.length) {
      setStatus(`Loaded ${repos.length === 1 ? "" : `${repos.length} models: `}${repos.join(", ")}`);
      rememberHfRepoIdsLoaded(repos);
      void refreshModelRepoSuggestions();
      modelRepoInput.value = "";
    } else {
      setStatus("Models loaded");
    }
  } catch (err) {
    setStatus(String(err));
  } finally {
    setModelLoaderBusy(false);
    void fetchLoadedModels();
  }
}

async function unloadSelectedModelFromToolbar() {
  const poolKeyB64 = modelsLoadedSelect.value;
  if (!poolKeyB64) {
    setStatus("Select a loaded model in the list");
    return;
  }
  setModelLoaderBusy(true);
  try {
    const r = await fetch(`${apiBase}/api/models/unload`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ pool_keys_b64: [poolKeyB64] }),
    });
    const j = (await readApiJson(r)) as {
      detail?: string;
      unloaded?: LoadedModelInfo[];
      loaded_now?: LoadedModelInfo[];
    };
    if (!r.ok) throw new Error(j.detail || r.statusText);
    await fetchKernelVars();
    const unloaded = Array.isArray(j.unloaded) ? j.unloaded : [];
    const label = unloaded[0]?.repo_id ?? "checkpoint";
    setStatus(
      unloaded.length
        ? `Unloaded ${label} globally (${unloaded.length} binding${unloaded.length === 1 ? "" : "s"})`
        : "That checkpoint had no bindings in any open experiment",
    );
  } catch (err) {
    setStatus(String(err));
  } finally {
    setModelLoaderBusy(false);
    void fetchLoadedModels();
  }
}

async function unloadAllModelsFromToolbar() {
  setModelLoaderBusy(true);
  try {
    const r = await fetch(`${apiBase}/api/models/unload`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ names: null }),
    });
    const j = (await readApiJson(r)) as {
      detail?: string;
      unloaded?: LoadedModelInfo[];
      loaded_now?: LoadedModelInfo[];
    };
    if (!r.ok) throw new Error(j.detail || r.statusText);
    await fetchKernelVars();
    const unloaded = Array.isArray(j.unloaded) ? j.unloaded.length : 0;
    setStatus(
      unloaded
        ? `Unloaded ${unloaded} binding${unloaded === 1 ? "" : "s"} globally`
        : "No Stonesoup model bindings in any open experiment",
    );
  } catch (err) {
    setStatus(String(err));
  } finally {
    setModelLoaderBusy(false);
    void fetchLoadedModels();
  }
}

function isVariablesPanelExpanded(): boolean {
  return !kernelVarsDock.classList.contains("collapsed");
}

/** Show or hide the Variables bottom pane (shared dock with Console; only one expanded at a time). */
function setVariablesPanelExpanded(expanded: boolean) {
  if (expanded) {
    setConsoleExpanded(false);
  }
  kernelVarsDock.classList.toggle("collapsed", !expanded);
  kernelVarsToggle.setAttribute("aria-expanded", expanded ? "true" : "false");
  localStorage.setItem(KERNEL_VARS_EXPANDED_KEY, expanded ? "1" : "0");
  if (expanded) void fetchKernelVars();
}

/** Apply prefs; if both saved open, Console wins (single bottom pane). */
function initBottomDock() {
  const prefConsole = localStorage.getItem(STONESOUP_CONSOLE_EXPANDED_KEY) === "1";
  let prefVars = kernelVarsStartExpanded();
  if (prefConsole && prefVars) {
    localStorage.setItem(KERNEL_VARS_EXPANDED_KEY, "0");
    prefVars = false;
  }
  setVariablesPanelExpanded(prefVars);
  setConsoleExpanded(prefConsole);
}

kernelVarsToggle.addEventListener("click", () => {
  setVariablesPanelExpanded(!isVariablesPanelExpanded());
});

kernelVarsCollapse.addEventListener("click", () => {
  setVariablesPanelExpanded(false);
});

kernelVarsRefresh.addEventListener("click", () => void fetchKernelVars());

async function clearVariablesCurrentExperimentKernel() {
  try {
    const r = await fetch(`${apiBase}/api/kernel/free-memory`, { method: "POST" });
    const j = (await readApiJson(r)) as {
      detail?: string;
      cleared?: unknown[];
    };
    if (!r.ok) throw new Error(j.detail || r.statusText);
    await fetchKernelVars();
    void fetchLoadedModels();
    const n = Array.isArray(j.cleared) ? j.cleared.length : 0;
    setStatus(
      n > 0
        ? `Cleared ${n} variable${n === 1 ? "" : "s"} in this experiment`
        : "No user variables to clear",
    );
  } catch (e) {
    setStatus(String(e));
  }
}

kernelVarsFreeMemory.addEventListener("click", () => void clearVariablesCurrentExperimentKernel());

function trimConsoleBuffer(s: string): string {
  if (s.length <= CONSOLE_BUFFER_MAX) return s;
  return s.slice(s.length - CONSOLE_BUFFER_MAX);
}

function renderConsoleBuffer() {
  stonesoupConsolePre.textContent = appLogBuffer;
  if (stonesoupConsoleRoot.classList.contains("stonesoup-console--expanded")) {
    stonesoupConsolePre.scrollTop = stonesoupConsolePre.scrollHeight;
  }
}

function schedulePersistConsoleBuffer() {
  window.clearTimeout(saveConsoleTimer);
  saveConsoleTimer = window.setTimeout(() => {
    saveConsoleTimer = 0;
    try {
      localStorage.setItem(STONESOUP_CONSOLE_LOG_LS_KEY, appLogBuffer);
    } catch {
      /* quota or private mode */
    }
  }, 300);
}

function appendAppLogChunk(text: string) {
  if (!text) return;
  appLogBuffer = trimConsoleBuffer(foldCarriageReturns(appLogBuffer + text));
  renderConsoleBuffer();
  schedulePersistConsoleBuffer();
}

function clearConsoleBuffer() {
  appLogBuffer = "";
  stonesoupConsolePre.textContent = "";
  schedulePersistConsoleBuffer();
}

function loadPersistedConsoleBuffer() {
  try {
    const raw = localStorage.getItem(STONESOUP_CONSOLE_LOG_LS_KEY);
    if (typeof raw === "string" && raw.length > 0) {
      appLogBuffer = trimConsoleBuffer(foldCarriageReturns(raw));
    }
  } catch {
    /* ignore */
  }
  renderConsoleBuffer();
}

function setConsoleExpanded(expanded: boolean) {
  if (expanded) {
    setVariablesPanelExpanded(false);
  }
  stonesoupConsoleRoot.classList.toggle("stonesoup-console--expanded", expanded);
  localStorage.setItem(STONESOUP_CONSOLE_EXPANDED_KEY, expanded ? "1" : "0");
  btnConsoleToggle.setAttribute("aria-expanded", expanded ? "true" : "false");
  if (expanded) {
    stonesoupConsolePre.scrollTop = stonesoupConsolePre.scrollHeight;
  }
}

btnConsoleToggle.addEventListener("click", () => {
  setConsoleExpanded(!stonesoupConsoleRoot.classList.contains("stonesoup-console--expanded"));
});

btnConsoleCollapse.addEventListener("click", () => {
  setConsoleExpanded(false);
});

btnConsoleClear.addEventListener("click", () => {
  clearConsoleBuffer();
});

stonesoupConsolePanel.addEventListener("click", async (e) => {
  const t = e.target as HTMLElement;
  if (t.closest("button")) return;
  const text = appLogBuffer.replace(/\s+$/, "");
  if (!text) {
    setStatus("Console is empty");
    return;
  }
  try {
    await navigator.clipboard.writeText(text);
    setStatus("Console copied");
    stonesoupConsolePre.classList.remove("stonesoup-console-pre--copied");
    void stonesoupConsolePre.offsetWidth;
    stonesoupConsolePre.classList.add("stonesoup-console-pre--copied");
    window.setTimeout(() => stonesoupConsolePre.classList.remove("stonesoup-console-pre--copied"), 500);
  } catch {
    setStatus("Copy failed (clipboard permission?)");
  }
});

loadPersistedConsoleBuffer();

/** Keepalive for dev proxies (e.g. Vite ``/ws``) that drop idle WebSockets. */
let wsKeepaliveTimer: ReturnType<typeof setInterval> | null = null;
/** Pending auto-reconnect from ``onclose``; cleared on every ``connectWs()`` so explicit reconnects (e.g. after reset) are not stomped. */
let wsReconnectTimer: ReturnType<typeof setTimeout> | null = null;

function connectWs() {
  if (wsReconnectTimer != null) {
    window.clearTimeout(wsReconnectTimer);
    wsReconnectTimer = null;
  }
  if (wsKeepaliveTimer != null) {
    window.clearInterval(wsKeepaliveTimer);
    wsKeepaliveTimer = null;
  }
  ws?.close();
  const sock = new WebSocket(wsUrl());
  ws = sock;
  sock.onopen = () => {
    wsDot.classList.add("on");
    wsDot.title = "Live reload: connected";
    wsDot.setAttribute("aria-label", "WebSocket connected");
    wsKeepaliveTimer = window.setInterval(() => {
      if (ws === sock && sock.readyState === WebSocket.OPEN) {
        sock.send("ping");
      }
    }, 25000);
  };
  sock.onclose = () => {
    /** ``connectWs()`` already replaced ``ws``; do not schedule another reconnect for this dead socket. */
    if (ws !== sock) return;
    if (wsKeepaliveTimer != null) {
      window.clearInterval(wsKeepaliveTimer);
      wsKeepaliveTimer = null;
    }
    wsDot.classList.remove("on");
    wsDot.title = "Live reload: disconnected (retrying…)";
    wsDot.setAttribute("aria-label", "WebSocket disconnected");
    wsReconnectTimer = window.setTimeout(() => {
      wsReconnectTimer = null;
      connectWs();
    }, 2000);
  };
  sock.onmessage = (ev) => {
    try {
      const data = JSON.parse(ev.data as string) as {
        type?: string;
        cell_index?: number;
        stream?: string;
        text?: string;
      };
      if (data.type === "cells") {
        const cells = (data as { cells?: Cell[] }).cells;
        if (!Array.isArray(cells)) return;
        applyCellsFromServer({
          revision: (data as { revision?: number }).revision ?? revision,
          path: (data as { path?: string | null }).path ?? null,
          cells,
          changed_cell_indices: (data as { changed_cell_indices?: unknown }).changed_cell_indices,
          repo_root: (data as { repo_root?: unknown }).repo_root,
        });
        setStatus(`rev ${revision} · ${cells.length} cells`);
      } else if (data.type === "run_start") {
        const ci = Number(data.cell_index);
        if (!Number.isInteger(ci)) return;
        setCellRunningState(ci, true);
        prepareCellStreamUi(ci);
      } else if (data.type === "run_stream") {
        const ci = Number(data.cell_index);
        const t = typeof data.text === "string" ? data.text : "";
        if (Number.isInteger(ci) && t) appendCellStreamChunk(ci, t);
      } else if (data.type === "run_end") {
        const ci = Number(data.cell_index);
        if (Number.isInteger(ci) && (data as { ok?: boolean }).ok) {
          staleCells.delete(ci);
          syncCellStaleClassForIndex(ci);
          renderPipelineBar();
        }
        scheduleKernelVarsRefresh();
        // Cell code can call ``stonesoup.load_model`` without the HTTP model-load path,
        // so re-fetch the HF bundle list for the toolbar (same as ``app_log_end`` /models).
        void fetchLoadedModels();
      } else if (data.type === "app_log_start") {
        const op =
          typeof (data as { op?: string }).op === "string"
            ? (data as { op: string }).op
            : "";
        const sep =
          op === "model_load"
            ? "\n\n── model load ──\n"
            : op === "model_unload"
              ? "\n\n── model unload ──\n"
              : "\n\n── log ──\n";
        appLogBuffer = trimConsoleBuffer(appLogBuffer + sep);
        renderConsoleBuffer();
        schedulePersistConsoleBuffer();
      } else if (data.type === "app_log") {
        const t = typeof data.text === "string" ? data.text : "";
        appendAppLogChunk(t);
      } else if (data.type === "app_log_end") {
        const ok = Boolean((data as { ok?: boolean }).ok);
        const errRaw = (data as { error?: string }).error;
        const err = typeof errRaw === "string" ? errRaw : "";
        const op =
          typeof (data as { op?: string }).op === "string" ? (data as { op: string }).op : "";
        if (!ok && err) {
          appendAppLogChunk(`\n[error] ${err}\n`);
        }
        scheduleKernelVarsRefresh();
        if (op === "model_load") {
          void fetchLoadedModels();
          if (ok) {
            setStatus("Model load finished");
            void refreshModelRepoSuggestions();
          } else {
            setStatus(
              err
                ? `Model load failed — ${err.length > 120 ? `${err.slice(0, 120)}…` : err}`
                : "Model load failed",
            );
          }
        } else if (op === "model_unload") {
          void fetchLoadedModels();
          if (ok) {
            setStatus("Model unload finished");
          } else {
            setStatus(
              err
                ? `Model unload failed — ${err.length > 120 ? `${err.slice(0, 120)}…` : err}`
                : "Model unload failed",
            );
          }
        }
      }
    } catch {
      /* ignore */
    }
  };
}

/** Matplotlib `tab20` listed colormap (dark/light pairs per hue family). */
const TAB20: readonly string[] = [
  "#1f77b4",
  "#aec7e8",
  "#ff7f0e",
  "#ffbb78",
  "#2ca02c",
  "#98df8a",
  "#d62728",
  "#ff9896",
  "#9467bd",
  "#c5b0d5",
  "#8c564b",
  "#c49c94",
  "#e377c2",
  "#f7b6d2",
  "#7f7f7f",
  "#c7c7c7",
  "#bcbd22",
  "#dbdb8d",
  "#17becf",
  "#9edae5",
];

/** FNV-1a 32-bit hash — stable tab20 pick from cell title (not list index). */
function hashStringToUint32(s: string): number {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

function tab20AccentFromKey(accentKey: string): string {
  return TAB20[hashStringToUint32(accentKey) % TAB20.length]!;
}

/** Same key as pipeline chips: ``meta?.title ?? `Cell ${index}` `` (empty title hashes as empty string). */
function cellAccentKey(index: number): string {
  return lastCells.find((c) => c.index === index)?.title ?? `Cell ${index}`;
}

function applyCellColorVars(el: HTMLElement, accentKey: string) {
  el.style.setProperty("--cell-accent", tab20AccentFromKey(accentKey));
}

/** View zoom for the cell canvas (logical layout unchanged; wrapper size × scale for scroll extents). */
let cellsViewScale = 1;
const CELLS_ZOOM_MIN = 0.35;
const CELLS_ZOOM_MAX = 2.5;

function applyCellsZoomLayout() {
  const s = cellsViewScale;
  const w = Math.max(1, cellsCanvas.offsetWidth);
  const h = Math.max(1, cellsCanvas.offsetHeight);
  cellsZoomWrap.style.width = `${Math.ceil(w * s)}px`;
  cellsZoomWrap.style.height = `${Math.ceil(h * s)}px`;
  if (s === 1) {
    cellsCanvas.style.transform = "";
    cellsCanvas.style.transformOrigin = "";
  } else {
    cellsCanvas.style.transform = `scale(${s})`;
    cellsCanvas.style.transformOrigin = "0 0";
  }
}

function relayoutCanvasBounds() {
  const padX = 48;
  /** Extra space below the lowest cell so the canvas can scroll vertically. */
  const padBottom = 420;
  let maxBottom = 0;
  let maxRight = 0;
  cellsCanvas.querySelectorAll<HTMLElement>(".cell").forEach((c) => {
    maxBottom = Math.max(maxBottom, c.offsetTop + c.offsetHeight);
    maxRight = Math.max(maxRight, c.offsetLeft + c.offsetWidth);
  });
  const minH = Math.max(
    maxBottom + padBottom,
    cellsEl.clientHeight + 240,
  );
  const minW = Math.max(maxRight + padX, cellsEl.clientWidth);
  cellsCanvas.style.minHeight = `${minH}px`;
  cellsCanvas.style.minWidth = `${minW}px`;
  applyCellsZoomLayout();
}

/** Per watched file: zoom scale and ``.cells`` scroll offsets (pan position). */
const CELLS_VIEW_LS_KEY = "stonesoup_cells_view_v1";
type CellsViewSnapshot = { scale: number; scrollLeft: number; scrollTop: number };

function parseCellsViewStore(): Record<string, CellsViewSnapshot> {
  try {
    const raw = localStorage.getItem(CELLS_VIEW_LS_KEY);
    if (!raw) return {};
    const x = JSON.parse(raw) as unknown;
    if (x && typeof x === "object" && !Array.isArray(x)) return x as Record<string, CellsViewSnapshot>;
  } catch {
    /* ignore */
  }
  return {};
}

function writeCellsViewStore(all: Record<string, CellsViewSnapshot>) {
  try {
    localStorage.setItem(CELLS_VIEW_LS_KEY, JSON.stringify(all));
  } catch {
    /* quota */
  }
}

function readCellsViewForPath(pathKey: string): CellsViewSnapshot | null {
  if (!pathKey || pathKey === "_unset") return null;
  const raw = parseCellsViewStore()[pathKey];
  if (!raw || typeof raw !== "object") return null;
  const scale = Number(raw.scale);
  const scrollLeft = Number(raw.scrollLeft);
  const scrollTop = Number(raw.scrollTop);
  if (!Number.isFinite(scale) || !Number.isFinite(scrollLeft) || !Number.isFinite(scrollTop)) return null;
  return { scale, scrollLeft, scrollTop };
}

let cellsViewSaveTimer = 0;
function schedulePersistCellsView() {
  window.clearTimeout(cellsViewSaveTimer);
  cellsViewSaveTimer = window.setTimeout(() => {
    cellsViewSaveTimer = 0;
    const pathKey = layoutStoragePath();
    if (!pathKey || pathKey === "_unset") return;
    const all = parseCellsViewStore();
    all[pathKey] = {
      scale: cellsViewScale,
      scrollLeft: cellsEl.scrollLeft,
      scrollTop: cellsEl.scrollTop,
    };
    writeCellsViewStore(all);
  }, 200);
}

/** Restore zoom + scroll for ``pathKey`` after layout; returns false if nothing saved. */
function applySavedCellsView(pathKey: string): boolean {
  const saved = readCellsViewForPath(pathKey);
  if (!saved) return false;
  cellsViewScale = Math.min(CELLS_ZOOM_MAX, Math.max(CELLS_ZOOM_MIN, saved.scale));
  applyCellsZoomLayout();
  relayoutCanvasBounds();
  const sl = saved.scrollLeft;
  const st = saved.scrollTop;
  requestAnimationFrame(() => {
    const maxL = Math.max(0, cellsEl.scrollWidth - cellsEl.clientWidth);
    const maxT = Math.max(0, cellsEl.scrollHeight - cellsEl.clientHeight);
    cellsEl.scrollLeft = Math.min(Math.max(0, sl), maxL);
    cellsEl.scrollTop = Math.min(Math.max(0, st), maxT);
    requestAnimationFrame(() => {
      cellsEl.scrollLeft = Math.min(Math.max(0, sl), maxL);
      cellsEl.scrollTop = Math.min(Math.max(0, st), maxT);
      schedulePersistCellsView();
    });
  });
  return true;
}

/** Run after grid positions and ``relayoutCanvasBounds`` (same rAF chain as layout). */
function consumePendingAutoLayoutScroll(): void {
  if (!pendingScrollAfterAutoLayout) return;
  pendingScrollAfterAutoLayout = false;
  const g = CELLS_PAN_GUTTER_PX;
  relayoutCanvasBounds();
  applyCellsZoomLayout();
  requestAnimationFrame(() => {
    const maxL = Math.max(0, cellsEl.scrollWidth - cellsEl.clientWidth);
    const maxT = Math.max(0, cellsEl.scrollHeight - cellsEl.clientHeight);
    cellsEl.scrollLeft = Math.min(g, maxL);
    cellsEl.scrollTop = Math.min(g, maxT);
    requestAnimationFrame(() => {
      const maxL2 = Math.max(0, cellsEl.scrollWidth - cellsEl.clientWidth);
      const maxT2 = Math.max(0, cellsEl.scrollHeight - cellsEl.clientHeight);
      cellsEl.scrollLeft = Math.min(g, maxL2);
      cellsEl.scrollTop = Math.min(g, maxT2);
      schedulePersistCellsView();
    });
  });
}

function scheduleScrollAfterAutoLayoutIfNeeded(): void {
  if (!pendingScrollAfterAutoLayout) return;
  requestAnimationFrame(() => {
    consumePendingAutoLayoutScroll();
  });
}

let cellOutputMaxOverlayEl: HTMLDivElement | null = null;
/** Zoom factor for expanded output (1 = 100%). */
let cellOutputMaxZoom = 1;
const CELL_OUTPUT_MAX_ZOOM_MIN = 0.5;
const CELL_OUTPUT_MAX_ZOOM_MAX = 2.5;
const CELL_OUTPUT_MAX_ZOOM_STEP = 1.1;

function onCellOutputMaxOverlayKeydown(e: KeyboardEvent) {
  if (e.key === "Escape") closeCellOutputMaxOverlay();
}

function applyCellOutputMaxZoomToDom() {
  const el = cellOutputMaxOverlayEl?.querySelector<HTMLElement>(".cell-output-max-scroll");
  if (!el) return;
  const z = Math.min(CELL_OUTPUT_MAX_ZOOM_MAX, Math.max(CELL_OUTPUT_MAX_ZOOM_MIN, cellOutputMaxZoom));
  cellOutputMaxZoom = z;
  el.style.zoom = String(z);
}

function cellOutputMaxZoomIn() {
  cellOutputMaxZoom *= CELL_OUTPUT_MAX_ZOOM_STEP;
  applyCellOutputMaxZoomToDom();
}

function cellOutputMaxZoomOut() {
  cellOutputMaxZoom /= CELL_OUTPUT_MAX_ZOOM_STEP;
  applyCellOutputMaxZoomToDom();
}

function ensureCellOutputMaxOverlay(): HTMLDivElement {
  if (cellOutputMaxOverlayEl) return cellOutputMaxOverlayEl;
  const root = document.createElement("div");
  root.id = "cell-output-max-overlay";
  root.className = "cell-output-max-overlay";
  root.hidden = true;
  root.setAttribute("role", "dialog");
  root.setAttribute("aria-modal", "true");
  root.setAttribute("aria-labelledby", "cell-output-max-heading");
  root.innerHTML = `
    <div class="cell-output-max-backdrop" aria-hidden="true"></div>
    <div class="cell-output-max-dialog">
      <div class="cell-output-max-toolbar">
        <h2 class="cell-output-max-heading" id="cell-output-max-heading">Output</h2>
        <div class="cell-output-max-toolbar-actions">
          <button type="button" class="cell-output-max-zoom btn-icon" data-cell-output-zoom="out" title="Zoom out" aria-label="Zoom out">−</button>
          <button type="button" class="cell-output-max-zoom btn-icon" data-cell-output-zoom="in" title="Zoom in" aria-label="Zoom in">+</button>
          <button type="button" class="cell-output-max-close btn-icon" aria-label="Close expanded output">✕</button>
        </div>
      </div>
      <div class="cell-output-max-scroll">
        <div class="cell-output-max-content"></div>
      </div>
    </div>
  `;
  document.body.appendChild(root);
  const backdrop = root.querySelector(".cell-output-max-backdrop")!;
  const closeBtn = root.querySelector(".cell-output-max-close")!;
  backdrop.addEventListener("click", () => closeCellOutputMaxOverlay());
  closeBtn.addEventListener("click", () => closeCellOutputMaxOverlay());
  root.querySelectorAll<HTMLButtonElement>("[data-cell-output-zoom]").forEach((btn) => {
    btn.addEventListener("click", (ev) => {
      ev.stopPropagation();
      const dir = btn.dataset.cellOutputZoom;
      if (dir === "in") cellOutputMaxZoomIn();
      else if (dir === "out") cellOutputMaxZoomOut();
    });
  });
  cellOutputMaxOverlayEl = root;
  return root;
}

function closeCellOutputMaxOverlay() {
  const el = cellOutputMaxOverlayEl;
  if (!el || el.hidden) return;
  el.hidden = true;
  document.body.style.overflow = "";
  document.removeEventListener("keydown", onCellOutputMaxOverlayKeydown);
}

function openCellOutputMaximize(cellIndex: number) {
  const outEl = cellsEl.querySelector<HTMLElement>(`[data-out="${cellIndex}"]`);
  if (!outEl) return;
  const t = lastCells.find((c) => c.index === cellIndex)?.title?.trim();
  const overlay = ensureCellOutputMaxOverlay();
  const heading = overlay.querySelector(".cell-output-max-heading")!;
  heading.textContent = t ? `Cell ${cellIndex} — ${t}` : `Cell ${cellIndex}`;
  const contentHost = overlay.querySelector<HTMLDivElement>(".cell-output-max-content")!;
  const raw = outEl.innerHTML.trim();
  cellOutputMaxZoom = 2;
  if (!raw) {
    contentHost.innerHTML = '<p class="cell-output-max-empty">No output yet.</p>';
  } else {
    const wrap = document.createElement("div");
    wrap.className = outEl.className;
    wrap.innerHTML = outEl.innerHTML;
    contentHost.replaceChildren(wrap);
  }
  applyCellOutputMaxZoomToDom();
  overlay.hidden = false;
  document.body.style.overflow = "hidden";
  document.addEventListener("keydown", onCellOutputMaxOverlayKeydown);
  overlay.querySelector<HTMLButtonElement>(".cell-output-max-close")?.focus();
}

function flattenCellIndices(steps: PipelineStep[]): number[] {
  const out: number[] = [];
  for (const s of steps) {
    if (s.kind === "cell") out.push(s.index);
    else out.push(...flattenCellIndices(s.body));
  }
  return out;
}

function getParentList(prefix: number[], pIdx: number): PipelineStep[] | null {
  const root = pipelines[pIdx];
  if (!root) return null;
  if (prefix.length === 0) return root;
  const first = prefix[0]!;
  const rest = prefix.slice(1);
  const step0 = root[first];
  if (!step0 || step0.kind !== "loop") return null;
  let body = step0.body;
  for (let d = 0; d < rest.length; d++) {
    const idx = rest[d]!;
    const inner = body[idx];
    if (!inner || inner.kind !== "loop") return null;
    body = inner.body;
  }
  return body;
}

function resolveListIndex(
  path: number[],
  pIdx: number,
): { list: PipelineStep[]; index: number } | null {
  if (path.length === 0) return null;
  const prefix = path.slice(0, -1);
  const index = path[path.length - 1]!;
  const list = getParentList(prefix, pIdx);
  if (!list || index < 0 || index >= list.length) return null;
  return { list, index };
}

function stepAtPath(path: number[], pIdx: number): PipelineStep | null {
  const r = resolveListIndex(path, pIdx);
  return r ? r.list[r.index]! : null;
}

/** List where pipeline steps are inserted: root, or a loop's `body`. */
function getBodyListForDrop(loopPath: number[] | null, pIdx: number): PipelineStep[] | null {
  if (loopPath === null) return pipelines[pIdx] ?? null;
  const st = stepAtPath(loopPath, pIdx);
  if (!st || st.kind !== "loop") return null;
  return st.body;
}

function insertCellInPipeline(
  cellIndex: number,
  bodyLoopPath: number[] | null,
  at: number,
  pIdx: number,
) {
  if (cellIndex < 0 || cellIndex >= lastCells.length) return;
  const list = getBodyListForDrop(bodyLoopPath, pIdx);
  if (!list) return;
  const n = Math.max(0, Math.min(at, list.length));
  list.splice(n, 0, { kind: "cell", index: cellIndex });
}

function insertNewLoopInPipeline(bodyLoopPath: number[] | null, at: number, pIdx: number) {
  const list = getBodyListForDrop(bodyLoopPath, pIdx);
  if (!list) return;
  const n = Math.max(0, Math.min(at, list.length));
  const loop: PipelineStep = { kind: "loop", iterations: [{}], body: [] };
  list.splice(n, 0, loop);
}

/** True if drop target list is inside the subtree of the loop at `movedLoopPath` (invalid move). */
function isDropInsideMovedLoop(movedLoopPath: number[], bodyLoopPath: number[] | null): boolean {
  if (bodyLoopPath === null) return false;
  if (bodyLoopPath.length < movedLoopPath.length) return false;
  for (let i = 0; i < movedLoopPath.length; i++) {
    if (bodyLoopPath[i] !== movedLoopPath[i]) return false;
  }
  return true;
}

/** Move a cell or loop step from `fromPath` into list `getBodyListForDrop(bodyLoopPath, toPIdx)` at index `at`. */
function movePipelineStep(
  fromPath: number[],
  fromPIdx: number,
  bodyLoopPath: number[] | null,
  toPIdx: number,
  at: number,
) {
  const fromRes = resolveListIndex(fromPath, fromPIdx);
  if (!fromRes) return;
  const step = fromRes.list[fromRes.index];
  if (!step) return;
  if (
    fromPIdx === toPIdx &&
    step.kind === "loop" &&
    isDropInsideMovedLoop(fromPath, bodyLoopPath)
  ) {
    return;
  }
  const toList = getBodyListForDrop(bodyLoopPath, toPIdx);
  if (!toList) return;
  const fromList = fromRes.list;
  const fromIdx = fromRes.index;
  let insertAt = Math.max(0, Math.min(at, toList.length));

  if (toList === fromList) {
    if (fromIdx === insertAt) return;
    fromList.splice(fromIdx, 1);
    if (fromIdx < insertAt) insertAt -= 1;
    toList.splice(insertAt, 0, step);
  } else {
    fromList.splice(fromIdx, 1);
    insertAt = Math.max(0, Math.min(insertAt, toList.length));
    toList.splice(insertAt, 0, step);
  }
}

const DND_PAYLOAD = "text/plain";

type DndPayload =
  | { kind: "canvas"; cellIndex: number }
  /** Cell or loop step at this path in the pipeline tree */
  | { kind: "move"; fromPath: number[]; fromPipeline: number }
  /** Reorder entire pipeline strip (row) */
  | { kind: "reorder_strip"; fromPipeline: number };

function loopRunCount(step: PipelineStep & { kind: "loop" }): number {
  const n = step.iterations?.length ?? 0;
  return n < 1 ? 1 : n;
}

function findLoopTextarea(pathJson: string, pIdx: number): HTMLTextAreaElement | null {
  const stack = document.getElementById("pipelines-stack");
  if (!stack) return null;
  for (const ta of stack.querySelectorAll<HTMLTextAreaElement>("textarea[data-loop-path]")) {
    if (ta.dataset.loopPath === pathJson && Number(ta.dataset.loopPipeline) === pIdx) return ta;
  }
  return null;
}

/** Parse open textarea into model; returns error message or null. */
function applyLoopEditorToModel(pathJson: string, pIdx: number): string | null {
  const path = JSON.parse(pathJson) as number[];
  const st = stepAtPath(path, pIdx);
  if (!st || st.kind !== "loop") return null;
  const ta = findLoopTextarea(pathJson, pIdx);
  if (!ta) return null;
  const parsed = parseIterationsJson(ta.value);
  if (!parsed.ok) return parsed.error;
  st.iterations = parsed.iterations;
  savePipeline();
  return null;
}

const PIPELINE_RETURN_SVG =
  '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 12 14" focusable="false"><path d="M7.5 1.5v8.2M7.5 9.7H3.2M5 8l-1.8 1.7L5 11.5" fill="none" stroke="currentColor" stroke-width="1.35" stroke-linecap="round" stroke-linejoin="round"/></svg>';

const PIPELINE_FLOW_NEXT_SVG =
  '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 14 8" focusable="false"><path d="M1 4h7.5M7 2.2L9.5 4 7 5.8" fill="none" stroke="currentColor" stroke-width="1.35" stroke-linecap="round" stroke-linejoin="round"/></svg>';

/**
 * Root strip only: flex-grow gap with drop zone full-bleed and → centered (vertically + in spare width).
 */
function appendFlowBetweenSteps(
  parent: HTMLElement,
  at: number,
  dropLoopAttr: string,
  pIdx: number,
) {
  const wrap = document.createElement("span");
  wrap.className = "pipeline-flow-between";
  const dz = document.createElement("div");
  dz.className = "pipeline-drop-zone pipeline-drop-zone--between";
  dz.dataset.dropLoop = dropLoopAttr;
  dz.dataset.dropAt = String(at);
  dz.dataset.dropPipeline = String(pIdx);
  dz.title = "Drop here (cell or loop)";
  const sep = document.createElement("span");
  sep.className = "pipeline-flow-sep pipeline-flow-sep-next";
  sep.setAttribute("aria-hidden", "true");
  sep.innerHTML = PIPELINE_FLOW_NEXT_SVG;
  wrap.append(dz, sep);
  parent.appendChild(wrap);
}

function createPipelineWrapReturnEl(): HTMLElement {
  const span = document.createElement("span");
  span.className = "pipeline-flow-sep pipeline-flow-sep-return pipeline-flow-sep-wrap";
  span.setAttribute("aria-hidden", "true");
  span.innerHTML = PIPELINE_RETURN_SVG;
  return span;
}

/** L-shaped “continued from line above” marker at the start of wrapped rows. */
function createPipelineContinueEl(): HTMLElement {
  const span = document.createElement("span");
  span.className = "pipeline-flow-sep pipeline-flow-continue";
  span.setAttribute("aria-hidden", "true");
  span.innerHTML =
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 14 20" focusable="false"><g fill="none" stroke="currentColor" stroke-width="1.35" stroke-linecap="round" stroke-linejoin="round"><path d="M5 1.5v11"/><path d="M5 12.5h6"/><path d="M9 10.5l2.2 2-2.2 2"/></g></svg>';
  return span;
}

function isPipelineLayoutInjected(el: HTMLElement): boolean {
  return (
    el.classList.contains("pipeline-flow-wrap-bridge") ||
    el.classList.contains("pipeline-flow-continue")
  );
}

/**
 * Group flex children into visual rows. Do not use `offsetTop`: with
 * `align-items: center`, siblings on the same line often have different
 * offsetTop and were mis-split into fake “wrapped” rows.
 */
function clusterPipelineFlowRows(children: HTMLElement[]): HTMLElement[][] {
  if (children.length === 0) return [];
  const rows: HTMLElement[][] = [];
  let cur: HTMLElement[] = [children[0]!];
  let rowMaxBottom = children[0]!.getBoundingClientRect().bottom;
  /** Same flex line ⇔ vertical overlap with the row so far (not offsetTop — center-aligned items differ). */
  const subpixelSlop = 2;
  for (let i = 1; i < children.length; i++) {
    const el = children[i]!;
    const r = el.getBoundingClientRect();
    if (r.top < rowMaxBottom + subpixelSlop) {
      cur.push(el);
      rowMaxBottom = Math.max(rowMaxBottom, r.bottom);
    } else {
      rows.push(cur);
      cur = [el];
      rowMaxBottom = r.bottom;
    }
  }
  rows.push(cur);
  return rows;
}

/**
 * After each visual row except the last: full-width line with ↵ on the right.
 * Before each continuation row: “from previous line” glyph (indented via CSS).
 */
function updatePipelineLineBreakMarkers() {
  const stack = document.getElementById("pipelines-stack");
  if (!stack) return;
  for (const flow of stack.querySelectorAll<HTMLElement>(".pipeline-chips-flow")) {
    flow.querySelectorAll(".pipeline-flow-wrap-bridge, .pipeline-flow-continue").forEach((el) => {
      el.remove();
    });

    const children = [...flow.children].filter(
      (el): el is HTMLElement => el instanceof HTMLElement && !isPipelineLayoutInjected(el),
    );
    if (children.length < 2) continue;

    const rows = clusterPipelineFlowRows(children);
    if (rows.length < 2) continue;

    for (let r = 0; r < rows.length - 1; r++) {
      const firstNext = rows[r + 1]![0]!;
      const bridge = document.createElement("div");
      bridge.className = "pipeline-flow-wrap-bridge";
      bridge.setAttribute("aria-hidden", "true");
      bridge.appendChild(createPipelineWrapReturnEl());
      flow.insertBefore(bridge, firstNext);
      flow.insertBefore(createPipelineContinueEl(), firstNext);
    }
  }
}

function renderPipelineBar() {
  const stack = document.getElementById("pipelines-stack");
  if (!stack) return;
  stack.replaceChildren();

  const renderLevel = (
    parent: HTMLElement,
    steps: PipelineStep[],
    pathPrefix: number[],
    bodyLoopPath: number[] | null,
    pIdx: number,
  ) => {
    const dropLoopAttr = bodyLoopPath === null ? "" : JSON.stringify(bodyLoopPath);
    const appendDropZone = (at: number) => {
      const dz = document.createElement("div");
      dz.className = "pipeline-drop-zone";
      dz.dataset.dropLoop = dropLoopAttr;
      dz.dataset.dropAt = String(at);
      dz.dataset.dropPipeline = String(pIdx);
      dz.title = "Drop here (cell or loop)";
      parent.appendChild(dz);
    };

    const renderStepAt = (i: number) => {
      const step = steps[i]!;
      const path = [...pathPrefix, i];
      const pathJson = JSON.stringify(path);
      if (step.kind === "cell") {
        const idx = step.index;
        const meta = lastCells.find((c) => c.index === idx);
        const title = meta?.title ?? `Cell ${idx}`;
        const chip = document.createElement("span");
        chip.className = "pipeline-chip pipeline-chip-cell";
        if (staleCells.has(idx)) chip.classList.add("pipeline-chip-stale");
        applyCellColorVars(chip, title);
        chip.draggable = true;
        chip.dataset.pipelineChipDrag = pathJson;
        chip.dataset.cellIndex = String(idx);
        chip.title = `${title} — drag to reorder in pipeline`;
        const idxSpan = document.createElement("span");
        idxSpan.className = "chip-idx";
        idxSpan.textContent = String(idx);
        idxSpan.title = "Cell index in file (0-based), same as on the canvas";
        const titleSpan = document.createElement("span");
        titleSpan.className = "chip-title";
        titleSpan.title = title;
        const shortTitle =
          [...title].length > 6
            ? `${[...title].slice(0, 6).join("")}…`
            : title;
        titleSpan.textContent = shortTitle;
        const bRm = document.createElement("button");
        bRm.type = "button";
        bRm.dataset.pPath = pathJson;
        bRm.dataset.pPipeline = String(pIdx);
        bRm.dataset.pRemove = "1";
        bRm.title = "Remove";
        bRm.textContent = "×";
        chip.append(idxSpan, titleSpan, bRm);
        parent.appendChild(chip);
      } else {
        const nRuns = loopRunCount(step);
        const expKey = loopExpandedKey(pIdx, pathJson);
        const expanded = loopConfigExpanded.has(expKey);
        const wrap = document.createElement("div");
        wrap.className = "pipeline-nest";
        const head = document.createElement("div");
        head.className = "pipeline-nest-head";
        const loopGrip = document.createElement("span");
        loopGrip.className = "pipeline-loop-drag";
        loopGrip.draggable = true;
        loopGrip.dataset.pipelineLoopDrag = pathJson;
        loopGrip.title = "Drag to move loop in pipeline";
        loopGrip.textContent = "⠿";
        loopGrip.setAttribute("aria-hidden", "true");
        const bToggle = document.createElement("button");
        bToggle.type = "button";
        bToggle.className = "pipeline-nest-toggle";
        bToggle.dataset.loopTogglePath = pathJson;
        bToggle.dataset.loopPipeline = String(pIdx);
        bToggle.title = expanded
          ? "Hide iteration JSON (applies edits)"
          : "Edit loop iterations (JSON array)";
        const lab = document.createElement("span");
        lab.className = "pipeline-nest-label";
        lab.textContent = "↻ Loop";
        const countEl = document.createElement("span");
        countEl.className = "pipeline-nest-count";
        countEl.textContent = `${nRuns}×`;
        countEl.title = `${nRuns} pass${nRuns === 1 ? "" : "es"} through loop body`;
        const chev = document.createElement("span");
        chev.className = "pipeline-nest-chevron";
        chev.textContent = expanded ? "▾" : "▸";
        chev.setAttribute("aria-hidden", "true");
        bToggle.append(lab, countEl, chev);
        const bRm = document.createElement("button");
        bRm.type = "button";
        bRm.className = "pipeline-nest-remove";
        bRm.dataset.removeLoopPath = pathJson;
        bRm.dataset.removeLoopPipeline = String(pIdx);
        bRm.title = "Remove entire loop";
        bRm.textContent = "×";
        head.append(loopGrip, bToggle, bRm);

        const bodyWrap = document.createElement("div");
        bodyWrap.className = "pipeline-nest-body";
        renderLevel(bodyWrap, step.body, path, path, pIdx);
        wrap.append(head, bodyWrap);
        if (expanded) {
          const config = document.createElement("div");
          config.className = "pipeline-nest-config";
          const ta = document.createElement("textarea");
          ta.className = "pipeline-nest-iter";
          ta.rows = 4;
          ta.spellcheck = false;
          ta.dataset.loopPath = pathJson;
          ta.dataset.loopPipeline = String(pIdx);
          ta.value = iterationsToJson(step.iterations);
          ta.title = "Example: [1, 2, 3] or [{\"lr\": 0.01}, {\"lr\": 0.1}]";
          const hint = document.createElement("span");
          hint.className = "pipeline-nest-config-hint";
          hint.textContent =
            "One array element = one pass. Numbers/strings/lists use LOOP_ITEM and LOOP_INDEX in cells; objects add each key as a global plus LOOP_INDEX.";
          const bDone = document.createElement("button");
          bDone.type = "button";
          bDone.className = "pipeline-nest-done";
          bDone.dataset.loopDonePath = pathJson;
          bDone.dataset.loopPipeline = String(pIdx);
          bDone.textContent = "Done";
          bDone.title = "Apply JSON and collapse";
          config.append(ta, hint, bDone);
          wrap.append(config);
        }
        parent.appendChild(wrap);
      }
    };

    if (bodyLoopPath !== null) {
      for (let i = 0; i <= steps.length; i++) {
        appendDropZone(i);
        if (i >= steps.length) break;
        renderStepAt(i);
      }
    } else if (steps.length === 0) {
      appendDropZone(0);
    } else {
      appendDropZone(0);
      for (let i = 0; i < steps.length; i++) {
        renderStepAt(i);
        if (i < steps.length - 1) {
          appendFlowBetweenSteps(parent, i + 1, dropLoopAttr, pIdx);
        }
      }
      appendDropZone(steps.length);
    }
  };

  const nStrips = pipelines.length;
  for (let slot = 0; slot <= nStrips; slot++) {
    const stripZone = document.createElement("div");
    stripZone.className = "pipeline-strip-drop-zone";
    stripZone.dataset.stripReorderAt = String(slot);
    stripZone.title = "Drop here to place this pipeline row";
    stack.appendChild(stripZone);

    if (slot >= nStrips) break;

    const pIdx = slot;
    const program = pipelines[pIdx]!;
    const block = document.createElement("div");
    block.className = "pipeline-block";

    const shell = document.createElement("div");
    shell.className = "pipeline-chips";
    shell.dataset.pipelineIndex = String(pIdx);
    if (pipelines.length > 1) {
      shell.classList.add("pipeline-chips--reorderable");
      shell.draggable = true;
      shell.dataset.pipelineStripFrom = String(pIdx);
      shell.title = "Drag from an empty area of this row to reorder pipelines";
    }

    const flowScroll = document.createElement("div");
    flowScroll.className = "pipeline-chips-flow-scroll";
    const flow = document.createElement("div");
    flow.className = "pipeline-chips-flow";
    renderLevel(flow, program, [], null, pIdx);
    flowScroll.appendChild(flow);

    const actions = document.createElement("div");
    actions.className = "pipeline-block-actions";
    const bRun = document.createElement("button");
    bRun.type = "button";
    bRun.className = "primary btn-icon";
    bRun.dataset.pipelineRun = String(pIdx);
    bRun.title = "Run this pipeline (stops on first error)";
    bRun.setAttribute("aria-label", "Run pipeline");
    bRun.textContent = "▶";
    const bAbort = document.createElement("button");
    bAbort.type = "button";
    bAbort.className = "btn-icon";
    bAbort.dataset.pipelineAbort = String(pIdx);
    bAbort.title = "Abort pipeline (stops before the next cell)";
    bAbort.setAttribute("aria-label", "Abort pipeline");
    bAbort.disabled = true;
    bAbort.textContent = "⏹";
    const bAddAll = document.createElement("button");
    bAddAll.type = "button";
    bAddAll.className = "btn-icon";
    bAddAll.dataset.pipelineAddAll = String(pIdx);
    bAddAll.title =
      "Chain: append every file cell not already in this pipeline (indices 0…n−1 in order at the end)";
    bAddAll.setAttribute("aria-label", "Chain all missing cells from file into this pipeline");
    bAddAll.innerHTML =
      '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"/><path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"/></svg>';
    bAddAll.disabled = lastCells.length === 0;
    const bClr = document.createElement("button");
    bClr.type = "button";
    bClr.className = "btn-icon";
    bClr.dataset.pipelineClear = String(pIdx);
    bClr.title = "Clear this pipeline";
    bClr.setAttribute("aria-label", "Clear pipeline");
    bClr.textContent = "✕";
    actions.append(bRun, bAbort, bAddAll, bClr);

    shell.append(flowScroll, actions);
    block.appendChild(shell);

    const stripTools = document.createElement("div");
    stripTools.className = "pipeline-block-strip-tools";
    if (pipelines.length > 1) {
      const bStripRm = document.createElement("button");
      bStripRm.type = "button";
      bStripRm.className = "btn-icon pipeline-strip-remove";
      bStripRm.dataset.removePipelineStrip = String(pIdx);
      bStripRm.draggable = false;
      bStripRm.title = "Remove this pipeline";
      bStripRm.setAttribute("aria-label", "Remove pipeline");
      bStripRm.textContent = "−";
      stripTools.appendChild(bStripRm);
    }
    if (pIdx === pipelines.length - 1) {
      const bAdd = document.createElement("button");
      bAdd.type = "button";
      bAdd.className = "btn-icon pipeline-add-more";
      bAdd.draggable = false;
      bAdd.title = "Add another pipeline";
      bAdd.setAttribute("aria-label", "Add pipeline");
      bAdd.textContent = "＋";
      bAdd.addEventListener("click", () => {
        pipelines.push([]);
        savePipeline();
        renderPipelineBar();
        highlightPipelineCells();
      });
      stripTools.appendChild(bAdd);
    }
    if (stripTools.childElementCount > 0) block.appendChild(stripTools);

    stack.appendChild(block);
  }

  stack.querySelectorAll<HTMLButtonElement>("[data-remove-pipeline-strip]").forEach((btn) => {
    btn.addEventListener("click", (ev) => {
      ev.stopPropagation();
      const i = Number(btn.dataset.removePipelineStrip);
      if (pipelines.length <= 1 || !Number.isInteger(i)) return;
      pipelines.splice(i, 1);
      clearLoopExpanded();
      savePipeline();
      renderPipelineBar();
      highlightPipelineCells();
    });
  });

  stack.querySelectorAll<HTMLButtonElement>("[data-pipeline-run]").forEach((btn) => {
    btn.addEventListener("click", (ev) => {
      ev.stopPropagation();
      const i = Number(btn.dataset.pipelineRun);
      if (!Number.isInteger(i)) return;
      void runPipeline(i);
    });
  });

  stack.querySelectorAll<HTMLButtonElement>("[data-pipeline-abort]").forEach((btn) => {
    btn.addEventListener("click", (ev) => {
      ev.stopPropagation();
      const i = Number(btn.dataset.pipelineAbort);
      if (!Number.isInteger(i)) return;
      activePipelineAbortControllers.get(i)?.abort();
    });
  });

  stack.querySelectorAll<HTMLButtonElement>("[data-pipeline-clear]").forEach((btn) => {
    btn.addEventListener("click", (ev) => {
      ev.stopPropagation();
      const i = Number(btn.dataset.pipelineClear);
      if (!Number.isInteger(i) || !pipelines[i]) return;
      pipelines[i] = [];
      clearLoopExpanded();
      savePipeline();
      renderPipelineBar();
      highlightPipelineCells();
    });
  });

  stack.querySelectorAll<HTMLButtonElement>("[data-pipeline-add-all]").forEach((btn) => {
    btn.addEventListener("click", (ev) => {
      ev.stopPropagation();
      const i = Number(btn.dataset.pipelineAddAll);
      if (!Number.isInteger(i)) return;
      appendMissingFileCellsToPipeline(i);
    });
  });

  stack.querySelectorAll<HTMLButtonElement>("[data-p-remove]").forEach((btn) => {
    btn.addEventListener("click", (ev) => {
      ev.stopPropagation();
      const path = JSON.parse(btn.dataset.pPath!) as number[];
      const pIdx = Number(btn.dataset.pPipeline);
      const r = resolveListIndex(path, pIdx);
      if (!r) return;
      r.list.splice(r.index, 1);
      clearLoopExpanded();
      savePipeline();
      renderPipelineBar();
      highlightPipelineCells();
    });
  });

  stack.querySelectorAll<HTMLButtonElement>("[data-remove-loop-path]").forEach((btn) => {
    btn.addEventListener("click", (ev) => {
      ev.stopPropagation();
      const path = JSON.parse(btn.dataset.removeLoopPath!) as number[];
      const pIdx = Number(btn.dataset.removeLoopPipeline);
      const r = resolveListIndex(path, pIdx);
      if (!r) return;
      r.list.splice(r.index, 1);
      clearLoopExpanded();
      savePipeline();
      renderPipelineBar();
      highlightPipelineCells();
    });
  });

  stack.querySelectorAll<HTMLButtonElement>("[data-loop-toggle-path]").forEach((btn) => {
    btn.addEventListener("click", (ev) => {
      ev.stopPropagation();
      const pathJson = btn.dataset.loopTogglePath!;
      const pIdx = Number(btn.dataset.loopPipeline);
      const expKey = loopExpandedKey(pIdx, pathJson);
      if (loopConfigExpanded.has(expKey)) {
        const err = applyLoopEditorToModel(pathJson, pIdx);
        if (err) {
          setStatus(err);
          return;
        }
        loopConfigExpanded.delete(expKey);
        setStatus("Loop iterations saved");
      } else {
        loopConfigExpanded.add(expKey);
      }
      renderPipelineBar();
      highlightPipelineCells();
      if (loopConfigExpanded.has(expKey)) {
        requestAnimationFrame(() => {
          findLoopTextarea(pathJson, pIdx)?.focus();
        });
      }
    });
  });

  stack.querySelectorAll<HTMLButtonElement>("[data-loop-done-path]").forEach((btn) => {
    btn.addEventListener("click", (ev) => {
      ev.stopPropagation();
      const pathJson = btn.dataset.loopDonePath!;
      const pIdx = Number(btn.dataset.loopPipeline);
      const expKey = loopExpandedKey(pIdx, pathJson);
      const err = applyLoopEditorToModel(pathJson, pIdx);
      if (err) {
        setStatus(err);
        return;
      }
      loopConfigExpanded.delete(expKey);
      setStatus("Loop iterations saved");
      renderPipelineBar();
      highlightPipelineCells();
    });
  });

  stack.querySelectorAll<HTMLTextAreaElement>("[data-loop-path]").forEach((ta) => {
    ta.addEventListener("change", () => {
      const path = JSON.parse(ta.dataset.loopPath!) as number[];
      const pIdx = Number(ta.dataset.loopPipeline);
      const st = stepAtPath(path, pIdx);
      if (!st || st.kind !== "loop") return;
      const parsed = parseIterationsJson(ta.value);
      if (!parsed.ok) {
        setStatus(parsed.error);
        ta.value = iterationsToJson(st.iterations);
        return;
      }
      st.iterations = parsed.iterations;
      savePipeline();
      setStatus("Loop iterations updated");
    });
  });

  syncPipelineAbortButtons();

  applyAllPipelineChipRunningClasses();

  requestAnimationFrame(() => {
    updatePipelineLineBreakMarkers();
  });
}

function highlightPipelineCells() {
  const inPipe = new Set<number>();
  for (const p of pipelines) {
    for (const idx of flattenCellIndices(p)) inPipe.add(idx);
  }
  cellsCanvas.querySelectorAll<HTMLElement>(".cell[data-pipeline-cell-drag]").forEach((el) => {
    const i = Number(el.dataset.pipelineCellDrag);
    if (Number.isInteger(i)) el.classList.toggle("pipeline-step", inPipe.has(i));
  });
}

const CELL_PIPELINE_CHIP_HOVER = "cell-pipeline-chip-hover";

function clearPipelineChipCanvasHover() {
  cellsCanvas.querySelectorAll<HTMLElement>(`.cell.${CELL_PIPELINE_CHIP_HOVER}`).forEach((el) => {
    el.classList.remove(CELL_PIPELINE_CHIP_HOVER);
  });
}

function setPipelineChipCanvasHover(idx: number) {
  clearPipelineChipCanvasHover();
  const cell = cellsCanvas.querySelector<HTMLElement>(`.cell[data-pipeline-cell-drag="${idx}"]`);
  cell?.classList.add(CELL_PIPELINE_CHIP_HOVER);
}

let pipelineChipCanvasHoverBound = false;
function bindPipelineChipCanvasHover() {
  if (pipelineChipCanvasHoverBound) return;
  pipelineChipCanvasHoverBound = true;
  const stack = document.getElementById("pipelines-stack");
  if (!stack) return;
  stack.addEventListener("mouseover", (e) => {
    const chip = (e.target as Element).closest<HTMLElement>(".pipeline-chip.pipeline-chip-cell");
    if (!chip || !stack.contains(chip)) return;
    const ci = Number(chip.dataset.cellIndex);
    if (!Number.isInteger(ci)) return;
    setPipelineChipCanvasHover(ci);
  });
  stack.addEventListener("mouseout", (e) => {
    const chip = (e.target as Element).closest<HTMLElement>(".pipeline-chip.pipeline-chip-cell");
    if (!chip || !stack.contains(chip)) return;
    const rel = e.relatedTarget as Element | null;
    if (rel && chip.contains(rel)) return;
    if (rel && rel.closest(".pipeline-chip.pipeline-chip-cell")) return;
    clearPipelineChipCanvasHover();
  });
}

function appendToPipeline(idx: number) {
  const last = pipelines[pipelines.length - 1];
  if (!last) return;
  last.push({ kind: "cell", index: idx });
  savePipeline();
  renderPipelineBar();
  highlightPipelineCells();
}

/** Append cells 0..n-1 in file order at the end of this pipeline’s root, skipping indices already present anywhere in the tree. */
function appendMissingFileCellsToPipeline(pIdx: number) {
  if (!Number.isInteger(pIdx) || !pipelines[pIdx]) return;
  const n = lastCells.length;
  if (n === 0) {
    setStatus("No cells loaded");
    return;
  }
  const program = pipelines[pIdx]!;
  const present = new Set(flattenCellIndices(program));
  let added = 0;
  for (let i = 0; i < n; i++) {
    if (!present.has(i)) {
      program.push({ kind: "cell", index: i });
      present.add(i);
      added++;
    }
  }
  if (added === 0) {
    setStatus(`Pipeline ${pIdx + 1}: every cell is already in this pipeline`);
    return;
  }
  savePipeline();
  renderPipelineBar();
  highlightPipelineCells();
  setStatus(`Pipeline ${pIdx + 1}: added ${added} cell${added === 1 ? "" : "s"} (file order)`);
}

function scheduleLayoutAndLines() {
  requestAnimationFrame(() => {
    relayoutCanvasBounds();
  });
}

function computeCellGridParams(): { pad: number; gap: number; cellW: number; cols: number } {
  const pad = 12;
  const gap = 28;
  /** Cell geometry is in canvas logical px; viewport is CSS px on the scaled zoom-wrap → divide by zoom. */
  const s = cellsViewScale > 0 ? cellsViewScale : 1;
  const viewportW = cellsEl.clientWidth / s;
  const usableW = Math.max(240, viewportW - 2 * pad);
  let cellW = Math.min(520, Math.max(260, Math.min(440, usableW)));
  let cols = Math.max(1, Math.floor((usableW + gap) / (cellW + gap)));
  cellW = Math.min(520, Math.max(240, Math.floor((usableW - (cols - 1) * gap) / cols)));
  return { pad, gap, cellW, cols };
}

/** Stack rows using measured heights. */
function packGridRows(
  nodes: HTMLElement[],
  cols: number,
  cellW: number,
  pad: number,
  gap: number,
) {
  const n = nodes.length;
  let y = pad;
  for (let r = 0; r * cols < n; r++) {
    let rowMax = 0;
    for (let c = 0; c < cols; c++) {
      const i = r * cols + c;
      if (i >= n) break;
      const cell = nodes[i]!;
      cell.style.left = `${pad + c * (cellW + gap)}px`;
      cell.style.top = `${y}px`;
      rowMax = Math.max(rowMax, cell.offsetHeight);
    }
    y += rowMax + gap;
  }
}

function syncCellPositionsFromDom(nodes: HTMLElement[]) {
  nodes.forEach((cell, i) => {
    cellPositions.set(i, {
      left: parseFloat(cell.style.left) || 0,
      top: parseFloat(cell.style.top) || 0,
    });
  });
}

/** Pack cells into the default grid once, then snapshot into ``manualLayoutByCellIdx`` and cookie (stable across reload/live parse until user clicks auto-layout). */
function placeDefaultGridAndPersist() {
  const { pad, gap, cellW, cols } = computeCellGridParams();
  const nodes = [...cellsCanvas.querySelectorAll<HTMLElement>(".cell[data-pipeline-cell-drag]")];
  lastLayoutCols = cols;
  nodes.forEach((cell, i) => {
    cell.classList.remove("cell-custom-geometry");
    cell.dataset.cellIndex = String(i);
    cell.style.width = `${cellW}px`;
    cell.style.height = "";
    cell.style.minHeight = "";
    cell.style.left = `${pad + (i % cols) * (cellW + gap)}px`;
    cell.style.top = `${pad}px`;
  });
  requestAnimationFrame(() => {
    packGridRows(nodes, cols, cellW, pad, gap);
    syncCellPositionsFromDom(nodes);
    snapshotCurrentLayoutToManualMap();
    scheduleSaveCellLayouts();
    scheduleLayoutAndLines();
    scheduleScrollAfterAutoLayoutIfNeeded();
  });
}

function applyFloatingLayout() {
  const { pad, gap, cellW, cols } = computeCellGridParams();
  const nodes = [...cellsCanvas.querySelectorAll<HTMLElement>(".cell[data-pipeline-cell-drag]")];
  if (manualLayoutByCellIdx.size === 0 && nodes.length > 0) {
    if (!(needsDefaultCellGrid || pendingUserAutoLayout)) {
      loadManualLayoutsForPath(layoutStoragePath());
    }
    if (manualLayoutByCellIdx.size === 0) {
      placeDefaultGridAndPersist();
      needsDefaultCellGrid = false;
      pendingUserAutoLayout = false;
      return;
    }
    needsDefaultCellGrid = false;
    pendingUserAutoLayout = false;
  }

  lastLayoutCols = cols;
  nodes.forEach((cell, i) => {
    cell.dataset.cellIndex = String(i);
    const idx = Number(cell.dataset.pipelineCellDrag);
    const saved = Number.isInteger(idx) ? manualLayoutByCellIdx.get(idx) : undefined;
    if (saved) {
      cell.classList.add("cell-custom-geometry");
      cell.style.left = `${Math.max(0, saved.left)}px`;
      cell.style.top = `${Math.max(0, saved.top)}px`;
      cell.style.width = `${Math.max(CELL_LAYOUT_MIN_W, saved.width)}px`;
      /* Compact cards save a small height; do not floor to CELL_LAYOUT_MIN_H or every header-only cell becomes a tall empty box after any full re-layout. */
      const showOutForLayout = Number.isInteger(idx) ? cellHasExpandedOutputUi(idx) : false;
      const compact = Number.isInteger(idx) && !showOutForLayout;
      /* Do not set inline min-height: 0 here — it overrides `.cell-custom-geometry { min-height: 200px }` and
         sticks until the next full layout, so cells collapse after pipeline runs. Let classes control min-height. */
      if (compact) {
        cell.style.height = "";
        cell.style.minHeight = "";
      } else {
        cell.style.height = `${Math.max(CELL_LAYOUT_MIN_H, saved.height)}px`;
        cell.style.minHeight = "";
      }
    } else {
      cell.classList.remove("cell-custom-geometry");
      cell.style.width = `${cellW}px`;
      cell.style.height = "";
      cell.style.minHeight = "";
      cell.style.left = `${pad}px`;
      cell.style.top = `${pad}px`;
    }
  });

  requestAnimationFrame(() => {
    let stackY = pad;
    for (const cell of nodes) {
      const idx = Number(cell.dataset.pipelineCellDrag);
      if (!Number.isInteger(idx) || manualLayoutByCellIdx.has(idx)) {
        stackY = Math.max(stackY, cell.offsetTop + cell.offsetHeight + gap);
      }
    }
    for (const cell of nodes) {
      const idx = Number(cell.dataset.pipelineCellDrag);
      if (!Number.isInteger(idx) || manualLayoutByCellIdx.has(idx)) continue;
      cell.style.top = `${stackY}px`;
      stackY += cell.offsetHeight + gap;
    }
    syncCellPositionsFromDom(nodes);
    scheduleLayoutAndLines();
    scheduleScrollAfterAutoLayoutIfNeeded();
  });
}

function renderCells(cells: Cell[], path: string | null) {
  lastCells = cells;
  lastPath = path;
  const p = path ?? pathInput.value.trim();
  const pathForEditorLink = (path?.trim() || pathInput.value.trim()) || "";
  const pathChangedForScrollReset = p !== lastLayoutPath;
  if (p !== lastLayoutPath || cells.length !== lastLayoutCount) {
    cellPositions.clear();
    lastLayoutCols = -1;
    if (p !== lastLayoutPath) {
      if (!p) {
        manualLayoutByCellIdx.clear();
      } else {
        loadManualLayoutsForPath(layoutStoragePath());
      }
      pipelines = loadPipelines(cells.length);
      if (pipelines.length === 0) pipelines = [[]];
      clearLoopExpanded();
    } else {
      const valid = new Set(cells.map((c) => c.index));
      for (const k of [...manualLayoutByCellIdx.keys()]) {
        if (!valid.has(k)) manualLayoutByCellIdx.delete(k);
      }
      pipelines = pipelines.map((pl) => sanitizeProgram(pl, cells.length));
      clearLoopExpanded();
    }
    lastLayoutPath = p;
    lastLayoutCount = cells.length;
    if (cells.length > 0 && manualLayoutByCellIdx.size === 0) {
      needsDefaultCellGrid = true;
    }
  }

  const validMarkerKeys = new Set(cells.map((c) => c.marker_key).filter(Boolean));
  for (const k of [...cellRunInputDraft.keys()]) {
    if (!validMarkerKeys.has(k)) cellRunInputDraft.delete(k);
  }
  for (const c of cells) {
    if (!c.cell_input) cellRunInputDraft.delete(c.marker_key);
  }

  cellsCanvas.innerHTML = "";

  for (const c of cells) {
    const stale = staleCells.has(c.index);
    const div = document.createElement("div");
    div.className = "cell";
    div.draggable = false;
    div.dataset.pipelineCellDrag = String(c.index);
    div.title = stale
      ? "Source changed on disk — re-run to clear. Drag title bar: move on canvas or drop on pipeline · corner → resize"
      : "Drag title bar to move · drop on pipeline bar to add · corner to resize";
    if (stale) div.classList.add("cell-stale");
    applyCellColorVars(div, cellAccentKey(c.index));
    const prev = outputs.get(c.marker_key);
    const showOut = shouldRevealOutputStrip(prev);
    const outRendered = showOut && prev ? renderOutputInnerHtml(prev, c.marker_key) : null;
    const outRichClass = outRendered?.richLayout ? " out-rich" : "";
    if (!showOut) div.classList.add("cell-compact");
    const slRaw = c.start_line;
    const slNum = slRaw === undefined || slRaw === null ? NaN : Number(slRaw);
    const startLine =
      Number.isFinite(slNum) && slNum >= 1 ? Math.floor(slNum) : 1;
    const editorHref = pathForEditorLink
      ? cellEditorHref(pathForEditorLink, startLine)
      : "";
    const editorName = getEditorPref() === "vscode" ? "VS Code" : "Cursor";
    const codeControl =
      editorHref !== ""
        ? `<a class="toggle cell-cursor-link cell-cursor-link--icon" draggable="false" href="${escapeHtmlAttr(editorHref)}" title="Open this cell in ${editorName} (deeplink)" rel="noopener" aria-label="Open this cell in ${editorName}">${CELL_ICON_PYTHON}</a>`
        : `<span class="toggle cell-cursor-link cell-cursor-link--disabled cell-cursor-link--icon" title="Watch a file to open in ${editorName}" aria-label="Open in ${editorName} unavailable: watch a file first">${CELL_ICON_PYTHON}</span>`;
    const runInputHtml = c.cell_input
      ? `<input type="text" class="cell-run-input" draggable="false" data-run-input="${c.index}" placeholder="CELL_INPUT" spellcheck="false" title="Injected as CELL_INPUT · Ctrl+Enter or ⌘+Enter to run this cell" aria-label="Cell run input" />`
      : "";
    div.innerHTML = `
      <div class="cell-body">
        <div class="cell-head">
          <div class="cell-head-main">
            <span class="cell-idx" title="Cell index in file (0-based)">${c.index}</span>
            <span class="cell-updated-badge" draggable="false" ${stale ? "" : "hidden"} title="This cell's code changed on disk; run it to clear">Updated</span>
            <span class="cell-title">${escapeHtml(c.title)}</span>
          </div>
          <div class="cell-head-actions">
            ${codeControl}
            <button type="button" class="btn-chain btn-icon-cell" draggable="false" data-pipeline-add="${c.index}" title="Append to pipeline" aria-label="Append to pipeline">${CELL_ICON_ADD}</button>
            ${runInputHtml}
            <button type="button" class="btn-icon-cell btn-output-maximize" draggable="false" data-output-maximize="${c.index}" title="Expand output (full screen)" aria-label="Expand output">${CELL_ICON_OUTPUT_MAXIMIZE}</button>
            <button type="button" class="primary btn-icon-cell" draggable="false" data-run="${c.index}" title="Run this cell" aria-label="Run this cell">${CELL_ICON_RUN_CELL}</button>
          </div>
        </div>
        <div class="cell-output-block" data-output-block="${c.index}" style="display:${showOut ? "flex" : "none"}">
          <div class="out-label-row" draggable="false">
            <span class="out-label" draggable="false">Output</span>
            <div class="out-label-meta" draggable="false"></div>
          </div>
          <div class="out ${prev && !prev.ok ? "err" : prev ? "ok" : "out-pending"}${outRichClass}" draggable="false" data-out="${c.index}" data-marker-key="${escapeHtmlAttr(c.marker_key)}" title="Click to copy">${outRendered ? outRendered.html : ""}</div>
        </div>
      </div>
      <div class="cell-resize-handle" draggable="false" title="Drag corner to resize"></div>
    `;
    cellsCanvas.appendChild(div);
    syncOutLabelRowForCell(c.index);
    const runInp = div.querySelector<HTMLInputElement>(`[data-run-input="${c.index}"]`);
    if (runInp) {
      runInp.value = cellRunInputDraft.get(c.marker_key) ?? "";
      const stopHeadDrag = (e: Event) => e.stopPropagation();
      runInp.addEventListener("pointerdown", stopHeadDrag);
      runInp.addEventListener("mousedown", stopHeadDrag);
      runInp.addEventListener("input", () => cellRunInputDraft.set(c.marker_key, runInp.value));
      runInp.addEventListener("keydown", (e: KeyboardEvent) => {
        if (e.key !== "Enter" || (!e.ctrlKey && !e.metaKey)) return;
        e.preventDefault();
        void runCell(c.index);
      });
    }
    div.querySelectorAll<HTMLAnchorElement>("a.cell-cursor-link[href]").forEach((a) => {
      const stop = (e: Event) => e.stopPropagation();
      a.addEventListener("click", stop);
      a.addEventListener("pointerdown", stop);
    });
  }

  cellsCanvas.querySelectorAll("[data-run]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const idx = Number((btn as HTMLButtonElement).dataset.run);
      runCell(idx);
    });
  });
  cellsCanvas.querySelectorAll<HTMLButtonElement>("[data-pipeline-add]").forEach((btn) => {
    btn.addEventListener("click", (ev) => {
      ev.stopPropagation();
      appendToPipeline(Number(btn.dataset.pipelineAdd));
    });
  });
  cellsCanvas.querySelectorAll<HTMLButtonElement>("[data-output-maximize]").forEach((btn) => {
    btn.addEventListener("click", (ev) => {
      ev.stopPropagation();
      openCellOutputMaximize(Number(btn.dataset.outputMaximize));
    });
  });

  const loopPal = document.createElement("div");
  loopPal.className = "loop-palette loop-palette--slot";
  loopPal.title = "Drag into a pipeline row to insert a loop (short drag from the left)";
  loopPal.innerHTML = `
    <div class="loop-palette-head">
      <span class="loop-palette-grip" aria-hidden="true">⠿</span>
      <span class="loop-palette-label">↻ Loop</span>
    </div>
  `;
  loopPaletteSlot.replaceChildren(loopPal);

  applyFloatingLayout();
  renderPipelineBar();
  highlightPipelineCells();

  if (runningCellIndices.size > 0) {
    requestAnimationFrame(() => {
      for (const idx of [...runningCellIndices]) {
        restoreLiveExecutionUiAfterCellRender(idx);
      }
      applyFloatingLayout();
    });
  }

  if (pathChangedForScrollReset) {
    const g = CELLS_PAN_GUTTER_PX;
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        requestAnimationFrame(() => {
          if (layoutStoragePath() !== "_unset" && applySavedCellsView(layoutStoragePath())) {
            return;
          }
          cellsViewScale = 1;
          applyCellsZoomLayout();
          relayoutCanvasBounds();
          requestAnimationFrame(() => {
            cellsEl.scrollLeft = g;
            cellsEl.scrollTop = g;
            requestAnimationFrame(() => {
              cellsEl.scrollLeft = g;
              cellsEl.scrollTop = g;
            });
          });
        });
      });
    });
  }

  syncAbortButton();
}

function formatOut(o: CellOutput) {
  let s = "";
  if (o.stdout) s += o.stdout;
  if (o.stderr) s += (s ? "\n" : "") + o.stderr;
  if (!s) s = o.ok ? "(finished — no stdout/stderr)" : "(failed)";
  return s;
}

/** Rich output only when the cell sent ``# stonesoup:render=html|md``; otherwise plain text. */
function presetRichKind(o: CellOutput): "html" | "markdown" | null {
  if (o.renderHint === "html") return "html";
  if (o.renderHint === "markdown") return "markdown";
  return null;
}

function showOutputRichToggleForCell(o: CellOutput | undefined): boolean {
  if (!o || !hasCellBodyOutput(o)) return false;
  return presetRichKind(o) !== null;
}

function outputPlainToggleLabel(preset: "html" | "markdown", asPlain: boolean): string {
  if (asPlain) return "Text";
  return preset === "html" ? "HTML" : "MD";
}

function outputPlainToggleTitle(preset: "html" | "markdown", asPlain: boolean): string {
  const rich = preset === "html" ? "HTML" : "Markdown";
  if (asPlain) return `Showing escaped text. Click to render ${rich}.`;
  return `Showing ${rich}. Click to view as plain text.`;
}

/** Kernel HTML / MD only with explicit first-line hint; sanitized before ``innerHTML`` (XSS-safe). */
function renderStdoutHtml(
  stdout: string,
  asPlainText: boolean,
  renderHint: StdoutKind | null | undefined,
): { html: string; rich: boolean } {
  if (!stdout.trim()) return { html: "", rich: false };
  if (asPlainText) {
    return { html: escapeHtml(stdout), rich: false };
  }
  if (renderHint === "html") {
    return { html: DOMPurify.sanitize(stdout), rich: true };
  }
  if (renderHint === "markdown") {
    const raw = marked.parse(stdout, { async: false }) as string;
    return { html: DOMPurify.sanitize(raw), rich: true };
  }
  return { html: escapeHtml(stdout), rich: false };
}

function renderOutputInnerHtml(o: CellOutput, outputKey: string): { html: string; richLayout: boolean } {
  const hasBody = Boolean(o.stdout.trim() || o.stderr.trim());
  if (!hasBody) {
    return { html: escapeHtml(formatOut(o)), richLayout: false };
  }
  const asPlain = cellStdoutPlainText.has(outputKey);
  const { html: outHtml, rich: stdoutRich } = renderStdoutHtml(o.stdout, asPlain, o.renderHint);
  let html = outHtml;
  if (o.stderr.trim()) {
    html += `<pre class="stonesoup-stderr">${escapeHtml(o.stderr)}</pre>`;
  }
  return { html, richLayout: stdoutRich };
}

/** Creates/updates/removes the HTML/MD ↔ plain chip in the output header. */
function syncOutLabelRowForCell(index: number) {
  const mk = markerKeyForCellIndex(index);
  if (!mk) return;
  const o = outputs.get(mk);
  const block = cellsEl.querySelector<HTMLElement>(`[data-output-block="${index}"]`);
  if (!block) return;
  const row = block.querySelector<HTMLElement>(".out-label-row");
  if (!row) return;
  let meta = row.querySelector<HTMLElement>(".out-label-meta");
  if (!meta) {
    meta = document.createElement("div");
    meta.className = "out-label-meta";
    meta.draggable = false;
    row.appendChild(meta);
  }

  const running = runningCellIndices.has(index);
  let durEl = meta.querySelector<HTMLElement>(`[data-out-duration="${index}"]`);
  if (!durEl) {
    durEl = document.createElement("span");
    durEl.className = "out-duration";
    durEl.draggable = false;
    durEl.dataset.outDuration = String(index);
    meta.insertBefore(durEl, meta.firstChild);
  }
  const ds = o?.durationSec;
  if (!running && typeof ds === "number" && Number.isFinite(ds) && ds >= 0) {
    durEl.textContent = formatDurationSec(ds);
    durEl.title = `Cell execution time: ${ds.toFixed(4)} s`;
    durEl.hidden = false;
  } else {
    durEl.textContent = "";
    durEl.removeAttribute("title");
    durEl.hidden = true;
  }

  const wantChip = Boolean(o && showOutputRichToggleForCell(o));
  let chip = meta.querySelector<HTMLButtonElement>(`[data-out-plain-toggle="${mk}"]`);
  if (!wantChip) {
    chip?.remove();
    return;
  }
  const preset = presetRichKind(o!);
  if (!preset) {
    chip?.remove();
    return;
  }
  const asPlain = cellStdoutPlainText.has(mk);
  if (!chip) {
    chip = document.createElement("button");
    chip.type = "button";
    chip.className = "out-mode-chip";
    chip.draggable = false;
    chip.dataset.outPlainToggle = mk;
    chip.setAttribute("aria-label", "Toggle plain text");
    chip.addEventListener("click", (ev) => {
      ev.stopPropagation();
      const key = chip!.dataset.outPlainToggle;
      if (!key) return;
      if (cellStdoutPlainText.has(key)) cellStdoutPlainText.delete(key);
      else cellStdoutPlainText.add(key);
      const i = cellIndexForMarkerKey(key);
      if (i !== undefined) refreshCellOutputView(i);
    });
    meta.appendChild(chip);
  }
  chip.textContent = outputPlainToggleLabel(preset, asPlain);
  chip.title = outputPlainToggleTitle(preset, asPlain);
}

function refreshCellOutputView(index: number) {
  syncOutLabelRowForCell(index);
  const mk = markerKeyForCellIndex(index);
  if (!mk) return;
  const o = outputs.get(mk);
  const outEl = cellsEl.querySelector<HTMLElement>(`[data-out="${index}"]`);
  if (!outEl || !o) return;
  const reveal = shouldRevealOutputStrip(o);
  if (!reveal) return;
  if (outEl.classList.contains("out-streaming")) return;
  const r = renderOutputInnerHtml(o, mk);
  outEl.className = "out " + (o.ok ? "ok" : "err") + (r.richLayout ? " out-rich" : "");
  outEl.innerHTML = r.html;
}

/** True when there is real stdout/stderr to show (placeholders like “no stdout” are not shown in the UI). */
function hasCellBodyOutput(o: { stdout: string; stderr: string } | undefined): boolean {
  if (!o) return false;
  return Boolean(o.stdout.trim() || o.stderr.trim());
}

/** Open the output strip for failed runs even if stderr/stdout are empty (kernel/network edge cases). */
function shouldRevealOutputStrip(
  o: { stdout: string; stderr: string; ok?: boolean } | undefined,
): boolean {
  if (!o) return false;
  if (o.ok === false) return true;
  return hasCellBodyOutput(o);
}

function setCellOutputBlockVisible(index: number, visible: boolean) {
  const block = cellsEl.querySelector<HTMLElement>(`[data-output-block="${index}"]`);
  if (block) block.style.display = visible ? "flex" : "none";
}

/** True while the output strip is shown (running/streaming or real output), even if `outputs` is not updated yet. */
function isOutputStripVisible(index: number): boolean {
  const block = cellsEl.querySelector<HTMLElement>(`[data-output-block="${index}"]`);
  if (!block) return false;
  const d = (block.style.display || "").toLowerCase();
  if (d === "flex" || d === "block") return true;
  /* Fallback: inline style can be empty in edge cases; rely on computed display. */
  return getComputedStyle(block).display !== "none";
}

/** Output strip should stay uncollapsed: saved result, visible strip, or cell currently executing (tqdm / model load). */
function cellHasExpandedOutputUi(index: number): boolean {
  const mk = markerKeyForCellIndex(index);
  return (
    (mk != null && shouldRevealOutputStrip(outputs.get(mk))) ||
    isOutputStripVisible(index) ||
    runningCellIndices.has(index)
  );
}

/** Toolbar lives in `.cell-body`; compact = no output so the card is header-only. */
function syncCellCompactClassForIndex(index: number) {
  const cell = cellsCanvas.querySelector<HTMLElement>(`.cell[data-pipeline-cell-drag="${index}"]`);
  if (!cell) return;
  const showOut = cellHasExpandedOutputUi(index);
  cell.classList.toggle("cell-compact", !showOut);
}

function syncCellStaleClassForIndex(index: number) {
  const cell = cellsCanvas.querySelector<HTMLElement>(`.cell[data-pipeline-cell-drag="${index}"]`);
  if (!cell) return;
  const stale = staleCells.has(index);
  cell.classList.toggle("cell-stale", stale);
  cell.title = stale
    ? "Source changed — re-run to clear. Drag title bar: move or drop on pipeline · corner resize"
    : "Drag title bar: move or drop on pipeline · corner resize";
  const badge = cell.querySelector<HTMLElement>(".cell-updated-badge");
  if (badge) badge.hidden = !stale;
}

function escapeHtml(s: string) {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

async function postWatch() {
  const path = pathInput.value.trim();
  if (!path) {
    setStatus("Pick a folder and file");
    return;
  }
  btnWatch.disabled = true;
  try {
    const r = await fetch(`${apiBase}/api/watch`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path }),
    });
    const j = (await readApiJson(r)) as {
      detail?: string;
      revision?: number;
      n_cells?: number;
      path?: string | null;
      cells?: Cell[];
      changed_cell_indices?: unknown;
      repo_root?: unknown;
    };
    if (!r.ok) throw new Error(j.detail || r.statusText);
    saveWatchPathCookie(path);
    if (Array.isArray(j.cells)) {
      applyCellsFromServer({
        revision: Number(j.revision) || 0,
        path: typeof j.path === "string" || j.path === null ? j.path : null,
        cells: j.cells,
        changed_cell_indices: j.changed_cell_indices,
        repo_root: j.repo_root,
      });
    } else {
      outputs.clear();
      cellStdoutPlainText.clear();
      staleCells.clear();
      revision = j.revision ?? revision;
    }
    setStatus(`watching · rev ${j.revision} · ${j.n_cells} cells`);
    scheduleKernelVarsRefresh();
    void fetchLoadedModels();
  } catch (e) {
    setStatus(String(e));
  } finally {
    btnWatch.disabled = false;
  }
}

async function runCell(index: number, inject?: Record<string, unknown> | null) {
  if (runningCellIndices.has(index)) {
    setStatus(`Cell ${index} is already running`);
    return;
  }
  const btn = cellsEl.querySelector<HTMLButtonElement>(`[data-run="${index}"]`);
  /* Match WebSocket run_start order: running flag first so compact/layout never clips live output (tqdm). */
  setCellRunningState(index, true);
  prepareCellStreamUi(index);
  appendCellRunHeaderToConsole(index);
  syncOutLabelRowForCell(index);
  if (btn) btn.disabled = true;
  try {
    const body: Record<string, unknown> = {
      cell_index: index,
      inject: mergeCellRunInject(index, inject),
    };
    const r = await fetch(`${apiBase}/api/run`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const j = (await readApiJson(r)) as {
      ok?: boolean;
      stdout?: string;
      stderr?: string;
      detail?: string;
    };
    if (!r.ok) throw new Error(j.detail || r.statusText);
    const peeled = peelStonesoupRenderHint(
      foldCarriageReturns(typeof j.stdout === "string" ? j.stdout : ""),
    );
    const nextOut: CellOutput = {
      stdout: peeled.body,
      stderr: foldCarriageReturns(typeof j.stderr === "string" ? j.stderr : ""),
      ok: Boolean(j.ok),
    };
    const rawDur = (j as { duration_sec?: unknown }).duration_sec;
    if (typeof rawDur === "number" && Number.isFinite(rawDur)) nextOut.durationSec = rawDur;
    if (peeled.renderHint != null) nextOut.renderHint = peeled.renderHint;
    const mk = markerKeyForCellIndex(index);
    if (mk) {
      if (presetRichKind(nextOut) === null) cellStdoutPlainText.delete(mk);
      outputs.set(mk, nextOut);
    }
    const streamed = (cellStreamBufferByIndex.get(index) ?? "").length > 0;
    if (
      !streamed &&
      (nextOut.stdout.trim() || nextOut.stderr.trim())
    ) {
      if (nextOut.stdout) appendAppLogChunk(nextOut.stdout);
      if (nextOut.stderr.trim()) {
        appendAppLogChunk(
          (nextOut.stdout.trim() ? "\n" : "") + nextOut.stderr,
        );
      }
    }
    const outEl = cellsEl.querySelector<HTMLElement>(`[data-out="${index}"]`);
    const reveal = shouldRevealOutputStrip(nextOut);
    setCellOutputBlockVisible(index, reveal);
    if (outEl) {
      outEl.classList.remove("out-streaming");
      const baseClass = reveal ? "out " + (j.ok ? "ok" : "err") : "out out-pending";
      if (reveal) {
        refreshCellOutputView(index);
      } else {
        outEl.className = baseClass;
        outEl.textContent = "";
        syncOutLabelRowForCell(index);
      }
    }
    syncCellCompactClassForIndex(index);
    schedulePersistOutputs();
    scheduleLayoutAndLines();
    if (!j.ok) {
      const line =
        nextOut.stderr.trim().split("\n")[0] || nextOut.stdout.trim().split("\n")[0] || "";
      setStatus(
        line
          ? `Cell ${index} failed — ${line.length > 100 ? `${line.slice(0, 100)}…` : line}`
          : `Cell ${index} failed — see output below`,
      );
    }
    if (j.ok) {
      staleCells.delete(index);
      syncCellStaleClassForIndex(index);
      renderPipelineBar();
    }
  } catch (e) {
    appendAppLogChunk(`\n${String(e)}\n`);
    const nextOut = { stdout: "", stderr: String(e), ok: false };
    const mkErr = markerKeyForCellIndex(index);
    if (mkErr) outputs.set(mkErr, nextOut);
    const outEl = cellsEl.querySelector<HTMLElement>(`[data-out="${index}"]`);
    setCellOutputBlockVisible(index, true);
    if (outEl) {
      outEl.classList.remove("out-streaming");
      outEl.className = "out err";
      outEl.classList.remove("out-rich");
      outEl.textContent = String(e);
      syncOutLabelRowForCell(index);
    }
    syncCellCompactClassForIndex(index);
    schedulePersistOutputs();
    scheduleLayoutAndLines();
  } finally {
    setCellRunningState(index, false);
    syncOutLabelRowForCell(index);
    if (btn) btn.disabled = false;
  }
}

async function resetServer() {
  btnReset.disabled = true;
  const watchPath =
    pathInput.value.trim() ||
    readWatchPathCookie() ||
    (lastPath?.trim() ?? "");
  try {
    const r = await fetch(`${apiBase}/api/reset`, { method: "POST" });
    if (!r.ok) throw new Error(await r.text());
    outputs.clear();
    cellStdoutPlainText.clear();
    staleCells.clear();
    clearPersistedOutputsForPath(layoutStoragePath());
    cellsEl.querySelectorAll<HTMLElement>("[data-output-block]").forEach((el) => {
      el.style.display = "none";
    });
    cellsEl.querySelectorAll(".out").forEach((el) => {
      el.textContent = "";
      el.className = "out out-pending";
    });
    cellsCanvas.querySelectorAll<HTMLElement>(".cell[data-pipeline-cell-drag]").forEach((cell) => {
      const idx = Number(cell.dataset.pipelineCellDrag);
      if (Number.isInteger(idx)) syncCellCompactClassForIndex(idx);
    });
    scheduleLayoutAndLines();
    setStatus("Restarting server…");
    for (let i = 0; i < 60; i++) {
      await new Promise<void>((resolve) => setTimeout(resolve, 250));
      try {
        const ping = await fetch(`${apiBase}/api/cells`);
        if (ping.ok) {
          connectWs();
          if (watchPath) {
            if (!pathInput.value.trim()) pathInput.value = watchPath;
            await postWatch();
          } else {
            setStatus("Server restarted — pick a file and click Watch.");
          }
          scheduleKernelVarsRefresh();
          void fetchLoadedModels();
          return;
        }
      } catch {
        /* still down */
      }
    }
    setStatus("Server did not respond in time — refresh the page when the backend is up.");
  } catch (e) {
    setStatus(String(e));
  } finally {
    btnReset.disabled = false;
  }
}

type RunStepsResult =
  | { ok: true; nRuns: number }
  | { ok: false; label: string; aborted?: boolean; nRuns?: number };

/** One controller per pipeline index while that pipeline is running (enables Abort). */
const activePipelineAbortControllers = new Map<number, AbortController>();

function syncPipelineAbortButtons() {
  document.querySelectorAll<HTMLButtonElement>("[data-pipeline-abort]").forEach((btn) => {
    const i = Number(btn.dataset.pipelineAbort);
    btn.disabled = !Number.isInteger(i) || !activePipelineAbortControllers.has(i);
  });
}

async function runSteps(
  steps: PipelineStep[],
  base: Record<string, unknown> | undefined,
  signal: AbortSignal,
): Promise<RunStepsResult> {
  let nRuns = 0;
  for (const step of steps) {
    if (signal.aborted) {
      return { ok: false, label: "aborted", aborted: true, nRuns };
    }
    if (step.kind === "cell") {
      await runCell(step.index, base);
      nRuns++;
      const mkStep = markerKeyForCellIndex(step.index);
      const o = mkStep ? outputs.get(mkStep) : undefined;
      if (!o?.ok) return { ok: false, label: `cell ${step.index}` };
    } else {
      const iters = step.iterations.length ? step.iterations : [{}];
      for (let i = 0; i < iters.length; i++) {
        if (signal.aborted) {
          return { ok: false, label: "aborted", aborted: true, nRuns };
        }
        const merged: Record<string, unknown> = { ...(base ?? {}) };
        const patch = iterationToInject(iters[i], i);
        if (patch) Object.assign(merged, patch);
        const sub = await runSteps(step.body, merged, signal);
        if (!sub.ok) {
          if (sub.aborted) return sub;
          return {
            ok: false,
            label: `loop ${i + 1}/${iters.length} → ${sub.label}`,
          };
        }
        nRuns += sub.nRuns;
      }
    }
  }
  return { ok: true, nRuns };
}

function programIsEmpty(steps: PipelineStep[]): boolean {
  return flattenCellIndices(steps).length === 0;
}

/** Apply textarea values into the program so Run uses latest edits (no blur needed). */
function syncLoopTextareasFromDom(): string | null {
  const stack = document.getElementById("pipelines-stack");
  if (!stack) return null;
  for (const ta of stack.querySelectorAll<HTMLTextAreaElement>("[data-loop-path]")) {
    const path = JSON.parse(ta.dataset.loopPath!) as number[];
    const pIdx = Number(ta.dataset.loopPipeline);
    const st = stepAtPath(path, pIdx);
    if (!st || st.kind !== "loop") continue;
    const parsed = parseIterationsJson(ta.value);
    if (!parsed.ok) return parsed.error;
    st.iterations = parsed.iterations;
  }
  savePipeline();
  return null;
}

async function runPipeline(pIdx: number) {
  const btn = document.querySelector<HTMLButtonElement>(`[data-pipeline-run="${pIdx}"]`);
  if (btn) btn.disabled = true;
  const ac = new AbortController();
  activePipelineAbortControllers.set(pIdx, ac);
  syncPipelineAbortButtons();
  const n = lastCells.length;
  const domErr = syncLoopTextareasFromDom();
  if (domErr) {
    setStatus(domErr);
    activePipelineAbortControllers.delete(pIdx);
    syncPipelineAbortButtons();
    if (btn) btn.disabled = false;
    return;
  }
  const steps = sanitizeProgram(pipelines[pIdx] ?? [], n);
  if (programIsEmpty(steps)) {
    setStatus(`Pipeline ${pIdx + 1} is empty — use the add-to-pipeline (+) icon on a cell or drag ↻ Loop below`);
    activePipelineAbortControllers.delete(pIdx);
    syncPipelineAbortButtons();
    if (btn) btn.disabled = false;
    return;
  }
  try {
    const r = await runSteps(steps, undefined, ac.signal);
    if (!r.ok) {
      if (r.aborted) {
        const k = r.nRuns ?? 0;
        setStatus(`Pipeline ${pIdx + 1} aborted after ${k} cell run${k === 1 ? "" : "s"}`);
        return;
      }
      setStatus(`Pipeline ${pIdx + 1} stopped · ${r.label}`);
      return;
    }
    setStatus(`Pipeline ${pIdx + 1} finished (${r.nRuns} cell runs)`);
  } finally {
    activePipelineAbortControllers.delete(pIdx);
    syncPipelineAbortButtons();
    if (btn) btn.disabled = false;
  }
}

btnCellsAutoLayout.addEventListener("click", () => {
  resetCellsToAutoLayout();
});

type DocumentWithLegacyFs = Document & {
  webkitFullscreenElement?: Element | null;
  webkitExitFullscreen?: () => Promise<void>;
};
type ElementWithLegacyFs = Element & {
  webkitRequestFullscreen?: () => Promise<void>;
};

function documentFullscreenElement(): Element | null {
  const d = document as DocumentWithLegacyFs;
  return document.fullscreenElement ?? d.webkitFullscreenElement ?? null;
}

function syncWorkspaceFullscreenUi() {
  const active = documentFullscreenElement() === workspaceEl;
  btnWorkspaceFullscreen.classList.toggle("is-fullscreen", active);
  btnWorkspaceFullscreen.setAttribute("aria-pressed", active ? "true" : "false");
  btnWorkspaceFullscreen.setAttribute("aria-label", active ? "Exit fullscreen" : "Enter fullscreen");
  btnWorkspaceFullscreen.title = active
    ? "Exit fullscreen"
    : "Fullscreen workspace (cells, console, variables)";
}

document.addEventListener("fullscreenchange", syncWorkspaceFullscreenUi);
document.addEventListener("webkitfullscreenchange", syncWorkspaceFullscreenUi);

btnWorkspaceFullscreen.addEventListener("click", () => {
  void (async () => {
    try {
      if (documentFullscreenElement() === workspaceEl) {
        const d = document as DocumentWithLegacyFs;
        if (document.exitFullscreen) await document.exitFullscreen();
        else await d.webkitExitFullscreen?.();
      } else {
        const w = workspaceEl as HTMLElement & ElementWithLegacyFs;
        if (w.requestFullscreen) await w.requestFullscreen();
        else await w.webkitRequestFullscreen?.();
      }
    } catch (e) {
      setStatus(`Fullscreen: ${String(e)}`);
    }
  })();
});

btnWatch.addEventListener("click", () => postWatch());
btnReset.addEventListener("click", () => resetServer());
btnAbort.addEventListener("click", () => void postRunAbort());

async function postRunAbort() {
  try {
    const r = await fetch(`${apiBase}/api/run/abort`, { method: "POST" });
    const j = (await readApiJson(r)) as { ok?: boolean; detail?: string; signaled?: boolean };
    if (!r.ok) throw new Error(typeof j.detail === "string" ? j.detail : r.statusText);
    if (j.signaled === false) {
      setStatus("Abort: no cell run was active.");
    } else {
      setStatus("Abort requested…");
    }
  } catch (e) {
    setStatus(String(e));
  }
}
btnEditorToggle.addEventListener("click", () => {
  const next: EditorPref = getEditorPref() === "cursor" ? "vscode" : "cursor";
  setEditorPref(next);
  btnEditorToggle.dataset.editorPref = next;
  if (lastCells.length) renderCells(lastCells, lastPath);
});
btnModelLoad.addEventListener("click", () => void loadModelsFromToolbar());
btnModelUnloadOne.addEventListener("click", () => void unloadSelectedModelFromToolbar());
btnModelUnloadAll.addEventListener("click", () => void unloadAllModelsFromToolbar());
modelsLoadedSelect.addEventListener("change", () => copyLoadedModelRepoIdToClipboard());
modelRepoInput.addEventListener("keydown", (e) => {
  if (e.key !== "Enter") return;
  e.preventDefault();
  void loadModelsFromToolbar();
});
modelRepoInput.addEventListener("input", () => syncModelRepoInputWidthCh());

window.addEventListener("resize", () => {
  if (lastCells.length) scheduleLayoutAndLines();
  schedulePersistCellsView();
  requestAnimationFrame(() => updatePipelineLineBreakMarkers());
});

function clearPipelineDropHighlights() {
  document.querySelectorAll(".pipeline-drop-zone.is-drag-over, .pipeline-strip-drop-zone.is-drag-over").forEach(
    (el) => {
      el.classList.remove("is-drag-over");
    },
  );
}

/**
 * Resolve drop target from pointer. Zones are thin inserts; pad rects first, then map any point
 * inside a `.pipeline-chips-flow` to the nearest zone in that flow (no tiny score cutoff — the
 * old stack-wide `bestScore < 100` missed drops on the right side of an empty/wide strip).
 */
function hitTestPipelineDropZone(clientX: number, clientY: number): HTMLElement | null {
  const stack = document.getElementById("pipelines-stack");
  if (!stack) return null;
  const zones = [...stack.querySelectorAll<HTMLElement>(".pipeline-drop-zone")];
  if (zones.length === 0) return null;

  const pad = 28;
  for (const z of zones) {
    const r = z.getBoundingClientRect();
    if (
      clientX >= r.left - pad &&
      clientX <= r.right + pad &&
      clientY >= r.top - pad &&
      clientY <= r.bottom + pad
    ) {
      return z;
    }
  }

  const shells = [...stack.querySelectorAll<HTMLElement>(".pipeline-chips")];
  let flow: HTMLElement | null = null;
  const shellPad = 8;
  for (const shell of shells) {
    const sr = shell.getBoundingClientRect();
    if (
      clientX < sr.left - shellPad ||
      clientX > sr.right + shellPad ||
      clientY < sr.top - shellPad ||
      clientY > sr.bottom + shellPad
    ) {
      continue;
    }
    flow = shell.querySelector<HTMLElement>(".pipeline-chips-flow");
    break;
  }
  if (!flow) return null;

  const flowZones = zones.filter((z) => flow!.contains(z));
  if (flowZones.length === 0) return null;

  let best: HTMLElement | null = null;
  let bestScore = Infinity;
  for (const z of flowZones) {
    const r = z.getBoundingClientRect();
    const midX = (r.left + r.right) / 2;
    const midY = (r.top + r.bottom) / 2;
    const dx = Math.abs(clientX - midX);
    const dy = Math.abs(clientY - midY);
    const score = dx * 1.15 + dy;
    if (score < bestScore) {
      bestScore = score;
      best = z;
    }
  }
  return best;
}

/** Drop target for reordering whole pipeline rows (between strips). */
function hitTestStripReorderZone(clientX: number, clientY: number): HTMLElement | null {
  const stack = document.getElementById("pipelines-stack");
  if (!stack) return null;
  const zones = [...stack.querySelectorAll<HTMLElement>(".pipeline-strip-drop-zone")];
  if (zones.length === 0) return null;
  const pad = 12;
  let best: HTMLElement | null = null;
  let bestArea = Infinity;
  for (const z of zones) {
    const r = z.getBoundingClientRect();
    if (
      clientX >= r.left - pad &&
      clientX <= r.right + pad &&
      clientY >= r.top - pad &&
      clientY <= r.bottom + pad
    ) {
      const area = Math.max(1, r.width) * Math.max(1, r.height);
      if (area < bestArea) {
        bestArea = area;
        best = z;
      }
    }
  }
  return best;
}

function parseDndPayload(dt: DataTransfer): DndPayload | null {
  try {
    const raw = dt.getData(DND_PAYLOAD);
    if (!raw) return null;
    const x = JSON.parse(raw) as DndPayload;
    if (x.kind === "canvas" && typeof x.cellIndex === "number" && Number.isInteger(x.cellIndex))
      return x;
    if (x.kind === "reorder_strip") {
      const rs = x as { kind: string; fromPipeline?: number };
      if (typeof rs.fromPipeline === "number" && Number.isInteger(rs.fromPipeline)) {
        return { kind: "reorder_strip", fromPipeline: rs.fromPipeline };
      }
    }
    if (x.kind === "move" && Array.isArray(x.fromPath)) {
      const mp = x as { fromPath: number[]; fromPipeline?: number };
      const fromPipeline =
        typeof mp.fromPipeline === "number" && Number.isInteger(mp.fromPipeline)
          ? mp.fromPipeline
          : 0;
      return { kind: "move", fromPath: mp.fromPath, fromPipeline };
    }
  } catch {
    return null;
  }
  return null;
}

type CanvasHeadDragState =
  | {
      kind: "cell";
      el: HTMLElement;
      pointerId: number;
      startX: number;
      startY: number;
      origL: number;
      origT: number;
      cellIndex: number;
    }
  | {
      kind: "loop";
      el: HTMLElement;
      pointerId: number;
      startX: number;
      startY: number;
      /** Viewport coordinates; element uses `position: fixed` while dragging. */
      origL: number;
      origT: number;
      viewportDrag: true;
    };
type CellResizeGeomState = {
  el: HTMLElement;
  pointerId: number;
  /** Viewport position of cell top-left (border box); width/height = cursor − these so the corner tracks the pointer */
  vLeft: number;
  vTop: number;
};

let canvasHeadDragGeom: CanvasHeadDragState | null = null;
let cellResizeGeom: CellResizeGeomState | null = null;

/** Background drag on `.cells` (not on a `.cell`) pans the scroll viewport. */
let cellsPanState: { pointerId: number; lastX: number; lastY: number } | null = null;

function attachCellGeomWindowListeners() {
  window.addEventListener("pointermove", onCellGeomWindowMove);
  window.addEventListener("pointerup", onCellGeomWindowEnd);
  window.addEventListener("pointercancel", onCellGeomWindowEnd);
}

function detachCellGeomWindowListeners() {
  window.removeEventListener("pointermove", onCellGeomWindowMove);
  window.removeEventListener("pointerup", onCellGeomWindowEnd);
  window.removeEventListener("pointercancel", onCellGeomWindowEnd);
}

/** Map viewport (post-transform) pixel deltas to `.cells-canvas` layout pixels when zoom ≠ 1. */
function layoutDeltaFromViewport(d: number): number {
  const s = cellsViewScale;
  return s > 0 ? d / s : d;
}

function onCellGeomWindowMove(e: PointerEvent) {
  if (cellResizeGeom && e.pointerId === cellResizeGeom.pointerId) {
    e.preventDefault();
    const { el, vLeft, vTop } = cellResizeGeom;
    /** `getBoundingClientRect` is visual (scaled); width/height are layout (pre-scale). */
    const wVis = e.clientX - vLeft;
    const hVis = e.clientY - vTop;
    el.style.width = `${Math.max(CELL_LAYOUT_MIN_W, layoutDeltaFromViewport(wVis))}px`;
    el.style.height = `${Math.max(CELL_LAYOUT_MIN_H, layoutDeltaFromViewport(hVis))}px`;
    el.style.minHeight = "0";
    relayoutCanvasBounds();
    return;
  }
  if (canvasHeadDragGeom && e.pointerId === canvasHeadDragGeom.pointerId) {
    e.preventDefault();
    const { el, startX, startY, origL, origT, kind } = canvasHeadDragGeom;
    clearPipelineDropHighlights();
    const z = hitTestPipelineDropZone(e.clientX, e.clientY);
    const overPipeline = Boolean(z && pipelineRow.contains(z));
    const dx = layoutDeltaFromViewport(e.clientX - startX);
    const dy = layoutDeltaFromViewport(e.clientY - startY);
    if (overPipeline && z) {
      /* Snap preview to home: release here only adds to pipeline, not a canvas move */
      el.style.left = `${origL}px`;
      el.style.top = `${origT}px`;
      z.classList.add("is-drag-over");
    } else {
      el.style.left = `${origL + dx}px`;
      el.style.top = `${origT + dy}px`;
    }
    if (kind === "cell") relayoutCanvasBounds();
  }
}

function onCellGeomWindowEnd(e: PointerEvent) {
  if (cellResizeGeom && e.pointerId === cellResizeGeom.pointerId) {
    try {
      cellResizeGeom.el.releasePointerCapture(e.pointerId);
    } catch {
      /* already released */
    }
    snapshotCurrentLayoutToManualMap();
    scheduleSaveCellLayouts();
    cellResizeGeom = null;
    detachCellGeomWindowListeners();
    return;
  }
  if (canvasHeadDragGeom && e.pointerId === canvasHeadDragGeom.pointerId) {
    try {
      canvasHeadDragGeom.el.releasePointerCapture(e.pointerId);
    } catch {
      /* already released */
    }
    clearPipelineDropHighlights();
    const z = hitTestPipelineDropZone(e.clientX, e.clientY);
    let inserted = false;
    if (z && pipelineRow.contains(z)) {
      const loopRaw = z.dataset.dropLoop ?? "";
      let bodyLoopPath: number[] | null = null;
      if (loopRaw !== "") {
        try {
          bodyLoopPath = JSON.parse(loopRaw) as number[];
        } catch {
          /* ignore */
        }
      }
      const at = Number(z.dataset.dropAt);
      const toPIdx = Number(z.dataset.dropPipeline);
      if (Number.isInteger(at) && at >= 0 && Number.isInteger(toPIdx) && toPIdx >= 0) {
        const { el, origL, origT, kind } = canvasHeadDragGeom;
        if (kind === "cell") {
          el.style.left = `${origL}px`;
          el.style.top = `${origT}px`;
          insertCellInPipeline(canvasHeadDragGeom.cellIndex, bodyLoopPath, at, toPIdx);
          inserted = true;
        } else if (kind === "loop") {
          resetLoopPaletteSlotPosition(el);
          insertNewLoopInPipeline(bodyLoopPath, at, toPIdx);
          inserted = true;
        }
        if (inserted) {
          clearLoopExpanded();
          savePipeline();
          renderPipelineBar();
          highlightPipelineCells();
          setStatus("Pipeline updated");
        }
      }
    }
    if (canvasHeadDragGeom.kind === "loop" && !inserted) {
      resetLoopPaletteSlotPosition(canvasHeadDragGeom.el);
    }
    snapshotCurrentLayoutToManualMap();
    scheduleSaveCellLayouts();
    canvasHeadDragGeom = null;
    detachCellGeomWindowListeners();
  }
}

let cellGeometryBound = false;
function bindCellGeometryInteractions() {
  if (cellGeometryBound) return;
  cellGeometryBound = true;

  pipelineRow.addEventListener(
    "pointerdown",
    (e: PointerEvent) => {
      if (e.button !== 0) return;
      const t = e.target as HTMLElement;
      if (!pipelineRow.contains(t)) return;
      const loopHead = t.closest(".loop-palette-head");
      if (!loopHead || !pipelineRow.contains(loopHead)) return;
      if (t.closest("button, a, input, textarea, select")) return;
      const loopPal = loopHead.closest<HTMLElement>(".loop-palette");
      if (!loopPal || !loopPaletteSlot.contains(loopPal)) return;
      bringCellToFront(loopPal);
      e.preventDefault();
      e.stopPropagation();
      if (manualLayoutByCellIdx.size === 0) snapshotCurrentLayoutToManualMap();
      const br = loopPal.getBoundingClientRect();
      loopPal.classList.add("loop-palette--dragging");
      loopPal.style.position = "fixed";
      loopPal.style.left = `${br.left}px`;
      loopPal.style.top = `${br.top}px`;
      loopPal.style.width = `${br.width}px`;
      loopPal.style.zIndex = String(++cellZStackCounter);
      canvasHeadDragGeom = {
        kind: "loop",
        el: loopPal,
        pointerId: e.pointerId,
        startX: e.clientX,
        startY: e.clientY,
        origL: br.left,
        origT: br.top,
        viewportDrag: true,
      };
      loopPal.setPointerCapture(e.pointerId);
      attachCellGeomWindowListeners();
    },
    true,
  );

  cellsCanvas.addEventListener(
    "pointerdown",
    (e: PointerEvent) => {
      if (e.button !== 0) return;
      const t = e.target as HTMLElement;
      if (!cellsCanvas.contains(t)) return;

      const rh = t.closest(".cell-resize-handle");
      if (rh) {
        const cell = rh.closest<HTMLElement>(".cell[data-pipeline-cell-drag]");
        if (!cell || !cellsCanvas.contains(cell)) return;
        bringCellToFront(cell);
        e.preventDefault();
        e.stopPropagation();
        if (manualLayoutByCellIdx.size === 0) snapshotCurrentLayoutToManualMap();
        cell.classList.add("cell-custom-geometry");
        void cell.offsetWidth;
        const br = cell.getBoundingClientRect();
        cellResizeGeom = {
          el: cell,
          pointerId: e.pointerId,
          vLeft: br.left,
          vTop: br.top,
        };
        cell.setPointerCapture(e.pointerId);
        attachCellGeomWindowListeners();
        return;
      }

      const head = t.closest(".cell-head");
      if (!head || !cellsCanvas.contains(head)) return;
      if (t.closest("button, a, input, textarea, select")) return;
      const cell = head.closest<HTMLElement>(".cell[data-pipeline-cell-drag]");
      if (!cell || !cellsCanvas.contains(cell)) return;
      const cellIndex = Number(cell.dataset.pipelineCellDrag);
      if (!Number.isInteger(cellIndex)) return;
      bringCellToFront(cell);
      e.preventDefault();
      e.stopPropagation();
      if (manualLayoutByCellIdx.size === 0) snapshotCurrentLayoutToManualMap();
      canvasHeadDragGeom = {
        kind: "cell",
        el: cell,
        pointerId: e.pointerId,
        startX: e.clientX,
        startY: e.clientY,
        origL: parseFloat(cell.style.left) || cell.offsetLeft,
        origT: parseFloat(cell.style.top) || cell.offsetTop,
        cellIndex,
      };
      cell.setPointerCapture(e.pointerId);
      attachCellGeomWindowListeners();
    },
    true,
  );

  cellsCanvas.addEventListener(
    "pointerdown",
    (e: PointerEvent) => {
      if (e.button !== 0) return;
      const tgt = e.target as HTMLElement;
      const cell = tgt.closest<HTMLElement>(".cell[data-pipeline-cell-drag]");
      if (cell && cellsCanvas.contains(cell)) bringCellToFront(cell);
    },
    false,
  );
}

/** Browsers often hide `getData` until `drop`; use this for `dragover` feedback. */
let activePipelineDnd: DndPayload | null = null;

let pipelineDndBound = false;
function bindPipelineDnD() {
  if (pipelineDndBound) return;
  pipelineDndBound = true;
  const pipelineRow = document.getElementById("pipeline-row");

  document.addEventListener(
    "dragstart",
    (e) => {
      activePipelineDnd = null;
      const raw = e.target;
      const t =
        raw instanceof HTMLElement ? raw : (raw as Node).parentElement ?? null;
      if (!t) return;
      const pl = t.closest<HTMLElement>("[data-pipeline-loop-drag]");
      if (pl && pipelineRow && pipelineRow.contains(pl)) {
        const shell = pl.closest<HTMLElement>(".pipeline-chips");
        const fromPipeline = Number(shell?.dataset.pipelineIndex);
        if (!Number.isInteger(fromPipeline)) return;
        const fromPath = JSON.parse(pl.dataset.pipelineLoopDrag!) as number[];
        const st = stepAtPath(fromPath, fromPipeline);
        if (!st || st.kind !== "loop") return;
        const payload: DndPayload = { kind: "move", fromPath, fromPipeline };
        activePipelineDnd = payload;
        e.dataTransfer!.setData(DND_PAYLOAD, JSON.stringify(payload));
        e.dataTransfer!.effectAllowed = "move";
        return;
      }
      const pg = t.closest<HTMLElement>("[data-pipeline-chip-drag]");
      if (pg && pipelineRow && pipelineRow.contains(pg)) {
        const fromBtn = t.closest("button");
        if (fromBtn && pg.contains(fromBtn)) return;
        const shell = pg.closest<HTMLElement>(".pipeline-chips");
        const fromPipeline = Number(shell?.dataset.pipelineIndex);
        if (!Number.isInteger(fromPipeline)) return;
        const fromPath = JSON.parse(pg.dataset.pipelineChipDrag!) as number[];
        const payload: DndPayload = { kind: "move", fromPath, fromPipeline };
        activePipelineDnd = payload;
        e.dataTransfer!.setData(DND_PAYLOAD, JSON.stringify(payload));
        e.dataTransfer!.effectAllowed = "move";
        return;
      }
      const stripShell = t.closest<HTMLElement>("[data-pipeline-strip-from]");
      if (stripShell && pipelineRow && pipelineRow.contains(stripShell)) {
        const blocked = t.closest<HTMLElement>(
          "button, a, input, textarea, select, [data-pipeline-chip-drag], [data-pipeline-loop-drag], .pipeline-drop-zone",
        );
        if (blocked && stripShell.contains(blocked)) {
          e.preventDefault();
          return;
        }
        const fromPipeline = Number(stripShell.dataset.pipelineStripFrom);
        if (!Number.isInteger(fromPipeline) || pipelines.length <= 1) return;
        const payload: DndPayload = { kind: "reorder_strip", fromPipeline };
        activePipelineDnd = payload;
        e.dataTransfer!.setData(DND_PAYLOAD, JSON.stringify(payload));
        e.dataTransfer!.effectAllowed = "move";
      }
    },
    true,
  );

  document.addEventListener("dragend", () => {
    activePipelineDnd = null;
    clearPipelineDropHighlights();
  });

  document.addEventListener("dragover", (e) => {
    if (!pipelineRow || !activePipelineDnd) {
      clearPipelineDropHighlights();
      return;
    }
    if (!pipelineRow.contains(e.target as Node)) {
      clearPipelineDropHighlights();
      return;
    }
    const dt = e.dataTransfer;
    if (!dt) return;
    e.preventDefault();
    clearPipelineDropHighlights();
    if (activePipelineDnd.kind === "reorder_strip") {
      dt.dropEffect = "move";
      const sz = hitTestStripReorderZone(e.clientX, e.clientY);
      if (sz) sz.classList.add("is-drag-over");
      return;
    }
    dt.dropEffect = activePipelineDnd.kind === "move" ? "move" : "copy";
    const z = hitTestPipelineDropZone(e.clientX, e.clientY);
    if (z) z.classList.add("is-drag-over");
  });

  document.addEventListener("drop", (e) => {
    if (!pipelineRow || !pipelineRow.contains(e.target as Node)) return;
    e.preventDefault();
    clearPipelineDropHighlights();
    const payload = parseDndPayload(e.dataTransfer!);
    if (!payload) return;

    if (payload.kind === "reorder_strip") {
      const sz = hitTestStripReorderZone(e.clientX, e.clientY);
      if (!sz || !pipelineRow.contains(sz)) return;
      const to = Number(sz.dataset.stripReorderAt);
      if (!Number.isInteger(to) || to < 0 || to > pipelines.length) return;
      if (!movePipelineStrip(payload.fromPipeline, to)) return;
      savePipeline();
      renderPipelineBar();
      highlightPipelineCells();
      setStatus("Pipeline order updated");
      return;
    }

    const z = hitTestPipelineDropZone(e.clientX, e.clientY);
    if (!z || !pipelineRow.contains(z)) return;
    const loopRaw = z.dataset.dropLoop ?? "";
    let bodyLoopPath: number[] | null = null;
    if (loopRaw !== "") {
      try {
        bodyLoopPath = JSON.parse(loopRaw) as number[];
      } catch {
        return;
      }
    }
    const at = Number(z.dataset.dropAt);
    if (!Number.isInteger(at) || at < 0) return;
    const toPIdx = Number(z.dataset.dropPipeline);
    if (!Number.isInteger(toPIdx) || toPIdx < 0) return;

    if (payload.kind === "canvas") {
      insertCellInPipeline(payload.cellIndex, bodyLoopPath, at, toPIdx);
    } else {
      movePipelineStep(payload.fromPath, payload.fromPipeline, bodyLoopPath, toPIdx, at);
    }
    clearLoopExpanded();
    savePipeline();
    renderPipelineBar();
    highlightPipelineCells();
    setStatus("Pipeline updated");
  });
}

const WHEEL_SCROLL_EPS = 1;

function overflowAllowsScrollY(el: HTMLElement): boolean {
  const y = getComputedStyle(el).overflowY;
  return y === "auto" || y === "scroll" || y === "overlay";
}

function overflowAllowsScrollX(el: HTMLElement): boolean {
  const x = getComputedStyle(el).overflowX;
  return x === "auto" || x === "scroll" || x === "overlay";
}

function elementCanScrollY(el: HTMLElement, dy: number): boolean {
  if (!overflowAllowsScrollY(el)) return false;
  if (el.scrollHeight <= el.clientHeight + WHEEL_SCROLL_EPS) return false;
  if (dy > 0) return el.scrollTop + WHEEL_SCROLL_EPS < el.scrollHeight - el.clientHeight;
  if (dy < 0) return el.scrollTop > WHEEL_SCROLL_EPS;
  return false;
}

function elementCanScrollX(el: HTMLElement, dx: number): boolean {
  if (!overflowAllowsScrollX(el)) return false;
  if (el.scrollWidth <= el.clientWidth + WHEEL_SCROLL_EPS) return false;
  if (dx > 0) return el.scrollLeft + WHEEL_SCROLL_EPS < el.scrollWidth - el.clientWidth;
  if (dx < 0) return el.scrollLeft > WHEEL_SCROLL_EPS;
  return false;
}

/** True if default wheel behavior would scroll something inside ``cell`` (not the main ``.cells`` viewport). */
function cellInnerAbsorbsWheel(origin: Element, cell: Element, dy: number, dx: number): boolean {
  let n: Element | null = origin;
  while (n && cell.contains(n)) {
    if (n instanceof HTMLElement) {
      if (dy !== 0 && elementCanScrollY(n, dy)) return true;
      if (dx !== 0 && elementCanScrollX(n, dx)) return true;
    }
    if (n === cell) break;
    n = n.parentElement;
  }
  return false;
}

function bindCellsViewportPan() {
  cellsEl.addEventListener("pointerdown", (e: PointerEvent) => {
    if (e.button !== 0 || cellsPanState) return;
    const t = e.target as HTMLElement;
    if (!cellsEl.contains(t) || t.closest(".cell")) return;
    e.preventDefault();
    cellsPanState = { pointerId: e.pointerId, lastX: e.clientX, lastY: e.clientY };
    cellsEl.classList.add("cells--panning");
    try {
      cellsEl.setPointerCapture(e.pointerId);
    } catch {
      /* ignore */
    }
  });

  cellsEl.addEventListener("pointermove", (e: PointerEvent) => {
    if (!cellsPanState || e.pointerId !== cellsPanState.pointerId) return;
    e.preventDefault();
    const dx = e.clientX - cellsPanState.lastX;
    const dy = e.clientY - cellsPanState.lastY;
    cellsEl.scrollLeft -= dx;
    cellsEl.scrollTop -= dy;
    cellsPanState.lastX = e.clientX;
    cellsPanState.lastY = e.clientY;
  });

  const endCellsPan = (e: PointerEvent) => {
    if (!cellsPanState || e.pointerId !== cellsPanState.pointerId) return;
    try {
      cellsEl.releasePointerCapture(e.pointerId);
    } catch {
      /* ignore */
    }
    cellsEl.classList.remove("cells--panning");
    cellsPanState = null;
    schedulePersistCellsView();
  };
  cellsEl.addEventListener("pointerup", endCellsPan);
  cellsEl.addEventListener("pointercancel", endCellsPan);

  /**
   * Background: wheel zooms (no `.cells` scroll). Over a `.cell`: scroll only inner ``pre`` / ``.out`` when
   * they overflow; otherwise prevent default so the main panel does not pan.
   */
  cellsEl.addEventListener(
    "wheel",
    (e: WheelEvent) => {
      const raw = e.target;
      const origin = raw instanceof Element ? raw : (raw as Node).parentElement;
      if (!origin || !cellsEl.contains(origin)) return;
      const cell = origin.closest(".cell");
      if (cell && cellsEl.contains(cell)) {
        if (cellInnerAbsorbsWheel(origin, cell, e.deltaY, e.deltaX)) return;
        e.preventDefault();
        return;
      }
      e.preventDefault();
      const oldS = cellsViewScale;
      const step = Math.exp(-e.deltaY * 0.002);
      const newS = Math.min(CELLS_ZOOM_MAX, Math.max(CELLS_ZOOM_MIN, oldS * step));
      if (Math.abs(newS - oldS) < 1e-4) return;
      const rect = cellsEl.getBoundingClientRect();
      /** Mouse in scrollport coords (border excluded; aligns with scrollLeft / scrollTop). */
      const mx = e.clientX - rect.left - cellsEl.clientLeft;
      const my = e.clientY - rect.top - cellsEl.clientTop;
      const sl0 = cellsEl.scrollLeft;
      const st0 = cellsEl.scrollTop;
      const g = CELLS_PAN_GUTTER_PX;
      cellsViewScale = newS;
      applyCellsZoomLayout();
      /** Zoom-wrap origin is offset by pan-arena padding `(g,g)`; keep point under cursor fixed. */
      const sl1 = g + (sl0 + mx - g) * (newS / oldS) - mx;
      const st1 = g + (st0 + my - g) * (newS / oldS) - my;
    cellsEl.scrollLeft = sl1;
    cellsEl.scrollTop = st1;
      requestAnimationFrame(() => {
        cellsEl.scrollLeft = sl1;
        cellsEl.scrollTop = st1;
        schedulePersistCellsView();
      });
    },
    { passive: false },
  );

  cellsEl.addEventListener("scroll", schedulePersistCellsView, { passive: true });
}

bindCellGeometryInteractions();
bindCellsViewportPan();
bindPipelineDnD();
bindPipelineChipCanvasHover();
renderPipelineBar();
initBottomDock();
connectWs();
void fetchLoadedModels();
void refreshModelRepoSuggestions();
