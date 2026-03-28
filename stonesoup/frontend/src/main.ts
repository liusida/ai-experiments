import "./style.css";
import DOMPurify from "dompurify";
import hljs from "highlight.js/lib/core";
import python from "highlight.js/lib/languages/python";
import { marked } from "marked";

hljs.registerLanguage("python", python);

type Cell = {
  index: number;
  title: string;
  source: string;
  marker_key: string;
  cell_input?: boolean;
};

/** Manual resize / saved layout: must leave room below `.cell-head` for output/code */
const CELL_LAYOUT_MIN_W = 220;
const CELL_LAYOUT_MIN_H = 200;

/** Symmetric empty margin around the zoom strip so panning can continue past content on all sides. */
const CELLS_PAN_GUTTER_PX = 2400;

const apiBase = import.meta.env.DEV ? "" : "http://127.0.0.1:8765";

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

/** Per-watched-file cell positions/sizes; one cookie JSON object keyed by repo-relative path */
const CELL_LAYOUTS_COOKIE = "stonesoup_cell_layouts_v1";
type CellLayoutTuple = [number, number, number, number];
type CellLayoutsFileMap = Record<string, CellLayoutTuple>;

function parseCellLayoutsCookie(): Record<string, CellLayoutsFileMap> {
  const prefix = `${CELL_LAYOUTS_COOKIE}=`;
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

function writeCellLayoutsCookie(all: Record<string, CellLayoutsFileMap>) {
  document.cookie = `${CELL_LAYOUTS_COOKIE}=${encodeURIComponent(JSON.stringify(all))}; Path=/; Max-Age=${WATCH_PATH_COOKIE_MAX_AGE}; SameSite=Lax`;
}


function resetCellsToAutoLayout(): void {
  manualLayoutByCellIdx.clear();
  cellPositions.clear();
  lastLayoutCols = -1;
  const pathKey = layoutStoragePath();
  if (pathKey && pathKey !== "_unset") {
    const all = parseCellLayoutsCookie();
    delete all[pathKey];
    writeCellLayoutsCookie(all);
  }
  applyFloatingLayout();
  setStatus("Cells: automatic layout");
}

function layoutStoragePath(): string {
  return pathInput.value.trim() || lastPath || "_unset";
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
    const all = parseCellLayoutsCookie();
    all[pathKey] = rec;
    writeCellLayoutsCookie(all);
  }, 250);
}

const defaultPath =
  (urlParams.get("path") || "").trim() ||
  readWatchPathCookie().trim() ||
  "experiments/2026-03-23-Embedding/demo.py";

/** List *.py from this repo-relative folder; default `experiments` shows all dated experiment scripts. */
const scriptPickerDir =
  (urlParams.get("dir") ?? "").trim() || "experiments";

app.innerHTML = `
  <div class="toolbar">
    <span class="ws-dot" id="ws-dot" title="Live reload: disconnected" aria-label="WebSocket disconnected"></span>
    <span class="script-picker">
      <select id="folder-select" title="Experiment folder under list root" aria-label="Folder"></select>
      <select id="file-select" title="Python file in folder" aria-label="File"></select>
    </span>
    <input type="hidden" id="path-input" />
    <button type="button" class="primary" id="btn-watch">Watch</button>
    <button type="button" id="btn-reset">Reset kernel</button>
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
    <div class="kernel-vars-dock" id="kernel-vars-dock">
      <div class="kernel-vars-panel" id="kernel-vars-panel">
        <div class="kernel-vars-toolbar">
          <span class="kernel-vars-title">Kernel variables</span>
          <button type="button" class="btn-icon" id="kernel-vars-refresh" title="Refresh list">⟳</button>
          <button type="button" class="btn-icon" id="kernel-vars-collapse" title="Hide panel">▾</button>
        </div>
        <div class="kernel-vars-scroll">
          <table class="kernel-vars-table" aria-label="Kernel variables">
            <thead><tr><th>Name</th><th>Type</th><th>Value</th></tr></thead>
            <tbody id="kernel-vars-tbody"></tbody>
          </table>
          <p class="kernel-vars-empty" id="kernel-vars-empty" hidden>No user variables (only builtins).</p>
        </div>
      </div>
      <div class="kernel-vars-dock-bar">
        <button type="button" class="cells-auto-fab" id="btn-cells-auto-layout" title="Discard saved positions and reflow cells into the automatic grid (fixes overlap after output or resize)" aria-label="Automatic cell layout">↻</button>
        <button type="button" class="kernel-vars-chip" id="kernel-vars-toggle" aria-expanded="false" title="Show kernel variables">
          <span class="kernel-vars-chip-icon" aria-hidden="true">{ }</span>
          <span class="kernel-vars-chip-count" id="kernel-vars-count"></span>
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
const statusToastEl = app.querySelector<HTMLDivElement>("#status-toast")!;
const btnCellsAutoLayout = app.querySelector<HTMLButtonElement>("#btn-cells-auto-layout")!;
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
const kernelVarsCount = app.querySelector<HTMLSpanElement>("#kernel-vars-count")!;

const KERNEL_VARS_EXPANDED_KEY = "stonesoup_kernel_vars_expanded";

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
  const idx = Number(out.dataset.out);
  if (Number.isNaN(idx)) return;
  const o = outputs.get(idx);
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

type ScriptFileEntry = { rel: string; label: string };

/** Sentinel: ``*.py`` directly under the list root (group key; not a real folder name). */
const SCRIPT_PICKER_ROOT_FOLDER = "__ss_root__";

/** First path segment under list root → files (full repo-relative path + label under that folder). */
let scriptPickerGroups: Map<string, ScriptFileEntry[]> = new Map();

function normalizeRelPath(p: string): string {
  return p.replace(/\\/g, "/").replace(/\/+$/, "");
}

/**
 * Group ``*.py`` paths by the first directory under ``root`` (posix). Files directly under ``root``
 * use ``SCRIPT_PICKER_ROOT_FOLDER``.
 */
function groupPyFilesUnderRoot(root: string, files: string[]): Map<string, ScriptFileEntry[]> {
  const r = normalizeRelPath(root);
  const groups = new Map<string, ScriptFileEntry[]>();
  for (const relRaw of files) {
    const rel = relRaw.replace(/\\/g, "/");
    if (!rel.toLowerCase().endsWith(".py")) continue;
    if (rel === r) continue;
    if (!rel.startsWith(r + "/")) continue;
    const rest = rel.slice(r.length + 1);
    const slash = rest.indexOf("/");
    const folderKey = slash === -1 ? SCRIPT_PICKER_ROOT_FOLDER : rest.slice(0, slash);
    const tail = slash === -1 ? rest : rest.slice(slash + 1);
    if (!tail.toLowerCase().endsWith(".py")) continue;
    const list = groups.get(folderKey) ?? [];
    list.push({ rel, label: tail });
    groups.set(folderKey, list);
  }
  for (const list of groups.values()) {
    list.sort((a, b) => a.label.localeCompare(b.label));
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

async function fetchPyFilesUnderDir(dir: string): Promise<string[]> {
  const params = new URLSearchParams({ dir, recursive: "true" });
  const r = await fetch(`${apiBase}/api/py-files?${params}`);
  const j = (await r.json()) as { files?: string[] };
  if (!r.ok) throw new Error((j as { detail?: string }).detail || r.statusText);
  return j.files ?? [];
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
    let files = await fetchPyFilesUnderDir(listDir);
    scriptPickerGroups = groupPyFilesUnderRoot(listDir, files);

    const want = pathInput.value.trim().replace(/\\/g, "/");
    let chosenKey = pickFolderKeyForPath(want, scriptPickerGroups);
    if (
      chosenKey === null &&
      want &&
      want.startsWith("experiments/") &&
      listDir !== "experiments"
    ) {
      files = await fetchPyFilesUnderDir("experiments");
      listDir = "experiments";
      scriptPickerGroups = groupPyFilesUnderRoot("experiments", files);
      chosenKey = pickFolderKeyForPath(want, scriptPickerGroups);
    }

    const keys = [...scriptPickerGroups.keys()].sort((a, b) => {
      if (a === SCRIPT_PICKER_ROOT_FOLDER) return 1;
      if (b === SCRIPT_PICKER_ROOT_FOLDER) return -1;
      return a.localeCompare(b);
    });

    for (const key of keys) {
      const opt = document.createElement("option");
      opt.value = key;
      opt.textContent = folderPickerLabel(key);
      folderSelect.appendChild(opt);
    }

    if (chosenKey === null && keys.length > 0) {
      chosenKey = keys[0]!;
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
      } else if (entries.length > 0) {
        fileSelect.value = entries[0]!.rel;
        pathInput.value = entries[0]!.rel;
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
  if (entries.length === 0) {
    return;
  }
  fileSelect.value = entries[0]!.rel;
  pathInput.value = entries[0]!.rel;
  void postWatch();
});

fileSelect.addEventListener("change", () => {
  const v = fileSelect.value.trim();
  if (!v) return;
  pathInput.value = v;
  void postWatch();
});

void populateScriptPicker();

let revision = 0;
let lastCells: Cell[] = [];
let lastPath: string | null = null;
let ws: WebSocket | null = null;
const expanded = new Set<number>();
/** Cell indices whose source changed on disk since last successful run (merged from server + cleared on run). */
const staleCells = new Set<number>();

/** Per-cell run input text; merged into kernel as ``CELL_INPUT`` (survives UI re-render). */
const cellRunInputDraft = new Map<number, string>();

/** How stdout is interpreted when not forced to plain via chip; only ``html`` / ``markdown`` enable rich rendering (first-line hint). */
type StdoutKind = "text" | "html" | "markdown";

/** Kernel result for one cell run; ``renderHint`` from optional first stdout line ``# stonesoup:render=…`` (stripped from ``stdout``). */
type CellOutput = { stdout: string; stderr: string; ok: boolean; renderHint?: StdoutKind | null };
const outputs = new Map<number, CellOutput>();

/** When set, stdout is shown escaped (toggle chip); only meaningful for HTML/MD preset outputs. */
const cellStdoutPlainText = new Set<number>();

const STONESOUP_RENDER_FIRST_LINE = /^\s*#\s*stonesoup:render\s*=\s*(auto|text|html|markdown|md)\s*$/i;

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

function cellRunInputValue(index: number): string {
  const el = cellsEl.querySelector<HTMLInputElement>(`[data-run-input="${index}"]`);
  if (el) return el.value;
  return cellRunInputDraft.get(index) ?? "";
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

/** Drop output / expanded state for cell indices that no longer exist after a re-parse. */
function pruneOutputsAndExpanded(cellCount: number) {
  for (const k of [...outputs.keys()]) {
    if (!Number.isInteger(k) || k < 0 || k >= cellCount) outputs.delete(k);
  }
  for (const k of [...expanded]) {
    if (!Number.isInteger(k) || k < 0 || k >= cellCount) expanded.delete(k);
  }
  for (const k of [...cellStdoutPlainText.keys()]) {
    if (!Number.isInteger(k) || k < 0 || k >= cellCount) cellStdoutPlainText.delete(k);
  }
}

function pruneStaleCells(cellCount: number) {
  for (const k of [...staleCells]) {
    if (!Number.isInteger(k) || k < 0 || k >= cellCount) staleCells.delete(k);
  }
}
/** Last-known cell geometry for reflow heuristics; cleared when watched path or cell count changes. */
const cellPositions = new Map<number, { left: number; top: number }>();
/** When non-empty, canvas uses saved left/top/width/height per file cell index (from cookie or after drag). */
const manualLayoutByCellIdx = new Map<number, { left: number; top: number; width: number; height: number }>();
function loadManualLayoutsForPath(pathKey: string) {
  manualLayoutByCellIdx.clear();
  if (!pathKey || pathKey === "_unset") return;
  const all = parseCellLayoutsCookie();
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
/** Last grid column count used for auto layout; when viewport changes columns, we reflow the grid. */
let lastLayoutCols = -1;

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

function setCellRunningState(index: number, running: boolean) {
  const cell = cellsCanvas.querySelector<HTMLElement>(`.cell[data-pipeline-cell-drag="${index}"]`);
  if (!cell) return;
  cell.classList.toggle("cell-running", running);
  if (running) bringCellToFront(cell);
}


/** Clear output and show the output strip so streamed stdout/stderr can appear while the cell runs. */
function prepareCellStreamUi(index: number) {
  const outEl = cellsEl.querySelector<HTMLElement>(`[data-out="${index}"]`);
  if (outEl) {
    outEl.textContent = "";
    outEl.className = "out out-streaming";
  }
  setCellOutputBlockVisible(index, true);
  syncCellCompactClassForIndex(index);
  applyFloatingLayout();
}

function appendCellStreamChunk(index: number, text: string) {
  const outEl = cellsEl.querySelector<HTMLElement>(`[data-out="${index}"]`);
  if (outEl) outEl.textContent += text;
}

function applyCellsFromServer(
  data: {
    revision: number;
    path: string | null;
    cells: Cell[];
    changed_cell_indices?: unknown;
  },
  opts?: { forceResetOutputs?: boolean },
) {
  const incomingPath = data.path ?? null;
  const pathChanged = (incomingPath ?? "") !== (lastPath ?? "");
  if (pathChanged || opts?.forceResetOutputs) {
    outputs.clear();
    cellStdoutPlainText.clear();
    expanded.clear();
    staleCells.clear();
  }
  revision = data.revision;
  const cells = data.cells;
  if (!pathChanged && !opts?.forceResetOutputs) {
    pruneOutputsAndExpanded(cells.length);
    pruneStaleCells(cells.length);
  }
  const changed = data.changed_cell_indices;
  if (Array.isArray(changed)) {
    for (const x of changed) {
      const i = Number(x);
      if (Number.isInteger(i) && i >= 0 && i < cells.length) staleCells.add(i);
    }
  }
  try {
    renderCells(cells, incomingPath);
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

async function fetchKernelVars() {
  try {
    const r = await fetch(`${apiBase}/api/kernel/vars`);
    const j = (await r.json()) as { vars?: { name: string; type: string; preview: string }[] };
    if (!r.ok || !Array.isArray(j.vars)) return;
    const n = j.vars.length;
    kernelVarsCount.textContent = n ? String(n) : "";
    kernelVarsTbody.replaceChildren();
    for (const row of j.vars) {
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

function initKernelVarsDock() {
  const open = kernelVarsStartExpanded();
  kernelVarsDock.classList.toggle("collapsed", !open);
  kernelVarsToggle.setAttribute("aria-expanded", open ? "true" : "false");
  void fetchKernelVars();
}

kernelVarsToggle.addEventListener("click", () => {
  if (!kernelVarsDock.classList.contains("collapsed")) return;
  kernelVarsDock.classList.remove("collapsed");
  kernelVarsToggle.setAttribute("aria-expanded", "true");
  localStorage.setItem(KERNEL_VARS_EXPANDED_KEY, "1");
  void fetchKernelVars();
});

kernelVarsCollapse.addEventListener("click", () => {
  kernelVarsDock.classList.add("collapsed");
  kernelVarsToggle.setAttribute("aria-expanded", "false");
  localStorage.setItem(KERNEL_VARS_EXPANDED_KEY, "0");
});

kernelVarsRefresh.addEventListener("click", () => void fetchKernelVars());

function connectWs() {
  ws?.close();
  ws = new WebSocket(wsUrl());
  ws.onopen = () => {
    wsDot.classList.add("on");
    wsDot.title = "Live reload: connected";
    wsDot.setAttribute("aria-label", "WebSocket connected");
  };
  ws.onclose = () => {
    wsDot.classList.remove("on");
    wsDot.title = "Live reload: disconnected (retrying…)";
    wsDot.setAttribute("aria-label", "WebSocket disconnected");
    setTimeout(connectWs, 2000);
  };
  ws.onmessage = (ev) => {
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
        scheduleKernelVarsRefresh();
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

function tab20Accent(index: number): string {
  const i = Math.max(0, Math.floor(index));
  return TAB20[i % TAB20.length]!;
}

function applyCellColorVars(el: HTMLElement, index: number) {
  el.style.setProperty("--cell-accent", tab20Accent(index));
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
  | { kind: "move"; fromPath: number[]; fromPipeline: number };

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
        applyCellColorVars(chip, idx);
        const dragGrip = document.createElement("span");
        dragGrip.className = "pipeline-chip-drag";
        dragGrip.draggable = true;
        dragGrip.dataset.pipelineChipDrag = pathJson;
        dragGrip.title = "Drag to move in pipeline";
        dragGrip.textContent = "⠿";
        dragGrip.setAttribute("aria-hidden", "true");
        const idxSpan = document.createElement("span");
        idxSpan.className = "chip-idx";
        idxSpan.textContent = String(idx);
        idxSpan.title = "Cell index in file (0-based), same as on the canvas";
        const titleSpan = document.createElement("span");
        titleSpan.className = "chip-title";
        titleSpan.title = title;
        titleSpan.textContent = title;
        const bRm = document.createElement("button");
        bRm.type = "button";
        bRm.dataset.pPath = pathJson;
        bRm.dataset.pPipeline = String(pIdx);
        bRm.dataset.pRemove = "1";
        bRm.title = "Remove";
        bRm.textContent = "×";
        chip.append(dragGrip, idxSpan, titleSpan, bRm);
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

  for (let pIdx = 0; pIdx < pipelines.length; pIdx++) {
    const program = pipelines[pIdx]!;
    const block = document.createElement("div");
    block.className = "pipeline-block";

    if (pipelines.length > 1) {
      const toolbar = document.createElement("div");
      toolbar.className = "pipeline-block-toolbar";
      const bStripRm = document.createElement("button");
      bStripRm.type = "button";
      bStripRm.className = "btn-icon pipeline-strip-remove";
      bStripRm.dataset.removePipelineStrip = String(pIdx);
      bStripRm.title = "Remove this pipeline";
      bStripRm.setAttribute("aria-label", "Remove pipeline");
      bStripRm.textContent = "−";
      toolbar.appendChild(bStripRm);
      block.appendChild(toolbar);
    }

    const shell = document.createElement("div");
    shell.className = "pipeline-chips";
    shell.dataset.pipelineIndex = String(pIdx);

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
    stack.appendChild(block);
  }

  const addWrap = document.createElement("div");
  addWrap.className = "pipeline-add-strip";
  const bAdd = document.createElement("button");
  bAdd.type = "button";
  bAdd.className = "btn-icon";
  bAdd.title = "Add another pipeline";
  bAdd.setAttribute("aria-label", "Add pipeline");
  bAdd.textContent = "＋";
  bAdd.addEventListener("click", () => {
    pipelines.push([]);
    savePipeline();
    renderPipelineBar();
    highlightPipelineCells();
  });
  addWrap.appendChild(bAdd);
  stack.appendChild(addWrap);

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
  const viewportW = cellsEl.clientWidth;
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

function applyFloatingLayout() {
  const { pad, gap, cellW, cols } = computeCellGridParams();
  const nodes = [...cellsCanvas.querySelectorAll<HTMLElement>(".cell[data-pipeline-cell-drag]")];
  if (manualLayoutByCellIdx.size === 0) {
    const needHorizontalReflow = lastLayoutCols !== cols || cellPositions.size === 0;
    if (needHorizontalReflow) {
      lastLayoutCols = cols;
    }
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
      scheduleLayoutAndLines();
    });
    return;
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
      /* Compact cards save a small height; do not floor to CELL_LAYOUT_MIN_H or every header-only cell becomes a tall empty box after any full re-layout (e.g. toggling Code). */
      const showOutForLayout =
        hasCellBodyOutput(outputs.get(idx)) || isOutputStripVisible(idx);
      const compact =
        Number.isInteger(idx) && !expanded.has(idx) && !showOutForLayout;
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
  });
}

function renderCells(cells: Cell[], path: string | null) {
  lastCells = cells;
  lastPath = path;
  const p = path ?? pathInput.value.trim();
  const pathChangedForScrollReset = p !== lastLayoutPath;
  if (p !== lastLayoutPath || cells.length !== lastLayoutCount) {
    cellPositions.clear();
    lastLayoutCols = -1;
    if (p !== lastLayoutPath) {
      if (!p) {
        manualLayoutByCellIdx.clear();
      } else {
        loadManualLayoutsForPath(p);
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
  }

  const validCellIdx = new Set(cells.map((c) => c.index));
  for (const k of [...cellRunInputDraft.keys()]) {
    if (!validCellIdx.has(k)) cellRunInputDraft.delete(k);
  }
  for (const c of cells) {
    if (!c.cell_input) cellRunInputDraft.delete(c.index);
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
    applyCellColorVars(div, c.index);
    const prev = outputs.get(c.index);
    const showOut = hasCellBodyOutput(prev);
    const outRendered = showOut && prev ? renderOutputInnerHtml(prev, c.index) : null;
    const outRichClass = outRendered?.richLayout ? " out-rich" : "";
    const exp = expanded.has(c.index);
    if (!exp && !showOut) div.classList.add("cell-compact");
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
            <button type="button" class="toggle" draggable="false" data-toggle="${c.index}" title="${exp ? "Hide source" : "Show source"}">${exp ? "Hide" : "Code"}</button>
            <button type="button" class="btn-chain" draggable="false" data-pipeline-add="${c.index}" title="Append to pipeline">+ chain</button>
            ${runInputHtml}
            <button type="button" class="primary" draggable="false" data-run="${c.index}">Run</button>
          </div>
        </div>
        <div class="cell-code-panel" style="display:${exp ? "block" : "none"}">
          <pre class="source full"><code class="language-python hljs">${safeHighlightPython(c.source)}</code></pre>
        </div>
        <div class="cell-output-block" data-output-block="${c.index}" style="display:${showOut ? "flex" : "none"}">
          <div class="out-label-row" draggable="false">
            <span class="out-label" draggable="false">Output</span>
          </div>
          <div class="out ${prev && !prev.ok ? "err" : prev ? "ok" : "out-pending"}${outRichClass}" draggable="false" data-out="${c.index}" title="Click to copy">${outRendered ? outRendered.html : ""}</div>
        </div>
      </div>
      <div class="cell-resize-handle" draggable="false" title="Drag corner to resize"></div>
    `;
    cellsCanvas.appendChild(div);
    syncOutLabelRowForCell(c.index);
    const runInp = div.querySelector<HTMLInputElement>(`[data-run-input="${c.index}"]`);
    if (runInp) {
      runInp.value = cellRunInputDraft.get(c.index) ?? "";
      const stopHeadDrag = (e: Event) => e.stopPropagation();
      runInp.addEventListener("pointerdown", stopHeadDrag);
      runInp.addEventListener("mousedown", stopHeadDrag);
      runInp.addEventListener("input", () => cellRunInputDraft.set(c.index, runInp.value));
      runInp.addEventListener("keydown", (e: KeyboardEvent) => {
        if (e.key !== "Enter" || (!e.ctrlKey && !e.metaKey)) return;
        e.preventDefault();
        void runCell(c.index);
      });
    }
  }

  cellsCanvas.querySelectorAll("[data-run]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const idx = Number((btn as HTMLButtonElement).dataset.run);
      runCell(idx);
    });
  });
  cellsCanvas.querySelectorAll("[data-toggle]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const idx = Number((btn as HTMLButtonElement).dataset.toggle);
      if (expanded.has(idx)) expanded.delete(idx);
      else expanded.add(idx);
      renderCells(lastCells, lastPath);
    });
  });

  cellsCanvas.querySelectorAll<HTMLButtonElement>("[data-pipeline-add]").forEach((btn) => {
    btn.addEventListener("click", (ev) => {
      ev.stopPropagation();
      appendToPipeline(Number(btn.dataset.pipelineAdd));
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

  if (pathChangedForScrollReset) {
    const g = CELLS_PAN_GUTTER_PX;
    requestAnimationFrame(() => {
      cellsEl.scrollLeft = g;
      cellsEl.scrollTop = g;
      requestAnimationFrame(() => {
        cellsEl.scrollLeft = g;
        cellsEl.scrollTop = g;
      });
    });
  }
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

function renderOutputInnerHtml(o: CellOutput, cellIndex: number): { html: string; richLayout: boolean } {
  const hasBody = Boolean(o.stdout.trim() || o.stderr.trim());
  if (!hasBody) {
    return { html: escapeHtml(formatOut(o)), richLayout: false };
  }
  const asPlain = cellStdoutPlainText.has(cellIndex);
  const { html: outHtml, rich: stdoutRich } = renderStdoutHtml(o.stdout, asPlain, o.renderHint);
  let html = outHtml;
  if (o.stderr.trim()) {
    html += `<pre class="stonesoup-stderr">${escapeHtml(o.stderr)}</pre>`;
  }
  return { html, richLayout: stdoutRich || Boolean(o.stderr.trim()) };
}

/** Creates/updates/removes the HTML/MD ↔ plain chip in the output header. */
function syncOutLabelRowForCell(index: number) {
  const o = outputs.get(index);
  const block = cellsEl.querySelector<HTMLElement>(`[data-output-block="${index}"]`);
  if (!block) return;
  const row = block.querySelector<HTMLElement>(".out-label-row");
  if (!row) return;
  const wantChip = Boolean(o && showOutputRichToggleForCell(o));
  let chip = row.querySelector<HTMLButtonElement>(`[data-out-plain-toggle="${index}"]`);
  if (!wantChip) {
    chip?.remove();
    return;
  }
  const preset = presetRichKind(o!);
  if (!preset) {
    chip?.remove();
    return;
  }
  const asPlain = cellStdoutPlainText.has(index);
  if (!chip) {
    chip = document.createElement("button");
    chip.type = "button";
    chip.className = "out-mode-chip";
    chip.draggable = false;
    chip.dataset.outPlainToggle = String(index);
    chip.setAttribute("aria-label", "Toggle plain text");
    chip.addEventListener("click", (ev) => {
      ev.stopPropagation();
      const i = Number(chip!.dataset.outPlainToggle);
      if (!Number.isInteger(i)) return;
      if (cellStdoutPlainText.has(i)) cellStdoutPlainText.delete(i);
      else cellStdoutPlainText.add(i);
      refreshCellOutputView(i);
    });
    row.appendChild(chip);
  }
  chip.textContent = outputPlainToggleLabel(preset, asPlain);
  chip.title = outputPlainToggleTitle(preset, asPlain);
}

function refreshCellOutputView(index: number) {
  syncOutLabelRowForCell(index);
  const o = outputs.get(index);
  const outEl = cellsEl.querySelector<HTMLElement>(`[data-out="${index}"]`);
  if (!outEl || !o) return;
  const visible = hasCellBodyOutput(o);
  if (!visible) return;
  if (outEl.classList.contains("out-streaming")) return;
  const r = renderOutputInnerHtml(o, index);
  outEl.className = "out " + (o.ok ? "ok" : "err") + (r.richLayout ? " out-rich" : "");
  outEl.innerHTML = r.html;
}

/** True when there is real stdout/stderr to show (placeholders like “no stdout” are not shown in the UI). */
function hasCellBodyOutput(o: { stdout: string; stderr: string } | undefined): boolean {
  if (!o) return false;
  return Boolean(o.stdout.trim() || o.stderr.trim());
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
  return d === "flex" || d === "block";
}

/** Toolbar lives in `.cell-body`; compact = no code + no output so the card is header-only. */
function syncCellCompactClassForIndex(index: number) {
  const cell = cellsCanvas.querySelector<HTMLElement>(`.cell[data-pipeline-cell-drag="${index}"]`);
  if (!cell) return;
  const showOut =
    hasCellBodyOutput(outputs.get(index)) || isOutputStripVisible(index);
  const exp = expanded.has(index);
  cell.classList.toggle("cell-compact", !exp && !showOut);
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

function highlightPython(source: string): string {
  try {
    return hljs.highlight(source, { language: "python", ignoreIllegals: true }).value;
  } catch {
    return escapeHtml(source);
  }
}

function safeHighlightPython(source: string): string {
  try {
    return highlightPython(source);
  } catch (err) {
    console.error("stonesoup: highlightPython failed", err);
    return escapeHtml(source);
  }
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
    const j = (await r.json()) as {
      detail?: string;
      revision?: number;
      n_cells?: number;
      path?: string | null;
      cells?: Cell[];
      changed_cell_indices?: unknown;
    };
    if (!r.ok) throw new Error(j.detail || r.statusText);
    saveWatchPathCookie(path);
    if (Array.isArray(j.cells)) {
      applyCellsFromServer(
        {
          revision: Number(j.revision) || 0,
          path: typeof j.path === "string" || j.path === null ? j.path : null,
          cells: j.cells,
          changed_cell_indices: j.changed_cell_indices,
        },
        { forceResetOutputs: true },
      );
    } else {
      outputs.clear();
      cellStdoutPlainText.clear();
      expanded.clear();
      staleCells.clear();
      revision = j.revision ?? revision;
    }
    setStatus(`watching · rev ${j.revision} · ${j.n_cells} cells`);
    scheduleKernelVarsRefresh();
  } catch (e) {
    setStatus(String(e));
  } finally {
    btnWatch.disabled = false;
  }
}

async function runCell(index: number, inject?: Record<string, unknown> | null) {
  const btn = cellsEl.querySelector<HTMLButtonElement>(`[data-run="${index}"]`);
  prepareCellStreamUi(index);
  setCellRunningState(index, true);
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
    const j = await r.json();
    if (!r.ok) throw new Error(j.detail || r.statusText);
    const peeled = peelStonesoupRenderHint(typeof j.stdout === "string" ? j.stdout : "");
    const nextOut: CellOutput = {
      stdout: peeled.body,
      stderr: j.stderr || "",
      ok: j.ok,
    };
    if (peeled.renderHint != null) nextOut.renderHint = peeled.renderHint;
    if (presetRichKind(nextOut) === null) cellStdoutPlainText.delete(index);
    outputs.set(index, nextOut);
    const outEl = cellsEl.querySelector<HTMLElement>(`[data-out="${index}"]`);
    const visible = hasCellBodyOutput(nextOut);
    setCellOutputBlockVisible(index, visible);
    if (outEl) {
      outEl.classList.remove("out-streaming");
      const baseClass = visible ? "out " + (j.ok ? "ok" : "err") : "out out-pending";
      if (visible) {
        refreshCellOutputView(index);
      } else {
        outEl.className = baseClass;
        outEl.textContent = "";
        syncOutLabelRowForCell(index);
      }
    }
    syncCellCompactClassForIndex(index);
    applyFloatingLayout();
    if (j.ok) {
      staleCells.delete(index);
      syncCellStaleClassForIndex(index);
      renderPipelineBar();
    }
  } catch (e) {
    const nextOut = { stdout: "", stderr: String(e), ok: false };
    outputs.set(index, nextOut);
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
    applyFloatingLayout();
  } finally {
    setCellRunningState(index, false);
    if (btn) btn.disabled = false;
  }
}

async function resetKernel() {
  btnReset.disabled = true;
  try {
    const r = await fetch(`${apiBase}/api/reset`, { method: "POST" });
    if (!r.ok) throw new Error(await r.text());
    outputs.clear();
    cellStdoutPlainText.clear();
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
    applyFloatingLayout();
    setStatus(`rev ${revision} · kernel reset`);
    scheduleKernelVarsRefresh();
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
      const o = outputs.get(step.index);
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
    setStatus(`Pipeline ${pIdx + 1} is empty — use + chain or drag ↻ Loop below`);
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

btnWatch.addEventListener("click", () => postWatch());
btnReset.addEventListener("click", () => resetKernel());

window.addEventListener("resize", () => {
  if (lastCells.length) applyFloatingLayout();
  requestAnimationFrame(() => updatePipelineLineBreakMarkers());
});

function clearPipelineDropHighlights() {
  document.querySelectorAll(".pipeline-drop-zone.is-drag-over").forEach((el) => {
    el.classList.remove("is-drag-over");
  });
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

function parseDndPayload(dt: DataTransfer): DndPayload | null {
  try {
    const raw = dt.getData(DND_PAYLOAD);
    if (!raw) return null;
    const x = JSON.parse(raw) as DndPayload;
    if (x.kind === "canvas" && typeof x.cellIndex === "number" && Number.isInteger(x.cellIndex))
      return x;
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
        const shell = pg.closest<HTMLElement>(".pipeline-chips");
        const fromPipeline = Number(shell?.dataset.pipelineIndex);
        if (!Number.isInteger(fromPipeline)) return;
        const fromPath = JSON.parse(pg.dataset.pipelineChipDrag!) as number[];
        const payload: DndPayload = { kind: "move", fromPath, fromPipeline };
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
    dt.dropEffect = activePipelineDnd.kind === "move" ? "move" : "copy";
    clearPipelineDropHighlights();
    const z = hitTestPipelineDropZone(e.clientX, e.clientY);
    if (z) z.classList.add("is-drag-over");
  });

  document.addEventListener("drop", (e) => {
    if (!pipelineRow || !pipelineRow.contains(e.target as Node)) return;
    e.preventDefault();
    clearPipelineDropHighlights();
    const payload = parseDndPayload(e.dataTransfer!);
    if (!payload) return;
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
      });
    },
    { passive: false },
  );
}

bindCellGeometryInteractions();
bindCellsViewportPan();
bindPipelineDnD();
renderPipelineBar();
initKernelVarsDock();
connectWs();
postWatch();
