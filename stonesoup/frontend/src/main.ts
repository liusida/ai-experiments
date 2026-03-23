import "./style.css";

type Cell = { index: number; title: string; source: string; marker_key: string };

const apiBase = import.meta.env.DEV ? "" : "http://127.0.0.1:8765";

function wsUrl(): string {
  if (import.meta.env.DEV) {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    return `${proto}//${location.host}/ws`;
  }
  return "ws://127.0.0.1:8765/ws";
}

const app = document.querySelector<HTMLDivElement>("#app")!;

const defaultPath =
  new URLSearchParams(location.search).get("path") ||
  "2026-03-23-Embedding/demo.py";

app.innerHTML = `
  <div class="toolbar">
    <span class="ws-dot" id="ws-dot" title="WebSocket"></span>
    <input type="text" id="path-input" placeholder="path under repo" spellcheck="false" />
    <button type="button" class="primary" id="btn-watch">Watch</button>
    <button type="button" id="btn-reset">Reset kernel</button>
    <span class="status" id="status"></span>
  </div>
  <div class="pipeline-row" id="pipeline-row">
    <span class="pipeline-label">Pipelines</span>
    <div class="pipelines-stack" id="pipelines-stack"></div>
  </div>
  <div class="workspace">
    <div class="cells" id="cells"><div class="cells-canvas" id="cells-canvas"></div></div>
  </div>
`;

const pathInput = app.querySelector<HTMLInputElement>("#path-input")!;
const btnWatch = app.querySelector<HTMLButtonElement>("#btn-watch")!;
const btnReset = app.querySelector<HTMLButtonElement>("#btn-reset")!;
const statusEl = app.querySelector<HTMLSpanElement>("#status")!;
const cellsEl = app.querySelector<HTMLDivElement>("#cells")!;
const cellsCanvas = app.querySelector<HTMLDivElement>("#cells-canvas")!;
const wsDot = app.querySelector<HTMLSpanElement>("#ws-dot")!;

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

let revision = 0;
let lastCells: Cell[] = [];
let lastPath: string | null = null;
let ws: WebSocket | null = null;
const outputs = new Map<number, { stdout: string; stderr: string; ok: boolean }>();
const expanded = new Set<number>();
/** Cell indices whose source changed on disk since last successful run (merged from server + cleared on run). */
const staleCells = new Set<number>();

/** Drop output / expanded state for cell indices that no longer exist after a re-parse. */
function pruneOutputsAndExpanded(cellCount: number) {
  for (const k of [...outputs.keys()]) {
    if (!Number.isInteger(k) || k < 0 || k >= cellCount) outputs.delete(k);
  }
  for (const k of [...expanded]) {
    if (!Number.isInteger(k) || k < 0 || k >= cellCount) expanded.delete(k);
  }
}

function pruneStaleCells(cellCount: number) {
  for (const k of [...staleCells]) {
    if (!Number.isInteger(k) || k < 0 || k >= cellCount) staleCells.delete(k);
  }
}
/** Last-known cell geometry for reflow heuristics; cleared when watched path or cell count changes. */
const cellPositions = new Map<number, { left: number; top: number }>();
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

function setStatus(msg: string) {
  statusEl.textContent = msg;
}

function setCellRunningState(index: number, running: boolean) {
  const cell = cellsCanvas.querySelector<HTMLElement>(`.cell[data-pipeline-cell-drag="${index}"]`);
  if (cell) cell.classList.toggle("cell-running", running);
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
  reflowCellStack();
}

function appendCellStreamChunk(index: number, text: string) {
  const outEl = cellsEl.querySelector<HTMLElement>(`[data-out="${index}"]`);
  if (outEl) outEl.textContent += text;
}

function connectWs() {
  ws?.close();
  ws = new WebSocket(wsUrl());
  ws.onopen = () => {
    wsDot.classList.add("on");
  };
  ws.onclose = () => {
    wsDot.classList.remove("on");
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
        const incomingPath = (data as { path?: string | null }).path ?? null;
        const pathChanged = (incomingPath ?? "") !== (lastPath ?? "");
        if (pathChanged) {
          outputs.clear();
          expanded.clear();
          staleCells.clear();
        }
        revision = (data as { revision?: number }).revision ?? revision;
        const cells = (data as { cells: Cell[] }).cells;
        if (!pathChanged) {
          pruneOutputsAndExpanded(cells.length);
          pruneStaleCells(cells.length);
        }
        const changed = (data as { changed_cell_indices?: unknown }).changed_cell_indices;
        if (Array.isArray(changed)) {
          for (const x of changed) {
            const i = Number(x);
            if (Number.isInteger(i) && i >= 0 && i < cells.length) staleCells.add(i);
          }
        }
        renderCells(cells, incomingPath);
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

function relayoutCanvasBounds() {
  const padX = 48;
  /** Extra space below the lowest cell so the canvas can scroll vertically. */
  const padBottom = 420;
  let maxBottom = 0;
  let maxRight = 0;
  cellsCanvas.querySelectorAll<HTMLElement>(".cell, .loop-palette").forEach((c) => {
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
  | { kind: "new-loop" }
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

    const flow = document.createElement("div");
    flow.className = "pipeline-chips-flow";
    renderLevel(flow, program, [], null, pIdx);

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

    shell.append(flow, actions);
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

/** Re-pack vertical positions after cell heights change (e.g. compact ↔ expanded) without resetting horizontal grid. */
function reflowCellStack() {
  const { pad, gap, cellW, cols } = computeCellGridParams();
  const nodes = [...cellsCanvas.querySelectorAll<HTMLElement>(".cell")];
  requestAnimationFrame(() => {
    packGridRows(nodes, cols, cellW, pad, gap);
    syncCellPositionsFromDom(nodes);
    const lp = cellsCanvas.querySelector<HTMLElement>(".loop-palette");
    if (lp) {
      let maxBottom = pad;
      for (const cell of nodes) {
        maxBottom = Math.max(maxBottom, cell.offsetTop + cell.offsetHeight);
      }
      lp.style.left = `${pad}px`;
      lp.style.top = `${maxBottom + gap}px`;
      lp.style.width = `${cellW}px`;
    }
    scheduleLayoutAndLines();
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
  const nodes = [...cellsCanvas.querySelectorAll<HTMLElement>(".cell")];
  const needHorizontalReflow = lastLayoutCols !== cols || cellPositions.size === 0;

  if (needHorizontalReflow) {
    lastLayoutCols = cols;
  }
  nodes.forEach((cell, i) => {
    cell.dataset.cellIndex = String(i);
    cell.style.width = `${cellW}px`;
    cell.style.left = `${pad + (i % cols) * (cellW + gap)}px`;
    cell.style.top = `${pad}px`;
  });
  requestAnimationFrame(() => {
    packGridRows(nodes, cols, cellW, pad, gap);
    syncCellPositionsFromDom(nodes);
    const lp = cellsCanvas.querySelector<HTMLElement>(".loop-palette");
    if (lp) {
      let maxBottom = pad;
      for (const cell of nodes) {
        maxBottom = Math.max(maxBottom, cell.offsetTop + cell.offsetHeight);
      }
      lp.style.left = `${pad}px`;
      lp.style.top = `${maxBottom + gap}px`;
      lp.style.width = `${cellW}px`;
    }
    scheduleLayoutAndLines();
  });
}

function renderCells(cells: Cell[], path: string | null) {
  lastCells = cells;
  lastPath = path;
  const p = path ?? pathInput.value.trim();
  if (p !== lastLayoutPath || cells.length !== lastLayoutCount) {
    cellPositions.clear();
    lastLayoutCols = -1;
    if (p !== lastLayoutPath) {
      pipelines = loadPipelines(cells.length);
      if (pipelines.length === 0) pipelines = [[]];
      clearLoopExpanded();
    } else {
      pipelines = pipelines.map((pl) => sanitizeProgram(pl, cells.length));
      clearLoopExpanded();
    }
    lastLayoutPath = p;
    lastLayoutCount = cells.length;
  }

  cellsCanvas.innerHTML = "";

  for (const c of cells) {
    const stale = staleCells.has(c.index);
    const div = document.createElement("div");
    div.className = "cell";
    div.draggable = true;
    div.dataset.pipelineCellDrag = String(c.index);
    div.title = stale
      ? "Source changed on disk — re-run to clear. Drag into pipeline (anywhere except output and buttons)"
      : "Drag into pipeline (anywhere except output and buttons)";
    if (stale) div.classList.add("cell-stale");
    applyCellColorVars(div, c.index);
    const prev = outputs.get(c.index);
    const showOut = hasCellBodyOutput(prev);
    const exp = expanded.has(c.index);
    if (!exp && !showOut) div.classList.add("cell-compact");
    div.innerHTML = `
      <div class="cell-body">
        <div class="cell-head">
          <span class="cell-pipeline-drag" aria-hidden="true">⠿</span>
          <span class="cell-idx" title="Cell index in file (0-based)">${c.index}</span>
          <span class="cell-updated-badge" draggable="false" ${stale ? "" : "hidden"} title="This cell's code changed on disk; run it to clear">Updated</span>
          <span class="cell-title">${escapeHtml(c.title)}</span>
          <button type="button" class="toggle" draggable="false" data-toggle="${c.index}" title="${exp ? "Hide source" : "Show source"}">${exp ? "Hide" : "Code"}</button>
          <button type="button" class="btn-chain" draggable="false" data-pipeline-add="${c.index}" title="Append to pipeline">+ chain</button>
          <button type="button" class="primary" draggable="false" data-run="${c.index}">Run</button>
        </div>
        <div class="cell-code-panel" style="display:${exp ? "block" : "none"}">
          <pre class="source full">${escapeHtml(c.source)}</pre>
        </div>
        <div class="cell-output-block" data-output-block="${c.index}" style="display:${showOut ? "flex" : "none"}">
          <div class="out-label" draggable="false">Output</div>
          <div class="out ${prev && !prev.ok ? "err" : prev ? "ok" : "out-pending"}" draggable="false" data-out="${c.index}" title="Click to copy">${showOut && prev ? escapeHtml(formatOut(prev)) : ""}</div>
        </div>
      </div>
    `;
    cellsCanvas.appendChild(div);
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
  loopPal.className = "loop-palette";
  loopPal.draggable = true;
  loopPal.dataset.pipelineNewLoop = "1";
  loopPal.title = "Drag into pipeline to insert an empty loop";
  loopPal.innerHTML = `
    <div class="loop-palette-head">
      <span class="loop-palette-grip" aria-hidden="true">⠿</span>
      <span class="loop-palette-label">↻ Loop</span>
      <span class="loop-palette-hint">Drop on pipeline bar</span>
    </div>
  `;
  cellsCanvas.appendChild(loopPal);

  applyFloatingLayout();
  renderPipelineBar();
  highlightPipelineCells();
}

function formatOut(o: { stdout: string; stderr: string; ok: boolean }) {
  let s = "";
  if (o.stdout) s += o.stdout;
  if (o.stderr) s += (s ? "\n" : "") + o.stderr;
  if (!s) s = o.ok ? "(finished — no stdout/stderr)" : "(failed)";
  return s;
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

/** Toolbar lives in `.cell-body`; compact = no code + no output so the card is header-only. */
function syncCellCompactClassForIndex(index: number) {
  const cell = cellsCanvas.querySelector<HTMLElement>(`.cell[data-pipeline-cell-drag="${index}"]`);
  if (!cell) return;
  const showOut = hasCellBodyOutput(outputs.get(index));
  const exp = expanded.has(index);
  cell.classList.toggle("cell-compact", !exp && !showOut);
}

function syncCellStaleClassForIndex(index: number) {
  const cell = cellsCanvas.querySelector<HTMLElement>(`.cell[data-pipeline-cell-drag="${index}"]`);
  if (!cell) return;
  const stale = staleCells.has(index);
  cell.classList.toggle("cell-stale", stale);
  cell.title = stale
    ? "Source changed on disk — re-run to clear. Drag into pipeline (anywhere except output and buttons)"
    : "Drag into pipeline (anywhere except output and buttons)";
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
    setStatus("Enter a path");
    return;
  }
  btnWatch.disabled = true;
  try {
    const r = await fetch(`${apiBase}/api/watch`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path }),
    });
    const j = await r.json();
    if (!r.ok) throw new Error(j.detail || r.statusText);
    outputs.clear();
    expanded.clear();
    staleCells.clear();
    setStatus(`watching · rev ${j.revision} · ${j.n_cells} cells`);
    revision = j.revision;
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
    const body: Record<string, unknown> = { cell_index: index };
    if (inject != null) body.inject = inject;
    const r = await fetch(`${apiBase}/api/run`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const j = await r.json();
    if (!r.ok) throw new Error(j.detail || r.statusText);
    const nextOut = {
      stdout: j.stdout || "",
      stderr: j.stderr || "",
      ok: j.ok,
    };
    outputs.set(index, nextOut);
    const outEl = cellsEl.querySelector(`[data-out="${index}"]`);
    const visible = hasCellBodyOutput(nextOut);
    setCellOutputBlockVisible(index, visible);
    if (outEl) {
      outEl.className = visible ? "out " + (j.ok ? "ok" : "err") : "out out-pending";
      outEl.textContent = visible ? formatOut(nextOut) : "";
    }
    syncCellCompactClassForIndex(index);
    reflowCellStack();
    if (j.ok) {
      staleCells.delete(index);
      syncCellStaleClassForIndex(index);
      renderPipelineBar();
    }
  } catch (e) {
    const nextOut = { stdout: "", stderr: String(e), ok: false };
    outputs.set(index, nextOut);
    const outEl = cellsEl.querySelector(`[data-out="${index}"]`);
    setCellOutputBlockVisible(index, true);
    if (outEl) {
      outEl.className = "out err";
      outEl.textContent = String(e);
    }
    syncCellCompactClassForIndex(index);
    reflowCellStack();
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
    reflowCellStack();
    setStatus(`rev ${revision} · kernel reset`);
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
    if (x.kind === "new-loop") return { kind: "new-loop" };
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
      const nlp = t.closest<HTMLElement>("[data-pipeline-new-loop]");
      if (nlp && cellsCanvas.contains(nlp)) {
        const payload: DndPayload = { kind: "new-loop" };
        activePipelineDnd = payload;
        e.dataTransfer!.setData(DND_PAYLOAD, JSON.stringify(payload));
        e.dataTransfer!.effectAllowed = "copy";
        return;
      }
      const cg = t.closest<HTMLElement>("[data-pipeline-cell-drag]");
      if (cg && cellsEl.contains(cg)) {
        if (t.closest("button, textarea, input, select, .out, .out-label")) {
          e.preventDefault();
          return;
        }
        const cellIndex = Number(cg.dataset.pipelineCellDrag);
        if (!Number.isInteger(cellIndex)) return;
        const payload: DndPayload = { kind: "canvas", cellIndex };
        activePipelineDnd = payload;
        e.dataTransfer!.setData(DND_PAYLOAD, JSON.stringify(payload));
        e.dataTransfer!.effectAllowed = "copy";
        return;
      }
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
    } else if (payload.kind === "new-loop") {
      insertNewLoopInPipeline(bodyLoopPath, at, toPIdx);
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

bindPipelineDnD();
renderPipelineBar();
connectWs();
postWatch();
