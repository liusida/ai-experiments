# %% Qwen3.5 vocab embedding shape
import stonesoup
MODEL_ID = "Qwen/Qwen3.5-0.8B"

model, processor = stonesoup.load_model(MODEL_ID)
tokenizer = processor.tokenizer
emb = model.get_input_embeddings().weight
print(emb.shape)

# %% Sample tokens
for tid in [*range(12), 256, 10_000, emb.shape[0] - 1]:
    tok = tokenizer.convert_ids_to_tokens([tid])[0]
    print(f"{tid:7d} -> {tok!r}")
for name in ("eos_token_id", "pad_token_id", "bos_token_id"):
    tid = getattr(tokenizer, name, None)
    if tid is not None:
        tok = tokenizer.convert_ids_to_tokens([tid])[0]
        print(f"{name:16s} {tid:7d} -> {tok!r}")

# %% Count top-k tokens by popularity
import stonesoup
import torch
from datasets import load_dataset

TOP_K = 2**14
BATCH = 128
V = emb.shape[0]
counts = torch.zeros(V, dtype=torch.long)
_ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
_buf: list[str] = []
for _row_i, row in enumerate(_ds):
    t = row["text"].strip()
    if t:
        _buf.append(t)
    if len(_buf) >= BATCH:
        for ids in tokenizer(_buf, add_special_tokens=False, padding=False, truncation=False)[
            "input_ids"
        ]:
            counts += torch.bincount(torch.tensor(ids, dtype=torch.long), minlength=V)
        _buf.clear()
    if _row_i > 0 and _row_i % 2048 == 0:
        stonesoup.check_abort()
if _buf:
    for ids in tokenizer(_buf, add_special_tokens=False, padding=False, truncation=False)["input_ids"]:
        counts += torch.bincount(torch.tensor(ids, dtype=torch.long), minlength=V)

_topv, _topi = torch.topk(counts, k=min(TOP_K, V))
print(f"wikitext-2-raw-v1 train — tokens in vocab: {V}, nonzero ids: {(counts > 0).sum().item()}")
SHOW = min(25, _topi.numel())
for _rank, (_c, _tid) in enumerate(zip(_topv.tolist()[:SHOW], _topi.tolist()[:SHOW]), 1):
    _tok = tokenizer.convert_ids_to_tokens([int(_tid)])[0]
    print(f"{_rank:3d}  id={int(_tid):7d}  count={int(_c):9d}  {_tok!r}")
if _topi.numel() > SHOW:
    print(f"... ({SHOW} of {_topi.numel()} ids by count; full list in _topi)")

# %% Pick common-token embedding matrix
common_token_ids = _topi.to(dtype=torch.long).contiguous()
emb_common = emb.index_select(0, common_token_ids.to(device=emb.device))
print("common_token_ids:", common_token_ids.shape, "emb_common:", tuple(emb_common.shape))

# %% Pairwise cosine similarity (common embeddings)
import sys
import matplotlib.pyplot as plt
import torch
import stonesoup

N = emb_common.shape[0]
with torch.inference_mode():
    U = torch.nn.functional.normalize(emb_common.float(), dim=-1)
    cos = U @ U.T
print(
    f"cos shape {tuple(cos.shape)} (~{cos.numel() * 4 / 1e9:.2f} GiB float32)",
    file=sys.stderr,
    flush=True,
)

_d = cos.diag()
mean_off = (cos.sum() - _d.sum()) / (N * (N - 1))
_sq = cos * cos
mean_sq_off = (_sq.sum() - (_d * _d).sum()) / (N * (N - 1))
std_off = (mean_sq_off - mean_off * mean_off).clamp(min=0).sqrt()

_c = cos.clone()
_c.fill_diagonal_(float("inf"))
min_off = _c.min().item()
_c.fill_diagonal_(float("-inf"))
max_off = _c.max().item()
print(
    f"off-diagonal cosine: mean={mean_off:.5f} std={std_off:.5f} min={min_off:.5f} max={max_off:.5f}",
    file=sys.stderr,
    flush=True,
)

HIST_SAMPLE = 2**20
tri_i, tri_j = torch.triu_indices(N, N, offset=1, device=cos.device)
_k = torch.randint(0, tri_i.numel(), (HIST_SAMPLE,), device=cos.device)
_sample = cos[tri_i[_k], tri_j[_k]].detach().float().cpu().numpy()
plt.hist(_sample, bins=120, color="steelblue", alpha=0.85)
plt.xlabel("cosine similarity"); plt.ylabel("count"); plt.title(f"pairwise cos sim (sample {HIST_SAMPLE})")
plt.yscale("log")
stonesoup.show()
plt.close("all")

# %% HTML: all token pairs with cos > 0.5 (paginated in-browser; under outputs/…/embeddings/)
import html
import json
import sys

import stonesoup
import torch
from stonesoup import STONESOUP_RENDER_HTML
from stonesoup.experiment.paths import repo_root

COS_MIN = 0.4
WRITE_BATCH = 4096
HTML_NAME = "common_token_pairs_cos.html"
JSON_NAME = "common_token_pairs_cos_data.json"

with torch.inference_mode():
    tri_u, tri_v = torch.triu_indices(N, N, offset=1, device=cos.device)
    vals_uv = cos[tri_u, tri_v]
    sel = vals_uv > COS_MIN
    n_high = int(sel.sum().item())

print(
    f"upper-tri pairs cos > {COS_MIN}: {n_high} of {vals_uv.numel()} — writing {HTML_NAME} + {JSON_NAME}",
    file=sys.stderr,
    flush=True,
)

_ct = common_token_ids.cpu()
out_dir = stonesoup.plot_dir()
html_path = out_dir / HTML_NAME
json_path = out_dir / JSON_NAME
root = repo_root()

pairs_list: list[dict] = []
if n_high > 0:
    _sub_v = vals_uv[sel]
    _sub_i = tri_u[sel]
    _sub_j = tri_v[sel]
    _order = torch.argsort(_sub_v, descending=True)
    _sub_v = _sub_v[_order]
    _sub_i = _sub_i[_order]
    _sub_j = _sub_j[_order]
    for _start in range(0, n_high, WRITE_BATCH):
        _end = min(_start + WRITE_BATCH, n_high)
        _va = _sub_v[_start:_end].float().cpu()
        _ia = _sub_i[_start:_end].cpu()
        _ja = _sub_j[_start:_end].cpu()
        _tis = _ct[_ia].tolist()
        _tjs = _ct[_ja].tolist()
        _toks_i = tokenizer.convert_ids_to_tokens(_tis)
        _toks_j = tokenizer.convert_ids_to_tokens(_tjs)
        for _k in range(_end - _start):
            _rank = _start + _k + 1
            pairs_list.append(
                {
                    "rank": _rank,
                    "cos": round(float(_va[_k]), 3),
                    "token_i": str(_toks_i[_k]),
                    "token_j": str(_toks_j[_k]),
                    "vocab_i": int(_tis[_k]),
                    "vocab_j": int(_tjs[_k]),
                }
            )
        if _start > 0 and _start % (WRITE_BATCH * 32) == 0:
            stonesoup.check_abort()

_payload = {
    "model_id": MODEL_ID,
    "cos_min": COS_MIN,
    "n_common_rows": N,
    "total_pairs": n_high,
    "pairs": pairs_list,
}
with json_path.open("w", encoding="utf-8") as _jf:
    json.dump(_payload, _jf, ensure_ascii=False, separators=(",", ":"))

_data_url_js = json.dumps(JSON_NAME)
_html_page = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__MODEL_ID__ — cos &gt; __COS_MIN__ (paginated)</title>
<style>
body { font-family: system-ui, sans-serif; margin: 1rem; }
#meta { margin-bottom: 0.75rem; }
.pager { display: flex; flex-wrap: wrap; align-items: center; gap: 0.5rem; margin: 0.75rem 0; }
.pager button { padding: 0.35rem 0.65rem; cursor: pointer; min-height: 2.25rem; }
.pager .pages { display: flex; flex-wrap: wrap; gap: 0.25rem; align-items: center; }
.pager button.page-num { min-width: 2.25rem; }
.pager button.active { font-weight: 700; background: #2563eb; color: #fff; border-color: #2563eb; }
.pager .gap { padding: 0 0.25rem; color: #666; }
.table-scroll { overflow-x: auto; -webkit-overflow-scrolling: touch; margin: 0.5rem 0; }
table { border-collapse: collapse; width: 100%; max-width: 100%; }
th, td { border: 1px solid #ccc; padding: 6px 8px; font-size: 13px; text-align: left; word-break: break-all; }
th { background: #f0f0f0; position: sticky; top: 0; z-index: 1; box-shadow: 0 1px 0 #ccc; }
tr:nth-child(even) { background: #fafafa; }
#err { color: #b91c1c; }
#search-wrap { display: flex; flex-wrap: wrap; align-items: center; gap: 0.5rem; margin: 0.5rem 0 0.25rem; }
#search-wrap label { font-weight: 600; }
#search-input { min-width: 18rem; width: 100%; max-width: 24rem; padding: 0.35rem 0.5rem; font-size: 16px; }
#search-hint { font-size: 12px; color: #555; max-width: 40rem; }
.vocab-sub { font-size: 0.78em; color: #6b7280; font-weight: 400; margin-left: 0.2em; white-space: nowrap; }
.col-rank { width: 2.5rem; font-size: 0.75em; color: #9ca3af; font-weight: 400; text-align: right; font-variant-numeric: tabular-nums; }
th.col-rank { background: #f5f5f5; color: #b0b0b0; }
@media (max-width: 47.99rem) {
  body { margin: 0.75rem; }
  #search-input { min-width: 0; max-width: none; flex: 1 1 12rem; }
  th { position: static; z-index: auto; box-shadow: none; }
}
</style></head><body>
<p id="meta"></p>
<div id="search-wrap">
<label for="search-input">Search</label>
<input type="search" id="search-input" placeholder="substring in token_i or token_j only (matches display _ and raw Ġ/▁)" autocomplete="off" />
<span id="search-hint"></span>
</div>
<p id="err"></p>
<div class="pager" id="pager-top"></div>
<div class="table-scroll" role="region" aria-label="Token pair cosine table">
<table>
<thead><tr><th class="col-rank">#</th><th>cos</th><th>token_i</th><th>token_j</th></tr></thead>
<tbody id="tbody"></tbody>
</table>
</div>
<div class="pager" id="pager-bottom"></div>
<script>
const PAGE_SIZE = 100;
const DATA_URL = __DATA_URL__;
let pairs = [];
let totalPages = 0;
let currentPage = 1;
let searchQuery = "";

function el(tag, text) {
  const e = document.createElement(tag);
  if (text != null) e.textContent = text;
  return e;
}

function escHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function displayTokenPiece(t) {
  if (t == null) return "";
  let s = String(t);
  s = s.split("Ġ").join("_");
  s = s.split("▁").join("_");
  return s;
}

function appendTokenWithVocab(tr, row, which) {
  const td = document.createElement("td");
  const tok = which === "i" ? row.token_i : row.token_j;
  const vid = which === "i" ? row.vocab_i : row.vocab_j;
  td.appendChild(document.createTextNode(displayTokenPiece(tok)));
  const sub = document.createElement("span");
  sub.className = "vocab-sub";
  sub.textContent = " (" + vid + ")";
  td.appendChild(sub);
  tr.appendChild(td);
}

function rowMatchesSearch(row, q) {
  if (!q) return true;
  const ql = q.toLowerCase();
  const ti = displayTokenPiece(row.token_i).toLowerCase();
  const tj = displayTokenPiece(row.token_j).toLowerCase();
  const rawTi = String(row.token_i).toLowerCase();
  const rawTj = String(row.token_j).toLowerCase();
  return (
    ti.includes(ql) ||
    tj.includes(ql) ||
    rawTi.includes(ql) ||
    rawTj.includes(ql)
  );
}

function getFilteredPairs() {
  const q = searchQuery.trim();
  if (!q) return pairs;
  return pairs.filter((row) => rowMatchesSearch(row, q));
}

function updateSearchHint() {
  const elh = document.getElementById("search-hint");
  if (!elh) return;
  const n = getFilteredPairs().length;
  if (!searchQuery.trim()) {
    elh.textContent = "";
    return;
  }
  elh.textContent = n + " match" + (n === 1 ? "" : "es") + " (of " + pairs.length + " pairs).";
}

function renderTable(page) {
  const tb = document.getElementById("tbody");
  tb.replaceChildren();
  if (!pairs.length) return;
  const list = getFilteredPairs();
  totalPages = Math.max(1, Math.ceil(list.length / PAGE_SIZE));
  const p = Math.max(1, Math.min(page, totalPages));
  currentPage = p;
  if (!list.length) {
    const tr = document.createElement("tr");
    const td = document.createElement("td");
    td.colSpan = 4;
    td.style.textAlign = "center";
    td.style.color = "#666";
    td.textContent = searchQuery.trim() ? "No pairs match this search." : "No data.";
    tr.appendChild(td);
    tb.appendChild(tr);
    updateSearchHint();
    return;
  }
  const start = (p - 1) * PAGE_SIZE;
  const slice = list.slice(start, Math.min(start + PAGE_SIZE, list.length));
  for (const row of slice) {
    const tr = document.createElement("tr");
    const tdRank = document.createElement("td");
    tdRank.className = "col-rank";
    tdRank.textContent = row.rank;
    tr.appendChild(tdRank);
    const tdCos = document.createElement("td");
    tdCos.textContent = Number(row.cos).toFixed(3);
    tr.appendChild(tdCos);
    appendTokenWithVocab(tr, row, "i");
    appendTokenWithVocab(tr, row, "j");
    tb.appendChild(tr);
  }
  updateSearchHint();
}

function pageNumbersToShow(cur, total) {
  const out = [];
  if (total <= 12) {
    for (let i = 1; i <= total; i++) out.push(i);
    return out;
  }
  out.push(1);
  const windowStart = Math.max(2, cur - 2);
  const windowEnd = Math.min(total - 1, cur + 2);
  if (windowStart > 2) out.push("…");
  for (let i = windowStart; i <= windowEnd; i++) out.push(i);
  if (windowEnd < total - 1) out.push("…");
  if (total > 1) out.push(total);
  return out;
}

function refreshPagers() {
  buildPager("pager-top");
  buildPager("pager-bottom");
}

function buildPager(containerId) {
  const host = document.getElementById(containerId);
  host.replaceChildren();
  if (!pairs.length) return;

  const list = getFilteredPairs();
  const tp = Math.max(1, Math.ceil(list.length / PAGE_SIZE));
  const cp = Math.min(currentPage, tp);

  const prev = el("button", "Prev");
  prev.disabled = cp <= 1 || list.length === 0;
  prev.onclick = () => { renderTable(cp - 1); refreshPagers(); };

  const next = el("button", "Next");
  next.disabled = cp >= tp || list.length === 0;
  next.onclick = () => { renderTable(cp + 1); refreshPagers(); };

  let labelText = "Page " + cp + " / " + tp;
  if (searchQuery.trim()) {
    labelText += " — " + list.length + " matches (of " + pairs.length + " total)";
  } else {
    labelText += " — " + pairs.length + " pairs";
  }
  const label = el("span", labelText);

  const jump = el("span");
  jump.append("Go to ");
  const inp = document.createElement("input");
  inp.type = "number";
  inp.min = 1;
  inp.max = tp;
  inp.value = String(cp);
  inp.style.width = "5rem";
  inp.addEventListener("change", () => {
    const v = parseInt(inp.value, 10);
    if (!Number.isFinite(v)) return;
    renderTable(v);
    refreshPagers();
  });
  jump.appendChild(inp);

  const pagesWrap = el("div");
  pagesWrap.className = "pages";
  for (const p of pageNumbersToShow(cp, tp)) {
    if (p === "…") {
      const s = el("span", "…");
      s.className = "gap";
      pagesWrap.appendChild(s);
      continue;
    }
    const b = el("button", String(p));
    b.className = "page-num" + (p === cp ? " active" : "");
    b.onclick = () => { renderTable(p); refreshPagers(); };
    pagesWrap.appendChild(b);
  }

  host.append(prev, label, next, jump, pagesWrap);
}

async function load() {
  document.getElementById("err").textContent = "";
  document.getElementById("meta").innerHTML = "";
  try {
    const r = await fetch(DATA_URL);
    if (!r.ok) throw new Error(r.status + " " + r.statusText);
    const data = await r.json();
    pairs = data.pairs || [];
    const total = data.total_pairs ?? pairs.length;
    searchQuery = "";
    const searchInp = document.getElementById("search-input");
    if (searchInp) searchInp.value = "";
    const mid = data.model_id != null ? data.model_id : "";
    document.title = mid + " — cos > " + data.cos_min;
    document.getElementById("meta").innerHTML =
      "<strong>" + escHtml(mid) + "</strong> — <strong>cos &gt; " + data.cos_min + "</strong> — " +
      data.n_common_rows +
      " common-embedding rows — <strong>" + total + "</strong> pairs — " +
      PAGE_SIZE +
      " per page. Search filters rows by <code>token_i</code> / <code>token_j</code> substrings only.";
    if (searchInp) {
      searchInp.addEventListener("input", () => {
        searchQuery = searchInp.value;
        currentPage = 1;
        renderTable(1);
        refreshPagers();
      });
    }
    renderTable(1);
    refreshPagers();
  } catch (e) {
    document.getElementById("err").textContent =
      "Could not load " + DATA_URL + " (open via HTTP same folder as this HTML, e.g. /outputs/…). " + e;
  }
}
load();
</script>
</body></html>
"""
_html_page = (
    _html_page.replace("__MODEL_ID__", html.escape(MODEL_ID))
    .replace("__COS_MIN__", str(COS_MIN))
    .replace("__DATA_URL__", _data_url_js)
)

with html_path.open("w", encoding="utf-8") as _hf:
    _hf.write(_html_page)

_rel_html = html_path.resolve().relative_to(root).as_posix()
_rel_json = json_path.resolve().relative_to(root).as_posix()
_href = "/" + _rel_html
print(STONESOUP_RENDER_HTML, end="")
print(
    f'<p>Token pairs cos &gt; {COS_MIN}: <strong>{n_high}</strong> — '
    f'<a href="{html.escape(_href)}">{html.escape(HTML_NAME)}</a> '
    f"(paginated view; data <code>{html.escape(_rel_json)}</code>).</p>",
    flush=True,
)
