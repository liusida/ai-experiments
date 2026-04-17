# %% Imports
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import stonesoup
from stonesoup.experiment import ensure_pad_token_via_eos, inner_tokenizer

# %% Config
MODEL_ID = "Qwen/Qwen3.5-0.8B"
PROMPT = "Here is a story about a cat:"
GEN_TOKENS = 20

# %% Load model
model, proc = stonesoup.load_model(MODEL_ID, torch_dtype="bfloat16")
model.eval()
model.requires_grad_(False)

tokenizer = inner_tokenizer(proc)
ensure_pad_token_via_eos(tokenizer)

device = next(model.parameters()).device

_inner = model.model
text_module = getattr(_inner, "language_model", _inner)
final_norm = text_module.norm
lm_head = model.lm_head

# %% Load trained lens

class AffineTranslator(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.weight = nn.Parameter(torch.eye(d))
        self.bias = nn.Parameter(torch.zeros(d))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return (h.float() @ self.weight.T + self.bias).to(h.dtype)


class TunedLens(nn.Module):
    def __init__(self, n_probed: int, d: int):
        super().__init__()
        self.translators = nn.ModuleList([AffineTranslator(d) for _ in range(n_probed)])


ckpt_path = stonesoup.script_dir() / "tuned_lens.pt"
ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
lens = TunedLens(ckpt["n_layers"], ckpt["d_model"]).to(device)
lens.load_state_dict(ckpt["state_dict"])
lens.eval()
print(f"Loaded tuned lens from {ckpt_path} ({ckpt['n_layers']} layers, d={ckpt['d_model']})", flush=True)


def decode(h: torch.Tensor) -> torch.Tensor:
    return lm_head(final_norm(h))

# %% Apply tuned lens to prompt # stonesoup:cell-input
import html as html_mod

prompt = globals().get("CELL_INPUT", "") or PROMPT
prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
prompt_len = prompt_ids.shape[1]

# Generate continuation so we have ground-truth next tokens for every position
with torch.no_grad():
    gen_ids = model.generate(
        prompt_ids, max_new_tokens=GEN_TOKENS, do_sample=False, use_cache=True,
    )
all_ids = gen_ids[:, :prompt_len + GEN_TOKENS]
seq_len = all_ids.shape[1]

# Forward the full sequence to get hidden states
with torch.no_grad():
    out = text_module(input_ids=all_ids, output_hidden_states=True, use_cache=False)
    hidden_states = out.hidden_states
    final_logits = lm_head(hidden_states[-1])

# Header = input token at each position; cells = lens prediction (next token) at that position
all_tokens = [tokenizer.decode(all_ids[0, t]) for t in range(seq_len)]

# grid[layer][pos] = (top1_token, top1_prob)
grid: list[list[tuple[str, float]]] = []
with torch.no_grad():
    for l, h in enumerate(hidden_states[:-1]):
        lens_logits = decode(lens.translators[l](h))
        probs = F.softmax(lens_logits[0].float(), dim=-1)
        top1_probs, top1_ids = probs.max(dim=-1)
        grid.append([(tokenizer.decode(top1_ids[p]), top1_probs[p].item()) for p in range(seq_len)])

    final_probs = F.softmax(final_logits[0].float(), dim=-1)
    f_top1_probs, f_top1_ids = final_probs.max(dim=-1)
    final_row = [(tokenizer.decode(f_top1_ids[p]), f_top1_probs[p].item()) for p in range(seq_len)]

def _esc(s: str) -> str:
    return html_mod.escape(s)

CELL_H = 18

def _display_tok(tok: str) -> str:
    return _esc(tok) if tok.strip() else "-"

def _cell(tok: str, prob: float, bold: bool = False, border_left: bool = False) -> str:
    pct = prob * 100
    weight = "font-weight:bold;" if bold else ""
    bl = "border-left:2px solid #888;" if border_left else ""
    return (
        f'<td style="padding:0 4px;text-align:right;white-space:nowrap;{weight}{bl}'
        f'background:linear-gradient(0deg,rgba(34,139,34,0.25) {pct:.0f}%,transparent {pct:.0f}%)">'
        f'{_display_tok(tok)}</td>'
    )

generated_text = tokenizer.decode(all_ids[0, prompt_len:])
gen_start = prompt_len

lines: list[str] = []
lines.append(stonesoup.STONESOUP_RENDER_HTML)
lines.append(f"<p><b>Model:</b> <code>{_esc(MODEL_ID)} with self-trained Tuned Lens {ckpt['n_layers']} layers, d={ckpt['d_model']}</code></p>")
lines.append(f"<p><b>Prompt:</b> <code>{_esc(prompt)}</code></p>")
lines.append(f"<p><b>Generated:</b> <code>{_esc(generated_text)}</code></p>")
lines.append('<div style="overflow-x:auto">')
lines.append(
    f'<table style="border-collapse:collapse;font-family:monospace;font-size:11px;'
    f'line-height:{CELL_H}px">'
)

# Header row
lines.append("<tr>")
lines.append('<th style="padding:0 4px;border-bottom:2px solid #888;text-align:right">Layer</th>')
for pos in range(seq_len):
    color = "color:#07a;" if pos >= gen_start else ""
    bl = "border-left:2px solid #888;" if pos == gen_start else ""
    lines.append(
        f'<th style="padding:0 4px;border-bottom:2px solid #888;text-align:left;'
        f'white-space:nowrap;{color}{bl}">{_display_tok(all_tokens[pos])}</th>'
    )
lines.append("</tr>")

# One row per layer
for l, layer_row in enumerate(grid):
    lines.append("<tr>")
    lines.append(f'<td style="padding:0 4px;text-align:right;border-right:1px solid #ccc">{l}</td>')
    for pos, (tok, prob) in enumerate(layer_row):
        lines.append(_cell(tok, prob, border_left=(pos == gen_start)))
    lines.append("</tr>")

# Final model row
lines.append('<tr style="border-top:2px solid #888">')
lines.append('<td style="padding:0 4px;text-align:right;border-right:1px solid #ccc;font-weight:bold">Final</td>')
for pos, (tok, prob) in enumerate(final_row):
    lines.append(_cell(tok, prob, bold=True, border_left=(pos == gen_start)))
lines.append("</tr>")

lines.append("</table></div>")
print("\n".join(lines), flush=True)
