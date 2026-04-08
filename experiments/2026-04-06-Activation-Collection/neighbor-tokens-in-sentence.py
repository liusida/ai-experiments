# %% Imports, model, long sentence
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import stonesoup

MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
LONG_SENTENCE = (
    "The old observatory sat on a breezy ridge where undergraduates once learned to read "
    "sketches of nebulae by lantern light; now its dome stuck slightly off-center, "
    "and pigeons negotiated custody of the railing with the somber professionalism of "
    "small-time attorneys, while the town below debated whether to fund repairs or "
    "convert the building into a seasonal cafe that served honest espresso and dubious "
    "mythology about the astronomers who had slept on cots beside the brass transit "
    "when thunderstorms rolled inland from the sea, a controversy that dragged on through "
    "three rainy budget seasons until the mayor proposed a compromise involving string "
    "lights, interpretive plaques, and a single refurbished telescope that nobody trusted "
    "to hold collimation; volunteers cataloged peeling negatives in acid-free sleeves, "
    "discovered coffee rings on logbook margins that looked like accidental galaxies, and "
    "argued gently about whether the building sighed at night from wood cooling or from "
    "old shame; a teenager with a cheap monopod filmed bats trading the slit of twilight "
    "above the slit of the shutter while cicadas argued with distant traffic; the librarian "
    "from the college archive arrived with a wheeled cart of half-digitized ledgers, "
    "insisting that footnotes mattered more than optics, and that the true sky was "
    "archival dust rising in the projector beam; meanwhile contractors measured settling "
    "and shrugged as if settling were a personality trait concrete shared with humans; "
    "someone left a thermos of soup on the cast-iron spiral stair, a kindness that felt "
    "almost political; swifts stitched the air between chimney and weather vane; fog idled "
    "in the valley until afternoon, when it burned off to reveal a water tower painted "
    "like a target, as if the town expected cosmic archery; by dusk the ridge wind picked "
    "up, rattling loose sheets on a clipboard, and the observatory again pretended it had "
    "not been waiting all century for anyone patient enough to listen."
)

model, tokenizer = stonesoup.load_model(MODEL_ID)
model.eval()
device = next(model.parameters()).device
inner_tok = getattr(tokenizer, "tokenizer", tokenizer)
MODEL_BASENAME = MODEL_ID.replace("/", "__").replace(":", "-")
print("Loaded:", MODEL_ID, device, flush=True)

# %% Forward: hidden states, cross-layer cosine matrices
@torch.inference_mode()
def cross_layer_cos_matrices(
    text: str,
) -> tuple[torch.Tensor, torch.Tensor, int, int, list[int]]:
    """Return (M_adj, M_same, S, T, ids).

    M_adj[l,l'] = mean_i cos(H[l,i], H[l',i+1]) over adjacent pairs.
    M_same[l,l'] = mean_i cos(H[l,i], H[l',i]) over the same token i.
    """
    enc = inner_tok(text, return_tensors="pt", return_attention_mask=True, add_special_tokens=True)
    ids = enc["input_ids"].to(device)
    out = model(
        input_ids=ids,
        attention_mask=enc["attention_mask"].to(device),
        output_hidden_states=True,
        use_cache=False,
    )
    hs = out.hidden_states
    H = torch.stack([h[0].float() for h in hs], dim=0)
    H = F.normalize(H, dim=-1)
    S, T, _ = H.shape
    if T < 2:
        raise ValueError("need at least two tokens")
    m_adj = torch.einsum("s t d, u t d -> s u", H[:, :-1, :], H[:, 1:, :]) / float(T - 1)
    m_same = torch.einsum("s t d, u t d -> s u", H, H) / float(T)
    return m_adj, m_same, S, T, ids[0].tolist()


M_adj, M_same, n_stages, tok_len, id_list = cross_layer_cos_matrices(LONG_SENTENCE)
M_adj_np = M_adj.cpu().numpy()
M_same_np = M_same.cpu().numpy()
diag = np.diag(M_adj_np)
print(
    f"stages={n_stages} (incl. embeddings as 0)  seq_len={tok_len}  pairs={tok_len - 1}",
    flush=True,
)
print("decode (abridged):", inner_tok.decode(id_list, skip_special_tokens=False)[:200] + "…", flush=True)

# %% Plot: adjacent-pair heatmap | same-token heatmap | neighbor diagonal line
_tick_step = max(1, n_stages // 12)
ticks = np.arange(0, n_stages, _tick_step)
fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(17.5, 5.2))

im0 = ax0.imshow(M_adj_np, vmin=0.0, vmax=1.0, cmap="Blues", aspect="equal", origin="lower")
ax0.set_xlabel("layer l' · hidden at token i+1")
ax0.set_ylabel("layer l · hidden at token i")
ax0.set_xticks(ticks)
ax0.set_yticks(ticks)
ax0.set_title(
    "Adjacent tokens · mean over i of cos(h[i,l], h[i+1,l'])\n"
    f"{tok_len - 1} pairs · {MODEL_ID}"
)
fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04, label="cosine")

im1 = ax1.imshow(M_same_np, vmin=0.0, vmax=1.0, cmap="Blues", aspect="equal", origin="lower")
ax1.set_xlabel("layer l' (same token i)")
ax1.set_ylabel("layer l (same token i)")
ax1.set_xticks(ticks)
ax1.set_yticks(ticks)
ax1.set_title(
    "Same token i · mean over i of cos(h[i,l], h[i,l'])\n"
    f"{tok_len} positions"
)
fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label="cosine")

layers = np.arange(n_stages)
ax2.plot(layers, diag, color="seagreen", linewidth=1.4)
ax2.fill_between(layers, diag, alpha=0.2, color="seagreen")
ax2.set_xlabel("layer (same l = l', stage 0 = embeddings)")
ax2.set_ylabel("cosine")
ax2.set_title("Diagonal of left panel:\nsame-layer neighbor cos")
ax2.set_ylim(0, 1.05)
ax2.grid(True, alpha=0.25)

fig.tight_layout()
stonesoup.show(fig, basename=f"{MODEL_BASENAME}_neighbor_and_same_tok_cross_layer_cos")
