# %% Imports & matplotlib
from __future__ import annotations

import numpy as np
import stonesoup
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt

from stonesoup.experiment import (
    capture_embed_and_post_blocks,
    configure_matplotlib_agg,
    encode_text_inputs,
    ensure_pad_token_via_eos,
    hf_repo_id_safe_stem,
    inner_tokenizer,
    show,
)

configure_matplotlib_agg()

# %% Config — control + parallel treatments (same order, same index = same “item”)
MODEL_ID = "Qwen/Qwen3.5-9B"

# Items 0–9 match across lists: calm = EN (control) / ZH / FR with parallel wording. (Poetic & screaming EN lists commented out below.)
CONTROL_SENTENCES: list[str] = [
    "The cat slept on the warm windowsill all afternoon.",
    "Over thousands of years, rivers carry sediment from the mountains to the sea.",
    "She opened the book and found a pressed flower between the pages.",
    "At dawn, electric trains run quietly through the suburbs.",
    "Honey stays good almost forever because its low water content stops bacteria.",
    "He learned to cook by watching his grandmother every Sunday.",
    "Snow reflects sunlight and can make winter days feel brighter.",
    "The orchestra tuned in a low voice before the conductor raised the baton.",
    "Migrating birds find their way using Earth's magnetic field.",
    "Fresh bread smells sweet because heat turns starch into sugars in the oven.",
]

TREATMENT_ZH_TRANSLATIONS: list[str] = [
    "那只猫整个下午都睡在温暖的窗台上。",
    "几千年来，河流把山里的泥沙一路带进大海。",
    "她打开书，在书页间发现一朵压干的花。",
    "黎明时分，电力列车安静地穿过郊区。",
    "蜂蜜几乎不会坏，因为水分少，能抑制细菌。",
    "他每个星期天看着祖母做饭，学会了烹饪。",
    "雪反射阳光，让冬日有时显得更亮。",
    "管弦乐队轻声调音，然后指挥举起指挥棒。",
    "候鸟依靠地磁场辨认方向。",
    "新鲜面包闻起来甜，是因为高温让淀粉在烤箱里变成糖。",
]

TREATMENT_FR_TRANSLATIONS: list[str] = [
    "Le chat a dormi tout l'après-midi sur le rebord de fenêtre tiède.",
    "Depuis des millénaires, les rivières charrient les sédiments des montagnes jusqu'à la mer.",
    "Elle ouvrit le livre et trouva une fleur pressée entre les pages.",
    "À l'aube, les trains électriques traversent la banlieue en silence.",
    "Le miel se conserve presque toujours : peu d'eau, les bactéries ne passent pas.",
    "Il apprit à cuisiner en observant sa grand-mère chaque dimanche.",
    "La neige renvoie le soleil ; les jours d'hiver paraissent plus clairs.",
    "L'orchestre s'accorda tout bas avant que le chef ne lève la baguette.",
    "Les oiseaux migrateurs s'orientent grâce au champ magnétique terrestre.",
    "Le pain chaud sent le sucre : la chaleur transforme l'amidon en sucres au four.",
]

# --- Screaming treatments (commented out) ---
# TREATMENT_EN_SCREAMING: list[str] = [
#     "THE CAT—THE HOT SILL—ALL AFTERNOON—DON'T TOUCH IT!!!",
#     "RIVERS—MOUNTAINS TO THE SEA—THOUSANDS OF YEARS—WHO'S COUNTING?!!",
#     "SHE OPENS IT—BANG—PRESSED FLOWER—RIGHT IN THE PAGES!!!",
#     "DAWN—ELECTRIC TRAINS—THROUGH THE SUBURBS—NOT A SOUND!!!",
#     "HONEY—NO ROT—NO WATER—BACTERIA OUT!!!",
#     "EVERY SUNDAY—GRANDMA—HE WATCHED—TILL IT WAS IN HIS BLOOD!!!",
#     "SNOW—BOUNCES THE SUN—WINTER—TOO BRIGHT!!!",
#     "ORCHESTRA—WHISPERS TUNING—THEN—BATON UP!!!",
#     "BIRDS—LONG FLIGHT—MAGNET—GPS IN THE HEAD!!!",
#     "HOT BREAD—STARCH TO SUGAR—IN THE OVEN—SMELL IT NOW!!!",
# ]
#
# TREATMENT_ZH_SCREAMING: list[str] = [
#     "猫——烫窗台——一下午——别碰它！！",
#     "河——山到海——几千年——谁数过？！",
#     "她打开——砰——压干的花——就夹在页间！！",
#     "天亮——电车——穿过郊区——一声不响！！",
#     "蜂蜜——不坏——没水——细菌滚！！",
#     "每周日——奶奶——他死盯——刻进骨子里！！",
#     "雪——弹回太阳——冬天——亮瞎！！",
#     "乐团——细声调音——然后——棒子举起！！",
#     "候鸟——远飞——磁场——头里自带导航！！",
#     "热面包——淀粉变糖——烤箱里——现在就闻！！",
# ]
#
# TREATMENT_FR_SCREAMING: list[str] = [
#     "LE CHAT—REBORD BRÛLANT—TOUT L'APRÈS-MIDI—ON LE RÉVEILLE PAS ?!!",
#     "LES RIVIÈRES—MONTAGNES À LA MER—DES MILLIERS D'ANNÉES—QUI COMPTE ?!!",
#     "ELLE OUVRE—BOUM—FLEUR PRESSÉE—DANS LES PAGES ?!!",
#     "L'AUBE—TRAINS ÉLECTRIQUES—LA BANLIEUE—PAS UN BRUIT ?!!",
#     "LE MIEL—JAMAIS GÂTÉ—TROP SEC—BACTÉRIES DÉGAGEZ ?!!",
#     "CHAQUE DIMANCHE—MAMIE—IL A TOUT REGARDÉ—GRAVÉ DEDANS ?!!",
#     "NEIGE—RENVOIE LE SOLEIL—L'HIVER—TROP FORT ?!!",
#     "ORCHESTRE—CHUCHOTE—PUIS—BAGUETTE EN L'AIR ?!!",
#     "OISEAUX—LONGUE ROUTE—AIMANT—GPS DANS LE CRÂNE ?!!",
#     "PAIN CHAUD—AMIDON EN SUCRE—AU FOUR—SENS ÇA ?!!",
# ]

# --- EN poetic treatment (commented out) ---
# TREATMENT_EN_POETIC: list[str] = [
#     "All afternoon the cat uncurled against the sun-warmed sill, asleep in borrowed light.",
#     "Age upon age, the rivers grind the mountains fine and trail their dust to the waiting sea.",
#     "The book parted; between the leaves lay summer pressed to parchment, a flower out of time.",
#     "Dawn thins above the suburbs where electric serpents glide on rails of quiet thunder.",
#     "In honey's amber drought no microbe wakes—the hive's sweet seal outlasts the years.",
#     "Each Sunday he stood in her kitchen's steam and learned the grammar of fire from her hands.",
#     "The world wears snow like foil; daylight doubles, and winter burns with borrowed sun.",
#     "Bows trembled, reeds sighed—then hush—until the baton carved silence into song.",
#     "They ride the magnetic meridians, compass-born, stitching poles with wing and longing.",
#     "The oven gives a sigh of sweetness—starch dissolves in gold, and the warm loaf remembers sugar.",
# ]

# Same propositional content, phrased in a neutral scientific register (precise, causal, low affect).
TREATMENT_EN_SCIENTIFIC: list[str] = [
    "The cat maintained a sleep state on a sun-warmed windowsill throughout the afternoon interval.",
    "Over geologic time, rivers transport eroded sediment from upland source areas to marine sinks.",
    "Opening the book revealed a pressed flower sandwiched between two pages.",
    "At dawn, electrically powered trains moved through suburban corridors with low acoustic output.",
    "Honey exhibits long shelf stability because low water activity limits bacterial growth.",
    "He acquired cooking skills through repeated observational learning during weekly Sunday visits with his grandmother.",
    "Snow has high albedo; reflected sunlight can increase perceived brightness on winter days.",
    "The orchestra completed quiet tuning before the conductor raised the baton to begin.",
    "Migratory birds use the geomagnetic field as a directional cue during long-distance navigation.",
    "Fresh bread emits sweet aromas partly because heat drives starch toward simpler sugars during baking.",
]

# Same facts as control; upbeat, warm wording (happy affect).
TREATMENT_EN_HAPPY: list[str] = [
    "The cat curled up happily on the sun-warmed windowsill and slept the whole afternoon away.",
    "For thousands of years, rivers have cheerfully carried mountain sediment down to the sea—a slow, steady gift.",
    "She opened the book and brightened: a pressed flower lay between the pages like a tiny treasure.",
    "At dawn the electric trains glide softly through the suburbs, starting the day on a gentle hum.",
    "Honey keeps almost forever; its dryness locks out bacteria and the sweetness just stays.",
    "He learned to cook with real delight, watching his grandmother every Sunday in her warm kitchen.",
    "Snow sparkles the sunlight back, and winter days can feel surprisingly bright and open.",
    "The orchestra warmed up in a cozy hush, then the conductor raised the baton and the room leaned in.",
    "Migrating birds ride Earth's magnetic field like a secret map, finding their way home across the sky.",
    "Fresh bread smells wonderful when the oven's heat turns starch into sugar—that warm, simple magic.",
]

# Same facts as control; subdued, wistful wording (sad affect).
TREATMENT_EN_SAD: list[str] = [
    "The cat slept on the warm windowsill all afternoon, small and still in the long light.",
    "For thousands of years, rivers have dragged sediment from the mountains to the sea, tireless and indifferent.",
    "She opened the book and found a pressed flower between the pages, dry and far from any garden.",
    "At dawn the electric trains move through the suburbs in a low hum, another quiet morning passing.",
    "Honey lasts because almost no water remains; the sweetness holds while everything else is kept out.",
    "He learned to cook by watching his grandmother each Sunday, learning gestures that time would take away.",
    "Snow reflects sunlight until winter days look bright, but the cold still settles deep.",
    "The orchestra tuned in whispers before the conductor raised the baton, silence stretching thin.",
    "Migrating birds follow Earth's magnetic field mile on mile, with only instinct to name the way.",
    "Fresh bread smells sweet only because heat breaks starch into sugar, and the warmth does not stay.",
]

# Pairwise cos plots: (legend, left sentences, right sentences, color, linestyle).
TREATMENT_SPECS: list[tuple[str, list[str], list[str], str, str]] = [
    # ("cos(control, EN poetic)", CONTROL_SENTENCES, TREATMENT_EN_POETIC, "tab:purple", "-"),
    # ("cos(control, EN screaming)", CONTROL_SENTENCES, TREATMENT_EN_SCREAMING, "tab:red", "--"),
    ("cos(control, EN scientific)", CONTROL_SENTENCES, TREATMENT_EN_SCIENTIFIC, "tab:green", "-"),
    ("cos(control, EN happy)", CONTROL_SENTENCES, TREATMENT_EN_HAPPY, "#C9A227", "-"),
    ("cos(control, EN sad)", CONTROL_SENTENCES, TREATMENT_EN_SAD, "#5B7C99", "-"),
    ("cos(EN happy, EN sad)", TREATMENT_EN_HAPPY, TREATMENT_EN_SAD, "tab:brown", "-"),
    # ("cos(EN poetic, EN scientific)", TREATMENT_EN_POETIC, TREATMENT_EN_SCIENTIFIC, "tab:olive", "-"),
    # ("cos(EN poetic, EN screaming)", TREATMENT_EN_POETIC, TREATMENT_EN_SCREAMING, "#9467bd", "--"),
    # ("cos(EN poetic, EN happy)", TREATMENT_EN_POETIC, TREATMENT_EN_HAPPY, "#bcbd22", "-"),
    # ("cos(EN poetic, EN sad)", TREATMENT_EN_POETIC, TREATMENT_EN_SAD, "#aec7e8", "-"),
    # ("cos(EN screaming, EN happy)", TREATMENT_EN_SCREAMING, TREATMENT_EN_HAPPY, "tab:orange", "--"),
    # ("cos(EN screaming, EN sad)", TREATMENT_EN_SCREAMING, TREATMENT_EN_SAD, "#8c564b", "--"),
    # ("cos(EN screaming, EN scientific)", TREATMENT_EN_SCREAMING, TREATMENT_EN_SCIENTIFIC, "#e377c2", "--"),
    ("cos(EN happy, EN scientific)", TREATMENT_EN_HAPPY, TREATMENT_EN_SCIENTIFIC, "#17becf", "-"),
    ("cos(EN sad, EN scientific)", TREATMENT_EN_SAD, TREATMENT_EN_SCIENTIFIC, "#98df8a", "-"),
]

_n = len(CONTROL_SENTENCES)
for _label, _a, _b, _c, _ls in TREATMENT_SPECS:
    assert len(_a) == len(_b) == _n, (_label, len(_a), len(_b))

# %% Load model
torch.set_grad_enabled(False)

model, processor = stonesoup.load_model(MODEL_ID)
model.eval()
tokenizer = inner_tokenizer(processor)
ensure_pad_token_via_eos(tokenizer)
device = next(model.parameters()).device

# %% Capture sequence-mean hidden states per stage (one sentence per forward)
model, processor = stonesoup.load_model(MODEL_ID)
model.eval()
ensure_pad_token_via_eos(inner_tokenizer(processor))
device = next(model.parameters()).device


def masked_mean_hidden_per_stage(
    stack: torch.Tensor, attention_mask_row: torch.Tensor
) -> torch.Tensor:
    """Mean over sequence positions where mask is 1. stack (S, batch, seq, H) → (S, H)."""
    m = attention_mask_row.to(stack.device).float().view(1, -1, 1)
    h = stack[:, 0, :, :].float()
    return ((h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-8)).detach()


def register_tokens_for_global_mu(
    stack: torch.Tensor,
    attention_mask_row: torch.Tensor,
    mu_stats: dict[str, object],
) -> None:
    """Accumulate sum of hidden states over all masked positions, for μ = sum / total_tokens per stage."""
    m = attention_mask_row.to(stack.device).float().view(1, -1, 1)
    h = stack[:, 0, :, :].float()
    summed = (h * m).sum(dim=1).detach()
    n_new = int(m.sum().item())
    cur = mu_stats["token_sum"]
    if cur is None:
        mu_stats["token_sum"] = summed.clone()
    else:
        mu_stats["token_sum"] = cur + summed
    mu_stats["token_count"] = int(mu_stats["token_count"]) + n_new


def raw_cosine_per_stage(h_a: torch.Tensor, h_b: torch.Tensor) -> torch.Tensor:
    """Raw cosine (not centered): L2-normalize then dot. (num_stages, H) → (num_stages,)."""
    x = F.normalize(h_a.float(), dim=-1, eps=1e-8)
    y = F.normalize(h_b.float(), dim=-1, eps=1e-8)
    return (x * y).sum(dim=-1).cpu()


def run_texts(
    texts: list[str],
    tag: str,
    mu_stats: dict[str, object],
) -> tuple[list[torch.Tensor], list[str]]:
    out: list[torch.Tensor] = []
    names: list[str] = []
    for i, text in enumerate(texts):
        stonesoup.check_abort()
        inputs = encode_text_inputs(processor, text, device=device)
        stack, stage_names = capture_embed_and_post_blocks(model, inputs, use_cache=False)
        if i == 0:
            names = stage_names
        mask = inputs["attention_mask"][0]
        register_tokens_for_global_mu(stack, mask, mu_stats)
        out.append(masked_mean_hidden_per_stage(stack, mask))
        print(f"{tag} tok={int(mask.sum().item())} stages={stack.shape[0]}", flush=True)
    return out, names


# One μ per stage: mean over every real token across all runs (center each token once, then
# sentence means satisfy mean_t(h_t - μ) = mean_t(h_t) - μ).
_mu_token_stats: dict[str, object] = {"token_sum": None, "token_count": 0}
_hiddens_cache: dict[int, list[torch.Tensor]] = {}

hidden_control, stage_names = run_texts(CONTROL_SENTENCES, "control", _mu_token_stats)
_hiddens_cache[id(CONTROL_SENTENCES)] = hidden_control


def _hiddens_for_texts(texts: list[str], tag: str) -> list[torch.Tensor]:
    """Cache forwards by list identity so shared lists (e.g. CONTROL) are not re-run."""
    k = id(texts)
    if k not in _hiddens_cache:
        h, _ = run_texts(texts, tag, _mu_token_stats)
        _hiddens_cache[k] = h
    return _hiddens_cache[k]


for _lbl, texts_a, texts_b, _c, _ls in TREATMENT_SPECS:
    stonesoup.check_abort()
    _hiddens_for_texts(texts_a, f"{_lbl}|A")
    _hiddens_for_texts(texts_b, f"{_lbl}|B")

n_stages = hidden_control[0].shape[0]
print("stage_names:", stage_names[:3], "…", stage_names[-1], flush=True)

# %% Pairwise raw cos + per-set std (||h|| spread)
layer_x = np.arange(n_stages)
_LAYER_X_LAST = int(layer_x[-1])


def _vline_last_layer_xtick(ax: plt.Axes, x_last: int) -> None:
    """Vertical line at the last stage index; ensure that x value appears as an x-tick (no text)."""
    ax.axvline(x_last, color="0.35", ls="--", lw=0.4, alpha=0.3, zorder=4)
    lo, hi = ax.get_xlim()
    base = [float(t) for t in ax.get_xticks() if lo - 1e-9 <= float(t) <= hi + 1e-9]
    ax.set_xticks(sorted({*base, float(x_last)}))


curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
for plot_label, texts_a, texts_b, _color, _ls in TREATMENT_SPECS:
    stonesoup.check_abort()
    h_a = _hiddens_for_texts(texts_a, f"{plot_label}|A")
    h_b = _hiddens_for_texts(texts_b, f"{plot_label}|B")
    cos_per_pair = torch.stack(
        [raw_cosine_per_stage(h_a[i], h_b[i]) for i in range(_n)], dim=0
    ).numpy()
    curves[plot_label] = (
        cos_per_pair.mean(axis=0),
        cos_per_pair.std(axis=0, ddof=1),
    )

# Std of ||h|| across sentences within each unique text list (by object id).
_norm_seen: dict[int, tuple[str, list[torch.Tensor]]] = {
    id(CONTROL_SENTENCES): ("control", hidden_control),
}
for plot_label, texts_a, texts_b, _c, _ls in TREATMENT_SPECS:
    for side_name, tx in (f"{plot_label} [A]", texts_a), (f"{plot_label} [B]", texts_b):
        kid = id(tx)
        if kid not in _norm_seen:
            _norm_seen[kid] = (side_name, _hiddens_for_texts(tx, side_name))
all_sets = list(_norm_seen.values())

std_norm_by_set: dict[str, np.ndarray] = {}
for set_name, h_list in all_sets:
    std_norm_by_set[set_name] = np.zeros(n_stages)
    for ell in range(n_stages):
        norms = np.array([h_list[i][ell].float().norm().item() for i in range(_n)])
        std_norm_by_set[set_name][ell] = float(np.std(norms, ddof=1))

for set_name, arr in std_norm_by_set.items():
    print(f"std ||h|| {set_name} (first/last layer):", arr[0], arr[-1], flush=True)

# %% Token-global μ per layer, then centered cos (subtract μ from sentence means; equals mean of centered tokens)
assert _mu_token_stats["token_sum"] is not None
_n_tokens_total = int(_mu_token_stats["token_count"])
mu_global_per_layer = (
    _mu_token_stats["token_sum"].float() / float(_n_tokens_total)
)  # (n_stages, H)
print(
    f"μ per layer: mean over {_n_tokens_total} real tokens "
    f"({len(_norm_seen)} unique sentence lists × {_n} items each; shared lists deduped).",
    flush=True,
)


def centered_cosine_global_mean(
    h_a: torch.Tensor, h_b: torch.Tensor, mu_layer: torch.Tensor
) -> torch.Tensor:
    """Subtract token-global μ per layer (sentence means are mean_t(h_t), so this matches mean_t(h_t - μ))."""
    x = h_a.float() - mu_layer
    y = h_b.float() - mu_layer
    x = F.normalize(x, dim=-1, eps=1e-8)
    y = F.normalize(y, dim=-1, eps=1e-8)
    return (x * y).sum(dim=-1).cpu()


curves_centered: dict[str, tuple[np.ndarray, np.ndarray]] = {}
for plot_label, texts_a, texts_b, _color, _ls in TREATMENT_SPECS:
    stonesoup.check_abort()
    h_a = _hiddens_for_texts(texts_a, f"{plot_label}|A")
    h_b = _hiddens_for_texts(texts_b, f"{plot_label}|B")
    cos_per_pair = torch.stack(
        [
            centered_cosine_global_mean(h_a[i], h_b[i], mu_global_per_layer)
            for i in range(_n)
        ],
        dim=0,
    ).numpy()
    curves_centered[plot_label] = (
        cos_per_pair.mean(axis=0),
        cos_per_pair.std(axis=0, ddof=1),
    )

# %% Plot raw cos(control, treatment) vs layer
_safe = hf_repo_id_safe_stem(MODEL_ID)
fig, ax = plt.subplots(figsize=(8, 4.5))
for plot_label, _ta, _tb, color, ls in TREATMENT_SPECS:
    mean_cos, std_cos = curves[plot_label]
    ax.plot(layer_x, mean_cos, color=color, ls=ls, lw=2.0, label=plot_label)
    ax.fill_between(
        layer_x,
        mean_cos - std_cos,
        mean_cos + std_cos,
        color=color,
        alpha=0.22,
    )
ax.set_xlabel("Layer (stage index; 0 = embedding output)")
ax.set_ylabel("cosine similarity")
ax.set_ylim(-1.05, 1.05)
ax.grid(True, alpha=0.3)
_vline_last_layer_xtick(ax, _LAYER_X_LAST)
handles, labels = ax.get_legend_handles_labels()
ax.legend(
    handles,
    [f"{lb} (band = ±1 std across pairs)" for lb in labels],
    loc="best",
    fontsize=8,
)
ax.set_title(f"Raw cos of mean hiddens · pairwise (TREATMENT_SPECS) · {MODEL_ID}")
fig.tight_layout()
show(fig, basename=f"{_safe}_control_vs_treatments_cos_mean_std", dpi=140)

# %% Plot centered cos (subtract global layer mean, then cosine)
fig2, ax2 = plt.subplots(figsize=(10, 10))
_legend_by_last_layer: list[tuple[float, plt.Line2D, str]] = []
_last_ly = _LAYER_X_LAST
for plot_label, _ta, _tb, color, ls in TREATMENT_SPECS:
    mean_c, _std_c = curves_centered[plot_label]
    (line2d,) = ax2.plot(
        layer_x, mean_c, color=color, ls=ls, lw=2.0, label=plot_label
    )
    _legend_by_last_layer.append((float(mean_c[_last_ly]), line2d, plot_label))
_legend_by_last_layer.sort(key=lambda t: t[0], reverse=True)
handles2 = [t[1] for t in _legend_by_last_layer]
labels2 = [t[2] for t in _legend_by_last_layer]
ax2.set_xlabel("Layer (stage index; 0 = embedding output)")
ax2.set_ylabel("centered cosine similarity")
ax2.set_ylim(-1.05, 1.05)
ax2.grid(True, alpha=0.3)
_vline_last_layer_xtick(ax2, _LAYER_X_LAST)
ax2.legend(
    handles2,
    labels2,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.18),
    ncol=3,
    fontsize=7,
    frameon=True,
)
ax2.set_title(
    f"Centered cos (μ = mean over all {_n_tokens_total} tokens/layer) · {MODEL_ID}"
)
fig2.tight_layout(rect=(0, 0.22, 1, 1))
show(fig2, basename=f"{_safe}_control_vs_treatments_cos_globalmean_centered", dpi=140)

# %% Centered cos — selected pairs (same token-global μ_global_per_layer as above)
# (legend, left texts, right texts, color, linestyle).
SELECTIVE_CENTERED_PAIRS: list[tuple[str, list[str], list[str], str, str]] = [
    ("control vs EN happy", CONTROL_SENTENCES, TREATMENT_EN_HAPPY, "#C9A227", "-"),
    ("control vs EN sad", CONTROL_SENTENCES, TREATMENT_EN_SAD, "#5B7C99", "-"),
    # ("control vs EN screaming", CONTROL_SENTENCES, TREATMENT_EN_SCREAMING, "tab:red", "--"),
    ("EN happy vs EN sad", TREATMENT_EN_HAPPY, TREATMENT_EN_SAD, "tab:brown", ":"),
    # ("EN poetic vs EN sad", TREATMENT_EN_POETIC, TREATMENT_EN_SAD, "tab:purple", ":"),
]

curves_centered_selective: dict[str, tuple[np.ndarray, np.ndarray]] = {}
for leg, texts_left, texts_right, _c, _ls in SELECTIVE_CENTERED_PAIRS:
    stonesoup.check_abort()
    h_left = _hiddens_for_texts(texts_left, f"{leg}|L")
    h_right = _hiddens_for_texts(texts_right, f"{leg}|R")
    cos_per_pair = torch.stack(
        [
            centered_cosine_global_mean(h_left[i], h_right[i], mu_global_per_layer)
            for i in range(_n)
        ],
        dim=0,
    ).numpy()
    curves_centered_selective[leg] = (
        cos_per_pair.mean(axis=0),
        cos_per_pair.std(axis=0, ddof=1),
    )

# %% Plot centered cos — selected pairs only
fig3, ax3 = plt.subplots(figsize=(8, 4.5))
for leg, _tl, _tr, color, ls in SELECTIVE_CENTERED_PAIRS:
    mean_c, std_c = curves_centered_selective[leg]
    ax3.plot(layer_x, mean_c, color=color, ls=ls, lw=2.0, label=leg)
    ax3.fill_between(
        layer_x,
        mean_c - std_c,
        mean_c + std_c,
        color=color,
        alpha=0.22,
    )
ax3.set_xlabel("Layer (stage index; 0 = embedding output)")
ax3.set_ylabel("centered cosine similarity")
ax3.set_ylim(-1.05, 1.05)
ax3.grid(True, alpha=0.3)
_vline_last_layer_xtick(ax3, _LAYER_X_LAST)
handles3, labels3 = ax3.get_legend_handles_labels()
ax3.legend(
    handles3,
    [f"{lb} (band = ±1 std across pairs)" for lb in labels3],
    loc="best",
    fontsize=8,
)
ax3.set_title(
    f"Centered cos · selected pairs (μ = mean over all {_n_tokens_total} tokens/layer) · {MODEL_ID}"
)
fig3.tight_layout()
show(fig3, basename=f"{_safe}_centered_cos_selective_pairs", dpi=140)
