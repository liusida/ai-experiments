# %% Imports & config
from __future__ import annotations

import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F

import stonesoup
from stonesoup.experiment import (
    configure_matplotlib_agg,
    decoder_blocks,
    ensure_pad_token_via_eos,
    hf_repo_id_safe_stem,
    inner_tokenizer,
)

configure_matplotlib_agg()

MODELS: list[str] = [
    "meta-llama/llama-3.2-3B",
    "google/gemma-2-2b",
    # "Qwen/Qwen3.5-9B",
]

SENTENCES: list[str] = [
    # Expository English (science)
    "Chloroplasts use chlorophyll to absorb photons and store energy in ATP and NADPH. "
    "Those carriers then power the Calvin cycle, which fixes carbon into sugars the cell can use.",
    # Spoken dialogue
    '"Did you remember the keys?"\n"On the hook—unless the cat knocked them down again."',
    # Chinese (informative)
    "月球围绕地球公转，同一面始终朝向地球；潮汐主要由月球引力引起。",
    "The Moon orbits the Earth, always showing the same face toward Earth; tides are mainly caused by the Moon's gravitational pull.",
    "הירח סובב סביב כדור הארץ, כשאותו צד פונה תמיד לכיוון כדור הארץ; גאות ושפל נגרמים בעיקר על ידי כוח המשיכה של הירח.",
    # Children’s storybook tone
    "The little boat wished for wings, so the wind stitched clouds into sails and pushed it upstream.",
    # Statutory / legal style
    "Where a party fails to perform without excuse, the non-breaching party may seek damages as provided herein.",
    # Recipe / procedural
    "Whisk eggs with salt, fold in warm rice off the heat, then sprinkle nori without over-stirring.",
    # Poetry-ish (line breaks as in source)
    "Fog on the pier—\nA gull borrows the moon\nAnd flies away.",
    # Kid-friendly fable
    "The fox promised grapes were sour anyway, but the crow still laughed from the high branch.",
    # News headline + lead
    "City council delays vote: residents packed the hall, some holding signs that read “Fix the pipes first.”",
    # Text-message / informal
    "omw — grab a table near the window?? coffee’s on me if traffic eats me alive lol",
    # Second Chinese (colloquial narrative)
    "周末我想去爬山，如果下雨就改在家里煮火锅、看电影。",
    # Technical / spec tone
    "Requirement: latency p99 under 120 ms; fallback path must degrade gracefully without data loss.",
    # Courtroom dialogue
    "Your Honor, the exhibit is authenticated under Rule 902—the chain of custody is unbroken.",
    # Sports play-by-play
    "She fakes left, splits two defenders, and curls one into the top corner—stadium erupts.",
    # Academic philosophy (dense)
    "Normative claims concern what ought to be; descriptive claims concern what is—confusing them risks the is-ought gap.",
    # Product blurb / marketing
    "This jacket repels drizzle, packs into its pocket, and weighs less than your phone—trail-tested.",
    # Medical chart note style
    "Patient reports intermittent vertigo; differential includes BPPV versus orthostatic hypotension.",
    # Email closings / formal
    "Please find the revised figures attached. I remain available for a brief call next Tuesday.",
    # Myth / epic register
    "When the river refused the oath, the old king broke his crown and scattered the shards downstream.",
    # Code-adjacent comment (natural language)
    "# TODO: replace O(n^2) pairing with hash map once we confirm key distribution in prod logs.",
    # Japanese
    "光合作用是植物が二酸化炭素と水から糖を作る仕組みで、酸素が副産物として出る。",
    # Spanish
    "El metro llegó con retraso, pero al menos había asiento libre junto a la ventana.",
    # French
    "Si la bibliothèque ferme à dix-neuf heures, il faudra rendre le livre avant la sonnerie.",
    # German
    "Der Wind trug den Geruch von Kiefern über den See, und die Segel knisterten leise.",
    # Korean
    "비가 그치면 산책로를 따라 걸어가서, 카페에서 따뜻한 차 한 잔을 마시고 싶다.",
    # Hindi (Devanagari)
    "बारिश के बाद फुहारें हवा में इत्र की तरह फैलती हैं और बच्चे कल्लों पर छलांग लगाते हैं।",
    # Arabic
    "في الصباح الباكر، استمعت إلى الأذان وهو يمتزج مع خرير الماء في الحارة الضيقة.",
    # Portuguese (Brazil)
    "A feira da praça tinha goiaba madura, queijo de cabra e um violeiro que não parava de cantar.",
    # Italian
    "Preferisco il caffè ristretto dopo pranzo, ma oggi mi hanno offerto solo una tisana alla menta.",
    # Russian
    "Старый трамвай медленно поднимался по мосту, а за окном мелькали огни наводнённых улиц.",
    # Weather report register
    "Overnight lows near freezing; patchy frost in sheltered valleys before sunrise tomorrow.",
    # README / OSS tone
    "Contributions welcome—open an issue first if you plan to change the public Python API surface.",
    # Parenthetical academic
    "We operationalize “surprise” as negative log-probability under the fitted n-gram baseline (see §3.2).",
    # Interview transcript style
    "[Interviewer] What changed after the refactor? [Engineer] Honestly—we could finally profile without fear.",
    # Late-night internal monologue
    "If I reply now it’ll look obsessive; if I wait, indifferent—so I draft, delete, draft again.",
    # Appalachian-flavored English (literary)
    "You couldn’t lie to that creek if you tried—the stones remember whose boots crossed last spring.",
    # Second-person microfiction
    "You choose the sealed envelope; behind the door, someone exhales as if they’d held it for years.",
    # Board-game rulebook
    "Discard down to seven cards at the end of your turn; ties for first resolve clockwise from dealer.",
    # Radio ad
    "Local listeners save fifteen percent on winter tires this week only—mention code GRIP when you book.",
    # Slack / workplace
    "Looping in Legal on the draft SLA—no ship date until we get sign-off on the liability cap language.",
    # ESL learner sentence (correct but stiff)
    "I am learning how to explain my opinion in a polite way when I disagree with my colleague.",
    # Pirate / theatrical (playful)
    "Avast—yield the chart, or we’ll measure the depth with ye… creatively, says I.",
    # Math textbook tone
    "Let f be continuous on [a, b]; then f attains both a maximum and a minimum on that interval.",
    # knitting/craft blog
    "Slip the first stitch purlwise each row for a tidy edge on the ribbing before you join in the round.",
    # Third Chinese (newsy)
    "据气象台预报，今夜沿海风力增强，渔船应及早回港避风。",
    # Third Japanese
    "図書館の返却期限を過ぎたので、延長の手続きをオンラインで済ませた。",
    # Vietnamese
    "Chiều nay trời mát, tôi đạp xe qua cầu và ngửi mùi hoa sữa thoang thoảng hai bên đường.",
    # Thai
    "เช้านี้ตลาดสดคึกคัก มีข้าวเหนียวมะม่วงและน้ำพริกเผาที่หอมกรุ่นเป็นพิเศษ",
    # Indonesian
    "Anak-anak menunggu gerobak sate keliling sambil menghitung mobil lewat di depan gang sempit.",
    # Turkish
    "Fırından çıkan ekmeğin kabuğu çıtır, içi ise buğday kokusuyla dolup taşıyordu.",
    # Polish
    "Na peronie stał zapomniany parasol; nikt nie wiedział, czyje łokcie zgubiły go w pośpiechu.",
    # Dutch
    "De fietspaden waren glad door de ijzel, dus we liepen liever langs het kanaal met de handen in de zakken.",
    # Ancient-history pastiche
    "The envoy carried no gifts—only a clay tablet scratched with a treaty the king had not yet read aloud.",
    # Science fiction ship log
    "Day 412: No anomalies detected; the spectroscope still hums like distant insects in dry grass.",
    # Haiku-ish (English)
    "Icicles drip—\neach holds a shard of sky\nbefore it breaks.",
    # Recipe (baking, terse)
    "Proof dough until doubled; punch down, shape loaves, bake at 220°C until the crust sings hollow.",
    # Therapist soap-note shorthand
    "Client reports improved sleep hygiene; PHQ-9 down 3 pts; plan to continue weekly CBT homework.",
    # Kindergarten teacher voice
    "Everybody find a yellow crayon—and remember, we share sky with the birds, not with our elbows.",
    # Sports interview cliché
    "We’re taking it one game at a time; the locker room stays focused on fundamentals and each other.",
    # Grant proposal sentence
    "The proposed work will release reproducible artifacts under the MIT license alongside preprocessed splits.",
]
SENTENCES = SENTENCES[:10]
MAX_LENGTH = 128
SKIP_FIRST_TOKENS = 1


# %% Align tokens & capture activations

def _span_map(
    proc, sentence: str, *, max_length: int, skip_first: int,
) -> dict[tuple[int, int], int]:
    tok = inner_tokenizer(proc)
    ensure_pad_token_via_eos(tok)
    enc = tok(
        sentence,
        return_offsets_mapping=True,
        return_tensors="pt",
        add_special_tokens=True,
        max_length=max_length,
        truncation=True,
    )
    om = enc["offset_mapping"][0]
    seq_len = int(enc["attention_mask"][0].sum().item())
    out: dict[tuple[int, int], int] = {}
    for t in range(skip_first, seq_len):
        a, b = int(om[t][0]), int(om[t][1])
        if b > a:
            out[(a, b)] = t
    return out


def _mid_hook_target(block: torch.nn.Module) -> torch.nn.Module | None:
    """Find the submodule whose pre-hook input is the residual stream mid-point.

    Gemma-2 (sandwich norm) has *both* attributes; ``pre_feedforward_layernorm``
    sits after the first residual add, so it must be checked first.
    """
    # Gemma-2 sandwich norm: pre_feedforward_layernorm input = residual after attn+norm+add
    if hasattr(block, "pre_feedforward_layernorm"):
        return block.pre_feedforward_layernorm
    # Llama / Mistral / Qwen pre-norm: post_attention_layernorm input = residual after attn+add
    if hasattr(block, "post_attention_layernorm"):
        return block.post_attention_layernorm
    return None


def capture_embed_mid_post(
    model: torch.nn.Module, inputs: dict, **forward_kw,
) -> tuple[torch.Tensor, list[str]]:
    """Capture embedding, mid-layer (post-attn residual), and post-layer for every block."""
    blocks = decoder_blocks(model)
    captured: list[torch.Tensor] = []

    def _post(_mod, _inp, x):
        h = x[0] if isinstance(x, tuple) else x
        captured.append(h.detach())

    def _pre(_mod, inp):
        h = inp[0] if isinstance(inp, tuple) else inp
        captured.append(h.detach())

    emb = model.get_input_embeddings()
    hooks = [emb.register_forward_hook(_post)]
    names = ["embed"]

    for i, block in enumerate(blocks):
        mid_target = _mid_hook_target(block)
        if mid_target is not None:
            hooks.append(mid_target.register_forward_pre_hook(_pre))
            names.append(f"L{i}_mid")
        hooks.append(block.register_forward_hook(_post))
        names.append(f"L{i}_post")

    try:
        with torch.inference_mode():
            model(**{**inputs, **forward_kw})
    finally:
        for h in hooks:
            h.remove()

    if len(captured) != len(names):
        raise RuntimeError(
            f"capture_embed_mid_post: expected {len(names)} stages, got {len(captured)}"
        )
    return torch.stack(captured, dim=0), names


align_by_repo: dict[str, list[list[int]]] = {r: [] for r in MODELS}
common_spans: list[list[tuple[int, int]]] = []

for sentence in SENTENCES:
    stonesoup.check_abort()
    maps: dict[str, dict[tuple[int, int], int]] = {}
    for rid in MODELS:
        _, proc = stonesoup.load_model(rid)
        maps[rid] = _span_map(proc, sentence, max_length=MAX_LENGTH, skip_first=SKIP_FIRST_TOKENS)
    common = sorted(set.intersection(*(set(m.keys()) for m in maps.values())))
    common_spans.append(common)
    for rid in MODELS:
        align_by_repo[rid].append([maps[rid][sp] for sp in common])

token_labels = [
    SENTENCES[si][a:b] for si, spans in enumerate(common_spans) for a, b in spans
]
print(f"Aligned tokens ({len(token_labels)}): {token_labels}", flush=True)

acts: dict[str, torch.Tensor] = {}
stage_names_by_repo: dict[str, list[str]] = {}
for repo_id in MODELS:
    stonesoup.check_abort()
    model, proc = stonesoup.load_model(repo_id)
    model.eval()
    device = next(model.parameters()).device
    tok = inner_tokenizer(proc)
    ensure_pad_token_via_eos(tok)
    chunks: list[torch.Tensor] = []
    snames: list[str] = []
    for sentence, idxs in zip(SENTENCES, align_by_repo[repo_id], strict=True):
        if not idxs:
            continue
        enc = tok(
            sentence,
            return_tensors="pt",
            return_attention_mask=True,
            add_special_tokens=True,
            max_length=MAX_LENGTH,
            truncation=True,
        )
        inputs = {k: v.to(device) for k, v in enc.items()}
        stack, snames = capture_embed_mid_post(model, inputs, use_cache=False)
        seq_len = min(int(inputs["attention_mask"][0].sum()), stack.shape[2])
        all_layers = stack[:, 0, :seq_len, :].detach().float()
        idx_t = torch.as_tensor(idxs, device=device)
        chunks.append(all_layers[:, idx_t, :])
    acts[repo_id] = torch.cat(chunks, dim=1)
    stage_names_by_repo[repo_id] = snames
    print(
        f"{repo_id}: shape={tuple(acts[repo_id].shape)} (stages, tokens, dim)",
        flush=True,
    )

# %% plot one layer for model[0]
def plot_layer_rsm(repo_id: str, layer_index: int):
    snames = stage_names_by_repo[repo_id]
    normed = F.normalize(acts[repo_id][layer_index], dim=-1, eps=1e-8)
    sim = (normed @ normed.T).cpu().numpy()
    n = len(token_labels)
    fig_size = max(6, n * 0.32)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(sim, vmin=-1, vmax=1, cmap="RdBu_r", aspect="equal", interpolation="nearest")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(token_labels, rotation=90, fontsize=7, fontfamily="monospace")
    ax.set_yticklabels(token_labels, fontsize=7, fontfamily="monospace")
    short = repo_id.split("/")[-1]
    layer_name = snames[layer_index] if layer_index < len(snames) else f"L{layer_index}"
    ax.set_title(f"RSM (cosine sim) — {short} {layer_name}", fontsize=12, pad=10)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("cosine similarity", fontsize=10)
    fig.tight_layout()
    stonesoup.show(fig, basename=f"rsm_{hf_repo_id_safe_stem(repo_id)}_L{layer_index}", dpi=144)

plot_layer_rsm(MODELS[0], 0)

# %% plot one layer for model[1]
plot_layer_rsm(MODELS[0], 1)

# %% plot the delta
def plot_rsm_delta(repo_id_0: str, layer_index_0: int, repo_id_1: str, layer_index_1: int):
    rsm0 = F.normalize(acts[repo_id_0][layer_index_0], dim=-1, eps=1e-8)
    rsm0 = (rsm0 @ rsm0.T).cpu()
    rsm1 = F.normalize(acts[repo_id_1][layer_index_1], dim=-1, eps=1e-8)
    rsm1 = (rsm1 @ rsm1.T).cpu()
    delta = (rsm0 - rsm1).numpy()
    n = len(token_labels)
    fig_size = max(6, n * 0.32)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(delta, vmin=-0.5, vmax=0.5, cmap="RdBu_r", aspect="equal", interpolation="nearest")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(token_labels, rotation=90, fontsize=7, fontfamily="monospace")
    ax.set_yticklabels(token_labels, fontsize=7, fontfamily="monospace")
    short0 = repo_id_0.split("/")[-1]
    short1 = repo_id_1.split("/")[-1]
    ax.set_title(
        f"RSM delta: {short0} L{layer_index_0} − {short1} L{layer_index_1}",
        fontsize=12, pad=10,
    )
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Δ cosine similarity", fontsize=10)
    fig.tight_layout()
    stonesoup.show(
        fig,
        basename=f"rsm_delta_{hf_repo_id_safe_stem(repo_id_0)}_L{layer_index_0}"
                 f"_vs_{hf_repo_id_safe_stem(repo_id_1)}_L{layer_index_1}",
        dpi=144,
    )
for i in range(27):
    plot_rsm_delta(MODELS[0], i+1, MODELS[1], i)

# %% MODELS[0] — RSM heatmaps (all layers, one figure per model)
def plot_rsm_heatmaps(repo_id: str):
    act = acts[repo_id]
    snames = stage_names_by_repo[repo_id]
    n_stages, n_tokens = act.shape[0], act.shape[1]

    n_cols = 8
    n_rows = -(-n_stages // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.5, n_rows * 2.5))
    axes_flat = axes.flatten()

    for i in range(n_stages):
        stonesoup.check_abort()
        normed = F.normalize(act[i], dim=-1, eps=1e-8)
        sim = (normed @ normed.T).cpu().numpy()
        ax = axes_flat[i]
        ax.imshow(sim, vmin=-1, vmax=1, cmap="RdBu_r", aspect="equal", interpolation="nearest")
        ax.set_title(snames[i], fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    for i in range(n_stages, len(axes_flat)):
        axes_flat[i].set_visible(False)

    short = repo_id.split("/")[-1]
    fig.suptitle(
        f"Token RSM (cosine similarity, all layers) — {short}",
        fontsize=14,
        y=1.01,
    )
    fig.tight_layout()
    stonesoup.show(fig, basename=f"rsm_all_layers_{hf_repo_id_safe_stem(repo_id)}", dpi=120)

plot_rsm_heatmaps(MODELS[0])

# %% MODELS[1] — RSM heatmaps (all layers, one figure per model)
plot_rsm_heatmaps(MODELS[1])

# %% CKA heatmap (cosine-sim RSMs, layer × layer across models)
m0, m1 = MODELS[0], MODELS[1]
n0, n1 = acts[m0].shape[0], acts[m1].shape[0]
n_tokens = acts[m0].shape[1]


def _centered_rsm_vec(act_layer: torch.Tensor) -> torch.Tensor:
    """Cosine-similarity RSM, double-centered, then flattened."""
    normed = F.normalize(act_layer.double(), dim=-1, eps=1e-8)
    rsm = normed @ normed.T
    H = torch.eye(n_tokens, device=rsm.device, dtype=rsm.dtype) - 1.0 / n_tokens
    return (H @ rsm @ H).reshape(-1)


vecs0 = torch.stack([_centered_rsm_vec(acts[m0][i]) for i in range(n0)])
vecs1 = torch.stack([_centered_rsm_vec(acts[m1][i]) for i in range(n1)])

cka = (F.normalize(vecs0, dim=-1) @ F.normalize(vecs1, dim=-1).T).cpu().numpy()

fig, ax = plt.subplots(figsize=(max(8, n1 * 0.3), max(6, n0 * 0.3)))
im = ax.imshow(cka, vmin=0.6, vmax=1, cmap="Blues", aspect="equal", interpolation="none")
ax.set_xlabel(m1.split("/")[-1], fontsize=12)
ax.set_ylabel(m0.split("/")[-1], fontsize=12)
ax.set_xticks(range(n1))
ax.set_yticks(range(n0))
ax.set_xticklabels(stage_names_by_repo[m1], rotation=90, fontsize=7)
ax.set_yticklabels(stage_names_by_repo[m0], fontsize=7)
ax.set_title("CKA (cosine-sim RSM) — layer × layer", fontsize=13, pad=10)
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cb.set_label("CKA", fontsize=10)
fig.tight_layout()
stonesoup.show(fig, basename="cka_cosrsm_layer_vs_layer", dpi=144)
