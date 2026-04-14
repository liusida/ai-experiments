# %% Imports
from __future__ import annotations

import numpy as np
import stonesoup
import torch
from anatome.distance import (
    linear_cka_distance,
    orthogonal_procrustes_distance,
    pwcca_distance,
    svcca_distance,
)
from stonesoup.experiment import (
    capture_embed_and_post_blocks,
    configure_matplotlib_agg,
    ensure_pad_token_via_eos,
    hf_repo_id_safe_stem,
    inner_tokenizer,
    show,
)

# %% Knobs — model, layers, and long prompt text
MODEL_ID = "Qwen/Qwen3.5-0.8B"
# METHOD = "pwcca"
# METHOD = "svcca"
METHOD = "lincka"
MAX_SEQ = 16384
# Long text → many token rows (PWCCA/SVCCA want D ≥ hidden size).
# Sixty short passages: mixed genres, registers, and languages (non-sink tokens concatenated).
SENTENCES: list[str] = [
    # Expository English (science)
    "Chloroplasts use chlorophyll to absorb photons and store energy in ATP and NADPH. "
    "Those carriers then power the Calvin cycle, which fixes carbon into sugars the cell can use.",
    # Spoken dialogue
    '"Did you remember the keys?"\n"On the hook—unless the cat knocked them down again."',
    # Chinese (informative)
    "月球围绕地球公转，同一面始终朝向地球；潮汐主要由月球引力引起。",
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
PROMPT = "Here are some sentences:\n\n" + "\n\n".join(SENTENCES)

def pairwise_layer_distance(a: torch.Tensor, b: torch.Tensor, method: str) -> torch.Tensor:
    """PWCCA/SVCCA call ``torch.linalg.svd`` inside anatome; fall back to ``qr`` / float64 on failure."""
    if method == "pwcca":
        last_err: BaseException | None = None
        for backend, dtype in (
            ("svd", torch.float32),
            ("qr", torch.float32),
            ("svd", torch.float64),
            ("qr", torch.float64),
        ):
            aa, bb = a.to(dtype=dtype), b.to(dtype=dtype)
            try:
                return pwcca_distance(aa, bb, backend=backend)
            except torch.linalg.LinAlgError as e:
                last_err = e
        raise last_err if last_err else RuntimeError("pwcca fallback exhausted")
    if method == "svcca":
        last_err = None
        for backend, dtype in (
            ("svd", torch.float32),
            ("qr", torch.float32),
            ("svd", torch.float64),
            ("qr", torch.float64),
        ):
            aa, bb = a.to(dtype=dtype), b.to(dtype=dtype)
            try:
                return svcca_distance(aa, bb, accept_rate=0.99, backend=backend)
            except torch.linalg.LinAlgError as e:
                last_err = e
        raise last_err if last_err else RuntimeError("svcca fallback exhausted")
    if method == "lincka":
        return linear_cka_distance(a, b, reduce_bias=False)
    if method == "opd":
        return orthogonal_procrustes_distance(a, b)
    raise ValueError(
        f"unknown method={method!r} (use pwcca, svcca, lincka, opd)",
    )


# %% Load checkpoint (shared pool — same as toolbar Load)
model, proc = stonesoup.load_model(MODEL_ID)
device = next(model.parameters()).device
tok = inner_tokenizer(proc)
ensure_pad_token_via_eos(tok)
model.eval()

# %% Encode on device (truncation for very long concatenated text)
# Call ``tok`` directly so truncation works without relying on a reloaded ``encode_text_inputs``.
enc = tok(
    PROMPT,
    return_tensors="pt",
    return_attention_mask=True,
    add_special_tokens=True,
    truncation=True,
    max_length=MAX_SEQ,
)
inputs = {k: v.to(device) for k, v in enc.items()}
ids = inputs["input_ids"]
n_emb = model.get_input_embeddings().weight.shape[0]
hi, lo = int(ids.max()), int(ids.min())
if lo < 0 or hi >= n_emb:
    raise ValueError(
        f"input_ids range [{lo}, {hi}] incompatible with embedding rows {n_emb} "
        "(wrong tokenizer vs model, or stale kernel). Re-run the Load cell after changing MODEL_ID."
    )

# %% Forward: embedding + post-block hiddens (one pass)
stack, names = capture_embed_and_post_blocks(model, inputs, use_cache=False)
n_stage = stack.shape[0]
hidden = int(stack.shape[-1])
acts = [stack[i].reshape(-1, hidden).float() for i in range(n_stage)]

# %% Pairwise similarity (1 − distance)
sim = np.ones((n_stage, n_stage), dtype=np.float64)
for i in range(n_stage):
    for j in range(i + 1, n_stage):
        stonesoup.check_abort()
        print(f"Computing similarity between {names[i]} and {names[j]}")
        s = 1.0 - pairwise_layer_distance(acts[i], acts[j], METHOD).item()
        sim[i, j] = s
        sim[j, i] = s

# %% Plot heatmap

configure_matplotlib_agg()
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(max(6.0, 0.35 * n_stage), max(5.0, 0.35 * n_stage)))
im = ax.imshow(sim, cmap="Blues", aspect="equal", vmin=0.0, vmax=1.0)
ax.set_xticks(np.arange(n_stage))
ax.set_yticks(np.arange(n_stage))
ax.set_xticklabels(names, rotation=75, ha="right", fontsize=7)
ax.set_yticklabels(names, fontsize=7)
ax.set_title(f"{METHOD}: pairwise layer similarity ({MODEL_ID}, D×H = {acts[0].shape[0]}×{hidden})")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.tight_layout()
show(fig, basename=f"{hf_repo_id_safe_stem(MODEL_ID)}_{METHOD}_layer_similarity_heatmap", dpi=140)

iu = np.triu_indices(n_stage, k=1)
off = sim[iu]
print(
    f"{METHOD}: matrix {n_stage}×{n_stage}, "
    f"off-diagonal min={off.min():.6f} max={off.max():.6f}",
    flush=True,
)
