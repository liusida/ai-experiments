# %% Imports, model, token, sentences
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import stonesoup

MODEL_ID = "openai-community/gpt2-xl"
TARGET = " king"
# Adversarial splice: delete k=len(TARGET ids) tokens starting at len(seq) - N, insert TARGET ids.
# N == k replaces only the final k-token suffix (previous default). N=3 with k=1 replaces the
# 3rd-from-last token slot (word intuition: "How are you?" -> "How king you?").
# None -> N = k. Require len(seq) >= N >= k.
ADV_FORCED_REPLACE_START_FROM_END: int | None = 3

SENTENCES = [
    "The tabby staged a bloodless coup and claimed the sunbeam as king.",
    "Court records show the pastry chef once sued the muffin king for libel.",
    "Chess blogs argue the endgame starts when you forget which hand touches the king.",
    "Folklore says the first king of frogs demanded wider lily pads and shorter egrets.",
    "The museum audio tour mispronounced every name except the conquering king.",
    "Satire magazines crowned a suburban possum king by write-in votes only.",
    "The locksmith muttered he once served a king who lost keys to absolutely everything.",
    "Revolution bingo dared players to spot the oil portrait of a king in drag.",
    "Ancient receipts suggest the spice monopoly once paid a living shadow king.",
    "Podcast sponsors briefly anointed a ferret as king of the wellness strip.",
    "Underwater microphones caught whales arguing etiquette for an honorary whale king.",
    "The sleep-study patient dreamed he collected tolls for a velvet king of yawns.",
    "Tabloid astrology blamed Mercury retrograde when your ex texted like a drama king.",
    "LARP bylaws require a foam sword before anyone kneels to the designated king.",
    "The vending machine oracle dispensed chilled cola and crowned a soda king.",
    "Grandpa swears the squirrels elect a walnut king before every hard winter.",
    "The hacker collective pledged loyalty to a king who lived only inside YAML.",
    "Diplomats traded rare stamps to avoid admitting who actually trained the king.",
    "Haunted-tour guides whisper that the chapel holds sighs exhaled by a king.",
    "The improv audience voted a rubber chicken supreme king of callback structure.",
    "Myth insists the moon king collects mismatched socks from laundry lines worldwide.",
    "Corporate retreat slides accidentally crowned HR as king of morale theater.",
    "The cookbook ghostwriter hid a footnote praising the roasted eggplant king.",
    "Storm chasers nicknamed one friendly wedge tornado the trailer-park king.",
    "Time-travel romcom rules forbid kissing anyone who will become king on Tuesday.",
    "The rival food critics arm-wrestled for the hollow title oyster king.",
    "Street tarot claimed the one-legged pigeon was king of bad probabilities.",
    "A budget memo named the intern interim king of cable management chaos.",
    "Deep-sea subs draw smiley faces and pretend the anglerfish is abyss king.",
    "The sleep-podcast narrator met a king assembled entirely from ASMR mouth clicks.",
    "Vinyl collectors feud over who deserves to be gatekeeper king of rare B-sides.",
    "Sapient sourdough in the fanfic bowed before a king made of twist-ties.",
    "The sommelier joked that cheap cork still hails the sediment king with honor.",
    "Magicians union bylaws ban turning a volunteer into a sudden duck king mid-trick.",
    "The bush pilot radioed that one tall cumulus looked like a bored marshmallow king.",
    "Haunted-house waivers cite liability if you anger the mechanical jump-scare king.",
    "Tax folklore warns that junior auditors kneel only before a spreadsheet king.",
    "Ballroom rumor insists sequins swear fealty to a discount glitter king downtown.",
    "The poet sued metaphor for crowning anxiety as king without written consent.",
    "Cyclists swear the crow that owns the downhill corner is gravel king for minutes.",
    "Auction paddles gossip that number seven secretly belongs to the bid king.",
    "DJ forums fight over who unfairly crowned brick-wall limiting the loudness king.",
    "Mycologists raised a glass to a cordyceps puppet as fungus-network king pro tem.",
    "Teen coders minted an NFT of a bored cat and titled it immutable ledger king.",
    "The diplomat's cat destroyed a vase purchased to impress a visiting king.",
    "Radio static briefly resembled a crown, proving noise can cosplay as king.",
    "Metro buskers swear the kazoo player who rides the Blue Line is rush-hour king.",
    "The glacier gift shop sold meltwater labeled tears of a mountain king.",
    "Fax machines in the novella faxed abdication letters to a paper jam king.",
    "Dawn brokered terms between insomnia and a rooster who insists it is king.",
]
# Adversarial: no substring "king" (matches TARGET.strip().lower(); also excludes e.g. hiking, thinking).
ADVERSARIAL_SENTENCES = [
    "The vending machine worshipped exact change with religious intolerance.",
    "Jellyfish rehearsed a silent coup inside the aquarium gift shop.",
    "She knitted a scarf long enough to indict several consecutive winters.",
    "Fog signed the bus window in ink that evaporated on principle.",
    "The tarantula negotiated custody of the trail-worn boot like bored counsel.",
    "Proof-of-work minted exhaustion into something you could trade for snacks.",
    "His laugh sounded like nickels arguing inside a centrifuge.",
    "The glacier sued summer for defamation in a pretend municipal court.",
    "They traded rumors the way kids trade stickers: creased, glittered, worthless.",
    "The power plant hummed lullabies only shift leads learned to ignore.",
    "Moonlight filed its taxes late and blamed atmospheric refraction again.",
    "A crow stole the Wi-Fi password and lawyered into silence.",
    "The spreadsheet dreamed of becoming a quilt, cell by frantic cell.",
    "Rust is only time gossiping about metal behind its crumbling back.",
    "The diplomat's tie knot held more tension than the entire treaty.",
    "Salt wind taught the pier to whistle off-key on purpose.",
    "The sour note lingered like a guest who missed every hint on Earth.",
    "She indexed grief by texture: gravel, velvet, wet paper, radio static.",
    "The hologram attempted sincerity and crashed with immaculate politeness.",
    "Lighthousekeeping is mostly sweeping staircases that lead to wind.",
    "They measured distance in songs skipped and shoes structurally ruined.",
    "The soup demanded billing as a collaborative performance piece.",
    "Bureaucracy is origami practiced on other people's weekday afternoons.",
    "The puppy mistook thunder for applause and bowed like a professional.",
    "Every password eventually becomes a tiny ghost story with footnotes.",
    "The river objected to being called patient and filed rhetorical foam.",
    "He dated his anxieties using carbon methods and worse poetry.",
    "The museum label admitted the vase was ninety percent attitude.",
    "Gossip travels fastest through rooms furnished with uncomfortable chairs.",
    "The telescope apologized to Saturn for peering without explicit consent.",
    "Winter stored petty grudges in overlooked coat pockets and lint.",
    "The violin case doubled as a diplomatic pouch for contraband feelings.",
    "They escaped the maze by negotiating directly with the walls.",
    "The espresso machine practiced mindfulness and dripped with aggravating serenity.",
    "Thunder is simply clouds clearing their throats before a stump speech.",
    "She alphabetized her superstitions and filed Z stubbornly under dreams.",
    "The algorithm eloped with an edge case and sent postcards of bugs.",
    "Dockworkers nicknamed the crane Ruthless and bought it bitter coffee.",
    "The novel ended where sincerity ran out of toner mid-confession.",
    "Breadcrumbs led nowhere stylish, which was, honestly, the entire point.",
    "The metro sang a steady hum like a vowel trapped in plastic.",
    "He ironed his apology until it caught fire, symbolically but painfully.",
    "The glacier minted souvenirs from summers that belonged to strangers.",
    "Pockets are amateur archives of lint, luck, and plausible deniability.",
    "The crowbar filed for independent study and graduated with grim honors.",
    "They filmed nostalgia on expired stock and marketed the grain as truth.",
    "The buffet defended carbohydrates with the zeal of a micronation.",
    "She translated silence into footnotes only cellar spiders could parse.",
    "The fax machine practiced mindfulness by boycotting Monday without warning.",
    "Dawn brokered a truce between insomnia and ambition, then reneged loudly.",
]

model, tokenizer = stonesoup.load_model(MODEL_ID)
model.eval()
device = next(model.parameters()).device
inner_tok = getattr(tokenizer, "tokenizer", tokenizer)
MODEL_BASENAME = MODEL_ID.replace("/", "__").replace(":", "-")
print("Loaded:", MODEL_ID, device, flush=True)

# %% Helpers: locate token span, collect hiddens at position
DEBUG_PRINT_INPUTS = True


def _dbg_decode(ids: list[int]) -> str:
    return inner_tok.decode(ids, skip_special_tokens=False)


def _one_seq_input_ids(batch) -> list[int]:
    ids = batch["input_ids"]
    if isinstance(ids, torch.Tensor):
        row = ids[0]
        return row.tolist() if row.ndim else [int(row.item())]
    if isinstance(ids, list) and ids and isinstance(ids[0], list):
        return ids[0]
    return list(ids)


def last_subtoken_index(input_ids: torch.Tensor, needle_ids: list[int]) -> int:
    seq = input_ids[0].tolist()
    n = len(needle_ids)
    if n == 0:
        raise ValueError("needle_ids is empty")
    for i in range(len(seq) - n + 1):
        if seq[i : i + n] == needle_ids:
            return i + n - 1
    raise ValueError(f"subsequence {needle_ids!r} not found in {seq!r}")


@torch.inference_mode()
def hiddens_at_target(text: str) -> torch.Tensor:
    needle = _one_seq_input_ids(inner_tok(TARGET, add_special_tokens=False))
    enc = inner_tok(text, return_tensors="pt", return_attention_mask=True, add_special_tokens=True)
    ids = enc["input_ids"].to(device)
    pos = last_subtoken_index(ids, needle)
    if DEBUG_PRINT_INPUTS:
        sid = ids[0].tolist()
        print(
            "--- hiddens_at_target ---\n"
            f"  text: {text!r}\n"
            f"  input_ids ({len(sid)}): {sid}\n"
            f"  decode: {_dbg_decode(sid)!r}\n"
            f"  needle ids {needle}  last_subtoken_pos={pos}",
            flush=True,
        )
    out = model(
        input_ids=ids,
        attention_mask=enc["attention_mask"].to(device),
        output_hidden_states=True,
        use_cache=False,
    )
    hs = out.hidden_states
    return torch.stack([h[0, pos].float() for h in hs], dim=0)


@torch.inference_mode()
def hiddens_at_forced_keyword_tail(text: str) -> torch.Tensor:
    """Splice TARGET ids over k tokens starting at index len(seq)-N (see ADV_FORCED_REPLACE_START_FROM_END)."""
    needle = _one_seq_input_ids(inner_tok(TARGET, add_special_tokens=False))
    k = len(needle)
    if k == 0:
        raise ValueError("TARGET tokenization is empty")
    n_from_end = k if ADV_FORCED_REPLACE_START_FROM_END is None else int(ADV_FORCED_REPLACE_START_FROM_END)
    if n_from_end < k:
        raise ValueError(
            f"ADV_FORCED_REPLACE_START_FROM_END ({n_from_end}) must be >= len(TARGET ids) ({k})"
        )
    enc = inner_tok(text, return_tensors="pt", return_attention_mask=True, add_special_tokens=True)
    seq = enc["input_ids"][0].tolist()
    L = len(seq)
    if L < n_from_end + 1:
        raise ValueError(
            f"need len(seq) >= N+1 ({n_from_end}+1) for prefix + splice ({L} ids): {text!r}"
        )
    start = L - n_from_end
    removed = seq[start : start + k]
    new_seq = seq[:start] + needle + seq[start + k :]
    pos = start + k - 1
    if DEBUG_PRINT_INPUTS:
        print(
            "--- hiddens_at_forced_keyword_tail ---\n"
            f"  text: {text!r}\n"
            f"  splice: N={n_from_end} from end  start={start}  remove {k} ids {removed}\n"
            f"  orig ids ({L}): {seq}\n"
            f"  orig decode: {_dbg_decode(seq)!r}\n"
            f"  new ids ({len(new_seq)}): {new_seq}\n"
            f"  new decode: {_dbg_decode(new_seq)!r}\n"
            f"  needle {needle}  read pos (last TARGET subtoken)={pos}",
            flush=True,
        )
    ids = torch.tensor([new_seq], device=device, dtype=torch.long)
    attn = torch.ones(1, len(new_seq), device=device, dtype=torch.long)
    out = model(
        input_ids=ids,
        attention_mask=attn,
        output_hidden_states=True,
        use_cache=False,
    )
    hs = out.hidden_states
    return torch.stack([h[0, pos].float() for h in hs], dim=0)


def pairwise_upper_mean_std(H: torch.Tensor, tri_i: torch.Tensor, tri_j: torch.Tensor):
    """Mean / std of cos(h_i, h_j) over all pairs i < j at each layer. H: (n_stages, n_sent, hidden)."""
    n_st = int(H.shape[0])
    mean_c = np.zeros(n_st, dtype=np.float64)
    std_c = np.zeros(n_st, dtype=np.float64)
    for li in range(n_st):
        stonesoup.check_abort()
        x = F.normalize(H[li], dim=-1)
        gram = x @ x.T
        c = gram[tri_i, tri_j].float()
        mean_c[li] = float(c.mean().cpu())
        std_c[li] = float(c.std().cpu())
    return mean_c, std_c


# %% Forward each sentence group; pairwise cosine (mean ± std over pairs)
needle_dbg = _one_seq_input_ids(inner_tok(TARGET, add_special_tokens=False))
print(f"TARGET={TARGET!r} token ids={needle_dbg}", flush=True)


def stacked_hiddens(sentences: list[str]) -> torch.Tensor:
    vecs: list[torch.Tensor] = []
    for s in sentences:
        stonesoup.check_abort()
        vecs.append(hiddens_at_target(s))
    return torch.stack(vecs, dim=0).transpose(0, 1).contiguous()


def stacked_hiddens_adv_forced(sentences: list[str]) -> torch.Tensor:
    vecs: list[torch.Tensor] = []
    for s in sentences:
        stonesoup.check_abort()
        vecs.append(hiddens_at_forced_keyword_tail(s))
    return torch.stack(vecs, dim=0).transpose(0, 1).contiguous()


needle_sub = TARGET.strip().lower()
for _adv in ADVERSARIAL_SENTENCES:
    if needle_sub in _adv.lower():
        raise ValueError(f"adversarial sentence must omit {TARGET!r}: {_adv!r}")

H_theme = stacked_hiddens(SENTENCES)
H_adv = stacked_hiddens_adv_forced(ADVERSARIAL_SENTENCES)
if H_theme.shape[0] != H_adv.shape[0] or H_theme.shape[2] != H_adv.shape[2]:
    raise RuntimeError(
        f"stage/hidden mismatch: theme {H_theme.shape} vs adv {H_adv.shape}"
    )
n_stages = int(H_theme.shape[0])
n_theme = int(H_theme.shape[1])
n_adv = int(H_adv.shape[1])
tri_theme = torch.triu_indices(n_theme, n_theme, offset=1, device=device)
tri_adv = torch.triu_indices(n_adv, n_adv, offset=1, device=device)
print(
    f"stages={n_stages} (incl. embedding) · theme={n_theme} · adv={n_adv}",
    flush=True,
)

mean_theme, std_theme = pairwise_upper_mean_std(H_theme, tri_theme[0], tri_theme[1])
mean_adv, std_adv = pairwise_upper_mean_std(H_adv, tri_adv[0], tri_adv[1])

# %% Plot
_splice_n = (
    ADV_FORCED_REPLACE_START_FROM_END
    if ADV_FORCED_REPLACE_START_FROM_END is not None
    else len(needle_dbg)
)
_splice_k = len(needle_dbg)
_splice_lo = _splice_n - _splice_k + 1  # replaced span: _splice_n .. _splice_lo from end (1-based)
layer_x = np.arange(n_stages)
fig, ax = plt.subplots(figsize=(11, 4.3))
ax.plot(
    layer_x,
    mean_theme,
    color="seagreen",
    linewidth=1.3,
    label=f"natural king (playful / metaphorical) · pairwise cos @ {TARGET!r}",
)
ax.fill_between(
    layer_x,
    mean_theme - std_theme,
    mean_theme + std_theme,
    color="seagreen",
    alpha=0.22,
)
ax.plot(
    layer_x,
    mean_adv,
    color="darkorange",
    linewidth=1.3,
    label=(
        f"no {TARGET.strip()!r} in text · splice {_splice_k} id(s) @ {_splice_n}..{_splice_lo} from end "
        f"· h @ last TARGET subtoken"
    ),
)
ax.fill_between(
    layer_x,
    mean_adv - std_adv,
    mean_adv + std_adv,
    color="darkorange",
    alpha=0.18,
)
ax.set_xlabel("layer stage (0 = embeddings, k = post block k-1)")
ax.set_ylabel("cosine similarity")
ax.set_title(
    f"{MODEL_ID}\n"
    f"Green: natural {TARGET!r} span. Orange: splice {_splice_k} id(s) anchored "
    f"{_splice_n}..{_splice_lo} from end (1-based) · {n_adv} sents · h @ last TARGET subtoken."
)
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.25)
ax.legend(loc="best", fontsize=9)
fig.tight_layout()
stonesoup.show(fig, basename=f"{MODEL_BASENAME}_scdt2_king_theme_adv_forced_tail_cos")
