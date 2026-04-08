# %% Imports & load model
from __future__ import annotations

import random

import torch
import torch.nn.functional as F
import stonesoup

MODEL_ID = "Qwen/Qwen3-8B-Base"
# Base LMs: use raw user text only (no ``apply_chat_template`` / role tokens). Set True for
# instruct checkpoints if you want HF chat formatting.
USE_CHAT_TEMPLATE = False
# Never use ``k=0`` for control (first user token on both sides inflates cosine mid-depth).
CONTROL_EXCLUDE_FIRST_USER_TOKEN = True

model, tokenizer = stonesoup.load_model(MODEL_ID)
model.eval()
device = next(model.parameters()).device
# VL bundles return a Processor: never ``processor(text)`` for encoding — that routes to
# image decoding. Always use the wrapped tokenizer for text ids / offsets.
inner_tok = getattr(tokenizer, "tokenizer", tokenizer)
_apply_chat_template = getattr(
    tokenizer, "apply_chat_template", None
) or getattr(inner_tok, "apply_chat_template", None)
print(
    "Loaded:",
    MODEL_ID,
    device,
    f"chat_template={'on' if USE_CHAT_TEMPLATE else 'off'}",
    flush=True,
)

# %% Helpers: backbone layers, prompt encoding, homonym index, single-sequence forward


def _decoder_blocks(model: torch.nn.Module) -> torch.nn.ModuleList:
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    if hasattr(model, "model"):
        inner = model.model
        if hasattr(inner, "language_model") and hasattr(inner.language_model, "layers"):
            return inner.language_model.layers
        if hasattr(inner, "layers"):
            return inner.layers
    raise TypeError(
        "Cannot find decoder layers: need transformer.h, model.language_model.layers, "
        f"or model.layers. Got {type(model).__name__}"
    )


def _chat_prompt_ids_and_mask(
    sentence: str,
) -> tuple[torch.Tensor, torch.Tensor, str, bool]:
    """Return ids, mask, prompt string, and ``add_special_tokens`` used for forward / offsets."""
    messages = [{"role": "user", "content": sentence.strip()}]
    if (
        not USE_CHAT_TEMPLATE
        or getattr(inner_tok, "chat_template", None) is None
        or _apply_chat_template is None
    ):
        prompt = sentence.strip()
        enc = inner_tok(
            prompt,
            return_tensors="pt",
            return_attention_mask=True,
            add_special_tokens=True,
        )
        return enc["input_ids"].to(device), enc["attention_mask"].to(device), prompt, True
    prompt = _apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    enc = inner_tok(
        prompt,
        return_tensors="pt",
        return_attention_mask=True,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    return (
        enc["input_ids"].to(device),
        enc["attention_mask"].to(device),
        prompt,
        False,
    )


def _offset_pairs(enc: dict) -> list[tuple[int | None, int | None]]:
    """``(seq_len, 2)`` offset rows as a list of ``(start, end)`` (HF + torch layouts)."""
    om = enc["offset_mapping"]
    if isinstance(om, torch.Tensor):
        t = om
        if t.dim() == 3:
            t = t[0]
        # (seq, 2)
        out: list[tuple[int | None, int | None]] = []
        for i in range(t.shape[0]):
            out.append((int(t[i, 0]), int(t[i, 1])))
        return out
    if isinstance(om, list) and om and isinstance(om[0], list):
        row = om[0]
    else:
        row = om
    pairs: list[tuple[int | None, int | None]] = []
    for span in row:
        if span is None:
            pairs.append((None, None))
        elif isinstance(span, (list, tuple)) and len(span) >= 2:
            pairs.append((int(span[0]), int(span[1])))
        else:
            raise TypeError(f"Unexpected offset span type {type(span)!r}: {span!r}")
    return pairs


def _homonym_token_index_last_subword(
    prompt: str, needle: str, *, add_special_tokens: bool
) -> int:
    """Index of the last token overlapping ``needle`` (must appear exactly once in ``prompt``)."""
    if prompt.count(needle) != 1:
        raise ValueError(
            f"needle {needle!r} must occur exactly once in prompt; got {prompt.count(needle)}"
        )
    char_start = prompt.index(needle)
    char_end = char_start + len(needle)
    enc = inner_tok(
        prompt,
        add_special_tokens=add_special_tokens,
        return_offsets_mapping=True,
    )
    overlapping: list[int] = []
    for i, (s, e) in enumerate(_offset_pairs(enc)):
        if s is None or e is None or e <= s:
            continue
        if s < char_end and e > char_start:
            overlapping.append(i)
    if not overlapping:
        raise RuntimeError(f"No token overlaps needle {needle!r} in prompt")
    return overlapping[-1]


def _user_content_token_lo_hi(
    prompt: str,
    user_sentence: str,
    *,
    add_special_tokens: bool,
) -> tuple[int, int]:
    """
    Inclusive token bounds ``(lo, hi)`` for tokens overlapping the verbatim user text
    (``user_sentence.strip()`` once in ``prompt``). Used so a “aligned” control compares
    the **same offset inside the user sentence**, not the same global index (chat markup
    breaks global alignment).
    """
    body = user_sentence.strip()
    if prompt.count(body) != 1:
        raise ValueError(
            "user sentence must appear exactly once as a substring of the rendered prompt; "
            f"got count={prompt.count(body)} sentence={body[:100]!r}"
        )
    c0 = prompt.index(body)
    c1 = c0 + len(body)
    enc = inner_tok(
        prompt,
        add_special_tokens=add_special_tokens,
        return_offsets_mapping=True,
    )
    idxs: list[int] = []
    for i, (s, e) in enumerate(_offset_pairs(enc)):
        if s is None or e is None or e <= s:
            continue
        if s < c1 and e > c0:
            idxs.append(i)
    if not idxs:
        raise RuntimeError("No token overlaps user text span in prompt")
    return min(idxs), max(idxs)


def _eligible_control_k_distinct_ids(
    ids_a: torch.Tensor,
    ids_b: torch.Tensor,
    *,
    ua_lo: int,
    ub_lo: int,
    m_user: int,
    homonym_rel_excluded: set[int],
) -> tuple[list[int], list[int], bool]:
    """
    Offsets ``k`` into the aligned user-token spans: start from ``k`` not in
    ``homonym_rel_excluded``, then drop any ``k`` where A and B share the same token id
    (otherwise layer-0 cos_sim is trivially 1 from the embedding lookup).

    Returns ``(eligible_k, same_id_ks, used_fallback)``. If no distinct-id ``k`` exists,
    ``eligible_k`` falls back to homonym-excluded-only and ``used_fallback`` is True.
    """
    base = [k for k in range(m_user) if k not in homonym_rel_excluded]
    if not base:
        base = list(range(m_user))
    distinct: list[int] = []
    same_id: list[int] = []
    for k in base:
        ia = int(ids_a[0, ua_lo + k].item())
        ib = int(ids_b[0, ub_lo + k].item())
        if ia == ib:
            same_id.append(k)
        else:
            distinct.append(k)
    if distinct:
        return distinct, same_id, False
    return base, same_id, True


def _control_homonym_excluded_relative_offsets(
    m_user: int,
    homonym_rel_a: int,
    homonym_rel_b: int,
    *,
    exclude_first_user_token: bool,
) -> set[int]:
    """
    Relative offsets ``k`` excluded when sampling the control: homonym slots, and optionally
    ``k=0`` (first user token). If excluding ``k=0`` leaves no ``k``, drop only that rule
    and keep homonym exclusions.
    """
    ex_h = {r for r in (homonym_rel_a, homonym_rel_b) if 0 <= r < m_user}
    if not exclude_first_user_token or m_user <= 0:
        return ex_h
    ex = set(ex_h) | {0}
    if not any(k not in ex for k in range(m_user)):
        return ex_h
    return ex


def _control_pair_legend_label(
    text_a: str, text_b: str, pair_i: int, *, max_side: int = 20
) -> str:
    """One-line legend for control plots: decoded token at control index A vs B."""

    def _one(s: str) -> str:
        if not s:
            return "∅"
        vis = s.replace("\n", "⏎").replace("\r", "")
        if len(vis) > max_side:
            return vis[: max_side - 1] + "…"
        return vis

    return f"{_one(text_a)} | {_one(text_b)} ({pair_i})"


def run_hidden_streams(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, int, torch.Tensor]:
    """
    One forward, batch size 1. Returns ``activations`` of shape
    ``(1 + num_stages, seq_len, hidden)`` (pre block 0 + after each decoder block),
    ``num_blocks``, and ``logits`` ``(1, seq_len, vocab)`` for next-token prediction.
    """
    blocks = _decoder_blocks(model)
    captured: list[torch.Tensor] = []

    def save_pre_layer0(_module, inputs: tuple) -> None:
        captured.append(inputs[0].detach())

    def save_layer_output(_module, _inp, out: torch.Tensor | tuple) -> None:
        hidden = out[0] if isinstance(out, tuple) else out
        captured.append(hidden.detach())

    hooks = [blocks[0].register_forward_pre_hook(save_pre_layer0)]
    hooks += [layer.register_forward_hook(save_layer_output) for layer in blocks]
    try:
        with torch.inference_mode():
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
    finally:
        for h in hooks:
            h.remove()

    logits = getattr(out, "logits", None)
    if logits is None:
        raise TypeError(
            f"Model forward returned no logits (got {type(out).__name__}). "
            "Causal LM outputs are required for next-token prediction."
        )

    stacked = torch.stack(captured, dim=0)
    if stacked.dim() == 4 and stacked.shape[1] == 1:
        stacked = stacked.squeeze(1)
    return stacked, len(blocks), logits


def print_next_token_at_position(
    logits: torch.Tensor,
    pos: int,
    *,
    side: str,
    top_k: int = 8,
) -> None:
    """Print top-``top_k`` next tokens from ``logits[0, pos]`` (prediction for token at ``pos + 1``)."""
    seq_len = logits.shape[1]
    if pos < 0 or pos >= seq_len:
        print(
            f"  next-token [{side}]: invalid pos={pos} (seq_len={seq_len})",
            flush=True,
        )
        return
    row = logits[0, pos].float()
    probs = torch.softmax(row, dim=-1)
    k = min(int(top_k), int(probs.shape[-1]))
    top_p, top_i = torch.topk(probs, k=k)
    print(
        f"  next-token at homonym index [{side}] (targets position {pos + 1}):",
        flush=True,
    )
    for rank, (p, tid) in enumerate(zip(top_p.tolist(), top_i.tolist()), start=1):
        tok = inner_tok.decode([tid], skip_special_tokens=False)
        disp = repr(tok) if (not tok or tok.isspace() or "\n" in tok) else tok
        print(f"    {rank:2d}. p={p:.5f}  id={tid:6d}  {disp}", flush=True)


# %% Homonym sentence pairs & per-layer cosine similarity at homonym position
# Each row: (sentence_a, sentence_b, sentence_a_prime, needle).
# Third sentence: **same sense of ``needle`` as in A**, but a **different situation** overall (not
# a full-sentence paraphrase of A). ``needle`` appears once in A, B, and the third sentence.
HOMONYM_PAIRS: list[tuple[str, str, str, str]] = [
    (
        "I deposit some money into the bank.",
        "I took a walk along the river and then sat on the bank.",
        "The branch manager at the downtown bank froze cards after skimmers hit two sidewalk ATMs.",
        "bank",
    ),
    (
        "He used a firm bass tone when he spoke to the crowd.",
        "For dinner we baked lemon bass with herbs.",
        "The subwoofer buzzed the windows when the DJ dropped a filthy bass line at two a.m.",
        "bass",
    ),
    (
        "Old apartment buildings may still have lead water pipes.",
        "A confident guide will lead the group across the ridge.",
        "Renovators chipped brittle lead paint off sills while fans whirred in the corridor.",
        "lead",
    ),
    (
        "Archery practice started with a simple wooden bow.",
        "The cast lined up to bow deeply when the curtain fell.",
        "An Olympic hopeful practiced drawing a recurve bow until her shoulders shook.",
        "bow",
    ),
    (
        "Camels cross the open desert for days without much water.",
        "A soldier must not desert a guard post without permission.",
        "Orbital radar mapped ancient channels under the Namib desert that dunes had half buried.",
        "desert",
    ),
    (
        "She chose a bright polish for her nails before the interview.",
        "It takes patience to polish silver until you see your face.",
        "The salon recalled a glitter polish after clients reported stinging cuticles.",
        "polish",
    ),
    (
        "After the audit parents argued printed guidelines were still not fair to tiny struggling rural districts.",
        "Fireworks shimmered on the river when the long dormant riverside fair finally reopened after many quiet summers.",
        "Season ticket holders swore the referee's overtime call was not fair after replay spread online.",
        "fair",
    ),
    (
        "Deep mud tracks implied a restless bear had circled the lodge dumpsters throughout the windy mountain night.",
        "Honestly I cannot bear listening to that smug podcast host for one more condescending minute tonight.",
        "Biologists whispered when a grizzly bear shouldered into the river among thrashing salmon.",
        "bear",
    ),
    (
        "Hikers purify icy water from a limestone spring seeping under roots at the bottom of the shaded ravine.",
        "Faulty stage traps may spring upward without rehearsal cues during the noisy melee closing the pirate battle scene.",
        "Ranchers fenced a trampled spring that cattle had churned into calf deep mud.",
        "spring",
    ),
    (
        "Field biologists vanish every rainy season into foggy upland forest that swallows trails without mercy.",
        "Line cooks learned to season enormous cast iron pans until satin black polymer built up after slow oven cycles.",
        "Fire chiefs hire extra crews before dry season turns chaparral into matchstick hills.",
        "season",
    ),
    (
        "A rogue curling wave smashed the dinghy broadside and rolled it toward jagged rocks beyond the pier lights.",
        "Sleepy kids began to wave tiny glow sticks in slow arcs when the parade fire truck loomed out of swirling fog.",
        "Locals argued which wave of the set would close out first when the tide drained the inside bar.",
        "wave",
    ),
    (
        "Headlights stabbed through sleet seconds before the eastbound train shuddered past empty silos beyond town limits.",
        "Ultramarathon hopefuls train through knee deep slush so spring tempos on race day feel almost insultingly smooth.",
        "Dispatchers halted a coal train miles short of a bridge inspectors had flagged as unsafe.",
        "train",
    ),
    (
        "The vintage lens breathed in and out hunting sharp focus while strobes flared behind swaying costumed bodies.",
        "Clinicians coached her to focus on slow diaphragm breath whenever dread climbed like heat into her tightening throat.",
        "A macro lens hunted focus again whenever a moth blundered through the hot ring light.",
        "focus",
    ),
    (
        "Rust flaked off the compass bezel until the needle found a steady north point above the torn topo quadrangle.",
        "Union stewards kept trying to point toward phantom hours missing from pay stubs during the bitter arbitration hearing.",
        "The briefing circled a foggy coastal point where patrol planes once caught smugglers landing at dawn.",
        "point",
    ),
    (
        "Volunteers ferried a rescued pitcher plant toward misty greenhouse doors past racks of muddy boots and cable reels.",
        "Prosecutors believed foreign agents managed to plant malware inside outdated routers weeks before the summit convened.",
        "Crews uprooted an invasive plant whose sap raised burns that lingered for days.",
        "plant",
    ),
    (
        "Tenants highlighted every vague rent clause in the renewal contract before confronting the smirking building manager upstairs.",
        "Inspectors fear century old iron trusses will contract dangerously during the polar cold front forecast for next weekend.",
        "Angel investors balked when counsel crossed liability caps in the seed round contract.",
        "contract",
    ),
    (
        "Delivery bots garbled the handwritten farm address scrawled on a receipt magneted beside a noisy rattling refrigerator.",
        "The governor will address frightened coastal towns only when FEMA clears road crews into the shattered highway corridor.",
        "Paramedics blamed a rain smeared address painted on the curb for losing critical seconds.",
        "address",
    ),
    (
        "Foggy coastal terraces still produce tart heirloom apples that stain the crates deep garnet at the winter market.",
        "Shoppers dig toward the back hunting unbruised produce buried under pyramids of glossy waxy fruit near closing time.",
        "Backlot plots by the warehouse district still produce chilies hot enough for single jar batches.",
        "produce",
    ),
    (
        "Stewards quietly refuse to approve concessions that strip paid breaks from crews unloading trucks through bitter nights.",
        "Opossums ripped into plastic bags of kitchen refuse stacked where overworked sanitation crews missed another alley pickup.",
        "The wetlands board may refuse permits if the survey omits vernal pool shading.",
        "refuse",
    ),
    (
        "Career counselors demanded a ruthlessly honest one page resume listing every fellowship without hollow trendy leadership verbs.",
        "The bedraggled pit band waited until ushers signaled they could resume the overture after the false fire alarm ended.",
        "Hiring partners scoffed at a four page resume in an industry that claims it reads only one.",
        "resume",
    ),
    (
        "We wedged sleeping pads and stoves into one absurdly compact car before crawling toward the clogged coastal highway.",
        "She eased the rattling makeup compact open beneath hot klieg lights and powdered cheeks before the live television hit.",
        "Evacuees jammed heirlooms into a compact go bag before the evacuation map turned orange.",
        "compact",
    ),
    (
        "Commentators dreaded a joyless defensive chess match stretching past curfew inside the half empty convention center hall.",
        "None of the metric bolts from the tin seemed to match the stripped threads on the hand pump buried in clay.",
        "The fifth set of that semifinal match ended after both players limped between serves.",
        "match",
    ),
    (
        "Silver birch bark sloughed onto our boots while we slid toward the half frozen marsh glowering under violet sunset.",
        "Terriers lose their voices when they bark nonstop at Amazon vans lurching up the steep cracked apartment driveway.",
        "Street vendors toasted cinnamon bark until the alley smelled like winter markets.",
        "bark",
    ),
    (
        "That chipped enamel mug on the sill was always mine according to stubborn cousins who never forgot Grandma's stories.",
        "Dangerous gases once flooded a coal mine before federal inspectors padlocked the slope and fined the neglectful operator.",
        "The radiator nook by the fire escape was mine long before roommates hashed out the chore wheel.",
        "mine",
    ),
    (
        "She wrapped the lumpy present in old comic panels after every glittery downtown shop had locked tight for the blizzard.",
        "Almost no city council allies were present when the mayor initialed the confidential waterfront easement inside a side chamber.",
        "The toddler shredded the present ribbon before anyone finished passing the salad.",
        "present",
    ),
    (
        "Childhood neighbors drifted until nobody on the thread felt emotionally close after years of careful polite distance.",
        "Ushers hissed at stragglers to close the carved mahogany doors before the quartet began the fragile opening adagio.",
        "The siblings only felt close again once their father entered hospice care across town.",
        "close",
    ),
    (
        "Controllers granted exactly one more minute before automated safeties would trip the experimental fusion containment vent sequence.",
        "Icon restorers laid down minute strokes of gold leaf along cracked halos until saints seemed gently luminous under spotlights.",
        "Tower chatter stopped for one endless minute after a drone crossed the glide slope at rush hour.",
        "minute",
    ),
    (
        "Historians sparred over whether the corroded bronze object fit better with maritime tools or forgotten ritual censers.",
        "Elder fishermen stood to object passionately when engineers outlined a jetty that would erase tidal pools nursery grounds.",
        "Interns cocooned the cracked object in acid free tissue before its crate rolled toward customs.",
        "object",
    ),
    (
        "Quantum thermodynamics stayed the most dreaded subject printed in giant type on the battered department recruiting poster.",
        "Imperial armies routinely subject conquered ports to suffocating tariffs whenever rebel governors refused tribute after naval blockades.",
        "Transfers whispered that one subject handed a midterm whose curve still haunts alumni forums.",
        "subject",
    ),
]

HOMONYM_COS_BY_PAIR: list[list[float]] = []
# Control: same offset ``k`` into the **user-sentence** token span; skip same token id across
# A/B; optionally skip k=0 (``CONTROL_EXCLUDE_FIRST_USER_TOKEN``).
HOMONYM_COS_CONTROL_BY_PAIR: list[list[float]] = []
HOMONYM_PAIR_LABELS: list[str] = []
HOMONYM_CONTROL_PAIR_LABELS: list[str] = []
# A vs third sentence at ``needle`` (same sense as A, different overall scenario).
HOMONYM_PARAPHRASE_COS_BY_PAIR: list[list[float]] = []
HOMONYM_PARAPHRASE_PAIR_LABELS: list[str] = []

for pair_i, (sent_a, sent_b, sent_a_prime, needle) in enumerate(HOMONYM_PAIRS):
    stonesoup.check_abort()
    ids_a, mask_a, prompt_a, add_spec_a = _chat_prompt_ids_and_mask(sent_a)
    ids_b, mask_b, prompt_b, add_spec_b = _chat_prompt_ids_and_mask(sent_b)
    ids_p, mask_p, prompt_p, add_spec_p = _chat_prompt_ids_and_mask(sent_a_prime)
    if not (add_spec_a == add_spec_b == add_spec_p):
        raise ValueError("Mixed tokenization modes in one pair (unexpected).")
    idx_a = _homonym_token_index_last_subword(
        prompt_a, needle, add_special_tokens=add_spec_a
    )
    idx_b = _homonym_token_index_last_subword(
        prompt_b, needle, add_special_tokens=add_spec_b
    )
    idx_p = _homonym_token_index_last_subword(
        prompt_p, needle, add_special_tokens=add_spec_p
    )

    acts_a, n_blocks, logits_a = run_hidden_streams(ids_a, mask_a)
    acts_b, n_blocks_b, logits_b = run_hidden_streams(ids_b, mask_b)
    acts_p, n_blocks_p, _logp = run_hidden_streams(ids_p, mask_p)
    assert n_blocks_b == n_blocks == n_blocks_p
    num_stages = acts_a.shape[0]
    cos_per_layer: list[float] = []
    cos_paraphrase: list[float] = []
    for li in range(num_stages):
        ha = acts_a[li, idx_a].float()
        hb = acts_b[li, idx_b].float()
        hp = acts_p[li, idx_p].float()
        cos = F.cosine_similarity(ha.unsqueeze(0), hb.unsqueeze(0), dim=-1).item()
        cos_per_layer.append(cos)
        cos_paraphrase.append(
            F.cosine_similarity(ha.unsqueeze(0), hp.unsqueeze(0), dim=-1).item()
        )

    seq_a = int(acts_a.shape[1])
    seq_b = int(acts_b.shape[1])
    ua_lo, ua_hi = _user_content_token_lo_hi(
        prompt_a, sent_a, add_special_tokens=add_spec_a
    )
    ub_lo, ub_hi = _user_content_token_lo_hi(
        prompt_b, sent_b, add_special_tokens=add_spec_b
    )
    len_ua = ua_hi - ua_lo + 1
    len_ub = ub_hi - ub_lo + 1
    m_user = min(len_ua, len_ub)
    rel_ha = idx_a - ua_lo
    rel_hb = idx_b - ub_lo
    exclude_rel = _control_homonym_excluded_relative_offsets(
        m_user,
        rel_ha,
        rel_hb,
        exclude_first_user_token=CONTROL_EXCLUDE_FIRST_USER_TOKEN,
    )
    eligible_k, same_id_ks, ctrl_distinct_fallback = _eligible_control_k_distinct_ids(
        ids_a,
        ids_b,
        ua_lo=ua_lo,
        ub_lo=ub_lo,
        m_user=m_user,
        homonym_rel_excluded=exclude_rel,
    )
    rng_ctrl = random.Random(42 + pair_i)
    k_ctrl = rng_ctrl.choice(eligible_k)
    r_ctrl_a = ua_lo + k_ctrl
    r_ctrl_b = ub_lo + k_ctrl
    cos_control: list[float] = []
    for li in range(num_stages):
        ha_c = acts_a[li, r_ctrl_a].float()
        hb_c = acts_b[li, r_ctrl_b].float()
        cos_c = F.cosine_similarity(ha_c.unsqueeze(0), hb_c.unsqueeze(0), dim=-1).item()
        cos_control.append(cos_c)

    HOMONYM_COS_BY_PAIR.append(cos_per_layer)
    HOMONYM_COS_CONTROL_BY_PAIR.append(cos_control)
    HOMONYM_PAIR_LABELS.append(f"{needle} ({pair_i})")
    HOMONYM_PARAPHRASE_COS_BY_PAIR.append(cos_paraphrase)
    HOMONYM_PARAPHRASE_PAIR_LABELS.append(f"{needle} A|A' ({pair_i})")

    print(f"\n--- Pair {pair_i}: needle={needle!r} ---", flush=True)
    print(f"  A: {sent_a}", flush=True)
    print(f"  B: {sent_b}", flush=True)
    print(f"  token index (A, B): {idx_a}, {idx_b}", flush=True)
    cid_a = int(ids_a[0, r_ctrl_a].item())
    cid_b = int(ids_b[0, r_ctrl_b].item())
    c_tok_a = inner_tok.convert_ids_to_tokens([cid_a])[0]
    c_tok_b = inner_tok.convert_ids_to_tokens([cid_b])[0]
    _to_str = getattr(inner_tok, "convert_tokens_to_string", None)
    if _to_str is not None:
        c_str_a = _to_str([c_tok_a])
        c_str_b = _to_str([c_tok_b])
    else:
        c_str_a = inner_tok.decode(
            [cid_a], skip_special_tokens=False
        )
        c_str_b = inner_tok.decode(
            [cid_b], skip_special_tokens=False
        )
    HOMONYM_CONTROL_PAIR_LABELS.append(
        _control_pair_legend_label(c_str_a, c_str_b, pair_i)
    )
    same_ctrl_id = cid_a == cid_b
    ctx_radius = 3
    lo_a = max(0, r_ctrl_a - ctx_radius)
    hi_a = min(seq_a, r_ctrl_a + ctx_radius + 1)
    lo_b = max(0, r_ctrl_b - ctx_radius)
    hi_b = min(seq_b, r_ctrl_b + ctx_radius + 1)
    ctx_decode_a = inner_tok.decode(
        ids_a[0, lo_a:hi_a].tolist(), skip_special_tokens=False
    )
    ctx_decode_b = inner_tok.decode(
        ids_b[0, lo_b:hi_b].tolist(), skip_special_tokens=False
    )
    _parts = [
        f"user-text offset k={k_ctrl}",
        f"m_user={m_user}",
        f"user tok A [{ua_lo},{ua_hi}], B [{ub_lo},{ub_hi}]",
        f"global idx A={r_ctrl_a}, B={r_ctrl_b}",
        f"excluded homonym rel {sorted(exclude_rel)}",
    ]
    if same_id_ks:
        _parts.append(f"skipped same-token-id k {sorted(same_id_ks)}")
    if ctrl_distinct_fallback:
        _parts.append("FALLBACK no distinct-id k left (homonym-excluded only)")
    print(f"  control: {'; '.join(_parts)}", flush=True)
    print(
        f"    A @ r text: {c_str_a!r}  (id={cid_a}, piece={c_tok_a!r})",
        flush=True,
    )
    print(
        f"    B @ r text: {c_str_b!r}  (id={cid_b}, piece={c_tok_b!r})",
        flush=True,
    )
    print(
        f"    context A ids[{lo_a}:{hi_a}] decode: {ctx_decode_a!r}",
        flush=True,
    )
    print(
        f"    context B ids[{lo_b}:{hi_b}] decode: {ctx_decode_b!r}",
        flush=True,
    )
    print(
        f"    same token id at r={same_ctrl_id} "
        f"(if True, early cos_sim≈1 is common: identical embedding lookup for that id)",
        flush=True,
    )
    print_next_token_at_position(logits_a, idx_a, side="A")
    print_next_token_at_position(logits_b, idx_b, side="B")
    for li, c in enumerate(cos_per_layer):
        label = "pre_block_0" if li == 0 else f"post_block_{li - 1}"
        print(f"  layer {li:2d} ({label}): cos_sim = {c:.6f}", flush=True)
    print(f"  (decoder blocks: {n_blocks})", flush=True)


# %% Plot cosine similarity vs layer (one line per pair)
import matplotlib.pyplot as plt

MODEL_BASENAME = MODEL_ID.replace("/", "__").replace(":", "-")

if not HOMONYM_COS_BY_PAIR:
    print("Run the homonym pairs cell first.", flush=True)
else:
    n_st = len(HOMONYM_COS_BY_PAIR[0])
    x_layers = list(range(n_st))
    n_pairs = len(HOMONYM_COS_BY_PAIR)
    fig_w = min(13.0, 9.0 + 0.04 * n_pairs)
    ncol = max(5, min(8, (n_pairs + 4) // 5))
    n_legend_rows = (n_pairs + ncol - 1) // ncol
    fig_h = 5.4 + 0.38 * n_legend_rows
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    for ys, label in zip(HOMONYM_COS_BY_PAIR, HOMONYM_PAIR_LABELS, strict=True):
        if len(ys) != n_st:
            raise ValueError(
                f"Inconsistent stage count: expected {n_st}, got {len(ys)} for {label}"
            )
        ax.plot(x_layers, ys, marker="o", markersize=4, linewidth=1.5, label=label)
    ax.set_xlabel("layer stage (0 = pre block 0, k = post block k-1)")
    ax.set_ylabel("cosine similarity")
    ax.set_title(
        f"{MODEL_ID}\nHomonym token hidden states: cos sim vs depth"
    )
    ax.grid(True, alpha=0.3)
    ax.set_xticks(x_layers)
    ax.set_ylim(0, 1.05)
    handles, labels = ax.get_legend_handles_labels()
    # Reserve ``bottom_margin`` for a multi-column legend under the axes (figure y is bottom=0).
    bottom_margin = min(0.55, 0.26 + 0.042 * n_legend_rows)
    legend_anchor_y = bottom_margin - 0.015
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, legend_anchor_y),
        ncol=ncol,
        fontsize=8,
        frameon=True,
        fancybox=False,
        edgecolor="0.75",
        columnspacing=0.9,
        handletextpad=0.5,
    )
    fig.subplots_adjust(top=0.92, bottom=bottom_margin, left=0.09, right=0.98)
    stonesoup.show(fig, basename=f"{MODEL_BASENAME}_homonym_cos_vs_layer")


# %% Plot A vs A' at ``needle`` (same keyword sense, different sentence situation)
import matplotlib.pyplot as plt

MODEL_BASENAME = MODEL_ID.replace("/", "__").replace(":", "-")

if not HOMONYM_PARAPHRASE_COS_BY_PAIR:
    print("Run the homonym pairs cell first.", flush=True)
elif len(HOMONYM_PARAPHRASE_COS_BY_PAIR) != len(HOMONYM_PARAPHRASE_PAIR_LABELS):
    raise ValueError("Paraphrase cos list length mismatch")
else:
    n_st_p = len(HOMONYM_PARAPHRASE_COS_BY_PAIR[0])
    x_p = list(range(n_st_p))
    n_pp = len(HOMONYM_PARAPHRASE_COS_BY_PAIR)
    fig_wp = min(13.0, 9.0 + 0.04 * n_pp)
    ncol_p = max(5, min(8, (n_pp + 4) // 5))
    n_lr_p = (n_pp + ncol_p - 1) // ncol_p
    fig_hp = 5.4 + 0.38 * n_lr_p
    fig_p, ax_p = plt.subplots(figsize=(fig_wp, fig_hp))
    for ys, lab in zip(
        HOMONYM_PARAPHRASE_COS_BY_PAIR, HOMONYM_PARAPHRASE_PAIR_LABELS, strict=True
    ):
        if len(ys) != n_st_p:
            raise ValueError(f"Expected {n_st_p} stages, got {len(ys)} for {lab}")
        ax_p.plot(x_p, ys, marker="o", markersize=4, linewidth=1.5, label=lab)
    ax_p.set_xlabel("layer stage (0 = pre block 0, k = post block k-1)")
    ax_p.set_ylabel("cosine similarity")
    ax_p.set_title(
        f"{MODEL_ID}\nSame keyword sense, different scenario: cos(needle in A, needle in A') vs depth"
    )
    ax_p.grid(True, alpha=0.3)
    ax_p.set_xticks(x_p)
    ax_p.set_ylim(0, 1.05)
    h_p, lab_p = ax_p.get_legend_handles_labels()
    bm_p = min(0.55, 0.26 + 0.042 * n_lr_p)
    fig_p.legend(
        h_p,
        lab_p,
        loc="upper center",
        bbox_to_anchor=(0.5, bm_p - 0.015),
        ncol=ncol_p,
        fontsize=8,
        frameon=True,
        fancybox=False,
        edgecolor="0.75",
        columnspacing=0.9,
        handletextpad=0.5,
    )
    fig_p.subplots_adjust(top=0.92, bottom=bm_p, left=0.09, right=0.98)
    stonesoup.show(fig_p, basename=f"{MODEL_BASENAME}_homonym_paraphrase_cos_vs_layer")


# %% Violin plot: A vs A' (same keyword sense, different scenario) per layer
import matplotlib.pyplot as plt
import numpy as np

MODEL_BASENAME = MODEL_ID.replace("/", "__").replace(":", "-")

if not HOMONYM_PARAPHRASE_COS_BY_PAIR:
    print("Run the homonym pairs cell first.", flush=True)
elif len(HOMONYM_PARAPHRASE_COS_BY_PAIR) != len(HOMONYM_COS_BY_PAIR):
    print("Re-run homonym pairs cell so paraphrase rows match homonym rows.", flush=True)
else:
    n_st_pv = len(HOMONYM_PARAPHRASE_COS_BY_PAIR[0])
    mat_pv = np.asarray(HOMONYM_PARAPHRASE_COS_BY_PAIR, dtype=np.float64)
    if mat_pv.ndim != 2 or mat_pv.shape[1] != n_st_pv:
        raise ValueError(
            f"Expected HOMONYM_PARAPHRASE_COS_BY_PAIR as (n_pairs, {n_st_pv}), got {mat_pv.shape}"
        )
    x_pv = np.arange(n_st_pv)
    dataset_pv = [mat_pv[:, li] for li in range(n_st_pv)]
    fig_pv, ax_pv = plt.subplots(figsize=(11, 5.2))
    parts_pv = ax_pv.violinplot(
        dataset_pv,
        positions=x_pv,
        widths=0.82,
        showmeans=True,
        showmedians=True,
        showextrema=True,
    )
    for pc in parts_pv["bodies"]:
        pc.set_facecolor("seagreen")
        pc.set_alpha(0.45)
        pc.set_edgecolor("0.35")
        pc.set_linewidth(0.8)
    for key in ("cmeans", "cmedians", "cbars", "cmins", "cmaxes"):
        if key in parts_pv and parts_pv[key] is not None:
            parts_pv[key].set_color("0.25")
            parts_pv[key].set_linewidth(1.0)
    if "cmeans" in parts_pv and parts_pv["cmeans"] is not None:
        parts_pv["cmeans"].set_linestyle("--")
    ax_pv.set_xlabel("layer stage (0 = pre block 0, k = post block k-1)")
    ax_pv.set_ylabel("cosine similarity")
    ax_pv.set_title(
        f"{MODEL_ID}\nA vs A': same keyword sense, different scenario — cos sim per layer"
    )
    ax_pv.set_xticks(x_pv)
    ax_pv.set_ylim(0, 1.05)
    ax_pv.grid(True, alpha=0.3, axis="y")
    fig_pv.tight_layout()
    stonesoup.show(fig_pv, basename=f"{MODEL_BASENAME}_homonym_paraphrase_cos_vs_layer_violin")


# %% Plot control cosine similarity vs layer (one line per pair)
import matplotlib.pyplot as plt

MODEL_BASENAME = MODEL_ID.replace("/", "__").replace(":", "-")

if not HOMONYM_COS_CONTROL_BY_PAIR:
    print("Run the homonym pairs cell first (control trajectories are filled there).", flush=True)
else:
    n_st_c = len(HOMONYM_COS_CONTROL_BY_PAIR[0])
    x_layers_c = list(range(n_st_c))
    n_pairs_c = len(HOMONYM_COS_CONTROL_BY_PAIR)
    if n_pairs_c != len(HOMONYM_CONTROL_PAIR_LABELS):
        raise ValueError(
            f"Control rows ({n_pairs_c}) != HOMONYM_CONTROL_PAIR_LABELS "
            f"({len(HOMONYM_CONTROL_PAIR_LABELS)}); re-run homonym pairs cell"
        )
    fig_w_c = min(13.0, 9.0 + 0.04 * n_pairs_c)
    ncol_c = max(5, min(8, (n_pairs_c + 4) // 5))
    n_legend_rows_c = (n_pairs_c + ncol_c - 1) // ncol_c
    fig_h_c = 5.4 + 0.38 * n_legend_rows_c
    fig_c, ax_c = plt.subplots(figsize=(fig_w_c, fig_h_c))
    for ys, label in zip(
        HOMONYM_COS_CONTROL_BY_PAIR, HOMONYM_CONTROL_PAIR_LABELS, strict=True
    ):
        if len(ys) != n_st_c:
            raise ValueError(
                f"Inconsistent stage count: expected {n_st_c}, got {len(ys)} for {label}"
            )
        ax_c.plot(
            x_layers_c, ys, marker="o", markersize=4, linewidth=1.5, label=label
        )
    ax_c.set_xlabel("layer stage (0 = pre block 0, k = post block k-1)")
    ax_c.set_ylabel("cosine similarity")
    ax_c.set_title(
        f"{MODEL_ID}\nControl hidden states (aligned user offset, distinct ids): cos sim vs depth"
    )
    ax_c.grid(True, alpha=0.3)
    ax_c.set_xticks(x_layers_c)
    ax_c.set_ylim(0, 1.05)
    handles_c, labels_c = ax_c.get_legend_handles_labels()
    bottom_margin_c = min(0.55, 0.26 + 0.042 * n_legend_rows_c)
    legend_anchor_y_c = bottom_margin_c - 0.015
    fig_c.legend(
        handles_c,
        labels_c,
        loc="upper center",
        bbox_to_anchor=(0.5, legend_anchor_y_c),
        ncol=ncol_c,
        fontsize=8,
        frameon=True,
        fancybox=False,
        edgecolor="0.75",
        columnspacing=0.9,
        handletextpad=0.5,
    )
    fig_c.subplots_adjust(top=0.92, bottom=bottom_margin_c, left=0.09, right=0.98)
    stonesoup.show(fig_c, basename=f"{MODEL_BASENAME}_homonym_control_cos_vs_layer")


# %% Violin plot: cos sim across homonym pairs at each layer
import matplotlib.pyplot as plt
import numpy as np

MODEL_BASENAME = MODEL_ID.replace("/", "__").replace(":", "-")

if not HOMONYM_COS_BY_PAIR:
    print("Run the homonym pairs cell first.", flush=True)
else:
    n_st = len(HOMONYM_COS_BY_PAIR[0])
    mat = np.asarray(HOMONYM_COS_BY_PAIR, dtype=np.float64)
    if mat.ndim != 2 or mat.shape[1] != n_st:
        raise ValueError(
            f"Expected HOMONYM_COS_BY_PAIR as (n_pairs, {n_st}), got {mat.shape}"
        )
    x_layers = np.arange(n_st)
    dataset = [mat[:, li] for li in range(n_st)]
    fig_v, ax_v = plt.subplots(figsize=(11, 5.2))
    parts = ax_v.violinplot(
        dataset,
        positions=x_layers,
        widths=0.82,
        showmeans=True,
        showmedians=True,
        showextrema=True,
    )
    for pc in parts["bodies"]:
        pc.set_facecolor("steelblue")
        pc.set_alpha(0.5)
        pc.set_edgecolor("0.35")
        pc.set_linewidth(0.8)
    for key in ("cmeans", "cmedians", "cbars", "cmins", "cmaxes"):
        if key in parts and parts[key] is not None:
            parts[key].set_color("0.25")
            parts[key].set_linewidth(1.0)
    if "cmeans" in parts and parts["cmeans"] is not None:
        parts["cmeans"].set_linestyle("--")
    ax_v.set_xlabel("layer stage (0 = pre block 0, k = post block k-1)")
    ax_v.set_ylabel("cosine similarity")
    ax_v.set_title(
        f"{MODEL_ID}\nHomonym pairs: cos sim distribution per layer (one violin per stage)"
    )
    ax_v.set_xticks(x_layers)
    ax_v.set_ylim(0, 1.05)
    ax_v.grid(True, alpha=0.3, axis="y")
    fig_v.tight_layout()
    stonesoup.show(fig_v, basename=f"{MODEL_BASENAME}_homonym_cos_vs_layer_violin")


# %% Violin plot (control): random offset inside **user** token span on each side
import matplotlib.pyplot as plt
import numpy as np

MODEL_BASENAME = MODEL_ID.replace("/", "__").replace(":", "-")

if not HOMONYM_COS_CONTROL_BY_PAIR:
    print("Run the homonym pairs cell first.", flush=True)
elif len(HOMONYM_COS_CONTROL_BY_PAIR) != len(HOMONYM_COS_BY_PAIR):
    print(
        "Control matrix length mismatch; re-run the homonym pairs cell.",
        flush=True,
    )
else:
    n_st = len(HOMONYM_COS_CONTROL_BY_PAIR[0])
    mat_c = np.asarray(HOMONYM_COS_CONTROL_BY_PAIR, dtype=np.float64)
    if mat_c.ndim != 2 or mat_c.shape[1] != n_st:
        raise ValueError(
            f"Expected HOMONYM_COS_CONTROL_BY_PAIR as (n_pairs, {n_st}), got {mat_c.shape}"
        )
    x_layers = np.arange(n_st)
    dataset_c = [mat_c[:, li] for li in range(n_st)]
    fig_vc, ax_vc = plt.subplots(figsize=(11, 5.2))
    parts_c = ax_vc.violinplot(
        dataset_c,
        positions=x_layers,
        widths=0.82,
        showmeans=True,
        showmedians=True,
        showextrema=True,
    )
    for pc in parts_c["bodies"]:
        pc.set_facecolor("darkorange")
        pc.set_alpha(0.45)
        pc.set_edgecolor("0.35")
        pc.set_linewidth(0.8)
    for key in ("cmeans", "cmedians", "cbars", "cmins", "cmaxes"):
        if key in parts_c and parts_c[key] is not None:
            parts_c[key].set_color("0.25")
            parts_c[key].set_linewidth(1.0)
    if "cmeans" in parts_c and parts_c["cmeans"] is not None:
        parts_c["cmeans"].set_linestyle("--")
    ax_vc.set_xlabel("layer stage (0 = pre block 0, k = post block k-1)")
    ax_vc.set_ylabel("cosine similarity")
    ax_vc.set_title(
        f"{MODEL_ID}\nControl: cos sim at same user-sentence offset, distinct token ids (not homonym)"
    )
    ax_vc.set_xticks(x_layers)
    ax_vc.set_ylim(0, 1.05)
    ax_vc.grid(True, alpha=0.3, axis="y")
    fig_vc.tight_layout()
    stonesoup.show(fig_vc, basename=f"{MODEL_BASENAME}_homonym_cos_control_random_pos_violin")


# %% Violin plot: homonym + control + A|A' on one axes (grouped per layer)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

MODEL_BASENAME = MODEL_ID.replace("/", "__").replace(":", "-")

def _style_violin_parts(
    parts: dict,
    *,
    face: str,
    alpha: float,
    edge: str = "0.35",
    line_alpha: float = 1.0,
) -> None:
    for pc in parts["bodies"]:
        pc.set_facecolor(face)
        pc.set_alpha(alpha)
        pc.set_edgecolor(edge)
        pc.set_linewidth(0.75)
    for key in ("cmeans", "cmedians", "cbars", "cmins", "cmaxes"):
        if key in parts and parts[key] is not None:
            parts[key].set_color("0.2")
            parts[key].set_alpha(line_alpha)
            parts[key].set_linewidth(0.9)
    if "cmeans" in parts and parts["cmeans"] is not None:
        parts["cmeans"].set_linestyle("--")


if (
    not HOMONYM_COS_BY_PAIR
    or not HOMONYM_COS_CONTROL_BY_PAIR
    or not HOMONYM_PARAPHRASE_COS_BY_PAIR
):
    print("Run the homonym pairs cell first (all three cos matrices required).", flush=True)
elif not (
    len(HOMONYM_COS_BY_PAIR)
    == len(HOMONYM_COS_CONTROL_BY_PAIR)
    == len(HOMONYM_PARAPHRASE_COS_BY_PAIR)
):
    print("Row count mismatch; re-run the homonym pairs cell.", flush=True)
else:
    n_st3 = len(HOMONYM_COS_BY_PAIR[0])
    mat_h = np.asarray(HOMONYM_COS_BY_PAIR, dtype=np.float64)
    mat_o = np.asarray(HOMONYM_COS_CONTROL_BY_PAIR, dtype=np.float64)
    mat_p = np.asarray(HOMONYM_PARAPHRASE_COS_BY_PAIR, dtype=np.float64)
    if mat_h.shape != mat_o.shape or mat_h.shape != mat_p.shape:
        raise ValueError(
            f"Shape mismatch homonym {mat_h.shape} control {mat_o.shape} A' {mat_p.shape}"
        )
    x_ctr = np.arange(n_st3, dtype=np.float64)
    # Wider violins than default grouping; ``off >= wv`` keeps the three from merging into one lump.
    wv = 0.34
    off = 0.36
    ds_h = [mat_h[:, li] for li in range(n_st3)]
    ds_o = [mat_o[:, li] for li in range(n_st3)]
    ds_p = [mat_p[:, li] for li in range(n_st3)]
    fig_3v, ax_3v = plt.subplots(figsize=(14.0, 5.5))
    ph = ax_3v.violinplot(
        ds_h,
        positions=x_ctr - off,
        widths=wv,
        showmeans=True,
        showmedians=True,
        showextrema=True,
    )
    po = ax_3v.violinplot(
        ds_o,
        positions=x_ctr,
        widths=wv,
        showmeans=True,
        showmedians=True,
        showextrema=True,
    )
    pp = ax_3v.violinplot(
        ds_p,
        positions=x_ctr + off,
        widths=wv,
        showmeans=True,
        showmedians=True,
        showextrema=True,
    )
    _vio_alpha = 0.58
    _vio_line_alpha = 0.28
    _style_violin_parts(
        ph, face="steelblue", alpha=_vio_alpha, line_alpha=_vio_line_alpha
    )
    _style_violin_parts(
        po, face="darkorange", alpha=_vio_alpha, line_alpha=_vio_line_alpha
    )
    _style_violin_parts(
        pp, face="seagreen", alpha=_vio_alpha, line_alpha=_vio_line_alpha
    )
    ax_3v.set_xlabel("layer stage (0 = pre block 0, k = post block k-1)")
    ax_3v.set_ylabel("cosine similarity")
    ax_3v.set_title(
        f"{MODEL_ID}\nGrouped violins: homonym (blue) · control offset (orange) · A vs A' (green)"
    )
    ax_3v.set_xticks(x_ctr)
    ax_3v.set_xticklabels([str(int(i)) for i in x_ctr])
    ax_3v.set_xlim(-0.85, n_st3 - 1 + 0.85)
    ax_3v.set_ylim(0, 1.05)
    ax_3v.grid(True, alpha=0.28, axis="y")
    ax_3v.legend(
        handles=[
            Patch(
                facecolor="seagreen",
                alpha=0.58,
                edgecolor="0.35",
                label="Control (same token, same meaning)",
            ),
            Patch(facecolor="steelblue", alpha=0.58, edgecolor="0.35", label="Homonym (same token, different meaning)"),
            Patch(
                facecolor="darkorange",
                alpha=0.58,
                edgecolor="0.35",
                label="Control (different tokens)",
            ),

        ],
        loc="lower right",
        framealpha=0.92,
        fontsize=9,
    )
    fig_3v.tight_layout()
    stonesoup.show(fig_3v, basename=f"{MODEL_BASENAME}_homonym_cos_three_violins_overlay")


# %% Sanity: one control pair re-forward — cos=1 vs identical vs collinear
# ``cos_sim = 1`` means **same direction** (unit vectors match): collinear, not necessarily ``a == b``.
# ``||nrm_a - nrm_b||`` is L2 distance between ``a/||a||`` and ``b/||b||``; ~0 ⇒ parallel same way.
# If ``k=0`` on both sides you are comparing **first user tokens** (``He`` vs ``For``): identical
# causal/positional role, so mid-layer directions can nearly align; ``allclose`` still False.
SANITY_CTRL_PAIR_I = 1  # bass → control legend like ``He | For (1)``
_s0, _s1, _, _need = HOMONYM_PAIRS[SANITY_CTRL_PAIR_I]
_iA, _mA, _pA, _adA = _chat_prompt_ids_and_mask(_s0)
_iB, _mB, _pB, _adB = _chat_prompt_ids_and_mask(_s1)
_ixA = _homonym_token_index_last_subword(_pA, _need, add_special_tokens=_adA)
_ixB = _homonym_token_index_last_subword(_pB, _need, add_special_tokens=_adB)
_u0A, _u1A = _user_content_token_lo_hi(_pA, _s0, add_special_tokens=_adA)
_u0B, _u1B = _user_content_token_lo_hi(_pB, _s1, add_special_tokens=_adB)
_mu = min(_u1A - _u0A + 1, _u1B - _u0B + 1)
_ex = _control_homonym_excluded_relative_offsets(
    _mu,
    _ixA - _u0A,
    _ixB - _u0B,
    exclude_first_user_token=CONTROL_EXCLUDE_FIRST_USER_TOKEN,
)
_ek, _, _ = _eligible_control_k_distinct_ids(
    _iA, _iB, ua_lo=_u0A, ub_lo=_u0B, m_user=_mu, homonym_rel_excluded=_ex
)
_k = random.Random(42 + SANITY_CTRL_PAIR_I).choice(_ek)
_rA, _rB = _u0A + _k, _u0B + _k
_tidA = int(_iA[0, _rA].item())
_tidB = int(_iB[0, _rB].item())
print(
    f"pair {SANITY_CTRL_PAIR_I} control k={_k} pos=({_rA},{_rB}) "
    f"ids=({_tidA},{_tidB}) "
    f"{inner_tok.decode([_tidA], skip_special_tokens=False)!r} vs "
    f"{inner_tok.decode([_tidB], skip_special_tokens=False)!r}",
    flush=True,
)
_actA, _, _ = run_hidden_streams(_iA, _mA)
_actB, _, _ = run_hidden_streams(_iB, _mB)
for _li in range(int(_actA.shape[0])):
    _a = _actA[_li, _rA].float()
    _b = _actB[_li, _rB].float()
    _cos = float(F.cosine_similarity(_a.unsqueeze(0), _b.unsqueeze(0), dim=-1))
    _na = float(_a.norm())
    _nb = float(_b.norm())
    _uh = (_a / _na - _b / _nb).norm().item() if _na and _nb else float("nan")
    _eq = bool(torch.allclose(_a, _b, rtol=0.0, atol=1e-4))
    print(
        f"L{_li:2d}  cos={_cos:.6f}  ||a||={_na:.2f}  ||b||={_nb:.2f}  "
        f"||nrm_a-nrm_b||={_uh:.4e}  allclose(a,b)={_eq}",
        flush=True,
    )
