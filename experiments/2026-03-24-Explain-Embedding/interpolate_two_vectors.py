# %% Imports & setup
"""Interpolate and extrapolate along the SLERP great circle between two embedding rows; extrapolation depth follows angle(A,B) to span almost a full 360 deg in ``t`` (same ``dt`` as ``[0,1]``). Top-3 vocab tokens by cosine similarity per step (A and B included).

Cache: ``data/embedding-layers/<hf_id_slashes_to__>.pt`` with a ``weight`` tensor ``(V, d)``.

**Run in Stonesoup:** **Imports** → **Load cache** → **Token A** / **B** (optional) → **Resolve** → optional **Midpoint** / **SLERP** / **Record+hist** / **unit-circle top-3 plot**.

**CLI:** ``uv run python …`` runs all blocks in file order.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
THIS_DIR = Path(__file__).resolve().parent
CACHE_DIR = REPO_ROOT / "data" / "embedding-layers"

MODEL_NAME = "Qwen/Qwen3.5-2B"
TOKEN_A = "0"
TOKEN_B = "9"
N_STEPS = 5  # t in [0, 1]: endpoints and interior, step dt = 1/(N_STEPS-1)


def extrap_steps_each_side_for_full_turn(
    dt: float,
    omega_rad: float,
    *,
    angular_slack: float = 1e-4,
) -> int:
    """Symmetric extrapolation count ``n``: use ``t`` in ``[-n*dt, 1+n*dt]`` (plus ``[0,1]`` grid).

    Target: total parameter span ``(1 + 2*n*dt) * omega_rad ~ 2*pi * (1 - angular_slack)``
    (almost one full rotation along the A–B great circle, in the same ``dt`` steps as interpolation).

    Subtract one step per side after the ceil so the two extrapolation arms do not quite meet
    (avoids piled-up, near-duplicate ticks at the wrap).
    """
    if omega_rad <= 1e-12 or dt <= 0:
        return 0
    span_t = (2.0 * math.pi / omega_rad) * (1.0 - angular_slack)
    need = span_t - 1.0
    if need <= 0:
        return 0
    n_raw = max(0, int(math.ceil(need / (2.0 * dt) - 1e-12)))
    return max(0, n_raw - 1)


def cache_path(model_name: str) -> Path:
    return CACHE_DIR / f"{model_name.replace('/', '__')}.pt"


def slerp_unit(a: torch.Tensor, b: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """``a``, ``b``: (d,) unit. ``t``: (n,) any real — great circle through A,B. Returns (n, d) unit rows."""
    dot = (a * b).sum().clamp(-1.0, 1.0)
    omega = torch.acos(dot)
    sin_o = torch.sin(omega)
    if sin_o.abs() < 1e-6:
        m = (1.0 - t).unsqueeze(-1) * a + t.unsqueeze(-1) * b
        return m / m.norm(dim=-1, keepdim=True)
    s0 = torch.sin((1.0 - t) * omega) / sin_o
    s1 = torch.sin(t * omega) / sin_o
    return s0.unsqueeze(-1) * a + s1.unsqueeze(-1) * b


def single_token_id(text: str, tokenizer) -> int:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"{text!r} encodes to {len(ids)} tokens {ids}; pick strings that are one token.")
    return ids[0]


def wrap_deg_signed180(deg: float) -> float:
    """Map degrees to (-180, 180]; same physical heading mod one full turn."""
    return (deg + 180.0) % 360.0 - 180.0


def plot_safe_token(s: str, *, max_len: int = 14) -> str:
    """Text safe for default matplotlib fonts: ASCII substitutes, drop CJK, strip control chars.

    Valid ASCII like ``\\x14`` (glyph 20) still encodes as ASCII but is non-printable and triggers
    DejaVu warnings; ``str.isprintable()`` removes those.
    """
    raw = s.encode("ascii", "replace").decode("ascii")
    t = "".join(c for c in raw if c.isprintable())
    return (t[: max_len - 1] + "...") if len(t) > max_len else t


# %% Load embedding cache & tokenizer
path = cache_path(MODEL_NAME)
if not path.is_file():
    raise FileNotFoundError(f"Missing cache: {path}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
payload = torch.load(path, map_location=str(device), weights_only=False)
W = payload["weight"].float()
if W.dim() != 2:
    raise ValueError(f"expected 2d weight, got {W.shape}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# %% Token A # stonesoup:cell-input
_in = globals().get("CELL_INPUT", "")
if _in:
    TOKEN_A = _in

# %% Token B # stonesoup:cell-input
_in = globals().get("CELL_INPUT", "")
if _in:
    TOKEN_B = _in

# %% Resolve A, B token ids & unit vectors
ia = single_token_id(TOKEN_A, tokenizer)
ib = single_token_id(TOKEN_B, tokenizer)
a = W[ia] / W[ia].norm()
b = W[ib] / W[ib].norm()

# %% Midpoint M cos and angles
t_mid = torch.tensor([0.5], device=a.device, dtype=a.dtype)
M = slerp_unit(a, b, t_mid)[0]

cos_ab = float((a * b).sum().clamp(-1.0, 1.0))
cos_am = float((a * M).sum().clamp(-1.0, 1.0))
cos_bm = float((b * M).sum().clamp(-1.0, 1.0))
rad_ab = math.acos(cos_ab)
rad_am = math.acos(cos_am)
rad_bm = math.acos(cos_bm)

print(f"A={TOKEN_A!r}  B={TOKEN_B!r}")
print(f"cos(A,B)={cos_ab:.6f}  angle(A,B)={rad_ab:.6f} rad ({math.degrees(rad_ab):.4f} deg)")
print(f"cos(A,M)={cos_am:.6f}  angle(A,M)={rad_am:.6f} rad ({math.degrees(rad_am):.4f} deg)")
print(f"cos(B,M)={cos_bm:.6f}  angle(B,M)={rad_bm:.6f} rad ({math.degrees(rad_bm):.4f} deg)")
print(f"||M||={float(M.norm()):.6f}  (SLERP midpoint t=0.5)")

# %% SLERP path & top-3 vocab neighbors
dt = 1.0 / max(N_STEPS - 1, 1)
omega_rad_path = float(math.acos(float((a * b).sum().clamp(-1.0, 1.0))))
n_extrap = extrap_steps_each_side_for_full_turn(dt, omega_rad_path)
dtype = a.dtype
t_left = torch.arange(-n_extrap, 0, device=device, dtype=dtype) * dt
t_mid = torch.linspace(0.0, 1.0, N_STEPS, device=device, dtype=dtype)
t_right = 1.0 + torch.arange(1, n_extrap + 1, device=device, dtype=dtype) * dt
t = torch.cat([t_left, t_mid, t_right])
V = slerp_unit(a, b, t)
n_pts = int(t.shape[0])

omega_deg = math.degrees(omega_rad_path)

Wn = W / (W.norm(dim=1, keepdim=True) + 1e-8)
cos = V @ Wn.T

top_k = min(3, cos.shape[1])
top_val, top_j = torch.topk(cos, k=top_k, dim=1)

print(f"model={MODEL_NAME}  cache={path.name}  A={TOKEN_A!r} id={ia}  B={TOKEN_B!r} id={ib}")
print(
    f"cos(A,B)={math.cos(omega_rad_path):.6f}  angle(A,B)={omega_deg:.4f} deg  "
    f"N_STEPS={N_STEPS}  n_extrap_each_side={n_extrap}  dt={dt:.6g}  n_pts={n_pts}  "
    f"(~360 deg in t*omega, slack 1e-4, minus 1 step/side vs ceil wrap)"
)
print("arc_A_deg = wrap180(t * angle(A,B)) in (-180, 180] — same direction mod 360.\n")
for k in range(n_pts):
    tk = float(t[k])
    if k > 0:
        prev = float(t[k - 1])
        if prev < 0 <= tk:
            print("--- t in [0, 1] : interpolate (A at t=0, B at t=1) ---")
        if prev <= 1 < tk:
            print("--- t > 1 : extrapolate past B ---")
    arc_a_deg = wrap_deg_signed180(tk * omega_deg)
    bits = []
    for r in range(top_k):
        j = int(top_j[k, r])
        bits.append(f"{float(top_val[k, r]):.6f}  {tokenizer.decode([j])!r}")
    print(f"t={tk:+7.4f}  arc_A={arc_a_deg:+9.2f} deg  " + "  |  ".join(bits))

# %% Record SLERP vector at t (for vocab cosine plot)
T_RECORD = -1.8
t_record = torch.tensor([T_RECORD], device=a.device, dtype=a.dtype)
V_t_record = slerp_unit(a, b, t_record)[0]
print(f"T_RECORD={T_RECORD}  ||V_t_record||={float(V_t_record.norm()):.6f}")

# %% Plot histogram of cos(V_t_record, all vocab rows)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PLOTS_DIR = THIS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

Wn = W / (W.norm(dim=1, keepdim=True) + 1e-8)
v = V_t_record / (V_t_record.norm() + 1e-8)
with torch.inference_mode():
    cos_all = (v @ Wn.T).float().cpu().numpy().ravel()

fig, ax = plt.subplots(figsize=(8, 4), dpi=140)
ax.hist(cos_all, bins=120, color="steelblue", edgecolor="white", linewidth=0.3)
ax.axvline(float(cos_all[ia]), color="coral", linewidth=1.2, label=f"A {TOKEN_A!r}")
ax.axvline(float(cos_all[ib]), color="seagreen", linewidth=1.2, label=f"B {TOKEN_B!r}")
ax.set_xlabel("cos(V, embedding[j])")
ax.set_ylabel("count")
ax.set_title(
    f"SLERP(A,B) at t={T_RECORD} — cosine to all {len(cos_all):,} vocab rows"
)
ax.legend(loc="upper right", fontsize=8)
fig.tight_layout()
_out = PLOTS_DIR / f"vocab_cosine_hist_t{T_RECORD}.png"
fig.savefig(_out)
plt.close(fig)
print(f"wrote {_out}")

# %% Unit circle: SLERP ticks + top-3 neighbors (2D plane spanned by A, B)
import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, TextArea, VPacker

PLOTS_DIR.mkdir(parents=True, exist_ok=True)

cos_ab_unit = float((a * b).sum().clamp(-1.0, 1.0))
omega_rad = float(math.acos(cos_ab_unit))  # angle(A,B) in R^d; a,b unit rows
dt_c = 1.0 / max(N_STEPS - 1, 1)
n_extrap_c = extrap_steps_each_side_for_full_turn(dt_c, omega_rad)
dtype_c = a.dtype
t_left_c = torch.arange(-n_extrap_c, 0, device=device, dtype=dtype_c) * dt_c
t_mid_c = torch.linspace(0.0, 1.0, N_STEPS, device=device, dtype=dtype_c)
t_right_c = 1.0 + torch.arange(1, n_extrap_c + 1, device=device, dtype=dtype_c) * dt_c
t_c = torch.cat([t_left_c, t_mid_c, t_right_c])
print(f"unit circle: n_extrap_each_side={n_extrap_c}  (same ~360 deg rule as SLERP cell)")
V_c = slerp_unit(a, b, t_c)
Wn_c = W / (W.norm(dim=1, keepdim=True) + 1e-8)
with torch.inference_mode():
    cos_c = V_c @ Wn_c.T
    tk_c = min(3, cos_c.shape[1])
    top_val_c, top_j_c = torch.topk(cos_c, k=tk_c, dim=1)
top_j_c = top_j_c.cpu()
top_val_c = top_val_c.cpu()
t_np = t_c.detach().float().cpu().numpy()
# Display frame: A at top-left, B at top-right, mirror symmetry across +y axis.
# θ_A = π/2 + ω/2, θ_B = π/2 − ω/2  ⇒  |θ_A − θ_B| = ω = arccos(a·b).
# Same plane as span{a,b}; rotation preserves dot product, so cos(θ_A−θ_B) = a·b.
angles = np.pi / 2.0 + omega_rad / 2.0 - t_np * omega_rad
xc = np.cos(angles)
yc = np.sin(angles)

fig2, ax2 = plt.subplots(figsize=(12, 12), dpi=160)
ring = np.linspace(0, 2 * math.pi, 500)
ax2.plot(np.cos(ring), np.sin(ring), color="0.5", linewidth=0.9, alpha=0.5)
theta_a = math.pi / 2.0 + omega_rad / 2.0
theta_b = math.pi / 2.0 - omega_rad / 2.0
a_x, a_y = math.cos(theta_a), math.sin(theta_a)
b_x, b_y = math.cos(theta_b), math.sin(theta_b)
_arc_ab_plot = abs(theta_a - theta_b)
_dot_plot = a_x * b_x + a_y * b_y
if abs(_arc_ab_plot - omega_rad) > 1e-6 or abs(_dot_plot - cos_ab_unit) > 1e-4:
    print(
        "WARNING circle geometry mismatch:",
        f"arc={math.degrees(_arc_ab_plot):.6f} deg vs ω={math.degrees(omega_rad):.6f} deg;",
        f"2D dot={_dot_plot:.8f} vs cos(a,b)={cos_ab_unit:.8f}",
    )
else:
    print(
        "circle OK: minor arc A–B =",
        f"{math.degrees(_arc_ab_plot):.4f} deg == arccos(a·b);",
        f"2D dot {_dot_plot:.6f} == cos(a,b) {cos_ab_unit:.6f}",
    )
_arr_kw = dict(arrowstyle="->", mutation_scale=16, lw=1.5, shrinkA=0, shrinkB=4)
ax2.annotate(
    "",
    xy=(a_x, a_y),
    xytext=(0.0, 0.0),
    arrowprops={**_arr_kw, "color": "coral", "alpha": 0.9},
    zorder=4,
)
ax2.annotate(
    "",
    xy=(b_x, b_y),
    xytext=(0.0, 0.0),
    arrowprops={**_arr_kw, "color": "seagreen", "alpha": 0.9},
    zorder=4,
)
ax2.scatter([a_x], [a_y], c="coral", s=140, zorder=6, edgecolors="white", linewidths=0.5)
ax2.annotate(
    f"A\n{plot_safe_token(repr(TOKEN_A), max_len=48)}",
    (a_x * 1.15, a_y * 1.15),
    fontsize=14,
    ha="center",
    va="center",
    color="darkred",
)
ax2.scatter([b_x], [b_y], c="seagreen", s=140, zorder=6, edgecolors="white", linewidths=0.5)
ax2.annotate(
    f"B\n{plot_safe_token(repr(TOKEN_B), max_len=48)}",
    (b_x * 1.15, b_y * 1.15),
    fontsize=14,
    ha="center",
    va="center",
    color="darkgreen",
)

_aob_deg = math.degrees(omega_rad)
_arc_r = 0.36
_arc = mpatches.Arc(
    (0.0, 0.0),
    2.0 * _arc_r,
    2.0 * _arc_r,
    angle=0.0,
    theta1=math.degrees(theta_b),
    theta2=math.degrees(theta_a),
    color="0.45",
    linewidth=1.1,
    zorder=3,
    linestyle=(0, (3, 3)),
)
ax2.add_patch(_arc)
ax2.annotate(
    f"{_aob_deg:.2f} deg",
    (0.0, _arc_r + 0.08),
    fontsize=11,
    ha="center",
    va="bottom",
    color="0.2",
    zorder=5,
)

rad_lbl = 1.4
_tick_box_props = dict(
    boxstyle="round,pad=0.3",
    facecolor="white",
    edgecolor="0.65",
    linewidth=0.6,
    alpha=0.94,
)
for k in range(len(t_np)):
    x, y = float(xc[k]), float(yc[k])
    ax2.plot([0.92 * x, x], [0.92 * y, y], color="0.35", linewidth=0.85, zorder=3)
    ax2.scatter([x], [y], c="steelblue", s=24, zorder=5, edgecolors="white", linewidths=0.4)
    stack = []
    for r in range(tk_c):
        j = int(top_j_c[k, r])
        csim = float(top_val_c[k, r])
        tok = plot_safe_token(tokenizer.decode([j]), max_len=22)
        stack.append(
            TextArea(
                tok,
                textprops=dict(size=9.0, color="0.08", weight="medium", family="sans-serif"),
            )
        )
        stack.append(
            TextArea(
                f"{csim:+.3f}",
                textprops=dict(size=5.0, color="0.42", family="monospace"),
            )
        )
    pack = VPacker(children=stack, align="center", pad=0, sep=2)
    ab = AnnotationBbox(
        pack,
        (x * rad_lbl, y * rad_lbl),
        frameon=True,
        box_alignment=(0.5, 0.5),
        bboxprops=_tick_box_props,
        zorder=7,
    )
    ax2.add_artist(ab)

ax2.set_aspect("equal")
ax2.set_xlim(-1.78, 1.78)
ax2.set_ylim(-1.78, 1.78)
ax2.set_title(
    f"\nTop 3 tokens at each extrapolation/interpolation (by angle) points."
    f"\n\n{MODEL_NAME}",
    fontsize=18,
)
ax2.axis("off")
fig2.subplots_adjust(bottom=0.07)
_out2 = PLOTS_DIR / "slerp_unit_circle_top3_neighbors.png"
fig2.savefig(_out2, bbox_inches="tight")
plt.close(fig2)
print(f"wrote {_out2}")
