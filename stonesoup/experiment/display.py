"""Display helpers: rich cell output from Matplotlib without a GUI."""

from __future__ import annotations

import uuid
import warnings
from pathlib import Path
from typing import Any

from .paths import plot_dir, repo_root, script_path

# Bundled under ``misc/fonts/`` (Noto Sans / Hebrew + Noto Sans CJK). Override with env if needed:
# STONESOUP_MATPLOTLIB_NOTO_DIR (dir containing NotoSans-*.ttf / Hebrew), STONESOUP_MATPLOTLIB_CJK_FONT (.ttc path).


def _bundled_font_dir() -> Path:
    """Directory with bundled Noto files.

    Prefer walking up from :func:`script_path` so CJK loads even when ``repo_root()`` does not
    match the checkout (e.g. ``STONESOUP_ROOT`` / install layout).
    """
    markers = ("NotoSansCJK-Regular.ttc", "NotoSans-Regular.ttf")
    try:
        for ancestor in script_path().resolve().parents:
            d = ancestor / "misc" / "fonts"
            if all((d / m).is_file() for m in markers):
                return d
    except RuntimeError:
        pass
    return repo_root() / "misc" / "fonts"


def _default_noto_sans_dir() -> Path:
    return _bundled_font_dir()


def _default_cjk_font_path() -> Path:
    return _bundled_font_dir() / "NotoSansCJK-Regular.ttc"


def _register_font_file(path: Path) -> str | None:
    import matplotlib.font_manager as fm
    from matplotlib.font_manager import FontProperties

    path = path.resolve()
    if not path.is_file():
        return None
    try:
        fm.fontManager.addfont(str(path))
    except Exception:
        return None
    try:
        return FontProperties(fname=str(path)).get_name()
    except Exception:
        return None


def _ensure_matplotlib_font_stack() -> None:
    """Register Noto sans / CJK / Hebrew and set ``font.family`` to a multi-family list.

    Matplotlib only chains glyph fallback when ``font.family`` is a **concrete** list of families.

    Fonts load from ``misc/fonts`` at the repo root unless env vars override paths.
    Order: Latin, then CJK, then Hebrew (better for mixed LTR/RTL strings than CJK-after-Hebrew).

    Noto **Color** Emoji cannot be loaded by Matplotlib's FreeType layer on typical Linux installs
    (``addfont`` / ``FT2Font`` fail on ``NotoColorEmoji.ttf``); color emoji in labels may be omitted
    with a suppressed missing-glyph warning during ``savefig``.
    """
    import os

    import matplotlib as mpl

    noto_dir = Path(
        (os.environ.get("STONESOUP_MATPLOTLIB_NOTO_DIR") or "").strip()
        or str(_default_noto_sans_dir()),
    )
    cjk_path = Path(
        (os.environ.get("STONESOUP_MATPLOTLIB_CJK_FONT") or "").strip()
        or str(_default_cjk_font_path()),
    )

    # CJK before Hebrew so per-character fallback still reaches CJK in mixed strings.
    register_paths = [
        noto_dir / "NotoSans-Regular.ttf",
        cjk_path,
        noto_dir / "NotoSansHebrew-Regular.ttf",
    ]
    seen: set[str] = set()
    families: list[str] = []
    for p in register_paths:
        name = _register_font_file(p)
        if name and name not in seen:
            seen.add(name)
            families.append(name)

    if not families:
        return

    # Last resort for any remaining codepoints (not covered by Noto above).
    if "DejaVu Sans" not in families:
        families.append("DejaVu Sans")

    mpl.rcParams["font.family"] = families


def _apply_font_family_to_figure(fig: Any) -> None:
    """Point all ``Text`` artists at the current ``font.family`` (list) for glyph fallback.

    ``Text`` snapshots generic ``sans-serif`` at creation time; updating ``rcParams`` alone does not
    change existing labels, so CJK would stay on DejaVu until this runs.
    """
    import matplotlib as mpl
    import matplotlib.text

    fam = mpl.rcParams.get("font.family")
    if not fam:
        return
    fam_list: list[str] = [fam] if isinstance(fam, str) else list(fam)

    for obj in fig.findobj(lambda x: isinstance(x, matplotlib.text.Text)):
        try:
            obj.set_fontfamily(fam_list)
        except (TypeError, ValueError, AttributeError):
            pass


def show(fig: Any | None = None, *, dpi: int = 120, format: str = "png", **savefig_kw: Any) -> str:
    """Save the current (or given) Matplotlib figure using :func:`plot_dir` (``outputs/stonesoup/``) and print HTML.

    Same directory as ``plot_dir()`` (``outputs/stonesoup/<repo-relative script path>/``). The
    backend serves under ``/outputs/…`` (Vite proxies ``/outputs`` in dev). Stdout starts with
    ``# stonesoup:render=html`` for the following ``<img>`` HTML.

    Typical use::

        import matplotlib.pyplot as plt
        plt.plot([0, 1], [0, 1])
        stonesoup.show()

    Returns the repo-relative POSIX path (``outputs/stonesoup/...``).
    """
    import matplotlib.figure

    from stonesoup import STONESOUP_RENDER_HTML

    if fig is None:
        import matplotlib.pyplot as plt

        fig = plt.gcf()

    if not isinstance(fig, matplotlib.figure.Figure):
        raise TypeError("stonesoup.show() expects a matplotlib.figure.Figure or use the current figure (pass None).")

    root = repo_root()
    out_dir = plot_dir()

    fmt = format.lower().lstrip(".")
    if fmt not in ("png", "svg", "pdf", "webp"):
        raise ValueError(f"Unsupported format {format!r}; try png, svg, pdf, or webp.")

    name = f"{uuid.uuid4().hex}.{fmt}"
    abs_path = out_dir / name
    kw = {"dpi": dpi, "bbox_inches": "tight", **savefig_kw}
    _ensure_matplotlib_font_stack()
    _apply_font_family_to_figure(fig)
    # Color emoji (e.g. Noto Color Emoji) is not FT2Font-loadable here; hide noisy missing-glyph churn.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message=r"Glyph .+ missing from font",
        )
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message=r"Matplotlib currently does not support Hebrew natively",
        )
        fig.savefig(abs_path, format=fmt, **kw)

    rel_posix = (abs_path.relative_to(root)).as_posix()
    src = f"/{rel_posix}"

    # Two-part stdout: render hint then HTML (see peelStonesoupRenderHint in the UI).
    print(STONESOUP_RENDER_HTML, end="")
    print(
        f'<p class="stonesoup-show"><img src="{src}" alt="stonesoup.show()" loading="lazy" '
        f'style="max-width:100%;height:auto" /></p>',
        flush=True,
    )

    return rel_posix
