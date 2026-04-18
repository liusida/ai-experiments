"""Display helpers: rich cell output from Matplotlib without a GUI."""

from __future__ import annotations

import uuid
import warnings
from pathlib import Path
from typing import Any

from .paths import outputs_dir, repo_root, script_path


def configure_matplotlib_agg() -> None:
    """Use the Agg backend (no GUI). Safe to call once per Stonesoup kernel."""
    import matplotlib

    matplotlib.use("Agg")


def output_url_path(path: Path, *, cache_bust: bool = True) -> str:
    """Build a web path under the repo for static files (e.g. ``/outputs/…?t=…``).

    ``path`` must be under :func:`repo_root`. Same cache-busting query as :func:`show` when
    ``cache_bust`` is True.
    """
    root = repo_root()
    abs_path = path.resolve()
    rel = abs_path.relative_to(root)
    rel_posix = rel.as_posix()
    if cache_bust:
        t_ms = abs_path.stat().st_mtime_ns // 1_000_000
        return f"/{rel_posix}?t={t_ms}"
    return f"/{rel_posix}"

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


def configure_matplotlib_unicode_fonts() -> None:
    """Register bundled Noto (Latin + CJK) and disable Unicode-minus so mixed scripts render reliably.

    Call once after :func:`configure_matplotlib_agg`. For tokenizer labels with Han characters,
    also call :func:`apply_matplotlib_fonts_to_figure` on each figure before ``tight_layout`` /
    :func:`show` so existing :class:`~matplotlib.text.Text` artists pick up the font stack.
    """
    import matplotlib as mpl

    _ensure_matplotlib_font_stack()
    mpl.rcParams["axes.unicode_minus"] = False


def apply_matplotlib_fonts_to_figure(fig: Any) -> None:
    """Re-apply the current ``font.family`` stack to all text on ``fig`` (CJK tick labels, titles)."""
    _apply_font_family_to_figure(fig)


def show(
    fig: Any | None = None,
    *,
    dpi: int = 120,
    format: str = "png",
    basename: str | None = None,
    emit_render_hint: bool = True,
    **savefig_kw: Any,
) -> str:
    """Save the current (or given) Matplotlib figure using :func:`outputs_dir` and print HTML.

    **You do not need to pass a figure.** With ``fig=None`` (the default), the **current** pyplot
    figure is used—whatever you last drew with ``matplotlib.pyplot`` (e.g. ``plt.imshow``,
    ``plt.plot``, ``plt.subplots``). You never need ``fig = plt.gcf()`` or ``stonesoup.show(fig)``
    unless you want a non-current figure explicitly.

    Same directory as :func:`outputs_dir`: ``outputs/<…>/<script_stem>/`` under the repo (leading
    ``experiments/`` stripped from the watched path; see :mod:`stonesoup.experiment.paths`).
    The backend serves those files under ``/outputs/…`` (Vite proxies ``/outputs`` in dev). By
    default, stdout starts with ``# stonesoup:render=html`` for the following ``<img>`` HTML (the UI
    only reads that hint from the **first** line of the cell’s combined stdout).

    With ``emit_render_hint=True`` (default), the hint is printed only **once per cell**; later
    ``show()`` / ``display()`` calls omit it. If you ``print()`` other text before the first rich
    output, put ``print(stonesoup.STONESOUP_RENDER_HTML, end="")`` at the **very start** of the cell
    so line 1 is still the hint, or call ``stonesoup.mark_render_hint_emitted()`` after your manual
    hint. Use ``emit_render_hint=False`` on any call to suppress the hint for that call only.

    If ``basename`` is set, the file is ``{basename}.{format}`` (no directory components; ``.`` in
    ``basename`` is allowed so you can pass ``foo.v2`` for ``foo.v2.png``). Otherwise a random name
    is used.

    After a successful ``savefig``, the figure is closed with ``matplotlib.pyplot.close(fig)`` to
    release memory; do not reuse the same ``Figure`` instance afterward.

    Typical use::

        import matplotlib.pyplot as plt
        plt.plot([0, 1], [0, 1])
        stonesoup.show()

    Heatmap / image only::

        plt.imshow(xxt)
        stonesoup.show()

    Returns the repo-relative POSIX path (under ``outputs/…``, same rule as :func:`outputs_dir`).
    """
    import matplotlib.figure
    import matplotlib.pyplot as plt

    from stonesoup import STONESOUP_RENDER_HTML

    if fig is None:
        fig = plt.gcf()

    if not isinstance(fig, matplotlib.figure.Figure):
        raise TypeError("stonesoup.show() expects a matplotlib.figure.Figure or use the current figure (pass None).")

    root = repo_root()
    out_dir = outputs_dir()

    fmt = format.lower().lstrip(".")
    if fmt not in ("png", "svg", "pdf", "webp"):
        raise ValueError(f"Unsupported format {format!r}; try png, svg, pdf, or webp.")

    # Never forward Stonesoup-only options to ``Figure.savefig`` (e.g. if ``basename`` slipped into
    # ``**savefig_kw`` from an outdated wrapper or duplicate kwarg).
    if basename is None and "basename" in savefig_kw:
        basename = savefig_kw.pop("basename")
    else:
        savefig_kw.pop("basename", None)

    if basename is not None:
        raw = basename.strip()
        stem_path = Path(raw)
        if not raw or len(stem_path.parts) != 1 or stem_path.is_absolute():
            raise ValueError(
                "show(basename=...): use one path segment, no extension "
                f"(saved as {{basename}}.{fmt}); got {basename!r}"
            )
        name = f"{stem_path.name}.{fmt}"
    else:
        name = f"{uuid.uuid4().hex}.{fmt}"
    abs_path = out_dir / name
    # Never forward Stonesoup-only flags to ``Figure.savefig`` (e.g. legacy wrappers that put
    # ``emit_render_hint`` in ``**kwargs`` when ``show`` had no explicit parameter).
    _emit = savefig_kw.pop("emit_render_hint", emit_render_hint)
    emit_render_hint = bool(_emit)
    kw = {"dpi": dpi, "bbox_inches": "tight", **savefig_kw}
    _ensure_matplotlib_font_stack()
    _apply_font_family_to_figure(fig)
    # Ensure implicit ``plt.imshow`` / pyplot state is fully drawn before ``savefig`` (Agg backend).
    try:
        fig.canvas.draw()
    except Exception:
        pass
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
    # Bust browser cache when the same path is overwritten (e.g. fixed basename).
    t_ms = abs_path.stat().st_mtime_ns // 1_000_000
    src = f"/{rel_posix}?t={t_ms}"

    try:
        # Two-part stdout: render hint then HTML (see peelStonesoupRenderHint in the UI).
        if emit_render_hint:
            from stonesoup.backend.render_hint_state import (
                mark_rich_render_hint_emitted,
                rich_render_hint_already_emitted,
            )

            if not rich_render_hint_already_emitted():
                print(STONESOUP_RENDER_HTML, end="")
                mark_rich_render_hint_emitted()
        print(
            f'<p class="stonesoup-show"><img src="{src}" alt="stonesoup.show()" loading="lazy" '
            f'style="max-width:100%;height:auto" /></p>',
            flush=True,
        )
    finally:
        plt.close(fig)

    return rel_posix


def _rich_html_fragment(
    obj: Any,
    *,
    max_rows: int,
    max_cols: int,
    **to_html_kw: Any,
) -> str | None:
    """Return HTML string or None if no rich representation is available.

    :class:`pandas.DataFrame` / :class:`pandas.Series` and :class:`pandas.io.formats.style.Styler`
    are rendered with :meth:`~pandas.DataFrame.to_html` so ``max_rows`` / ``max_cols`` are honored.
    They must be handled *before* generic ``_repr_html_``: pandas' HTML repr ignores those limits
    and follows global display options instead.
    """
    try:
        import pandas as pd
    except ImportError:
        pd = None  # type: ignore[assignment]

    if pd is not None:
        if isinstance(obj, (pd.DataFrame, pd.Series)):
            return obj.to_html(max_rows=max_rows, max_cols=max_cols, **to_html_kw)
        try:
            from pandas.io.formats.style import Styler
        except ImportError:
            Styler = None  # type: ignore[assignment,misc]
        if Styler is not None and isinstance(obj, Styler):
            try:
                return obj.to_html(max_rows=max_rows, max_cols=max_cols, **to_html_kw)
            except Exception:
                pass

    repr_html = getattr(obj, "_repr_html_", None)
    if callable(repr_html):
        try:
            out = repr_html()
            if isinstance(out, str) and out.strip():
                return out
        except Exception:
            pass

    return None


def display(
    obj: Any,
    *,
    max_rows: int = 30,
    max_cols: int = 20,
    emit_render_hint: bool = True,
    **to_html_kw: Any,
) -> None:
    """Print *obj* as HTML in the cell output (requires first stdout line ``# stonesoup:render=html``).

    :class:`pandas.DataFrame`, :class:`pandas.Series`, and :class:`pandas.io.formats.style.Styler`
    use :meth:`~pandas.DataFrame.to_html` with ``max_rows`` / ``max_cols``. Other objects use
    ``_repr_html_()`` when present and non-empty.

    If nothing applies, prints :func:`repr` only (plain text, no render hint).

    With ``emit_render_hint=True`` (default), the hint is printed only **once per cell run**; later
    ``display()`` / ``show()`` calls omit it. If you print ``STONESOUP_RENDER_HTML`` yourself first,
    call :func:`stonesoup.mark_render_hint_emitted` so helpers do not duplicate the hint.

    If you ``print()`` other text before the first rich output, emit ``STONESOUP_RENDER_HTML`` at
    the **very start** of the cell so the first line remains the hint (same as :func:`show`).
    """
    from stonesoup import STONESOUP_RENDER_HTML
    from stonesoup.backend.render_hint_state import (
        mark_rich_render_hint_emitted,
        rich_render_hint_already_emitted,
    )

    body = _rich_html_fragment(obj, max_rows=max_rows, max_cols=max_cols, **to_html_kw)
    if body is None:
        print(repr(obj))
        return
    wrapped = f'<div class="stonesoup-rich-html">{body}</div>'
    if emit_render_hint and not rich_render_hint_already_emitted():
        print(STONESOUP_RENDER_HTML, end="")
        mark_rich_render_hint_emitted()
    print(wrapped, flush=True)


def emit_html_output_hint() -> None:
    """Print ``# stonesoup:render=html`` and mark that hint as emitted for this cell run.

    Use as the **first** stdout in the cell (before any other ``print``) so the UI renders the rest
    as HTML.     Same as ``print(stonesoup.STONESOUP_RENDER_HTML, end="")`` plus
    ``stonesoup.mark_render_hint_emitted()`` — convenient when you build HTML by hand.
    """
    from stonesoup import STONESOUP_RENDER_HTML
    from stonesoup.backend.render_hint_state import mark_rich_render_hint_emitted

    print(STONESOUP_RENDER_HTML, end="")
    mark_rich_render_hint_emitted()
