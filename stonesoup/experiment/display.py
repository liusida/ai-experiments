"""Display helpers: rich cell output from Matplotlib without a GUI."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from .paths import plot_dir, repo_root


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
    fig.savefig(abs_path, format=fmt, **kw)

    rel_posix = (abs_path.relative_to(root)).as_posix()
    src = f"/{rel_posix}"

    # Two-part stdout: render hint then HTML (see peelStonesoupRenderHint in the UI).
    print(STONESOUP_RENDER_HTML, end="")
    print(
        f'<p class="stonesoup-show"><img src="{src}" alt="stonesoup.show()" loading="lazy" /></p>',
        flush=True,
    )

    return rel_posix
