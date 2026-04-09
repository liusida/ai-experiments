"""Rich stdout line prefixes (aligned with :mod:`stonesoup`).

Re-exported for convenience from ``stonesoup.experiment`` so new scripts can import one module.
"""

from __future__ import annotations

from stonesoup import (
    STONESOUP_RENDER_AUTO,
    STONESOUP_RENDER_HTML,
    STONESOUP_RENDER_MARKDOWN,
    STONESOUP_RENDER_MD,
    STONESOUP_RENDER_TEXT,
    stonesoup_render_prefix,
)

__all__ = [
    "STONESOUP_RENDER_AUTO",
    "STONESOUP_RENDER_HTML",
    "STONESOUP_RENDER_MARKDOWN",
    "STONESOUP_RENDER_MD",
    "STONESOUP_RENDER_TEXT",
    "stonesoup_render_prefix",
]
