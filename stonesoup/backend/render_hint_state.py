"""Track whether the HTML render hint was already emitted in the current cell run.

``Kernel.run_cell`` resets this at the start of each execution so ``stonesoup.display`` /
``stonesoup.show`` only print ``# stonesoup:render=html`` once per cell by default.
"""

from __future__ import annotations

import threading

_local = threading.local()


def reset_render_hint_for_cell() -> None:
    """Clear state; call once at the beginning of each cell run."""
    _local.emitted = False


def rich_render_hint_already_emitted() -> bool:
    return bool(getattr(_local, "emitted", False))


def mark_rich_render_hint_emitted() -> None:
    """Record that the hint line was written (by a helper or manually)."""
    _local.emitted = True
