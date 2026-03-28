"""Stonesoup: watched # %% cells + local Python kernel + floating UI."""

from __future__ import annotations

__version__ = "0.1.0"

# First stdout line: UI strips it. Default output is plain text; use HTML/MD lines only for rich panes.
# See EXPERIMENT_PYTHON.md (stdout render hint).
STONESOUP_RENDER_AUTO = "# stonesoup:render=auto\n"
STONESOUP_RENDER_TEXT = "# stonesoup:render=text\n"
STONESOUP_RENDER_HTML = "# stonesoup:render=html\n"
STONESOUP_RENDER_MARKDOWN = "# stonesoup:render=markdown\n"
STONESOUP_RENDER_MD = "# stonesoup:render=md\n"


def stonesoup_render_prefix(mode: str) -> str:
    """First stdout line for Stonesoup (trailing newline). Rich: html, markdown/md; plain: text, auto."""
    m = mode.strip().lower()
    if m in ("auto", "text", "html", "markdown", "md"):
        return f"# stonesoup:render={m}\n"
    raise ValueError(f"Unknown stonesoup render mode: {mode!r}")
