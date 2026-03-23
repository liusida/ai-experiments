"""Parse ``# %%`` / ``#%%`` cells and execute them in a shared namespace."""

from __future__ import annotations

import builtins
import hashlib
import re
import traceback
import warnings
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from typing import Any, Callable

# VS Code / Spyder style: # %%, # %%, optional title after %%.
CELL_START = re.compile(r"^\s*#\s*%%(.*)$")


def fingerprint_marker_line(marker_line: str | None) -> str:
    """
    Stable id for a cell opener: SHA-256 of the raw ``# %%`` line (no trailing newline),
    or a fixed sentinel when the file starts with content before any marker (implicit head cell).
    """
    if marker_line is None:
        payload = b"__stonesoup_implicit__"
    else:
        payload = marker_line.rstrip("\r\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


@dataclass
class Cell:
    index: int
    title: str
    source: str
    marker_key: str

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "title": self.title,
            "source": self.source,
            "marker_key": self.marker_key,
        }


def parse_cells(text: str) -> list[Cell]:
    """Split a Python file into cells at lines matching ``# %%`` or ``#%%``."""
    lines = text.splitlines(keepends=True)
    cells: list[Cell] = []
    i = 0
    n = len(lines)

    while i < n:
        title = f"Cell {len(cells)}"
        marker_line: str | None = None
        m = CELL_START.match(lines[i])
        if m:
            marker_line = lines[i]
            t = m.group(1).strip()
            title = t if t else title
            i += 1
        chunk: list[str] = []
        while i < n and not CELL_START.match(lines[i]):
            chunk.append(lines[i])
            i += 1
        source = "".join(chunk).rstrip("\n")
        mk = fingerprint_marker_line(marker_line)
        cells.append(Cell(index=len(cells), title=title, source=source, marker_key=mk))

    return cells


class _StreamSink:
    """File-like writer: buffers full text and forwards each ``write`` to ``on_chunk`` (for live UI)."""

    __slots__ = ("_parts", "_on_chunk")

    def __init__(self, on_chunk: Callable[[str], None] | None) -> None:
        self._parts: list[str] = []
        self._on_chunk = on_chunk

    def write(self, s: str) -> int:
        self._parts.append(s)
        if s and self._on_chunk is not None:
            self._on_chunk(s)
        return len(s)

    def flush(self) -> None:
        pass

    def getvalue(self) -> str:
        return "".join(self._parts)


class Kernel:
    """One persistent global namespace for sequential ``exec`` of cell sources."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.globals: dict = {
            "__name__": "__main__",
            "__builtins__": builtins,
        }

    def _apply_inject(self, inject: dict[str, Any] | None) -> None:
        """Merge JSON-safe keys into the namespace (no ``__*`` keys)."""
        if not inject:
            return
        for k, v in inject.items():
            if isinstance(k, str) and k.startswith("__"):
                continue
            self.globals[k] = v

    def run_cell(
        self,
        source: str,
        *,
        inject: dict[str, Any] | None = None,
        source_path: str | None = None,
        on_stdout_chunk: Callable[[str], None] | None = None,
        on_stderr_chunk: Callable[[str], None] | None = None,
    ) -> tuple[str, str, bool]:
        """
        Execute ``source`` in the shared namespace.

        If ``inject`` is set, assign those names into the namespace before running
        the cell (shallow merge; later pipeline steps and iterations can overwrite).

        Optional ``on_stdout_chunk`` / ``on_stderr_chunk`` are invoked for each ``write``
        (for streaming to the UI).

        Returns (stdout, stderr, ok). On failure, stderr includes traceback.
        """
        out_sink = _StreamSink(on_stdout_chunk)
        err_sink = _StreamSink(on_stderr_chunk)
        ok = True
        try:
            # Reserved pipeline names: defined every run so single-cell Run does not NameError;
            # client inject overwrites before exec.
            self.globals["LOOP_INDEX"] = None
            self.globals["LOOP_ITEM"] = None
            self._apply_inject(inject)
            if source_path is not None:
                self.globals["__file__"] = source_path
            code = compile(source, "<stonesoup cell>", "exec")
            with redirect_stdout(out_sink), redirect_stderr(err_sink):
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=r".*Matplotlib GUI outside of the main thread.*",
                        category=UserWarning,
                    )
                    exec(code, self.globals, self.globals)
        except BaseException:
            ok = False
            err_sink.write(traceback.format_exc())
        return out_sink.getvalue(), err_sink.getvalue(), ok
