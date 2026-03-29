"""Parse ``# %%`` / ``#%%`` cells and execute them in a shared namespace."""

from __future__ import annotations

import ast
import builtins
import hashlib
import linecache
import re
import traceback
import warnings
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from types import CodeType
from typing import Any, Callable

# Headless matplotlib before any cell can ``import matplotlib.pyplot`` (avoids TkAgg in worker
# threads: tk ``__del__`` / "main thread is not in main loop" / Tcl_AsyncDelete).
try:
    import matplotlib

    matplotlib.use("Agg")
except ImportError:
    pass


_DEFAULT_TORCH_CUDA_MEMORY_FRACTION = 0.8


def _stonesoup_apply_cuda_memory_fraction_cap() -> None:
    """Cap PyTorch CUDA memory fraction before first CUDA alloc (unified RAM / OS headroom).

    Default **0.8** if unset. Override with ``STONESOUP_CUDA_MEMORY_FRACTION`` or
    ``AI_TORCH_CUDA_MEMORY_FRACTION`` (``0.05``…``1.0``; use ``1`` to relax the cap).
    """
    import os

    raw = os.environ.get("STONESOUP_CUDA_MEMORY_FRACTION", "").strip()
    if not raw:
        raw = os.environ.get("AI_TORCH_CUDA_MEMORY_FRACTION", "").strip()
    try:
        frac = float(raw) if raw else _DEFAULT_TORCH_CUDA_MEMORY_FRACTION
        if not (0.05 <= frac <= 1.0):
            return
        import torch

        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(frac, 0)
    except Exception:
        pass


_stonesoup_apply_cuda_memory_fraction_cap()

# VS Code / Spyder style: # %%, # %%, optional title after %%.
CELL_START = re.compile(r"^\s*#\s*%%(.*)$")

# Optional suffix on the same line, e.g. ``# %% Title # stonesoup:cell-input`` or ``# %% # stonesoup:cell-input``.
CELL_INPUT_SUFFIX = re.compile(r"\s*#\s*stonesoup:\s*cell-input\s*$", re.IGNORECASE)


_STONESOUP_CELL_FILENAME = "<stonesoup cell>"


def _prime_linecache_for_exec(display_name: str, source: str) -> None:
    """So tracebacks can show source for synthetic filenames; ``mtime`` None skips stat in checkcache."""
    lines = source.splitlines(keepends=True)
    if not lines:
        lines = ["\n"]
    else:
        lines = [line if line.endswith("\n") else line + "\n" for line in lines]
    linecache.cache[display_name] = (len(source), None, lines, display_name)


def _adjust_syntax_error_to_file(exc: SyntaxError, source_path: str, start_line_1: int) -> None:
    """Map cell-local ``SyntaxError`` line numbers to positions in ``source_path``."""
    delta = start_line_1 - 1
    exc.filename = source_path
    if exc.lineno is not None:
        exc.lineno += delta
    if exc.end_lineno is not None:
        exc.end_lineno += delta


def _compile_cell(
    source: str,
    source_path: str | None,
    start_line: int | None,
) -> CodeType:
    if source_path and start_line is not None and start_line >= 1:
        if not source.strip():
            return compile(source, source_path, "exec")
        try:
            tree = ast.parse(source, filename=source_path, mode="exec")
            ast.increment_lineno(tree, start_line - 1)
            return compile(tree, source_path, "exec")
        except SyntaxError as exc:
            _adjust_syntax_error_to_file(exc, source_path, start_line)
            raise
        except Exception:
            pass

    _prime_linecache_for_exec(_STONESOUP_CELL_FILENAME, source)
    return compile(source, _STONESOUP_CELL_FILENAME, "exec")


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
    start_line: int
    cell_input: bool = False

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "title": self.title,
            "source": self.source,
            "marker_key": self.marker_key,
            "start_line": self.start_line,
            # Alias for JSON consumers / tooling that expect camelCase
            "startLine": self.start_line,
            "cell_input": self.cell_input,
        }


def parse_cells(text: str) -> list[Cell]:
    """Split a Python file into cells at lines matching ``# %%`` or ``#%%``.

    A cell shows a run-input field in the UI when the marker line ends with
    ``# stonesoup:cell-input`` (case-insensitive). The suffix is stripped from the title.
    """
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
            marker_line_1 = i + 1
            raw_rest = m.group(1).strip()
            wants_input = bool(CELL_INPUT_SUFFIX.search(raw_rest))
            t = CELL_INPUT_SUFFIX.sub("", raw_rest).strip() if wants_input else raw_rest.strip()
            title = t if t else title
            i += 1
            # Deeplink: first line of cell *body* (after ``# %%``); if none, stay on marker line.
            start_line_1 = (i + 1) if i < n else marker_line_1
        else:
            wants_input = False
            start_line_1 = i + 1
        chunk: list[str] = []
        while i < n and not CELL_START.match(lines[i]):
            chunk.append(lines[i])
            i += 1
        source = "".join(chunk).rstrip("\n")
        mk = fingerprint_marker_line(marker_line)
        cells.append(
            Cell(
                index=len(cells),
                title=title,
                source=source,
                marker_key=mk,
                start_line=start_line_1,
                cell_input=wants_input,
            )
        )

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
        start_line: int | None = None,
        on_stdout_chunk: Callable[[str], None] | None = None,
        on_stderr_chunk: Callable[[str], None] | None = None,
    ) -> tuple[str, str, bool]:
        """
        Execute ``source`` in the shared namespace.

        If ``inject`` is set, assign those names into the namespace before running
        the cell (shallow merge; later pipeline steps and iterations can overwrite).

        When ``source_path`` and ``start_line`` (1-based first line of the cell body in that
        file) are set, compile so tracebacks point at the watched file and matching line numbers.

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
            code = _compile_cell(source, source_path, start_line)
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

    def snapshot_globals_for_ui(self, *, max_preview: int = 120) -> list[dict[str, str]]:
        """JSON-safe name / type / repr preview for UI (excludes dunder and ``__builtins__``)."""
        skip = frozenset(
            {
                "__builtins__",
                "__name__",
                "__doc__",
                "__package__",
                "__loader__",
                "__spec__",
                "__file__",
                "__cached__",
            }
        )
        # Injected each run for pipelines; not useful in the variables panel.
        skip_pipeline = frozenset({"LOOP_INDEX", "LOOP_ITEM"})
        rows: list[dict[str, str]] = []
        for name in sorted(self.globals.keys(), key=lambda s: (s.lower(), s)):
            if name in skip or name in skip_pipeline or name.startswith("__"):
                continue
            val = self.globals[name]
            typ = type(val).__name__
            try:
                preview = repr(val)
            except Exception as exc:  # noqa: BLE001 — user code in repr
                preview = f"<repr error: {exc!r}>"
            if len(preview) > max_preview:
                preview = preview[: max_preview - 1] + "…"
            rows.append({"name": name, "type": typ, "preview": preview})
        return rows
