#!/usr/bin/env python3
"""Download Alice in Wonderland (Gutenberg-style plain text) into ``data/text/``.

Same mirrors and cache filename as ``embedding_qwen35_statistics.py``.

From repo root, **curl** (no Python)::

    mkdir -p data/text && curl -fL --retry 3 --connect-timeout 30 -A "curl" -o data/text/gutenberg_11-0_alice_in_wonderland.txt "https://raw.githubusercontent.com/GITenberg/Alice-s-Adventures-in-Wonderland_11/master/11-0.txt"

This script::

    uv run python experiments/2026-03-23-Embedding/download_alice_cache.py
    uv run python experiments/2026-03-23-Embedding/download_alice_cache.py --force
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import urllib.request
from urllib.error import URLError

# Keep in sync with ``embedding_qwen35_statistics.py`` (Alice cell).
SOURCE_URLS: tuple[str, ...] = (
    "https://raw.githubusercontent.com/GITenberg/Alice-s-Adventures-in-Wonderland_11/master/11-0.txt",
    "https://www.gutenberg.org/files/11/11-0.txt",
    "https://gutenberg.org/files/11/11-0.txt",
    "http://www.gutenberg.org/files/11/11-0.txt",
)
HTTP_UA = "StonesoupAliceCacheCLI/1.0 (+https://github.com/liusida/ai-experiments)"
# Single float: urllib/http.client passes timeout to socket.settimeout (no tuple support here).
URL_TIMEOUT_S = 240.0
MIN_BYTES = 5000

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUT = REPO_ROOT / "data" / "text" / "gutenberg_11-0_alice_in_wonderland.txt"


def fetch_first_ok() -> tuple[bytes, str]:
    last: BaseException | None = None
    for url in SOURCE_URLS:
        req = urllib.request.Request(url, headers={"User-Agent": HTTP_UA})
        try:
            print(f"trying {url}", flush=True)
            with urllib.request.urlopen(req, timeout=URL_TIMEOUT_S) as resp:
                data = resp.read()
            if len(data) < MIN_BYTES:
                raise ValueError(f"body too small ({len(data)} B)")
            print(f"  ok — {len(data):,} bytes", flush=True)
            return data, url
        except (URLError, TimeoutError, OSError, ValueError) as e:
            last = e
            print(f"  failed: {e}", flush=True)
    raise SystemExit(f"All mirrors failed. Last error: {last!r}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUT,
        help=f"output path (default: {DEFAULT_OUT})",
    )
    ap.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="download even if the output file already exists",
    )
    args = ap.parse_args()
    out: Path = args.output.resolve()
    if out.exists() and not args.force:
        print(f"Already exists (use --force to re-download):\n  {out}", flush=True)
        return 0

    out.parent.mkdir(parents=True, exist_ok=True)
    raw, used = fetch_first_ok()
    text = raw.decode("utf-8", errors="replace")
    out.write_text(text, encoding="utf-8")
    print(f"Wrote {len(text):,} chars\n  {out}\nSource: {used}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
