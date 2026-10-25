#!/usr/bin/env python3
"""Thin wrapper around `rse ingest`."""

from __future__ import annotations

import sys
from pathlib import Path

# Importable when run as a file from a checkout where the package is not
# installed, which is the normal state of a fresh GPU host: `python
# scripts/x.py` puts `scripts/` on sys.path, not the repository root.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import sys

from rse.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["ingest", *sys.argv[1:]]))
