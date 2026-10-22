#!/usr/bin/env python3
"""Thin wrapper around `rse ingest`."""

from __future__ import annotations

import sys

from rse.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["ingest", *sys.argv[1:]]))
