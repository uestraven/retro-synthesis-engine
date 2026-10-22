#!/usr/bin/env python3
"""Thin wrapper around `rse fetch` for people who prefer scripts to entry points."""

from __future__ import annotations

import sys

from rse.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["fetch", *sys.argv[1:]]))
