#!/usr/bin/env python3
"""Thin wrapper around `rse inspect`."""

from __future__ import annotations

import sys

from rse.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["inspect", *sys.argv[1:]]))
