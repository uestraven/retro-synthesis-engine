"""Operational denominator assertions. These run in production, not in tests.

This project has produced **three separate denominator failures**, each giving
a plausible result with a clean verdict line:

1. Level 1 reported 1,257,064 serialized tokens where the loss consumed
   1,255,751 scored targets -- a difference of exactly one per window.
2. The A/B harness filtered decoded notes to `onset < 25.0`, dropping an
   attack that quantised to exactly 25.0, and reported it as a codec loss.
3. `legal_at` had no snapshot handling, so a baseline-selection run scored
   1,298 of 8,000 intended targets -- 16% -- and printed "SELECTED" anyway.

A metric's formula can be correct, its output plausible and its verdict clean
while the population it measured is wrong. **A passing unit test with the
wrong expected denominator is not a safeguard**, so these assertions:

- run on the normal execution path, not under `pytest`;
- fail the process with a nonzero exit status;
- name the mismatched population in the diagnostic;
- are never satisfied by a hard-coded historical count. Expected values are
  recomputed from the frozen manifests.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class PopulationMismatch(RuntimeError):
    """The scoring procedure did not consume the intended observations."""


@dataclass
class Accounting:
    """A machine-readable record of every registered metric's population."""

    stage: str
    records: list[dict[str, Any]] = field(default_factory=list)

    def check(self, name: str, *, observed: int, intended: int,
              population: str, strict: bool = True) -> None:
        """Assert that `observed` equals `intended`, and record either way."""
        ok = observed == intended
        self.records.append({
            "metric": name, "population": population,
            "observed": observed, "intended": intended,
            "delta": observed - intended, "ok": ok,
        })
        if not ok and strict:
            raise PopulationMismatch(
                f"{self.stage}/{name}: consumed {observed:,} observations but "
                f"{intended:,} were eligible in {population} "
                f"(delta {observed - intended:+,})")

    def note(self, name: str, value: Any) -> None:
        self.records.append({"metric": name, "value": value})

    @property
    def ok(self) -> bool:
        return all(r.get("ok", True) for r in self.records)

    def write(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({
            "schema": "snes_v2.accounting.v1",
            "stage": self.stage,
            "all_populations_reconcile": self.ok,
            "records": self.records,
        }, indent=1))

    def exit_if_failed(self) -> None:
        """Fail the RUN, not just a test. Preserves a nonzero exit status."""
        if not self.ok:
            bad = [r for r in self.records if not r.get("ok", True)]
            for r in bad:
                print(f"  POPULATION MISMATCH {self.stage}/{r['metric']}: "
                      f"observed {r['observed']:,} vs intended "
                      f"{r['intended']:,} in {r['population']}",
                      file=sys.stderr)
            sys.exit(2)


def eligible_from_manifest(window_dir: str | Path, split: str) -> dict[str, int]:
    """Recompute eligibility from the frozen window manifest.

    Deliberately reads the manifest rather than trusting a count quoted in
    prose. If the snapshot-mask definition changes, this changes with it.
    """
    manifest = json.loads((Path(window_dir) / "MANIFEST.json").read_text())
    entry = manifest["splits"][split]
    return {
        "n_windows": entry["n_windows"],
        "scored_targets": entry["scored_targets"],
        "padding_positions": entry["padding_positions"],
        "context": manifest["context"],
        "pad_id": manifest["pad_id"],
    }
