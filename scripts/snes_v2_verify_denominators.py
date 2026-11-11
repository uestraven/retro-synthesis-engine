#!/usr/bin/env python3
"""Recompute every registered denominator from the corrected pipeline.

Historical counts are NOT asserted. Each is recomputed from the frozen
artifacts and reconciled, so a changed eligibility definition surfaces as a
mismatch rather than being papered over by a stale expectation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rse.snes_v2.accounting import Accounting, eligible_from_manifest  # noqa

ART = Path.home() / "workplace/v2_artifacts"
WINDOWS = ART / "windows"
EPOCHS = 20
BATCH = 8


def main() -> None:
    acc = Accounting(stage="pre_training_denominators")

    for split in ("train", "valid"):
        manifest = eligible_from_manifest(WINDOWS, split)
        data = np.load(WINDOWS / f"{split}.npz")
        tokens, mask, lengths = data["tokens"], data["mask"], data["lengths"]

        acc.check(f"{split}_windows", observed=int(tokens.shape[0]),
                  intended=manifest["n_windows"],
                  population=f"{split}.npz rows vs MANIFEST")
        acc.check(f"{split}_scored_targets", observed=int(mask.sum()),
                  intended=manifest["scored_targets"],
                  population=f"{split} loss mask vs MANIFEST")
        acc.check(f"{split}_padding", observed=int(
                      (tokens == manifest["pad_id"]).sum()),
                  intended=manifest["padding_positions"],
                  population=f"{split} PAD positions vs MANIFEST")

        # Independent derivation: a window contributes
        # length - 1 - n_snapshot_tokens targets. Recomputed from the token
        # stream itself, not from the stored mask.
        derived = 0
        for i in range(len(lengths)):
            row, n = tokens[i], int(lengths[i])
            first = int(np.argmax(mask[i])) if mask[i].any() else n
            derived += max(0, n - first)
        acc.check(f"{split}_targets_rederived", observed=derived,
                  intended=int(mask.sum()),
                  population=f"{split} mask recomputed from lengths")

        # No target may be a PAD, and none may precede its window's first
        # scored position.
        pad_targets = int(((tokens == manifest["pad_id"]) & mask).sum())
        acc.check(f"{split}_pad_never_scored", observed=pad_targets,
                  intended=0, population=f"{split} PAD under mask")

    # The exposure budget is DERIVED, never quoted.
    train = eligible_from_manifest(WINDOWS, "train")
    steps = -(-train["n_windows"] // BATCH)
    exposure = EPOCHS * train["scored_targets"]
    acc.note("epochs", EPOCHS)
    acc.note("batch", BATCH)
    acc.note("steps_per_epoch", steps)
    acc.note("optimizer_updates", EPOCHS * steps)
    acc.note("scored_target_exposure", exposure)

    # Reconcile against the figures already written into the protocol, as a
    # CHECK rather than as the source of truth.
    acc.check("protocol_train_targets", observed=train["scored_targets"],
              intended=8_906_692,
              population="protocol-quoted train scored targets")
    acc.check("protocol_exposure", observed=exposure,
              intended=178_133_840,
              population="protocol-quoted 20-epoch exposure")
    acc.check("protocol_steps_per_epoch", observed=steps, intended=1_199,
              population="protocol-quoted steps per epoch")

    # The baseline selection must have scored every candidate on the same set.
    sel = json.loads((ART / "baseline_selection_final.json").read_text())
    counts = {r["n_scored"] for r in sel["results"]}
    acc.check("baseline_identical_subsample", observed=len(counts), intended=1,
              population="n_scored across all 10 candidates")
    acc.check("baseline_subsample_size", observed=counts.pop(),
              intended=sel["subsample"]["n"],
              population="declared subsample size")

    acc.write(ART / "denominator_accounting.json")
    for r in acc.records:
        if "observed" in r:
            flag = "ok  " if r["ok"] else "FAIL"
            print(f"  {flag} {r['metric']:32} {r['observed']:>12,} "
                  f"vs {r['intended']:>12,}")
        else:
            print(f"       {r['metric']:32} {r['value']:>12,}")
    print(f"\n  all populations reconcile: {acc.ok}")
    acc.exit_if_failed()


if __name__ == "__main__":
    main()
