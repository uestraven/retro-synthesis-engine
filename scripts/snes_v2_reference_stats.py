#!/usr/bin/env python3
"""Precompute the frozen reference statistics for sampling selection.

Run on this desktop, shipped in the bundle: the pod has no release.

Per §3 of the frozen procedure: 60 TRAINING tracks, shuffle seed 0,
**round-tripped through G3** so the generators are not charged for the
representation's own quantisation. Metrics are per-excerpt, then median.
Validation and test are not used.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rse.snes_lm.release import Release  # noqa: E402
from rse.snes_v2 import codec as cx  # noqa: E402
from rse.snes_v2 import delta as dx  # noqa: E402
from rse.snes_v2 import grammar as gx  # noqa: E402
from rse.snes_v2 import statistics as sx  # noqa: E402

N_TRACKS = 60
SHUFFLE_SEED = 0
GRID_HZ = 4000.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", default="~/snesmdb-releases/snesmdb-50game-rc1")
    parser.add_argument("--out", default=str(
        Path.home() / "workplace/v2_artifacts/reference_stats.json"))
    args = parser.parse_args()

    raw = json.loads((Path.home() /
        "workplace/v2_artifacts/delta_vocab_k512.json").read_text())
    vocab = gx.Vocab(delta=dx.DeltaCodec(
        direct={int(k): v for k, v in raw["direct"].items()},
        n_observed=raw["n_distinct_train"]))

    release = Release(args.release)
    tracks = release.tracks(split="train")
    random.Random(SHUFFLE_SEED).shuffle(tracks)
    tracks = tracks[:N_TRACKS]

    per_track = []
    names = []
    for track in tracks:
        events = release.events(track)
        if not events:
            continue
        groups, _ = cx.encode(events, grid_hz=GRID_HZ)
        parsed, _ = gx.parse(gx.serialize(groups, vocab), vocab)
        notes = gx.notes_from_groups(parsed)
        stats = sx.excerpt_statistics(notes, grid_hz=GRID_HZ)
        if stats is None:
            continue
        per_track.append(stats)
        names.append(str(getattr(track, "track_id", None) or ""))

    reference = sx.aggregate(per_track)
    payload = {
        "schema": "snes_v2.reference_stats.v1",
        "population": f"{len(per_track)} training tracks, shuffle seed "
                      f"{SHUFFLE_SEED}, round-tripped through G3",
        "n_tracks": len(per_track),
        "tracks": names,
        "grid_hz": GRID_HZ,
        "aggregation": "per-excerpt statistics, then median; intervals pooled",
        "reference": reference,
    }
    Path(args.out).write_text(json.dumps(payload, indent=1))
    print(f"  {len(per_track)} tracks")
    print(f"  median attacks/sec        {reference['attacks_per_sec']:.4f}")
    print(f"  median simultaneity       {reference['mean_simultaneity']:.4f}")
    print(f"  interval histogram bins   {len(reference['interval_histogram'])}")
    print(f"  written to {args.out}")


if __name__ == "__main__":
    main()
