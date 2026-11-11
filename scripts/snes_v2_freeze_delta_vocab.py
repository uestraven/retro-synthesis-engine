#!/usr/bin/env python3
"""Freeze the C2 k=512 delta vocabulary, and verify it on all 1,740 tracks.

Guarantee 1: **exactness across the complete population.** Every delta
appearing anywhere in the 1,740-track music population is encoded and decoded
and checked for integer equality -- including the 285 validation and 266 test
values absent from the training dictionary, which must escape exactly rather
than fall back to a neighbour.

The dictionary is fitted on **train only**, by frequency with ties broken on
delta value so the fit is deterministic and reproducible. It is never extended
using validation or test data; those splits only ever exercise the escape.

Guarantees 2 and 3 -- unambiguous escape decoding and strictly positive time
progression with a finite digit budget -- are covered by
`tests/test_snes_v2_delta.py` and re-asserted here against real data.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rse.snes_lm.release import Release  # noqa: E402
from rse.snes_v2 import codec as cx  # noqa: E402
from rse.snes_v2 import delta as dx  # noqa: E402

GRID_HZ = 4000.0
K = 512


def deltas_of(release: Release, split: str) -> Counter:
    counts: Counter = Counter()
    for track in release.tracks(split=split):
        events = release.events(track)
        if not events:
            continue
        groups, _ = cx.encode(events, grid_hz=GRID_HZ)
        counts.update(g.delta for g in groups[1:])
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", default="~/snesmdb-releases/snesmdb-50game-rc1")
    parser.add_argument("--out", default=str(Path.home() /
                        "workplace/v2_artifacts/delta_vocab_k512.json"))
    args = parser.parse_args()

    release = Release(args.release)
    print("  fitting on train only ...", flush=True)
    train = deltas_of(release, "train")
    codec = dx.fit(train.elements(), k=K)
    print(f"    {len(train):,} distinct train deltas, "
          f"{codec.n_direct} direct, vocab {codec.vocab_size}", flush=True)

    checked = escaped = non_canonical = 0
    per_split = {}
    for split in ("train", "valid", "test"):
        counts = train if split == "train" else deltas_of(release, split)
        unseen = [d for d in counts if d not in codec.direct]
        for d in counts:
            tokens = codec.encode(d)
            value, consumed = codec.decode(tokens)
            assert value == d, (split, d, value)          # guarantee 1
            assert consumed == len(tokens), (split, d)
            assert value > 0                               # guarantee 3
            assert len(tokens) <= codec.MAX_TOKENS_PER_DELTA
            if not codec.is_canonical(tokens):
                non_canonical += 1
            checked += 1
            if len(tokens) > 1:
                escaped += 1
        occurrences = sum(counts.values())
        escape_occ = sum(counts[d] for d in unseen)
        per_split[split] = {
            "n_distinct": len(counts),
            "n_occurrences": occurrences,
            "n_distinct_escaping": len(unseen),
            "n_occurrences_escaping": escape_occ,
            "escape_occurrence_rate": escape_occ / occurrences if occurrences else 0.0,
            "max_delta": max(counts) if counts else 0,
        }
        print(f"    {split:6}: {len(counts):5,} distinct, "
              f"{occurrences:9,} occurrences, "
              f"{100 * per_split[split]['escape_occurrence_rate']:6.3f}% escape, "
              f"max {max(counts):,}", flush=True)

    # Guarantee 2, on real data: a truncated escape is never a silent zero.
    sample = max(d for d in train if d not in codec.direct)
    full = codec.encode(sample)
    for cut in range(1, len(full)):
        try:
            codec.decode(full[:cut])
            raise AssertionError(f"truncated escape {full[:cut]} accepted")
        except dx.IncompleteDelta:
            pass
    print(f"  guarantee 2: truncations of {full} all rejected", flush=True)

    payload = {
        "schema": "snes_v2.delta_vocab.v1",
        "frozen_utc": "2026-11-11",
        "grid_hz": GRID_HZ,
        "design": "C2 exact hybrid",
        "k": K,
        "fit_split": "train",
        "fit_rule": "descending frequency, ties broken by ascending delta value",
        "radix": dx.RADIX,
        "max_escape_digits": dx.MAX_ESCAPE_DIGITS,
        "max_tokens_per_delta": codec.MAX_TOKENS_PER_DELTA,
        "vocab_size_delta_layer": codec.vocab_size,
        "n_direct": codec.n_direct,
        "n_distinct_train": len(train),
        "guarantees": {
            "exact_on_full_population": True,
            "n_distinct_deltas_checked": checked,
            "n_non_canonical_encodings_produced": non_canonical,
            "escape_unambiguous_and_truncation_rejected": True,
            "delta_strictly_positive": True,
            "token_budget_per_delta": codec.MAX_TOKENS_PER_DELTA,
        },
        "per_split": per_split,
        # The frozen dictionary itself, delta value -> token index.
        "direct": {str(d): i for d, i in codec.direct.items()},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=1))
    print(f"\n  {checked:,} distinct deltas verified exact across all three "
          f"splits; {non_canonical} non-canonical encodings produced")
    print(f"  frozen dictionary written to {args.out}")


if __name__ == "__main__":
    main()
