#!/usr/bin/env python3
"""Decision 2: C1 exact vocabulary versus C2 exact hybrid, on full train.

Both designs are exact. Neither quantises a delta, including long rests --
those carry phrase-boundary and silence information, and there is no reason to
spend the codec's certified 0.125 ms bound at the token layer.

**Both need the escape machinery**, because a vocabulary fitted on training
data cannot cover validation, test, or generated performances. So the contest
is `N_distinct` against `k`, with escape frequency as the price.

Selection rule, declared before measuring: prefer C1 for simplicity unless C2
gives a substantial vocabulary reduction WITHOUT materially reducing musical
context. Vocabulary size alone is not the objective -- an escape-heavy scheme
trades embeddings for sequence length and makes long-range structure harder to
model.

Reported for each design: distinct deltas, head coverage, total vocabulary,
tokens per second, seconds per 1024 tokens, exact round-trip, and escape
validity. Fitted on train; coverage also reported on validation and test to
show what the fixed vocabulary misses.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rse.snes_lm.release import Release  # noqa: E402
from rse.snes_v2 import codec as cx  # noqa: E402
from rse.snes_v2 import delta as dx  # noqa: E402

GRID_HZ = 4000.0
HEAD_SIZES = (128, 256, 512)


def collect(release: Release, split: str):
    """Per-track group deltas and span, at the frozen 4000 Hz grid."""
    out = []
    for track in release.tracks(split=split):
        events = release.events(track)
        if not events:
            continue
        groups, _ = cx.encode(events, grid_hz=GRID_HZ)
        deltas = [g.delta for g in groups[1:]]
        n_actions = sum(len(g.actions) for g in groups)
        span = max(float(e["onset_sec"]) for e in events) or 1e-9
        out.append({"deltas": deltas, "n_groups": len(groups),
                    "n_actions": n_actions, "span": span})
    return out


def evaluate(name: str, codec: dx.DeltaCodec, tracks) -> dict:
    """Candidate C token totals: one factored token per action, plus deltas."""
    per_sec, totals, escapes, total_deltas = [], [], 0, 0
    for t in tracks:
        delta_tokens = sum(codec.n_tokens(d) for d in t["deltas"])
        escapes += sum(1 for d in t["deltas"] if d not in codec.direct)
        total_deltas += len(t["deltas"])
        tokens = delta_tokens + t["n_actions"]
        totals.append(tokens)
        per_sec.append(tokens / t["span"])
    median_rate = st.median(per_sec)
    return {
        "design": name,
        "n_direct": codec.n_direct,
        "vocab_delta_layer": codec.vocab_size,
        "median_tokens": st.median(totals),
        "p95_tokens": sorted(totals)[int(0.95 * len(totals))],
        "max_tokens": max(totals),
        "median_tokens_per_sec": median_rate,
        "seconds_in_1024": 1024 / median_rate,
        "escape_rate": escapes / total_deltas if total_deltas else 0.0,
        "n_escapes": escapes,
        "n_deltas": total_deltas,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", default="~/snesmdb-releases/snesmdb-50game-rc1")
    parser.add_argument("--out", default=str(Path.home() /
                        "workplace/v2_artifacts/delta_compare.json"))
    args = parser.parse_args()

    release = Release(args.release)
    print(f"  collecting deltas at {GRID_HZ:.0f} Hz ...", flush=True)
    train = collect(release, "train")
    print(f"    train {len(train)} tracks", flush=True)

    all_deltas = [d for t in train for d in t["deltas"]]
    counts = Counter(all_deltas)
    print(f"  full-train deltas: {len(all_deltas):,} occurrences, "
          f"{len(counts):,} distinct, max {max(counts):,}", flush=True)

    ordered = sorted(counts, key=lambda d: (-counts[d], d))
    total = sum(counts.values())
    coverage = {}
    for k in HEAD_SIZES:
        covered = sum(counts[d] for d in ordered[:k])
        coverage[str(k)] = covered / total
        print(f"    top {k:4}: covers {100 * covered / total:.4f}% of occurrences")

    designs = [("C1 exact vocabulary", dx.fit(all_deltas))]
    for k in HEAD_SIZES:
        designs.append((f"C2 hybrid k={k}", dx.fit(all_deltas, k=k)))

    # Exactness and escape validity, verified on every distinct train delta.
    for name, codec in designs:
        for d in counts:
            tokens = codec.encode(d)
            got, nxt = codec.decode(tokens)
            assert got == d and nxt == len(tokens), (name, d)
        # Truncation must be verified on a delta that actually ESCAPES.
        # Using max(counts) was wrong: under C1 every observed delta has a
        # direct token, so its encoding is one token and slicing it to [:1]
        # yields the complete encoding with nothing to detect.
        escaping = max(counts) + 1          # unseen under every design
        assert escaping not in codec.direct
        full = codec.encode(escaping)
        assert len(full) > 1, (name, full)
        assert codec.decode(full) == (escaping, len(full))
        for truncated in range(1, len(full)):
            try:
                codec.decode(full[:truncated])
                raise AssertionError(
                    f"{name}: truncated escape {full[:truncated]} not detected")
            except dx.IncompleteDelta:
                pass
    print("  exact round-trip verified on every distinct train delta; "
          "truncated escapes rejected", flush=True)

    results = [evaluate(name, codec, train) for name, codec in designs]
    print(f"\n  {'design':24} {'direct':>7} {'vocab':>7} {'median':>8} "
          f"{'tok/s':>7} {'sec/1024':>9} {'escape%':>9}")
    for r in results:
        print(f"  {r['design']:24} {r['n_direct']:7,} {r['vocab_delta_layer']:7,} "
              f"{r['median_tokens']:8,.0f} {r['median_tokens_per_sec']:7.1f} "
              f"{r['seconds_in_1024']:8.1f}s {100 * r['escape_rate']:8.4f}%")

    # What a train-fitted vocabulary misses on held-out splits.
    held = {}
    for split in ("valid", "test"):
        tracks = collect(release, split)
        ds = [d for t in tracks for d in t["deltas"]]
        unseen = sum(1 for d in ds if d not in counts)
        held[split] = {"n_deltas": len(ds), "n_unseen": unseen,
                       "unseen_rate": unseen / len(ds) if ds else 0.0,
                       "n_distinct": len(set(ds)),
                       "n_distinct_unseen": len(set(ds) - set(counts))}
        print(f"  {split}: {unseen:,} of {len(ds):,} delta occurrences unseen in "
              f"train ({100 * held[split]['unseen_rate']:.4f}%), "
              f"{held[split]['n_distinct_unseen']:,} distinct values")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "schema": "snes_v2.delta_compare.v1",
        "grid_hz": GRID_HZ,
        "radix": dx.RADIX,
        "max_escape_digits": dx.MAX_ESCAPE_DIGITS,
        "n_delta_occurrences": len(all_deltas),
        "n_distinct_train": len(counts),
        "max_delta_train": max(counts),
        "head_coverage": coverage,
        "designs": results,
        "held_out_unseen": held,
    }, indent=1))
    print(f"\n  written to {args.out}")


if __name__ == "__main__":
    main()
