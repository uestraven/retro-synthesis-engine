#!/usr/bin/env python3
"""Milestone 2, decision 1: does 4000 Hz cost sequence length? Measured.

The codec's **action count is not the model's sequence length**. Actions become
tokens only under a chosen serialization, so the question "does 4000 Hz
materially lengthen sequences" cannot be answered from the audit's 1.916
actions per source event. Three candidate serializations are measured here
against the Level 1 representation.

**Why the grid should NOT lengthen sequences:** groups carry one positive time
delta each, and empty ticks are never enumerated. A finer grid changes the
delta's numeric range, not the number of groups. That is the hypothesis; this
script tests it by also measuring at 250 Hz and 1000 Hz.

Candidates, all representing the SAME validated grouped events:

- **A, explicit ends** — `DELTA` per group, then `START PITCH` per start,
  `END_SLOT` per end, `PITCHMOD_SLOT PITCH` per pitch change. Lifetimes are
  explicit in the token stream.
- **B, duration-bearing starts** — `DELTA`, then `NOTE PITCH DUR` per start
  and `PITCHMOD_SLOT PITCH` per change. No end tokens; the decoder derives
  ends. Closest in shape to the Level 1 four-token event.
- **C, compound** — `DELTA` per group plus one factored token per action,
  fields predicted by separate heads (the Compound Word approach).

**What each candidate does NOT decide** is which constraints the model must
learn versus which the decoder guarantees. Under A and B the decoder still
allocates voices, so no candidate here gives the model credit for
eight-voice occupancy. That separation is decision 3 and is not settled by
token counts.

Level 1 comparison is reported honestly: it used 4 tokens per attack and
**dropped legato entirely**, so the new representation carries 277,329 events
Level 1 could not express. A pure tokens-per-track comparison flatters Level 1.

Train split only.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rse.snes_lm.release import Release  # noqa: E402
from rse.snes_v2 import codec as cx  # noqa: E402

GRIDS = (250.0, 1000.0, 4000.0)
CONTEXTS = (1024, 2048, 4096)


def count_tokens(groups) -> dict[str, int]:
    """Token counts per candidate, plus the delta alphabet actually used."""
    a = b = c = 0
    deltas: set[int] = set()
    for index, group in enumerate(groups):
        if index > 0 or group.delta:
            deltas.add(group.delta)
        starts = sum(1 for x in group.actions if x.kind == "start")
        ends = sum(1 for x in group.actions if x.kind == "end")
        pitches = sum(1 for x in group.actions if x.kind == "pitch")
        a += 1 + 2 * starts + 1 * ends + 2 * pitches
        b += 1 + 3 * starts + 2 * pitches
        c += 1 + starts + ends + pitches
    return {"A": a, "B": b, "C": c, "distinct_deltas": len(deltas),
            "max_delta": max(deltas) if deltas else 0}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", default="~/snesmdb-releases/snesmdb-50game-rc1")
    parser.add_argument("--n-tracks", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=str(Path.home() /
                        "workplace/v2_artifacts/token_budget.json"))
    args = parser.parse_args()

    release = Release(args.release)
    tracks = release.tracks(split="train")
    random.Random(args.seed).shuffle(tracks)
    tracks = tracks[:args.n_tracks]

    rows = []
    for track in tracks:
        events = release.events(track)
        if not events:
            continue
        attacks = sum(1 for e in events if e.get("origin") != "legato_step")
        span = max(float(e["onset_sec"]) for e in events) or 1e-9
        row = {"n_events": len(events), "n_attacks": attacks, "span_sec": span,
               # Level 1: four tokens per attack, legato NOT representable.
               "level1_tokens": 4 * attacks + 2, "grids": {}}
        for grid in GRIDS:
            groups, _ = cx.encode(events, grid_hz=grid)
            row["grids"][str(grid)] = count_tokens(groups) | {
                "n_groups": len(groups)}
        rows.append(row)

    report = {
        "schema": "snes_v2.token_budget.v1",
        "split": "train",
        "n_tracks": len(rows),
        "candidates": {
            "A": "DELTA + (START PITCH) + (END_SLOT) + (PITCHMOD_SLOT PITCH)",
            "B": "DELTA + (NOTE PITCH DUR) + (PITCHMOD_SLOT PITCH), no ends",
            "C": "DELTA + one factored token per action",
        },
        "level1_note": "4 tokens per attack, legato not representable",
        "per_track": rows,
    }

    print(f"  {len(rows)} train tracks; Level 1 median "
          f"{st.median(r['level1_tokens'] for r in rows):,.0f} tokens/track\n")
    print(f"  {'grid':>7} {'cand':>5} {'median':>9} {'p95':>9} {'max':>9} "
          f"{'vs L1':>7} {'tok/s':>7} {'sec in 1024':>12} {'deltas':>7}")
    summary = {}
    for grid in GRIDS:
        key = str(grid)
        for cand in ("A", "B", "C"):
            toks = [r["grids"][key][cand] for r in rows]
            l1 = [r["level1_tokens"] for r in rows]
            ratio = sum(toks) / sum(l1)
            per_sec = [r["grids"][key][cand] / r["span_sec"] for r in rows]
            median_per_sec = st.median(per_sec)
            summary[f"{key}:{cand}"] = {
                "median_tokens": st.median(toks),
                "p95_tokens": sorted(toks)[int(0.95 * len(toks))],
                "max_tokens": max(toks),
                "ratio_vs_level1": ratio,
                "median_tokens_per_sec": median_per_sec,
                "seconds_in_1024": 1024 / median_per_sec,
                "distinct_deltas_max": max(
                    r["grids"][key]["distinct_deltas"] for r in rows),
            }
            s = summary[f"{key}:{cand}"]
            print(f"  {grid:7.0f} {cand:>5} {s['median_tokens']:9,.0f} "
                  f"{s['p95_tokens']:9,} {s['max_tokens']:9,} "
                  f"{ratio:6.2f}x {median_per_sec:7.1f} "
                  f"{s['seconds_in_1024']:11.1f}s "
                  f"{s['distinct_deltas_max']:7,}")
    report["summary"] = summary
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1))

    print("\n  does the grid lengthen sequences? (candidate C, median tokens)")
    for grid in GRIDS:
        print(f"    {grid:7.0f} Hz: {summary[f'{grid}:C']['median_tokens']:9,.0f}"
              f"   distinct deltas up to {summary[f'{grid}:C']['distinct_deltas_max']:,}")
    print(f"\n  written to {args.out}")


if __name__ == "__main__":
    main()
