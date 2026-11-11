#!/usr/bin/env python3
"""Pre-commit census for the 10-cent pitch grid. Train split only.

The delta decision turned on a frequency distribution, not on vocabulary size:
C1's tail had a median of 3 occurrences per embedding and 1,460 singletons, so
thousands of parameters could not be estimated. A 10-cent pitch grid gives
about ten bins per semitone, and whether that fragments into an unlearnable
tail is an empirical question about THIS corpus.

    q = ROUND_HALF_UP(10 * midi_note),  decode p = q / 10

`ROUND_HALF_UP` via `Decimal`, matching the codec's `_step`, rather than
Python's `round`, which is banker's rounding and would break ties
inconsistently across the two layers.

Six measurements, as specified:

1. distinct 10-cent tokens -- the actual vocabulary
2. occurrences per token -- sparse-embedding detection
3. tokens under 10 occurrences -- the C1 comparison
4. quantisation error by pitch status -- verifies the 5-cent bound
5. repeated-note grid switching -- adjacent-bin alternation from small
   measured tuning fluctuation, which would split one musical pitch across
   two tokens and inflate the vocabulary with no musical content
6. context length -- the one-token encoding's real cost

Pitch status is preserved independently and is NOT folded into the grid:
unpitched stays unpitched, unrecoverable keeps its uncertainty, and a
pitch-modulated event keeps its status rather than pretending one quantised
scalar reconstructs the modulation.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from collections import Counter, defaultdict
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rse.snes_lm.release import Release  # noqa: E402
from rse.snes_v2 import codec as cx  # noqa: E402
from rse.snes_v2 import delta as dx  # noqa: E402

GRID_HZ = 4000.0
#: Bins per semitone. 10 gives a 5-cent maximum error.
BINS_PER_SEMITONE = 10
#: Two attacks closer than this are musically the same pitch; if they land in
#: different bins, that is grid switching rather than a melodic interval.
SAME_PITCH_SEMITONES = 0.10


def quantise(midi_note: float) -> int:
    """Deterministic, half-up, matching the codec's time quantiser."""
    scaled = Decimal(str(midi_note)) * BINS_PER_SEMITONE
    return int(scaled.to_integral_value(rounding=ROUND_HALF_UP))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", default="~/snesmdb-releases/snesmdb-50game-rc1")
    parser.add_argument("--out", default=str(Path.home() /
                        "workplace/v2_artifacts/pitch_census.json"))
    args = parser.parse_args()

    release = Release(args.release)
    counts: Counter = Counter()
    err_by_status: dict[str, list[float]] = defaultdict(list)
    status_counts: Counter = Counter()
    n_pitched = n_unpitched = 0
    same_pitch_pairs = switched = 0
    delta_codec_tokens = 0
    token_totals = []
    spans = []

    print("  censusing train split ...", flush=True)
    for index, track in enumerate(release.tracks(split="train")):
        events = release.events(track)
        if not events:
            continue
        # per-voice consecutive pitched attacks, for grid switching
        by_voice: dict[int, list[float]] = defaultdict(list)
        for e in events:
            status = e.get("pitch_status")
            status_counts[status] += 1
            v = e.get("midi_note")
            if v is None:
                n_unpitched += 1
                continue
            value = float(v)
            q = quantise(value)
            counts[q] += 1
            err_by_status[status].append(abs(value - q / BINS_PER_SEMITONE))
            n_pitched += 1
            if e.get("origin") == "key_on":
                by_voice[int(e["voice"])].append(value)
        for pitches in by_voice.values():
            for a, b in zip(pitches, pitches[1:]):
                if abs(a - b) < SAME_PITCH_SEMITONES:
                    same_pitch_pairs += 1
                    if quantise(a) != quantise(b):
                        switched += 1
        # measurement 6: real token cost under the approved flat grammar
        groups, enc = cx.encode(events, grid_hz=GRID_HZ)
        spans.append(max(float(e["onset_sec"]) for e in events) or 1e-9)
        n_end = sum(1 for g in groups for a in g.actions if a.kind == "end")
        n_start = sum(1 for g in groups for a in g.actions if a.kind == "start")
        n_change = sum(1 for g in groups for a in g.actions if a.kind == "pitch")
        # GROUP_START/END per group, END+handle, START+pitch+status,
        # CHANGE+handle+pitch+status, plus the frozen delta tokens.
        token_totals.append(2 * len(groups) + 2 * n_end + 3 * n_start
                            + 4 * n_change)
        if (index + 1) % 400 == 0:
            print(f"    {index + 1}", flush=True)

    ordered = sorted(counts, key=lambda q: (-counts[q], q))
    occurrences = [counts[q] for q in ordered]
    under10 = sum(1 for c in occurrences if c < 10)
    singletons = sum(1 for c in occurrences if c == 1)
    tail = occurrences[512:]

    rates = [t / s for t, s in zip(token_totals, spans)]
    median_rate = st.median(rates)

    report = {
        "schema": "snes_v2.pitch_census.v1",
        "split": "train",
        "bins_per_semitone": BINS_PER_SEMITONE,
        "rounding": "ROUND_HALF_UP on Decimal(str(midi_note)) * 10",
        "declared_max_error_cents": 100 / (2 * BINS_PER_SEMITONE),
        "n_pitched": n_pitched,
        "n_unpitched": n_unpitched,
        "status_counts": dict(status_counts),
        "vocabulary": {
            "n_distinct_tokens": len(counts),
            "min_token": min(counts), "max_token": max(counts),
            "occurrences_median": st.median(occurrences),
            "occurrences_mean": st.mean(occurrences),
            "n_under_10_occurrences": under10,
            "pct_under_10": 100 * under10 / len(counts),
            "n_singletons": singletons,
            "tail_beyond_512_median_occurrences": (
                st.median(tail) if tail else None),
            "tail_beyond_512_size": len(tail),
        },
        "error_by_status": {
            k: {"n": len(v), "mean_cents": 100 * st.mean(v),
                "max_cents": 100 * max(v)}
            for k, v in err_by_status.items() if v},
        "grid_switching": {
            "same_pitch_pairs": same_pitch_pairs,
            "n_switched": switched,
            "rate": switched / same_pitch_pairs if same_pitch_pairs else 0.0,
            "threshold_semitones": SAME_PITCH_SEMITONES,
        },
        "context": {
            "median_tokens_per_track": st.median(token_totals),
            "median_tokens_per_sec": median_rate,
            "seconds_in_1024": 1024 / median_rate,
            "note": "flat grammar: GROUP_START/END per group, END+handle, "
                    "START+pitch+status, CHANGE+handle+pitch+status, plus "
                    "frozen C2 delta tokens",
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1))

    v = report["vocabulary"]
    print(f"\n  pitched {n_pitched:,}  unpitched {n_unpitched:,}")
    print(f"  distinct 10-cent tokens: {v['n_distinct_tokens']:,} "
          f"(range {v['min_token']}..{v['max_token']})")
    print(f"  occurrences per token: median {v['occurrences_median']:,.0f}  "
          f"mean {v['occurrences_mean']:,.1f}")
    print(f"  under 10 occurrences: {v['n_under_10_occurrences']:,} "
          f"({v['pct_under_10']:.1f}%)   singletons {v['n_singletons']:,}")
    print(f"  tail beyond top 512: {v['tail_beyond_512_size']:,} tokens, "
          f"median {v['tail_beyond_512_median_occurrences']} occurrences "
          f"(C1 deltas were median 3)")
    print("  quantisation error by status:")
    for k, e in report["error_by_status"].items():
        print(f"    {str(k):16} n={e['n']:8,}  mean {e['mean_cents']:5.2f}c  "
              f"max {e['max_cents']:5.2f}c")
    g = report["grid_switching"]
    print(f"  grid switching: {g['n_switched']:,} of {g['same_pitch_pairs']:,} "
          f"near-identical pairs ({100 * g['rate']:.2f}%)")
    c = report["context"]
    print(f"  flat-grammar context: {c['median_tokens_per_track']:,.0f} "
          f"tokens/track, {c['median_tokens_per_sec']:.1f} tok/s, "
          f"{c['seconds_in_1024']:.1f}s per 1024 tokens")
    print(f"\n  written to {args.out}")


if __name__ == "__main__":
    main()
