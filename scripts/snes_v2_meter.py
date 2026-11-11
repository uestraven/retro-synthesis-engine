#!/usr/bin/env python3
"""Is a bar length measurable, or only assumed? (RSE v2, step 3)

RSE v2 proposes an explicit A-A'-B-A phrase plan. That needs a unit. The
release supplies two row grids -- the driver's own timer tick, a hardware
measurement with 80 distinct rates, and a 100 Hz uniform grid that
`DATASET.md` labels "a convention, stated as one" -- but **no time signature,
no bar count, and no rows-per-bar field anywhere**. Only 573 of 2,228 tracks
carry a software divisor.

So before a phrase plan is registered, the question is whether a bar-scale
periodicity can be *recovered from the onsets* or whether it has to be
asserted. This script measures it directly, on TRAIN only.

Method: bin attack onsets into 10 ms cells (the uniform grid), then take the
normalised autocorrelation of the onset-count sequence and find the strongest
lag in a plausible bar range. A track whose onsets really are organised in
bars should show a clear peak at the bar length and weaker peaks at its
multiples.

What a negative result would mean: if most tracks show no dominant lag, then
"eight-bar melody" has no measured referent in this corpus, and v2 must either
define phrases in ROWS, restrict to the divisor-bearing subset, or register
rows-per-bar as an explicit assumption with its own validation. The 16
rows/bar figure is a TRACKER convention and is not verified for SNES drivers.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics as st
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rse.snes_lm.release import Release  # noqa: E402

BIN_SEC = 0.010
ANALYSE_SEC = 60.0
#: Plausible bar lengths. Below 0.3 s is a pulse, above 8 s is a section.
MIN_LAG_SEC, MAX_LAG_SEC = 0.3, 8.0
#: A peak this strong is treated as a real periodicity rather than noise.
STRONG_PEAK = 0.30


def analyse(events) -> dict | None:
    onsets = [float(e["onset_sec"]) for e in events
              if e.get("origin") == "key_on" and float(e["onset_sec"]) < ANALYSE_SEC]
    if len(onsets) < 40:
        return None
    n = int(ANALYSE_SEC / BIN_SEC)
    counts = np.zeros(n)
    for o in onsets:
        counts[min(int(o / BIN_SEC), n - 1)] += 1
    counts -= counts.mean()
    if not counts.any():
        return None
    full = np.correlate(counts, counts, mode="full")[n - 1:]
    if full[0] <= 0:
        return None
    ac = full / full[0]
    lo, hi = int(MIN_LAG_SEC / BIN_SEC), int(MAX_LAG_SEC / BIN_SEC)
    window = ac[lo:hi]
    if not len(window):
        return None
    best = int(np.argmax(window)) + lo
    peak = float(window.max())
    lag = best * BIN_SEC
    # Does the lag's double also peak? A real bar implies periodic multiples.
    double = best * 2
    harmonic = float(ac[double]) if double < len(ac) else None
    # Median nonzero inter-onset gap: the pulse, for a rows-per-bar estimate.
    gaps = [b - a for a, b in zip(sorted(onsets), sorted(onsets)[1:]) if b - a > 1e-6]
    pulse = st.median(gaps) if gaps else None
    return {
        "best_lag_sec": round(lag, 3),
        "peak_autocorr": round(peak, 4),
        "harmonic_autocorr": round(harmonic, 4) if harmonic is not None else None,
        "strong": peak >= STRONG_PEAK,
        "pulse_sec": round(pulse, 4) if pulse else None,
        "lag_over_pulse": round(lag / pulse, 2) if pulse else None,
        "n_onsets": len(onsets),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", default="~/snesmdb-releases/snesmdb-50game-rc1")
    parser.add_argument("--n-tracks", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=str(Path.home() /
                        "workplace/v2_artifacts/meter.json"))
    args = parser.parse_args()

    release = Release(args.release)
    tracks = release.tracks(split="train")       # TRAIN ONLY
    random.Random(args.seed).shuffle(tracks)

    rows = []
    for track in tracks[:args.n_tracks]:
        r = analyse(release.events(track))
        if r:
            rows.append(r)

    strong = [r for r in rows if r["strong"]]
    report = {
        "schema": "snes_v2.meter.v1",
        "split": "train",
        "n_analysed": len(rows),
        "n_strong_periodicity": len(strong),
        "fraction_strong": len(strong) / len(rows) if rows else None,
        "strong_peak_threshold": STRONG_PEAK,
        "bin_sec": BIN_SEC,
        "lag_search_sec": [MIN_LAG_SEC, MAX_LAG_SEC],
        "median_peak_autocorr": st.median(r["peak_autocorr"] for r in rows) if rows else None,
        "median_best_lag_sec": st.median(r["best_lag_sec"] for r in rows) if rows else None,
        "median_pulse_sec": st.median(r["pulse_sec"] for r in rows
                                      if r["pulse_sec"]) if rows else None,
        "median_lag_over_pulse": st.median(r["lag_over_pulse"] for r in rows
                                           if r["lag_over_pulse"]) if rows else None,
        "note": "No ground-truth meter exists in the release. A strong "
                "autocorrelation peak is evidence of periodicity, not proof "
                "of a notated bar.",
        "per_track": rows,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1))

    print(f"  analysed {len(rows)} train tracks")
    print(f"  strong periodicity (peak >= {STRONG_PEAK}): {len(strong)} "
          f"({100*len(strong)/len(rows):.1f}%)")
    print(f"  median peak autocorrelation {report['median_peak_autocorr']:.3f}")
    print(f"  median best lag {report['median_best_lag_sec']:.3f} s")
    print(f"  median pulse   {report['median_pulse_sec']:.4f} s")
    print(f"  median lag/pulse {report['median_lag_over_pulse']:.2f}  "
          f"(a tracker convention would be 16)")
    if strong:
        lags = sorted(r["best_lag_sec"] for r in strong)
        print(f"  strong-track lags: p10 {lags[len(lags)//10]:.2f} s  "
              f"median {st.median(lags):.2f} s  p90 {lags[9*len(lags)//10]:.2f} s")
    print(f"\n  written to {args.out}")


if __name__ == "__main__":
    main()
