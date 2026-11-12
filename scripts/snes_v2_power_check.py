#!/usr/bin/env python3
"""Addendum D2: achievable precision at the ACTUAL analysable pair count.

Run after both evaluation batches exist and **before any listener is
recruited**. It answers one question: at the pair count the 20.0 s window
actually leaves, can the registered criteria resolve anything?

The criteria, unchanged from `V2_MILESTONE3_PROTOCOL.md`:

- practical threshold: adjusted preference >= 0.60
- superiority: the two-way clustered 95% CI lower bound > 0.50
- **both must pass**

Judgements are SIMULATED. No listener is involved and no result about the
model is produced here. The output is the design's resolving power, which is
a property of the pair count and the bootstrap, not of the audio.

The simulation gives each pair and each listener its own random effect,
because the registered bootstrap resamples both and a simulation with
independent judgements would overstate precision -- the pair-level cluster is
exactly what a 6-pair design cannot average away.
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

from rse.snes_v2 import preference as px  # noqa: E402

#: Registered design, from V2_MILESTONE3_PROTOCOL.md.
N_LISTENERS = 20
#: Spread of the per-pair and per-listener random effects, on the preference
#: scale. A pair that genuinely favours one arm, and a listener with a
#: consistent lean, are both real; assuming neither would flatter the design.
PAIR_SD = 0.15
LISTENER_SD = 0.10
TIE_RATE = 0.15
N_SIMULATIONS = 200


def simulate(n_pairs: int, true_preference: float, *, seed: int,
             n_listeners: int = N_LISTENERS) -> list[px.Judgement]:
    rng = random.Random(seed)
    pair_effect = {f"p{i}": rng.gauss(0.0, PAIR_SD) for i in range(n_pairs)}
    listener_effect = {f"L{i}": rng.gauss(0.0, LISTENER_SD)
                       for i in range(n_listeners)}
    out: list[px.Judgement] = []
    for listener, le in listener_effect.items():
        for pair, pe in pair_effect.items():
            p = min(0.999, max(0.001, true_preference + pe + le))
            draw = rng.random()
            if draw < TIE_RATE:
                choice = "tie"
            else:
                choice = "v2" if rng.random() < p else "baseline"
            out.append(px.Judgement(listener=listener, pair=pair,
                                    choice=choice))
    return out


def resolving_power(n_pairs: int, true_preference: float) -> dict:
    widths, lows, points, both_pass = [], [], [], 0
    for sim in range(N_SIMULATIONS):
        judgements = simulate(n_pairs, true_preference, seed=1000 + sim)
        result = px.analyse(judgements, n_replicates=2000,
                            seed=px.BOOTSTRAP_SEED + sim)
        widths.append(result.ci_high - result.ci_low)
        lows.append(result.ci_low)
        points.append(result.adjusted_preference)
        if result.threshold_met and result.superiority_established:
            both_pass += 1
    return {
        "n_pairs": n_pairs,
        "true_preference": true_preference,
        "median_point_estimate": st.median(points),
        "median_ci_width": st.median(widths),
        "median_ci_low": st.median(lows),
        "power_both_criteria": both_pass / N_SIMULATIONS,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=int, required=True,
                        help="the ACTUAL analysable pair count")
    parser.add_argument("--out", default=str(
        Path.home() / "workplace/v2_artifacts/power_check.json"))
    args = parser.parse_args()

    print(f"  registered criteria: preference >= {px.PRACTICAL_THRESHOLD}, "
          f"CI low > 0.50, BOTH required")
    print(f"  {N_LISTENERS} listeners; pair SD {PAIR_SD}, listener SD "
          f"{LISTENER_SD}, tie rate {TIE_RATE}")
    print(f"  {N_SIMULATIONS} simulations per cell; judgements SIMULATED\n")

    rows = []
    # The actual count, and the designed count for contrast.
    for n_pairs in sorted({args.pairs, 8, 24}):
        print(f"  --- {n_pairs} pairs ---")
        for true_preference in (0.50, 0.60, 0.70, 0.80, 0.90):
            row = resolving_power(n_pairs, true_preference)
            rows.append(row)
            print(f"    true {true_preference:.2f}  "
                  f"point {row['median_point_estimate']:.3f}  "
                  f"CI width {row['median_ci_width']:.3f}  "
                  f"CI low {row['median_ci_low']:.3f}  "
                  f"power {row['power_both_criteria']:.0%}")

    actual = [r for r in rows if r["n_pairs"] == args.pairs]
    # Smallest true effect this design detects at >=80% power.
    detectable = [r["true_preference"] for r in actual
                  if r["power_both_criteria"] >= 0.80]
    payload = {
        "schema": "snes_v2.power_check.v1",
        "status": "prospective; run before any listener was recruited",
        "judgements": "SIMULATED; no result about the model is produced here",
        "analysable_pairs": args.pairs,
        "n_listeners": N_LISTENERS,
        "criteria": {"practical_threshold": px.PRACTICAL_THRESHOLD,
                     "superiority": "two-way clustered 95% CI low > 0.50",
                     "both_required": True},
        "simulation": {"pair_sd": PAIR_SD, "listener_sd": LISTENER_SD,
                       "tie_rate": TIE_RATE,
                       "n_simulations": N_SIMULATIONS},
        "minimum_detectable_preference_at_80pct_power":
            min(detectable) if detectable else None,
        "rows": rows,
    }
    Path(args.out).write_text(json.dumps(payload, indent=1))
    print(f"\n  minimum detectable preference at >=80% power, "
          f"{args.pairs} pairs: "
          f"{min(detectable) if detectable else 'NONE in the swept range'}")
    print(f"  written to {args.out}")


if __name__ == "__main__":
    main()
