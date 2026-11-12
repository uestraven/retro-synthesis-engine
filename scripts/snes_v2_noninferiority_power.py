#!/usr/bin/env python3
"""Sample size for the model-vs-genuine noninferiority endpoint.

Milestone 3's paired study was declared underpowered *after* its stimuli
existed. This runs the equivalent calculation **before** the new protocol is
frozen, so the registered seed population is derived from the criterion
rather than inherited from the previous design.

The criterion under test, stated as a noninferiority claim:

    P = P(model excerpt preferred over a genuine SNES excerpt), ties = 0.5
    noninferiority established if the two-way clustered 95% CI
    lower bound for P exceeds 0.50 - margin

Power is the probability of establishing that when the model is **truly
equal** to genuine music (P = 0.50). The sweep also reports the rate at
which a genuinely inferior model (P = 0.40, 0.35) would wrongly pass, which
is the error a noninferiority margin can hide if it is set too wide.

Judgements are simulated with per-pair and per-listener random effects,
because the registered bootstrap resamples both clusters.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rse.snes_v2 import preference as px  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "power_check", Path(__file__).with_name("snes_v2_power_check.py"))
_power = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_power)

N_SIMULATIONS = 200


def evaluate(n_pairs: int, n_listeners: int, true_preference: float,
             margin: float, pairs_per_listener: int | None = None) -> dict:
    floor = 0.50 - margin
    passes, lows, widths, points = 0, [], [], []
    for sim in range(N_SIMULATIONS):
        judgements = _power.simulate(n_pairs, true_preference,
                                     seed=5000 + sim,
                                     n_listeners=n_listeners,
                                     pairs_per_listener=pairs_per_listener)
        result = px.analyse(judgements, n_replicates=2000,
                            seed=px.BOOTSTRAP_SEED + sim)
        lows.append(result.ci_low)
        widths.append(result.ci_high - result.ci_low)
        points.append(result.adjusted_preference)
        if result.ci_low > floor:
            passes += 1
    return {
        "n_pairs": n_pairs, "n_listeners": n_listeners,
        "pairs_per_listener": pairs_per_listener,
        "n_judgements": n_listeners * (pairs_per_listener or n_pairs),
        "true_preference": true_preference, "margin": margin,
        "noninferiority_floor": floor,
        "median_point": st.median(points),
        "median_ci_low": st.median(lows),
        "median_ci_width": st.median(widths),
        "pass_rate": passes / N_SIMULATIONS,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--margin", type=float, default=0.10)
    parser.add_argument("--target-power", type=float, default=0.80)
    parser.add_argument("--listeners", type=int, nargs="+", default=[20, 30])
    parser.add_argument("--pairs", type=int, nargs="+",
                        default=[12, 20, 30, 40, 50, 60])
    parser.add_argument("--pairs-per-listener", type=int, default=None,
                        help="fixed rotation; omit to cross fully (an upper "
                             "bound on power, and infeasible past ~30 pairs)")
    parser.add_argument("--out", default=str(
        Path.home() / "workplace/v2_artifacts/noninferiority_power.json"))
    args = parser.parse_args()

    floor = 0.50 - args.margin
    print(f"  noninferiority margin {args.margin}: establish if CI low > "
          f"{floor:.2f}")
    print(f"  target power {args.target_power:.0%} at a TRUE preference of "
          f"0.50 (model equal to genuine)")
    print(f"  pair SD {_power.PAIR_SD}, listener SD {_power.LISTENER_SD}, "
          f"tie rate {_power.TIE_RATE}; {N_SIMULATIONS} sims per cell\n")

    rows = []
    print("  listeners  pairs  judgem  true P  median CI low  width   "
          "pass rate")
    for n_listeners in args.listeners:
        for n_pairs in args.pairs:
            for true_preference in (0.50, 0.40, 0.35):
                row = evaluate(n_pairs, n_listeners, true_preference,
                               args.margin, args.pairs_per_listener)
                rows.append(row)
                label = ("power" if true_preference == 0.50
                         else "FALSE PASS")
                print(f"  {n_listeners:9d}  {n_pairs:5d}  "
                      f"{row['n_judgements']:6d}  "
                      f"{true_preference:.2f}    {row['median_ci_low']:.3f}"
                      f"        {row['median_ci_width']:.3f}   "
                      f"{row['pass_rate']:.0%}  ({label})")
        print()

    # Smallest pair count reaching target power at true 0.50, per listener count.
    required: dict[str, int | None] = {}
    for n_listeners in args.listeners:
        ok = [r["n_pairs"] for r in rows
              if r["n_listeners"] == n_listeners
              and r["true_preference"] == 0.50
              and r["pass_rate"] >= args.target_power]
        required[str(n_listeners)] = min(ok) if ok else None

    payload = {
        "schema": "snes_v2.noninferiority_power.v1",
        "status": "prospective; run before the new protocol was frozen and "
                  "before any stimulus existed",
        "judgements": "SIMULATED; no result about the model is produced here",
        "criterion": {
            "outcome": "P(model preferred over genuine), ties scored 0.5",
            "margin": args.margin,
            "noninferiority_floor": floor,
            "rule": f"two-way clustered 95% CI lower bound > {floor:.2f}",
        },
        "simulation": {"pair_sd": _power.PAIR_SD,
                       "listener_sd": _power.LISTENER_SD,
                       "tie_rate": _power.TIE_RATE,
                       "n_simulations": N_SIMULATIONS},
        "target_power": args.target_power,
        "pairs_per_listener": args.pairs_per_listener,
        "assignment": ("fully crossed (upper bound on power)"
                       if args.pairs_per_listener is None
                       else f"fixed rotation, {args.pairs_per_listener} "
                            f"pairs per listener"),
        "required_pairs_by_listener_count": required,
        "rows": rows,
    }
    Path(args.out).write_text(json.dumps(payload, indent=1))
    print(f"  pairs required for {args.target_power:.0%} power at true 0.50:")
    for listeners, n in required.items():
        print(f"    {listeners} listeners: "
              f"{n if n else 'NONE in the swept range'}")
    print(f"  written to {args.out}")


if __name__ == "__main__":
    main()
