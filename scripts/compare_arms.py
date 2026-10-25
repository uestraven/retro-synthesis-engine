"""Compact per-arm comparison across battery runs, with paired differences.

Reads one or more `generation_battery` JSONs and prints a short table instead
of the per-sample rows. Paired by seed, because matched seeds are the whole
reason for running 36 of them: an unpaired median difference at this sample
size is swamped by which seeds happened to land in which arm.

Each arm is labelled by its structural temperature and whether EOS was
isolated, so runs with and without isolation can be passed together and read
as one experiment.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Importable when run as a file from a checkout where the package is not
# installed, which is the normal state of a fresh GPU host.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import json
import statistics as st
from typing import Any

#: Reported for every arm. Keep short: the point is a readable console table.
METRICS = (
    "notes_per_bar",
    "ioi_entropy",
    "n_distinct_iois",
    "pitch_class_entropy",
    "ngram_repetition_rate",
    "motif_note_participation",
    "top_duration_share",
)


def arm_label(row: dict[str, Any], isolate: bool) -> str:
    structural = row.get("structural_temperature")
    base = "none" if structural is None else f"T{structural:g}"
    return f"{base}{'+iso' if isolate else ''}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", type=Path, nargs="+")
    parser.add_argument(
        "--baseline",
        default=None,
        help="arm label to compute paired differences against; defaults to the first arm seen",
    )
    args = parser.parse_args()

    arms: dict[str, list[dict[str, Any]]] = {}
    checkpoints: set[str] = set()
    for path in args.files:
        payload = json.loads(path.read_text())
        isolate = bool(payload.get("isolate_eos", False))
        checkpoints.add(Path(payload.get("checkpoint", "?")).name)
        for row in payload["rows"]:
            arms.setdefault(arm_label(row, isolate), []).append(row)

    if not arms:
        print("error: no rows found", file=sys.stderr)
        return 2
    order = list(arms)
    baseline = args.baseline or order[0]

    print(f"checkpoints: {', '.join(sorted(checkpoints))}")
    header = (
        f"\n{'arm':<10}{'n':>4}{'8bar%':>7}{'bars':>6}{'ev':>6}{'n/bar':>7}"
        f"{'4chan':>7}{'perc%':>7}{'ioiH':>7}{'IOIs':>6}{'pitchH':>8}{'ngram':>7}"
    )
    print(header)
    print("-" * (len(header) - 1))
    for name in order:
        group = arms[name]
        done = [r for r in group if r["completed_8_bars"]]
        print(
            f"{name:<10}{len(group):>4}{len(done) / len(group):>7.0%}"
            f"{st.median([r['n_bars'] for r in group]):>6.1f}"
            f"{st.median([r['n_events'] for r in group]):>6.0f}"
            f"{st.median([r['notes_per_bar'] for r in group]):>7.1f}"
            f"{sum(1 for r in group if r['n_channels_used'] == 4) / len(group):>7.0%}"
            f"{sum(1 for r in group if r['n_percussion'] > 0) / len(group):>7.0%}"
            f"{st.median([r['ioi_entropy'] for r in group]):>7.2f}"
            f"{st.median([r['n_distinct_iois'] for r in group]):>6.1f}"
            f"{st.median([r['pitch_class_entropy'] for r in group]):>8.2f}"
            f"{st.median([r['ngram_repetition_rate'] for r in group]):>7.3f}"
        )

    print("\ncompleted 8-bar samples only (premature endings excluded):")
    sub = f"{'arm':<10}{'n':>4}{'n/bar':>8}{'ioiH':>8}{'IOIs':>6}{'pitchH':>8}"
    print(sub)
    print("-" * len(sub))
    for name in order:
        done = [r for r in arms[name] if r["completed_8_bars"]]
        if not done:
            print(f"{name:<10}{0:>4}{'-':>8}{'-':>8}{'-':>6}{'-':>8}")
            continue
        print(
            f"{name:<10}{len(done):>4}"
            f"{st.median([r['notes_per_bar'] for r in done]):>8.1f}"
            f"{st.median([r['ioi_entropy'] for r in done]):>8.2f}"
            f"{st.median([r['n_distinct_iois'] for r in done]):>6.1f}"
            f"{st.median([r['pitch_class_entropy'] for r in done]):>8.2f}"
        )

    print(f"\npaired per-seed differences against {baseline}:")
    base_by_seed = {r["seed"]: r for r in arms.get(baseline, [])}
    pair = f"{'arm':<10}{'metric':<26}{'median d':>10}{'win/loss':>10}{'n':>4}"
    print(pair)
    print("-" * len(pair))
    for name in order:
        if name == baseline:
            continue
        for metric in METRICS:
            deltas = [
                r[metric] - base_by_seed[r["seed"]][metric]
                for r in arms[name]
                if r["seed"] in base_by_seed
                and isinstance(r.get(metric), (int, float))
                and isinstance(base_by_seed[r["seed"]].get(metric), (int, float))
            ]
            if not deltas:
                continue
            wins = sum(1 for d in deltas if d > 0)
            print(
                f"{name:<10}{metric:<26}{st.median(deltas):>+10.3f}"
                f"{f'{wins}/{len(deltas) - wins}':>10}{len(deltas):>4}"
            )
    print("\ntermination reasons per arm:")
    for name in order:
        from collections import Counter

        counts = Counter(r["termination"] for r in arms[name])
        print(f"  {name:<10}{dict(counts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
