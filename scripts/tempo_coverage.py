"""How many pieces the widened tempo range recovers.

`fit_grid` does not clip a row rate outside the Tempo token range -- it
*declines the fit entirely* and falls back to the legacy one-step-per-frame
grid. So the tempo bounds silently decide which pieces get a tracker-row
representation at all, and the old 40-360 range was rejecting pieces whose
rates were perfectly measurable.

This reports eligibility by range, per piece and per window, so the widening is
justified by a count rather than by the two out-of-range examples that prompted
it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from typing import Any

from rse.events.pulse import detect_pulse
from rse.tokenizers.frozen import TEMPO_MAX, TEMPO_MIN
from rse.training.corpus import corpus_from_release

RANGES = ((40, 360), (TEMPO_MIN, TEMPO_MAX))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="nesmdb")
    parser.add_argument("--release", default="midi")
    parser.add_argument("--splits", nargs="+", default=["train", "valid"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", type=Path, default=Path("tempo_coverage.json"))
    args = parser.parse_args()

    payload: dict[str, Any] = {"ranges": [list(r) for r in RANGES], "splits": {}}

    for split in args.splits:
        print(f"corpus    : loading {split}...", flush=True)
        entries, _info = corpus_from_release(
            args.dataset, args.release, limit=args.limit, splits=(split,)
        )
        rates: list[float] = []
        no_pulse = 0
        for index, (composition, _s, _p) in enumerate(entries, 1):
            pulse = detect_pulse(composition)
            if not pulse.found or pulse.period_sec <= 0:
                no_pulse += 1
                continue
            rates.append(60.0 / (pulse.period_sec * 4))
            if index % 200 == 0:
                print(f"          : {index}/{len(entries)}", flush=True)

        entry: dict[str, Any] = {
            "n_pieces": len(entries),
            "n_no_pulse": no_pulse,
            "n_with_pulse": len(rates),
            "bpm_min": round(min(rates), 2) if rates else None,
            "bpm_median": round(median(rates), 2) if rates else None,
            "bpm_max": round(max(rates), 2) if rates else None,
            "by_range": {},
        }
        for low, high in RANGES:
            eligible = [r for r in rates if low <= r <= high]
            entry["by_range"][f"{low}-{high}"] = {
                "n_eligible": len(eligible),
                "share_of_pieces": round(len(eligible) / len(entries), 5) if entries else 0.0,
                "n_rejected_out_of_range": len(rates) - len(eligible),
            }
        base = entry["by_range"][f"{RANGES[0][0]}-{RANGES[0][1]}"]["n_eligible"]
        wide = entry["by_range"][f"{RANGES[1][0]}-{RANGES[1][1]}"]["n_eligible"]
        entry["recovered_pieces"] = wide - base
        entry["recovered_share"] = round((wide - base) / len(entries), 5) if entries else 0.0
        payload["splits"][split] = entry

    args.out.write_text(json.dumps(payload, indent=2) + "\n")

    for split, entry in payload["splits"].items():
        print(f"\n{split}: {entry['n_pieces']} pieces, {entry['n_with_pulse']} with a pulse")
        print(f"  fitted BPM: {entry['bpm_min']} .. {entry['bpm_median']} .. {entry['bpm_max']}")
        for name, row in entry["by_range"].items():
            print(
                f"  range {name:>8}: {row['n_eligible']:>5} eligible "
                f"({row['share_of_pieces']:.1%}), "
                f"{row['n_rejected_out_of_range']} rejected for rate"
            )
        print(
            f"  recovered by widening: {entry['recovered_pieces']} pieces "
            f"({entry['recovered_share']:.2%})"
        )
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
