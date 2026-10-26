"""What does real NES music do across a 16-bar span? The corpus control.

The paired length experiment reported that generated 16-bar pieces restate
almost nothing: median per-bar overlap 0.115, one exactly-repeated bar in 24
pieces. That is only a finding if the corpus does something different, and
there is good reason to doubt it does -- a tracker author writing a 16-bar
loop has every incentive to vary the second half rather than paste the first.

So this measures the same three things on the corpus, through the same code
(`rse.analysis.restatement`), and prints both distributions side by side.

**Train split only.** The official test split stays reserved for final
evaluation, and this reference will be used to judge generated output, which
makes it a selection surface.

A 16-bar corpus span is two consecutive 8-bar cache windows from one
composition, which the cache supports directly: every window records its
`composition_id` and `start_bar`. Spans are taken at every available offset
rather than only from bar 0, because a piece's opening is not representative
of how it develops later.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import json
import random
import statistics as st
from collections import defaultdict
from typing import Any

from rse.analysis.restatement import NGRAM, analyse_adjacent, ticks_per_bar
from rse.tokenizers.frozen import frozen_config
from rse.tokenizers.tokenizer import RseTokenizer

#: Metrics summarised, in report order.
METRICS = (
    "exact_restatement_mean",
    "exact_restatement_bars",
    "contour_recurrence",
    "interval_recurrence",
    "pitch_overlap",
    "new_pitches",
    "lost_pitches",
)


def quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "p10": ordered[int(0.10 * (len(ordered) - 1))],
        "p25": ordered[int(0.25 * (len(ordered) - 1))],
        "median": st.median(ordered),
        "p75": ordered[int(0.75 * (len(ordered) - 1))],
        "p90": ordered[int(0.90 * (len(ordered) - 1))],
        "mean": st.fmean(ordered),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("caches/d4_full.jsonl"))
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--max-spans",
        type=int,
        default=1500,
        help="sampled deterministically; the full set is far larger than needed",
    )
    parser.add_argument("--seed", type=int, default=20261026)
    parser.add_argument("--ngram", type=int, default=NGRAM)
    parser.add_argument(
        "--generated",
        type=Path,
        default=Path("paired_length.json"),
        help="the paired run, re-measured through the same code for comparison",
    )
    parser.add_argument("--out", type=Path, default=Path("restatement_reference.json"))
    args = parser.parse_args()

    tokenizer = RseTokenizer(frozen_config())
    bar_ticks = ticks_per_bar()

    # Index the cache by composition so adjacent windows can be paired.
    by_piece: dict[str, dict[int, list[int]]] = defaultdict(dict)
    kept = 0
    with args.cache.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("split") != args.split:
                continue
            if int(row.get("n_bars", 0)) != 8:
                continue
            by_piece[row["composition_id"]][int(row["start_bar"])] = row["ids"]
            kept += 1
    print(f"{kept} 8-bar {args.split} windows across {len(by_piece)} compositions")

    spans: list[tuple[str, int]] = []
    for piece, windows in by_piece.items():
        for start in sorted(windows):
            if start + 8 in windows:
                spans.append((piece, start))
    print(f"{len(spans)} adjacent 16-bar spans available")
    if not spans:
        print("error: no adjacent window pairs in this split", file=sys.stderr)
        return 2

    sampled = spans
    if args.max_spans and len(spans) > args.max_spans:
        sampled = random.Random(args.seed).sample(spans, args.max_spans)
        sampled.sort()
    print(f"measuring {len(sampled)} spans", flush=True)

    rows: list[dict[str, Any]] = []
    for index, (piece, start) in enumerate(sampled):
        windows = by_piece[piece]
        try:
            first = tokenizer.decode(windows[start]).composition
            second = tokenizer.decode(windows[start + 8]).composition
        except Exception:  # noqa: BLE001 - a bad window is skipped, not fatal
            continue
        report = analyse_adjacent(
            first, second, n_bars=8, bar_ticks=bar_ticks, ngram=args.ngram
        )
        rows.append({"composition_id": piece, "start_bar": start, **report.to_dict()})
        if (index + 1) % 300 == 0:
            print(f"  {index + 1}/{len(sampled)}", flush=True)

    corpus = {
        metric: quantiles([r[metric] for r in rows if r[metric] is not None])
        for metric in METRICS
    }
    n_exact = sum(1 for r in rows if r["exact_restatement_bars"] > 0)

    generated: dict[str, Any] = {}
    gen_rows: list[dict[str, Any]] = []
    if args.generated.exists():
        payload = json.loads(args.generated.read_text())
        for pair in payload.get("pairs", []):
            development = pair.get("long", {}).get("development")
            if pair.get("usable") and development:
                gen_rows.append({"name": pair["name"], **development})
        # The paired run's own numbers used `end_tick / n_bars` for bar
        # boundaries; this reference uses the grid. Only the fields both
        # compute the same way are compared, and the paired run is re-measured
        # from its stored compositions in `paired_restatement.py`.
        generated = {
            "n": len(gen_rows),
            "exact_restatement_mean": quantiles(
                [r["restatement_mean"] for r in gen_rows]
            ),
            "n_with_exact_bar": sum(
                1 for r in gen_rows if r["restatement_exact_bars"] > 0
            ),
        }

    args.out.write_text(
        json.dumps(
            {
                "cache": str(args.cache),
                "split": args.split,
                "ngram": args.ngram,
                "n_windows": kept,
                "n_compositions": len(by_piece),
                "n_spans_available": len(spans),
                "n_spans_measured": len(rows),
                "seed": args.seed,
                "corpus": corpus,
                "corpus_spans_with_any_exact_bar": n_exact,
                "generated_from_paired_run": generated,
                "spans": rows,
            },
            indent=2,
        )
        + "\n"
    )

    head = f"{'metric':<26}{'p10':>8}{'p25':>8}{'median':>9}{'p75':>8}{'p90':>8}{'mean':>8}"
    print(f"\ncorpus, {args.split} split, {len(rows)} spans of 16 bars\n{head}")
    print("-" * len(head))
    for metric in METRICS:
        q = corpus.get(metric)
        if not q:
            continue
        print(
            f"  {metric:<24}{q['p10']:>8.3f}{q['p25']:>8.3f}{q['median']:>9.3f}"
            f"{q['p75']:>8.3f}{q['p90']:>8.3f}{q['mean']:>8.3f}"
        )
    print(
        f"\ncorpus spans with at least one exactly-restated bar: "
        f"{n_exact}/{len(rows)} ({n_exact / len(rows):.1%})"
    )
    if gen_rows:
        gq = generated["exact_restatement_mean"]
        cq = corpus["exact_restatement_mean"]
        print(
            f"\ngenerated (paired run, n={len(gen_rows)}): "
            f"exact restatement median {gq['median']:.3f} "
            f"vs corpus {cq['median']:.3f}"
        )
        print(
            f"generated spans with any exact bar: "
            f"{generated['n_with_exact_bar']}/{len(gen_rows)}"
        )
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
