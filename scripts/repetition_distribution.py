"""Is NES-MDB's repetition unimodal, or two distinct compositional behaviours?

The sampling sweep found the models bistable: they either lock into a loop
(n-gram interval repetition 0.7-1.0, runs of 3-4 identical bars) or produce no
repetition at all (0.0, every bar unique), with the corpus median of 0.483
sitting between the two attractors. That was read as a failure to represent
typical structure. It only means that if the corpus itself has a typical
structure to represent.

If real NES music is bimodal -- loop-driven pieces at one end, melody-driven
pieces at the other -- then an acceptance band built around the median punishes
a generator for producing legitimate music, and 0.483 is a target no single
track exhibits.

What this reports
-----------------
Per composition and per 8-bar window, kept separate, because 63% of pieces
yield exactly one window while one yields 96, so pooling windows weights a few
long pieces heavily.

Histograms at several bin widths, since an apparent peak that moves with the
binning is an artefact.

Stratified by piece length: under 8 bars, 8-16, over 16. 37% of NES-MDB is
shorter than eight bars, and those pieces produce a padded single window whose
repetition is not a musical measurement.

Eligible n-gram counts, always. `ngram_repetition_rate` needs five lead pitches
to form one 4-gram and returns 0.0 below that, so a sparse piece scores a
spurious zero. Pieces with too little material are counted and excluded rather
than recorded as having no repetition.

Joint structure, because repetition alone cannot tell a motif recurring with
variation from a bar duplicated four times: n-gram rate against unique-bar
fraction, longest identical-bar run, note density and stepwise motion.

Original against round-tripped, so the answer is about what the model was
actually exposed to and not only about the source files.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from rse import paths
from rse.hardware.profile import load_profile
from rse.tokenizers.tokenizer import RseTokenizer
from rse.training.evaluate import (
    REPETITION_NGRAM,
    bar_signatures,
    evaluate_composition,
    melody_pitches,
)

#: A piece needs this many eligible 4-grams before its repetition rate means
#: anything. Below it the score is an artefact of having no material.
MIN_GRAMS = 8


def eligible_grams(composition: Any) -> int:
    """How many melodic n-grams a composition can form."""
    pitches = melody_pitches(composition, "P1") or melody_pitches(composition)
    intervals = max(0, len(pitches) - 1)
    return max(0, intervals - REPETITION_NGRAM + 1)


def longest_run(composition: Any) -> int:
    signatures = bar_signatures(composition)
    longest = run = 0
    for index in range(1, len(signatures)):
        run = run + 1 if signatures[index] == signatures[index - 1] else 0
        longest = max(longest, run + 1 if run else 0)
    return longest


def histogram(values: list[float], bins: int) -> list[tuple[str, int]]:
    counts = [0] * bins
    for value in values:
        index = min(bins - 1, int(value * bins))
        counts[index] += 1
    width = 1.0 / bins
    return [(f"{i * width:.2f}-{(i + 1) * width:.2f}", c) for i, c in enumerate(counts)]


def describe(label: str, values: list[float]) -> dict[str, Any]:
    if not values:
        return {"label": label, "n": 0}
    ordered = sorted(values)
    return {
        "label": label,
        "n": len(ordered),
        "median": round(statistics.median(ordered), 4),
        "mean": round(statistics.fmean(ordered), 4),
        "p10": round(ordered[int(0.10 * (len(ordered) - 1))], 4),
        "p25": round(ordered[int(0.25 * (len(ordered) - 1))], 4),
        "p75": round(ordered[int(0.75 * (len(ordered) - 1))], 4),
        "p90": round(ordered[int(0.90 * (len(ordered) - 1))], 4),
        "zero_fraction": round(sum(1 for v in ordered if v <= 0.01) / len(ordered), 4),
        "high_fraction": round(sum(1 for v in ordered if v >= 0.80) / len(ordered), 4),
        "middle_fraction": round(sum(1 for v in ordered if 0.35 <= v <= 0.65) / len(ordered), 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="nesmdb")
    parser.add_argument("--release", default="midi")
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=None, help="cap pieces, for a dry run")
    parser.add_argument("--window-bars", type=int, default=8)
    parser.add_argument("--datasets-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("repetition_report.json"))
    parser.add_argument(
        "--skip-roundtrip", action="store_true", help="original only; roughly halves the runtime"
    )
    args = parser.parse_args()

    from rse.training.corpus import corpus_from_release
    from rse.training.dataset import bar_count, slice_bars

    profile = load_profile("nes", configs_dir=paths.HARDWARE_CONFIGS_DIR)
    tokenizer = RseTokenizer()

    entries, _info = corpus_from_release(
        args.dataset,
        args.release,
        datasets_dir=args.datasets_dir,
        limit=args.limit,
        splits=(args.split,),
        seed=0,
    )
    print(f"loaded {len(entries)} pieces from '{args.split}'", flush=True)

    pieces: list[dict[str, Any]] = []
    windows: list[dict[str, Any]] = []
    skipped_short = 0

    for index, (composition, _split, _cid) in enumerate(entries):
        if index and index % 250 == 0:
            print(f"  {index}/{len(entries)}", flush=True)
        if not composition.sounding_events():
            continue
        fit = tokenizer.fit_grid(composition)
        ticks_per_bar = fit.step_ticks * tokenizer.config.steps_per_bar
        total_bars = bar_count(composition, ticks_per_bar=ticks_per_bar)

        grams = eligible_grams(composition)
        metrics = evaluate_composition(composition, profile)
        record = {
            "bars": total_bars,
            "eligible_grams": grams,
            "ngram": metrics.ngram_repetition_rate,
            "unique_bar_fraction": metrics.unique_bar_fraction,
            "longest_run": longest_run(composition),
            "stepwise": metrics.stepwise_fraction,
            "notes_per_second": (
                metrics.n_notes / max(composition.time.seconds(composition.end_tick), 1e-9)
            ),
            "kind": "original",
        }
        if grams < MIN_GRAMS:
            skipped_short += 1
            record["excluded"] = True
        pieces.append(record)

        if not args.skip_roundtrip:
            decoded = tokenizer.decode(tokenizer.encode(composition).ids).composition
            if decoded.sounding_events():
                rt = evaluate_composition(decoded, profile)
                pieces.append(
                    {
                        "bars": total_bars,
                        "eligible_grams": eligible_grams(decoded),
                        "ngram": rt.ngram_repetition_rate,
                        "unique_bar_fraction": rt.unique_bar_fraction,
                        "longest_run": longest_run(decoded),
                        "stepwise": rt.stepwise_fraction,
                        "notes_per_second": (
                            rt.n_notes / max(decoded.time.seconds(decoded.end_tick), 1e-9)
                        ),
                        "kind": "roundtrip",
                        "excluded": eligible_grams(decoded) < MIN_GRAMS,
                    }
                )

        # Windows, only where the piece is long enough to hold one without
        # padding. A padded window's repetition is not a musical measurement.
        if total_bars >= args.window_bars:
            for start in range(0, total_bars - args.window_bars + 1, args.window_bars):
                window = slice_bars(
                    composition, start, args.window_bars, ticks_per_bar=ticks_per_bar
                )
                if not window.sounding_events():
                    continue
                wg = eligible_grams(window)
                wm = evaluate_composition(window, profile)
                windows.append(
                    {
                        "piece_bars": total_bars,
                        "eligible_grams": wg,
                        "ngram": wm.ngram_repetition_rate,
                        "unique_bar_fraction": wm.unique_bar_fraction,
                        "longest_run": longest_run(window),
                        "stepwise": wm.stepwise_fraction,
                        "excluded": wg < MIN_GRAMS,
                    }
                )

    def usable(rows: list[dict[str, Any]], kind: str | None = None) -> list[dict[str, Any]]:
        return [
            r for r in rows if not r.get("excluded") and (kind is None or r.get("kind") == kind)
        ]

    report: dict[str, Any] = {
        "split": args.split,
        "window_bars": args.window_bars,
        "min_eligible_grams": MIN_GRAMS,
        "n_pieces_loaded": len(entries),
        "n_pieces_excluded_too_little_material": skipped_short,
    }

    orig = usable(pieces, "original")
    print(
        f"\n{'=' * 72}\nPER COMPOSITION (original), n={len(orig)} usable, "
        f"{skipped_short} excluded for under {MIN_GRAMS} eligible 4-grams"
    )
    report["per_composition_original"] = describe("original", [r["ngram"] for r in orig])
    for name, value in report["per_composition_original"].items():
        if name != "label":
            print(f"  {name:20} {value}")

    print("\n  histogram of n-gram repetition, three bin widths:")
    for bins in (5, 10, 20):
        counts = histogram([r["ngram"] for r in orig], bins)
        peak = max(c for _, c in counts)
        print(f"    {bins} bins: " + "  ".join(f"{lab}:{c}" for lab, c in counts if c))
        report[f"histogram_{bins}"] = counts
        del peak

    if not args.skip_roundtrip:
        rt = usable(pieces, "roundtrip")
        report["per_composition_roundtrip"] = describe("roundtrip", [r["ngram"] for r in rt])
        print(f"\nPER COMPOSITION (round-trip), n={len(rt)}")
        for name, value in report["per_composition_roundtrip"].items():
            if name != "label":
                print(f"  {name:20} {value}")

    win = usable(windows)
    report["per_window"] = describe("windows", [r["ngram"] for r in win])
    print(f"\nPER {args.window_bars}-BAR WINDOW (unpadded only), n={len(win)}")
    for name, value in report["per_window"].items():
        if name != "label":
            print(f"  {name:20} {value}")

    print("\nSTRATIFIED BY PIECE LENGTH (original compositions)")
    bands = (("under 8 bars", 0, 7), ("8-16 bars", 8, 16), ("over 16 bars", 17, 10**9))
    report["by_length"] = {}
    print(f"  {'band':16}{'n':>6}{'median':>9}{'zero%':>8}{'high%':>8}{'middle%':>9}")
    for label, low, high in bands:
        subset = [r for r in orig if low <= r["bars"] <= high]
        stats = describe(label, [r["ngram"] for r in subset])
        report["by_length"][label] = stats
        if stats["n"]:
            print(
                f"  {label:16}{stats['n']:>6}{stats['median']:>9.3f}"
                f"{100 * stats['zero_fraction']:>7.0f}%{100 * stats['high_fraction']:>7.0f}%"
                f"{100 * stats['middle_fraction']:>8.0f}%"
            )

    # On ROUND-TRIPPED data. Bar signatures are exact (offset, channel,
    # value) tuples, so unquantized MIDI micro-timing means no two bars ever
    # match: unique_bar_fraction reads 1.000 and longest_run 0.00 in every
    # band, telling us nothing. Quantizing is what makes bar-level structure
    # visible, and the quantized form is also what the model was trained on.
    joint_source = usable(pieces, "roundtrip") if not args.skip_roundtrip else orig
    print(
        "\nJOINT STRUCTURE on "
        + ("round-tripped" if not args.skip_roundtrip else "original")
        + " data: is high repetition motif reuse or literal looping?"
    )
    print(
        f"  {'ngram band':14}{'n':>6}{'uniqbar':>9}"
        f"{'longest run':>13}{'notes/sec':>11}{'stepwise':>10}"
    )
    report["joint"] = {}
    for low, high in ((0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)):
        subset = [r for r in joint_source if low <= r["ngram"] < high]
        if not subset:
            continue
        row = {
            "n": len(subset),
            "unique_bar_fraction": round(
                statistics.fmean(r["unique_bar_fraction"] for r in subset), 3
            ),
            "longest_run": round(statistics.fmean(r["longest_run"] for r in subset), 2),
            "notes_per_second": round(statistics.fmean(r["notes_per_second"] for r in subset), 1),
            "stepwise": round(statistics.fmean(r["stepwise"] for r in subset), 3),
        }
        report["joint"][f"{low:.1f}-{high:.1f}"] = row
        print(
            f"  {f'{low:.1f}-{high:.1f}':14}{row['n']:>6}{row['unique_bar_fraction']:>9.3f}"
            f"{row['longest_run']:>13.2f}{row['notes_per_second']:>11.1f}"
            f"{row['stepwise']:>10.3f}"
        )

    report["pieces"] = pieces
    report["windows"] = windows
    args.out.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n  wrote {args.out}")
    print(
        f"  eligible-gram counts: median {statistics.median([r['eligible_grams'] for r in orig])}"
    )
    print(
        f"  excluded {skipped_short} of {len(entries)} pieces "
        f"({100 * skipped_short / max(1, len(entries)):.0f}%) for insufficient material"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
