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
from collections import Counter
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


def exact_ngram_repetition_rate(composition: Any, n: int = REPETITION_NGRAM) -> float:
    """Share of absolute-pitch n-grams that recur. Supplementary only.

    The project's long-standing `ngram_repetition_rate` is already
    interval-based and therefore transposition-invariant: a motif restated a
    fifth higher counts as recurrence. This is the stricter exact-pitch
    companion, so the two together separate "the same shape elsewhere" from
    "the same notes again". Exact-pitch recurrence should be the lower of the
    two; a large gap means the music transposes its motifs.
    """
    pitches = melody_pitches(composition, "P1") or melody_pitches(composition)
    if len(pitches) < n:
        return 0.0
    grams = [tuple(pitches[i : i + n]) for i in range(len(pitches) - n + 1)]
    if not grams:
        return 0.0
    counts = Counter(grams)
    return sum(c for c in counts.values() if c > 1) / len(grams)


def bar_reuse(composition: Any) -> dict[str, float]:
    """Separate consecutive bar duplication from non-adjacent bar reuse.

    These are musically opposite. A bar repeated immediately is a stalled
    loop; the same bar returning after two others is form. Both raise a plain
    repeated-bar count, which is why that count could not tell the model's
    locked loops from the corpus's motif reuse.
    """
    signatures = bar_signatures(composition)
    if len(signatures) < 2:
        return {"consecutive": 0.0, "nonconsecutive": 0.0, "longest_run": 0.0}
    consecutive = sum(1 for i in range(1, len(signatures)) if signatures[i] == signatures[i - 1])
    nonconsecutive = 0
    for i in range(len(signatures)):
        for j in range(i + 2, len(signatures)):
            if signatures[i] == signatures[j]:
                nonconsecutive += 1
                break
    longest = run = 0
    for i in range(1, len(signatures)):
        run = run + 1 if signatures[i] == signatures[i - 1] else 0
        longest = max(longest, run + 1 if run else 0)
    span = len(signatures)
    return {
        "consecutive": consecutive / (span - 1),
        "nonconsecutive": nonconsecutive / span,
        "longest_run": float(longest),
    }


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

    rows: list[dict[str, Any]] = []
    excluded = 0

    def measure(comp: Any, piece: str, kind: str, piece_bars: int) -> dict[str, Any] | None:
        nonlocal excluded
        if not comp.sounding_events():
            return None
        grams = eligible_grams(comp)
        m = evaluate_composition(comp, profile)
        reuse = bar_reuse(comp)
        row = {
            "piece": piece,
            "kind": kind,
            "piece_bars": piece_bars,
            "eligible_grams": grams,
            "ngram_interval": m.ngram_repetition_rate,
            "ngram_exact": exact_ngram_repetition_rate(comp),
            "unique_bar_fraction": m.unique_bar_fraction,
            "bars_consecutive_dup": reuse["consecutive"],
            "bars_nonconsecutive_reuse": reuse["nonconsecutive"],
            "longest_run": reuse["longest_run"],
            "stepwise": m.stepwise_fraction,
            "pitch_entropy": m.pitch_class_entropy,
            "in_scale": m.in_scale_fraction,
            "notes_per_second": m.n_notes / max(comp.time.seconds(comp.end_tick), 1e-9),
            "excluded": grams < MIN_GRAMS,
        }
        if row["excluded"]:
            excluded += 1
        return row

    for index, (composition, _split, cid) in enumerate(entries):
        if index and index % 250 == 0:
            print(f"  {index}/{len(entries)}", flush=True)
        if not composition.sounding_events():
            continue
        piece = str(cid)
        fit = tokenizer.fit_grid(composition)
        ticks_per_bar = fit.step_ticks * tokenizer.config.steps_per_bar
        total_bars = bar_count(composition, ticks_per_bar=ticks_per_bar)

        for row in (measure(composition, piece, "orig_piece", total_bars),):
            if row:
                rows.append(row)
        if not args.skip_roundtrip:
            decoded = tokenizer.decode(tokenizer.encode(composition).ids).composition
            row = measure(decoded, piece, "rt_piece", total_bars)
            if row:
                rows.append(row)

        # Equivalent 8-bar windows for original and round-trip. Only where the
        # piece can hold a whole window: a padded window's repetition is not a
        # musical measurement.
        if total_bars >= args.window_bars:
            for start_bar in range(0, total_bars - args.window_bars + 1, args.window_bars):
                window = slice_bars(
                    composition, start_bar, args.window_bars, ticks_per_bar=ticks_per_bar
                )
                row = measure(window, piece, "orig_window", total_bars)
                if row:
                    rows.append(row)
                if not args.skip_roundtrip:
                    rt = tokenizer.decode(tokenizer.encode(window).ids).composition
                    row = measure(rt, piece, "rt_window", total_bars)
                    if row:
                        rows.append(row)

    def usable(kind: str, **where: Any) -> list[dict[str, Any]]:
        out = [r for r in rows if r["kind"] == kind and not r["excluded"]]
        for key, (low, high) in where.items():
            out = [r for r in out if low <= r[key] <= high]
        return out

    def uniq(subset: list[dict[str, Any]]) -> int:
        return len({r["piece"] for r in subset})

    report: dict[str, Any] = {
        "split": args.split,
        "window_bars": args.window_bars,
        "min_eligible_grams": MIN_GRAMS,
        "n_pieces_loaded": len(entries),
        "n_rows_excluded_too_little_material": excluded,
        "note_on_metrics": (
            "ngram_interval is the long-standing metric and is already "
            "transposition-invariant; ngram_exact is the stricter absolute-pitch "
            "companion. bars_consecutive_dup is a stalled loop, "
            "bars_nonconsecutive_reuse is form -- both raise a plain repeated-bar "
            "count, which is why that count could not separate them."
        ),
        "generated_material": (
            "absent: no checkpoint available locally, so the generated arm of the "
            "comparison is not included and must be run against a trained model"
        ),
    }

    print(f"\n{'=' * 78}")
    print(
        f"{'group':14}{'n':>7}{'pieces':>8}{'interval':>10}{'exact':>8}"
        f"{'uniqbar':>9}{'consec':>8}{'nonconsec':>11}{'run':>6}"
    )
    for kind in ("orig_piece", "rt_piece", "orig_window", "rt_window"):
        subset = usable(kind)
        if not subset:
            continue
        stats = describe(kind, [r["ngram_interval"] for r in subset])
        report[kind] = {
            **stats,
            "unique_pieces": uniq(subset),
            "ngram_exact_median": round(statistics.median([r["ngram_exact"] for r in subset]), 4),
            "unique_bar_fraction_median": round(
                statistics.median([r["unique_bar_fraction"] for r in subset]), 4
            ),
            "bars_consecutive_dup_median": round(
                statistics.median([r["bars_consecutive_dup"] for r in subset]), 4
            ),
            "bars_nonconsecutive_reuse_median": round(
                statistics.median([r["bars_nonconsecutive_reuse"] for r in subset]), 4
            ),
            "longest_run_mean": round(statistics.fmean([r["longest_run"] for r in subset]), 3),
        }
        r = report[kind]
        print(
            f"{kind:14}{r['n']:>7}{r['unique_pieces']:>8}{r['median']:>10.3f}"
            f"{r['ngram_exact_median']:>8.3f}{r['unique_bar_fraction_median']:>9.3f}"
            f"{r['bars_consecutive_dup_median']:>8.3f}"
            f"{r['bars_nonconsecutive_reuse_median']:>11.3f}"
            f"{r['longest_run_mean']:>6.2f}"
        )

    print("\nDISTRIBUTION of interval repetition, rt_window (what the model saw)")
    target = usable("rt_window") or usable("rt_piece")
    for bins in (5, 10, 20):
        counts = histogram([r["ngram_interval"] for r in target], bins)
        report[f"histogram_rt_window_{bins}"] = counts
        print(f"  {bins:2} bins: " + " ".join(f"{lab}:{c}" for lab, c in counts if c))
    stats = describe("rt_window", [r["ngram_interval"] for r in target])
    report["rt_window_distribution"] = stats
    print(
        f"  zero {100 * stats['zero_fraction']:.0f}%  "
        f"middle(.35-.65) {100 * stats['middle_fraction']:.0f}%  "
        f"high(>=.8) {100 * stats['high_fraction']:.0f}%"
    )

    print("\nSTRATIFIED BY PIECE LENGTH (rt_window)")
    print(f"  {'band':14}{'n':>7}{'pieces':>8}{'median':>9}{'zero%':>7}{'mid%':>7}{'high%':>7}")
    report["by_length"] = {}
    for label, low, high in (
        ("8-16 bars", 8, 16),
        ("17-32 bars", 17, 32),
        ("over 32 bars", 33, 10**9),
    ):
        subset = usable("rt_window", piece_bars=(low, high))
        if not subset:
            continue
        stats = describe(label, [r["ngram_interval"] for r in subset])
        report["by_length"][label] = {**stats, "unique_pieces": uniq(subset)}
        print(
            f"  {label:14}{stats['n']:>7}{uniq(subset):>8}{stats['median']:>9.3f}"
            f"{100 * stats['zero_fraction']:>6.0f}%{100 * stats['middle_fraction']:>6.0f}%"
            f"{100 * stats['high_fraction']:>6.0f}%"
        )

    print("\nJOINT: within each interval-repetition band, is it loop or form?")
    print(
        f"  {'band':12}{'n':>7}{'pieces':>8}{'uniqbar':>9}{'consec':>8}"
        f"{'nonconsec':>11}{'run':>6}{'n/sec':>7}{'step':>7}"
    )
    report["joint"] = {}
    for low, high in ((0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)):
        subset = usable("rt_window", ngram_interval=(low, high - 1e-9))
        if not subset:
            continue
        row = {
            "n": len(subset),
            "unique_pieces": uniq(subset),
            "unique_bar_fraction": round(
                statistics.fmean(r["unique_bar_fraction"] for r in subset), 3
            ),
            "bars_consecutive_dup": round(
                statistics.fmean(r["bars_consecutive_dup"] for r in subset), 3
            ),
            "bars_nonconsecutive_reuse": round(
                statistics.fmean(r["bars_nonconsecutive_reuse"] for r in subset), 3
            ),
            "longest_run": round(statistics.fmean(r["longest_run"] for r in subset), 2),
            "notes_per_second": round(statistics.fmean(r["notes_per_second"] for r in subset), 1),
            "stepwise": round(statistics.fmean(r["stepwise"] for r in subset), 3),
        }
        report["joint"][f"{low:.1f}-{high:.1f}"] = row
        print(
            f"  {f'{low:.1f}-{high:.1f}':12}{row['n']:>7}{row['unique_pieces']:>8}"
            f"{row['unique_bar_fraction']:>9.3f}{row['bars_consecutive_dup']:>8.3f}"
            f"{row['bars_nonconsecutive_reuse']:>11.3f}{row['longest_run']:>6.2f}"
            f"{row['notes_per_second']:>7.1f}{row['stepwise']:>7.3f}"
        )

    report["rows"] = rows
    args.out.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n  wrote {args.out}")
    grams = [r["eligible_grams"] for r in usable("orig_piece")]
    print(f"  eligible 4-grams per piece: median {statistics.median(grams or [0])}")
    print(f"  excluded {excluded} rows for under {MIN_GRAMS} eligible 4-grams")
    print(
        "  generated material is NOT in this report: no checkpoint is available "
        "locally, so that arm must be measured against a trained model."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
