"""Controlled sampling sweep: can sampling alone recover NES-MDB's structure?

Fixes the checkpoints and varies only generation parameters, because the
diagnostic question is whether the structure already present in the training
representation can be sampled out of a trained model, or whether the model
never learned it. Retraining would answer a different question and cost more.

What it measures, and what may be compared with what
----------------------------------------------------
Three groups are kept apart on purpose:

- **corpus original** -- real music, the musical target.
- **corpus round-trip** -- the same music after encode/decode, which is the
  ceiling the representation allows. The gap to the original is representation
  loss, currently about 13% of n-gram repetition and 26% of note density.
- **generated** -- what the model produces.

Reference windows match the generation length. N-gram repetition is
sequence-length sensitive -- on the train split the median runs 0.592 over
full songs, 0.572 over sixteen bars, 0.531 over eight -- and the reference
items are complete pieces, so a whole-song figure is not a target for an
eight-bar sample.

Token-level numbers are reported in their own columns and must not be compared
against the decoded musical metrics. Repetition over a token stream is a
different quantity from repetition over pitches; it is comparable across
sampling configurations and nothing else. `Channel_*` is a valid note-on count:
verified equal to the decoded note count on 12 of 12 pieces.

Density is reported per second as well as per bar. `notes_per_bar` uses nominal
bars, which last 2.0 s for a MIDI-release reference at its fake 120 BPM but
1.6 s for a sample decoded at Tempo_150, so the per-bar figures are not
strictly like for like. Per-second density is representation-independent.

Ranking
-------
Validity and memorization are filters, not scores. Among survivors the penalty
is the weighted normalized distance *outside* each corpus interquartile range
-- repetition 3x, note density 2x, stepwise motion 1x -- tie-broken by distance
from the corpus median. Nothing is rewarded for exceeding the corpus on
repetition: a literal loop would score perfectly on that metric alone, which is
why unique-bar fraction and the longest run of identical consecutive bars are
reported beside it.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from rse import paths
from rse.hardware.constraints import ConstraintEngine
from rse.hardware.profile import load_profile
from rse.tokenizers.tokenizer import Conditioning
from rse.training.evaluate import (
    MEMORIZATION_THRESHOLD,
    aggregate,
    aggregate_by_piece,
    bar_signatures,
    corpus_fingerprints,
    evaluate_composition,
    reference_windows,
)

#: Weights for the ranked metrics, per the review's scoring decision.
WEIGHTS = {"ngram_repetition_rate": 3.0, "notes_per_bar": 2.0, "stepwise_fraction": 1.0}

#: Reported but never ranked on: a model can satisfy these while producing
#: nothing musical, so they catch regressions rather than indicate success.
GUARDRAILS = ("pitch_class_entropy", "in_scale_fraction", "n_channels_used")


def token_metrics(tokens: list[str], n: int = 4) -> dict[str, float]:
    """Repetition and note-on density over the raw token stream.

    Comparable across sampling configurations only. A token n-gram spans
    position, channel, pitch, velocity and duration tokens together, so its
    repetition rate is not the pitch n-gram rate and the two must not be put in
    the same column.
    """
    bars = max(1, sum(1 for t in tokens if t == "Bar"))
    note_ons = sum(1 for t in tokens if t.startswith("Channel_"))
    grams = [tuple(tokens[i : i + n]) for i in range(max(0, len(tokens) - n + 1))]
    repeated = 0
    if grams:
        counts = Counter(grams)
        repeated = sum(c for c in counts.values() if c > 1)
    return {
        "token_ngram_repetition": repeated / len(grams) if grams else 0.0,
        "token_note_ons_per_bar": note_ons / bars,
        "n_tokens": float(len(tokens)),
        "token_bars": float(bars),
    }


def structure_metrics(composition: Any) -> dict[str, float]:
    """Trivial-loop detection: how long an identical bar runs unbroken."""
    signatures = bar_signatures(composition)
    longest = run = 0
    for index in range(1, len(signatures)):
        run = run + 1 if signatures[index] == signatures[index - 1] else 0
        longest = max(longest, run + 1 if run else 0)
    return {"longest_identical_bar_run": float(longest)}


def slice_seconds(composition: Any, start_sec: float, length_sec: float) -> Any:
    """A window of real elapsed time, not of nominal bars.

    The MIDI release's tempo metadata is fake -- every file claims 120 BPM --
    so a nominal bar is 2.0 s there while a decoded sample's nominal bar lasts
    whatever its Tempo token says. Slicing by seconds compares like with like
    on density. Events are selected by their own mapped start time rather than
    by assuming a constant tick rate.
    """
    from dataclasses import replace as dc_replace

    time = composition.time
    kept = []
    offset = None
    for event in composition.sorted_events():
        at = time.seconds(event.tick)
        if at < start_sec or at >= start_sec + length_sec:
            continue
        if offset is None:
            offset = event.tick
        kept.append(dc_replace(event, tick=event.tick - offset))
    window = type(composition)(
        hardware_profile=composition.hardware_profile,
        ticks_per_beat=composition.ticks_per_beat,
        events=kept,
        tempo_map=list(composition.tempo_map[:1]),
        time_signatures=list(composition.time_signatures[:1]),
    )
    window.normalize()
    return window


def event_fingerprint(composition: Any) -> set[tuple[int, str, int]]:
    """A composition as a set of (time, channel, value), quantized to 24 Hz.

    Quantized so that a millisecond of timing difference does not register as
    different music, and compared as a set so order and duration do not.
    """
    from rse.events.schema import Note

    time = composition.time
    return {
        (
            int(round(time.seconds(event.tick) * 24)),
            event.channel_id,
            event.pitch if isinstance(event, Note) else event.value,
        )
        for event in composition.sounding_events()
    }


def inter_seed_similarity(compositions: list[Any]) -> float:
    """Mean pairwise Jaccard similarity between a configuration's seeds.

    The point of this is to catch a failure mode the repetition metric cannot
    see: a low temperature can make every sample internally repetitive while
    making every seed produce nearly the same piece. That raises the n-gram
    rate without improving the generator at all, so a configuration scoring
    well on repetition and also near 1.0 here has collapsed rather than
    learned. 0.0 means the seeds share nothing; 1.0 means they are identical.
    """
    prints = [event_fingerprint(c) for c in compositions]
    prints = [p for p in prints if p]
    if len(prints) < 2:
        return 0.0
    scores = []
    for i in range(len(prints)):
        for j in range(i + 1, len(prints)):
            union = prints[i] | prints[j]
            if union:
                scores.append(len(prints[i] & prints[j]) / len(union))
    return sum(scores) / len(scores) if scores else 0.0


def bootstrap_median(values: list[float], *, n: int = 1000, seed: int = 0) -> tuple[float, float]:
    """95% confidence interval for a median, resampling the observations.

    Resampled at the piece level by the caller. Twenty-six windows drawn from
    fourteen songs are not twenty-six independent observations, and a reference
    quoted without an interval invites treating a smoke-test population as a
    settled target.
    """
    import random as _random
    import statistics as _stats

    if len(values) < 2:
        return (values[0] if values else 0.0, values[0] if values else 0.0)
    rng = _random.Random(seed)
    medians = sorted(_stats.median(rng.choices(values, k=len(values))) for _ in range(n))
    return (medians[int(0.025 * n)], medians[int(0.975 * n)])


def outside(value: float, low: float, high: float) -> float:
    """Normalized distance outside a band; zero when inside it."""
    width = high - low
    if width <= 0:
        return 0.0
    if value < low:
        return (low - value) / width
    if value > high:
        return (value - high) / width
    return 0.0


def score(row: dict[str, Any], reference: Any) -> tuple[float, float]:
    """(primary penalty, tiebreaker). Lower is better for both."""
    penalty = tie = 0.0
    for metric, weight in WEIGHTS.items():
        low = reference.p25.get(metric)
        high = reference.p75.get(metric)
        median = reference.medians.get(metric)
        value = row.get(metric)
        if low is None or high is None or value is None or median is None:
            continue
        penalty += weight * outside(value, low, high)
        width = high - low
        tie += weight * (abs(value - median) / width if width > 0 else 0.0)
    return penalty, tie


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--temperatures", type=float, nargs="+", default=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--top-k", type=int, nargs="*", default=[])
    parser.add_argument("--top-p", type=float, nargs="*", default=[])
    parser.add_argument(
        "--untruncated",
        action="store_true",
        help="add a control arm with top_k and top_p both disabled, at every "
        "temperature. Establishes what the sampler's defaults of top_k=40 and "
        "top_p=0.95 -- active in every sample taken so far -- actually contribute",
    )
    parser.add_argument("--bars", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--dataset", default="nesmdb")
    parser.add_argument("--release", default="midi")
    parser.add_argument("--reference-split", default="train")
    parser.add_argument("--n-reference", type=int, default=60)
    parser.add_argument("--datasets-dir", type=Path, default=None)
    parser.add_argument("--configs-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument(
        "--keep-events",
        type=int,
        default=3,
        help="save this many generated event files per checkpoint, best-scoring first",
    )
    args = parser.parse_args()

    from rse.generation.sampler import SamplingConfig, sample
    from rse.training.corpus import corpus_from_release
    from rse.training.train import load_checkpoint

    args.out.mkdir(parents=True, exist_ok=True)
    profile = load_profile("nes", configs_dir=args.configs_dir or paths.HARDWARE_CONFIGS_DIR)
    engine = ConstraintEngine(profile)

    # --- references, computed once ----------------------------------------
    print("reference : loading corpus...", flush=True)
    entries, _info = corpus_from_release(
        args.dataset,
        args.release,
        datasets_dir=args.datasets_dir,
        limit=args.n_reference,
        splits=None if args.reference_split == "all" else (args.reference_split,),
        seed=0,
    )
    compositions = [c for c, _, _ in entries]
    fingerprints = corpus_fingerprints(compositions, profile)

    # The tokenizer for the reference round trip comes from the first
    # checkpoint, since the round trip is only meaningful for a given encoding.
    _m0, tokenizer0, _p0 = load_checkpoint(args.checkpoints[0])
    original_tagged, roundtrip_tagged = reference_windows(
        compositions,
        tokenizer0,
        profile,
        fingerprints=fingerprints,
        window_bars=args.bars,
        min_bars=args.bars,
    )
    # Two aggregations of the same windows. Per-window pools every window
    # equally, which lets long pieces dominate; per-piece gives each source
    # piece one vote. The ranking uses per-piece.
    corpus_by_window = aggregate("corpus original (per window)", [m for _, m in original_tagged])
    corpus_original = aggregate_by_piece("corpus original (per piece)", original_tagged)
    corpus_roundtrip = aggregate_by_piece("corpus round-trip (per piece)", roundtrip_tagged)
    unique_pieces = len({piece for piece, _ in original_tagged})
    print(
        f"reference : {len(original_tagged)} windows of {args.bars} bars from "
        f"{unique_pieces} unique pieces in '{args.reference_split}'",
        flush=True,
    )
    # Piece-level resampling: windows from one song are correlated, so the
    # interval is over pieces rather than over windows.
    by_piece: dict[str, list[float]] = {}
    for piece, metrics in original_tagged:
        by_piece.setdefault(piece, []).append(metrics.ngram_repetition_rate)
    piece_values = [sum(v) / len(v) for v in by_piece.values()]
    ngram_ci = bootstrap_median(piece_values)
    print(
        f"            ngram per piece "
        f"{corpus_original.medians['ngram_repetition_rate']:.3f} "
        f"IQR {corpus_original.p25['ngram_repetition_rate']:.3f}-"
        f"{corpus_original.p75['ngram_repetition_rate']:.3f}   "
        f"per window {corpus_by_window.medians['ngram_repetition_rate']:.3f}",
        flush=True,
    )
    print(
        f"            95% CI for the per-piece median, resampling {len(piece_values)} "
        f"pieces: {ngram_ci[0]:.3f}-{ngram_ci[1]:.3f}",
        flush=True,
    )

    # --- the sweep ---------------------------------------------------------
    # The sampler's own defaults are top_k=40 and top_p=0.95, so every sample
    # taken so far has had both active. Each arm below therefore states all
    # three values explicitly: the temperature arm holds the default nucleus
    # settings fixed, and the top_k and top_p arms disable the other truncation
    # so their effects stay identifiable rather than compounding.
    configs: list[dict[str, Any]] = [
        {"kind": "temperature", "temperature": t, "top_k": 40, "top_p": 0.95}
        for t in args.temperatures
    ]
    configs += [
        {"kind": "top_k", "temperature": 1.0, "top_k": k, "top_p": None} for k in args.top_k
    ]
    configs += [
        {"kind": "top_p", "temperature": 1.0, "top_k": None, "top_p": p} for p in args.top_p
    ]
    if args.untruncated:
        configs += [
            {"kind": "untruncated", "temperature": t, "top_k": None, "top_p": None}
            for t in args.temperatures
        ]

    rows: list[dict[str, Any]] = []
    # Kept so the best-scoring configurations can be written out for
    # listening; a ranked table is not a substitute for hearing it.
    produced: dict[tuple, Any] = {}
    for checkpoint in args.checkpoints:
        model, tokenizer, _payload = load_checkpoint(checkpoint)
        label = checkpoint.stem
        for config in configs:
            for seed in args.seeds:
                kwargs: dict[str, Any] = {
                    "seed": seed,
                    "temperature": config["temperature"],
                    "max_tokens": args.max_tokens,
                    "top_k": config["top_k"],
                    "top_p": config["top_p"],
                }
                result = sample(
                    model,
                    tokenizer,
                    conditioning=Conditioning(n_bars=args.bars),
                    config=SamplingConfig(**kwargs),
                )
                tokens = [tokenizer.vocabulary.token_of(i) for i in result.ids]
                raw_ok = not engine.validate(result.composition)
                playable, _repair = engine.apply(result.composition)
                metrics = evaluate_composition(playable, profile, corpus_fingerprints=fingerprints)
                seconds = playable.time.seconds(playable.end_tick) or 1e-9
                notes = playable.sounding_events()
                rests = sum(
                    1 for channel in profile.channel_ids if not playable.sounding_events(channel)
                )
                durations = [e.duration for e in notes]
                row: dict[str, Any] = {
                    "checkpoint": label,
                    "kind": config["kind"],
                    "temperature": config["temperature"],
                    "top_k": config["top_k"] or 0,
                    "top_p": config["top_p"] or 0.0,
                    "seed": seed,
                    **{k: getattr(metrics, k) for k in WEIGHTS},
                    **{k: getattr(metrics, k) for k in GUARDRAILS},
                    "notes_per_second": len(notes) / seconds,
                    "seconds": seconds,
                    "n_notes": metrics.n_notes,
                    "n_bars": metrics.n_bars,
                    "unique_bar_fraction": metrics.unique_bar_fraction,
                    "mean_note_duration_ticks": (
                        sum(durations) / len(durations) if durations else 0.0
                    ),
                    "silent_channels": rests,
                    "raw_hardware_compliant": raw_ok,
                    "hardware_compliant": metrics.hardware_compliant,
                    "loop_ok": metrics.loop_ok,
                    "flagged_as_memorized": bool(
                        metrics.max_corpus_similarity is not None
                        and metrics.max_corpus_similarity >= MEMORIZATION_THRESHOLD
                    ),
                    "max_corpus_similarity": metrics.max_corpus_similarity or 0.0,
                    "hit_token_limit": bool(getattr(result, "hit_token_limit", False)),
                    **token_metrics(tokens),
                    **structure_metrics(playable),
                }
                for channel in profile.channel_ids:
                    row[f"onsets_per_sec_{channel}"] = (
                        len(playable.sounding_events(channel)) / seconds
                    )
                produced[
                    (
                        label,
                        config["kind"],
                        config["temperature"],
                        config["top_k"] or 0,
                        config["top_p"] or 0.0,
                        seed,
                    )
                ] = playable
                penalty, tie = score(row, corpus_original)
                row["penalty"] = penalty
                row["tiebreak"] = tie
                rows.append(row)
                truncation = (
                    "untruncated"
                    if config["top_k"] is None and config["top_p"] is None
                    else f"k={config['top_k']} p={config['top_p']}"
                )
                print(
                    f"  {label:22} {config['kind']:11} "
                    f"t={config['temperature']:.1f} {truncation:18} seed={seed}  "
                    f"ngram={row['ngram_repetition_rate']:.3f} "
                    f"n/bar={row['notes_per_bar']:.1f} penalty={penalty:.2f}",
                    flush=True,
                )

    # --- duration-matched sensitivity check -------------------------------
    # The nominal-bar reference and the generated samples do not span the same
    # elapsed time: a MIDI-release bar is 2.0 s at the release's fake 120 BPM,
    # while a sample decoded at Tempo_150 has 1.6 s bars. Re-cut the reference
    # to the median generated duration and re-read the density figures. If the
    # verdict moves, the per-bar comparison was carrying a unit artifact.
    median_seconds = sorted(row["seconds"] for row in rows)[len(rows) // 2] if rows else 0.0
    duration_tagged: list[tuple[str, Any]] = []
    if median_seconds > 0:
        for index, composition in enumerate(compositions):
            if not composition.sounding_events():
                continue
            piece = str(composition.source_metadata.get("composition_id") or f"piece{index}")
            total = composition.time.seconds(composition.end_tick)
            if total < median_seconds:
                continue
            start = 0.0
            while start + median_seconds <= total:
                window = slice_seconds(composition, start, median_seconds)
                if window.sounding_events():
                    duration_tagged.append(
                        (
                            piece,
                            evaluate_composition(window, profile, corpus_fingerprints=fingerprints),
                        )
                    )
                start += median_seconds
    corpus_by_duration = (
        aggregate_by_piece("corpus original (duration-matched)", duration_tagged)
        if duration_tagged
        else None
    )
    if corpus_by_duration is not None:
        print(
            f"\n  duration-matched reference: windows of {median_seconds:.1f}s "
            f"(median generated length), {len(duration_tagged)} windows from "
            f"{len({q for q, _ in duration_tagged})} pieces"
        )
        print(
            f"    ngram {corpus_by_duration.medians['ngram_repetition_rate']:.3f} "
            f"(IQR {corpus_by_duration.p25['ngram_repetition_rate']:.3f}-"
            f"{corpus_by_duration.p75['ngram_repetition_rate']:.3f})   "
            f"notes/bar {corpus_by_duration.medians['notes_per_bar']:.1f}   "
            f"vs nominal-bar reference ngram "
            f"{corpus_original.medians['ngram_repetition_rate']:.3f} / notes/bar "
            f"{corpus_original.medians['notes_per_bar']:.1f}"
        )

    # --- outputs -----------------------------------------------------------
    fields = sorted({k for row in rows for k in row})
    csv_path = args.out / "sweep.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "reference": {
            "split": args.reference_split,
            "window_bars": args.bars,
            "unique_source_pieces": unique_pieces,
            "n_windows": len(original_tagged),
            "window_selection": (
                "consecutive non-overlapping windows of the piece's own fitted "
                "bars from bar zero; tails shorter than a full window dropped; "
                "empty windows dropped; pieces shorter than one window excluded"
            ),
            "aggregation": (
                "per piece: each source piece contributes the mean of its own "
                "windows, so one long piece does not outweigh several short ones"
            ),
            "original_per_piece": corpus_original.to_dict(),
            "original_per_window": corpus_by_window.to_dict(),
            "roundtrip_per_piece": corpus_roundtrip.to_dict(),
            "duration_matched": (
                corpus_by_duration.to_dict() if corpus_by_duration is not None else None
            ),
            "duration_matched_seconds": median_seconds,
        },
        "ranking_is_diagnostic_only": (
            "A rank orders configurations by distance from corpus statistics. "
            "It does not certify musical quality: no metric here sees phrase "
            "development, and listening remains required."
        ),
        "weights": WEIGHTS,
        "guardrails": list(GUARDRAILS),
        "rows": rows,
    }
    (args.out / "sweep.json").write_text(json.dumps(report, indent=2, default=str))

    # Rank configurations by mean penalty across seeds, after the filters.
    groups: dict[tuple, list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["checkpoint"], row["kind"], row["temperature"], row["top_k"], row["top_p"])
        groups.setdefault(key, []).append(row)

    ranked = []
    for key, group in groups.items():
        valid = [r for r in group if r["hardware_compliant"] and not r["flagged_as_memorized"]]
        if not valid:
            continue
        seed_compositions = [
            produced[(key[0], key[1], key[2], key[3], key[4], r["seed"])]
            for r in valid
            if (key[0], key[1], key[2], key[3], key[4], r["seed"]) in produced
        ]
        diversity = 1.0 - inter_seed_similarity(seed_compositions)
        means = {
            metric: sum(r[metric] for r in valid) / len(valid)
            for metric in (
                *WEIGHTS,
                *GUARDRAILS,
                "notes_per_second",
                "unique_bar_fraction",
                "longest_identical_bar_run",
                "token_ngram_repetition",
                "penalty",
                "tiebreak",
            )
        }
        spread = max(r["ngram_repetition_rate"] for r in valid) - min(
            r["ngram_repetition_rate"] for r in valid
        )
        ranked.append(
            {
                "key": key,
                "n_valid": len(valid),
                "ngram_spread": spread,
                "seed_diversity": diversity,
                "collapsed": bool(diversity < 0.5 and means["ngram_repetition_rate"] > 0.3),
                **means,
            }
        )
    ranked.sort(key=lambda r: (r["penalty"], r["tiebreak"]))

    print(
        f"\n{'rank':>4} {'checkpoint':22}{'cfg':>12}{'ngram':>8}{'n/sec':>7}"
        f"{'step':>7}{'uniqbar':>8}{'run':>5}{'div':>6}{'pen':>7}{'spread':>8}"
    )
    for index, row in enumerate(ranked[:20], start=1):
        ck, kind, t, k, p = row["key"]
        if kind == "temperature":
            cfg = f"t{t:.1f}"
        elif kind == "untruncated":
            cfg = f"raw t{t:.1f}"
        elif kind == "top_k":
            cfg = f"k{k}"
        else:
            cfg = f"p{p:.2f}"
        print(
            f"{index:>4} {ck:22}{cfg:>12}"
            f"{row['ngram_repetition_rate']:>8.3f}"
            f"{row['notes_per_second']:>7.1f}{row['stepwise_fraction']:>7.3f}"
            f"{row['unique_bar_fraction']:>8.3f}{row['longest_identical_bar_run']:>5.1f}"
            f"{row['seed_diversity']:>6.2f}{row['penalty']:>7.2f}"
            f"{row['ngram_spread']:>8.3f}"
            f"{'  COLLAPSED' if row['collapsed'] else ''}"
        )
    print(
        "\n  div = cross-seed diversity: 1.0 = the seeds share nothing, 0.0 = they\n"
        "  are identical. COLLAPSED marks repetition bought by every seed making\n"
        "  the same piece, which lifts the metric without improving the generator.\n"
        "  The rank orders distance from corpus statistics; it certifies nothing\n"
        "  musical, since no metric here sees phrase development."
    )
    print(
        f"\n  corpus target: ngram {corpus_original.medians['ngram_repetition_rate']:.3f} "
        f"(IQR {corpus_original.p25['ngram_repetition_rate']:.3f}-"
        f"{corpus_original.p75['ngram_repetition_rate']:.3f}), "
        f"notes/bar {corpus_original.medians['notes_per_bar']:.1f} "
        f"(IQR {corpus_original.p25['notes_per_bar']:.1f}-"
        f"{corpus_original.p75['notes_per_bar']:.1f})"
    )
    # Representative output, best-scoring first, for listening.
    if args.keep_events:
        from rse.jsonio import write_json

        events_dir = args.out / "events"
        events_dir.mkdir(parents=True, exist_ok=True)
        per_checkpoint: dict[str, int] = {}
        for entry in ranked:
            ck, kind, temperature, top_k, top_p = entry["key"]
            if per_checkpoint.get(ck, 0) >= args.keep_events:
                continue
            seed = args.seeds[0]
            composition = produced.get((ck, kind, temperature, top_k, top_p, seed))
            if composition is None:
                continue
            per_checkpoint[ck] = per_checkpoint.get(ck, 0) + 1
            if kind == "temperature":
                tag = f"t{temperature:.1f}"
            elif kind == "top_k":
                tag = f"k{top_k}"
            else:
                tag = f"p{top_p:.2f}"
            write_json(events_dir / f"{ck}_{tag}_s{seed}.json", composition.to_dict())
        print(f"  wrote {sum(per_checkpoint.values())} event files to {events_dir}")
        print("  render them with: rse render --record <file> --out <name>")

    print(f"  wrote {csv_path} and {args.out / 'sweep.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
