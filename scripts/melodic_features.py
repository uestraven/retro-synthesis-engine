"""Melodic feature extraction across NES-MDB's official splits.

Purpose: find out whether a melodic-quality score could define both a curated
training subset and a held-out evaluation set. The franchise route cannot --
Final Fantasy, Dragon Quest, Mega Man, Castlevania, Ninja Gaiden and Kirby are
563 tracks and every one is in the train split, with zero in test, because the
splits are composer-disjoint. A fine-tune on those games could be trained but
not measured. A feature-based score ignores game names, so it can find melodic
material inside valid and test.

Nothing here decides anything. Thresholds are reported at several levels to
show how coverage responds; weights and a composite score wait on the
repetition-distribution report and on listening.

Where each feature is measured, and why
---------------------------------------
Bar-level features are measured on **round-tripped** data only. Bar signatures
are exact (offset, channel, value) tuples, so unquantized MIDI micro-timing
means no two bars ever match: on original files unique-bar fraction reads 1.000
and identical-bar runs 0.00 for every piece, which says nothing. Quantizing is
what makes bar structure visible, and the quantized form is also what the model
trains on.

Pitch and interval features are measured on **original** MIDI, where the lead
line is untouched by the row collapse.

Rhythm features are reported on **both**, because they are the ones the
representation alters: the row grid quantizes onsets and folds intra-row events
together, so a figure that is varied in the source can read as uniform after a
round trip.

No composer metadata exists in the ingested records -- only game and track
name -- so game count is the diversity proxy. Composer disjointness is a
property of the official splits rather than something checkable per piece.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Importable when run as a file from a checkout where the package is not
# installed, which is the normal state of a fresh GPU host: `python
# scripts/x.py` puts `scripts/` on sys.path, not the repository root.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import json
import math
import re
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
    ngram_repetition_rate,
)

#: Minimum eligible 4-grams before motif figures mean anything.
MIN_GRAMS = 8

FRANCHISES = (
    r"FinalFantasy|DragonWarrior|DragonQuest|MegaMan|Rockman|Castlevania"
    r"|AkumajouDracula|NinjaGaiden|NinjaRyukenden|Kirby|Zelda"
)


def _game_of(name: str) -> str:
    """Game name from a NES-MDB relpath like `.../001_1942_00_01Start.mid`."""
    stem = name.split("/")[-1]
    parts = stem.split("_")
    return parts[1] if len(parts) > 1 else "unknown"


def _intervals(pitches: list[int]) -> list[int]:
    return [b - a for a, b in zip(pitches, pitches[1:], strict=False)]


def melodic_continuity(pitches: list[int]) -> dict[str, float]:
    """Interval size, direction changes, and span. Descriptive, not a verdict."""
    intervals = _intervals(pitches)
    if not intervals:
        return {"mean_abs_interval": 0.0, "direction_change_rate": 0.0, "melodic_span": 0.0}
    directions = [1 if i > 0 else (-1 if i < 0 else 0) for i in intervals]
    moving = [d for d in directions if d]
    changes = sum(1 for a, b in zip(moving, moving[1:], strict=False) if a != b)
    return {
        "mean_abs_interval": statistics.fmean(abs(i) for i in intervals),
        "direction_change_rate": changes / max(1, len(moving) - 1),
        "melodic_span": float(max(pitches) - min(pitches)),
    }


def motif_recurrence(pitches: list[int], n: int = REPETITION_NGRAM) -> dict[str, float]:
    """Exact-pitch and interval (transposition-invariant) recurrence.

    The gap between them is how much the music restates motifs at a different
    pitch. `interval` is the project's long-standing metric, reused unchanged.
    """
    interval_rate = ngram_repetition_rate(pitches, n)
    # n intervals span n+1 notes. Comparing against n-note pitch grams
    # measures different window lengths, and the shorter window repeats
    # more often, so the gap came out at exactly 0.000 for every piece.
    width = n + 1
    exact = 0.0
    if len(pitches) >= width:
        grams = [tuple(pitches[i : i + width]) for i in range(len(pitches) - width + 1)]
        counts = Counter(grams)
        exact = sum(c for c in counts.values() if c > 1) / len(grams)
    return {
        "motif_interval": interval_rate,
        "motif_exact": exact,
        "motif_transposed_gap": max(0.0, interval_rate - exact),
    }


def motif_variation(pitches: list[int], n: int = REPETITION_NGRAM) -> dict[str, float]:
    """Motifs that return altered rather than verbatim.

    A near-duplicate is an interval n-gram differing in exactly one position
    from another n-gram that occurs elsewhere. Counted only where there is no
    verbatim duplicate, so literal repetition is not scored as variation --
    that distinction is the whole point, since the model's failure is bar
    duplication while the corpus's strength is altered return.
    """
    intervals = _intervals(pitches)
    if len(intervals) < n:
        return {"variation_rate": 0.0, "n_eligible_grams": 0.0}
    grams = [tuple(intervals[i : i + n]) for i in range(len(intervals) - n + 1)]
    counts = Counter(grams)
    varied = 0
    unique = list(counts)
    for gram in grams:
        if counts[gram] > 1:
            continue
        for other in unique:
            if other == gram:
                continue
            if sum(1 for a, b in zip(gram, other, strict=False) if a != b) == 1:
                varied += 1
                break
    return {"variation_rate": varied / len(grams), "n_eligible_grams": float(len(grams))}


def bar_structure(composition: Any) -> dict[str, float]:
    """Bar diversity and reuse. ROUND-TRIPPED data only -- see module docstring."""
    signatures = bar_signatures(composition)
    if len(signatures) < 2:
        return {
            "bar_diversity": 1.0,
            "bars_consecutive_dup": 0.0,
            "bars_nonconsecutive_reuse": 0.0,
            "longest_identical_run": 0.0,
        }
    consecutive = sum(1 for i in range(1, len(signatures)) if signatures[i] == signatures[i - 1])
    nonconsecutive = sum(
        1
        for i in range(len(signatures))
        if any(signatures[i] == signatures[j] for j in range(i + 2, len(signatures)))
    )
    longest = run = 0
    for i in range(1, len(signatures)):
        run = run + 1 if signatures[i] == signatures[i - 1] else 0
        longest = max(longest, run + 1 if run else 0)
    return {
        "bar_diversity": len(set(signatures)) / len(signatures),
        "bars_consecutive_dup": consecutive / (len(signatures) - 1),
        "bars_nonconsecutive_reuse": nonconsecutive / len(signatures),
        "longest_identical_run": float(longest),
    }


def phrase_and_rhythm(composition: Any) -> dict[str, float]:
    """Activity, rests, onset spacing and rhythmic variety.

    Rests are reported as measurements, not treated as phrase boundaries: a
    rest can be articulation, and a phrase can turn without one.
    """
    time = composition.time
    events = composition.sounding_events()
    if not events:
        return {}
    starts = sorted(time.seconds(e.tick) for e in events)
    durations = [max(0.0, time.seconds(e.tick + e.duration) - time.seconds(e.tick)) for e in events]
    iois = [b - a for a, b in zip(starts, starts[1:], strict=False) if b > a]
    span = max(starts[-1] - starts[0], 1e-9)
    silence = sum(1 for gap in iois if gap >= 0.25)
    ioi_entropy = 0.0
    if iois:
        buckets = Counter(round(math.log2(max(g, 1e-3)) * 2) for g in iois)
        total = sum(buckets.values())
        ioi_entropy = -sum((c / total) * math.log2(c / total) for c in buckets.values() if c)
    return {
        "notes_per_second": len(events) / span,
        "mean_note_seconds": statistics.fmean(durations) if durations else 0.0,
        "note_length_cv": (
            statistics.pstdev(durations) / statistics.fmean(durations)
            if len(durations) > 1 and statistics.fmean(durations) > 0
            else 0.0
        ),
        "mean_ioi_seconds": statistics.fmean(iois) if iois else 0.0,
        "ioi_entropy": ioi_entropy,
        "rest_rate_per_second": silence / span,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="nesmdb")
    parser.add_argument("--release", default="midi")
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--datasets-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("melodic_features.json"))
    args = parser.parse_args()

    from rse.training.corpus import corpus_from_release

    profile = load_profile("nes", configs_dir=paths.HARDWARE_CONFIGS_DIR)
    tokenizer = RseTokenizer()
    rows: list[dict[str, Any]] = []

    for split in args.splits:
        entries, _info = corpus_from_release(
            args.dataset,
            args.release,
            datasets_dir=args.datasets_dir,
            limit=args.limit,
            splits=(split,),
            seed=0,
        )
        print(f"{split}: {len(entries)} pieces", flush=True)
        for index, (composition, _s, cid) in enumerate(entries):
            # composition_id is a hash; the game name lives in the source
            # relpath, e.g. nesmdb_midi/train/001_1942_00_01Start.mid
            name = str(
                (composition.source_metadata.get("source") or {}).get("relpath")
                or composition.source_metadata.get("name")
                or cid
            )
            if index and index % 300 == 0:
                print(f"  {split} {index}/{len(entries)}", flush=True)
            if not composition.sounding_events():
                continue
            pitches = melody_pitches(composition, "P1") or melody_pitches(composition)
            decoded = tokenizer.decode(tokenizer.encode(composition).ids).composition
            if not decoded.sounding_events():
                continue
            metrics = evaluate_composition(composition, profile)
            row: dict[str, Any] = {
                "piece": str(cid),
                "split": split,
                "name": name,
                "game": _game_of(name),
                "franchise": bool(re.search(FRANCHISES, name, re.I)),
                "seconds": composition.time.seconds(composition.end_tick),
                "n_notes": metrics.n_notes,
                "n_channels_used": metrics.n_channels_used,
                "pitch_entropy": metrics.pitch_class_entropy,
                "in_scale": metrics.in_scale_fraction,
                "stepwise": metrics.stepwise_fraction,
                **{f"orig_{k}": v for k, v in melodic_continuity(pitches).items()},
                **{f"orig_{k}": v for k, v in motif_recurrence(pitches).items()},
                **{f"orig_{k}": v for k, v in motif_variation(pitches).items()},
                **{f"rt_{k}": v for k, v in bar_structure(decoded).items()},
                **{f"orig_{k}": v for k, v in phrase_and_rhythm(composition).items()},
                **{f"rt_{k}": v for k, v in phrase_and_rhythm(decoded).items()},
            }
            row["eligible"] = row.get("orig_n_eligible_grams", 0) >= MIN_GRAMS
            rows.append(row)

    usable = [r for r in rows if r["eligible"]]
    report: dict[str, Any] = {
        "n_rows": len(rows),
        "n_eligible": len(usable),
        "min_eligible_grams": MIN_GRAMS,
        "feature_sources": {
            "pitch_and_interval": "original MIDI",
            "bar_structure": "round-tripped only (bar signatures never match on raw MIDI)",
            "rhythm": "both original and round-tripped, reported separately",
        },
        "composer_counts": "unavailable: records carry game and track name only",
        "rows": rows,
    }

    numeric = (
        [k for k, v in usable[0].items() if isinstance(v, (int, float)) and not isinstance(v, bool)]
        if usable
        else []
    )

    print(f"\n{'=' * 78}\nFEATURE DISTRIBUTIONS, eligible pieces only")
    print(f"{'feature':34}{'train':>13}{'valid':>13}{'test':>13}")
    report["distributions"] = {}
    for key in numeric:
        cells = []
        entry = {}
        for split in args.splits:
            vals = [r[key] for r in usable if r["split"] == split]
            if vals:
                med = statistics.median(vals)
                entry[split] = {
                    "median": round(med, 4),
                    "p25": round(sorted(vals)[int(0.25 * (len(vals) - 1))], 4),
                    "p75": round(sorted(vals)[int(0.75 * (len(vals) - 1))], 4),
                    "n": len(vals),
                }
                cells.append(f"{med:>13.3f}")
            else:
                cells.append(f"{'-':>13}")
        report["distributions"][key] = entry
        print(f"{key:34}" + "".join(cells))

    print(f"\n{'=' * 78}\nCANDIDATE SELECTION RULES (derived from train, applied unchanged)")
    print("These are illustrative thresholds to show how coverage responds.")
    print("Weights and a final score wait on the distribution report and listening.\n")
    rules = {
        "lenient": {
            "orig_motif_interval": 0.30,
            "rt_bar_diversity": 0.60,
            "orig_variation_rate": 0.05,
            "seconds": 8.0,
        },
        "moderate": {
            "orig_motif_interval": 0.40,
            "rt_bar_diversity": 0.70,
            "orig_variation_rate": 0.10,
            "seconds": 12.0,
        },
        "strict": {
            "orig_motif_interval": 0.50,
            "rt_bar_diversity": 0.75,
            "orig_variation_rate": 0.15,
            "seconds": 16.0,
        },
    }
    report["rules"] = {}
    print(
        f"{'rule':10}{'split':7}{'pieces':>8}{'games':>7}{'franchise':>11}"
        f"{'median s':>10}{'8-bar win':>11}"
    )
    for name, rule in rules.items():
        report["rules"][name] = {"thresholds": rule, "splits": {}}
        for split in args.splits:
            chosen = [
                r
                for r in usable
                if r["split"] == split and all(r.get(k, 0) >= v for k, v in rule.items())
            ]
            windows = sum(int(r["seconds"] // 16.0) for r in chosen)
            entry = {
                "pieces": len(chosen),
                "games": len({r["game"] for r in chosen}),
                "franchise_pieces": sum(1 for r in chosen if r["franchise"]),
                "median_seconds": (
                    round(statistics.median([r["seconds"] for r in chosen]), 1) if chosen else 0.0
                ),
                "approx_8bar_windows": windows,
            }
            report["rules"][name]["splits"][split] = entry
            print(
                f"{name:10}{split:7}{entry['pieces']:>8}{entry['games']:>7}"
                f"{entry['franchise_pieces']:>11}{entry['median_seconds']:>10.1f}"
                f"{windows:>11}"
            )

    args.out.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n  wrote {args.out}")
    print(f"  {len(usable)} of {len(rows)} pieces have >= {MIN_GRAMS} eligible 4-grams")
    print("  NOTE: thresholds above are illustrative. Nothing is tuned on valid or")
    print("  test, and the test split stays reserved for final evaluation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
