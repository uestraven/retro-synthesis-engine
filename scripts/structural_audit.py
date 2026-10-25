"""Structural audit: what the representation costs, and what zero repetition means.

Five questions, answered before any further GPU time.

1. The zero-repetition population. 22% of round-tripped windows score zero
   interval repetition. Some of that is music that genuinely never restates a
   motif; some is a piece with too few notes to form an n-gram at all. Those
   are different facts and must not be pooled.

2. Rhythmic loss, isolated. Round-tripping drops IOI entropy from 2.25 to 0.52
   and coarsens the mean inter-onset interval from 0.058 s to 0.158 s. But the
   row collapse also *removes* notes -- about 12% per piece, being intra-row
   arpeggio and vibrato writes -- and removing notes changes inter-onset
   intervals mechanically, whether or not any surviving onset moved. Comparing
   only the notes that survive separates quantization from deletion. Without
   that split the fourfold entropy drop cannot be attributed.

3. Full distributions rather than medians, for the rhythm features.

4. Joint behaviour: motif recurrence against rhythmic diversity against
   consecutive bar duplication, so "repetitive" is never read as "looping".

5. Piece-weighted references with uncertainty from resampling source pieces.
   Windows from one piece are correlated, so treating them as independent
   observations understates the interval.

Reads the existing repetition_train.json and melodic_features.json for items
1, 3, 4 and 5. Item 2 needs fresh computation and is sampled.
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
import random
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from rse import paths
from rse.events.schema import Note
from rse.hardware.profile import load_profile
from rse.tokenizers.tokenizer import RseTokenizer


def ioi_entropy(times: list[float]) -> float:
    """Entropy of inter-onset intervals, bucketed by log2 half-steps."""
    ordered = sorted(times)
    iois = [b - a for a, b in zip(ordered, ordered[1:], strict=False) if b > a]
    if not iois:
        return 0.0
    buckets = Counter(round(math.log2(max(g, 1e-3)) * 2) for g in iois)
    total = sum(buckets.values())
    return -sum((c / total) * math.log2(c / total) for c in buckets.values() if c)


def bootstrap_ci(values: list[float], *, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    if len(values) < 2:
        return (values[0] if values else 0.0, values[0] if values else 0.0)
    rng = random.Random(seed)
    medians = sorted(statistics.median(rng.choices(values, k=len(values))) for _ in range(n))
    return (medians[int(0.025 * n)], medians[int(0.975 * n)])


def quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)

    def q(p: float) -> float:
        return round(ordered[int(p * (len(ordered) - 1))], 4)

    return {
        "n": len(ordered),
        "p05": q(0.05),
        "p25": q(0.25),
        "median": round(statistics.median(ordered), 4),
        "p75": q(0.75),
        "p95": q(0.95),
        "mean": round(statistics.fmean(ordered), 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetition", type=Path, default=Path("repetition_train.json"))
    parser.add_argument("--melodic", type=Path, default=Path("melodic_features.json"))
    parser.add_argument("--matched-sample", type=int, default=400)
    parser.add_argument("--dataset", default="nesmdb")
    parser.add_argument("--release", default="midi")
    parser.add_argument("--datasets-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("structural_audit.json"))
    args = parser.parse_args()

    report: dict[str, Any] = {}
    rep = json.loads(args.repetition.read_text())
    mel = json.loads(args.melodic.read_text())
    rep_rows: list[dict[str, Any]] = rep["rows"]
    mel_rows: list[dict[str, Any]] = mel["rows"]

    # ---- 1. the zero-repetition population --------------------------------
    print(f"{'=' * 78}\n1. WHAT ZERO REPETITION MEANS (round-tripped 8-bar windows)")
    windows = [r for r in rep_rows if r["kind"] == "rt_window"]
    too_few = [r for r in windows if r.get("excluded")]
    eligible = [r for r in windows if not r.get("excluded")]
    zero = [r for r in eligible if r["ngram_interval"] <= 0.01]
    nonzero = [r for r in eligible if r["ngram_interval"] > 0.01]
    print(f"  windows total                      {len(windows)}")
    print(
        f"  excluded, under 8 eligible 4-grams {len(too_few)}"
        f"  ({100 * len(too_few) / max(1, len(windows)):.0f}%)"
    )
    print(f"  eligible                           {len(eligible)}")
    print(
        f"  of those, genuinely zero           {len(zero)}"
        f"  ({100 * len(zero) / max(1, len(eligible)):.0f}%)"
    )
    for label, subset in (("genuinely zero", zero), ("any repetition", nonzero)):
        if not subset:
            continue
        g = statistics.median([r["eligible_grams"] for r in subset])
        nps = statistics.median([r["notes_per_second"] for r in subset])
        step = statistics.median([r["stepwise"] for r in subset])
        pieces = len({r["piece"] for r in subset})
        print(
            f"    {label:16} grams median {g:6.0f}   notes/sec {nps:5.1f}"
            f"   stepwise {step:.3f}   uniq pieces {pieces}"
        )
    report["zero_population"] = {
        "windows": len(windows),
        "excluded_insufficient_grams": len(too_few),
        "eligible": len(eligible),
        "genuinely_zero": len(zero),
        "genuinely_zero_unique_pieces": len({r["piece"] for r in zero}),
        "zero_grams": quantiles([float(r["eligible_grams"]) for r in zero]),
        "nonzero_grams": quantiles([float(r["eligible_grams"]) for r in nonzero]),
    }

    # ---- 2. rhythmic loss, matched notes only ------------------------------
    print(f"\n{'=' * 78}\n2. RHYTHMIC LOSS: quantization or deletion?")
    from rse.training.corpus import corpus_from_release

    profile = load_profile("nes", configs_dir=paths.HARDWARE_CONFIGS_DIR)
    del profile
    tokenizer = RseTokenizer()
    entries, _info = corpus_from_release(
        args.dataset,
        args.release,
        datasets_dir=args.datasets_dir,
        limit=args.matched_sample,
        splits=("train",),
        seed=0,
    )
    all_orig, all_rt, matched_orig, matched_rt, kept_share = [], [], [], [], []
    for composition, _s, _cid in entries:
        if not composition.sounding_events():
            continue
        fit = tokenizer.fit_grid(composition)
        decoded = tokenizer.decode(tokenizer.encode(composition).ids).composition
        if not decoded.sounding_events():
            continue
        time_o, time_d = composition.time, decoded.time

        def key(event: Any, fit_: Any = fit) -> tuple[int, str, int]:
            row = int(round((event.tick - fit_.phase_ticks) / fit_.step_ticks))
            return (row, event.channel_id, event.pitch if isinstance(event, Note) else event.value)

        step_d = tokenizer.ticks_per_step(decoded.ticks_per_beat)
        originals = {}
        for event in composition.sounding_events():
            originals.setdefault(key(event), time_o.seconds(event.tick))
        decoded_keys = {}
        for event in decoded.sounding_events():
            row = int(round(event.tick / step_d))
            k = (row, event.channel_id, event.pitch if isinstance(event, Note) else event.value)
            decoded_keys.setdefault(k, time_d.seconds(event.tick))

        shared = [k for k in decoded_keys if k in originals]
        if len(shared) < 10:
            continue
        kept_share.append(len(shared) / max(1, len(originals)))
        all_orig.append(
            ioi_entropy([time_o.seconds(e.tick) for e in composition.sounding_events()])
        )
        all_rt.append(ioi_entropy([time_d.seconds(e.tick) for e in decoded.sounding_events()]))
        matched_orig.append(ioi_entropy([originals[k] for k in shared]))
        matched_rt.append(ioi_entropy([decoded_keys[k] for k in shared]))

    print(
        f"  pieces compared {len(matched_orig)}; notes surviving the row collapse: "
        f"median {100 * statistics.median(kept_share):.0f}%"
    )
    print(f"  {'IOI entropy':28}{'original':>11}{'round-trip':>12}{'drop':>8}")
    print(
        f"  {'all notes':28}{statistics.median(all_orig):>11.3f}"
        f"{statistics.median(all_rt):>12.3f}"
        f"{statistics.median(all_orig) - statistics.median(all_rt):>8.3f}"
    )
    print(
        f"  {'matched surviving notes only':28}{statistics.median(matched_orig):>11.3f}"
        f"{statistics.median(matched_rt):>12.3f}"
        f"{statistics.median(matched_orig) - statistics.median(matched_rt):>8.3f}"
    )
    report["rhythmic_loss"] = {
        "pieces": len(matched_orig),
        "median_notes_kept": round(statistics.median(kept_share), 4),
        "all_notes": {
            "original": quantiles(all_orig),
            "roundtrip": quantiles(all_rt),
        },
        "matched_notes": {
            "original": quantiles(matched_orig),
            "roundtrip": quantiles(matched_rt),
        },
        "interpretation": (
            "The gap on matched notes is quantization alone. The extra gap on "
            "all notes is deletion of intra-row events."
        ),
    }

    # ---- 3. full rhythm distributions --------------------------------------
    print(f"\n{'=' * 78}\n3. RHYTHM DISTRIBUTIONS, original against round-trip (whole pieces)")
    usable_mel = [r for r in mel_rows if r.get("eligible") and r["split"] == "train"]
    print(f"  {'feature':24}{'p05':>8}{'p25':>8}{'median':>9}{'p75':>8}{'p95':>8}")
    report["rhythm_distributions"] = {}
    for base in ("ioi_entropy", "mean_ioi_seconds", "note_length_cv", "notes_per_second"):
        for prefix in ("orig", "rt"):
            key_name = f"{prefix}_{base}"
            vals = [r[key_name] for r in usable_mel if key_name in r]
            if not vals:
                continue
            stats = quantiles(vals)
            report["rhythm_distributions"][key_name] = stats
            print(
                f"  {key_name:24}{stats['p05']:>8.3f}{stats['p25']:>8.3f}"
                f"{stats['median']:>9.3f}{stats['p75']:>8.3f}{stats['p95']:>8.3f}"
            )

    # ---- 4. joint: recurrence x rhythm x consecutive duplication ----------
    print(f"\n{'=' * 78}\n4. JOINT: recurrence against rhythmic diversity and bar duplication")
    print(
        f"  {'motif band':13}{'n':>7}{'pieces':>8}{'ioi_ent':>9}{'consec_dup':>12}"
        f"{'nonconsec':>11}{'variation':>11}"
    )
    report["joint"] = {}
    for low, high in ((0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)):
        subset = [r for r in usable_mel if low <= r.get("orig_motif_interval", -1) < high]
        if not subset:
            continue
        row = {
            "n": len(subset),
            "unique_pieces": len({r["piece"] for r in subset}),
            "rt_ioi_entropy": round(statistics.fmean(r["rt_ioi_entropy"] for r in subset), 3),
            "rt_bars_consecutive_dup": round(
                statistics.fmean(r["rt_bars_consecutive_dup"] for r in subset), 4
            ),
            "rt_bars_nonconsecutive_reuse": round(
                statistics.fmean(r["rt_bars_nonconsecutive_reuse"] for r in subset), 4
            ),
            "variation_rate": round(statistics.fmean(r["orig_variation_rate"] for r in subset), 4),
        }
        report["joint"][f"{low:.1f}-{high:.1f}"] = row
        print(
            f"  {f'{low:.1f}-{high:.1f}':13}{row['n']:>7}{row['unique_pieces']:>8}"
            f"{row['rt_ioi_entropy']:>9.3f}{row['rt_bars_consecutive_dup']:>12.4f}"
            f"{row['rt_bars_nonconsecutive_reuse']:>11.4f}{row['variation_rate']:>11.4f}"
        )

    # ---- 5. piece-weighted references with resampled uncertainty ----------
    print(f"\n{'=' * 78}\n5. PIECE-WEIGHTED REFERENCE, 95% CI by resampling PIECES")
    print(f"  {'metric':30}{'per window':>12}{'per piece':>11}{'95% CI':>20}")
    report["piece_weighted"] = {}
    for label, key_name in (
        ("interval repetition", "ngram_interval"),
        ("notes per second", "notes_per_second"),
        ("stepwise fraction", "stepwise"),
    ):
        per_window = [r[key_name] for r in eligible if key_name in r]
        grouped: dict[str, list[float]] = {}
        for r in eligible:
            if key_name in r:
                grouped.setdefault(r["piece"], []).append(r[key_name])
        per_piece = [statistics.fmean(v) for v in grouped.values()]
        if not per_piece:
            continue
        low, high = bootstrap_ci(per_piece)
        report["piece_weighted"][key_name] = {
            "per_window_median": round(statistics.median(per_window), 4),
            "per_piece_median": round(statistics.median(per_piece), 4),
            "ci95": [round(low, 4), round(high, 4)],
            "n_pieces": len(per_piece),
            "n_windows": len(per_window),
        }
        print(
            f"  {label:30}{statistics.median(per_window):>12.3f}"
            f"{statistics.median(per_piece):>11.3f}"
            f"{f'{low:.3f}-{high:.3f}':>20}"
        )

    args.out.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
