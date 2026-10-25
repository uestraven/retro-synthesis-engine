"""Gate 2: do the conditioning labels survive conversion to model-visible events?

The question is not whether the labels describe the source music. It is whether
they still describe what the model sees. Round-tripping drops IOI entropy from
2.358 to 0.686 on matched surviving notes -- 97% of that loss is quantizing
onsets rather than deleting notes -- so a rhythmic distinction present in the
MIDI may be absent from the token stream. Labelling a piece `driving` on
evidence the tokenizer destroys would teach the model nothing.

Phrase survival is the sharper worry and is reported separately. If the row
grid's rhythmic flattening pushes restatements above the `PHRASE_SAME`
threshold, every round-tripped return looks verbatim and `variation` becomes
unlearnable by construction. VARIATION -> STABLE, VARIATION -> UNKNOWN and
STABLE -> VARIATION are counted explicitly.

Thresholds are fitted on train only and then frozen. Nothing here is
recalibrated against validation or test; those splits are measured with the
frozen extractor.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from rse.conditioning import (
    PHRASE_BARS,
    PHRASE_RELATED,
    PHRASE_SAME,
    UNKNOWN,
    ConditioningRecord,
    _similarity,
    agreement,
    eligibility,
    extract_features,
    fit_thresholds,
    label,
    melody_line,
)
from rse.events.pulse import detect_pulse
from rse.events.schema import Note
from rse.tokenizers.tokenizer import RseTokenizer
from rse.tokenizers.vocab import TokenizerConfig

AXES = ("motif", "rhythm", "phrase")


def game_of(name: str) -> str:
    stem = name.split("/")[-1]
    parts = stem.split("_")
    return parts[1] if len(parts) > 1 else "unknown"


def phrase_pairs(composition: Any) -> list[tuple[float, int]]:
    """Similarity of every phrase pair, with the piece's phrase count.

    Returned per piece so the threshold audit can account for pairs within one
    composition being correlated: a piece contributing 40 pairs is not 40
    independent observations.
    """
    fit = detect_pulse(composition)
    if not fit.found or fit.period_ticks <= 0:
        return []
    channel, _ = melody_line(composition)
    if not channel:
        return []
    notes = sorted(
        (e for e in composition.sounding_events(channel) if isinstance(e, Note)),
        key=lambda e: e.tick,
    )
    if len(notes) < 24:
        return []
    span = fit.period_ticks * 16 * PHRASE_BARS
    phrases: list[list[tuple[int, int]]] = []
    start = notes[0].tick
    while start < composition.end_tick:
        inside = [n for n in notes if start <= n.tick < start + span]
        if len(inside) >= 4:
            phrases.append(
                [(n.pitch, int(round(n.duration / max(fit.period_ticks, 1e-9)))) for n in inside]
            )
        start += span
    out = []
    for i in range(len(phrases)):
        for j in range(i + 1, len(phrases)):
            out.append((_similarity(phrases[i], phrases[j]), len(phrases)))
    return out


def confusion(records: list[dict[str, Any]], axis: str) -> dict[str, dict[str, int]]:
    matrix: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in records:
        matrix[row["orig"][axis]][row["rt"][axis]] += 1
    return {k: dict(v) for k, v in matrix.items()}


def precision_recall(records: list[dict[str, Any]], axis: str) -> dict[str, dict[str, float]]:
    """Per-class precision and recall, treating the original label as truth."""
    labels = sorted({r["orig"][axis] for r in records} | {r["rt"][axis] for r in records})
    out: dict[str, dict[str, float]] = {}
    for name in labels:
        tp = sum(1 for r in records if r["orig"][axis] == name and r["rt"][axis] == name)
        fp = sum(1 for r in records if r["orig"][axis] != name and r["rt"][axis] == name)
        fn = sum(1 for r in records if r["orig"][axis] == name and r["rt"][axis] != name)
        out[name] = {
            "support": tp + fn,
            "precision": round(tp / (tp + fp), 4) if tp + fp else 0.0,
            "recall": round(tp / (tp + fn), 4) if tp + fn else 0.0,
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="nesmdb")
    parser.add_argument("--release", default="midi")
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--threshold-audit-sample", type=int, default=300)
    parser.add_argument("--datasets-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("conditioning_gate2.json"))
    parser.add_argument("--labels-out", type=Path, default=Path("conditioning_labels.jsonl"))
    # Codec selection, so the round-trip diagnostics can be re-run against a
    # new representation. D4 restores arpeggio and passage events that codec A
    # dropped, which changes motif eligibility and rhythm features -- the old
    # labels cannot simply be carried over.
    parser.add_argument("--codec", default="A")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--tempo-ratio", type=float, default=1.03)
    args = parser.parse_args()

    from rse.training.corpus import corpus_from_release

    tokenizer = RseTokenizer(
        TokenizerConfig(tempo_ratio=args.tempo_ratio)
        if args.codec == "A"
        else TokenizerConfig(
            codec=args.codec, sub_row_divisions=args.k, tempo_ratio=args.tempo_ratio
        )
    )
    print(
        f"codec     : {args.codec}"
        f"{'' if args.codec == 'A' else args.k}/{args.tempo_ratio}, vocab {len(tokenizer)}",
        flush=True,
    )
    loaded: dict[str, list[Any]] = {}
    for split in args.splits:
        entries, _info = corpus_from_release(
            args.dataset,
            args.release,
            datasets_dir=args.datasets_dir,
            limit=args.limit,
            splits=(split,),
            seed=0,
        )
        loaded[split] = entries
        print(f"{split}: {len(entries)} pieces", flush=True)

    # ---- features for both views -----------------------------------------
    rows: list[dict[str, Any]] = []
    for split, entries in loaded.items():
        for index, (composition, _s, cid) in enumerate(entries):
            if index and index % 300 == 0:
                print(f"  {split} {index}/{len(entries)}", flush=True)
            if not composition.sounding_events():
                continue
            decoded = tokenizer.decode(tokenizer.encode(composition).ids).composition
            if not decoded.sounding_events():
                continue
            name = str(
                (composition.source_metadata.get("source") or {}).get("relpath")
                or composition.source_metadata.get("name")
                or cid
            )
            rows.append(
                {
                    "piece": str(cid),
                    "split": split,
                    "game": game_of(name),
                    "orig_features": extract_features(composition),
                    "rt_features": extract_features(decoded),
                }
            )

    # ---- thresholds: train only, then frozen ------------------------------
    train_features = [r["orig_features"] for r in rows if r["split"] == "train"]
    thresholds = fit_thresholds(train_features, split="train")
    print(f"\nthresholds fitted on {thresholds.n_fitted} train pieces: {thresholds.to_dict()}")

    for row in rows:
        row["orig"] = label(row["orig_features"], thresholds)
        row["rt"] = label(row["rt_features"], thresholds)
        row["orig_elig"] = eligibility(row["orig_features"])
        row["rt_elig"] = eligibility(row["rt_features"])
        row["agree"] = agreement(row["orig"], row["rt"])

    report: dict[str, Any] = {
        "thresholds": thresholds.to_dict(),
        "threshold_provenance": "fitted on the train split only, then frozen",
        "composer": "null by design: no composer field exists in the ingested records",
        "n_pieces": len(rows),
    }

    # ---- eligibility ------------------------------------------------------
    print(f"\n{'=' * 78}\nELIGIBILITY, original against round-trip")
    print(f"  {'axis':8}{'split':7}{'orig':>8}{'round-trip':>12}{'both':>8}")
    report["eligibility"] = {}
    for axis in AXES:
        for split in args.splits:
            subset = [r for r in rows if r["split"] == split]
            if not subset:
                continue
            o = sum(1 for r in subset if r["orig_elig"][axis])
            t = sum(1 for r in subset if r["rt_elig"][axis])
            b = sum(1 for r in subset if r["orig_elig"][axis] and r["rt_elig"][axis])
            report["eligibility"][f"{axis}/{split}"] = {
                "n": len(subset),
                "orig": o,
                "rt": t,
                "both": b,
            }
            print(
                f"  {axis:8}{split:7}{100 * o / len(subset):>7.0f}%"
                f"{100 * t / len(subset):>11.0f}%{100 * b / len(subset):>7.0f}%"
            )

    # ---- label distributions ---------------------------------------------
    print(f"\n{'=' * 78}\nLABEL DISTRIBUTIONS (original / round-trip)")
    report["distributions"] = {}
    for axis in AXES:
        print(f"\n  {axis}")
        for split in args.splits:
            subset = [r for r in rows if r["split"] == split]
            if not subset:
                continue
            o = Counter(r["orig"][axis] for r in subset)
            t = Counter(r["rt"][axis] for r in subset)
            report["distributions"][f"{axis}/{split}"] = {"orig": dict(o), "rt": dict(t)}
            names = sorted(set(o) | set(t))
            cells = "  ".join(f"{n}:{o.get(n, 0)}/{t.get(n, 0)}" for n in names)
            print(f"    {split:6} {cells}")

    # ---- confusion, agreement, precision/recall ---------------------------
    print(f"\n{'=' * 78}\nORIGINAL -> ROUND-TRIP CONFUSION (train+valid+test pooled)")
    report["confusion"] = {}
    report["agreement"] = {}
    report["precision_recall"] = {}
    for axis in AXES:
        matrix = confusion(rows, axis)
        report["confusion"][axis] = matrix
        both = [r for r in rows if r["orig_elig"][axis] and r["rt_elig"][axis]]
        agreed = sum(1 for r in both if r["agree"][axis])
        overall = sum(1 for r in rows if r["agree"][axis])
        # Two different things, kept apart: whether the label class is
        # preserved, and whether the axis remains measurable at all.
        elig_same = sum(1 for r in rows if r["orig_elig"][axis] == r["rt_elig"][axis])
        lost = sum(1 for r in rows if r["orig_elig"][axis] and not r["rt_elig"][axis])
        gained = sum(1 for r in rows if not r["orig_elig"][axis] and r["rt_elig"][axis])
        report["agreement"][axis] = {
            "class_preservation_both_eligible": round(agreed / max(1, len(both)), 4),
            "n_both_eligible": len(both),
            "eligibility_preservation": round(elig_same / max(1, len(rows)), 4),
            "eligibility_lost": lost,
            "eligibility_gained": gained,
            "naive_agreement_all_pieces": round(overall / max(1, len(rows)), 4),
        }
        report["precision_recall"][axis] = precision_recall(rows, axis)
        stats = report["agreement"][axis]
        print(
            f"\n  {axis}: CLASS preservation "
            f"{100 * stats['class_preservation_both_eligible']:.0f}% "
            f"(n={stats['n_both_eligible']} both eligible); "
            f"ELIGIBILITY preserved {100 * stats['eligibility_preservation']:.0f}% "
            f"(lost {stats['eligibility_lost']}, gained {stats['eligibility_gained']})"
        )
        for source in sorted(matrix):
            targets = "  ".join(f"{k}:{v}" for k, v in sorted(matrix[source].items()))
            print(f"    {source:10} -> {targets}")
        print(f"    {'class':12}{'support':>9}{'precision':>11}{'recall':>9}")
        for name, stats in sorted(report["precision_recall"][axis].items()):
            print(
                f"    {name:12}{stats['support']:>9}{stats['precision']:>11.3f}"
                f"{stats['recall']:>9.3f}"
            )

    # ---- phrase survival, called out ---------------------------------------
    print(f"\n{'=' * 78}\nPHRASE LABEL SURVIVAL (the axis most at risk)")
    transitions = Counter((r["orig"]["phrase"], r["rt"]["phrase"]) for r in rows)
    named = {
        "variation -> stable": transitions.get(("variation", "stable"), 0),
        "variation -> unknown": transitions.get(("variation", UNKNOWN), 0),
        "variation -> variation": transitions.get(("variation", "variation"), 0),
        "variation -> contrast": transitions.get(("variation", "contrast"), 0),
        "stable -> variation": transitions.get(("stable", "variation"), 0),
    }
    origin_variation = sum(1 for r in rows if r["orig"]["phrase"] == "variation")
    report["phrase_survival"] = {
        **named,
        "n_original_variation": origin_variation,
        "variation_retained_fraction": (
            round(named["variation -> variation"] / origin_variation, 4)
            if origin_variation
            else None
        ),
    }
    accounted = sum(named[k] for k in named if k.startswith("variation ->"))
    report["phrase_survival"]["variation_rows_accounted"] = accounted
    report["phrase_survival"]["variation_unaccounted"] = origin_variation - accounted
    report["phrase_survival"]["all_transitions"] = {
        f"{a} -> {b}": n for (a, b), n in sorted(transitions.items())
    }
    for key, value in named.items():
        share = (
            f" ({100 * value / origin_variation:.0f}% of original variation)"
            if (origin_variation and key.startswith("variation ->"))
            else ""
        )
        print(f"  {key:26}{value:>7}{share}")

    # ---- combination support ----------------------------------------------
    print(
        f"  {'every transition from variation':26}"
        f"{accounted}/{origin_variation} accounted, "
        f"{origin_variation - accounted} unexplained"
    )
    print(f"\n{'=' * 78}\nSUPPORT FOR CONDITION COMBINATIONS (train, original labels)")
    train_rows = [r for r in rows if r["split"] == "train"]
    combos: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in train_rows:
        combos[(r["orig"]["motif"], r["orig"]["rhythm"], r["orig"]["phrase"])].append(r)
    report["combinations"] = {}
    print(f"  {'motif':9}{'rhythm':10}{'phrase':11}{'pieces':>8}{'games':>7}{'rt-agree':>10}")
    for key, subset in sorted(combos.items(), key=lambda kv: -len(kv[1]))[:18]:
        agree_all = sum(1 for r in subset if all(r["agree"][a] for a in AXES))
        entry = {
            "pieces": len(subset),
            "games": len({r["game"] for r in subset}),
            "all_axes_agree": agree_all,
        }
        report["combinations"]["|".join(key)] = entry
        print(
            f"  {key[0]:9}{key[1]:10}{key[2]:11}{len(subset):>8}{entry['games']:>7}{agree_all:>10}"
        )

    target = ("high", "driving", "variation")
    hit = combos.get(target, [])
    report["target_combination"] = {
        "labels": list(target),
        "train_pieces": len(hit),
        "train_games": len({r["game"] for r in hit}),
        "all_axes_survive_roundtrip": sum(1 for r in hit if all(r["agree"][a] for a in AXES)),
    }
    print(
        f"\n  TARGET high+driving+variation: {len(hit)} train pieces from "
        f"{len({r['game'] for r in hit})} games; "
        f"{report['target_combination']['all_axes_survive_roundtrip']} keep all three "
        f"labels after round-tripping"
    )

    # ---- combination support: single, two and three axis, both views -------
    print(f"\n{'=' * 78}\nCOMBINATION SUPPORT, unique compositions (train split)")
    print("  UNKNOWN is excluded from a combination: it means evidence was")
    print("  unavailable, not that the piece belongs to a musical category.\n")
    report["axis_combinations"] = {}
    subsets = (
        ("motif",),
        ("rhythm",),
        ("phrase",),
        ("motif", "rhythm"),
        ("motif", "phrase"),
        ("rhythm", "phrase"),
        ("motif", "rhythm", "phrase"),
    )
    for axes in subsets:
        print(f"  {'+'.join(axes)}")
        counts_o: dict[tuple[str, ...], set[str]] = defaultdict(set)
        counts_r: dict[tuple[str, ...], set[str]] = defaultdict(set)
        games_o: dict[tuple[str, ...], set[str]] = defaultdict(set)
        for r in train_rows:
            key_o = tuple(r["orig"][a] for a in axes)
            key_r = tuple(r["rt"][a] for a in axes)
            if UNKNOWN not in key_o:
                counts_o[key_o].add(r["piece"])
                games_o[key_o].add(r["game"])
            if UNKNOWN not in key_r:
                counts_r[key_r].add(r["piece"])
        entry = {}
        for key in sorted(set(counts_o) | set(counts_r), key=lambda k: -len(counts_o.get(k, ()))):
            n_o, n_r = len(counts_o.get(key, ())), len(counts_r.get(key, ()))
            entry["|".join(key)] = {
                "pieces_original": n_o,
                "pieces_roundtrip": n_r,
                "games_original": len(games_o.get(key, ())),
            }
            print(
                f"    {'|'.join(key):34}orig {n_o:>5}  rt {n_r:>5}  "
                f"games {len(games_o.get(key, ())):>4}"
            )
        report["axis_combinations"]["+".join(axes)] = entry

    # ---- phrase threshold audit -------------------------------------------
    print(f"\n{'=' * 78}\nPHRASE THRESHOLD AUDIT")
    sample = random.Random(11).sample(
        [r for r in loaded.get("train", [])],
        min(args.threshold_audit_sample, len(loaded.get("train", []))),
    )
    per_piece: list[list[float]] = []
    lengths: list[int] = []
    for composition, _s, _cid in sample:
        pairs = phrase_pairs(composition)
        if pairs:
            per_piece.append([p for p, _ in pairs])
            lengths.append(pairs[0][1])
    flat = [p for piece in per_piece for p in piece]
    if flat:
        ordered = sorted(flat)

        def q(p: float) -> float:
            return ordered[int(p * (len(ordered) - 1))]

        print(
            f"  population: ALL phrase pairs from {len(per_piece)} train pieces ({len(flat)} pairs)"
        )
        print(
            f"    median {statistics.median(ordered):.3f}  p90 {q(0.90):.3f}  "
            f"p99 {q(0.99):.3f}  max {ordered[-1]:.3f}"
        )
        # Piece-weighted, since pairs inside one composition are correlated.
        piece_medians = [statistics.median(p) for p in per_piece]
        print(
            f"    piece-weighted median of pair similarity: {statistics.median(piece_medians):.3f}"
        )
        above_related = sum(1 for p in flat if p >= PHRASE_RELATED) / len(flat)
        above_same = sum(1 for p in flat if p >= PHRASE_SAME) / len(flat)
        print(f"    pairs >= PHRASE_RELATED ({PHRASE_RELATED}): {100 * above_related:.1f}%")
        print(f"    pairs >= PHRASE_SAME    ({PHRASE_SAME}): {100 * above_same:.1f}%")
        report["phrase_thresholds"] = {
            "population": (
                "all phrase pairs within train pieces, pooled unweighted; "
                "pairs inside one composition are correlated, so pooled "
                "percentiles are dominated by a few long repetitive pieces"
            ),
            "n_pieces": len(per_piece),
            "n_pairs": len(flat),
            "median": round(statistics.median(ordered), 4),
            "p90": round(q(0.90), 4),
            "p99": round(q(0.99), 4),
            "piece_weighted_median": round(statistics.median(piece_medians), 4),
            "share_above_related": round(above_related, 4),
            "share_above_same": round(above_same, 4),
        }
        # Stability across piece length.
        print(f"\n    {'phrases/piece':16}{'n pairs':>9}{'median':>9}{'p90':>8}{'>=0.45':>9}")
        bands = (("2-3", 2, 3), ("4-6", 4, 6), ("7-12", 7, 12), ("13+", 13, 10**6))
        report["phrase_thresholds"]["by_length"] = {}
        for lab, low, high in bands:
            chosen = [
                p
                for piece, n in zip(per_piece, lengths, strict=False)
                if low <= n <= high
                for p in piece
            ]
            if not chosen:
                continue
            co = sorted(chosen)
            entry = {
                "n_pairs": len(co),
                "median": round(statistics.median(co), 4),
                "p90": round(co[int(0.9 * (len(co) - 1))], 4),
                "share_above_related": round(
                    sum(1 for p in co if p >= PHRASE_RELATED) / len(co), 4
                ),
            }
            report["phrase_thresholds"]["by_length"][lab] = entry
            print(
                f"    {lab:16}{entry['n_pairs']:>9}{entry['median']:>9.3f}"
                f"{entry['p90']:>8.3f}{100 * entry['share_above_related']:>8.1f}%"
            )

    # ---- write label records, BOTH views ----------------------------------
    # Original and round-trip labels, eligibility and features in one record,
    # so every combination count can be recomputed without re-running the
    # extraction. The first pass dumped original labels only, which made
    # round-tripped combination support impossible to derive afterwards.
    with args.labels_out.open("w") as handle:
        for row in rows:
            record = ConditioningRecord(
                piece_id=row["piece"],
                split=row["split"],
                window_index=0,
                features=row["orig_features"],
                eligibility=row["orig_elig"],
                conditioning=row["orig"],
                game=row["game"],
            )
            payload = record.to_dict()
            payload["roundtrip"] = {
                "conditioning": row["rt"],
                "eligibility": row["rt_elig"],
                "features": row["rt_features"].to_dict(),
            }
            payload["agreement"] = row["agree"]
            handle.write(json.dumps(payload) + "\n")

    args.out.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n  wrote {args.out} and {args.labels_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
