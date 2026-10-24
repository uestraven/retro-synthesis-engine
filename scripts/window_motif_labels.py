"""Gate 3A: motif labels for each model-visible 8-bar training example.

The labelling unit is the training example itself: a window's token ids decoded
back into events. Not the source piece, and not the window before
tokenization. A piece's label cannot be inherited by its windows -- a track can
open with a sparse statement and develop a recurring motif later, and 63% of
pieces contribute exactly one window while one contributes 96, so inheritance
would both mislabel windows and weight a few long pieces heavily.

A threshold problem this surfaces
---------------------------------
Gate 2 fitted thresholds on whole pieces: motif_low 0.6667, motif_high 0.898.
Whole pieces recur more than eight-bar excerpts of them simply by being longer,
so applying piece-fitted boundaries to windows biases labels toward `low`. The
report therefore shows the distribution under both the frozen piece-fitted
thresholds and boundaries re-fitted on train *windows* — still train-only, so
neither leaks — and the choice is left to the review rather than made here.

Sampling weights are emitted because windows per composition are unequal. Left
alone, training would see a 96-window piece 96 times as often as a one-window
piece, which is a form of the game-dominance problem the splits were designed
to avoid.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from rse.conditioning import (
    FEATURE_VERSION,
    MIN_MELODY_NOTES,
    UNKNOWN,
    Thresholds,
    melody_line,
    motif_note_participation,
)
from rse.tokenizers.tokenizer import RseTokenizer


def motif_label(participation: float, n_notes: int, thresholds: Thresholds) -> str:
    """HIGH/MEDIUM/LOW, or UNKNOWN when there is too little melody to judge.

    Insufficient evidence is never recorded as `low`: 22% of the corpus
    genuinely never restates a motif and those pieces are *more* stepwise than
    the rest, so `low` is a real musical behaviour and must not double as a
    missing-data marker.
    """
    if n_notes < MIN_MELODY_NOTES:
        return UNKNOWN
    if participation >= thresholds.motif_high:
        return "high"
    if participation <= thresholds.motif_low:
        return "low"
    return "medium"


def game_of(name: str) -> str:
    parts = name.split("/")[-1].split("_")
    return parts[1] if len(parts) > 1 else "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="nesmdb")
    parser.add_argument("--release", default="midi")
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    parser.add_argument("--window-bars", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--gate2", type=Path, default=Path("conditioning_gate2.json"))
    parser.add_argument("--datasets-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("window_motif_labels.jsonl"))
    parser.add_argument("--report", type=Path, default=Path("window_motif_report.json"))
    args = parser.parse_args()

    from rse.training.corpus import corpus_from_release
    from rse.training.dataset import windows_from_composition

    tokenizer = RseTokenizer()
    frozen = Thresholds(
        **{
            k: v
            for k, v in json.loads(args.gate2.read_text())["thresholds"].items()
            if k in Thresholds.__dataclass_fields__
        }
    )
    print(f"frozen piece-fitted thresholds: low {frozen.motif_low}, high {frozen.motif_high}")

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
        for index, (composition, piece_split, cid) in enumerate(entries):
            if index and index % 300 == 0:
                print(f"  {split} {index}/{len(entries)}", flush=True)
            name = str(
                (composition.source_metadata.get("source") or {}).get("relpath")
                or composition.source_metadata.get("name")
                or cid
            )
            windows = windows_from_composition(
                composition,
                tokenizer,
                split=piece_split,
                window_bars=args.window_bars,
                composition_id=str(cid),
            )
            for window in windows:
                # THE labelling unit: the training example as the model sees
                # it, recovered by decoding the very ids that would be fed in.
                decoded = tokenizer.decode(window.ids).composition
                channel, pitches = melody_line(decoded)
                participation = motif_note_participation(pitches) if pitches else 0.0
                rows.append(
                    {
                        "piece_id": str(cid),
                        "source_name": name,
                        "game": game_of(name),
                        "split": window.split,
                        "window_index": window.start_bar // args.window_bars,
                        "start_bar": window.start_bar,
                        "n_bars": window.n_bars,
                        "n_tokens": len(window.ids),
                        "lead_channel": channel,
                        "n_melody_notes": len(pitches),
                        "motif_note_participation": round(participation, 5),
                        "feature_version": FEATURE_VERSION,
                        "label_source": "roundtrip_window_tokens",
                    }
                )

    # --- labels under both threshold sets ----------------------------------
    train_rows = [r for r in rows if r["split"] == "train"]
    eligible_train = [
        r["motif_note_participation"] for r in train_rows if r["n_melody_notes"] >= MIN_MELODY_NOTES
    ]
    window_fitted = Thresholds(fitted_on="train-windows", n_fitted=len(train_rows))
    nonzero = sorted(v for v in eligible_train if v > 0.01)
    if len(nonzero) >= 10:
        window_fitted.motif_low = round(nonzero[int(0.33 * (len(nonzero) - 1))], 4)
        window_fitted.motif_high = round(nonzero[int(0.67 * (len(nonzero) - 1))], 4)
    print(
        f"train-window-fitted thresholds: low {window_fitted.motif_low}, "
        f"high {window_fitted.motif_high}"
    )

    for row in rows:
        row["label_frozen_piece_thresholds"] = motif_label(
            row["motif_note_participation"], row["n_melody_notes"], frozen
        )
        row["label_window_thresholds"] = motif_label(
            row["motif_note_participation"], row["n_melody_notes"], window_fitted
        )

    report: dict[str, Any] = {
        "feature_version": FEATURE_VERSION,
        "labelling_unit": "decoded token ids of each 8-bar training window",
        "thresholds_frozen_piece_fitted": frozen.to_dict(),
        "thresholds_window_fitted_train_only": window_fitted.to_dict(),
        "threshold_note": (
            "Gate 2 boundaries were fitted on whole pieces, which recur more "
            "than eight-bar excerpts of them, so applying them to windows "
            "biases toward low. Both are reported; neither leaks, since both "
            "are fitted on train only."
        ),
        "n_windows": len(rows),
        "test_split_note": (
            "test labels are computed but reserved; use them only for a final frozen evaluation"
        ),
    }

    print(f"\n{'=' * 78}\nLABEL DISTRIBUTIONS per split")
    for key, caption in (
        ("label_frozen_piece_thresholds", "frozen piece-fitted thresholds"),
        ("label_window_thresholds", "train-window-fitted thresholds"),
    ):
        print(f"\n  {caption}")
        print(f"    {'split':7}{'windows':>9}{'high':>8}{'medium':>9}{'low':>8}{'unknown':>10}")
        report[key] = {}
        for split in args.splits:
            subset = [r for r in rows if r["split"] == split]
            if not subset:
                continue
            counts = Counter(r[key] for r in subset)
            report[key][split] = dict(counts)
            print(
                f"    {split:7}{len(subset):>9}{counts.get('high', 0):>8}"
                f"{counts.get('medium', 0):>9}{counts.get('low', 0):>8}"
                f"{counts.get(UNKNOWN, 0):>10}"
            )

    # --- windows and unique compositions, plus game concentration ---------
    print(f"\n{'=' * 78}\nSUPPORT: windows, unique compositions, game concentration")
    report["support"] = {}
    for key in ("label_frozen_piece_thresholds", "label_window_thresholds"):
        report["support"][key] = {}
        print(f"\n  {key}")
        print(
            f"    {'split':7}{'label':9}{'windows':>9}{'pieces':>8}{'games':>7}{'top game %':>12}"
        )
        for split in ("train", "valid"):
            for name in ("high", "medium", "low", UNKNOWN):
                subset = [r for r in rows if r["split"] == split and r[key] == name]
                if not subset:
                    continue
                games = Counter(r["game"] for r in subset)
                top = max(games.values()) / len(subset)
                entry = {
                    "windows": len(subset),
                    "pieces": len({r["piece_id"] for r in subset}),
                    "games": len(games),
                    "top_game_share": round(top, 4),
                }
                report["support"][key][f"{split}/{name}"] = entry
                print(
                    f"    {split:7}{name:9}{entry['windows']:>9}{entry['pieces']:>8}"
                    f"{entry['games']:>7}{100 * top:>11.1f}%"
                )

    # --- unequal windows per composition ----------------------------------
    per_piece = Counter(r["piece_id"] for r in train_rows)
    counts = sorted(per_piece.values())
    print(f"\n{'=' * 78}\nWINDOWS PER COMPOSITION (train)")
    print(
        f"  median {statistics.median(counts)}  mean {statistics.fmean(counts):.2f}  "
        f"max {max(counts)}"
    )
    print(f"  a piece with {max(counts)} windows would be sampled {max(counts)}x as often")
    print("  as a one-window piece; sampling_weight in the manifest is 1/n_windows")
    report["windows_per_composition"] = {
        "median": statistics.median(counts),
        "mean": round(statistics.fmean(counts), 3),
        "max": max(counts),
        "weight_rule": "sampling_weight = 1 / windows_in_that_composition",
    }
    for row in rows:
        row["sampling_weight"] = round(1.0 / per_piece.get(row["piece_id"], 1), 6)

    # --- context headroom for the conditioning prefix ---------------------
    lengths = [r["n_tokens"] for r in rows]
    longest = max(lengths)
    print(f"\n{'=' * 78}\nCONTEXT HEADROOM with a 1-token conditioning prefix")
    print(f"  longest window now {longest} tokens -> {longest + 1} with the prefix")
    for limit in (2048, 2304):
        over_now = sum(1 for n in lengths if n > limit)
        over_then = sum(1 for n in lengths if n + 1 > limit)
        print(
            f"  over seq_len {limit}: {over_now} now, {over_then} with prefix "
            f"({100 * over_then / len(lengths):.2f}%)"
        )
        report[f"over_{limit}"] = {"now": over_now, "with_prefix": over_then}
    report["context"] = {
        "longest_window_tokens": longest,
        "with_one_token_prefix": longest + 1,
    }

    with args.out.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    args.report.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n  wrote {args.out} and {args.report}")

    # --- split integrity ---------------------------------------------------
    by_piece: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        by_piece[row["piece_id"]].add(row["split"])
    straddle = {p for p, s in by_piece.items() if len(s) > 1}
    print(f"  compositions appearing in more than one split: {len(straddle)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
