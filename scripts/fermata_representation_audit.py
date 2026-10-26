"""Can authentic NES music survive Fermata's vocabulary? Measured, two ways.

The feasibility question for NES adaptation: if encoding NES-MDB into
Fermata's token grammar loses essential musical structure, fine-tuning on it
teaches the model a damaged target. So fidelity is measured before any
training, which is the mistake the original native Transformer made.

Two comparisons, because one alone is misleading:

**A. original -> Fermata -> decoded.** Total fidelity against the real NES
events. This is what a fine-tune would actually be trained on.

**B. D4-quantised -> Fermata -> D4.** What Fermata's vocabulary costs
*relative to the representation already frozen*. Without B, D4's own
quantisation error gets attributed to Fermata and the pretrained model looks
worse than it is.

**Event loss and timing displacement are reported separately, at stated
tolerances.** A note moved by 10 ms is not a missing note, and conflating the
two is how an earlier audit in this project produced a false recall of 0.05 --
it compared raw ticks across two differently-scaled streams. Matching here
runs in seconds, per stream, with the tolerance printed beside every number.

One arithmetic result worth stating up front, because it was got wrong once:
D4 has 16 positions per quarter and Fermata 24 ticks, so a D4 slot is 1.5
ticks and half of them do not land on an integer. That does **not** lose
resolution. The 16 positions map to 16 *distinct* ticks
(0,2,3,5,6,8,9,11,12,14,15,17,18,20,21,23) and every slot recovers exactly;
the cost is displacement of at most half a tick, 10.4 ms at 120 BPM.
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
from collections import Counter
from typing import Any

from rse.events.convert import composition_from_intake
from rse.events.schema import Composition, Note
from rse.tokenizers.frozen import frozen_config
from rse.tokenizers.tokenizer import GridFit, RseTokenizer
from rse.transfer.converters import (
    NES_PROGRAMS,
    composition_to_score,
    score_to_composition,
)
from rse.transfer.fermata import SUBPOS, Vocab, decode, encode
from rse.transfer.score import BASS, MELODY, PERCUSSION

#: One tracker row, the convention the corpus was built on.
ROW_SEC = 0.1

#: Tolerances at which event recall is reported. Named, never implicit.
TOLERANCES_MS = (0.0, 5.0, 10.0, 25.0, 50.0)

CHANNELS = ("P1", "P2", "TR", "NO")

#: GM program back to NES channel. **Channel identity survives the Fermata
#: round trip only through this mapping.** Fermata's `<TRK_n>` carries a
#: program, not a role, so `decode` gives every pitched track the same role
#: and `score_to_composition` -- which assigns channels by role -- put all
#: three pitched voices on P1. Measured: 302 reconstructed events on P1
#: against 151 original, with P2 and TR empty. That is a property of the
#: representation worth reporting, not a detail to paper over.
PROGRAM_TO_CHANNEL = {program: channel for channel, program in NES_PROGRAMS.items()}

#: `score_to_composition` reads roles, so the roles are restored from the
#: programs before calling it.
PROGRAM_TO_ROLE = {80: MELODY, 81: "accompaniment", 38: BASS}


def restore_roles(score: Any) -> dict[str, int]:
    """Put each decoded track back on the role its program implies.

    Returns a count of tracks whose role had to be corrected, so the report
    can state how much channel identity depended on this step.
    """
    corrected = 0
    for track in score.tracks:
        if track.percussion and not track.notes:
            if track.role != PERCUSSION:
                track.role = PERCUSSION
                corrected += 1
            track.channel_id = "NO"
            continue
        role = PROGRAM_TO_ROLE.get(track.program or -1)
        if role is None:
            continue
        if track.role != role:
            corrected += 1
        track.role = role
        track.channel_id = PROGRAM_TO_CHANNEL.get(track.program or -1)
    return {"roles_corrected": corrected}


def grid_injectivity() -> dict[str, Any]:
    """Does every D4 position get its own Fermata tick?"""
    positions = 16
    mapping = {k: int(k * SUBPOS / positions + 0.5) for k in range(positions)}
    recovered = {
        k: int(round(tick * positions / SUBPOS)) for k, tick in mapping.items()
    }
    return {
        "d4_positions_per_quarter": positions,
        "fermata_ticks_per_quarter": SUBPOS,
        "ticks_used": sorted(set(mapping.values())),
        "injective": len(set(mapping.values())) == positions,
        "all_positions_recover": all(recovered[k] == k for k in mapping),
        "max_displacement_ticks": 0.5,
        "max_displacement_ms_at_bpm": {
            bpm: round((60.0 / bpm) / SUBPOS * 500, 2)
            for bpm in (76, 92, 120, 128, 150, 168, 180)
        },
    }


def events_in_seconds(
    composition: Composition,
) -> list[tuple[str, float, float, int, int]]:
    """(channel, onset_sec, duration_sec, voice, velocity) per sounding event.

    Seconds, because the two sides of every comparison carry different tick
    scales and comparing raw ticks across them is the error that produced a
    false 0.05 recall in the codec work.
    """
    end = composition.end_tick or 1
    total = composition.time.seconds(end)
    spt = (total / end) if total else 0.0
    out = []
    for event in composition.sounding_events():
        voice = (
            event.pitch if isinstance(event, Note) else getattr(event, "value", 0)
        )
        out.append(
            (
                event.channel_id,
                event.tick * spt,
                getattr(event, "duration", 0) * spt,
                int(voice),
                int(getattr(event, "velocity", 0)),
            )
        )
    return sorted(out, key=lambda e: (e[0], e[1]))


def match(
    original: list[tuple[str, float, float, int, int]],
    reconstructed: list[tuple[str, float, float, int, int]],
    *,
    tolerance_sec: float,
) -> dict[str, Any]:
    """One-to-one matching within a channel, on voice identity and onset.

    Greedy nearest-onset inside the tolerance, each reconstructed event used
    once, so a cluster of reconstructed notes cannot satisfy several originals
    and inflate recall.
    """
    matched: list[tuple[int, int]] = []
    used: set[int] = set()
    by_channel: dict[str, list[int]] = {}
    for index, event in enumerate(reconstructed):
        by_channel.setdefault(event[0], []).append(index)

    for oi, origin in enumerate(original):
        best, best_delta = None, None
        for ri in by_channel.get(origin[0], []):
            if ri in used:
                continue
            candidate = reconstructed[ri]
            if candidate[3] != origin[3]:
                continue
            delta = abs(candidate[1] - origin[1])
            if delta <= tolerance_sec and (best_delta is None or delta < best_delta):
                best, best_delta = ri, delta
        if best is not None:
            used.add(best)
            matched.append((oi, best))

    onset_errors = [
        abs(reconstructed[ri][1] - original[oi][1]) for oi, ri in matched
    ]
    duration_errors = [
        abs(reconstructed[ri][2] - original[oi][2]) for oi, ri in matched
    ]
    velocity_errors = [
        abs(reconstructed[ri][4] - original[oi][4]) for oi, ri in matched
    ]
    return {
        "tolerance_ms": round(tolerance_sec * 1000, 3),
        "n_original": len(original),
        "n_reconstructed": len(reconstructed),
        "n_matched": len(matched),
        "recall": round(len(matched) / len(original), 5) if original else None,
        "precision": round(len(matched) / len(reconstructed), 5)
        if reconstructed
        else None,
        "n_missing": len(original) - len(matched),
        "n_spurious": len(reconstructed) - len(matched),
        "onset_ms": _stats([e * 1000 for e in onset_errors]),
        "duration_ms": _stats([e * 1000 for e in duration_errors]),
        "velocity_abs": _stats(velocity_errors),
        "matched_pairs": matched,
    }


def _stats(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    ordered = sorted(values)
    return {
        "median": round(st.median(ordered), 4),
        "p95": round(ordered[int(0.95 * (len(ordered) - 1))], 4),
        "max": round(ordered[-1], 4),
        "mean": round(st.fmean(ordered), 4),
    }


def per_channel(
    original: list[tuple[str, float, float, int, int]],
    reconstructed: list[tuple[str, float, float, int, int]],
    *,
    tolerance_sec: float,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for channel in CHANNELS:
        o = [e for e in original if e[0] == channel]
        r = [e for e in reconstructed if e[0] == channel]
        if not o:
            continue
        result = match(o, r, tolerance_sec=tolerance_sec)
        result.pop("matched_pairs", None)
        out[channel] = result
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("/tmp/fermata_dl"))
    parser.add_argument(
        "--index",
        type=Path,
        default=Path("datasets/processed/nesmdb/midi/index.jsonl"),
        help="MIDI release: exprsco's 24 Hz frame rate cannot express the row grid",
    )
    parser.add_argument("--n", type=int, default=120)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument(
        "--splits",
        default="train,valid",
        help="test is reserved; composer-disjoint by the corpus's own split",
    )
    parser.add_argument("--row-sec", type=float, default=ROW_SEC)
    parser.add_argument(
        "--primary-tolerance-ms",
        type=float,
        default=25.0,
        help="tolerance for the headline recall figure, reported beside it",
    )
    parser.add_argument("--out", type=Path, default=Path("fermata_representation_audit.json"))
    args = parser.parse_args()

    vocab = Vocab.load(args.model / "tokenizer" / "vocab.json")
    tokenizer = RseTokenizer(frozen_config())
    root = args.index.parent
    wanted = {s for s in args.splits.split(",") if s}

    rows = [json.loads(line) for line in args.index.read_text().splitlines()]
    pool = [
        r
        for r in rows
        if r.get("split") in wanted
        and r.get("duplicate_of") in (None, "None", "")
        and int(r.get("n_notes", 0)) >= 32
    ]
    print(f"{len(rows)} indexed, {len(pool)} eligible in {sorted(wanted)}")
    sample = random.Random(args.seed).sample(pool, min(args.n, len(pool)))
    sample.sort(key=lambda r: r["composition_id"])

    grid = grid_injectivity()
    print(
        f"grid: {grid['d4_positions_per_quarter']} D4 positions -> "
        f"{len(grid['ticks_used'])} distinct Fermata ticks, "
        f"injective={grid['injective']}, all recover={grid['all_positions_recover']}"
    )

    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    primary = args.primary_tolerance_ms / 1000.0

    for index, row in enumerate(sample):
        path = root / row["record_path"]
        if not path.exists():
            path = path.with_suffix(".json")
        if not path.exists():
            failures.append({"id": row["composition_id"], "reason": "record_missing"})
            continue
        try:
            record = json.loads(path.read_text())
            # The corpus stores `rse-intake/0.1.0` records, not Composition
            # dicts, so Composition.from_dict rejects them on schema.
            original = composition_from_intake(record)
        except Exception as error:  # noqa: BLE001 - recorded, not fatal
            failures.append(
                {"id": row["composition_id"], "reason": f"load:{type(error).__name__}"}
            )
            continue
        if not original.sounding_events():
            failures.append({"id": row["composition_id"], "reason": "no_events"})
            continue

        bpm = 60.0 / (args.row_sec * 4)
        try:
            # -- arm A: original -> Fermata -> decoded ---------------------
            score_a, loss_a = composition_to_score(
                original,
                ticks_per_quarter=SUBPOS,
                tempo_bpm=bpm,
                row_sec=args.row_sec,
            )
            ids_a, encode_a = encode(score_a, vocab=vocab)
            back_a, decode_a = decode(ids_a, vocab=vocab)
            restored_a = restore_roles(back_a)
            comp_a = score_to_composition(back_a, row_sec=args.row_sec)
            if isinstance(comp_a, tuple):
                comp_a = comp_a[0]

            # -- arm B: D4-quantised -> Fermata -> D4 ----------------------
            # The fit MUST come from this composition's own tick scale. The
            # corpus carries ticks_per_beat=22050, not 480, and hard-coding
            # 480 made the D4 grid read row positions off the wrong scale: a
            # 1.719 s piece came back reporting 63.599 s, a 37x time error,
            # and the loss it produced would have been blamed on Fermata.
            fit = GridFit(
                step_ticks=float(tokenizer.ticks_per_step(original.ticks_per_beat)),
                phase_ticks=0.0,
                aligned=True,
                period_sec=args.row_sec,
            )
            d4 = tokenizer.decode(tokenizer.encode(original, fit=fit).ids).composition
            score_b, loss_b = composition_to_score(
                d4, ticks_per_quarter=SUBPOS, tempo_bpm=bpm, row_sec=args.row_sec
            )
            ids_b, encode_b = encode(score_b, vocab=vocab)
            back_b, _decode_b = decode(ids_b, vocab=vocab)
            restored_b = restore_roles(back_b)
            comp_b = score_to_composition(back_b, row_sec=args.row_sec)
            if isinstance(comp_b, tuple):
                comp_b = comp_b[0]
        except Exception as error:  # noqa: BLE001
            failures.append(
                {"id": row["composition_id"], "reason": f"convert:{type(error).__name__}: {error}"}
            )
            continue

        ev_original = events_in_seconds(original)
        ev_a = events_in_seconds(comp_a)
        ev_d4 = events_in_seconds(d4)
        ev_b = events_in_seconds(comp_b)

        entry: dict[str, Any] = {
            "composition_id": row["composition_id"],
            "name": row["name"],
            "split": row["split"],
            "n_notes_indexed": int(row["n_notes"]),
            "n_bars_indexed": int(row["n_bars"]),
            "channels_used": int(row["channels_used"]),
            "tempo_bpm_assumed": bpm,
            "n_tokens_fermata": len(ids_a),
            "roles_restored_a": restored_a,
            "roles_restored_b": restored_b,
            "arm_a": {
                t: {
                    k: v
                    for k, v in match(
                        ev_original, ev_a, tolerance_sec=t / 1000.0
                    ).items()
                    if k != "matched_pairs"
                }
                for t in TOLERANCES_MS
            },
            "arm_a_per_channel": per_channel(
                ev_original, ev_a, tolerance_sec=primary
            ),
            "arm_b": {
                t: {
                    k: v
                    for k, v in match(ev_d4, ev_b, tolerance_sec=t / 1000.0).items()
                    if k != "matched_pairs"
                }
                for t in TOLERANCES_MS
            },
            "arm_b_per_channel": per_channel(ev_d4, ev_b, tolerance_sec=primary),
            # NOT TRUSTWORTHY. This cross-comparison spans two tick scales
            # and two tempo labels, and its identity matching fails for a
            # reason not yet established -- recall 0.057 while onset errors
            # look sane at 8-15 ms. Arm B already isolates what Fermata's
            # vocabulary costs relative to D4, which is the attribution
            # question; this field is kept for later diagnosis and must not
            # be quoted as a result.
            "d4_vs_original_UNRELIABLE": {
                k: v
                for k, v in match(
                    ev_original, ev_d4, tolerance_sec=primary
                ).items()
                if k != "matched_pairs"
            },
            "conversion_loss_a": loss_a.to_dict() if hasattr(loss_a, "to_dict") else {},
            "conversion_loss_b": loss_b.to_dict() if hasattr(loss_b, "to_dict") else {},
            "encode_report_a": encode_a.to_dict() if hasattr(encode_a, "to_dict") else {},
            "decode_report_a": decode_a.to_dict() if hasattr(decode_a, "to_dict") else {},
            "encode_report_b": encode_b.to_dict() if hasattr(encode_b, "to_dict") else {},
        }
        results.append(entry)
        if (index + 1) % 20 == 0:
            print(f"  {index + 1}/{len(sample)} measured", flush=True)

    if not results:
        print("error: nothing measured", file=sys.stderr)
        for f in failures[:10]:
            print(f"  {f}", file=sys.stderr)
        return 2

    def gather(arm: str, tol: float, field: str) -> list[float]:
        return [
            r[arm][tol][field]
            for r in results
            if r[arm].get(tol) and r[arm][tol].get(field) is not None
        ]

    summary: dict[str, Any] = {"grid": grid}
    for arm in ("arm_a", "arm_b"):
        summary[arm] = {}
        for tol in TOLERANCES_MS:
            recalls = gather(arm, tol, "recall")
            if not recalls:
                continue
            summary[arm][f"{tol:g}ms"] = {
                "recall_median": round(st.median(recalls), 5),
                "recall_mean": round(st.fmean(recalls), 5),
                "n_pieces": len(recalls),
                "total_missing": sum(r[arm][tol]["n_missing"] for r in results),
                "total_spurious": sum(r[arm][tol]["n_spurious"] for r in results),
                "total_original": sum(r[arm][tol]["n_original"] for r in results),
            }
        onsets = [
            r[arm][args.primary_tolerance_ms]["onset_ms"]["median"]
            for r in results
            if r[arm].get(args.primary_tolerance_ms)
            and r[arm][args.primary_tolerance_ms]["onset_ms"]
        ]
        if onsets:
            summary[arm]["onset_ms_median_of_medians"] = round(st.median(onsets), 4)
            summary[arm]["onset_ms_worst_piece_median"] = round(max(onsets), 4)

    lengths = [r["n_tokens_fermata"] for r in results]
    summary["sequence_length"] = {
        "median": st.median(lengths),
        "p95": sorted(lengths)[int(0.95 * (len(lengths) - 1))],
        "max": max(lengths),
        "n_over_2048": sum(1 for v in lengths if v > 2048),
        "n_over_4096": sum(1 for v in lengths if v > 4096),
    }
    summary["channel_presence"] = {
        channel: sum(
            1 for r in results if channel in r["arm_a_per_channel"]
        )
        for channel in CHANNELS
    }
    summary["n_pieces"] = len(results)
    summary["n_failed"] = len(failures)
    summary["failures_by_reason"] = dict(
        Counter(f["reason"].split(":")[0] for f in failures)
    )

    args.out.write_text(
        json.dumps(
            {
                "primary_tolerance_ms": args.primary_tolerance_ms,
                "tolerances_ms": list(TOLERANCES_MS),
                "row_sec": args.row_sec,
                "splits": sorted(wanted),
                "summary": summary,
                "pieces": results,
                "failures": failures,
            },
            indent=2,
        )
        + "\n"
    )

    print(f"\n{len(results)} pieces measured, {len(failures)} failed "
          f"{summary['failures_by_reason']}")
    for arm, label in (
        ("arm_a", "A  original -> Fermata -> decoded"),
        ("arm_b", "B  D4 -> Fermata -> D4"),
    ):
        print(f"\n{label}")
        print(f"  {'tolerance':>10}{'recall':>10}{'missing':>10}{'spurious':>10}")
        for tol in TOLERANCES_MS:
            k = f"{tol:g}ms"
            if k not in summary[arm]:
                continue
            s = summary[arm][k]
            print(
                f"  {k:>10}{s['recall_median']:>10.4f}"
                f"{s['total_missing']:>10}{s['total_spurious']:>10}"
            )
        if "onset_ms_median_of_medians" in summary[arm]:
            print(
                f"  onset error: median {summary[arm]['onset_ms_median_of_medians']:.2f} ms, "
                f"worst piece median {summary[arm]['onset_ms_worst_piece_median']:.2f} ms"
            )
    print(f"\nsequence length: median {summary['sequence_length']['median']}, "
          f"p95 {summary['sequence_length']['p95']}, max {summary['sequence_length']['max']}, "
          f"{summary['sequence_length']['n_over_2048']} over 2048 tokens")
    print(f"channel presence: {summary['channel_presence']}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
