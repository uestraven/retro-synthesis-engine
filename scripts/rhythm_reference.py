"""Corpus rhythm statistics, as the reference every generation claim is read against.

Computed on **D4-reconstructed** eight-bar windows, not raw MIDI, because that
is what the model can actually represent and therefore the only fair target.
The distinction is large: IOI entropy on original MIDI is 2.236 over the same
windows but 1.479 after a D4 round trip, so quoting the raw figure overstates
the gap by 50%.

Piece-weighted throughout. An unusually long track contributes many windows,
and pooling them would let a handful of pieces set the reference -- the same
error that moved the motif thresholds when they were first fitted.

The metric set is deliberately wider than entropy alone. Pooled IOI entropy
cannot distinguish "uses many intervals across pieces" from "uses many within
one", and the saturated generation failure is invisible to it: a model emitting
one note per row on every channel has a single IOI and maximal Position-token
entropy at the same time. `active_rows_per_bar`, `channels_per_active_row` and
`longest_active_row_streak` see that directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Importable when run as a file from a checkout where the package is not
# installed, which is the normal state of a fresh GPU host.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import json
import statistics as st
from collections import Counter
from typing import Any

from rse.codec.benchmark import enforce_monophony, ioi_entropy
from rse.events.pulse import detect_pulse
from rse.events.schema import Composition, Noise, Note
from rse.tokenizers.frozen import frozen_config
from rse.tokenizers.tokenizer import GridFit, RseTokenizer
from rse.training.corpus import corpus_from_release

CHANNELS = ("P1", "P2", "TR", "NO")


def _window(composition: Composition, start: int, end: int) -> Composition:
    events: list[Note | Noise] = [
        type(e)(**{**vars(e), "tick": e.tick - start})
        for e in composition.sounding_events()
        if start <= e.tick < end
    ]
    return Composition(
        hardware_profile=composition.hardware_profile,
        ticks_per_beat=composition.ticks_per_beat,
        events=events,
        tempo_map=list(composition.tempo_map),
        time_signatures=list(composition.time_signatures),
    )


def window_stats(
    composition: Composition,
    *,
    row_ticks: float,
    rows_per_bar: int,
    divisions: int,
    n_bars: int,
) -> dict[str, Any]:
    """Every rhythm statistic for one window, on its own grid."""
    events = composition.sounding_events()
    if not events:
        return {}
    end = composition.end_tick or 1
    seconds = composition.time.seconds(end)
    spt = (seconds / end) if seconds else 0.0

    slot_ticks = row_ticks / divisions
    rows_total = rows_per_bar * n_bars

    # Which row, and which sub-row slot, each onset lands on.
    rows: Counter[int] = Counter()
    sub_slots: Counter[int] = Counter()
    per_channel_onsets: dict[str, list[int]] = {c: [] for c in CHANNELS}
    per_channel_rows: dict[str, set[int]] = {c: set() for c in CHANNELS}
    durations: Counter[int] = Counter()
    for event in events:
        slot = int(round(event.tick / slot_ticks)) if slot_ticks else 0
        row = slot // divisions
        rows[row] += 1
        sub_slots[slot % divisions] += 1
        if event.channel_id in per_channel_onsets:
            per_channel_onsets[event.channel_id].append(event.tick)
            per_channel_rows[event.channel_id].add(row)
        durations[max(1, int(round(event.duration / row_ticks)))] += 1

    active = sorted(rows)
    streak = best = 0
    previous: int | None = None
    for row in active:
        streak = streak + 1 if previous is not None and row == previous + 1 else 1
        best = max(best, streak)
        previous = row

    onsets = sorted({e.tick for e in events})
    iois = [b - a for a, b in zip(onsets, onsets[1:], strict=False)]
    ioi_counts = Counter(iois)

    per_channel: dict[str, Any] = {}
    for channel, ticks in per_channel_onsets.items():
        unique = sorted(set(ticks))
        channel_iois = [b - a for a, b in zip(unique, unique[1:], strict=False)]
        per_channel[channel] = {
            "n_notes": len(ticks),
            "notes_per_bar": round(len(ticks) / n_bars, 3),
            "active_rows": len(per_channel_rows[channel]),
            "ioi_entropy": round(ioi_entropy([t * spt for t in unique]), 4),
            "n_distinct_iois": len(set(channel_iois)),
            "dominant_ioi_share": (
                round(Counter(channel_iois).most_common(1)[0][1] / len(channel_iois), 4)
                if channel_iois
                else 0.0
            ),
        }

    total_notes = len(events)
    return {
        "n_events": total_notes,
        "notes_per_bar": round(total_notes / n_bars, 3),
        "active_rows_per_bar": round(len(active) / n_bars, 3),
        "silent_row_fraction": round(1.0 - len(active) / rows_total, 4) if rows_total else 0.0,
        "channels_per_active_row": round(st.fmean([rows[r] for r in active]), 4) if active else 0.0,
        "longest_active_row_streak": best,
        "n_silent_channels": sum(1 for c in CHANNELS if not per_channel_onsets[c]),
        "ioi_entropy": round(ioi_entropy([e.tick * spt for e in events]), 4),
        "n_distinct_iois": len(set(iois)),
        "dominant_ioi_share": (
            round(ioi_counts.most_common(1)[0][1] / len(iois), 4) if iois else 0.0
        ),
        "duration_1_share": round(durations.get(1, 0) / total_notes, 4),
        "n_distinct_durations": len(durations),
        # Share of onsets off the tracker row. The saturated generation mode
        # uses row-aligned onsets only, so this is the statistic that shows
        # whether D4's sub-row resolution is used at all.
        "off_row_onset_share": round(
            1.0 - sub_slots.get(0, 0) / max(1, sum(sub_slots.values())), 4
        ),
        "per_channel": per_channel,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="nesmdb")
    parser.add_argument("--release", default="midi")
    parser.add_argument("--splits", nargs="+", default=["train", "valid"])
    parser.add_argument("--limit", type=int, default=400)
    parser.add_argument("--window-bars", type=int, default=8)
    parser.add_argument("--out", type=Path, default=Path("rhythm_reference.json"))
    args = parser.parse_args()

    if "test" in args.splits:
        print("error: the test split is reserved", file=sys.stderr)
        return 2

    tokenizer = RseTokenizer(frozen_config())
    rows_per_bar = tokenizer.config.steps_per_bar
    divisions = tokenizer.config.onset_divisions
    payload: dict[str, Any] = {
        "window_bars": args.window_bars,
        "rows_per_bar": rows_per_bar,
        "sub_row_divisions": divisions,
        "onset_slots_per_bar": tokenizer.config.position_count,
        "basis": "D4-reconstructed windows, piece-weighted",
        "splits": {},
    }

    for split in args.splits:
        print(f"corpus    : loading {split}...", flush=True)
        entries, _info = corpus_from_release(
            args.dataset, args.release, limit=args.limit, splits=(split,)
        )
        # Piece-weighted: each composition contributes the mean of its windows,
        # so a long track cannot set the reference on its own.
        per_piece: list[dict[str, Any]] = []
        for index, (composition, _s, _p) in enumerate(entries, 1):
            pulse = detect_pulse(composition)
            if not pulse.found or pulse.period_sec <= 0:
                continue
            row_ticks = pulse.period_ticks
            width = int(round(row_ticks * rows_per_bar * args.window_bars))
            if width <= 0 or composition.end_tick < width:
                continue
            fit = GridFit(
                step_ticks=row_ticks,
                phase_ticks=0.0,
                aligned=True,
                period_sec=pulse.period_sec,
            )
            collected: list[dict[str, Any]] = []
            for start in range(0, max(1, composition.end_tick - width), width):
                source = _window(composition, start, start + width)
                if len(source.sounding_events()) < 8:
                    continue
                decoded = tokenizer.decode(tokenizer.encode(source, fit=fit).ids)
                final, _t = enforce_monophony(decoded.composition)
                stats = window_stats(
                    final,
                    row_ticks=float(tokenizer.ticks_per_step(480)),
                    rows_per_bar=rows_per_bar,
                    divisions=divisions,
                    n_bars=args.window_bars,
                )
                if stats:
                    collected.append(stats)
            if collected:
                piece: dict[str, Any] = {"n_windows": len(collected)}
                for key, value in collected[0].items():
                    if key == "per_channel":
                        continue
                    if isinstance(value, (int, float)):
                        piece[key] = st.fmean([c[key] for c in collected])
                piece["per_channel"] = {
                    channel: {
                        metric: st.fmean([c["per_channel"][channel][metric] for c in collected])
                        for metric in collected[0]["per_channel"][channel]
                    }
                    for channel in CHANNELS
                }
                per_piece.append(piece)
            if index % 100 == 0:
                print(f"          : {index}/{len(entries)}", flush=True)

        def quantiles(key: str, pieces: list[dict[str, Any]] = per_piece) -> dict[str, float]:
            values = [p[key] for p in pieces if key in p]
            if len(values) < 4:
                return {}
            q = st.quantiles(values, n=4)
            return {
                "median": round(st.median(values), 4),
                "p25": round(q[0], 4),
                "p75": round(q[2], 4),
                "p10": round(st.quantiles(values, n=10)[0], 4),
                "p90": round(st.quantiles(values, n=10)[8], 4),
            }

        metrics = [
            k
            for k, v in (per_piece[0].items() if per_piece else [])
            if isinstance(v, (int, float)) and k != "n_windows"
        ]
        entry: dict[str, Any] = {
            "n_pieces": len(per_piece),
            "n_windows": sum(p["n_windows"] for p in per_piece),
            "overall": {m: quantiles(m) for m in metrics},
            "per_channel": {
                channel: {
                    metric: round(
                        st.median([p["per_channel"][channel][metric] for p in per_piece]),
                        4,
                    )
                    for metric in (per_piece[0]["per_channel"][channel] if per_piece else {})
                }
                for channel in CHANNELS
            }
            if per_piece
            else {},
        }
        payload["splits"][split] = entry

    args.out.write_text(json.dumps(payload, indent=2) + "\n")

    for split, entry in payload["splits"].items():
        print(f"\n{split}: {entry['n_pieces']} pieces, {entry['n_windows']} windows")
        print(f"{'metric':<30}{'p10':>9}{'p25':>9}{'median':>9}{'p75':>9}{'p90':>9}")
        print("-" * 75)
        for metric, q in entry["overall"].items():
            if q:
                print(
                    f"{metric:<30}{q['p10']:>9.3f}{q['p25']:>9.3f}"
                    f"{q['median']:>9.3f}{q['p75']:>9.3f}{q['p90']:>9.3f}"
                )
        print("\n  per channel (median):")
        head = f"    {'metric':<22}" + "".join(f"{c:>10}" for c in CHANNELS)
        print(head)
        for metric in ("notes_per_bar", "ioi_entropy", "n_distinct_iois", "dominant_ioi_share"):
            cells = "".join(f"{entry['per_channel'][c].get(metric, 0):>10.3f}" for c in CHANNELS)
            print(f"    {metric:<22}{cells}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
