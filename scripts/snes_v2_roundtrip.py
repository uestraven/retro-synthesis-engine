#!/usr/bin/env python3
"""RSE v2 milestone 1: round-trip loss across candidate timing resolutions.

The milestone question, in one line:

    can genuine, heterogeneous SNES music be encoded and reconstructed in a
    form where time progression, simultaneity, note lifetimes and the
    eight-voice constraint are explicit -- without assuming stable melodic
    roles?

**The resolution is chosen from this measurement, not assumed.** 100 Hz is a
candidate because the release publishes a 100 Hz uniform grid, and
`DATASET.md` labels that grid "a convention, stated as one". It is not
automatically acceptable, so a ladder is swept and the cost of each rung
reported.

Notes are matched note-for-note by `note_id`, not by position: the decoder
reassigns voices on purpose, so positional matching would silently compare
different notes.

**Nothing is repaired silently.** Merged instants, indistinguishable starts,
voice steals, unmatched ends and orphaned pitch changes are all counted and
reported, and a resolution is not called lossless on the strength of a
tolerance it chose itself.

Train split only.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rse.snes_v2 import codec as cx  # noqa: E402
from rse.snes_lm.release import Release  # noqa: E402

LADDER = (1000.0, 500.0, 250.0, 200.0, 100.0, 50.0, 25.0)


def roundtrip(events, grid_hz: float) -> dict:
    groups, enc = cx.encode(events, grid_hz=grid_hz)
    notes, dec = cx.decode(groups, grid_hz=grid_hz)

    # Source starts, in the same note_id order the encoder assigned.
    starts = [e for e in events if e.get("origin") != "legato_step"]
    starts.sort(key=lambda e: (float(e["onset_sec"]), int(e["voice"])))
    by_id = {n["note_id"]: n for n in notes}

    onset_err, dur_err = [], []
    lost = pitch_lost = status_lost = 0
    for note_id, source in enumerate(starts):
        got = by_id.get(note_id)
        if got is None:
            lost += 1
            continue
        onset_err.append(abs(got["onset_sec"] - float(source["onset_sec"])))
        d = source.get("duration_sec")
        if d is not None and got.get("end_source") == "derived_duration":
            dur_err.append(abs(got["duration_sec"] - float(d)))
        if (source.get("midi_note") is None) != (got.get("midi_note") is None):
            pitch_lost += 1
        elif (source.get("midi_note") is not None
              and abs(float(source["midi_note"]) - got["midi_note"]) > 1e-9):
            pitch_lost += 1
        if source.get("pitch_status") != got.get("pitch_status"):
            status_lost += 1

    def q(values, p):
        if not values:
            return None
        s = sorted(values)
        return s[min(len(s) - 1, int(p * len(s)))]

    return {
        "grid_hz": grid_hz,
        "encode": enc.to_dict(),
        "decode": dec.to_dict(),
        "n_source_starts": len(starts),
        "n_reconstructed": len(notes),
        "n_lost_starts": lost,
        "onset_err_mean_ms": 1000 * st.mean(onset_err) if onset_err else None,
        "onset_err_p99_ms": 1000 * q(onset_err, 0.99) if onset_err else None,
        "onset_err_max_ms": 1000 * max(onset_err) if onset_err else None,
        "dur_err_mean_ms": 1000 * st.mean(dur_err) if dur_err else None,
        "dur_err_p99_ms": 1000 * q(dur_err, 0.99) if dur_err else None,
        "n_pitch_mismatch": pitch_lost,
        "n_pitch_status_mismatch": status_lost,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", default="~/snesmdb-releases/snesmdb-50game-rc1")
    parser.add_argument("--n-tracks", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=str(Path.home() /
                        "workplace/v2_artifacts/roundtrip.json"))
    args = parser.parse_args()

    release = Release(args.release)
    tracks = release.tracks(split="train")       # TRAIN ONLY
    random.Random(args.seed).shuffle(tracks)
    tracks = tracks[:args.n_tracks]
    tables = [release.events(t) for t in tracks]
    tables = [t for t in tables if t]
    print(f"  {len(tables)} train tracks, "
          f"{sum(len(t) for t in tables):,} source events", flush=True)

    per_grid = {}
    for grid_hz in LADDER:
        rows = [roundtrip(t, grid_hz) for t in tables]
        agg = {
            "grid_hz": grid_hz,
            "n_tracks": len(rows),
            "n_source_events": sum(r["encode"]["n_source_events"] for r in rows),
            "n_groups": sum(r["encode"]["n_groups"] for r in rows),
            "n_actions": sum(r["encode"]["n_actions"] for r in rows),
            "actions_per_source_event": (
                sum(r["encode"]["n_actions"] for r in rows)
                / sum(r["encode"]["n_source_events"] for r in rows)),
            "n_instants_merged": sum(r["encode"]["n_instants_merged"] for r in rows),
            "n_indistinguishable_starts": sum(
                r["encode"]["n_indistinguishable_starts"] for r in rows),
            "n_lost_starts": sum(r["n_lost_starts"] for r in rows),
            "n_pitch_mismatch": sum(r["n_pitch_mismatch"] for r in rows),
            "n_pitch_status_mismatch": sum(r["n_pitch_status_mismatch"] for r in rows),
            "n_voice_steals": sum(r["decode"]["n_voice_steals"] for r in rows),
            "n_unmatched_ends": sum(r["decode"]["n_unmatched_ends"] for r in rows),
            "n_orphan_pitch_changes": sum(
                r["decode"]["n_orphan_pitch_changes"] for r in rows),
            "peak_occupancy_max": max(r["decode"]["peak_occupancy"] for r in rows),
            "onset_err_mean_ms": st.mean(
                [r["onset_err_mean_ms"] for r in rows if r["onset_err_mean_ms"]]),
            "onset_err_p99_ms": max(
                [r["onset_err_p99_ms"] for r in rows if r["onset_err_p99_ms"]]),
            "dur_err_mean_ms": st.mean(
                [r["dur_err_mean_ms"] for r in rows if r["dur_err_mean_ms"]]),
            "dur_err_p99_ms": max(
                [r["dur_err_p99_ms"] for r in rows if r["dur_err_p99_ms"]]),
        }
        per_grid[str(grid_hz)] = agg
        print(f"  {grid_hz:7.1f} Hz  act/ev {agg['actions_per_source_event']:.3f}  "
              f"merged {agg['n_instants_merged']:7,}  indist {agg['n_indistinguishable_starts']:6,}  "
              f"lost {agg['n_lost_starts']:5,}  steals {agg['n_voice_steals']:6,}  "
              f"onset mean {agg['onset_err_mean_ms']:7.3f} ms p99 {agg['onset_err_p99_ms']:8.2f}  "
              f"dur mean {agg['dur_err_mean_ms']:7.3f} ms  peak {agg['peak_occupancy_max']}",
              flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "schema": "snes_v2.roundtrip.v1",
        "split": "train",
        "ladder_hz": list(LADDER),
        "n_tracks": len(tables),
        "note": "Resolution is selected from this sweep. Merged instants, "
                "indistinguishable starts, voice steals and mismatches are "
                "reported rather than absorbed into a tolerance.",
        "per_grid": per_grid,
    }, indent=1))
    print(f"\n  written to {args.out}")


if __name__ == "__main__":
    main()
