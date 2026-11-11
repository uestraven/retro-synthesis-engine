#!/usr/bin/env python3
"""RSE v2 milestone 1: full-population codec audit at 250 Hz.

All 1,384 train music tracks, same codec and definitions as the 60-track
sweep. 250 Hz is **provisionally selected, not validated** -- this decides it.

**Affected track identities are preserved.** A corpus-wide rate near zero is
not acceptance if a handful of compositions are badly damaged, so every
failing track is named with its counts. Nothing is repaired and no difficult
track is excluded.

Acceptance criteria, all strict:

- zero lost or duplicated distinguishable attacks
- zero unexplained lost or altered legato changes
- zero indistinguishable starts
- zero decoder-induced voice steals
- zero unpreserved same-voice retrigger ordering
- peak occupied voices never above eight
- timing error consistent with 250 Hz quantisation, with no unexplained
  systematic displacement

The last one is tested rather than asserted: half-up rounding to a 4 ms grid
bounds absolute onset error at 2 ms, so any track exceeding that, or a signed
mean far from zero, is a defect and not quantisation.

**Two claims are kept separate and only the first is in scope here:**

- *symbolic musical reconstruction* -- attacks, pitch changes, durations and
  timing preserved within measured bounds;
- *hardware-performance reconstruction* -- additionally preserving what is
  needed to reproduce original S-DSP voice behaviour.

The codec reassigns 12.6-74.6% of notes to different physical voices, so it
may satisfy the first while failing the second. It must not be called lossless
or hardware-equivalent merely because the reconstruction fits eight voices.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rse.snes_lm.release import Release  # noqa: E402
from rse.snes_v2 import codec as cx  # noqa: E402

GRID_HZ = 250.0   # default; overridable with --grid
QUANT_BOUND_MS = 1000.0 / GRID_HZ / 2.0


def _set_grid(grid_hz: float) -> None:
    """The grid is a module constant so `audit_track` reads one source."""
    global GRID_HZ, QUANT_BOUND_MS
    GRID_HZ = grid_hz
    QUANT_BOUND_MS = 1000.0 / grid_hz / 2.0


def audit_track(name: str, events) -> dict:
    groups, enc = cx.encode(events, grid_hz=GRID_HZ)
    notes, dec = cx.decode(groups, grid_hz=GRID_HZ)

    starts = [e for e in events if e.get("origin") != "legato_step"]
    starts.sort(key=lambda e: (float(e["onset_sec"]), int(e["voice"])))
    legato = [e for e in events if e.get("origin") == "legato_step"]
    by_id = {n["note_id"]: n for n in notes}

    signed, abs_on, dur_known, reassigned, matched = [], [], [], 0, 0
    for note_id, src in enumerate(starts):
        got = by_id.get(note_id)
        if got is None:
            continue
        matched += 1
        delta = got["onset_sec"] - float(src["onset_sec"])
        signed.append(delta)
        abs_on.append(abs(delta))
        d = src.get("duration_sec")
        if d is not None and got.get("end_source") == "derived_duration":
            dur_known.append(abs(got["duration_sec"] - float(d)))
        if int(got["voice"]) != int(src["voice"]):
            reassigned += 1

    span = max((float(e["onset_sec"]) for e in events), default=0.0) or 1e-9
    retained = sum(len(n.get("pitch_changes") or []) for n in notes)

    def q(values, p):
        if not values:
            return None
        s = sorted(values)
        return s[min(len(s) - 1, int(p * len(s)))]

    return {
        "track": name,
        "n_source_events": len(events),
        "n_attacks": len(starts),
        "n_decoded_notes": len(notes),
        "n_matched": matched,
        "n_lost_attacks": len(starts) - matched,
        "n_duplicated_notes": max(0, len(notes) - len(starts)),
        "n_legato_source": len(legato),
        "n_legato_retained": retained,
        "n_legato_lost": len(legato) - retained - enc.n_orphan_legato,
        "n_orphan_legato": enc.n_orphan_legato,
        "n_same_step_pitch_changes": enc.n_same_step_pitch_changes,
        "n_indistinguishable_starts": enc.n_indistinguishable_starts,
        "n_same_step_rekey_collisions": enc.n_same_step_rekey_collisions,
        # RECORDED NORMALISATION: ends clamped to the next key-on on the same
        # provenance voice. Reported separately from timing error, because a
        # note clamped from 1.0 s to 0.5 s is a correct 500 ms change that is
        # not quantisation, and folding it in would void the error bounds.
        "n_ends_clamped_to_rekey": enc.n_ends_clamped_to_rekey,
        "n_instants_merged": enc.n_instants_merged,
        "decoder_voice_steals": dec.n_voice_steals,
        "n_unmatched_ends": dec.n_unmatched_ends,
        "n_orphan_pitch_changes": dec.n_orphan_pitch_changes,
        "peak_occupied_voices": dec.peak_occupancy,
        "pct_reassigned": 100.0 * reassigned / matched if matched else None,
        "onset_abs_mean_ms": 1000 * st.mean(abs_on) if abs_on else None,
        "onset_abs_p95_ms": 1000 * q(abs_on, 0.95) if abs_on else None,
        "onset_abs_p99_ms": 1000 * q(abs_on, 0.99) if abs_on else None,
        "onset_abs_max_ms": 1000 * max(abs_on) if abs_on else None,
        "onset_signed_mean_ms": 1000 * st.mean(signed) if signed else None,
        "dur_known_mean_ms": 1000 * st.mean(dur_known) if dur_known else None,
        "dur_known_p95_ms": 1000 * q(dur_known, 0.95) if dur_known else None,
        "dur_known_p99_ms": 1000 * q(dur_known, 0.99) if dur_known else None,
        "dur_known_max_ms": 1000 * max(dur_known) if dur_known else None,
        "n_open_lifetimes": enc.n_ends_open,
        "actions_per_source_event": enc.actions_per_source_event,
        "n_groups": enc.n_groups,
        "groups_per_sec": enc.n_groups / span,
        "n_actions": enc.n_actions,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", default="~/snesmdb-releases/snesmdb-50game-rc1")
    parser.add_argument("--split", default="train")
    parser.add_argument("--out", default=str(Path.home() /
                        "workplace/v2_artifacts/full_audit.json"))
    parser.add_argument("--grid", type=float, default=GRID_HZ)
    args = parser.parse_args()
    _set_grid(args.grid)

    release = Release(args.release)
    tracks = release.tracks(split=args.split)
    print(f"  auditing {len(tracks)} {args.split} music tracks at {GRID_HZ} Hz",
          flush=True)

    rows = []
    for index, track in enumerate(tracks):
        name = str(getattr(track, "track_id", None)
                   or getattr(track, "name", None) or f"track{index}")
        events = release.events(track)
        if not events:
            continue
        rows.append(audit_track(name, events))
        if (index + 1) % 200 == 0:
            print(f"    {index + 1}/{len(tracks)}", flush=True)

    def total(key):
        return sum(r[key] or 0 for r in rows)

    def offenders(key):
        return [{"track": r["track"], key: r[key]} for r in rows if (r[key] or 0) > 0]

    criteria = {
        "zero_lost_attacks": total("n_lost_attacks") == 0,
        "zero_duplicated_notes": total("n_duplicated_notes") == 0,
        "zero_legato_lost": total("n_legato_lost") == 0,
        "zero_orphan_legato": total("n_orphan_legato") == 0,
        "zero_same_step_pitch_changes": total("n_same_step_pitch_changes") == 0,
        "zero_indistinguishable_starts": total("n_indistinguishable_starts") == 0,
        "zero_rekey_collisions": total("n_same_step_rekey_collisions") == 0,
        "zero_decoder_steals": total("decoder_voice_steals") == 0,
        "zero_unmatched_ends": total("n_unmatched_ends") == 0,
        "zero_orphan_pitch_changes": total("n_orphan_pitch_changes") == 0,
        "occupancy_within_eight": max(r["peak_occupied_voices"] for r in rows) <= 8,
        "onset_error_within_quantisation": max(
            r["onset_abs_max_ms"] or 0 for r in rows) <= QUANT_BOUND_MS + 1e-6,
    }
    report = {
        "schema": "snes_v2.full_audit.v1",
        "split": args.split,
        "grid_hz": GRID_HZ,
        "quantisation_bound_ms": QUANT_BOUND_MS,
        "n_tracks": len(rows),
        "n_source_events": total("n_source_events"),
        "n_attacks": total("n_attacks"),
        "n_decoded_notes": total("n_decoded_notes"),
        "n_legato_source": total("n_legato_source"),
        "n_legato_retained": total("n_legato_retained"),
        "totals": {k: total(k) for k in (
            "n_lost_attacks", "n_duplicated_notes", "n_legato_lost",
            "n_orphan_legato", "n_same_step_pitch_changes",
            "n_indistinguishable_starts", "n_same_step_rekey_collisions",
            "decoder_voice_steals", "n_unmatched_ends",
            "n_orphan_pitch_changes", "n_instants_merged",
            "n_ends_clamped_to_rekey")},
        "affected_tracks": {k: offenders(k) for k in (
            "n_lost_attacks", "n_duplicated_notes", "n_legato_lost",
            "n_orphan_legato", "n_same_step_pitch_changes",
            "n_indistinguishable_starts", "n_same_step_rekey_collisions",
            "decoder_voice_steals", "n_unmatched_ends",
            "n_orphan_pitch_changes", "n_ends_clamped_to_rekey")},
        "corpus_onset_abs_mean_ms": st.mean(
            [r["onset_abs_mean_ms"] for r in rows if r["onset_abs_mean_ms"]]),
        "corpus_onset_abs_max_ms": max(r["onset_abs_max_ms"] or 0 for r in rows),
        "corpus_onset_signed_mean_ms": st.mean(
            [r["onset_signed_mean_ms"] for r in rows
             if r["onset_signed_mean_ms"] is not None]),
        "corpus_dur_known_mean_ms": st.mean(
            [r["dur_known_mean_ms"] for r in rows if r["dur_known_mean_ms"]]),
        "corpus_dur_known_max_ms": max(r["dur_known_max_ms"] or 0 for r in rows),
        "n_open_lifetimes": total("n_open_lifetimes"),
        "peak_occupied_voices_max": max(r["peak_occupied_voices"] for r in rows),
        "reassignment_pct_median": st.median(
            [r["pct_reassigned"] for r in rows if r["pct_reassigned"] is not None]),
        "reassignment_pct_max": max(
            r["pct_reassigned"] or 0 for r in rows),
        "actions_per_source_event": total("n_actions") / total("n_source_events"),
        "groups_per_sec_median": st.median(r["groups_per_sec"] for r in rows),
        "sequence_length_actions": {
            "median": st.median(r["n_actions"] for r in rows),
            "p95": sorted(r["n_actions"] for r in rows)[int(0.95 * len(rows))],
            "max": max(r["n_actions"] for r in rows),
        },
        "acceptance": criteria,
        "all_criteria_pass": all(criteria.values()),
        "scope": "Symbolic musical reconstruction only. The codec reassigns "
                 "physical voices and therefore does NOT establish "
                 "hardware-performance reconstruction. Not to be described as "
                 "lossless or hardware-equivalent.",
        "per_track": rows,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1))

    print(f"\n  tracks {report['n_tracks']}, events {report['n_source_events']:,}, "
          f"attacks {report['n_attacks']:,}, legato {report['n_legato_source']:,} "
          f"(retained {report['n_legato_retained']:,})")
    print("  acceptance:")
    for k, v in criteria.items():
        n = report["totals"].get(k.replace("zero_", "n_"), "")
        print(f"    {'PASS' if v else 'FAIL'}  {k}")
    print(f"\n  onset abs mean {report['corpus_onset_abs_mean_ms']:.3f} ms, "
          f"max {report['corpus_onset_abs_max_ms']:.3f} ms "
          f"(bound {QUANT_BOUND_MS:.1f} ms), "
          f"signed mean {report['corpus_onset_signed_mean_ms']:+.4f} ms")
    print(f"  duration abs mean {report['corpus_dur_known_mean_ms']:.3f} ms, "
          f"max {report['corpus_dur_known_max_ms']:.3f} ms; "
          f"open lifetimes {report['n_open_lifetimes']:,}")
    print(f"  reassignment median {report['reassignment_pct_median']:.1f}%, "
          f"max {report['reassignment_pct_max']:.1f}%")
    print(f"  actions/event {report['actions_per_source_event']:.3f}, "
          f"groups/s median {report['groups_per_sec_median']:.1f}, "
          f"actions/track median {report['sequence_length_actions']['median']:,}")
    print(f"  ALL CRITERIA: {'PASS' if report['all_criteria_pass'] else 'FAIL'}")
    for k, v in report["affected_tracks"].items():
        if v:
            print(f"    {k}: {len(v)} track(s), e.g. {v[:3]}")
    print(f"\n  written to {args.out}")


if __name__ == "__main__":
    main()
