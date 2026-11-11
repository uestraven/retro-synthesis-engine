#!/usr/bin/env python3
"""Legato value-and-order verification. The last check before the codec freeze.

The full audit established that 220,809 legato pitch changes are retained by
COUNT. A count is not preservation: it cannot detect a substituted event, a
duplicated one, a reconstructed pitch where the source was unrecoverable, or a
reversed pair. This verifies each decoded pitch change against its source
`snesmdb.events.v4` record by STABLE IDENTITY.

Per the acceptance requirements, each decoded change is checked for:

- **parent identity** — attached to the same original parent attack, compared
  by source `event_id`, not by position;
- **event identity** — no substitutions, no duplicates, no omissions;
- **pitch status** — exact categorical preservation;
- **pitch value** — exact preservation when recoverable;
- **unrecoverable pitch** — remains explicitly unrecoverable. For the ~18.5%
  of legato steps whose pitch is unrecoverable, the correct outcome is
  PRESERVED UNCERTAINTY, never a reconstructed value;
- **temporal ordering** — same order within each parent's sequence;
- **time** — within the declared 1000 Hz quantisation bound of 0.5 ms.

Where quantisation collapses two changes onto one grid step, order must still
survive. The codec carries an explicit per-parent `sequence` index for exactly
this, so equal timestamps cannot silently reverse.

Acceptance: zero unexplained value, identity or ordering mismatches.
Mismatches are reported by category and by affected track.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rse.snes_lm.release import Release  # noqa: E402
from rse.snes_v2 import codec as cx  # noqa: E402

GRID_HZ = 1000.0
QUANT_BOUND_SEC = 1.0 / GRID_HZ / 2.0
UNRECOVERABLE = "unrecoverable"


def _set_grid(grid_hz: float) -> None:
    """Module constants, so `check_track` reads one source of truth."""
    global GRID_HZ, QUANT_BOUND_SEC
    GRID_HZ = grid_hz
    QUANT_BOUND_SEC = 1.0 / grid_hz / 2.0


def check_track(name: str, events) -> dict:
    groups, _ = cx.encode(events, grid_hz=GRID_HZ)
    notes, _ = cx.decode(groups, grid_hz=GRID_HZ)

    source_legato = [e for e in events if e.get("origin") == "legato_step"]
    starts = [e for e in events if e.get("origin") != "legato_step"]
    starts.sort(key=lambda e: (float(e["onset_sec"]), int(e["voice"])))
    parent_of_start = {i: e.get("event_id") for i, e in enumerate(starts)}

    # Source truth: parent event_id -> ordered list of its legato steps.
    by_parent: dict[object, list[dict]] = {}
    for e in source_legato:
        by_parent.setdefault(e.get("parent_event_id"), []).append(e)
    for v in by_parent.values():
        v.sort(key=lambda e: float(e["onset_sec"]))

    counts = dict(
        n_source=len(source_legato), n_decoded=0, n_verified=0,
        parent_mismatch=0, identity_missing=0, identity_duplicate=0,
        status_mismatch=0, value_mismatch=0,
        unrecoverable_reconstructed=0, order_mismatch=0, time_out_of_bound=0,
    )
    seen: set[object] = set()

    for note in notes:
        changes = note.get("pitch_changes") or []
        counts["n_decoded"] += len(changes)
        if not changes:
            continue
        parent_id = parent_of_start.get(note["note_id"])
        expected = by_parent.get(parent_id, [])
        # Ordering: compare the decoded sequence against the source sequence.
        decoded = sorted(changes, key=lambda c: (c["onset_sec"], c["sequence"]))
        if [c["source_event_id"] for c in decoded] != [
                e.get("event_id") for e in expected]:
            counts["order_mismatch"] += 1
        for got in decoded:
            eid = got["source_event_id"]
            if eid in seen:
                counts["identity_duplicate"] += 1
            seen.add(eid)
            src = next((e for e in expected if e.get("event_id") == eid), None)
            if src is None:
                counts["parent_mismatch"] += 1
                continue
            ok = True
            if src.get("pitch_status") != got.get("pitch_status"):
                counts["status_mismatch"] += 1
                ok = False
            s_note, g_note = src.get("midi_note"), got.get("midi_note")
            if src.get("pitch_status") == UNRECOVERABLE or s_note is None:
                if g_note is not None:
                    counts["unrecoverable_reconstructed"] += 1
                    ok = False
            elif g_note is None or abs(float(s_note) - float(g_note)) > 1e-9:
                counts["value_mismatch"] += 1
                ok = False
            if abs(got["onset_sec"] - float(src["onset_sec"])) > QUANT_BOUND_SEC + 1e-9:
                counts["time_out_of_bound"] += 1
                ok = False
            if ok:
                counts["n_verified"] += 1

    for e in source_legato:
        if e.get("event_id") not in seen and e.get("parent_event_id") in parent_of_start.values():
            counts["identity_missing"] += 1

    counts["track"] = name
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", default="~/snesmdb-releases/snesmdb-50game-rc1")
    parser.add_argument("--split", default="train")
    parser.add_argument("--out", default=str(Path.home() /
                        "workplace/v2_artifacts/legato_check.json"))
    parser.add_argument("--grid", type=float, default=GRID_HZ)
    args = parser.parse_args()
    _set_grid(args.grid)

    release = Release(args.release)
    tracks = release.tracks(split=args.split)
    print(f"  verifying legato on {len(tracks)} {args.split} tracks "
          f"at {GRID_HZ} Hz (bound {1000*QUANT_BOUND_SEC:.2f} ms)", flush=True)

    rows = []
    for index, track in enumerate(tracks):
        name = str(getattr(track, "track_id", None)
                   or getattr(track, "name", None) or f"track{index}")
        events = release.events(track)
        if not events:
            continue
        rows.append(check_track(name, events))
        if (index + 1) % 400 == 0:
            print(f"    {index + 1}/{len(tracks)}", flush=True)

    categories = ("parent_mismatch", "identity_missing", "identity_duplicate",
                  "status_mismatch", "value_mismatch",
                  "unrecoverable_reconstructed", "order_mismatch",
                  "time_out_of_bound")
    totals = {k: sum(r[k] for r in rows) for k in categories}
    report = {
        "schema": "snes_v2.legato_check.v1",
        "split": args.split,
        "grid_hz": GRID_HZ,
        "quantisation_bound_ms": 1000 * QUANT_BOUND_SEC,
        "n_tracks": len(rows),
        "n_source_legato": sum(r["n_source"] for r in rows),
        "n_decoded_pitch_changes": sum(r["n_decoded"] for r in rows),
        "n_fully_verified": sum(r["n_verified"] for r in rows),
        "mismatches_by_category": totals,
        "affected_tracks": {
            k: [{"track": r["track"], k: r[k]} for r in rows if r[k] > 0]
            for k in categories},
        "all_pass": all(v == 0 for v in totals.values()),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1))

    print(f"\n  source legato {report['n_source_legato']:,}  "
          f"decoded {report['n_decoded_pitch_changes']:,}  "
          f"fully verified {report['n_fully_verified']:,}")
    for k, v in totals.items():
        n = len(report["affected_tracks"][k])
        print(f"    {'PASS' if v == 0 else 'FAIL'}  {k:30} {v:7,}"
              + (f"  on {n} track(s)" if v else ""))
    print(f"  ALL: {'PASS' if report['all_pass'] else 'FAIL'}")
    print(f"\n  written to {args.out}")


if __name__ == "__main__":
    main()
