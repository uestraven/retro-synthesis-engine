#!/usr/bin/env python3
"""Decision 3, properties 1 and 4: G3 semantic round-trip and real efficiency.

**The comparison is against the FROZEN CODEC, not against G3 itself.** G3's
serializer and parser share `StreamState`'s transition logic, so comparing a
G3 encoding with a G3 decoding would let one defective transition agree
perfectly with itself. Instead the note table reconstructed from parsed G3
tokens is compared against `codec.decode`'s output -- a different decoder,
independently validated across 1,740 tracks in Milestone 1.

Two paths, compared note for note:

    events.v4 -> codec.encode -> groups -> codec.decode        -> notes_A
    events.v4 -> codec.encode -> groups -> G3 serialize+parse  -> notes_B

Property 4 measures the pipeline as a model would actually consume it:
group-aligned windows with serialized snapshots, and -- following Level 1,
where window boundaries and the next-token shift silently changed the loss
denominator -- **scored targets reported separately from serialized tokens**.
Snapshot positions are conditioning state, not model output, so they are
excluded from scored targets and reported as overhead.

The 27,841 disclosed duration normalisations belong to the codec and are kept
separate from anything this layer costs.
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
from rse.snes_v2 import delta as dx  # noqa: E402
from rse.snes_v2 import grammar as gx  # noqa: E402

GRID_HZ = 4000.0
CONTEXT = 1024
PITCH_TOLERANCE = 0.05 + 1e-9          # 5 cents


def load_vocab() -> gx.Vocab:
    raw = json.loads((Path.home() /
                      "workplace/v2_artifacts/delta_vocab_k512.json").read_text())
    codec = dx.DeltaCodec(direct={int(k): v for k, v in raw["direct"].items()},
                          n_observed=raw["n_distinct_train"])
    return gx.Vocab(delta=codec)


def notes_from_parsed(parsed, vocab: gx.Vocab) -> list[dict]:
    """Thin wrapper; the implementation now lives in grammar.notes_from_groups
    so generation reuses it instead of growing a third event walk."""
    return gx.notes_from_groups(parsed)


def _superseded_notes_from_parsed(parsed, vocab: gx.Vocab) -> list[dict]:
    """The original, kept only so the audit's provenance is legible."""
    tick = 0
    live: dict[int, dict] = {}
    out: list[dict] = []
    for group in parsed:
        tick += group.delta
        for slot in group.ends:
            note = live.pop(slot, None)
            if note is not None:
                note["end_tick"] = tick
                out.append(note)
        free = sorted(set(range(cx.N_HARDWARE_VOICES)) - set(live))
        for midi, status in group.starts:
            slot = free.pop(0)
            live[slot] = {"slot": slot, "start_tick": tick, "midi": midi,
                          "status": status, "changes": []}
        for slot, midi, status in group.changes:
            if slot in live:
                live[slot]["changes"].append((tick, midi, status))
    for note in live.values():
        note["end_tick"] = tick
        out.append(note)
    out.sort(key=lambda n: (n["start_tick"], n["slot"]))
    return out


def audit_track(name: str, events, vocab: gx.Vocab) -> dict:
    groups, enc = cx.encode(events, grid_hz=GRID_HZ)
    canonical, dec = cx.decode(groups, grid_hz=GRID_HZ)

    tokens = gx.serialize(groups, vocab)
    parsed, report = gx.parse(tokens, vocab)
    rebuilt = notes_from_parsed(parsed, vocab)

    canonical.sort(key=lambda e: (round(float(e["onset_sec"]) * GRID_HZ),
                                  int(e["voice"])))
    row = {
        "track": name,
        "n_canonical_notes": len(canonical),
        "n_rebuilt_notes": len(rebuilt),
        "n_starts": report.n_starts,
        "n_changes": report.n_changes,
        "n_canonical_changes": sum(
            len(e.get("pitch_changes") or []) for e in canonical),
        "pitch_mismatch": 0, "status_mismatch": 0, "onset_mismatch": 0,
        "slot_mismatch": 0, "pitch_err_max": 0.0,
        "delta_sum_tokens": sum(g.delta for g in parsed),
        "delta_sum_groups": sum(g.delta for g in groups[1:]) + groups[0].delta,
        "forced_closures": report.n_forced_group_closures,
        "n_tokens_whole_track": len(tokens),
    }
    for a, b in zip(canonical, rebuilt):
        if round(float(a["onset_sec"]) * GRID_HZ) != b["start_tick"]:
            row["onset_mismatch"] += 1
        if int(a["voice"]) != b["slot"]:
            row["slot_mismatch"] += 1
        if (a.get("midi_note") is None) != (b["midi"] is None):
            row["pitch_mismatch"] += 1
        elif a.get("midi_note") is not None:
            err = abs(float(a["midi_note"]) - b["midi"])
            row["pitch_err_max"] = max(row["pitch_err_max"], err)
            if err > PITCH_TOLERANCE:
                row["pitch_mismatch"] += 1
        if (a.get("pitch_status") or "estimated") != b["status"]:
            row["status_mismatch"] += 1

    # --- property 4: real windows ---
    windows = gx.build_windows(groups, vocab, context=CONTEXT)
    span = max(float(e["onset_sec"]) for e in events) or 1e-9
    snap_tokens = sum(w.n_snapshot_tokens for w in windows)
    lengths = [w.n_tokens for w in windows]
    # Scored targets: next-token shift, minus snapshot positions, which are
    # given conditioning rather than model output.
    scored = sum(w.n_tokens - 1 - w.n_snapshot_tokens for w in windows)
    covered = sum(w.last_group - w.first_group + 1 for w in windows)
    row.update({
        "n_windows": len(windows),
        "window_tokens_total": sum(lengths),
        "window_tokens_median": st.median(lengths),
        "snapshot_tokens": snap_tokens,
        "padding_tokens": sum(CONTEXT - n for n in lengths),
        "scored_targets": scored,
        "n_groups": len(groups),
        "groups_covered": covered,
        "windows_with_active_notes": sum(1 for w in windows if w.snapshot),
        "handles_at_entry": [len(w.snapshot) for w in windows],
        "tokens_per_sec": len(tokens) / span,
        "span_sec": span,
    })
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", default="~/snesmdb-releases/snesmdb-50game-rc1")
    parser.add_argument("--out", default=str(Path.home() /
                        "workplace/v2_artifacts/g3_audit.json"))
    parser.add_argument("--splits", default="train,valid,test")
    args = parser.parse_args()

    vocab = load_vocab()
    release = Release(args.release)
    print(f"  G3 vocabulary {vocab.size:,} tokens; context {CONTEXT}", flush=True)

    per_split = {}
    for split in args.splits.split(","):
        rows = []
        tracks = release.tracks(split=split)
        for index, track in enumerate(tracks):
            events = release.events(track)
            if not events:
                continue
            name = str(getattr(track, "track_id", None)
                       or getattr(track, "name", None) or f"t{index}")
            rows.append(audit_track(name, events, vocab))
            if (index + 1) % 400 == 0:
                print(f"    {split} {index + 1}/{len(tracks)}", flush=True)

        def total(key):
            return sum(r[key] for r in rows)

        bad = {k: [r["track"] for r in rows if r[k] > 0]
               for k in ("pitch_mismatch", "status_mismatch", "onset_mismatch",
                         "slot_mismatch", "forced_closures")}
        delta_exact = all(r["delta_sum_tokens"] == r["delta_sum_groups"]
                          for r in rows)
        rates = [r["tokens_per_sec"] for r in rows]
        handles = [h for r in rows for h in r["handles_at_entry"]]
        per_split[split] = {
            "n_tracks": len(rows),
            "n_canonical_notes": total("n_canonical_notes"),
            "n_rebuilt_notes": total("n_rebuilt_notes"),
            "n_starts": total("n_starts"),
            "n_changes": total("n_changes"),
            "n_canonical_changes": total("n_canonical_changes"),
            "delta_sums_exact": delta_exact,
            "pitch_err_max": max(r["pitch_err_max"] for r in rows),
            "mismatches": {k: total(k) for k in
                           ("pitch_mismatch", "status_mismatch",
                            "onset_mismatch", "slot_mismatch")},
            "affected_tracks": {k: v[:10] for k, v in bad.items() if v},
            "forced_closures": total("forced_closures"),
            "n_windows": total("n_windows"),
            "window_tokens_total": total("window_tokens_total"),
            "snapshot_tokens": total("snapshot_tokens"),
            "padding_tokens": total("padding_tokens"),
            "scored_targets": total("scored_targets"),
            "groups_total": total("n_groups"),
            "groups_covered": total("groups_covered"),
            "windows_with_active_notes": total("windows_with_active_notes"),
            "median_handles_at_entry": st.median(handles) if handles else 0,
            "median_tokens_per_sec": st.median(rates),
            "seconds_in_1024": 1024 / st.median(rates),
        }
        s = per_split[split]
        print(f"  {split}: notes {s['n_canonical_notes']:,} -> "
              f"{s['n_rebuilt_notes']:,}  changes {s['n_canonical_changes']:,} -> "
              f"{s['n_changes']:,}  mismatches {s['mismatches']}  "
              f"delta exact {s['delta_sums_exact']}  "
              f"pitch err max {s['pitch_err_max']:.4f}", flush=True)
        print(f"    windows {s['n_windows']:,}  scored {s['scored_targets']:,}  "
              f"snapshot {s['snapshot_tokens']:,}  pad {s['padding_tokens']:,}  "
              f"{s['median_tokens_per_sec']:.1f} tok/s  "
              f"{s['seconds_in_1024']:.1f}s/1024  "
              f"groups {s['groups_covered']:,}/{s['groups_total']:,}", flush=True)

    ok = all(
        sum(v["mismatches"].values()) == 0 and v["delta_sums_exact"]
        and v["forced_closures"] == 0
        and v["n_rebuilt_notes"] == v["n_canonical_notes"]
        and v["n_changes"] == v["n_canonical_changes"]
        and v["groups_covered"] == v["groups_total"]
        and v["pitch_err_max"] <= PITCH_TOLERANCE
        for v in per_split.values())
    payload = {
        "schema": "snes_v2.g3_audit.v1",
        "grid_hz": GRID_HZ, "context": CONTEXT,
        "vocab_size": vocab.size,
        "comparison": "G3 parse rebuilt against codec.decode, a DIFFERENT "
                      "decoder, so a shared defective transition cannot agree "
                      "with itself",
        "pitch_tolerance_semitones": PITCH_TOLERANCE,
        "scored_targets_note": "next-token shift, minus snapshot positions "
                               "which are conditioning rather than output",
        "per_split": per_split,
        "all_pass": ok,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=1))
    print(f"\n  ALL: {'PASS' if ok else 'FAIL'}\n  written to {args.out}")


if __name__ == "__main__":
    main()
