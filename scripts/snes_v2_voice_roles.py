#!/usr/bin/env python3
"""Is there a stable lead voice to train a melody model on? (RSE v2, step 2)

**There is no ground-truth lead-voice label in SNES-MDB.** `sample_id` is a
content identity, not an instrument label, and the same decoded BRR appears in
four unrelated games. So this script cannot measure whether a candidate voice
IS the melody. It measures something weaker and still decisive:

    does a stable, separable lead candidate exist at all?

If no voice holds a consistent high-register, pitch-varying role across a
track, then "train on the melody voice" has no referent in this corpus and
step 2 needs a different selection rule before any model is trained.

Three candidate criteria are scored, all computed per track over TRAIN only:

- `highest_median_pitch` — the pitched voice with the highest median note.
- `most_distinct_pitches` — the voice using the largest pitch vocabulary.
- `highest_interval_rate` — the voice that changes pitch most often per second.

For each, stability is the fraction of 5-second windows in which the criterion
picks the SAME voice it picks for the track overall. A criterion that flips
between voices every few seconds does not identify a part.

Agreement between criteria is also reported: three rules independently landing
on one voice is far better evidence than any single rule.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rse.snes_lm.release import Release  # noqa: E402

WINDOW_SEC = 5.0
MIN_PITCHED_SHARE = 0.5
MIN_EVENTS_PER_VOICE = 8


def voice_stats(events) -> dict[int, dict]:
    by_voice: dict[int, list] = collections.defaultdict(list)
    for e in events:
        if e.get("origin") != "key_on":
            continue
        by_voice[int(e["voice"])].append(e)
    out = {}
    for voice, evs in by_voice.items():
        pitched = [float(e["midi_note"]) for e in evs
                   if e.get("midi_note") is not None]
        if len(evs) < MIN_EVENTS_PER_VOICE:
            continue
        share = len(pitched) / len(evs)
        if share < MIN_PITCHED_SHARE or len(pitched) < 2:
            continue
        span = (max(float(e["onset_sec"]) for e in evs)
                - min(float(e["onset_sec"]) for e in evs)) or 1e-9
        intervals = sum(1 for a, b in zip(pitched, pitched[1:])
                        if abs(b - a) > 0.5)
        out[voice] = {
            "n_events": len(evs),
            "pitched_share": share,
            "median_pitch": st.median(pitched),
            "distinct_pitches": len({round(p) for p in pitched}),
            "interval_rate": intervals / span,
        }
    return out


CRITERIA = {
    "highest_median_pitch": lambda s: s["median_pitch"],
    "most_distinct_pitches": lambda s: s["distinct_pitches"],
    "highest_interval_rate": lambda s: s["interval_rate"],
}


def pick(stats: dict[int, dict], criterion) -> int | None:
    if not stats:
        return None
    return max(stats, key=lambda v: (CRITERIA[criterion](stats[v]), -v))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", default="~/snesmdb-releases/snesmdb-50game-rc1")
    parser.add_argument("--n-tracks", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=str(Path.home() /
                        "workplace/v2_artifacts/voice_roles.json"))
    args = parser.parse_args()

    release = Release(args.release)
    tracks = release.tracks(split="train")       # TRAIN ONLY
    random.Random(args.seed).shuffle(tracks)
    tracks = tracks[:args.n_tracks]

    per_track = []
    for track in tracks:
        events = release.events(track)
        if not events:
            continue
        stats = voice_stats(events)
        if len(stats) < 2:
            per_track.append({"n_candidate_voices": len(stats),
                              "usable": False})
            continue
        overall = {c: pick(stats, c) for c in CRITERIA}
        # stability over 5 s windows
        end = max(float(e["onset_sec"]) for e in events)
        stability = {}
        for c in CRITERIA:
            agree = total = 0
            t = 0.0
            while t < end:
                window = [e for e in events
                          if t <= float(e["onset_sec"]) < t + WINDOW_SEC]
                ws = voice_stats(window)
                if len(ws) >= 2:
                    total += 1
                    agree += int(pick(ws, c) == overall[c])
                t += WINDOW_SEC
            stability[c] = agree / total if total else None
        per_track.append({
            "usable": True,
            "n_candidate_voices": len(stats),
            "picks": overall,
            "stability": stability,
            "all_three_agree": len(set(overall.values())) == 1,
            "n_windows": total,
        })

    usable = [t for t in per_track if t["usable"]]
    report = {
        "schema": "snes_v2.voice_roles.v1",
        "split": "train",
        "n_tracks_examined": len(per_track),
        "n_usable": len(usable),
        "limitation": "No ground-truth lead-voice label exists in SNES-MDB. "
                      "This measures whether a stable separable candidate "
                      "exists, NOT whether it is the melody.",
        "window_sec": WINDOW_SEC,
        "per_track": per_track,
    }
    for c in CRITERIA:
        vals = [t["stability"][c] for t in usable
                if t["stability"].get(c) is not None]
        report[f"stability_{c}"] = {
            "median": st.median(vals) if vals else None,
            "mean": sum(vals) / len(vals) if vals else None,
            "fraction_above_0.8": (sum(1 for v in vals if v > 0.8) / len(vals)
                                   if vals else None),
            "n": len(vals),
        }
    agree = [t["all_three_agree"] for t in usable]
    report["fraction_all_three_criteria_agree"] = (
        sum(agree) / len(agree) if agree else None)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1))

    print(f"  tracks examined {len(per_track)}, usable (>=2 pitched voices) "
          f"{len(usable)}")
    print(f"  all three criteria agree on one voice: "
          f"{100*report['fraction_all_three_criteria_agree']:.1f}% of tracks")
    for c in CRITERIA:
        s = report[f"stability_{c}"]
        print(f"  {c:24} stability median {s['median']:.3f}  "
              f"mean {s['mean']:.3f}  >0.8 in {100*s['fraction_above_0.8']:.1f}%"
              f"  (n={s['n']})")
    print(f"\n  written to {args.out}")


if __name__ == "__main__":
    main()
