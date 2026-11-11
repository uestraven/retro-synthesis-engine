#!/usr/bin/env python3
"""Renderer-sensitivity calibration stimuli. No generated music involved.

This answers one question before Milestone 3 can be registered: **can this
renderer and listening task distinguish destroyed music from real music?**

Level 2's listening nulls were worthless because its 25 Hz positive control
moved sounding RMS by 4% and the waveform difference not at all, so
"indistinguishable" was the expected outcome regardless of the codec. A null
from an uncalibrated setup is not evidence.

Three arms, all from GENUINE validation music, all rendered identically:

- **genuine** -- unmodified. Should be judged coherent at >=90%.
- **pitch_permuted** (primary control) -- pitched values permuted among
  pitched events, with a fixed seed. Event times, note lifetimes, attack
  count and the overall pitch histogram are preserved EXACTLY, because a
  permutation is a bijection on the multiset. Unpitched events are untouched.
  This destroys melodic and harmonic relationships while leaving the
  superficial statistics identical, so it exercises what the model is
  supposed to learn rather than asking a listener to hear a small duration
  error. Should be judged coherent at <=20%.
- **phrase_shuffled** (secondary control) -- whole chunks reordered with
  their internal notes intact, cut on inferred bar boundaries. Tests
  sensitivity to long-range organisation, which pitch permutation does not
  isolate. Reported SEPARATELY with no assumed threshold.

Explicitly not used: silence, static noise, or deliberately unpleasant
synthesis. Those test whether a listener can recognise broken audio, not
broken music.

**The degradation is fixed in advance and is not escalated to reach a target
detection rate.** If a prespecified control fails to separate, that result is
recorded and the renderer or the task is investigated.

Validation split only -- the test split stays sealed.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rse.exporters.wav import write_wav  # noqa: E402
from rse.snes_lm.release import Release  # noqa: E402

#: Reuse the Level 2 renderer unchanged, so calibration measures THAT renderer.
_spec = importlib.util.spec_from_file_location(
    "ab", ROOT / "scripts" / "snes_v2_ab_render.py")
_ab = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ab)

EXCERPT_SEC = 50.0
N_PER_ARM = 10
BIN_SEC = 0.010
MIN_LAG, MAX_LAG = 0.3, 8.0
CHUNK_BARS = 4


def excerpt(events, seconds: float) -> list[dict]:
    return [e for e in events
            if e.get("origin") == "key_on"
            and float(e["onset_sec"]) < seconds
            and e.get("duration_sec") is not None]


def pitch_permute(events, rng: random.Random) -> list[dict]:
    """Permute pitched values; preserve times, lifetimes, counts, histogram."""
    out = [dict(e) for e in events]
    index = [i for i, e in enumerate(out) if e.get("midi_note") is not None]
    values = [out[i]["midi_note"] for i in index]
    rng.shuffle(values)
    for i, value in zip(index, values):
        out[i]["midi_note"] = value
    return out


#: Band width in semitones for the constrained permutation. A pitch may only
#: be replaced by another from its own band, bounding displacement.
BAND_SEMITONES = 3.0


def pitch_permute_constrained(events, rng: random.Random) -> list[dict]:
    """Permute WITHIN register bands, so no pitch moves far.

    Preserves the pitch multiset, timing, event count and note lifetimes
    exactly, like the free permutation, but a value can only be replaced by
    another from the same `BAND_SEMITONES` band. The free permutation destroys
    voice-leading and manufactures large register leaps, so detection there may
    be driven by unparseable jumps rather than by harmonic relationships; this
    arm exists to separate those two explanations.

    Bounding the swap distance does NOT by itself guarantee small melodic
    intervals -- adjacent notes in time can sit in different bands -- so the
    resulting interval distribution is measured and reported rather than
    assumed.
    """
    out = [dict(e) for e in events]
    bands: dict[int, list[int]] = {}
    for i, e in enumerate(out):
        if e.get("midi_note") is None:
            continue
        bands.setdefault(int(float(e["midi_note"]) // BAND_SEMITONES), []).append(i)
    for index in bands.values():
        values = [out[i]["midi_note"] for i in index]
        rng.shuffle(values)
        for i, value in zip(index, values):
            out[i]["midi_note"] = value
    return out


def interval_profile(events) -> dict:
    """Absolute melodic intervals between consecutive pitched attacks, per voice."""
    import statistics as _st
    by_voice: dict[int, list[float]] = {}
    for e in events:
        note = e.get("midi_note")
        if note is None:
            continue
        by_voice.setdefault(int(e["voice"]), []).append(float(note))
    steps = [abs(b - a) for pitches in by_voice.values()
             for a, b in zip(pitches, pitches[1:])]
    if not steps:
        return {}
    steps.sort()
    return {
        "n": len(steps),
        "median": round(_st.median(steps), 3),
        "mean": round(_st.mean(steps), 3),
        "p90": round(steps[int(0.90 * len(steps))], 3),
        "p99": round(steps[int(0.99 * len(steps))], 3),
        "max": round(steps[-1], 3),
        "frac_over_12_semitones": round(
            sum(1 for s in steps if s > 12) / len(steps), 4),
    }


def infer_bar_sec(events) -> float | None:
    """Bar length by onset autocorrelation; the measured 96.9%-recoverable one."""
    onsets = [float(e["onset_sec"]) for e in events]
    if len(onsets) < 40:
        return None
    n = int(EXCERPT_SEC / BIN_SEC)
    counts = np.zeros(n)
    for o in onsets:
        counts[min(int(o / BIN_SEC), n - 1)] += 1
    counts -= counts.mean()
    if not counts.any():
        return None
    full = np.correlate(counts, counts, mode="full")[n - 1:]
    if full[0] <= 0:
        return None
    ac = full / full[0]
    lo, hi = int(MIN_LAG / BIN_SEC), int(MAX_LAG / BIN_SEC)
    window = ac[lo:hi]
    best = int(np.argmax(window)) + lo
    # Accept only with harmonic confirmation, as the meter study required.
    if window.max() < 0.30:
        return None
    double = best * 2
    if double >= len(ac) or ac[double] < 0.20:
        return None
    return best * BIN_SEC


def phrase_shuffle(events, rng: random.Random) -> list[dict] | None:
    """Reorder whole chunks, preserving each chunk's internal notes."""
    bar = infer_bar_sec(events)
    if bar is None:
        return None
    chunk = bar * CHUNK_BARS
    n_chunks = max(2, int(EXCERPT_SEC / chunk))
    buckets: list[list[dict]] = [[] for _ in range(n_chunks)]
    for e in events:
        onset = float(e["onset_sec"])
        which = min(int(onset / chunk), n_chunks - 1)
        copy = dict(e)
        copy["_offset"] = onset - which * chunk
        # Truncate at the chunk boundary so a note cannot bridge a seam it
        # will no longer be adjacent to.
        copy["duration_sec"] = min(
            float(e["duration_sec"]), max(0.0, chunk - copy["_offset"]))
        buckets[which].append(copy)
    order = list(range(n_chunks))
    rng.shuffle(order)
    out: list[dict] = []
    for position, source in enumerate(order):
        base = position * chunk
        for e in buckets[source]:
            moved = dict(e)
            moved["onset_sec"] = base + e["_offset"]
            moved.pop("_offset")
            out.append(moved)
    out.sort(key=lambda e: (float(e["onset_sec"]), int(e["voice"])))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", default="~/snesmdb-releases/snesmdb-50game-rc1")
    parser.add_argument("--out-dir",
                        default=str(Path.home() / "workplace/v2_calibration"))
    parser.add_argument("--seed", type=int, default=1511)
    #: Tracks used in the FIRST calibration. Reusing them would create
    #: recognition and expectation effects even under fresh random filenames,
    #: because the listener already knows that set's outcome.
    parser.add_argument("--exclude", default=(
        "bmg-13,hag-05,hag-10,scv4-06,scv4-09,scv4-16,scv4-21,"
        "sf2t-03,sf2t-24,sm-29"))
    args = parser.parse_args()

    rng = random.Random(args.seed)
    release = Release(args.release)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Validation tracks with enough pitched material to judge.
    candidates = []
    for track in release.tracks(split="valid"):
        events = excerpt(release.events(track), EXCERPT_SEC)
        if len(events) < 120:
            continue
        pitched = sum(1 for e in events if e.get("midi_note") is not None)
        if pitched / len(events) < 0.6:
            continue
        name = str(getattr(track, "track_id", None) or "")
        if any(name.startswith(x) for x in args.exclude.split(",") if x):
            continue
        candidates.append((name, events))
    candidates.sort(key=lambda c: c[0])
    rng.shuffle(candidates)
    chosen = candidates[:N_PER_ARM]
    print(f"  {len(candidates)} eligible validation tracks; using {len(chosen)}",
          flush=True)

    keys, skipped = [], []
    for name, events in chosen:
        arms = {
            "genuine": events,
            "pitch_permuted_free": pitch_permute(events, rng),
            "pitch_permuted_constrained": pitch_permute_constrained(events, rng),
        }
        for arm, material in arms.items():
            if material is None:
                skipped.append({"track": name, "arm": arm,
                                "reason": "no bar length passed its validation"})
                continue
            segments = _ab.segments_from_published(material)
            audio, stats = _ab.render(segments)
            tag = rng.randrange(10 ** 6, 10 ** 7)
            fname = f"cal_{tag}.wav"
            write_wav(out_dir / fname, audio, _ab.SAMPLE_RATE)
            keys.append({"file": fname, "track": name, "arm": arm,
                         "n_events": len(material),
                         "peak": stats["peak"],
                         "limiter_engaged": stats["limiter_engaged"],
                         "interval_profile": interval_profile(material)})
            print(f"    {arm:16} {name[:22]:22} {len(material):5} ev  "
                  f"peak {stats['peak']:.3f}", flush=True)

    rng.shuffle(keys)
    (out_dir / "IDENTITY_KEY.json").write_text(json.dumps(
        {"note": "Do not open until all judgements are recorded.",
         "seed": args.seed, "skipped": skipped, "keys": keys}, indent=1))
    (out_dir / "LISTENING_SHEET.md").write_text(
        "# Renderer-sensitivity calibration\n\n"
        "**No RSE v2 output is in this set.** Every clip is genuine SNES-MDB\n"
        "validation music, either unmodified or deliberately damaged. The\n"
        "purpose is to check whether this renderer and this task can tell\n"
        "destroyed music from real music, BEFORE any model is trained.\n\n"
        "For each clip, answer one question:\n\n"
        "> **Score each clip 2, 1 or 0.**\n\n"
        "| score | label | definition |\n"
        "|---|---|---|\n"
        "| **2** | Coherent | Discernible musical phrases AND pitch "
        "relationships that sound intentional |\n"
        "| **1** | Partially coherent | Some musical organisation is audible, "
        "but one criterion is unclear or inconsistent |\n"
        "| **0** | Incoherent | No convincing phrase organisation, and no "
        "pitch relationships that sound intentional |\n\n"
        "Judge **structure**, not enjoyment, familiarity, production quality\n"
        "or stylistic preference. The timbre is deliberately plain in every\n"
        "clip and is not what is being tested. A dull but organised passage\n"
        "is **2**; an interesting but incoherent one is **0**. No\n"
        "music-theory knowledge is needed.\n\n"
        "Replace each `[ ]` with `[2]`, `[1]` or `[0]`.\n\n"
        "## Clips\n\n"
        + "\n".join(f"- [ ] `{k['file']}`" for k in sorted(
            keys, key=lambda k: k["file"]))
        + "\n")
    print(f"\n  {len(keys)} clips in {out_dir}")
    if skipped:
        print(f"  {len(skipped)} skipped: {skipped[:3]}")


if __name__ == "__main__":
    main()
