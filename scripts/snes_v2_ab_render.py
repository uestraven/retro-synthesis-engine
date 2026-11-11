#!/usr/bin/env python3
"""RSE v2 milestone 1 item 4: blind A/B of the 250 Hz reconstruction.

    A = published `snesmdb.events.v4` performance, rendered directly
    B = same performance -> 250 Hz codec -> decoder -> identical renderer

**Both sides are rendered from one common "segment" form**, so the comparison
isolates the codec rather than the renderer. A published legato_step is a
separate event row and becomes its own segment; a decoded note carrying pitch
changes is split into consecutive segments at those changes. Without this, B
would lose all legato movement for a RENDERER reason and the test would
measure the wrong thing.

The listening test assesses the WHOLE reconstruction, not only voice
reassignment: quantisation also shifts onsets and durations and merged 3,210
distinct instants at 250 Hz across the development sample.

Files are written with neutral randomised names and the identity key is kept
in a separate file, so the listener cannot know which recording is the
original. One pair is an IDENTICAL-SOURCE control: the same rendering twice
under two names, as a sanity check on the listening setup.

Five tracks cannot establish perceptual equivalence. They can reveal obvious
reconstruction defects before a full-corpus audit is paid for.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics as st
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rse.exporters.wav import write_wav  # noqa: E402
from rse.snes_lm import hardware_checks as hc  # noqa: E402
from rse.snes_lm.release import Release  # noqa: E402
from rse.snes_v2 import codec as cx  # noqa: E402

SAMPLE_RATE = 44_100
WINDOW_SEC = 25.0
VOICE_GAIN = 1.0 / 8.0
ATTACK_SEC = 0.004
RELEASE_SEC = 0.015
GRID_HZ = 250.0
#: Positive control. A real rung from the sweep (1,188 steals, 9.85 ms mean
#: onset error), not a synthetic degradation, so "can the setup hear a codec
#: difference at all" is answered by the codec itself.
CONTROL_GRID_HZ = 25.0
CONTROL_TRACKS = ("actr-01", "al3-03", "fz-01", "sai-03", "sf-04")


# --- one common segment form ---------------------------------------------

def segments_from_published(events) -> list[dict]:
    """Each event with a duration sounds as one segment, legato included."""
    out = []
    for e in events:
        d = e.get("duration_sec")
        if d is None:
            continue
        start = float(e["onset_sec"])
        out.append({"onset": start, "end": start + float(d),
                    "midi_note": e.get("midi_note"),
                    "voice": int(e["voice"])})
    return out


def segments_from_decoded(notes) -> list[dict]:
    """A decoded note is split at each pitch change into consecutive spans."""
    out = []
    for n in notes:
        start, end = float(n["onset_sec"]), float(n["onset_sec"]) + float(n["duration_sec"])
        changes = sorted(n.get("pitch_changes") or [],
                         key=lambda c: float(c["onset_sec"]))
        pitch = n.get("midi_note")
        cursor = start
        for change in changes:
            at = float(change["onset_sec"])
            if at <= cursor or at >= end:
                continue
            out.append({"onset": cursor, "end": at, "midi_note": pitch,
                        "voice": int(n["voice"])})
            cursor, pitch = at, change.get("midi_note")
        out.append({"onset": cursor, "end": end, "midi_note": pitch,
                    "voice": int(n["voice"])})
    return out


# --- one renderer over segments ------------------------------------------

def _tone(frequency: float, n: int) -> np.ndarray:
    t = np.arange(n, dtype=np.float64) / SAMPLE_RATE
    phase = 2.0 * np.pi * frequency * t
    return (np.sin(phase) + 0.20 * np.sin(3.0 * phase)
            + 0.08 * np.sin(5.0 * phase)) / 1.28


def _noise(n: int, rng) -> np.ndarray:
    raw = rng.standard_normal(n)
    out = np.empty(n); state = 0.0
    for i in range(n):
        state = 0.55 * state + 0.45 * raw[i]
        out[i] = state
    peak = np.max(np.abs(out)) or 1.0
    return out / peak


def _envelope(n: int) -> np.ndarray:
    env = np.ones(n)
    a = min(int(ATTACK_SEC * SAMPLE_RATE), n)
    if a > 0:
        env[:a] = np.linspace(0.0, 1.0, a)
    r = min(int(RELEASE_SEC * SAMPLE_RATE), n - a if n > a else 0)
    if r > 0:
        env[n - r:] *= np.linspace(1.0, 0.0, r)
    return env


def render(segments: list[dict]) -> tuple[np.ndarray, dict]:
    """Identical settings for both sides. Voice stealing applied to both."""
    events = [{"voice": s["voice"], "onset_sec": s["onset"],
               "duration_sec": max(0.0, s["end"] - s["onset"]),
               "midi_note": s["midi_note"]} for s in segments]
    stolen, n_clamped = hc.apply_voice_stealing(events)
    buffer = np.zeros(int(WINDOW_SEC * SAMPLE_RATE))
    rng = np.random.default_rng(20261111)
    pitched = unpitched = 0
    for e in stolen:
        length = int(float(e["duration_sec"]) * SAMPLE_RATE)
        start = int(float(e["onset_sec"]) * SAMPLE_RATE)
        if length <= 1 or start >= buffer.size:
            continue
        length = min(length, buffer.size - start)
        note = e.get("midi_note")
        if note is None:
            length = min(length, int(0.25 * SAMPLE_RATE))
            buffer[start:start + length] += (
                VOICE_GAIN * _noise(length, rng) * _envelope(length))
            unpitched += 1
            continue
        f = 440.0 * (2.0 ** ((float(note) - 69.0) / 12.0))
        if not (8.0 <= f <= SAMPLE_RATE / 2.0):
            continue
        buffer[start:start + length] += (
            VOICE_GAIN * _tone(f, length) * _envelope(length))
        pitched += 1
    peak = float(np.max(np.abs(buffer)))
    limited = peak > 1.0
    if limited:
        buffer = np.tanh(buffer)
    return buffer, {"n_segments": len(segments), "n_pitched": pitched,
                    "n_unpitched": unpitched, "n_clamped": n_clamped,
                    "peak": peak, "limiter_engaged": limited}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", default="~/snesmdb-releases/snesmdb-50game-rc1")
    parser.add_argument("--out-dir", default=str(Path.home() / "workplace/v2_ab_test"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--mixed-grids", action="store_true",
                        help="emit BOTH 250 Hz and 25 Hz pairs, "
                             "shuffled, so the null has a positive control")
    parser.add_argument("--legato-extra", action="store_true",
                        help="add an OPTIONAL sixth pair on a legato-bearing "
                             "track; legato is untested by the five controls")
    args = parser.parse_args()

    release = Release(args.release)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    chosen = []
    for track in release.tracks(split="test"):
        name = str(getattr(track, "name", None) or getattr(track, "track_id", ""))
        if name.startswith(CONTROL_TRACKS):
            chosen.append((name, track, False))
    chosen.sort(key=lambda c: c[0])

    if args.legato_extra:
        for track in release.tracks(split="test"):
            ev = [e for e in release.events(track)
                  if float(e["onset_sec"]) < WINDOW_SEC]
            leg = [e for e in ev if e.get("origin") == "legato_step"
                   and e.get("duration_sec") is not None]
            if len(ev) >= 40 and len(leg) >= 5:
                name = str(getattr(track, "name", None)
                           or getattr(track, "track_id", ""))
                chosen.append((name, track, True))
                break

    measurements, keys = [], []
    rng = random.Random(args.seed)
    grids = [GRID_HZ, CONTROL_GRID_HZ] if args.mixed_grids else [GRID_HZ]
    plan = [(n, t, x, g) for (n, t, x) in chosen for g in grids]
    rng.shuffle(plan)
    for name, track, is_extra, grid in plan:
        events = [e for e in release.events(track)
                  if float(e["onset_sec"]) < WINDOW_SEC]
        groups, enc = cx.encode(events, grid_hz=grid)
        notes, dec = cx.decode(groups, grid_hz=grid)
        # Do NOT filter decoded notes by the window: an attack at 24.999590 s
        # quantises to exactly 25.0 at 250 Hz, and filtering it reported a
        # codec loss that was really this harness truncating its own output.

        seg_a = segments_from_published(events)
        seg_b = segments_from_decoded(notes)
        audio_a, ra = render(seg_a)
        audio_b, rb = render(seg_b)

        # note-for-note error, matched by note_id
        starts = [e for e in events if e.get("origin") != "legato_step"]
        starts.sort(key=lambda e: (float(e["onset_sec"]), int(e["voice"])))
        by_id = {n["note_id"]: n for n in notes}
        on_err, du_err, reassigned, matched = [], [], 0, 0
        for note_id, src in enumerate(starts):
            got = by_id.get(note_id)
            if got is None:
                continue
            matched += 1
            on_err.append(abs(got["onset_sec"] - float(src["onset_sec"])))
            d = src.get("duration_sec")
            if d is not None and got.get("end_source") == "derived_duration":
                du_err.append(abs(got["duration_sec"] - float(d)))
            if int(got["voice"]) != int(src["voice"]):
                reassigned += 1

        n = min(audio_a.size, audio_b.size)
        diff = audio_a[:n] - audio_b[:n]
        rms = lambda x: float(np.sqrt(np.mean(x ** 2))) or 1e-12
        measurements.append({
            "track": name,
            "optional_legato_extra": is_extra,
            "grid_hz": grid,
            "n_published_events": len(events),
            "n_published_attacks": len(starts),
            "n_published_legato": sum(
                1 for e in events if e.get("origin") == "legato_step"),
            "n_decoded_notes": len(notes),
            "n_retained_pitch_changes": sum(
                len(x.get("pitch_changes") or []) for x in notes),
            "n_matched": matched,
            "n_lost_attacks": len(starts) - matched,
            "n_instants_merged": enc.n_instants_merged,
            "n_indistinguishable_starts": enc.n_indistinguishable_starts,
            "decoder_voice_steals": dec.n_voice_steals,
            "peak_occupied_voices": dec.peak_occupancy,
            "pct_notes_reassigned_voice": (
                100.0 * reassigned / matched if matched else None),
            "onset_err_mean_ms": 1000 * st.mean(on_err) if on_err else None,
            "onset_err_max_ms": 1000 * max(on_err) if on_err else None,
            "dur_err_mean_ms": 1000 * st.mean(du_err) if du_err else None,
            "dur_err_max_ms": 1000 * max(du_err) if du_err else None,
            "render_a": ra, "render_b": rb,
            "audio_rms_a": rms(audio_a), "audio_rms_b": rms(audio_b),
            "aligned_diff_rms_db_rel": 20 * np.log10(rms(diff) / rms(audio_a)),
            "audio_duration_diff_sec": 0.0,
        })

        pair = [("A_published", audio_a), (f"B_codec{int(grid)}", audio_b)]
        rng.shuffle(pair)
        tag = rng.randrange(10 ** 6, 10 ** 7)
        for index, (which, audio) in enumerate(pair):
            fname = f"clip_{tag}_{index + 1}.wav"
            write_wav(out_dir / fname, audio, SAMPLE_RATE)
            keys.append({"file": fname, "pair_tag": tag, "track": name,
                         "version": which, "grid_hz": grid,
                         "optional_extra": is_extra})

    # identical-source sanity control: one rendering written twice
    name, track, _ = chosen[0]
    events = [e for e in release.events(track)
              if float(e["onset_sec"]) < WINDOW_SEC]
    audio, _ = render(segments_from_published(events))
    tag = rng.randrange(10 ** 6, 10 ** 7)
    for index in (1, 2):
        fname = f"clip_{tag}_{index}.wav"
        write_wav(out_dir / fname, audio, SAMPLE_RATE)
        keys.append({"file": fname, "pair_tag": tag, "track": name,
                     "version": "IDENTICAL_SOURCE_CONTROL",
                     "optional_extra": False})

    (out_dir / "measurements.json").write_text(json.dumps({
        "schema": "snes_v2.ab_test.v1",
        "grid_hz": GRID_HZ,
        "window_sec": WINDOW_SEC,
        "renderer": "one common segment form for both sides; identical "
                    "synthesis, gain, percussion mapping and output format",
        "limitation": "Five tracks cannot establish perceptual equivalence. "
                      "Aligned waveform error is reported but is a poor "
                      "perceptual proxy: a few ms of note displacement "
                      "produces a large sample-wise difference while sounding "
                      "identical.",
        "measurements": measurements,
    }, indent=1))
    rng.shuffle(keys)
    (out_dir / "IDENTITY_KEY.json").write_text(json.dumps(
        {"note": "Do not open before listening.", "keys": keys}, indent=1))
    manifest = sorted({k["pair_tag"] for k in keys})
    (out_dir / "LISTENING_MANIFEST.md").write_text(
        "# Blind listening manifest\n\n"
        "Each pair shares a tag. Compare the two clips within a pair.\n"
        "One pair is an identical-source control.\n\n"
        + "\n".join(f"- pair `{t}`: `clip_{t}_1.wav` vs `clip_{t}_2.wav`"
                    for t in manifest)
        + "\n\nFor each pair: (1) can you distinguish them? (2) if so, what "
          "changed — timing, note length, overlap, percussion, texture? "
          "(3) does either sound less musically coherent?\n")

    for m in measurements:
        print(f"  {m['track'][:22]:22} att {m['n_published_attacks']:4} -> "
              f"notes {m['n_decoded_notes']:4}  lost {m['n_lost_attacks']:3}  "
              f"steals {m['decoder_voice_steals']:3}  peak {m['peak_occupied_voices']}  "
              f"reassigned {m['pct_notes_reassigned_voice']:5.1f}%  "
              f"onset {m['onset_err_mean_ms']:5.2f} ms  "
              f"dur {m['dur_err_mean_ms']:5.2f} ms  "
              f"wavdiff {m['aligned_diff_rms_db_rel']:+6.1f} dB", flush=True)
    print(f"\n  {len(keys)} clips in {out_dir}")
    print(f"  manifest: {out_dir/'LISTENING_MANIFEST.md'}")


if __name__ == "__main__":
    main()
