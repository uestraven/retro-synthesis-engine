#!/usr/bin/env python3
"""Render both Level 2 arms identically, for the diagnostic listening pilot.

**This is not a hardware-accurate renderer and no claim of accuracy is made.**
It exists to make rhythm, density and voice behaviour audible, nothing more.

**One fixed timbre for every voice, both arms.** The representation carries
only TIME, VOICE, PITCH and DUR -- there is no `sample_id`, so the events hold
no instrument information at all. Assigning timbres would mean inventing them,
and any invention could differ between arms and become a confound. A single
neutral tone is the symmetric, auditable choice.

**No per-excerpt normalisation.** Gain is fixed at 1/8 per voice, so eight
simultaneous voices sum to at most full scale. A quiet excerpt renders quiet.
Normalising each excerpt to a target level would erase exactly the density
difference the pilot is meant to expose.

**Nothing is excluded, repaired, or substituted.**

- An excerpt that fails the structural criterion is preserved as a failure.
  Its valid PREFIX may be rendered for diagnosis, labelled as a prefix. Its
  grammar is never repaired and no other seed is substituted for it.
- A runaway excerpt is rendered as the timeline it actually produced. Its music
  ends when it ends and the remainder is SILENCE -- never stretched, looped or
  regenerated to fill the window.
- Voice stealing is applied, so a thousand-note instant becomes the repeated
  instantaneous voice replacement the hardware would perform. Rendering
  unlimited overlapping voices would invent a sound the hardware cannot make
  and would risk extreme levels.

Every outcome appears in the manifest whether or not it produced audio.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rse.exporters.wav import write_wav  # noqa: E402
from rse.snes_lm import hardware_checks as hc  # noqa: E402
from rse.snes_lm import representation as rep  # noqa: E402

SAMPLE_RATE = 44_100
WINDOW_SEC = 25.0
#: Eight voices at 1/8 each cannot exceed full scale, so the limiter should
#: never engage. If it does, that is a finding and the manifest records it.
VOICE_GAIN = 1.0 / 8.0
ATTACK_SEC = 0.004
RELEASE_SEC = 0.015
CAP_EVENTS = 4_000


def _tone(frequency: float, n: int) -> np.ndarray:
    """A fixed, deliberately plain tone: fundamental plus two quiet odd partials."""
    t = np.arange(n, dtype=np.float64) / SAMPLE_RATE
    phase = 2.0 * np.pi * frequency * t
    return (np.sin(phase)
            + 0.20 * np.sin(3.0 * phase)
            + 0.08 * np.sin(5.0 * phase)) / 1.28


def _envelope(n: int) -> np.ndarray:
    env = np.ones(n, dtype=np.float64)
    attack = min(int(ATTACK_SEC * SAMPLE_RATE), n)
    if attack > 0:
        env[:attack] = np.linspace(0.0, 1.0, attack)
    release = min(int(RELEASE_SEC * SAMPLE_RATE), n - attack if n > attack else 0)
    if release > 0:
        env[n - release:] *= np.linspace(1.0, 0.0, release)
    return env


def _noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """A fixed percussive burst for an UNPITCHED event.

    `PITCH_NONE` is not absence of content -- the corpus's unpitched events are
    real percussion, and 2 of 60 control tracks are entirely unpitched. An
    earlier version of this script skipped them, which rendered a 323-event
    excerpt as pure silence and would have presented that silence to a listener
    as the model's output. Unpitched events therefore sound, with the same
    fixed treatment in both arms.
    """
    raw = rng.standard_normal(n)
    # Cheap one-pole lowpass, so the burst reads as a drum rather than a click.
    out = np.empty(n, dtype=np.float64)
    state = 0.0
    for i in range(n):
        state = 0.55 * state + 0.45 * raw[i]
        out[i] = state
    peak = np.max(np.abs(out)) or 1.0
    return out / peak


def render(events: list[dict]) -> tuple[np.ndarray, dict]:
    """Render AFTER voice stealing. Returns the buffer and render statistics."""
    stolen, n_clamped = hc.apply_voice_stealing(events)
    buffer = np.zeros(int(WINDOW_SEC * SAMPLE_RATE), dtype=np.float64)
    # Fixed seed: renders are reproducible and identical across arms.
    rng = np.random.default_rng(20261111)
    sounded = pitchless = zero_length = truncated = 0
    unpitched = no_duration = 0
    for event in stolen:
        note = event.get("midi_note")
        duration = event.get("duration_sec")
        if duration is None:
            no_duration += 1
            continue
        if note is None:
            # Unpitched: a percussive burst, not silence.
            start = int(float(event["onset_sec"]) * SAMPLE_RATE)
            length = int(float(duration) * SAMPLE_RATE)
            if length <= 1:
                zero_length += 1
                continue
            if start >= buffer.size:
                truncated += 1
                continue
            length = min(length, buffer.size - start, int(0.25 * SAMPLE_RATE))
            buffer[start:start + length] += (
                VOICE_GAIN * _noise(length, rng) * _envelope(length)
            )
            unpitched += 1
            continue
        start = int(float(event["onset_sec"]) * SAMPLE_RATE)
        length = int(float(duration) * SAMPLE_RATE)
        if length <= 1:
            zero_length += 1
            continue
        if start >= buffer.size:
            truncated += 1
            continue
        length = min(length, buffer.size - start)
        frequency = 440.0 * (2.0 ** ((float(note) - 69.0) / 12.0))
        if not (8.0 <= frequency <= SAMPLE_RATE / 2.0):
            pitchless += 1
            continue
        buffer[start:start + length] += (
            VOICE_GAIN * _tone(frequency, length) * _envelope(length)
        )
        sounded += 1

    peak = float(np.max(np.abs(buffer))) if buffer.size else 0.0
    limited = peak > 1.0
    reduction_db = 0.0
    if limited:
        # Fixed, reported limiting. Never silent: an excerpt that needed it is
        # an excerpt whose level the fixed gain did not bound as intended.
        buffer = np.tanh(buffer)
        new_peak = float(np.max(np.abs(buffer)))
        reduction_db = 20.0 * np.log10(new_peak / peak) if peak > 0 else 0.0
    return buffer, {
        "n_pitched_rendered": sounded,
        "n_unpitched_rendered": unpitched,
        "n_events_rendered": sounded + unpitched,
        "pitched_share": sounded / (sounded + unpitched)
                         if (sounded + unpitched) else None,
        "n_no_duration_skipped": no_duration,
        "n_out_of_range_skipped": pitchless,
        "n_zero_length_after_stealing": zero_length,
        "n_beyond_window": truncated,
        "n_clamped_by_voice_stealing": n_clamped,
        "voice_stealing_rate": n_clamped / len(events) if events else 0.0,
        "peak_before_limiting": peak,
        "limiter_engaged": limited,
        "limiter_gain_reduction_db": reduction_db,
    }


def valid_prefix(tokens: list[int], vocab: rep.Vocab) -> list[int]:
    """The longest leading run of WELL-FORMED events, for diagnosis only.

    Truncation at an event boundary, with the grammar left exactly as the model
    produced it. This repairs nothing: it shows how far the generator got.
    """
    body = [t for t in tokens if t not in (vocab.PAD, vocab.BOS, vocab.EOS)]
    expected = ("time", "voice", "pitch", "dur")
    cut = len(body)
    for index, token in enumerate(body):
        if vocab.group(token) != expected[index % 4]:
            cut = index
            break
    cut -= cut % 4
    return [vocab.BOS] + body[:cut] + [vocab.EOS]


def render_real_control(args, vocab, out_dir: Path, manifest: list[dict]) -> None:
    """Genuine SNES-MDB tracks through the IDENTICAL renderer.

    The diagnostic that separates two very different explanations of a poor
    listening result:

    - real music sounds like music here, generated music does not -> the
      representation and renderer preserve enough musical information and the
      generators are failing to organise it;
    - real music ALSO sounds incoherent -> the pipeline is destroying musical
      information and the listening failure cannot be attributed to the
      generators at all.

    Each track is rendered TWICE, which splits that second case further:

    - `real_published`: the release's own events, full symbolic fidelity. The
      upper bound of what this synthesiser can convey.
    - `real_roundtrip`: the same events through encode then decode, so they
      carry the tokenizer's quantisation -- 96 log-spaced time buckets, 32
      duration quantiles, semitone-rounded pitch. This is the fair comparison
      for the generators, which can only ever emit round-tripped material.

    A gap between those two arms measures the representation. A gap between
    `real_roundtrip` and the generators measures the generators.

    This is a control, not an opportunity to revise a registered outcome.
    """
    from rse.snes_lm.release import Release

    release = Release(args.release)
    chosen = []
    for track in release.tracks(split=args.real_split):
        events = release.events(track)
        if not events:
            continue
        window = [e for e in events if float(e["onset_sec"]) < WINDOW_SEC]
        if len(window) < 40:
            continue
        pitched = sum(1 for e in window if e.get("midi_note") is not None)
        if pitched / len(window) < 0.5:
            continue
        chosen.append((track, window))
        if len(chosen) >= args.real_tracks:
            break

    for index, (track, window) in enumerate(chosen):
        name = (getattr(track, "name", None) or getattr(track, "track_id", None)
                or f"track{index}")
        tokens = rep.encode(release.events(track), vocab).tokens
        roundtrip = [e for e in rep.decode(tokens, vocab)
                     if float(e["onset_sec"]) < WINDOW_SEC]
        for arm, events in (("real_published", window),
                            ("real_roundtrip", roundtrip)):
            arm_dir = out_dir / arm
            arm_dir.mkdir(parents=True, exist_ok=True)
            if not events:
                manifest.append({"arm": arm, "track": name,
                                 "render_status": "not_rendered_no_events"})
                continue
            span = max(float(e["onset_sec"]) for e in events)
            buffer, stats = render(events)
            safe = str(name).replace("/", "_").replace(":", "_")
            wav = write_wav(arm_dir / f"{arm}_{safe}.wav", buffer, SAMPLE_RATE)
            entry = {
                "arm": arm,
                "track": str(name),
                "split": args.real_split,
                "structurally_valid": True,
                "render_status": "full",
                "n_events": len(events),
                "span_sec": round(span, 4),
                "attacks_per_sec": round(len(events) / span, 3) if span else None,
                "wav": str(wav),
            }
            entry.update(stats)
            manifest.append(entry)
            print(f"  {arm:15} {str(name)[:28]:28} {len(events):5} ev  "
                  f"peak {stats['peak_before_limiting']:.3f}  "
                  f"steal {100*stats['voice_stealing_rate']:5.1f}%  "
                  f"pitched {100*(stats['pitched_share'] or 0):5.1f}%", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab", default="/tmp/snes_vocab.json")
    parser.add_argument("--model-tokens",
                        default="/tmp/snes_level2_raw_aligned/model_tokens.json")
    parser.add_argument("--baseline-tokens",
                        default="/tmp/snes_level2_raw/baseline_tokens.json")
    parser.add_argument("--out-dir",
                        default=str(Path.home() / "workplace/level2_renders"))
    parser.add_argument("--release", default="~/snesmdb-releases/snesmdb-50game-rc1")
    parser.add_argument("--real-tracks", type=int, default=0,
                        help="render N genuine tracks as a diagnostic control")
    parser.add_argument("--real-split", default="test")
    args = parser.parse_args()

    vocab = rep.read_vocab(args.vocab)
    out_dir = Path(args.out_dir)
    manifest: list[dict] = []

    if args.real_tracks:
        render_real_control(args, vocab, out_dir, manifest)

    for arm, path in (("transformer", args.model_tokens),
                      ("slotngram", args.baseline_tokens)):
        arm_dir = out_dir / arm
        arm_dir.mkdir(parents=True, exist_ok=True)
        for row in json.loads(Path(path).read_text()):
            seed = row["seed"]
            entry = {
                "arm": arm,
                "seed": seed,
                "structurally_valid": row["valid"],
                "invalid_reason": row.get("invalid_reason", ""),
                "n_tokens": len(row["tokens"]),
            }
            if row["valid"]:
                tokens = row["tokens"]
                entry["render_status"] = "full"
            else:
                tokens = valid_prefix(row["tokens"], vocab)
                entry["render_status"] = "valid_prefix_only"
                entry["prefix_tokens"] = len(tokens) - 2

            events = rep.decode(tokens, vocab)
            if not events:
                entry.update(render_status="not_rendered_no_events",
                             n_events=0, wav=None)
                manifest.append(entry)
                continue

            span = max(float(e["onset_sec"]) for e in events)
            entry.update(
                n_events=len(events),
                span_sec=round(span, 4),
                attacks_per_sec=round(len(events) / span, 3) if span > 0 else None,
                terminated_by_event_cap=len(events) >= CAP_EVENTS,
                silence_padding_sec=round(max(0.0, WINDOW_SEC - span), 4),
            )
            buffer, stats = render(events)
            entry.update(stats)
            name = f"{arm}_seed{seed:02d}"
            if entry["render_status"] == "valid_prefix_only":
                name += "_prefix"
            if entry.get("terminated_by_event_cap"):
                name += "_RUNAWAY"
            wav = write_wav(arm_dir / f"{name}.wav", buffer, SAMPLE_RATE)
            entry["wav"] = str(wav)
            manifest.append(entry)
            print(f"  {arm:11} seed {seed:2}  {entry['render_status']:22} "
                  f"{entry['n_events']:5} ev  peak {stats['peak_before_limiting']:.3f}"
                  f"{'  LIMITED' if stats['limiter_engaged'] else ''}", flush=True)

    path = out_dir / "manifest.json"
    path.write_text(json.dumps({
        "schema": "snes_lm.level2_render_manifest.v1",
        "sample_rate": SAMPLE_RATE,
        "window_sec": WINDOW_SEC,
        "voice_gain": VOICE_GAIN,
        "synthesis": "one fixed tone for all voices; the representation "
                     "carries no sample_id, so no instrument information "
                     "exists to render. Not hardware-accurate.",
        "normalisation": "none; fixed gain, identical across arms",
        "entries": manifest,
    }, indent=1))
    rendered = sum(1 for e in manifest if e.get("wav"))
    limited = sum(1 for e in manifest if e.get("limiter_engaged"))
    runaway = sum(1 for e in manifest if e.get("terminated_by_event_cap"))
    print(f"\n  {rendered}/{len(manifest)} rendered, {limited} limited, "
          f"{runaway} cap-terminated")
    print(f"  manifest: {path}")


if __name__ == "__main__":
    main()
