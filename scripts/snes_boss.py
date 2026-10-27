"""The first SNES-style boss theme: shared composer -> arranger -> sampler.

SNES-INSPIRED PREVIEW. Sample-based, eight-voice limited, stereo and
hardware-informed. It is not S-DSP-accurate: PCM with linear interpolation
rather than BRR with the hardware's Gaussian filter, floating-point envelopes
rather than the DSP's rate tables, and no ARAM enforcement. Those are separate
verification tasks and nothing here is authentic hardware output.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import json
from collections import Counter

import numpy as np

from rse.exporters.wav import write_wav
from rse.platform.roles import Role, concurrency
from rse.procedural.compose import compose
from rse.snes.aram import budget_for, largest_echo_that_fits
from rse.snes.arrange import VOICE_PLAN, allocate, arrange, percussion_settings
from rse.snes.brr import encode as brr_encode
from rse.snes.sampler import render
from rse.snes.samples import build_bank, manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("listen_snes"))
    parser.add_argument("--name", default="snes_boss")
    parser.add_argument("--style", default="boss")
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--reserved", type=int, default=0)
    parser.add_argument(
        "--echo-ms", type=int, default=64,
        help="echo delay; charged against the same 64 KiB as the samples",
    )
    parser.add_argument(
        "--no-brr", action="store_true",
        help="skip BRR quantisation (the earlier clean-PCM preview)",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 1. the shared composer, unchanged. Its NES defaults still apply.
    result = compose(
        style_name=args.style, seed=args.seed, target_seconds=args.seconds
    )
    # 2. the SNES arranger: re-realise the plan for polyphonic sampled voices.
    song = arrange(result)
    # 3. the allocator decides what physically fits.
    allocation = allocate(song, reserved=args.reserved)
    # 4. the sampler renders what the allocator assigned.
    bank = build_bank(brr=not args.no_brr)
    encoded = {
        name: brr_encode(
            asset.data,
            loop_start=asset.loop_start if asset.loop_end > asset.loop_start else None,
        )
        for name, asset in bank.items()
    }

    voices = dict(VOICE_PLAN)
    rendered = render(
        allocation.assignments,
        bank=bank,
        voices=voices,
        seconds_per_tick=_seconds_per_tick(result),
        allocation=allocation,
    )
    # Percussion is three instruments on one role; render those separately and
    # sum, since a single VoiceSettings cannot be a kick and a hat at once.
    drums = [a for a in allocation.assignments if a.note.role is Role.PERCUSSION]
    for value in sorted({a.note.value for a in drums}):
        hits = [a for a in drums if a.note.value == value]
        layer = render(
            hits,
            bank=bank,
            voices={Role.PERCUSSION: percussion_settings(value)},
            seconds_per_tick=_seconds_per_tick(result),
            peak=None,
        )
        _add(rendered, layer)

    # Re-normalise after the drum layers, with ONE scalar so panning survives.
    loudest = float(np.abs(rendered.samples).max())
    if loudest > 0:
        rendered.samples = (rendered.samples * (0.89 / loudest)).astype(np.float32)

    wav = args.out_dir / f"{args.name}.wav"
    write_wav(wav, rendered.samples, rendered.sample_rate)

    midi_path = args.out_dir / f"{args.name}.mid"
    try:
        from rse.events.convert import composition_to_intake
        from rse.exporters.midi import record_to_midi
        from rse.hardware.profile import load_profile

        record_to_midi(
            composition_to_intake(
                result.composition, profile=load_profile("nes")
            )
        ).save(str(midi_path))
    except Exception as error:  # noqa: BLE001 - reported, not fatal
        print(f"  score export failed: {type(error).__name__}: {error}")
        midi_path = None

    bank_manifest = manifest(bank, encoded=encoded)
    # The score's own size stands in for sequence data: one event per note is
    # a fair order-of-magnitude for a tracker-style sequence.
    sequence_bytes = len(allocation.assignments) * 4
    budget = budget_for(
        encoded, sequence_bytes=sequence_bytes, echo_delay_ms=args.echo_ms
    )
    report = {
        "renderer": "snes_sampler",
        "mode": "SNES-INSPIRED PREVIEW",
        "hardware_exact": False,
        "caveats": [
            "Samples ARE BRR-encoded and the byte counts are measured, but "
            "interpolation is a 4-point Catmull-Rom kernel plus a documented "
            "low-pass, NOT the S-DSP's own 4-tap Gaussian table.",
            "Floating-point ADSR, not the DSP's envelope rate tables.",
            "The ARAM budget is computed and reported; driver and runtime "
            "figures are assumptions, since RSE has no SPC700 driver.",
            "No SPC export and no emulator verification. A correct byte "
            "count is not an SPC file.",
        ],
        "seed": args.seed,
        "style": args.style,
        "config": {
            "style": args.style,
            "seed": args.seed,
            "seconds": args.seconds,
            "reserved_voices": args.reserved,
        },
        "music": {
            "tempo_bpm": round(result.style.tempo_bpm, 2),
            "mode": result.style.mode,
            "tonic": result.style.tonic,
            "form": [s.label for s in result.sections],
            "bars": sum(s.bars for s in result.sections),
            "seconds": round(rendered.seconds, 2),
        },
        "arrangement": song.to_dict(),
        "voice_plan": {
            role.value: {
                "instrument": settings.instrument,
                "pan": settings.pan,
                "gain": settings.gain,
                "echo_send": settings.echo,
            }
            for role, settings in voices.items()
        },
        "allocation": allocation.to_dict(),
        "peak_concurrency_by_role": {
            role.value: concurrency([p for p in song.parts if p.role is role])
            for role in Role
            if any(p.role is role for p in song.parts)
        },
        "audio": rendered.to_dict(),
        "instrument_bank": bank_manifest,
        "aram": budget.to_dict(),
        "brr": {
            "quantisation_applied": not args.no_brr,
            "per_sample": {n: e.to_dict() for n, e in encoded.items()},
            "longest_echo_that_fits_ms": largest_echo_that_fits(budget),
        },
        "files": {
            "wav": str(wav),
            "score": str(midi_path) if midi_path else None,
        },
    }
    out_json = args.out_dir / f"{args.name}.json"
    out_json.write_text(json.dumps(report, indent=2) + "\n")

    audio = rendered.to_dict()
    print("SNES-INSPIRED PREVIEW  (not hardware-exact)\n")
    print(f"style      {args.style}, seed {args.seed}")
    print(f"music      {report['music']['tempo_bpm']} BPM, "
          f"{result.style.mode}, {'-'.join(report['music']['form'])}, "
          f"{report['music']['bars']} bars, {audio['seconds']} s")
    extended_symbols = sorted(
        {v.intent.symbol for v in song.voicings if v.intent.seventh.value != "none"}
    )
    print(f"chords     {len(song.chords)} bars; extended in "
          f"{len(song.extended_bars)} ({', '.join(extended_symbols)})")
    print()
    print("parts by role:")
    for role, count in sorted(song.notes.items()):
        peak = report["peak_concurrency_by_role"].get(role)
        print(f"   {role:12} {count:>4} notes, peak {peak} voices")
    print()
    print(f"allocation {allocation.summary()}")
    if allocation.displacements:
        print("   displacements:",
              dict(Counter(d.role.value for d in allocation.displacements)))
    else:
        print("   no notes dropped or stolen")
    print()
    print(f"stereo     L peak {audio['peak_left']}, R peak {audio['peak_right']}, "
          f"correlation {audio['stereo_correlation']} "
          f"({'genuine stereo' if audio['stereo_correlation'] < 0.98 else 'NOT STEREO'})")
    print(f"clipping   {audio['clipped_frames']} frames before normalisation")
    print(f"interp     4-point Catmull-Rom + {11000} Hz softening "
          f"(NOT the S-DSP's Gaussian table)")
    print(f"brr        {'applied' if not args.no_brr else 'SKIPPED'}; "
          f"{bank_manifest['brr_bytes']} B encoded, measured")
    print()
    print(budget.report())
    if not budget.fits:
        print("  OVER THE 64 KiB BUDGET")
    if rendered.warnings:
        print("\nwarnings:")
        for warning in rendered.warnings[:6]:
            print(f"   {warning}")
    print(f"\nwrote {wav}")
    if midi_path:
        print(f"wrote {midi_path}")
    print(f"wrote {out_json}")
    return 0


def _seconds_per_tick(result) -> float:
    composition = result.composition
    end = composition.loop_end_tick or composition.end_tick or 1
    return composition.time.seconds(end) / end


def _add(target, layer) -> None:
    """Sum a rendered layer into the main mix, padding to the longer length."""
    if layer.samples.shape[0] > target.samples.shape[0]:
        pad = np.zeros(
            (layer.samples.shape[0] - target.samples.shape[0], 2), dtype=np.float32
        )
        target.samples = np.vstack([target.samples, pad])
    end = layer.samples.shape[0]
    target.samples[:end] += layer.samples
    target.notes_rendered += layer.notes_rendered
    target.warnings.extend(layer.warnings)


if __name__ == "__main__":
    raise SystemExit(main())
