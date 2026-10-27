"""The SNES generation path, shared by the API and the command line.

The same discipline the NES path already follows: one implementation, two
front ends. `scripts/snes_boss.py` and the website's `console: "snes"` branch
both call `generate_snes`, so neither can drift from the other about what a
SNES song is.

It reuses the NES composer unchanged. The composition is platform-neutral --
motifs, harmony, form, phrasing -- and only the arrangement and the rendering
are per-platform. So this module is thin by design: compose, arrange for eight
voices, allocate, render in stereo, and report what it cost.

**SNES-INSPIRED PREVIEW.** Samples are really BRR-encoded and the ARAM figures
are measured, but interpolation is a 4-point Catmull-Rom kernel plus a
documented low-pass rather than the S-DSP's own table, envelopes are floating
point rather than the hardware's rate tables, and there is no SPC export or
emulator verification. Every report this module produces says so.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from rse.exporters.wav import write_wav
from rse.platform.roles import Role, concurrency
from rse.procedural.compose import compose
from rse.snes.aram import budget_for, largest_echo_that_fits
from rse.snes.arrange import VOICE_PLAN, allocate, arrange, percussion_settings
from rse.snes.brr import encode as brr_encode
from rse.snes.sampler import render
from rse.snes.samples import build_bank, manifest

#: Echo presets, in milliseconds. Each costs ARAM, so the names are a budget
#: decision as much as a musical one: `cavern` spends 30 KiB of the 64 on
#: reverb, which is memory the instruments do not get.
ECHO_PRESETS: dict[str, int] = {
    "dry": 0,
    "room": 64,
    "hall": 128,
    "cavern": 240,
}

#: Styles that take a dry or near-dry setting by default, because reverb on
#: fast material smears the attacks that make it read as fast.
DEFAULT_ECHO: dict[str, str] = {
    "boss": "room",
    "action": "dry",
    "adventure": "hall",
    "town": "room",
    "tension": "cavern",
    "victory": "hall",
}


@dataclass
class SnesSong:
    """A rendered SNES song and the full account of what it cost."""

    samples: np.ndarray
    sample_rate: int
    result: Any
    arrangement: Any
    allocation: Any
    budget: Any
    bank_manifest: dict[str, Any]
    encoded: dict[str, Any]
    audio: dict[str, Any]
    echo_ms: int
    seed: int
    style: str
    warnings: list[str] = field(default_factory=list)

    @property
    def seconds(self) -> float:
        return self.samples.shape[0] / self.sample_rate

    def report(self) -> dict[str, Any]:
        music = {
            "tempo_bpm": round(self.result.style.tempo_bpm, 2),
            "mode": self.result.style.mode,
            "tonic": self.result.style.tonic,
            "form": [s.label for s in self.result.sections],
            "bars": sum(s.bars for s in self.result.sections),
            "seconds": round(self.seconds, 3),
        }
        extended = sorted(
            {
                v.intent.symbol
                for v in self.arrangement.voicings
                if v.intent.seventh.value != "none"
            }
        )
        return {
            "renderer": "snes_sampler",
            "mode": "SNES-INSPIRED PREVIEW",
            "hardware_exact": False,
            "caveats": [
                "Samples ARE BRR-encoded and the ARAM figures are measured.",
                "Interpolation is a 4-point Catmull-Rom kernel plus a "
                "documented low-pass, NOT the S-DSP's own 4-tap Gaussian "
                "table.",
                "Envelopes are floating point, not the DSP's rate tables.",
                "Driver and runtime memory are assumptions: RSE has no "
                "SPC700 driver.",
                "No SPC export and no emulator verification. A correct byte "
                "count is not an SPC file.",
            ],
            "console": "snes",
            "style": self.style,
            "seed": self.seed,
            "music": music,
            "extended_chords": extended,
            "arrangement": self.arrangement.to_dict(),
            "allocation": self.allocation.to_dict(),
            "peak_concurrency_by_role": {
                role.value: concurrency(
                    [p for p in self.arrangement.parts if p.role is role]
                )
                for role in Role
                if any(p.role is role for p in self.arrangement.parts)
            },
            "voice_plan": {
                role.value: {
                    "instrument": settings.instrument,
                    "pan": settings.pan,
                    "gain": settings.gain,
                    "echo_send": settings.echo,
                }
                for role, settings in VOICE_PLAN.items()
            },
            "echo": {
                "preset_ms": self.echo_ms,
                "aram_bytes": self.budget.echo_bytes,
                "longest_that_fits_ms": largest_echo_that_fits(self.budget),
            },
            "aram": self.budget.to_dict(),
            "brr": {n: e.to_dict() for n, e in self.encoded.items()},
            "instrument_bank": self.bank_manifest,
            "audio": self.audio,
            "warnings": list(self.warnings),
        }


def _seconds_per_tick(result: Any) -> float:
    composition = result.composition
    end = composition.loop_end_tick or composition.end_tick or 1
    return composition.time.seconds(end) / end


def generate_snes(
    *,
    style: str = "boss",
    seed: int,
    target_seconds: float = 60.0,
    echo: str | int | None = None,
    reserved_voices: int = 0,
    brr: bool = True,
) -> SnesSong:
    """Compose, arrange for eight voices, and render in stereo.

    `echo` takes a preset name, an explicit millisecond value, or None for the
    style's default. It is charged against ARAM, so a long reverb is paid for
    in instrument memory.
    """
    result = compose(style_name=style, seed=seed, target_seconds=target_seconds)
    song = arrange(result)
    allocation = allocate(song, reserved=reserved_voices)

    if echo is None:
        echo_ms = ECHO_PRESETS[DEFAULT_ECHO.get(style, "room")]
    elif isinstance(echo, str):
        if echo not in ECHO_PRESETS:
            raise ValueError(
                f"unknown echo preset {echo!r}; have {sorted(ECHO_PRESETS)}"
            )
        echo_ms = ECHO_PRESETS[echo]
    else:
        echo_ms = max(0, int(echo))

    bank = build_bank(brr=brr)
    encoded = {
        name: brr_encode(
            asset.data,
            loop_start=(
                asset.loop_start if asset.loop_end > asset.loop_start else None
            ),
        )
        for name, asset in bank.items()
    }

    seconds_per_tick = _seconds_per_tick(result)
    rendered = render(
        allocation.assignments,
        bank=bank,
        voices=dict(VOICE_PLAN),
        seconds_per_tick=seconds_per_tick,
        allocation=allocation,
    )
    # Percussion is three instruments on one role, so a single VoiceSettings
    # cannot be a kick and a hat at once: render each drum as its own layer
    # and sum.
    drums = [a for a in allocation.assignments if a.note.role is Role.PERCUSSION]
    for value in sorted({a.note.value for a in drums}):
        hits = [a for a in drums if a.note.value == value]
        layer = render(
            hits,
            bank=bank,
            voices={Role.PERCUSSION: percussion_settings(value)},
            seconds_per_tick=seconds_per_tick,
            peak=None,
        )
        _add(rendered, layer)

    # One scalar across both channels, so the panning survives.
    loudest = float(np.abs(rendered.samples).max())
    if loudest > 0:
        rendered.samples = (rendered.samples * (0.89 / loudest)).astype(np.float32)

    budget = budget_for(
        encoded,
        sequence_bytes=len(allocation.assignments) * 4,
        echo_delay_ms=echo_ms,
    )
    return SnesSong(
        samples=rendered.samples,
        sample_rate=rendered.sample_rate,
        result=result,
        arrangement=song,
        allocation=allocation,
        budget=budget,
        bank_manifest=manifest(bank, encoded=encoded),
        encoded=encoded,
        audio=rendered.to_dict(),
        echo_ms=echo_ms,
        seed=seed,
        style=style,
        warnings=list(rendered.warnings),
    )


def _add(target: Any, layer: Any) -> None:
    """Sum a rendered layer into the mix, padding to the longer length."""
    if layer.samples.shape[0] > target.samples.shape[0]:
        pad = np.zeros(
            (layer.samples.shape[0] - target.samples.shape[0], 2), dtype=np.float32
        )
        target.samples = np.vstack([target.samples, pad])
    end = layer.samples.shape[0]
    target.samples[:end] += layer.samples
    target.notes_rendered += layer.notes_rendered
    target.warnings.extend(layer.warnings)


def write_snes_artifacts(
    song: SnesSong, *, out_dir: Path, name: str
) -> dict[str, Any]:
    """Write the stereo WAV, the score, and the report."""
    import json

    out_dir.mkdir(parents=True, exist_ok=True)
    wav = out_dir / f"{name}.wav"
    write_wav(wav, song.samples, song.sample_rate)

    score: Path | None = out_dir / f"{name}.mid"
    error: str | None = None
    try:
        from rse.events.convert import composition_to_intake
        from rse.exporters.midi import record_to_midi
        from rse.hardware.profile import load_profile

        record_to_midi(
            composition_to_intake(
                song.result.composition, profile=load_profile("nes")
            )
        ).save(str(score))
    except Exception as failure:  # noqa: BLE001 - reported, not fatal
        error = f"{type(failure).__name__}: {failure}"
        score = None

    payload = song.report()
    payload["files"] = {
        "wav": str(wav),
        "score": str(score) if score else None,
    }
    if error:
        payload["score_error"] = error
    (out_dir / f"{name}.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def summary(song: SnesSong) -> str:
    """One block of text, for a terminal."""
    audio = song.audio
    extended = sorted(
        {
            v.intent.symbol
            for v in song.arrangement.voicings
            if v.intent.seventh.value != "none"
        }
    )
    lines = [
        "SNES-INSPIRED PREVIEW  (not hardware-exact)",
        "",
        f"style      {song.style}, seed {song.seed}",
        f"music      {song.result.style.tempo_bpm:.2f} BPM, "
        f"{song.result.style.mode}, "
        f"{'-'.join(s.label for s in song.result.sections)}, "
        f"{sum(s.bars for s in song.result.sections)} bars, "
        f"{song.seconds:.1f} s",
        f"chords     extended in {len(song.arrangement.extended_bars)} bars "
        f"({', '.join(extended)})",
        "",
        f"allocation {song.allocation.summary()}",
        f"stereo     correlation {audio['stereo_correlation']}, "
        f"clipping {audio['clipped_frames']} frames",
        f"echo       {song.echo_ms} ms = {song.budget.echo_bytes} B of ARAM",
        "",
        song.budget.report(),
    ]
    if not song.budget.fits:
        lines.append("  OVER THE 64 KiB BUDGET")
    return "\n".join(lines)


def aram_fraction(song: SnesSong) -> float:
    return song.budget.total / 65536.0


def peak_voices(song: SnesSong) -> int:
    return int(song.allocation.peak_voices)
