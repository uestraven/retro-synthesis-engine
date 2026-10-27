"""Rendering and level-matching for a finished composition.

Both functions lived in experiment scripts, which made the product entry point
import an experiment to produce a song: `prompt_to_song` reached into
`architecture_comparison` for `normalise_peak`, and that module imports the
Fermata codec at module scope, so the procedural path loaded Fermata's
vocabulary machinery on every run purely as a side effect of wanting a gain
calculation.

Nothing about the code changed in moving it here -- the experiment scripts
re-export these names, so the recorded listening batches stay reproducible.
"""

from __future__ import annotations

import wave
from pathlib import Path
from typing import Any

import numpy as np

from rse.events.schema import Composition

#: Peak target for every rendered file, in dBFS. Below 0 so normalisation
#: never has to clip a sample that was already near full scale.
PEAK_DBFS = -3.0


def normalise_peak(path: Path, *, dbfs: float = PEAK_DBFS) -> dict[str, Any]:
    """Rewrite a mono 16-bit WAV so its peak sits at `dbfs`.

    Equalising peak rather than loudness on purpose: a true loudness match
    would alter the perceived balance between a sparse piece and a dense one,
    which is part of what is being judged. Peak matching only removes the
    gross level cue that would identify the arm.
    """
    with wave.open(str(path), "rb") as handle:
        params = handle.getparams()
        frames = handle.readframes(handle.getnframes())
    samples = np.frombuffer(frames, dtype="<i2").astype(np.float64)
    peak = float(np.abs(samples).max()) if samples.size else 0.0
    if peak <= 0:
        return {"peak_before": 0.0, "gain": 1.0, "silent": True}
    target = (10.0 ** (dbfs / 20.0)) * 32767.0
    gain = target / peak
    scaled = np.clip(samples * gain, -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setparams(params)
        handle.writeframes(scaled.tobytes())
    return {
        "peak_before": round(peak / 32767.0, 5),
        "gain": round(gain, 5),
        "silent": False,
    }


def render_window(
    composition: Composition,
    destination: Path,
    *,
    profile_id: str = "nes",
    duty_by_channel: dict[str, int] | None = None,
) -> Path | None:
    """Render one composition through the same renderer every test uses.

    The same renderer for original and reconstruction on purpose: a codec
    difference must not be confused with a synthesis difference, which is the
    error that kept the synthesizer under suspicion for so long.

    `duty_by_channel` selects the pulse timbres. Omitted, the renderer uses its
    own defaults, which is what every procedural song used until now -- the
    composer never supplied a duty, so P1 was 50% and P2 25% in every song of
    every style ever generated.
    """
    from rse.events.convert import composition_to_intake
    from rse.exporters.wav import write_wav
    from rse.hardware.constraints import ConstraintEngine
    from rse.hardware.profile import load_profile
    from rse.renderers.base import RenderSettings, notes_from_record
    from rse.renderers.nes_apu import NesApuRenderer

    profile = load_profile(profile_id)
    # The same hardware repair the sampler applies, so the audio is what the
    # console could play rather than what the codec happens to describe.
    playable, _repair = ConstraintEngine(profile).apply(composition)
    record = composition_to_intake(playable, profile=profile)
    notes = notes_from_record(record)
    if not notes:
        return None
    renderer = NesApuRenderer(profile, duty_by_channel=duty_by_channel)
    # keep_stems=False: this path writes only the mix, and holding a decimated
    # copy of every channel forces all four oversampled channel buffers to
    # stay alive past the mix. At 4x oversampling that is 45 MB per channel
    # per minute, which is what put a 60-second render over a 512 MB instance
    # and returned a 502 instead of a song.
    result = renderer.render(notes, RenderSettings(tail_sec=0.0, keep_stems=False))
    destination.parent.mkdir(parents=True, exist_ok=True)
    return write_wav(destination, result.samples, result.sample_rate)
