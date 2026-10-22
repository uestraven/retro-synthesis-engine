# NES Renderer Survey

Why RSE synthesizes NES audio itself instead of driving an existing tool, what
the native renderer models, and what it still gets wrong. Proposal §7.1 left
three candidates open; this records what investigating them found.

## The candidates

### NES-MDB synthesis package — rejected

Proposal §7.1's first suggestion. Inspected version 0.1.8 from PyPI:

- **`nesmdb/apu.py` contains no synthesizer.** It holds register memory
  offsets, per-register bitmask definitions, the clock rates and the
  length-counter table. Useful as a reference; it produces no audio.
- **Audio goes through an external binary.** `nesmdb/vgm/vgm_to_wav.py` shells
  out to `vgm2wav`, built from a patched VGMPlay, located via `$VGMTOWAV` or
  `distutils.spawn.find_executable`. That means a C build step per platform,
  and a subprocess per render.
- **It is Python 2.** `cycle.py` and `convert.py` fail to parse on Python 3.11.
  `vgm_to_wav.py` imports `distutils`, removed from the standard library in
  3.12. (It also calls an undefined `wv` on its error path.)
- Licence: MIT, so its register tables are reusable with attribution.

Porting the Python is modest work. Shipping a per-platform C binary to a
Windows-first, offline-capable, local-only application is not.

### FamiStudio CLI — viable later, rejected for now

Can import MIDI and export WAV, and is genuinely chip-accurate. But it needs a
download plus a subprocess per render, and §7.1 already flags the real problem:
importing MIDI will not reconstruct an authentic instrument arrangement by
itself. It remains the best option for `.fur`-style project interoperability and
for NSF export, which is where a future adapter should go.

### Furnace CLI — viable later, rejected for now

Same shape of objection, plus authoring `.fur` project files is a larger job
than synthesizing four channels. It is the right target for Genesis (§7.2),
where writing an FM synthesizer from scratch would be a much worse trade.

### Native APU synthesizer — chosen

`rse/renderers/nes_apu.py`. No external binary, no subprocess, no download; runs
offline on Windows, Linux and macOS; NumPy is the only dependency; and every
component is unit-testable in isolation. For a proof of concept whose whole
purpose is to retire risk, being able to test the thing matters more than
borrowing someone else's accuracy.

## What the native renderer models

| Area | Implementation |
|---|---|
| Pulse ×2 | Four duty sequences (12.5/25/50/75%), 4-bit volume, 11-bit timer |
| Triangle | The fixed 32-step sequence, no volume control, 11-bit timer |
| Noise | 15-bit LFSR, both tap modes, NTSC period table |
| Pitch | Quantized to achievable timer values, so notes detune as on hardware |
| Sequencer phase | Advanced in CPU cycles, the clock the APU timers divide |
| Mixing | The APU's documented nonlinear pulse and triangle/noise mixing |
| Silencing | Pulse timers below 8 silence the channel, as the hardware does |
| Output stage | 90 Hz high-pass; oversampled render decimated to the output rate |
| Regions | NTSC (1.789773 MHz) and PAL (1.662607 MHz) clocks |

### Deliberate omissions

- **The 440 Hz output high-pass.** Real hardware has one. It strips most of the
  triangle's body, and emulators conventionally leave it out; so does this.
- **Hardware envelopes, sweep, length and linear counters.** Note on/off comes
  from the score and volume from velocity. This is the largest authenticity gap:
  NES music leans heavily on envelope and arpeggio effects for its character.
- **DPCM.** Out of scope for the MVP, as in the profile.
- **Cycle-exact emulation.** Phase advances in CPU cycles, but the frame
  counter, register writes and timing side effects are not simulated.

## Verification

Audio cannot be checked by ear in this environment, so correctness is
established numerically (`tests/test_nes_apu.py`, 46 tests):

- **Pitch** measured out of the rendered signal by autocorrelation, matched
  against the frequency the hardware timer can actually produce — within 1%
  across MIDI 33–108. Rendering a real composition and sampling 12 notes across
  P1, P2 and TR gave errors under 0.1%.
- **Pitch is measured by period, not by strongest FFT bin,** because at 12.5%
  duty the second harmonic exceeds the fundamental. That is correct behaviour
  for a narrow pulse, and an early version of the test wrongly flagged it.
- **Duty ratios** measured off the waveform tables: 0.125, 0.25, 0.5, 0.75.
- **LFSR** confirmed to have the full 32767-sample period and to sit gated-off
  almost exactly half the time, as a maximal-length register must.
- **Noise** confirmed broadband (no bin holds more than 1% of total energy) and
  brighter at lower period indexes.
- **Nonlinear mixing** confirmed: two pulses are louder than one but by
  distinctly less than double.
- **Aliasing** falls monotonically as oversampling rises, measured as the share
  of spectral energy that is not near a harmonic of the fundamental.
- **Determinism**: identical input gives bit-identical output.

One real bug was found this way. The duty sequencer was originally advanced at
the internal render rate, which quantized waveform transitions about ten times
more coarsely than hardware and made high notes measurably wrong — MIDI 93 and
104 could not be measured as their own pitch at all. Counting phase in CPU
cycles fixed it; those notes now land within 0.3%.

## What this does not prove

The renderer produces the right *frequencies*, with the right timing, and
plausibly NES-like timbre. Nobody has yet compared it against a real NES
recording. The honest next test is the one that needs the real dataset: render
an existing NES-MDB composition and A/B it against that song's own VGM
reference. Until then, "sounds authentic" is an expectation, not a measurement.

The missing envelopes are the likeliest source of a gap. Without them,
sustained notes hold a flat level where real NES instruments decay, which tends
to read as stiff rather than wrong.
