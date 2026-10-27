# First SNES-style boss theme — **ACCEPTED**

Rated 2026-10-27. Verdict: *"sounds much better. perfect"*, after four
revisions.

SNES-INSPIRED PREVIEW, not hardware: PCM with linear interpolation rather than
BRR with the S-DSP's Gaussian filter, floating-point ADSR rather than the
hardware's rate tables, no ARAM enforcement, no SPC export, no emulator
verification.

| | |
|---|---|
| music | 150.25 BPM, harmonic_minor, A-A'1-A'2-B-A'', 40 bars, 64.545 s |
| voices | peak 8 of 8, 0 dropped, 0 stolen |
| extended chords | 6 bars (Am9, Fmaj9, E9), 3 voices each, fifth omitted |
| stereo | correlation 0.9085, no clipping, no DC |
| sample memory | 17145 B BRR estimate = 26.2% of 64 KiB |
| seed | 4242, deterministic across hash seeds |

## The four revisions, and what each was

Worth keeping because the artifact took three wrong diagnoses before the right
one, and the pattern is instructive.

**r1 → r2: a loop buzz.** Sustained samples detuned their partials by up to
eleven cents, which makes a periodic loop impossible by construction. The
waveform jumped 4×–12× a normal sample step at every wrap, repeating at
**32.7 Hz** — a 32.7 Hz tone under every held note. The crossfade meant to
prevent it had been applied to the buffer's end rather than the loop boundary.
Also found: DC offset of +0.012, no output high-pass where the NES renderer
has one, and tiled noise that became a pitched rasp.

**r2 → r3: a bass approach note with no target.**
`bass + (2 if degree % 2 == 0 else -2)` — a whole tone chosen by whether the
degree *index* was even. It produced A2 → A3 → B2 → A2. Now steps into the next
bar's root, 39/39 bars.

**r3 → r4: the actual cause.** The listener analysed the MIDI and found it in
the P1 track. `vary_motif`'s ornament branch halved a note without checking it
was long enough, so a two-row note became two **one-row** notes and the second
landed on an odd row — the only off-eighth attack in the piece, once per bar.
An **NES V1 defect**, present in **54 of 240 songs** (action 23/40, boss
19/40), 536 notes. Fixed; 0 of 240 after.

## The lesson

Three diagnoses, two of them wrong, and both wrong ones were *real defects I
found by looking at the renderer*. The artifact was in the score the whole
time. **Trace a rendered-mix artifact to the score before the synthesiser** —
the MIDI is cheaper to read than the audio and it is unambiguous.
