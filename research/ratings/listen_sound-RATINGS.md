# Dry vs Expressive NES — 12 pairs, identical scores. **ACCEPTED.**

Rated 2026-10-26. Verdict: *"songs feel more diverse now, confirmed."*

The first non-null result after four consecutive nulls on composition. Every
pair was the same score twice — pitch, onset, duration, channel, progression
and structure byte-identical — with only the pulse duty cycles and the note
velocities differing. So the gain is attributable to performance and
instrumentation alone, not to any change in the notes.

`sound_design="expressive"` promoted to the default.

## What this settles

Composition, instrumentation and expression are separable, and RSE had been
improving only the first. Four experiments that rearranged material returned
no preference; the first one that changed *how the material is played* landed
immediately.

| experiment | what it changed | outcome |
|---|---|---|
| harmonic rhythm | composition | no preference (24 pairs) |
| harmonic vocabulary | composition | no preference (24 pairs) |
| key / tempo / form / arrangement | composition | no difference |
| phrase structure | composition | undecided, dropped |
| **duty + velocity** | **performance** | **accepted** |

## What it does NOT settle

The per-pair table was not filled in, so two things stay open:

- **Expression versus instrumentation was not separated.** Pairs 06, 09 and 11
  drew `hollow/classic`, which is the dry default, so those three tested
  velocity shaping *alone*. Whether they were audible is unrecorded. If
  envelopes are built next, that question is worth settling first: it decides
  whether effort belongs in dynamics or in timbre.
- **Style differentiation** — whether two different styles are now more
  distinguishable from each other — was the question the variety experiment
  failed to move, and it is unanswered here too.

## The hardware facts worth not re-deriving

**Three pulse timbres, not four.** `DUTY_SEQUENCES[3]` is exactly
`1 - DUTY_SEQUENCES[1]`: the 25% waveform inverted, identical to 0.0000 across
ten harmonics. 50% has no even harmonics (hollow), 25% has 0.70, 12.5% has
0.91 (bright, nasal).

**Volume is a 4-bit register.** Velocities 98–104 all land on register 12, so a
five-unit accent is silent. Dynamics must be authored in register space.

**The triangle has no volume register.** Bass dynamics do not exist on NES.

**Measure duty on the isolated channel stem, not the mix.** A whole-mix
spectral average called 12.5% and 25% indistinguishable, and called a real
duty change inaudible. Harmonic analysis of the P1 stem shows duty changes
moving its high/low energy ratio by 12–32%.
