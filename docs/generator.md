# The Rule-Based Generator

Phase 3's composer. It exists for two reasons the proposal is explicit about:
audio generation has to be provable independently of model quality (§12), and
the Transformer has to beat something before it has earned its place (§11.3).

## Structure first, notes second

Next-token prediction does not produce well-formed structure (§6.5), so nothing
here generates notes until the shape of the piece is settled.

```
make_plan()        key, mood, energy, tempo, form, and a chord per bar
      ↓
generate()         melody, accompaniment, bass and drums written into that frame
      ↓
notes_to_midi()    one named track per hardware channel
      ↓
parse_midi() + build_intake_record()      the pipeline's own parser and measurements
      ↓
NesApuRenderer + validate_loop()          audio, then both loop checks
```

The round-trip through the project's own parser is deliberate: a generated
composition is measured and validated by exactly the same code as a dataset
file, so `hardware_compliant` means the same thing for both, and the generator
cannot quietly exempt itself from the rules.

## The plan

| Field | Meaning |
|---|---|
| `seed` | Determines everything. Always recorded, even when it was chosen randomly |
| `key` | Tonic plus scale; the mood picks the scale unless overridden |
| `mood` | `heroic`, `dark`, `mysterious`, `melancholic` — selects scale and progression pool |
| `energy` | `low`, `medium`, `high` — drives tempo, note density, drum pattern |
| `form` | `AABA`, `ABAC`, `AABB`, `ABAB` |
| `sections` | One per form letter, with the chord degrees under its bars |

Phrases sharing a letter share a progression. That is what makes AABA hear as a
return rather than as four unrelated ideas.

Progressions are the ones game soundtracks actually lean on — `i-VI-III-VII`
for dark, `I-V-vi-IV` for heroic — not a general survey of harmony. Chords are
built by stacking scale steps, so quality follows the scale: the triad on
degree 0 is major in a major key and minor in a minor key, with no special
casing.

## The four parts

**Melody (P1)** — one motif per phrase letter, reused with variation. A rhythm
pattern is chosen per energy level, and pitches follow a random walk over scale
degrees, weighted toward stepwise motion with the occasional leap. Independent
random degrees do not read as a melody; a walk does. Each bar re-anchors to
that bar's chord, and the final bar resolves to the tonic so the loop closes.

**Accompaniment (P2)** — a held chord tone at low energy, arpeggios in eighths
or sixteenths above that. The arpeggio direction flips each bar so the
accompaniment does not become a loop of its own.

**Bass (TR)** — chord roots, with the fifth on the fourth beat at medium energy
and octave movement at high energy. Lands on the tonic root in the final bar.

**Drums (NO)** — kick, snare and hi-hat as three noise periods (13, 7 and 2:
lower value means a shorter period and a brighter sound). Built as one slot per
grid position, filled lowest-priority first, so a kick overwrites a hat instead
of sounding on top of it — the noise channel is monophonic, and two drums at
one position is a hardware violation, not a mix.

## Two details that matter more than they look

**Velocity varies by metrical position.** Downbeats 112, beats 96, off-beats
80, subdivisions 68. The renderer maps velocity onto the APU's 4-bit volume, so
without this every note comes out at one level and the result sounds lifeless.
This is a score-level stand-in for the hardware envelopes the renderer does not
model yet.

**Notes sound for 85% of their slot.** The gap is articulation. Holding every
note full-length sounds like an organ, not a chiptune.

## Loop validation

Both halves of §7.5, because a loop can be metrically perfect and still click,
or click-free and still leave a gap.

*Musical*: nothing may extend past the loop boundary, loop markers must be
present, and the final bar must touch the tonic.

*Acoustic*: trailing silence (a gap on repeat), seam loudness continuity, and
the raw seam step.

The seam checks took two attempts, and the first was worse than useless. It
compared the seam's sample step against a fixed threshold and flagged most
pieces — but chip audio steps full-scale between adjacent samples constantly,
because that is what a square wave *is*. Measurement showed the largest
internal adjacent step (0.77–1.28 of peak) exceeded every seam step
(0.23–0.72), so no seam was ever anomalous. Worse, the piece the old check
called cleanest was the one that actually started 8.1 dB louder than it ended —
an audible pump it could not see.

So the seam step is now compared against the signal's own sharpest internal
transition, which almost never fires and should not, and the real check is
loudness continuity across the seam, in dB. The 10 dB threshold is a starting
value, not a measured one; it wants tuning once there are real loops to listen
to.

## Known limitations

- **No envelopes.** Velocity shaping helps, but sustained notes still hold a
  flat level where real NES instruments decay. This is the biggest remaining
  authenticity gap and it belongs to the renderer, not here.
- **No arpeggio or vibrato effects**, which NES composers used constantly to
  imply chords on a monophonic channel.
- **Melody is a weighted random walk.** It stays in key and follows the
  harmony, but it has no sense of motif development beyond the per-repeat
  variation — no inversion, no sequence, no answering phrase.
- **Drums are three fixed sounds** on a fixed backbeat. No fills, no variation
  between sections.
- **Four fixed moods**, each with three progressions. Recognizable repetition
  across generated pieces is expected.

These are exactly the places a trained model should be able to beat rules, and
the comparison is the point: this baseline is what Phase 4 is measured against.
