# RSE Event Format (RSE-EF) and the Constraint Engine

Phase 2. `rse-ef/1.0.0`.

Two things live here: the canonical representation the whole system agrees on,
and the engine that decides what a given console can actually play.

## Why there are two formats

| | Intake record (`rse-intake/0.1.0`) | RSE-EF (`rse-ef/1.0.0`) |
|---|---|---|
| Purpose | Lossless landing zone for source data | What the system computes on |
| Shape | Mirrors MIDI closely, with derived fields alongside | Typed events, no derived fields |
| Produced by | `rse ingest` | `composition_from_intake` |
| Consumed by | statistics, de-duplication | tokenizer, constraint engine, renderers, exporters |

They are versioned separately on purpose. The intake record must never be
tidied up in a way that loses information, and RSE-EF must be free to change
shape as the tokenizer and constraint engine need it to.

## The format

```python
Composition(
    hardware_profile="nes",
    ticks_per_beat=480,
    events=[...],  # Note | Noise | Controller | Program | EffectEvent
    tempo_map=[TempoChange(tick, us_per_beat)],
    time_signatures=[TimeSignatureChange(tick, numerator, denominator)],
    loop_start_tick=0,
    loop_end_tick=30720,
    source_metadata={...},  # provenance, carried through unchanged
)
```

| Event | Carries |
|---|---|
| `Note` | `pitch`, `duration`, `velocity`, `instrument_id?`, `effects?` |
| `Noise` | `value`, `duration`, `velocity`, `instrument_id?`, `effects?` |
| `Controller` | `controller`, `value` |
| `Program` | `program` |
| `EffectEvent` | `effect`, `params` — duty, slide, vibrato, pitch bend, sweep |

### Two invariants

**A noise event has a `value`, never a `pitch`**, and the inapplicable key is
*absent* rather than null. Any code that transposes or harmonizes reads
`pitch`, so it cannot touch the noise channel by accident — the field simply
isn't there.

**Ticks are canonical.** `beat`, `bar` and `sec` are computed from the tempo
map through `composition.time`, never stored. The intake record stores them
alongside ticks for convenience; RSE-EF does not store them at all, so they
cannot drift out of agreement with the tick they describe.

Register-level effects stay effects. A MIDI pitch bend becomes
`EffectEvent(effect="pitch_bend")`, not a controller — it is a hardware
register change, and flattening it into continuous-controller data would lose
that distinction.

## Round-trip fidelity

MVP criterion 2 asks that the format round-trip compositions "within defined
timing and controller tolerances". The tolerance is a value you can print:

```python
Tolerance(timing_ticks=0, duration_ticks=0, velocity=0, controller_value=0)
```

Defaults are **exact**. Every path inside RSE is lossless for notes and
controllers, so a regression should fail the test rather than be absorbed by a
generous threshold. The knobs exist so that a lossy external format can be
measured against a stated budget instead of an opinion.

`compare_compositions` distinguishes three outcomes, which is the useful part:

- **missing / added** — the event is gone, or appeared from nowhere. Identity
  is `(type, channel, pitch-or-value-or-controller)`; a changed pitch is a
  *different note*, not a drifted one.
- **drifted** — matched, but a field moved: tick, duration, velocity, value.
- **notes** — something global changed: resolution, tempo map, loop boundaries.

Measured result: dataset files and generated compositions both round-trip
through MIDI with every event matched and zero drift, at exact tolerance.

```
$ rse validate --file song.mid --round-trip
352 events, profile nes
  hardware: playable as written
  loop: clean
  round trip: lossless (352/352 events matched)
```

## The constraint engine

Proposal 3.2 is explicit that overlapping notes must not simply be discarded
after generation, so the engine does two separate jobs and keeps them apart.

**Prevention** — `ChannelAllocator` tracks per-channel occupancy with half-open
intervals, so a note ending exactly where the next begins is legal monophony
rather than a collision. A generator can ask `can_place` before committing, or
`room_until` to find out how long a note could last.

**Repair** — `ConstraintEngine.apply` returns a playable copy plus a report of
everything it changed. `validate` is read-only and provably changes nothing.

### Resolution order

Keep the event if it can be kept; reject only what cannot be represented.

| Problem | Resolution | Recorded as |
|---|---|---|
| Pitch outside the channel's range | Transpose by octaves into range | `transposed` |
| Unpitched value out of range | Clamp to the nearest valid value | `clamped` |
| Note crosses the loop end | Truncate to the boundary | `shortened` |
| Note starts at or after the loop end | Drop it | `rejected` |
| A note is still sounding when the next begins | Truncate the **sounding** note | `shortened` |
| Two notes start at the same instant | Keep the louder, then the higher | `rejected` |
| Duration below the minimum | Drop it | `rejected` |
| Channel not in the profile | Drop it | `rejected` |

An octave transposition preserves a note's musical identity, which is why it is
tried before anything destructive. An unpitched value has no octave to move
by — it selects a timbre — so the nearest valid value is the honest substitute.

**Voice stealing runs in the hardware's direction.** A new note retriggers the
channel, so the note already sounding gets truncated to the newcomer's onset.
An earlier version of this code tried to shrink the *incoming* note to fit
before the existing one, and rejected it when it couldn't — backwards, and it
silently dropped music the console could have played. The test that caught it
is `test_overlapping_notes_are_shortened_so_both_survive`.

Repair is **idempotent**, and its output always validates clean — both are
tested, including against deliberately multi-violation input.

```
$ rse validate --record broken.json --fix repaired.json
  rejected   FM1  @0     no channel 'FM1' in this profile
  clamped    NO   @0     value 99 -> 16 to fit 1-16
  transposed P1   @0     pitch 12 -> 36 to fit 33-108
  shortened  P1   @0     duration 1200 -> 960 to end at the loop boundary
  shortened  P1   @0     duration 960 -> 240; the note at 240 retriggers the channel
  rejected   P1   @240   another note already starts here
```

## What this does not do yet

- **Nothing calls the engine automatically.** `rse generate` produces compliant
  output on its own — verified by asserting the engine finds nothing to fix —
  and `rse validate` is opt-in. Phase 4 is where model output, which has no
  such guarantee, must be run through it before rendering.
- **Multi-voice channels are untested in anger.** The code handles
  `max_simultaneous_notes > 1`, but every NES channel is monophonic, so that
  path has only synthetic coverage. Genesis FM channels will exercise it.
- **`channel_priority` is defined but unused.** Same-instant conflicts are
  resolved by velocity and then pitch, which is deterministic but ignores the
  role ranking proposal 3.2 describes. It needs cross-channel reallocation to
  be meaningful, which is a bigger change than it looks.
- **Effect events are carried, not validated.** The profile lists which effects
  each channel supports; nothing yet checks an `EffectEvent` against that list.
