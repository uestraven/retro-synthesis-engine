"""The rule-based composer.

Proposal 12 Phase 3 asks for this before any model exists, so that audio
generation can be proven to work independently of model quality — and so that
Phase 4 has a baseline to beat. Proposal 11.3 makes beating it a requirement:
if the Transformer cannot outdo these rules, the machine learning is not
earning its place.

Structure comes first (see `plan.py`), then notes are written *into* that
structure. Material is generated once per phrase letter and reused with
variation, so an AABA form hears as a return rather than four unrelated ideas.

Output is produced as MIDI and then run back through the project's own parser
and normalizer, so a generated composition is validated by exactly the same
code path as a dataset file, and arrives with the same measurements.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

import mido

from rse import INTAKE_SCHEMA_VERSION
from rse.data.midi_io import parse_midi
from rse.data.normalize import build_intake_record
from rse.data.provenance import USE_CLEARED, SourceRecord
from rse.generation.plan import CompositionPlan, make_plan
from rse.generation.theory import Key, chord_pitches, transpose_into_range
from rse.hardware.profile import HardwareProfile

#: Sixteenth notes per bar in 4/4.
STEPS_PER_BAR = 16

#: Playable register per channel, inside the profile's wider hardware limits.
#: Melody sits above accompaniment, which sits above bass, so the parts stay
#: distinguishable rather than crowding one octave.
REGISTERS: dict[str, tuple[int, int]] = {
    "P1": (60, 88),
    "P2": (55, 79),
    "TR": (33, 55),
}

#: Noise period values standing in for drums. Lower value is a shorter period
#: and so a brighter sound: 2 reads as a hat, 7 as a snare, 13 as a kick.
DRUM_HAT = 2
DRUM_SNARE = 7
DRUM_KICK = 13

#: Fraction of its slot a note actually sounds. The gap is what gives chiptune
#: its articulation; holding every note full-length sounds like an organ.
GATE = 0.85

#: Velocity by metrical position, which the renderer maps onto 4-bit volume.
#: Without this, every note renders at one level and the result sounds flat.
ACCENT_DOWNBEAT = 112
ACCENT_BEAT = 96
ACCENT_OFFBEAT = 80
ACCENT_SUBDIVISION = 68

#: Melodic rhythm patterns as `(offset, length)` in sixteenths, by energy.
RHYTHMS: dict[str, tuple[tuple[tuple[int, int], ...], ...]] = {
    "low": (
        ((0, 8), (8, 8)),
        ((0, 4), (4, 4), (8, 8)),
        ((0, 6), (6, 2), (8, 8)),
    ),
    "medium": (
        ((0, 4), (4, 2), (6, 2), (8, 4), (12, 4)),
        ((0, 2), (2, 2), (4, 4), (8, 4), (12, 4)),
        ((0, 4), (4, 4), (8, 2), (10, 2), (12, 4)),
    ),
    "high": (
        ((0, 2), (2, 2), (4, 2), (6, 2), (8, 2), (10, 2), (12, 4)),
        ((0, 2), (2, 1), (3, 1), (4, 2), (6, 2), (8, 4), (12, 2), (14, 2)),
        ((0, 1), (1, 1), (2, 2), (4, 2), (6, 2), (8, 2), (10, 2), (12, 4)),
    ),
}

#: Bass note length in sixteenths, by energy.
BASS_STEP: dict[str, int] = {"low": 8, "medium": 4, "high": 2}


@dataclass(frozen=True)
class GeneratedNote:
    """One note placed on the tick grid."""

    channel_id: str
    start_tick: int
    dur_ticks: int
    value: int
    velocity: int


@dataclass
class GeneratedComposition:
    """A finished composition in every representation the pipeline needs."""

    plan: CompositionPlan
    notes: list[GeneratedNote]
    midi: mido.MidiFile
    record: dict[str, Any]
    warnings: list[str] = field(default_factory=list)

    @property
    def composition_id(self) -> str:
        return self.record["composition_id"]


def generate(
    profile: HardwareProfile,
    plan: CompositionPlan | None = None,
    **plan_kwargs: Any,
) -> GeneratedComposition:
    """Compose a piece for `profile`, planning one first if none is given."""
    if plan is None:
        plan = make_plan(**plan_kwargs)
    elif plan_kwargs:
        raise TypeError("pass either a plan or plan keyword arguments, not both")
    if profile.profile_id != "nes":
        raise ValueError(
            f"the procedural generator writes for the 'nes' profile, not {profile.profile_id!r}"
        )

    rng = random.Random(plan.seed)
    notes: list[GeneratedNote] = []
    notes += _melody(plan, rng)
    notes += _accompaniment(plan, rng)
    notes += _bass(plan, rng)
    notes += _percussion(plan, rng)

    # Nothing may cross the loop boundary, or the seam clicks and the last bar
    # bleeds over the first on repeat.
    limit = plan.total_ticks
    notes = [
        GeneratedNote(
            n.channel_id,
            n.start_tick,
            min(n.dur_ticks, limit - n.start_tick),
            n.value,
            n.velocity,
        )
        for n in notes
        if n.start_tick < limit
    ]
    notes = [n for n in notes if n.dur_ticks > 0]
    notes.sort(key=lambda n: (n.start_tick, n.channel_id))

    midi = notes_to_midi(notes, plan, profile)
    record = _record_from_midi(midi, plan, profile)
    return GeneratedComposition(plan=plan, notes=notes, midi=midi, record=record)


# -- the four parts -----------------------------------------------------------


def _melody(plan: CompositionPlan, rng: random.Random) -> list[GeneratedNote]:
    """Melody on P1: one motif per phrase letter, varied on repeats."""
    motifs: dict[str, tuple[tuple[int, int], ...]] = {}
    contours: dict[str, tuple[int, ...]] = {}
    for section in plan.sections:
        if section.label in motifs:
            continue
        rhythm = rng.choice(RHYTHMS[plan.energy])
        motifs[section.label] = rhythm
        contours[section.label] = _contour(len(rhythm), rng)

    out: list[GeneratedNote] = []
    low, high = REGISTERS["P1"]
    for bar in range(plan.n_bars):
        section = plan.section_at(bar)
        rhythm = motifs[section.label]
        contour = contours[section.label]
        chord = plan.chord_at_bar(bar)
        is_final_bar = bar == plan.n_bars - 1
        # Repeats of a phrase get a small variation so the form reads as
        # development rather than as a copy-paste.
        repeat_index = (bar - section.start_bar) % max(1, len(section.chord_degrees))
        vary = repeat_index % 2 == 1

        for position, (offset, length) in enumerate(rhythm):
            degree = chord + contour[position]
            if vary:
                degree += rng.choice((0, 0, 1, -1))
            if is_final_bar and position == len(rhythm) - 1:
                degree = 0  # resolve to the tonic so the loop closes
            pitch = transpose_into_range(plan.key.degree_to_pitch(degree, octave=5), low, high)
            out.append(
                _note(
                    "P1",
                    plan,
                    bar,
                    offset,
                    length,
                    pitch,
                    _velocity(offset, emphasis=6 if position == 0 else 0),
                )
            )
    return out


def _accompaniment(plan: CompositionPlan, rng: random.Random) -> list[GeneratedNote]:
    """P2: sustained harmony at low energy, arpeggios above it."""
    out: list[GeneratedNote] = []
    low, high = REGISTERS["P2"]
    step = {"low": 8, "medium": 4, "high": 2}[plan.energy]

    for bar in range(plan.n_bars):
        chord = plan.chord_at_bar(bar)
        pitches = [
            transpose_into_range(p, low, high) for p in chord_pitches(plan.key, chord, octave=4)
        ]
        if plan.energy == "low":
            # Hold the third of the chord under the melody.
            out.append(_note("P2", plan, bar, 0, STEPS_PER_BAR, pitches[1], ACCENT_OFFBEAT))
            continue
        # Walk up and down the chord; the direction flips each bar so the
        # accompaniment does not turn into a loop of its own.
        order = pitches if bar % 2 == 0 else list(reversed(pitches))
        for index, offset in enumerate(range(0, STEPS_PER_BAR, step)):
            pitch = order[index % len(order)]
            if rng.random() < 0.12:  # occasional octave lift for interest
                pitch = transpose_into_range(pitch + 12, low, high)
            out.append(_note("P2", plan, bar, offset, step, pitch, _velocity(offset)))
    return out


def _bass(plan: CompositionPlan, rng: random.Random) -> list[GeneratedNote]:
    """TR: chord roots, with octave movement as energy rises."""
    out: list[GeneratedNote] = []
    low, high = REGISTERS["TR"]
    step = BASS_STEP[plan.energy]

    for bar in range(plan.n_bars):
        chord = plan.chord_at_bar(bar)
        root = transpose_into_range(plan.key.degree_to_pitch(chord, octave=2), low, high)
        fifth = transpose_into_range(plan.key.degree_to_pitch(chord + 4, octave=2), low, high)
        is_final_bar = bar == plan.n_bars - 1

        for index, offset in enumerate(range(0, STEPS_PER_BAR, step)):
            pitch = root
            if plan.energy == "high" and index % 2 == 1:
                pitch = transpose_into_range(root + 12, low, high)
            elif plan.energy == "medium" and index == 3:
                pitch = fifth
            if is_final_bar and offset >= STEPS_PER_BAR - step:
                pitch = transpose_into_range(plan.key.degree_to_pitch(0, octave=2), low, high)
            out.append(_note("TR", plan, bar, offset, step, pitch, _velocity(offset)))
    return out


def _percussion(plan: CompositionPlan, rng: random.Random) -> list[GeneratedNote]:
    """NO: a backbeat on the noise channel, busier as energy rises."""
    out: list[GeneratedNote] = []
    # Hats land between the beats: step 2 gives eighths, step 1 sixteenths.
    # A step of 4 would put every hat on a beat, where the declutter rule below
    # drops it, leaving no hats at all.
    hat_step = {"low": 0, "medium": 2, "high": 1}[plan.energy]

    for bar in range(plan.n_bars):
        # One slot per grid position, filled lowest priority first so a kick
        # overwrites a hat rather than sounding on top of it. The noise channel
        # is monophonic, so two drums at one position is not a mix — it is a
        # hardware violation.
        slots: dict[int, tuple[int, int, int]] = {}
        if hat_step:
            for offset in range(0, STEPS_PER_BAR, hat_step):
                if offset % 4 == 0:
                    continue  # leave the kick and snare positions uncluttered
                slots[offset] = (DRUM_HAT, 1, ACCENT_SUBDIVISION)
        if plan.energy == "high" and rng.random() < 0.4:
            slots[14] = (DRUM_SNARE, 2, ACCENT_OFFBEAT)  # a pickup into the next bar
        for offset in (4, 12):
            slots[offset] = (DRUM_SNARE, 2, ACCENT_BEAT)
        for offset in (0, 8):
            slots[offset] = (DRUM_KICK, 2, ACCENT_DOWNBEAT)

        offsets = sorted(slots)
        for index, offset in enumerate(offsets):
            value, length, velocity = slots[offset]
            next_offset = offsets[index + 1] if index + 1 < len(offsets) else STEPS_PER_BAR
            # Truncate to the gap before the next hit, so nothing can overlap.
            out.append(
                _note("NO", plan, bar, offset, min(length, next_offset - offset), value, velocity)
            )
    return out


# -- helpers ------------------------------------------------------------------


def _contour(length: int, rng: random.Random) -> tuple[int, ...]:
    """A melodic shape in scale steps, relative to the chord root.

    Stepwise motion with the occasional leap: a random walk over a scale reads
    as a melody, whereas independent random degrees do not.
    """
    degrees = [0]
    for _ in range(length - 1):
        move = rng.choices((-2, -1, 1, 2, 4), weights=(2, 4, 4, 2, 1))[0]
        nxt = degrees[-1] + move
        # Keep the line from wandering more than an octave from the chord root.
        if abs(nxt) > 7:
            nxt = degrees[-1] - move
        degrees.append(nxt)
    return tuple(degrees)


def _velocity(offset_steps: int, *, emphasis: int = 0) -> int:
    """Velocity from metrical position: downbeats loudest, subdivisions least."""
    if offset_steps == 0:
        base = ACCENT_DOWNBEAT
    elif offset_steps % 4 == 0:
        base = ACCENT_BEAT
    elif offset_steps % 2 == 0:
        base = ACCENT_OFFBEAT
    else:
        base = ACCENT_SUBDIVISION
    return max(1, min(127, base + emphasis))


def _note(
    channel_id: str,
    plan: CompositionPlan,
    bar: int,
    offset_steps: int,
    length_steps: int,
    value: int,
    velocity: int,
) -> GeneratedNote:
    step_ticks = plan.ticks_per_bar // STEPS_PER_BAR
    start = bar * plan.ticks_per_bar + offset_steps * step_ticks
    full = length_steps * step_ticks
    # Shorten to leave an articulation gap, but never below a 32nd note.
    duration = max(int(full * GATE), min(full, step_ticks // 2) or 1)
    return GeneratedNote(channel_id, start, duration, value, velocity)


def notes_to_midi(
    notes: list[GeneratedNote], plan: CompositionPlan, profile: HardwareProfile
) -> mido.MidiFile:
    """Write generated notes to MIDI, one named track per hardware channel."""
    midi = mido.MidiFile(type=1, ticks_per_beat=plan.ticks_per_beat)

    meta = mido.MidiTrack()
    meta.append(mido.MetaMessage("track_name", name="meta", time=0))
    meta.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(plan.tempo_bpm), time=0))
    meta.append(
        mido.MetaMessage("time_signature", numerator=plan.beats_per_bar, denominator=4, time=0)
    )
    if plan.loop:
        meta.append(mido.MetaMessage("marker", text="loop_start", time=0))
        meta.append(mido.MetaMessage("marker", text="loop_end", time=plan.total_ticks))
    midi.tracks.append(meta)

    for index, channel_id in enumerate(profile.channel_ids):
        track = mido.MidiTrack()
        track.append(mido.MetaMessage("track_name", name=channel_id, time=0))
        events: list[tuple[int, int, int, int]] = []
        for note in notes:
            if note.channel_id != channel_id:
                continue
            events.append((note.start_tick, 1, note.value, note.velocity))
            events.append((note.start_tick + note.dur_ticks, 0, note.value, 0))
        # Note-offs sort before note-ons at the same tick so a channel never
        # looks polyphonic to the validator.
        events.sort(key=lambda e: (e[0], e[1]))
        cursor = 0
        for tick, is_on, value, velocity in events:
            track.append(
                mido.Message(
                    "note_on" if is_on else "note_off",
                    channel=index,
                    note=value,
                    velocity=velocity,
                    time=tick - cursor,
                )
            )
            cursor = tick
        midi.tracks.append(track)
    return midi


def _record_from_midi(
    midi: mido.MidiFile, plan: CompositionPlan, profile: HardwareProfile
) -> dict[str, Any]:
    """Normalize generated MIDI through the pipeline's own parser.

    Round-tripping rather than hand-building a record means generated music is
    measured and validated by the same code as dataset music, so the hardware
    compliance figure means the same thing for both.
    """
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "generated.mid"
        midi.save(str(path))
        composition = parse_midi(path, profile)
        source = SourceRecord(
            source_id=f"generated:{plan.seed:010d}",
            dataset_id="generated",
            dataset_version=f"procedural/{INTAKE_SCHEMA_VERSION}",
            relpath=f"generated_{plan.seed}.mid",
            sha256="",
            size_bytes=path.stat().st_size,
            use_class=USE_CLEARED,
            license_note="Generated by RSE's procedural composer; no third-party material.",
            acquisition_method="procedural",
        )
        record = build_intake_record(composition, source, profile)

    record["generation"] = {
        "generator": "procedural/0.1.0",
        "plan": plan.to_dict(),
    }
    return record


def key_from_dict(data: dict[str, Any]) -> Key:
    """Rebuild a key from a serialized plan."""
    return Key(tonic=data["tonic"], scale=data["scale"])
