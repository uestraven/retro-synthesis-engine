"""Arranging a shared composition for eight sampled voices.

Not a waveform swap. The instruction was explicit that replacing P1/P2/TR/NO's
waveforms is the wrong thing, and the reason is musical: the NES composer
writes one monophonic harmony voice because that is all a second pulse channel
can be. Given eight voices, the same harmonic plan should produce a sustained
three-note string chord, a brass lead over it, an active bass and sampled
percussion -- parts that cannot exist on the NES at all.

So this arranger takes the composition's harmonic and melodic PLAN rather than
its NES events, and re-realises it:

    melody      -> brass lead, one voice
    harmony     -> string chord, up to three voices, via the voicing planner
    pad         -> choir, on the contrast section only
    bass        -> electric bass, with scalar approach notes
    percussion  -> kick, snare, hat as separate sampled voices

Extended chords are requested where the style asks for them and reduced by
`rse.harmony.voicing`, so a maj9 costs three harmony voices rather than five.

The allocator then decides what physically fits. This module does not enforce
the eight-voice ceiling itself -- it proposes parts and lets the allocator
report what it had to drop, because a part that silently vanishes is the defect
the whole role/voice split exists to prevent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rse.harmony.intent import ChordIntent, Function, Quality, Seventh
from rse.harmony.voicing import VoicingRequest, bass_pitch_for, plan_voicing
from rse.platform.roles import PartNote, Role, VoiceAllocator
from rse.snes.sampler import VoiceSettings

#: Which instrument plays which role, and where it sits in the stereo field.
#:
#: The lead stays centre and the bass stays centre: a melody that wanders
#: across the image is distracting and low frequencies carry no useful
#: directional information. Everything else is spread, which is clarity
#: rather than effect.
#:
#: The first pass used +/-0.35 and produced a channel correlation of 0.976 --
#: a stereo file with almost no stereo in it, because the centred lead, bass
#: and kick dominate the mix and are identical in both channels. The parts
#: that CAN move have to move further for the image to open up at all.
VOICE_PLAN: dict[Role, VoiceSettings] = {
    Role.LEAD: VoiceSettings(instrument="brass", pan=0.0, gain=1.00, echo=0.25),
    Role.HARMONY: VoiceSettings(instrument="strings", pan=-0.62, gain=0.68, echo=0.35),
    Role.PAD: VoiceSettings(instrument="choir", pan=0.58, gain=0.56, echo=0.45),
    Role.BASS: VoiceSettings(instrument="bass", pan=0.0, gain=0.80, echo=0.10),
    Role.PERCUSSION: VoiceSettings(instrument="snare", pan=0.22, gain=0.70),
    Role.COUNTERMELODY: VoiceSettings(
        instrument="choir", pan=0.55, gain=0.45, echo=0.30
    ),
}

#: Releases, in ticks. Each part's duration is sized so the release lands
#: INSIDE its own rhythmic slot rather than bleeding into the next note.
#:
#: Not a detail. A sustained chord whose release runs past the bar line is
#: still occupying its voices when the next chord starts, so three harmony
#: notes cost four voices at every bar boundary -- measured peak 4, not 3 --
#: and the arrangement asked for eleven voices on a console with eight. The
#: pad lost all eight of its notes to it. A release that ends as the next
#: chord arrives is also the musically correct shape.
HARMONY_RELEASE = 240
PAD_RELEASE = 320
BASS_RELEASE = 60
LEAD_RELEASE = 60


#: Percussion is three instruments on one role, chosen by the drum's value.
DRUM_INSTRUMENTS: dict[int, str] = {1: "kick", 6: "snare", 12: "hat"}


@dataclass
class ArrangedSong:
    """Parts, voicings and the allocation they produced."""

    parts: list[PartNote] = field(default_factory=list)
    chords: list[ChordIntent] = field(default_factory=list)
    voicings: list[Any] = field(default_factory=list)
    extended_bars: list[int] = field(default_factory=list)
    notes: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "parts": len(self.parts),
            "by_role": self.notes,
            "chords": [c.symbol for c in self.chords],
            "extended_chord_bars": list(self.extended_bars),
            "voicings": [v.to_dict() for v in self.voicings],
        }


#: Scale-degree semitone patterns, mirroring the NES composer's modes so a
#: shared plan means the same thing on both platforms.
SCALES: dict[str, tuple[int, ...]] = {
    "major": (0, 2, 4, 5, 7, 9, 11),
    "minor": (0, 2, 3, 5, 7, 8, 10),
    "harmonic_minor": (0, 2, 3, 5, 7, 8, 11),
    "dorian": (0, 2, 3, 5, 7, 9, 10),
    "phrygian": (0, 1, 3, 5, 7, 8, 10),
    "mixolydian": (0, 2, 4, 5, 7, 9, 10),
}


def chord_for(
    *,
    degree: int,
    tonic: int,
    mode: str,
    extended: bool,
    section: str,
) -> ChordIntent:
    """The chord a scale degree means, optionally extended.

    The triad quality comes from the mode, exactly as the NES composer derives
    it, so the harmonic plan is shared rather than reinvented. Extension is
    the platform-specific part: SNES can afford a seventh and a ninth, and the
    NES never could.
    """
    scale = SCALES[mode]
    root_pc = (tonic + scale[degree % len(scale)]) % 12
    third = (scale[(degree + 2) % 7] - scale[degree % 7]) % 12
    fifth = (scale[(degree + 4) % 7] - scale[degree % 7]) % 12
    if third == 3 and fifth == 6:
        quality = Quality.DIMINISHED
    elif third == 3:
        quality = Quality.MINOR
    elif third == 4 and fifth == 8:
        quality = Quality.AUGMENTED
    else:
        quality = Quality.MAJOR

    seventh = Seventh.NONE
    extensions: tuple[int, ...] = ()
    if extended and quality in (Quality.MAJOR, Quality.MINOR):
        # The seventh the mode supplies, so an extended chord stays diatonic.
        interval = (scale[(degree + 6) % 7] - scale[degree % 7]) % 12
        if interval == 11:
            # Diatonic, but a minor-major seventh is a harsh, unusual chord
            # and harmonic minor hands one out on the tonic. The repertoire
            # this is aiming at uses m7 there, so a minor triad takes the
            # minor seventh even where the mode offers the major one.
            seventh = (
                Seventh.MINOR if quality is Quality.MINOR else Seventh.MAJOR
            )
        elif interval == 10:
            seventh = Seventh.MINOR
        if seventh is not Seventh.NONE:
            extensions = (9,)

    function = Function.TONIC
    if degree == 4:
        function = Function.DOMINANT
    elif degree == 3:
        function = Function.SUBDOMINANT
    elif degree in (1, 5):
        function = Function.PREDOMINANT

    return ChordIntent(
        root=root_pc,
        quality=quality,
        seventh=seventh,
        extensions=extensions,
        function=function,
        degree=degree,
    )


def approach_note(
    *, target: int, from_pitch: int, tonic: int, mode: str
) -> int:
    """A scale step into `target`, from whichever side moves least.

    A real approach note leads somewhere. The first version was
    `bass + (2 if degree % 2 == 0 else -2)` -- a whole tone up or down chosen
    by whether the degree INDEX was even, with no reference to the next chord
    and no guarantee of being a step at all. In A harmonic minor it produced
    A2 -> A3 -> B2 -> A2: an octave leap up, a drop of a minor seventh to a
    whole tone above the root, then back. The listener heard that as a note
    that sounded out of place, which it was.

    The note is picked from the scale and the NEAREST one wins, so a semitone
    approach beats a whole tone: into A minor's tonic that selects the leading
    tone G#, which is the strongest approach available and the right answer.
    `from_pitch` only breaks an exact tie -- when the scale offers a step of
    the same size on both sides, the line approaches from the side it is
    already on.
    """
    scale = SCALES[mode]
    # Scale pitches within an octave either side of the target.
    candidates = [
        tonic + 12 * octave + semitone
        for octave in range(-4, 5)
        for semitone in scale
        if 0 < abs(tonic + 12 * octave + semitone - target) <= 2
    ]
    if not candidates:
        return target
    # Nearest to the target, breaking ties toward where the line already is.
    return min(
        candidates,
        key=lambda pitch: (abs(pitch - target), abs(pitch - from_pitch)),
    )


def arrange(
    result: Any,
    *,
    ticks_per_bar: int = 1920,
    harmony_voices: int = 3,
    contrast_harmony_voices: int = 3,
    extended_sections: tuple[str, ...] = ("B",),
) -> ArrangedSong:
    """Re-realise a procedural composition for sampled, polyphonic voices.

    `result` is a `ProceduralResult`. Its melody is taken as written -- the
    tune is the thing that works and this is not a composition change -- while
    the harmony, bass and percussion are rebuilt as polyphonic parts.
    """
    song = ArrangedSong()
    style = result.style
    tonic, mode = style.tonic, style.mode

    # -- lead: the composed melody, on one voice --------------------------
    for event in result.composition.sounding_events():
        if event.channel_id != "P1":
            continue
        song.parts.append(
            PartNote(
                role=Role.LEAD,
                tick=event.tick,
                duration=event.duration,
                value=event.pitch,
                velocity=event.velocity,
                instrument_id="brass",
                release=LEAD_RELEASE,
            )
        )

    # -- harmony, pad and bass: rebuilt from the harmonic plan ------------
    bar_offset = 0
    previous: tuple[int, ...] = ()
    previous_bass: int | None = None
    for section in result.sections:
        extended = section.label.startswith(extended_sections)
        for bar in range(section.bars):
            degree_spec = section.progression[bar % len(section.progression)]
            degree = degree_spec if isinstance(degree_spec, int) else degree_spec.degree
            chord = chord_for(
                degree=degree, tonic=tonic, mode=mode,
                extended=extended, section=section.label,
            )
            song.chords.append(chord)
            absolute_bar = bar_offset + bar
            if chord.seventh is not Seventh.NONE:
                song.extended_bars.append(absolute_bar)

            tick = absolute_bar * ticks_per_bar
            bass = bass_pitch_for(chord, near=previous_bass)
            # The contrast section adds a choir pad, and eight voices will not
            # hold a lead, three harmony notes, a bass, three drums AND a pad.
            # The pad is the orchestration change that makes B sound like a
            # different place, so it takes the voice rather than being dropped
            # -- which is what happened on the first run: all eight pad notes
            # lost, and the section's whole point with them.
            budget = contrast_harmony_voices if extended else harmony_voices
            voicing = plan_voicing(
                VoicingRequest(
                    intent=chord,
                    max_voices=budget,
                    bass=bass,
                    previous=previous,
                )
            )
            song.voicings.append(voicing)
            previous = voicing.pitches
            previous_bass = bass

            # Sustained harmony: one voice per pitch, held for the bar. This
            # is the thing the NES cannot do -- its single comp channel can
            # only arpeggiate a chord, never state it.
            for pitch in voicing.pitches:
                song.parts.append(
                    PartNote(
                        role=Role.HARMONY,
                        tick=tick,
                        duration=ticks_per_bar - HARMONY_RELEASE,
                        value=pitch,
                        velocity=74,
                        instrument_id="strings",
                        release=HARMONY_RELEASE,
                    )
                )

            # Choir pad, contrast section only: the orchestration change that
            # makes B sound like a different place.
            if extended:
                song.parts.append(
                    PartNote(
                        role=Role.PAD,
                        tick=tick,
                        duration=ticks_per_bar - PAD_RELEASE,
                        value=voicing.pitches[0] + 12,
                        velocity=62,
                        instrument_id="choir",
                        release=PAD_RELEASE,
                    )
                )

            # Active bass: root, an octave lift mid-bar, then a genuine step
            # into the NEXT bar's root. More than the NES bass, which
            # alternates root and fifth -- and the approach has to lead
            # somewhere, which the first version did not.
            quarter = ticks_per_bar // 4
            bass_hold = quarter - BASS_RELEASE
            next_degree_spec = _next_degree(result.sections, bar_offset + bar)
            if next_degree_spec is None:
                next_bass = bass
            else:
                next_chord = chord_for(
                    degree=next_degree_spec, tonic=tonic, mode=mode,
                    extended=False, section=section.label,
                )
                next_bass = bass_pitch_for(next_chord, near=bass)
            lift = bass + 12
            # The approach steps into the next root's UPPER octave, not its
            # low one. Targeting the low octave made the within-bar motion a
            # drop of 13 to 20 semitones -- the octave lift jumped up and the
            # approach fell straight back down, which is the awkward contour
            # the listener picked out even once the pitch itself was right.
            #
            # This way the lift-to-approach move is a step and the octave drop
            # lands on the bar line, where a bass re-establishing its register
            # is an idiomatic gesture rather than a stray leap.
            step = approach_note(
                target=next_bass + 12, from_pitch=lift, tonic=tonic, mode=mode
            )
            for offset, pitch, velocity in (
                (0, bass, 96),
                (quarter * 2, lift, 84),
                (quarter * 3, step, 80),
            ):
                song.parts.append(
                    PartNote(
                        role=Role.BASS, tick=tick + offset, duration=bass_hold,
                        value=pitch, velocity=velocity,
                        instrument_id="bass", release=BASS_RELEASE,
                    )
                )
        bar_offset += section.bars

    # -- percussion: the composed drum part, as sampled hits --------------
    for event in result.composition.sounding_events():
        if event.channel_id != "NO":
            continue
        song.parts.append(
            PartNote(
                role=Role.PERCUSSION,
                tick=event.tick,
                duration=event.duration,
                value=event.value,
                velocity=event.velocity,
                instrument_id=DRUM_INSTRUMENTS.get(event.value, "snare"),
                release=0,
            )
        )

    counts: dict[str, int] = {}
    for part in song.parts:
        counts[part.role.value] = counts.get(part.role.value, 0) + 1
    song.notes = counts
    return song


def _next_degree(sections: list[Any], absolute_bar: int) -> int | None:
    """The scale degree of the bar after `absolute_bar`, across section joins.

    Across joins on purpose: the approach note at the end of a section has to
    lead into the next section's first chord, which is exactly where a bass
    line most needs to connect.
    """
    offset = 0
    bars: list[int] = []
    for section in sections:
        for bar in range(section.bars):
            spec = section.progression[bar % len(section.progression)]
            bars.append(spec if isinstance(spec, int) else spec.degree)
        offset += section.bars
    index = absolute_bar + 1
    if index >= len(bars):
        # The last bar leads back to the first, since the track loops.
        return bars[0] if bars else None
    return bars[index]


def percussion_settings(value: int) -> VoiceSettings:
    """Percussion is three instruments sharing one role."""
    instrument = DRUM_INSTRUMENTS.get(value, "snare")
    pans = {"kick": 0.0, "snare": 0.22, "hat": -0.52}
    gains = {"kick": 0.95, "snare": 0.70, "hat": 0.38}
    return VoiceSettings(
        instrument=instrument,
        pan=pans.get(instrument, 0.0),
        gain=gains.get(instrument, 0.7),
    )


def allocate(song: ArrangedSong, *, voices: int = 8, reserved: int = 0):
    """Hand the parts to the allocator. It decides what fits."""
    return VoiceAllocator(
        [f"V{i}" for i in range(1, voices + 1)], reserved=reserved
    ).allocate(song.parts)
