"""Turning a chord into the fewest notes that still sound like it.

Cmaj9 is five pitch classes. A convincing SNES voicing of it is four notes,
and three of those are the harmony part:

    bass    C           the root, stated once and low
    harmony E  B  D      third, seventh, ninth
    omitted    G         the fifth

The fifth goes first because it carries no information. The root is implied by
it, and it says nothing about whether the chord is major, minor, dominant or
anything else. The third and the seventh are what make Cmaj9 that chord rather
than C major or C7, so they are the last to go. The ninth is the colour the
composer asked for, so it outranks the eleventh and thirteenth but yields to
the tones that define quality.

Two things make this a planner rather than a lookup table.

**It knows what the bass is doing.** A rootless voicing is only usable when
something else states the root; otherwise the chord loses its footing. So the
bass part is an input, not an afterthought, and the root becomes droppable only
once the bass has it.

**It knows where the previous chord was.** Voice leading is the difference
between a progression and a sequence of unrelated stacks: holding common tones
and moving the rest by the smallest available interval is most of what makes
changes sound smooth. The planner is given the previous voicing and prefers
registers that keep voices still.

Deliberately not a jazz-harmony grammar. Triads, sevenths, ninths, basic
inversions, omission and common-tone retention -- enough for the first SNES
track, and no attempt at altered dominants or reharmonisation.
"""

from __future__ import annotations

from dataclasses import dataclass

from rse.harmony.intent import (
    OMISSION_ORDER,
    ChordIntent,
    ChordVoicing,
    Quality,
    Tone,
)

#: Register the harmony part sits in, as MIDI pitches. Above the bass and
#: below a lead that is usually in the fifth octave, so the three parts do not
#: fight for the same space.
HARMONY_LOW = 52
HARMONY_HIGH = 79

#: Where the bass sits.
BASS_LOW = 28
BASS_HIGH = 52


@dataclass(frozen=True)
class VoicingRequest:
    """What the arranger has, and what it can spare."""

    intent: ChordIntent
    #: How many voices the harmony part may use. The whole constraint.
    max_voices: int
    #: Pitch the bass part is playing, if any. Lets the root be dropped.
    bass: int | None = None
    #: The lead's pitch, so the harmony does not double it in the same octave
    #: and muddy the melody.
    lead: int | None = None
    #: The previous voicing's pitches, for common-tone retention.
    previous: tuple[int, ...] = ()
    low: int = HARMONY_LOW
    high: int = HARMONY_HIGH
    #: Use a voice the bass freed, when the budget has room for it.
    #:
    #: Handing the root to the bass should not cost the harmony a voice.
    #: False reproduces the behaviour before that was fixed -- the bass tone
    #: removed and the freed voice left unused, so a three-voice budget
    #: returned a rootless two-note dyad, and `max_voices` had no effect at
    #: all above 2.
    #:
    #: True here, but NOT unconditionally true upstream: `generate_snes`
    #: decides from the voice reservation, because the extra note is free
    #: with 0 or 1 voices reserved and costs 15 dropped notes at 2 or more.
    #:
    #: Confirmed by listening, which is the evidence that matters -- the
    #: before/after pair was rendered and the fixed version judged better.
    #: Two NES changes reasoned out the same way were rejected by the same
    #: ear in the same session, so the measurement alone would not have been
    #: enough. False stays reachable to keep that comparison renderable, and
    #: because a deliberately thin, open texture is a real choice.
    spend_freed_voice: bool = True


def bass_pitch_for(
    intent: ChordIntent, *, low: int = BASS_LOW, high: int = BASS_HIGH,
    near: int | None = None,
) -> int:
    """The bass note, honouring the inversion.

    An inversion is a statement about the bass, so this is where it takes
    effect: a chord in first inversion puts its third here, and the harmony
    planner then knows the third is already covered.
    """
    tone = intent.bass_tone()
    pitch_class = (intent.root + intent.tones()[tone]) % 12
    candidates = [p for p in range(low, high + 1) if p % 12 == pitch_class]
    if not candidates:
        raise ValueError(f"no bass pitch for class {pitch_class} in {low}-{high}")
    if near is None:
        # Low in the range, which is where a bass line lives.
        return candidates[len(candidates) // 3]
    return min(candidates, key=lambda p: abs(p - near))


def plan_voicing(request: VoicingRequest) -> ChordVoicing:
    """Choose the pitches for one chord.

    Sheds tones in `OMISSION_ORDER` until the chord fits, then places what is
    left in the register that moves least from the previous chord.
    """
    intent = request.intent
    tones = intent.tones()
    if request.max_voices < 1:
        raise ValueError(f"max_voices must be at least 1, got {request.max_voices}")

    bass_tone: Tone | None = None
    if request.bass is not None:
        bass_class = request.bass % 12
        for tone, semitones in tones.items():
            if (intent.root + semitones) % 12 == bass_class:
                bass_tone = tone
                break

    # Start from everything, minus whatever the bass already states.
    keep = {t for t in tones if t is not bass_tone}

    # Shed in priority order until it fits. The root is only droppable once
    # the bass is holding it -- a rootless voicing with no root anywhere is
    # not a voicing of this chord.
    omitted: list[Tone] = []
    for tone in OMISSION_ORDER:
        if len(keep) <= request.max_voices:
            break
        if tone not in keep:
            continue
        if tone is Tone.ROOT and bass_tone is not Tone.ROOT:
            continue
        if tone is Tone.THIRD and intent.quality in (
            Quality.SUSPENDED_2, Quality.SUSPENDED_4
        ):
            # A suspension's "third" IS the suspended tone; dropping it makes
            # the chord a bare fifth.
            continue
        keep.discard(tone)
        omitted.append(tone)

    if len(keep) > request.max_voices:
        # Still too many: we are down to the defining tones and have to cut
        # into them. Reported, because this changes the harmony.
        for tone in reversed(OMISSION_ORDER):
            if len(keep) <= request.max_voices:
                break
            if tone in keep:
                keep.discard(tone)
                omitted.append(tone)

    # Spend what the bass freed. Handing the root to the bass is the right
    # orchestration -- the harmony should not double it in unison -- but
    # REMOVING it and then not using the voice it freed is how a three-voice
    # budget produced two notes.
    #
    # Measured before this: 34 of 40 bars in a boss track were rootless
    # dyads, 80% of the track had exactly two harmony notes sounding, and
    # max_voices had no effect at all above 2 -- a C major triad came back as
    # (E, G) whether it was given a budget of 2, 3 or 4. A rootless third and
    # fifth is harmonically ambiguous: C-E-G minus the C could be C major, A
    # minor or F major, so the chord's identity rested entirely on the bass
    # and a listener heard padding rather than harmony.
    #
    # Restored here rather than by not removing it above, because the removal
    # is correct when the budget really is tight: with two voices over a root
    # bass, (third, fifth) is the textbook choice and (root, third) wastes a
    # voice on a note the bass already has. This only ever adds a voice that
    # was going unused. `_place` puts it in the harmony register, 52 and up,
    # so it lands at least an octave above the bass instead of doubling it.
    if (
        request.spend_freed_voice
        and bass_tone is not None
        and len(keep) < request.max_voices
    ):
        keep.add(bass_tone)
        if bass_tone in omitted:
            omitted.remove(bass_tone)

    placed = _place(
        intent=intent,
        keep=keep,
        low=request.low,
        high=request.high,
        previous=request.previous,
        lead=request.lead,
    )
    ordered_tones = tuple(
        tone for _p, tone in sorted(
            ((p, t) for t, p in placed.items()), key=lambda pt: pt[0]
        )
    )
    return ChordVoicing(
        intent=intent,
        pitches=tuple(sorted(placed.values())),
        omitted=tuple(omitted),
        bass_pitches=(request.bass,) if request.bass is not None else (),
        tones=ordered_tones,
    )


def _place(
    *,
    intent: ChordIntent,
    keep: set[Tone],
    low: int,
    high: int,
    previous: tuple[int, ...],
    lead: int | None,
) -> dict[Tone, int]:
    """Choose an octave for each kept tone.

    Scored rather than computed: every candidate pitch gets a cost, and the
    cheapest wins. Distance from the previous voicing is the dominant term,
    which is what produces common-tone retention without a special case for
    it -- a tone that was already sounding has distance zero.
    """
    tones = intent.tones()
    chosen: dict[Tone, int] = {}
    # Low tones first, so the bottom of the chord is settled before the upper
    # extensions are placed above it.
    for tone in sorted(keep, key=lambda t: tones[t]):
        pitch_class = (intent.root + tones[tone]) % 12
        candidates = [p for p in range(low, high + 1) if p % 12 == pitch_class]
        if not candidates:
            continue

        def cost(pitch: int, tone: Tone = tone) -> float:
            score = 0.0
            if previous:
                score += min(abs(pitch - p) for p in previous) * 1.0
            else:
                # No history: favour the middle of the register.
                score += abs(pitch - (low + high) / 2) * 0.3
            # Do not double the lead in its own octave; it blurs the melody.
            if lead is not None and pitch % 12 == lead % 12:
                score += 6.0 if abs(pitch - lead) < 12 else 1.0
            # Keep the stack from collapsing onto one pitch.
            for other in chosen.values():
                if pitch == other:
                    score += 20.0
                elif abs(pitch - other) < 3:
                    score += 2.0
            # Upper extensions belong above the tones they colour.
            if tone in (Tone.NINTH, Tone.ELEVENTH, Tone.THIRTEENTH) and chosen:
                if pitch < max(chosen.values()):
                    score += 4.0
            return score

        chosen[tone] = min(candidates, key=cost)
    return chosen


def plan_progression(
    intents: list[ChordIntent],
    *,
    max_voices: int,
    lead: list[int] | None = None,
    with_bass: bool = True,
) -> list[ChordVoicing]:
    """Voice a whole progression, carrying voice leading between chords.

    Each chord is planned against the previous one's pitches, so common tones
    are held across the change rather than re-chosen independently.
    """
    out: list[ChordVoicing] = []
    previous: tuple[int, ...] = ()
    previous_bass: int | None = None
    for index, intent in enumerate(intents):
        bass = (
            bass_pitch_for(intent, near=previous_bass) if with_bass else None
        )
        voicing = plan_voicing(
            VoicingRequest(
                intent=intent,
                max_voices=max_voices,
                bass=bass,
                lead=lead[index] if lead and index < len(lead) else None,
                previous=previous,
            )
        )
        out.append(voicing)
        previous = voicing.pitches
        previous_bass = bass
    return out


def common_tones(first: ChordVoicing, second: ChordVoicing) -> int:
    """Pitches held unchanged across a chord change."""
    return len(set(first.pitches) & set(second.pitches))


def movement(first: ChordVoicing, second: ChordVoicing) -> int:
    """Total semitones moved, pairing each voice with its nearest successor.

    The measure voice leading is judged by: smaller is smoother.
    """
    if not first.pitches or not second.pitches:
        return 0
    return sum(
        min(abs(pitch - other) for other in second.pitches)
        for pitch in first.pitches
    )


def describe(voicing: ChordVoicing) -> str:
    """One line, naming what sounds and what was given up."""
    names = ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")
    sounding = " ".join(
        f"{names[p % 12]}{p // 12 - 1}" for p in voicing.pitches
    )
    text = f"{voicing.intent.symbol}: {sounding}"
    if voicing.bass_pitches:
        bass = voicing.bass_pitches[0]
        text += f"  bass {names[bass % 12]}{bass // 12 - 1}"
    if voicing.omitted:
        text += f"  (omit {', '.join(t.value for t in voicing.omitted)})"
    if not voicing.keeps_identity:
        text += "  [IDENTITY LOST]"
    return text


def fits(intent: ChordIntent, max_voices: int, *, with_bass: bool = True) -> bool:
    """Whether the chord survives being squeezed into `max_voices`.

    Checks identity rather than note count: a voicing that fits but has lost
    its third or seventh is not the chord that was asked for.
    """
    bass = bass_pitch_for(intent) if with_bass else None
    voicing = plan_voicing(
        VoicingRequest(intent=intent, max_voices=max_voices, bass=bass)
    )
    return voicing.keeps_identity
