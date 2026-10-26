"""Altered chords, and the melody knowing about the alteration.

The v1 composer stacks scale degrees, which cannot express a borrowed chord:
E major in A *natural* minor needs G#, and that scale has G. So `i-iv-V-i` in
natural minor was unreachable, and that dominant is the most useful chord the
old system lacked.

The integration requirement is the one the operator called most important: if
a borrowed chord sounds and the melody is still free to play any scale tone,
it will put the replaced note against the alteration -- G against G# -- and do
it repeatedly.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from rse.hardware.constraints import ConstraintEngine
from rse.hardware.profile import load_profile
from rse.procedural.chords import (
    QUALITY_INTERVALS,
    Chord,
    as_chord,
    melody_candidates,
    resolve,
)
from rse.procedural.compose import EXTRA_PROGRESSIONS, STYLES, compose

A, C = 9, 0
BASELINE = json.loads(
    (Path(__file__).parent / "data" / "procedural_v1_baseline.json").read_text()
)


def signature(result) -> str:
    events = [
        (
            e.channel_id,
            e.tick,
            getattr(e, "pitch", getattr(e, "value", 0)),
            e.duration,
            e.velocity,
        )
        for e in result.composition.sounding_events()
    ]
    return hashlib.sha256(json.dumps(events, sort_keys=True).encode()).hexdigest()[:16]


# -- the case the old system could not express ----------------------------


def test_natural_minor_dominant_is_minor_without_an_override() -> None:
    """Degree 4 of natural minor is v minor -- the thing being worked around."""
    voicing = resolve(4, tonic=A, mode="minor")
    assert [(A + s) % 12 for s in voicing.semitones] == [4, 7, 11]  # E G B
    assert voicing.quality == "minor"
    assert voicing.diatonic is True


def test_borrowed_dominant_in_natural_minor() -> None:
    """E-G#-B in A natural minor, flagged as not diatonic."""
    voicing = resolve((4, "major"), tonic=A, mode="minor")
    assert [(A + s) % 12 for s in voicing.semitones] == [4, 8, 11]  # E G# B
    assert voicing.quality == "major"
    assert voicing.diatonic is False


def test_harmonic_minor_dominant_needs_no_override() -> None:
    """Already diatonic there, so the new mechanism is not used."""
    voicing = resolve(4, tonic=A, mode="harmonic_minor")
    assert [(A + s) % 12 for s in voicing.semitones] == [4, 8, 11]
    assert voicing.diatonic is True


def test_phrygian_flat_two_needs_no_override() -> None:
    voicing = resolve(1, tonic=7, mode="phrygian")
    assert voicing.quality == "major"
    assert voicing.diatonic is True


# -- the safeguard --------------------------------------------------------


def test_alteration_suppresses_the_note_it_replaced() -> None:
    """G natural is removed while E major sounds; nothing else is."""
    voicing = resolve((4, "major"), tonic=A, mode="minor")
    avoided = {(A + a) % 12 for a in voicing.avoid}
    assert avoided == {7}, "only G natural should be suppressed"


def test_melody_keeps_its_scale_and_loses_only_the_replaced_note() -> None:
    """Not restricted to chord tones -- that would kill the passing tones."""
    voicing = resolve((4, "major"), tonic=A, mode="minor")
    chord_tones, scale_tones = melody_candidates(voicing, mode="minor")
    chord_classes = {(A + t) % 12 for t in chord_tones}
    scale_classes = {(A + t) % 12 for t in scale_tones}
    assert 8 in chord_classes, "G# must be available"
    assert 7 not in scale_classes, "G natural must not be"
    # Still a scale, not a chord: more than three pitch classes survive.
    assert len(scale_classes) >= 5


def test_diatonic_chords_suppress_nothing() -> None:
    for mode in ("minor", "major", "phrygian", "harmonic_minor"):
        for degree in range(7):
            assert resolve(degree, tonic=A, mode=mode).avoid == frozenset()


def test_no_clash_with_a_suppressed_tone_in_generated_music() -> None:
    """End to end: the borrowed chord never gets its replaced note over it."""
    style = STYLES["action"]
    result = compose(
        style_name="action", seed=900, target_seconds=60.0, vocabulary="extra"
    )
    melody = [
        e for e in result.composition.sounding_events() if e.channel_id == "P1"
    ]
    bar_ticks = 1920
    offset = 0
    clashes = []
    for section in result.sections:
        for bar in range(section.bars):
            spec = section.progression[bar % len(section.progression)]
            voicing = resolve(spec, tonic=style.tonic, mode=style.mode)
            if not voicing.avoid:
                continue
            avoided = {(style.tonic + a) % 12 for a in voicing.avoid}
            index = offset + bar
            for event in melody:
                if index * bar_ticks <= event.tick < (index + 1) * bar_ticks:
                    if event.pitch % 12 in avoided:
                        clashes.append((index, event.pitch % 12))
        offset += section.bars
    assert not clashes, f"melody used a suppressed tone: {clashes}"


# -- the opt-in contract --------------------------------------------------


@pytest.mark.parametrize("key", sorted(BASELINE))
def test_v1_unchanged_by_the_chord_system(key: str) -> None:
    style, _, seed = key.rpartition("_s")
    # `sound_design=None` on purpose. The baseline's job is to prove the
    # COMPOSITION has not moved, and the signature includes velocity, which
    # is now a performance decision rather than a compositional one. Pinning
    # the expressive render here would make every future sound-design change
    # look like a composition regression.
    result = compose(
        style_name=style, seed=int(seed), target_seconds=60.0, sound_design=None
    )
    assert signature(result) == BASELINE[key]
    assert result.vocabulary is None


def test_integer_progressions_still_mean_mode_derived() -> None:
    """v1 progressions are plain ints and must keep the old voicing path."""
    assert as_chord(3) == Chord(degree=3)
    assert resolve(3, tonic=A, mode="minor").degrees is not None


def test_extra_vocabulary_changes_the_chords_not_the_note_count() -> None:
    """Different harmony, same motif and rhythm, so the same event count."""
    plain = compose(style_name="action", seed=900, target_seconds=60.0)
    extra = compose(
        style_name="action", seed=900, target_seconds=60.0, vocabulary="extra"
    )
    assert signature(plain) != signature(extra)
    assert len(plain.composition.sounding_events()) == len(
        extra.composition.sounding_events()
    )


def test_unknown_vocabulary_is_rejected() -> None:
    with pytest.raises(KeyError, match="unknown vocabulary"):
        compose(style_name="boss", seed=1, vocabulary="jazz")


def test_unknown_quality_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown chord quality"):
        as_chord((4, "sus9"))


# -- every new family is playable and nameable ---------------------------


@pytest.mark.parametrize("style_name", sorted(EXTRA_PROGRESSIONS))
def test_new_families_resolve_and_play(style_name: str) -> None:
    style = STYLES[style_name]
    for progression in EXTRA_PROGRESSIONS[style_name]:
        for spec in progression:
            voicing = resolve(spec, tonic=style.tonic, mode=style.mode)
            assert len(voicing.semitones) == 3
            assert voicing.quality in QUALITY_INTERVALS


@pytest.mark.parametrize("style_name", sorted(STYLES))
@pytest.mark.parametrize("vocabulary", [None, "extra", "all"])
def test_every_vocabulary_is_hardware_valid(
    style_name: str, vocabulary: str | None
) -> None:
    result = compose(
        style_name=style_name, seed=77, target_seconds=60.0, vocabulary=vocabulary
    )
    engine = ConstraintEngine(load_profile("nes"))
    assert engine.validate(result.composition) == []
    assert result.composition.sounding_events()


def test_bass_plays_the_altered_root() -> None:
    """The bass must follow a borrowed chord, not the mode's version."""
    style = STYLES["action"]
    result = compose(
        style_name="action", seed=900, target_seconds=60.0, vocabulary="extra"
    )
    bass = {
        e.pitch % 12
        for e in result.composition.sounding_events()
        if e.channel_id == "TR"
    }
    # G is the borrowed dominant's root and must appear in the bass.
    assert (style.tonic + 7) % 12 in bass
