"""Conversion between D4 events and a model-native score."""

from __future__ import annotations

import pytest

from rse.events.schema import Composition, Noise, Note
from rse.transfer.converters import (
    GM_TO_NOISE,
    NOISE_TO_GM,
    composition_to_score,
    percussion_round_trip,
    score_to_composition,
)
from rse.transfer.score import PERCUSSION, Score, ScoreNote, ScorePercussion, Track

ROW_SEC = 0.1  # 150 BPM: a beat is 4 rows = 0.4 s


def _piece(events) -> Composition:
    comp = Composition(hardware_profile="nes", ticks_per_beat=480, events=list(events))
    comp.normalize()
    return comp


def _notes(channel, pairs, duration=120):
    return [
        Note(channel_id=channel, tick=t, pitch=p, duration=duration, velocity=96) for t, p in pairs
    ]


# -- the drum mapping ---------------------------------------------------------


def test_noise_to_gm_is_injective_and_covers_every_period():
    assert len(NOISE_TO_GM) == 16
    assert len(set(NOISE_TO_GM.values())) == 16
    assert set(NOISE_TO_GM) == set(range(1, 17))
    assert all(35 <= gm <= 81 for gm in NOISE_TO_GM.values()), "must be GM percussion"


def test_every_noise_period_round_trips_without_the_defect():
    report = percussion_round_trip(0)
    assert report["n_exact"] == 16
    assert report["n_lost"] == 0
    assert report["n_wrong_period"] == 0


def test_the_fermata_drum_shift_loses_percussion_detectably_not_silently():
    """Fermata's `midi-unpitched` values are 1-based but were read as 0-based,
    so drum notes come out one GM key high. The mapping is spaced two keys
    apart precisely so a shifted key lands outside it: percussion that is
    visibly absent is far safer than percussion that is quietly the wrong
    drum, which no onset or duration metric would catch."""
    report = percussion_round_trip(1)
    assert report["n_exact"] == 0
    assert report["n_wrong_period"] == 0, "a shift must never silently remap"
    assert report["n_lost"] == 16


def test_adjacent_keys_would_have_hidden_the_defect():
    """Guards the reasoning behind the spacing, not just its result."""
    adjacent = {period: 36 + period for period in range(1, 17)}
    back = {gm: p for p, gm in adjacent.items()}
    silently_wrong = sum(1 for p, gm in adjacent.items() if back.get(gm + 1) not in (None, p))
    assert silently_wrong >= 15, "adjacent keys hide a +1 shift"


# -- quantization -------------------------------------------------------------


def test_an_exact_grid_preserves_onsets_exactly():
    """48 TPQN is a multiple of RSE's beat/16, so nothing should move."""
    piece = _piece(_notes("P1", [(0, 60), (30, 62), (60, 64), (90, 67)]))
    score, loss = composition_to_score(
        piece, ticks_per_quarter=48, tempo_bpm=150.0, row_sec=ROW_SEC
    )
    assert loss.n_onset_collisions == 0
    assert loss.n_events_out == loss.n_events_in
    rebuilt, _ = score_to_composition(score, row_sec=ROW_SEC)
    assert len(rebuilt.sounding_events()) == 4


def test_a_coarse_grid_collides_onsets_and_reports_it():
    """Two events a sub-row apart can land on one tick at 8 TPQN. A
    monophonic channel cannot hold both, and the loss must be counted rather
    than letting the second silently overwrite the first."""
    # At 150 BPM a beat is 0.4 s; 8 TPQN gives 50 ms ticks, so events 25 ms
    # apart share a tick.
    piece = _piece(_notes("P1", [(0, 60), (12, 64)], duration=10))
    _score, loss = composition_to_score(
        piece, ticks_per_quarter=8, tempo_bpm=150.0, row_sec=ROW_SEC
    )
    assert loss.n_onset_collisions == 1
    assert loss.n_events_out < loss.n_events_in


def test_conversion_refuses_a_nonsensical_grid():
    piece = _piece(_notes("P1", [(0, 60)]))
    with pytest.raises(ValueError, match="bad grid"):
        composition_to_score(piece, ticks_per_quarter=0, tempo_bpm=150.0, row_sec=ROW_SEC)
    with pytest.raises(ValueError, match="bad grid"):
        composition_to_score(piece, ticks_per_quarter=24, tempo_bpm=150.0, row_sec=0.0)


# -- track and channel identity ----------------------------------------------


def test_channel_identity_survives_the_round_trip():
    """A conversion that forgets which track was the triangle cannot put it
    back, and the arranger would have to guess."""
    piece = _piece(
        _notes("P1", [(0, 72)])
        + _notes("P2", [(0, 67)])
        + _notes("TR", [(0, 43)])
        + [Noise(channel_id="NO", tick=0, value=5, duration=60, velocity=96)]
    )
    score, _ = composition_to_score(piece, ticks_per_quarter=24, tempo_bpm=150.0, row_sec=ROW_SEC)
    assert {t.channel_id for t in score.tracks} == {"P1", "P2", "TR", "NO"}
    rebuilt, _ = score_to_composition(score, row_sec=ROW_SEC)
    assert {e.channel_id for e in rebuilt.sounding_events()} == {"P1", "P2", "TR", "NO"}


def test_the_noise_channel_becomes_percussion_not_a_pitched_note():
    """A noise period is a timbre selector, not a pitch. Treating them as the
    same integer is what would make the round trip look lossless while
    changing which drum sounds."""
    piece = _piece([Noise(channel_id="NO", tick=0, value=9, duration=60, velocity=96)])
    score, _ = composition_to_score(piece, ticks_per_quarter=24, tempo_bpm=150.0, row_sec=ROW_SEC)
    track = score.track_for("NO")
    assert track is not None and track.role == PERCUSSION
    assert track.notes == []
    assert len(track.percussion) == 1
    assert track.percussion[0].key == NOISE_TO_GM[9]
    assert track.percussion[0].source_value == 9
    rebuilt, _ = score_to_composition(score, row_sec=ROW_SEC)
    noise = [e for e in rebuilt.sounding_events() if isinstance(e, Noise)]
    assert len(noise) == 1
    assert noise[0].value == 9


def test_nes_tracks_are_marked_monophonic():
    """The representation must carry the constraint, since a general model's
    tracks are not monophonic and the arranger needs to know the difference."""
    piece = _piece(_notes("P1", [(0, 60)]))
    score, _ = composition_to_score(piece, ticks_per_quarter=24, tempo_bpm=150.0, row_sec=ROW_SEC)
    assert all(t.monophonic for t in score.tracks)


# -- rapid arpeggios, the NES-specific case ----------------------------------


def test_a_rapid_arpeggio_survives_as_separate_sequential_notes():
    """An NES chord is one channel cycling pitches within a row. It must come
    back as the same sequence on one track -- not merged into simultaneous
    notes, which would change its musical meaning and imply polyphony the
    hardware cannot play."""
    piece = _piece(_notes("P1", [(0, 60), (30, 64), (60, 67)], duration=25))
    score, loss = composition_to_score(
        piece, ticks_per_quarter=24, tempo_bpm=150.0, row_sec=ROW_SEC
    )
    assert loss.n_onset_collisions == 0
    track = score.track_for("P1")
    assert track is not None
    ticks = [n.tick for n in track.notes]
    assert len(set(ticks)) == 3, "the three pitches must stay at distinct onsets"
    assert [n.pitch for n in sorted(track.notes, key=lambda n: n.tick)] == [60, 64, 67]


def test_overlapping_notes_in_a_general_score_are_assigned_to_one_channel():
    """A pretrained model will emit chords. They land on one NES channel and
    the conflict is the arranger's to resolve, so the conversion must not
    quietly drop the extra voices here."""
    score = Score(
        ticks_per_quarter=24,
        tempo_bpm=150.0,
        tracks=[
            Track(
                role="melody",
                channel_id="P1",
                notes=[
                    ScoreNote(tick=0, duration=24, pitch=60),
                    ScoreNote(tick=0, duration=24, pitch=64),
                    ScoreNote(tick=0, duration=24, pitch=67),
                ],
            )
        ],
    )
    rebuilt, loss = score_to_composition(score, row_sec=ROW_SEC)
    assert loss.n_events_out == 3, "all three reach the arranger"
    assert len({e.tick for e in rebuilt.sounding_events()}) == 1


def test_percussion_with_an_unmapped_key_is_counted_not_dropped_silently():
    score = Score(
        ticks_per_quarter=24,
        tempo_bpm=150.0,
        tracks=[
            Track(
                role=PERCUSSION,
                channel_id="NO",
                percussion=[ScorePercussion(tick=0, duration=6, key=99)],
            )
        ],
    )
    _rebuilt, loss = score_to_composition(score, row_sec=ROW_SEC)
    assert loss.n_percussion_shifted_lost == 1
    assert loss.notes


def test_gm_to_noise_is_the_exact_inverse():
    assert GM_TO_NOISE == {gm: p for p, gm in NOISE_TO_GM.items()}


def test_conversion_is_deterministic():
    piece = _piece(_notes("P1", [(0, 60), (30, 64)]) + _notes("TR", [(0, 43)]))
    first, _ = composition_to_score(piece, ticks_per_quarter=24, tempo_bpm=150.0, row_sec=ROW_SEC)
    second, _ = composition_to_score(piece, ticks_per_quarter=24, tempo_bpm=150.0, row_sec=ROW_SEC)
    assert first.to_dict() == second.to_dict()
