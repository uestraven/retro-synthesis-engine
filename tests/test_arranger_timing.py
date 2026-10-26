"""Tempo must not move the notes.

These lock the fix for the defect in `docs/lead_continuity_diagnosis.md`: the
arranger scaled every onset and duration by `score.tempo_bpm / 150`, so eight
composed bars became `8 * tempo_bpm / 150` arranged bars -- four at 76 BPM,
ten at 180. The lead only appeared to decay because a fixed eight-bar
measurement window was being read over a piece that had already ended.

`test_bar_count_is_tempo_invariant` is the fixture that fails under the old
scaling: it asserts eight bars in and eight bars out at 90, 120, 150, 180 and
210 BPM, and the old code produced 4.8, 6.4, 8.0, 9.6 and 11.2.
"""

from __future__ import annotations

import pytest

from rse.events.schema import Note
from rse.hardware.constraints import ConstraintEngine
from rse.hardware.profile import load_profile
from rse.transfer.arranger import ROWS_PER_BEAT, arrange_for_nes, row_seconds
from rse.transfer.score import (
    BASS,
    MELODY,
    PERCUSSION,
    Score,
    ScoreNote,
    ScorePercussion,
    Track,
)

#: Fermata's resolution, which is what the conversion has to come from.
SOURCE_TPQ = 24
TICKS_PER_BEAT = 480
BEATS_PER_BAR = 4
BARS = 8

TEMPOS = (90.0, 120.0, 150.0, 180.0, 210.0)


def build_score(
    *, tempo_bpm: float, bars: int = BARS, beats_per_bar: int = BEATS_PER_BAR
) -> Score:
    """Eight bars of quarter notes on a lead, a bass, and a drum per bar.

    Deliberately plain: a scale on the lead so ordering is checkable by
    pitch, a low sustained bass so register-based role assignment is
    unambiguous, and one drum hit per bar.
    """
    ticks_per_bar = SOURCE_TPQ * beats_per_bar
    lead = Track(role=MELODY, program=80, monophonic=True)
    bass = Track(role=BASS, program=38, monophonic=True)
    drums = Track(role=PERCUSSION, percussion=[])
    for bar in range(bars):
        for beat in range(beats_per_bar):
            tick = bar * ticks_per_bar + beat * SOURCE_TPQ
            lead.notes.append(
                ScoreNote(tick=tick, duration=SOURCE_TPQ, pitch=60 + (bar * 4 + beat) % 12)
            )
        bass.notes.append(
            ScoreNote(tick=bar * ticks_per_bar, duration=ticks_per_bar, pitch=36)
        )
        drums.percussion.append(
            ScorePercussion(tick=bar * ticks_per_bar, duration=1, key=36)
        )
    return Score(
        ticks_per_quarter=SOURCE_TPQ,
        tempo_bpm=tempo_bpm,
        beats_per_bar=beats_per_bar,
        tracks=[lead, bass, drums],
    )


def arranged_bars(composition, *, beats_per_bar: int = BEATS_PER_BAR) -> float:
    """Bars spanned by the arrangement, from the grid rather than the clock."""
    events = composition.sounding_events()
    if not events:
        return 0.0
    ticks_per_bar = TICKS_PER_BEAT * beats_per_bar
    return (max(e.tick for e in events) + 1) / ticks_per_bar


@pytest.mark.parametrize("tempo", TEMPOS)
def test_bar_count_is_tempo_invariant(tempo: float) -> None:
    """Eight composed bars are eight arranged bars at every tempo.

    The regression fixture. Under `scale = tempo_bpm / 150` this produced
    4.8, 6.4, 8.0, 9.6 and 11.2 bars respectively.
    """
    score = build_score(tempo_bpm=tempo)
    composition, _report = arrange_for_nes(
        score, row_sec=row_seconds(tempo), divisions=4
    )
    # The last onset sits in bar 8, so the span is just over 7 bars and at
    # most 8. Anything outside that means the music was rescaled.
    assert 7.0 < arranged_bars(composition) <= 8.0, (
        f"{tempo} BPM produced {arranged_bars(composition):.2f} bars"
    )


@pytest.mark.parametrize("tempo", TEMPOS)
def test_onsets_land_on_the_same_musical_positions(tempo: float) -> None:
    """Every source onset maps to source_ticks * 480/24, independent of tempo."""
    score = build_score(tempo_bpm=tempo)
    composition, _report = arrange_for_nes(
        score, row_sec=row_seconds(tempo), divisions=4
    )
    lead = sorted(
        (e for e in composition.sounding_events("P1") if isinstance(e, Note)),
        key=lambda e: e.tick,
    )
    expected = [
        n.tick * (TICKS_PER_BEAT // SOURCE_TPQ)
        for n in sorted(score.tracks[0].notes, key=lambda n: n.tick)
    ]
    assert [e.tick for e in lead] == expected


@pytest.mark.parametrize("tempo", TEMPOS)
def test_durations_scale_musically(tempo: float) -> None:
    """A quarter note is 480 destination ticks whatever the tempo."""
    score = build_score(tempo_bpm=tempo)
    composition, _report = arrange_for_nes(
        score, row_sec=row_seconds(tempo), divisions=4
    )
    lead = [e for e in composition.sounding_events("P1") if isinstance(e, Note)]
    assert lead
    assert {e.duration for e in lead} == {TICKS_PER_BEAT}


@pytest.mark.parametrize("tempo", TEMPOS)
def test_playback_tempo_is_the_requested_tempo(tempo: float) -> None:
    """Real time comes from the tempo map, and eight bars last 8*4*60/bpm."""
    score = build_score(tempo_bpm=tempo)
    composition, _report = arrange_for_nes(
        score, row_sec=row_seconds(tempo), divisions=4
    )
    assert composition.tempo_map
    # Microseconds per beat is an integer, so the round trip is lossy by up
    # to half a microsecond: 210 BPM stores as 285714 and reads back as
    # 210.00021. The stored integer is the exact assertion; the derived BPM
    # gets a tolerance that admits that rounding and nothing larger.
    assert composition.tempo_map[0].us_per_beat == round(60_000_000 / tempo)
    played = 60_000_000 / composition.tempo_map[0].us_per_beat
    assert played == pytest.approx(tempo, rel=1e-5)
    # Eight bars of 4/4 at `tempo`, measured through the composition's clock.
    eight_bars = TICKS_PER_BEAT * BEATS_PER_BAR * BARS
    assert composition.time.seconds(eight_bars) == pytest.approx(
        BARS * BEATS_PER_BAR * 60.0 / tempo, rel=1e-3
    )


@pytest.mark.parametrize("tempo", TEMPOS)
def test_duration_falls_as_tempo_rises(tempo: float) -> None:
    """Sanity on direction. The defect made duration *rise* with tempo."""
    score = build_score(tempo_bpm=tempo)
    composition, _report = arrange_for_nes(
        score, row_sec=row_seconds(tempo), divisions=4
    )
    eight_bars = TICKS_PER_BEAT * BEATS_PER_BAR * BARS
    seconds = composition.time.seconds(eight_bars)
    assert seconds == pytest.approx(1920.0 / tempo, rel=1e-3)


@pytest.mark.parametrize("tempo", TEMPOS)
def test_nothing_is_dropped_or_duplicated(tempo: float) -> None:
    score = build_score(tempo_bpm=tempo)
    composition, report = arrange_for_nes(
        score, row_sec=row_seconds(tempo), divisions=4
    )
    expected_in = sum(len(t.notes) + len(t.percussion) for t in score.tracks)
    assert report.n_events_in == expected_in
    assert report.n_out_of_range_dropped == 0
    assert report.n_polyphony_dropped == 0
    assert report.n_events_out == expected_in
    assert len(composition.sounding_events()) == expected_in


@pytest.mark.parametrize("tempo", TEMPOS)
def test_channel_identity_and_ordering_survive(tempo: float) -> None:
    """The lead keeps its pitch sequence, on one channel, in order."""
    score = build_score(tempo_bpm=tempo)
    composition, _report = arrange_for_nes(
        score, row_sec=row_seconds(tempo), divisions=4
    )
    source = [n.pitch for n in sorted(score.tracks[0].notes, key=lambda n: n.tick)]
    lead = sorted(
        (e for e in composition.sounding_events("P1") if isinstance(e, Note)),
        key=lambda e: e.tick,
    )
    assert [e.pitch for e in lead] == source
    assert {e.channel_id for e in lead} == {"P1"}
    # Percussion stays on the noise channel, one hit per bar.
    assert len(composition.sounding_events("NO")) == BARS


@pytest.mark.parametrize("tempo", TEMPOS)
def test_cross_bar_sustain_survives(tempo: float) -> None:
    """A bass note spanning a whole bar still spans a whole bar."""
    score = build_score(tempo_bpm=tempo)
    composition, _report = arrange_for_nes(
        score, row_sec=row_seconds(tempo), divisions=4
    )
    bass = [e for e in composition.sounding_events("TR") if isinstance(e, Note)]
    assert bass
    assert {e.duration for e in bass} == {TICKS_PER_BEAT * BEATS_PER_BAR}


@pytest.mark.parametrize("tempo", TEMPOS)
def test_hardware_constraints_still_pass(tempo: float) -> None:
    score = build_score(tempo_bpm=tempo)
    composition, _report = arrange_for_nes(
        score, row_sec=row_seconds(tempo), divisions=4
    )
    engine = ConstraintEngine(load_profile("nes"))
    assert engine.validate(composition) == []


@pytest.mark.parametrize("beats_per_bar", [3, 4, 6])
def test_other_meters_keep_their_bar_count(beats_per_bar: int) -> None:
    """Bar invariance is about bars, not about 4/4."""
    score = build_score(tempo_bpm=120.0, beats_per_bar=beats_per_bar)
    composition, _report = arrange_for_nes(
        score, row_sec=row_seconds(120.0), divisions=4
    )
    spanned = arranged_bars(composition, beats_per_bar=beats_per_bar)
    assert 7.0 < spanned <= 8.0, f"{beats_per_bar}/4 produced {spanned:.2f} bars"


def test_row_seconds_matches_the_grid() -> None:
    """`row_seconds` is the inverse of the tokenizer's `fit.bpm`."""
    for tempo in TEMPOS:
        period = row_seconds(tempo)
        assert 60.0 / (period * ROWS_PER_BEAT) == pytest.approx(tempo, rel=1e-9)
    with pytest.raises(ValueError):
        row_seconds(0.0)


def test_old_scaling_would_have_failed_bar_invariance() -> None:
    """Documents the defect numerically, so the fixture's purpose is legible.

    Not a test of current behaviour -- it asserts what the removed formula
    computed, so that a future reader can see why the parametrised bar-count
    test exists at these particular tempos.
    """
    for tempo, expected in zip(TEMPOS, (4.8, 6.4, 8.0, 9.6, 11.2), strict=True):
        old_scale = tempo / 150.0
        assert BARS * old_scale == pytest.approx(expected, rel=1e-9)
