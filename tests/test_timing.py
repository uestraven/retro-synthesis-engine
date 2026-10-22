from __future__ import annotations

import pytest

from rse.data.timing import MusicalTime, TempoChange, TimeSignatureChange


def test_defaults_are_inserted_when_the_file_declares_nothing():
    time = MusicalTime(480)
    assert time.tempo_changes[0].us_per_beat == 500_000
    assert time.tempo_changes[0].bpm == pytest.approx(120.0)
    assert time.time_signatures[0].numerator == 4
    assert time.is_constant_tempo and time.is_constant_meter


def test_rejects_nonpositive_resolution():
    with pytest.raises(ValueError, match="ticks_per_beat"):
        MusicalTime(0)


def test_beats_and_bars_in_common_time():
    time = MusicalTime(480)
    assert time.beat(960) == pytest.approx(2.0)
    assert time.bar_position(0) == (0, 0.0)
    assert time.bar_position(480 * 4) == (1, 0.0)
    bar, position = time.bar_position(480 * 6)
    assert (bar, position) == (1, pytest.approx(2.0))


def test_bars_follow_a_meter_change():
    # 4/4 for two bars, then 3/4.
    time = MusicalTime(
        480,
        time_signatures=[
            TimeSignatureChange(0, 4, 4),
            TimeSignatureChange(480 * 8, 3, 4),
        ],
    )
    assert time.bar_position(480 * 8) == (2, 0.0)
    assert time.bar_position(480 * 11) == (3, 0.0)  # one 3/4 bar later
    assert time.time_signature_at(480 * 9).numerator == 3


def test_seconds_integrate_a_tempo_change():
    # 120 BPM for 4 beats (2s), then 60 BPM.
    time = MusicalTime(
        480,
        tempo_changes=[TempoChange(0, 500_000), TempoChange(480 * 4, 1_000_000)],
    )
    assert time.seconds(480 * 4) == pytest.approx(2.0)
    assert time.seconds(480 * 6) == pytest.approx(4.0)
    assert not time.is_constant_tempo
    assert time.tempo_at(480 * 5).bpm == pytest.approx(60.0)


def test_tempo_declared_late_keeps_a_default_at_tick_zero():
    time = MusicalTime(480, tempo_changes=[TempoChange(960, 1_000_000)])
    assert time.tempo_at(0).bpm == pytest.approx(120.0)
    assert time.seconds(960) == pytest.approx(1.0)
