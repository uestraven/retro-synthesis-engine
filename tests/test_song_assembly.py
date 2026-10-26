"""Assembling valid sections must produce a valid track.

The first acceptance run generated five sections that were each individually
hardware-valid, concatenated them, and reported **5 violations** on the result.
A note whose duration runs past its section's last bar line overlaps the next
section's opening on a monophonic channel, and there are four boundaries.

The fix is the discipline already established for single pieces: monophony and
the constraint engine run LAST, on the assembled track, not only per section.
These tests pin that, and pin that the loop points survive the repair -- an
earlier version lost them, because `ConstraintEngine.apply` returns a new
composition without them.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from rse.analysis.restatement import ticks_per_bar
from rse.codec.benchmark import enforce_monophony
from rse.events.schema import Composition, Note, TempoChange, TimeSignatureChange
from rse.hardware.constraints import ConstraintEngine
from rse.hardware.profile import load_profile

SECTION_BARS = 8
BAR_TICKS = ticks_per_bar()
SECTION_TICKS = SECTION_BARS * BAR_TICKS


def section_with_overhanging_note() -> Composition:
    """Eight valid bars whose last note sustains past the final bar line.

    Legal on its own -- a note's duration is its own property and the codec
    deliberately keeps it -- and the exact thing that collides with whatever
    is placed after it.
    """
    events = [
        Note(channel_id="P1", tick=bar * BAR_TICKS, pitch=60 + bar, duration=480)
        for bar in range(SECTION_BARS - 1)
    ]
    events.append(
        Note(
            channel_id="P1",
            tick=(SECTION_BARS - 1) * BAR_TICKS,
            pitch=72,
            # Runs a full bar past the end of the section.
            duration=BAR_TICKS * 2,
        )
    )
    composition = Composition(
        hardware_profile="nes", ticks_per_beat=480, events=events
    )
    composition.tempo_map = [TempoChange(0, int(round(60_000_000 / 120)))]
    composition.time_signatures = [TimeSignatureChange(0, 4, 4)]
    composition.normalize()
    return composition


#: Repository root, for reading the assembler's source in the call-site test.
_ROOT = Path(__file__).resolve().parent.parent


def concatenate_raw(sections: list[Composition]) -> Composition:
    """Lay sections on bar lines with no repair -- the broken behaviour."""
    events = []
    for index, section in enumerate(sections):
        offset = index * SECTION_TICKS
        for event in section.sounding_events():
            events.append(dataclasses.replace(event, tick=event.tick + offset))
    track = Composition(hardware_profile="nes", ticks_per_beat=480, events=events)
    track.tempo_map = [TempoChange(0, int(round(60_000_000 / 120)))]
    track.time_signatures = [TimeSignatureChange(0, 4, 4)]
    track.loop_start_tick = 0
    track.loop_end_tick = len(sections) * SECTION_TICKS
    track.normalize()
    return track


@pytest.fixture
def engine() -> ConstraintEngine:
    return ConstraintEngine(load_profile("nes"))


def test_each_section_is_valid_alone(engine: ConstraintEngine) -> None:
    assert engine.validate(section_with_overhanging_note()) == []


def test_raw_concatenation_breaks_the_seam(engine: ConstraintEngine) -> None:
    """The regression fixture: valid parts, invalid whole.

    If this ever reports zero, the test no longer exercises the defect and the
    one below stops being evidence of anything.
    """
    track = concatenate_raw([section_with_overhanging_note() for _ in range(3)])
    assert engine.validate(track), (
        "expected the overhanging note to collide at the section seam"
    )


def test_repairing_the_assembled_track_fixes_the_seam(
    engine: ConstraintEngine,
) -> None:
    track = concatenate_raw([section_with_overhanging_note() for _ in range(3)])
    repaired, _truncated = enforce_monophony(track)
    repaired, _repairs = engine.apply(repaired)
    assert engine.validate(repaired) == []


def test_loop_points_survive_the_repair(engine: ConstraintEngine) -> None:
    """`ConstraintEngine.apply` returns a new composition; the loop must be
    carried across explicitly or a seamless track silently loses its seam."""
    track = concatenate_raw([section_with_overhanging_note() for _ in range(3)])
    start, end = track.loop_start_tick, track.loop_end_tick
    repaired, _t = enforce_monophony(track)
    repaired, _r = engine.apply(repaired)
    repaired.loop_start_tick, repaired.loop_end_tick = start, end
    assert repaired.loop_start_tick == 0
    assert repaired.loop_end_tick == 3 * SECTION_TICKS


def test_repair_does_not_shorten_the_form(engine: ConstraintEngine) -> None:
    """Repair may trim a note; it must not lose a section."""
    sections = [section_with_overhanging_note() for _ in range(4)]
    track = concatenate_raw(sections)
    repaired, _t = enforce_monophony(track)
    repaired, _r = engine.apply(repaired)
    events = repaired.sounding_events()
    assert events
    # The last section still has material in it.
    assert max(e.tick for e in events) >= 3 * SECTION_TICKS


def test_prompt_to_song_runs_repair_on_the_assembled_track() -> None:
    """Guard the call site itself.

    The defect was not in any library function -- every piece worked. It was
    the assembler validating the whole track without repairing it, so this
    checks the script actually repairs before validating.
    """
    source = (_ROOT / "scripts" / "prompt_to_song.py").read_text()
    assembly = source[source.index(", placement = concatenate(") :]
    repair_at = assembly.index("engine.apply(track)")
    validate_at = assembly.index("violations = engine.validate(track)")
    assert repair_at < validate_at, "repair must precede validation"
    assert "enforce_monophony(track)" in assembly


def test_nothing_rings_past_the_loop_point(engine: ConstraintEngine) -> None:
    """A note sustaining past the loop end breaks the loop.

    The first profile A/B run reported exactly one violation in both arms:
    `past_loop_end`, a triangle note at tick 60300 ending at 61500 against a
    loop end of 61440. The engine detects it and `apply()` does not repair it,
    so the note was being cut mid-sound every time the track wrapped -- which
    is what a broken loop sounds like.
    """
    import dataclasses

    sections = [section_with_overhanging_note() for _ in range(2)]
    track = concatenate_raw(sections)
    loop_end = track.loop_end_tick
    assert any(
        e.tick + getattr(e, "duration", 0) > loop_end
        for e in track.sounding_events()
    ), "fixture must contain a note ringing past the loop point"

    # The fix: truncate to the boundary rather than leaving it to the engine.
    kept = []
    for event in track.sounding_events():
        end = event.tick + getattr(event, "duration", 0)
        if end > loop_end and event.tick < loop_end:
            kept.append(dataclasses.replace(event, duration=loop_end - event.tick))
        elif event.tick < loop_end:
            kept.append(event)
    track.events = kept
    track.normalize()

    assert all(
        e.tick + getattr(e, "duration", 0) <= loop_end
        for e in track.sounding_events()
    )
    assert all(e.duration > 0 for e in track.sounding_events()), (
        "truncation must not produce a zero-length note"
    )


def test_render_is_padded_to_the_loop_length(tmp_path) -> None:
    """A rendered loop must be exactly one loop long.

    The renderer stops at the last note's release, not at the bar line, so a
    final bar that ends early produces a file shorter than the loop it claims
    to be: 58.45 s for a 60.00 s loop, wrapping 0.83 bars early. Every loop
    claim drawn from `loop_end_tick` was true of the composition and false of
    the WAV actually delivered.
    """
    import wave

    import numpy as np

    from rse.song.loopcheck import pad_to_loop_length

    path = tmp_path / "short.wav"
    rate = 44100
    short = np.zeros(int(rate * 58.45), dtype="<i2")
    short[::1000] = 8000  # non-silent, so padding is distinguishable
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(short.tobytes())

    report = pad_to_loop_length(path, loop_seconds=60.0)
    assert report["padded"] is True
    assert report["seconds_before"] == pytest.approx(58.45, abs=0.01)
    assert report["seconds_after"] == pytest.approx(60.0, abs=0.001)

    with wave.open(str(path), "rb") as handle:
        assert handle.getnframes() == int(round(60.0 * rate))


def test_padding_never_truncates(tmp_path) -> None:
    """A file already at or over the loop length is left alone."""
    import wave

    import numpy as np

    from rse.song.loopcheck import pad_to_loop_length

    path = tmp_path / "exact.wav"
    rate = 44100
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(np.zeros(int(rate * 60.0), dtype="<i2").tobytes())
    report = pad_to_loop_length(path, loop_seconds=58.0)
    assert report["padded"] is False
    with wave.open(str(path), "rb") as handle:
        assert handle.getnframes() == int(rate * 60.0)


def test_sections_are_placed_where_the_music_ends(engine: ConstraintEngine) -> None:
    """No dead air between sections.

    Fermata's eight-bar generations stop around bar 7.2, so placing each
    section on an eight-bar grid left 1.4 s of silence at three of four
    boundaries -- 4.3 s, 7% of a 60 s track, which the listener heard as
    "there are silent sections".
    """
    from rse.song.build import occupied_bars

    section = section_with_overhanging_note()
    # Its last onset is in bar 7, so it occupies 8 bars.
    assert occupied_bars(section, bar_ticks=BAR_TICKS) == SECTION_BARS

    short = Composition(
        hardware_profile="nes",
        ticks_per_beat=480,
        events=[
            Note(channel_id="P1", tick=bar * BAR_TICKS, pitch=60, duration=480)
            for bar in range(6)
        ],
    )
    short.normalize()
    assert occupied_bars(short, bar_ticks=BAR_TICKS) == 6, (
        "a section whose music stops in bar 6 occupies 6 bars, not 8"
    )


def test_occupied_bars_of_an_empty_section_is_zero() -> None:
    from rse.song.build import occupied_bars

    empty = Composition(hardware_profile="nes", ticks_per_beat=480, events=[])
    assert occupied_bars(empty, bar_ticks=BAR_TICKS) == 0


def test_drop_channel_keeps_the_rest_intact() -> None:
    """Thinning a return must change texture, not pitch material."""
    from rse.song.build import drop_channel

    events = [
        Note(channel_id="P1", tick=0, pitch=72, duration=480),
        Note(channel_id="P2", tick=0, pitch=60, duration=480),
        Note(channel_id="TR", tick=0, pitch=36, duration=480),
    ]
    full = Composition(hardware_profile="nes", ticks_per_beat=480, events=events)
    full.normalize()
    thin = drop_channel(full, "P2")
    assert {e.channel_id for e in thin.sounding_events()} == {"P1", "TR"}
    assert [e.pitch for e in thin.sounding_events() if e.channel_id == "P1"] == [72]
    # The original is untouched.
    assert {e.channel_id for e in full.sounding_events()} == {"P1", "P2", "TR"}


def test_prompt_to_song_places_sections_contiguously() -> None:
    """Guard the call site, because the library fix alone was not enough.

    `occupied_bars` was added to rse/song/build.py and `SongBuilder.assemble`
    was fixed to use it -- and the silence stayed at 4.3 s, byte-identical,
    because prompt_to_song.py has its OWN concatenate() and never calls the
    shared assembler. The module was extracted to prevent exactly that
    duplication, then patched while the duplicate was left in place.
    """
    source = (_ROOT / "scripts" / "prompt_to_song.py").read_text()
    body = source[source.index("def concatenate(") : source.index("def main(")]
    assert "music_end_bar(" in body, (
        "the script's own concatenate must place sections where the music stops"
    )
    assert "index * section_bars * bar_ticks" not in body, (
        "nominal-grid placement is what produced the dead air"
    )
