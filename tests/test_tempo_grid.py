"""Tempo on the NES is a frame count, not a free-running BPM.

The console has no tempo register: a music driver runs from the NMI once per
video frame and advances the tracker one row every whole number of frames. So
with four rows to the beat, only a ladder of tempos exists.

RSE composed in float seconds and produced tempos off that ladder. `action`
asked for 168 BPM, which is 5.37 frames per row -- 7% from anything real --
and because every layer agreed with that float, nothing flagged it. The
renderer played a tempo no console could output and the report presented it as
if it could.

These pin the grid, the snapping, and the invariant that matters: a Style's
reported BPM is always the BPM its frame count produces.
"""

from __future__ import annotations

import pytest

from rse.hardware.tempo import (
    MAX_FRAMES_PER_ROW,
    MIN_FRAMES_PER_ROW,
    NTSC_FRAME_RATE,
    PAL_FRAME_RATE,
    ROWS_PER_BEAT,
    Tempo,
    bpm_for_frames,
    frames_for_bpm,
    is_playable,
    ladder,
    row_seconds,
    snap,
)


def test_the_ntsc_frame_rate_is_the_real_one() -> None:
    """60 Hz is the approximation; the NES runs at 60.0988."""
    assert NTSC_FRAME_RATE == pytest.approx(60.0988, abs=1e-4)
    assert PAL_FRAME_RATE == pytest.approx(50.0070, abs=1e-4)


@pytest.mark.parametrize(
    ("frames", "bpm"),
    [
        (4, 225.37),
        (5, 180.30),
        (6, 150.25),
        (7, 128.78),
        (8, 112.69),
        (9, 100.16),
        (10, 90.15),
        (12, 75.12),
    ],
)
def test_the_tempo_ladder(frames: int, bpm: float) -> None:
    assert bpm_for_frames(frames) == pytest.approx(bpm, abs=0.01)


def test_six_frames_per_row_is_the_classic_tempo() -> None:
    """The 0.1 s row that the corpus work assumed: 6 frames, ~150 BPM."""
    assert row_seconds(6) == pytest.approx(0.0998, abs=1e-4)
    assert bpm_for_frames(6) == pytest.approx(150.25, abs=0.01)


def test_frames_and_bpm_are_inverses() -> None:
    for frames in range(MIN_FRAMES_PER_ROW, MAX_FRAMES_PER_ROW + 1):
        assert frames_for_bpm(bpm_for_frames(frames)) == pytest.approx(
            frames, abs=1e-9
        )


def test_the_ladder_is_ordered_and_complete() -> None:
    steps = ladder()
    assert [s.frames_per_row for s in steps] == list(
        range(MIN_FRAMES_PER_ROW, MAX_FRAMES_PER_ROW + 1)
    )
    # More frames per row is always slower.
    assert [s.bpm for s in steps] == sorted((s.bpm for s in steps), reverse=True)


@pytest.mark.parametrize(
    ("asked", "frames"),
    [(168, 5), (170, 5), (150, 6), (144, 6), (137, 7), (128, 7), (104, 9), (90, 10)],
)
def test_snapping_lands_on_the_grid(asked: float, frames: int) -> None:
    assert snap(asked).frames_per_row == frames
    assert is_playable(snap(asked).bpm)


def test_snapping_rounds_frames_not_bpm() -> None:
    """The ladder is not evenly spaced -- 180.30 to 150.25 is 30 BPM while
    100.16 to 90.15 is 10 -- so rounding in BPM space would bias toward the
    fast end. Rounding the frame count is what the driver would do."""
    # 165 BPM: 5.46 frames. Nearer 5 in frames; in BPM it is nearer 150.25
    # (14.75 away) than 180.30 (15.30 away), so the two disagree here.
    assert frames_for_bpm(165) == pytest.approx(5.46, abs=0.01)
    assert snap(165).frames_per_row == 5
    assert snap(165).bpm == pytest.approx(180.30, abs=0.01)


def test_unplayable_tempos_are_reported_as_such() -> None:
    assert not is_playable(168)
    assert not is_playable(170)
    assert not is_playable(120)
    assert is_playable(150.25)
    assert is_playable(128.78)


def test_snapping_is_clamped_to_the_playable_range() -> None:
    assert snap(10_000).frames_per_row == MIN_FRAMES_PER_ROW
    assert snap(1).frames_per_row == MAX_FRAMES_PER_ROW


def test_a_tempo_rejects_nonsense() -> None:
    with pytest.raises(ValueError):
        bpm_for_frames(0)
    with pytest.raises(ValueError):
        frames_for_bpm(0)


def test_rows_per_beat_changes_the_ladder() -> None:
    """Eight rows to the beat halves every tempo at the same frame count."""
    assert bpm_for_frames(6, rows_per_beat=8) == pytest.approx(
        bpm_for_frames(6, rows_per_beat=4) / 2, abs=0.01
    )
    assert ROWS_PER_BEAT == 4


def test_pal_is_a_different_ladder() -> None:
    """A PAL console runs slower, so the same frame count is a slower tempo.
    This is why PAL releases of NES games play their music flat."""
    assert bpm_for_frames(6, frame_rate=PAL_FRAME_RATE) == pytest.approx(
        125.02, abs=0.01
    )


def test_tempo_renders_both_units() -> None:
    assert str(Tempo(frames_per_row=6)) == "150.25 BPM (6 frames/row)"
    assert Tempo(frames_per_row=6).to_dict() == {
        "frames_per_row": 6,
        "bpm": 150.25,
        "row_ms": 99.836,
    }


# --- the invariant that matters -------------------------------------------


def test_every_preset_is_playable() -> None:
    """The defect: four of six presets asked for tempos no NES could play."""
    from rse.procedural.compose import STYLES

    for name, style in STYLES.items():
        assert is_playable(style.tempo_bpm), (
            f"{name} asks for {style.tempo_bpm} BPM, which is "
            f"{frames_for_bpm(style.tempo_bpm):.2f} frames per row"
        )


def test_a_style_reports_the_bpm_its_frame_count_produces() -> None:
    """Report and music cannot disagree, because BPM is derived from frames."""
    from rse.procedural.compose import STYLES

    for name, style in STYLES.items():
        assert style.tempo_bpm == pytest.approx(
            bpm_for_frames(style.frames_per_row), abs=0.01
        ), name
        assert style.row_sec == pytest.approx(
            row_seconds(style.frames_per_row), abs=1e-9
        )


def test_an_override_is_snapped_by_the_same_path_as_a_preset() -> None:
    """`dataclasses.replace` runs `__post_init__`, so a tempo from a prompt or
    a slider lands on the grid the same way a preset does."""
    import dataclasses

    from rse.procedural.compose import STYLES

    style = dataclasses.replace(STYLES["boss"], tempo_bpm=170)
    assert style.frames_per_row == 5
    assert style.tempo_bpm == pytest.approx(180.30, abs=0.01)
    assert is_playable(style.tempo_bpm)


def test_the_composition_plays_the_tempo_it_reports() -> None:
    """End to end: the tempo map must match the frame count, since this is the
    class of bug where every layer agrees with a wrong number."""
    from rse.procedural.compose import TICKS_PER_BAR, compose

    for style_name in ("boss", "action", "tension"):
        result = compose(style_name=style_name, seed=3, target_seconds=30.0)
        composition = result.composition
        sec_per_bar = composition.time.seconds(TICKS_PER_BAR) - (
            composition.time.seconds(0)
        )
        assert sec_per_bar == pytest.approx(
            4 * 60.0 / result.style.tempo_bpm, abs=1e-5
        ), style_name
        # The real constraint: a bar is a whole number of video frames. The
        # tempo map stores microseconds per beat as an integer, so allow the
        # half-microsecond that rounding costs -- about 1e-4 frames.
        frames = sec_per_bar * NTSC_FRAME_RATE
        expected = 16 * result.style.frames_per_row
        assert frames == pytest.approx(expected, abs=1e-3), (
            f"{style_name}: {frames} frames per bar, want {expected}"
        )


# --- the two bugs a browser found -----------------------------------------


def test_an_energy_control_does_not_forge_a_tempo_request() -> None:
    """`select()` applies the prompt's tempo only when the prompt carried an
    energy word, reasoning that such a tempo was asked for. Setting
    `plan.energy` from an energy SLIDER forged that signal, so the
    interpreter's unrequested 120 BPM default overrode every preset: a boss
    theme at "style default" played 112.69 BPM instead of 150.25.
    """
    from rse.procedural.compose import STYLES
    from rse.song.generate import generate_song

    preset = STYLES["boss"].tempo_bpm
    for energy in (None, "very_low", "medium", "very_high"):
        result = generate_song(
            "a dark boss battle theme", seed=7, energy=energy, target_seconds=60.0
        )
        assert result.meta["tempo_bpm"] == pytest.approx(preset, abs=0.01), (
            f"energy={energy!r} moved the tempo off the preset"
        )
        assert result.meta["frames_per_row"] == STYLES["boss"].frames_per_row


def test_a_tempo_the_prompt_names_is_still_honoured() -> None:
    """The fix must not go too far: a literal BPM in the prompt is a request,
    and it sets no energy, so testing energy alone dropped it."""
    from rse.song.generate import generate_song

    result = generate_song(
        "a battle theme at 200 bpm", seed=7, energy="high", target_seconds=60.0
    )
    # 200 BPM is 4.51 frames a row, so it snaps to 5 -> 180.30.
    assert result.meta["frames_per_row"] == 5
    assert result.meta["tempo_bpm"] == pytest.approx(180.30, abs=0.01)


def test_a_prompt_tempo_that_gets_snapped_says_so() -> None:
    """Reported on every route. It was reported only on the caller's route at
    first, so a tempo the prompt asked for was snapped in silence."""
    from rse.song.generate import generate_song

    result = generate_song(
        "a battle theme at 200 bpm", seed=7, target_seconds=60.0
    )
    notes = " ".join(result.meta["selection"]["notes"])
    assert "200 BPM is not playable" in notes
    assert "180.30" in notes


def test_no_report_claims_an_unplayable_tempo() -> None:
    """Whatever route a tempo takes, the number shown is the number played."""
    from rse.song.generate import finish_report, generate_song

    for kwargs in (
        {},
        {"tempo_bpm": 170},
        {"energy": "high"},
        {"tempo_bpm": 200, "energy": "low"},
    ):
        result = generate_song(
            "a dark boss battle theme", seed=7, target_seconds=60.0, **kwargs
        )
        played = result.meta["tempo_bpm"]
        assert is_playable(played), kwargs
        report = finish_report(result)
        shown = [
            item
            for bucket in report.values()
            for item in bucket
            if "chosen directly" in item["detail"]
        ]
        for item in shown:
            assert f"{played:.2f}" in item["detail"] or str(
                round(played)
            ) in item["detail"], item
