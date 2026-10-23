from __future__ import annotations

import pickle

import numpy as np
import pytest

from rse.data.exprsco import (
    CC_TIMBRE,
    CC_VOLUME,
    CHANNEL_ORDER,
    FRAME_TICKS,
    TICKS_PER_BEAT,
    TRIANGLE_VELOCITY,
    ExprscoError,
    UnsafePickleError,
    load_exprsco,
    score_to_composition,
    split_from_path,
)
from rse.data.normalize import build_intake_record
from rse.data.provenance import describe_file
from rse.events.convert import composition_from_intake
from rse.tokenizers.tokenizer import RseTokenizer

P1, P2, TR, NO = 0, 1, 2, 3
PITCH, VELOCITY, TIMBRE = 0, 1, 2


def make_score(frames: np.ndarray, *, rate: float = 24.0) -> tuple:
    return (rate, int(frames.shape[0] * 44100 / rate), frames.astype(np.uint8))


def write_score(path, frames, *, rate: float = 24.0):
    with open(path, "wb") as handle:
        pickle.dump(make_score(frames, rate=rate), handle)
    return path


def frames_of(n: int) -> np.ndarray:
    return np.zeros((n, 4, 3), dtype=np.uint8)


@pytest.fixture
def source(tmp_path):
    def build(frames, *, rate: float = 24.0, name="s.pkl"):
        path = write_score(tmp_path / name, frames, rate=rate)
        return path, describe_file(
            path, root=path.parent, dataset_id="t", dataset_version="t/exprsco"
        )

    return build


# -- safe loading -------------------------------------------------------------


def test_loads_a_well_formed_score(source):
    frames = frames_of(4)
    frames[:, P1, PITCH] = 60
    path, _record = source(frames)
    score = load_exprsco(path)
    assert score.rate == 24.0
    assert score.n_frames == 4
    assert score.duration_sec == pytest.approx(4 / 24)


class _Exploit:
    """The classic pickle attack shape: a reduce that calls out to the shell."""

    def __reduce__(self):
        import os

        return (os.system, ("true",))


def test_refuses_a_pickle_that_would_execute_code(tmp_path):
    """A downloaded pickle must not be able to run anything."""
    path = tmp_path / "evil.pkl"
    with path.open("wb") as handle:
        pickle.dump(_Exploit(), handle)

    # Caught as a security refusal, and as an exprsco problem, and as an
    # unpickling problem, so no caller can miss it.
    with pytest.raises(UnsafePickleError, match="refused to load"):
        load_exprsco(path)
    with pytest.raises(ExprscoError):
        load_exprsco(path)
    with pytest.raises(pickle.UnpicklingError):
        load_exprsco(path)


def test_refuses_a_non_tuple_payload(tmp_path):
    path = tmp_path / "bad.pkl"
    with path.open("wb") as handle:
        pickle.dump(np.zeros((2, 4, 3), dtype=np.uint8), handle)
    with pytest.raises(ExprscoError, match="expected a 3-tuple"):
        load_exprsco(path)


def test_refuses_a_wrongly_shaped_array(tmp_path):
    path = tmp_path / "shape.pkl"
    with path.open("wb") as handle:
        pickle.dump((24.0, 100, np.zeros((5, 3), dtype=np.uint8)), handle)
    with pytest.raises(ExprscoError, match="shaped"):
        load_exprsco(path)


def test_refuses_a_file_that_is_not_a_pickle(tmp_path):
    path = tmp_path / "nope.pkl"
    path.write_bytes(b"this is not a pickle")
    with pytest.raises(ExprscoError):
        load_exprsco(path)


# -- note extraction ----------------------------------------------------------


def test_a_held_pitch_becomes_one_note(nes_profile, source):
    frames = frames_of(6)
    frames[:, P1, PITCH] = 60
    frames[:, P1, VELOCITY] = 15
    path, _ = source(frames)
    composition = score_to_composition(load_exprsco(path), path, nes_profile)
    notes = composition.notes_for("P1")
    assert len(notes) == 1
    assert notes[0].tick == 0
    assert notes[0].dur_ticks == 6 * FRAME_TICKS
    assert notes[0].pitch == 60


def test_a_pitch_change_starts_a_new_note(nes_profile, source):
    frames = frames_of(4)
    frames[:2, P1, PITCH] = 60
    frames[2:, P1, PITCH] = 64
    frames[:, P1, VELOCITY] = 10
    path, _ = source(frames)
    notes = score_to_composition(load_exprsco(path), path, nes_profile).notes_for("P1")
    assert [(n.tick, n.dur_ticks, n.pitch) for n in notes] == [
        (0, 2 * FRAME_TICKS, 60),
        (2 * FRAME_TICKS, 2 * FRAME_TICKS, 64),
    ]


def test_zero_pitch_is_silence(nes_profile, source):
    frames = frames_of(6)
    frames[2:4, P1, PITCH] = 60
    frames[2:4, P1, VELOCITY] = 8
    path, _ = source(frames)
    notes = score_to_composition(load_exprsco(path), path, nes_profile).notes_for("P1")
    assert len(notes) == 1
    assert notes[0].tick == 2 * FRAME_TICKS
    assert notes[0].dur_ticks == 2 * FRAME_TICKS


def test_a_note_still_sounding_at_the_end_is_closed(nes_profile, source):
    frames = frames_of(3)
    frames[:, P2, PITCH] = 72
    frames[:, P2, VELOCITY] = 12
    path, _ = source(frames)
    notes = score_to_composition(load_exprsco(path), path, nes_profile).notes_for("P2")
    assert len(notes) == 1
    assert notes[0].dur_ticks == 3 * FRAME_TICKS


def test_noise_values_are_not_treated_as_pitches(nes_profile, source):
    frames = frames_of(4)
    frames[:, NO, PITCH] = 13
    frames[:, NO, VELOCITY] = 15
    path, _ = source(frames)
    composition = score_to_composition(load_exprsco(path), path, nes_profile)
    record = build_intake_record(
        composition,
        describe_file(path, root=path.parent, dataset_id="t", dataset_version="t/1"),
        nes_profile,
    )
    noise = [e for e in record["events"] if e["type"] == "noise"]
    assert noise
    assert all("value" in e and "pitch" not in e for e in noise)
    assert noise[0]["value"] == 13


def test_the_triangle_gets_an_audible_velocity(nes_profile, source):
    """exprsco stores velocity 0 for TR because it has no volume register."""
    frames = frames_of(4)
    frames[:, TR, PITCH] = 48
    frames[:, TR, VELOCITY] = 0
    path, _ = source(frames)
    notes = score_to_composition(load_exprsco(path), path, nes_profile).notes_for("TR")
    assert notes[0].velocity == TRIANGLE_VELOCITY


def test_four_bit_volume_scales_to_midi_velocity(nes_profile, source):
    frames = frames_of(2)
    frames[:, P1, PITCH] = 60
    frames[:, P1, VELOCITY] = 15
    path, _ = source(frames)
    notes = score_to_composition(load_exprsco(path), path, nes_profile).notes_for("P1")
    assert notes[0].velocity == 127


# -- envelope preservation ----------------------------------------------------


def test_volume_changes_within_a_note_become_controllers(nes_profile, source):
    """The envelope is most of what makes an NES instrument sound alive."""
    frames = frames_of(4)
    frames[:, P1, PITCH] = 60
    frames[:, P1, VELOCITY] = [15, 10, 5, 2]
    path, _ = source(frames)
    composition = score_to_composition(load_exprsco(path), path, nes_profile)
    volume = [
        c for c in composition.controls if c.channel_id == "P1" and c.data["control"] == CC_VOLUME
    ]
    # One note, but four distinct volume levels recorded across it.
    assert len(composition.notes_for("P1")) == 1
    assert len(volume) == 4
    assert [c.data["value"] for c in volume] == [127, 85, 42, 17]


def test_timbre_changes_become_controllers(nes_profile, source):
    frames = frames_of(4)
    frames[:, P1, PITCH] = 60
    frames[:, P1, VELOCITY] = 10
    frames[:, P1, TIMBRE] = [0, 0, 2, 2]
    path, _ = source(frames)
    composition = score_to_composition(load_exprsco(path), path, nes_profile)
    timbre = [
        c for c in composition.controls if c.channel_id == "P1" and c.data["control"] == CC_TIMBRE
    ]
    assert [(c.tick, c.data["value"]) for c in timbre] == [(0, 0), (2 * FRAME_TICKS, 2)]


# -- timing -------------------------------------------------------------------


def test_one_frame_is_one_grid_step(nes_profile, source):
    """The reason this release was chosen: zero quantization error."""
    frames = frames_of(16)
    # Notes on odd frames, which a coarser grid would smear.
    for index in range(0, 16, 3):
        frames[index, P1, PITCH] = 60 + index
        frames[index, P1, VELOCITY] = 10
    path, _ = source(frames)
    composition = score_to_composition(load_exprsco(path), path, nes_profile)
    record = build_intake_record(
        composition,
        describe_file(path, root=path.parent, dataset_id="t", dataset_version="t/1"),
        nes_profile,
    )
    encoded = RseTokenizer().encode(composition_from_intake(record))
    assert encoded.max_start_error_ticks == 0
    assert encoded.n_encoded == encoded.n_events


def test_wall_clock_duration_is_correct(nes_profile, source):
    frames = frames_of(48)  # exactly two seconds at 24 Hz
    frames[:, P1, PITCH] = 60
    frames[:, P1, VELOCITY] = 10
    path, _ = source(frames)
    composition = score_to_composition(load_exprsco(path), path, nes_profile)
    assert composition.ticks_per_beat == TICKS_PER_BEAT
    assert composition.time.seconds(48 * FRAME_TICKS) == pytest.approx(2.0, abs=0.001)


def test_an_unexpected_frame_rate_is_warned_about(nes_profile, source):
    frames = frames_of(4)
    frames[:, P1, PITCH] = 60
    path, _ = source(frames, rate=48.0)
    composition = score_to_composition(load_exprsco(path), path, nes_profile)
    assert any("frame rate" in w for w in composition.warnings)


# -- integration --------------------------------------------------------------


def test_a_score_produces_a_compliant_intake_record(nes_profile, source):
    frames = frames_of(24)
    frames[:, P1, PITCH] = 72
    frames[:, P1, VELOCITY] = 12
    frames[:, P2, PITCH] = 67
    frames[:, P2, VELOCITY] = 9
    frames[:, TR, PITCH] = 48
    frames[:12, NO, PITCH] = 13
    frames[:12, NO, VELOCITY] = 15
    path, record = source(frames)
    composition = score_to_composition(load_exprsco(path), path, nes_profile)
    intake = build_intake_record(composition, record, nes_profile)

    assert intake["measurements"]["hardware_compliant"] is True
    assert intake["measurements"]["channels_used"] == 4
    for channel in intake["channels"]:
        assert channel["max_simultaneous"] == 1
        assert channel["overlap_count"] == 0
        assert channel["out_of_range_values"] == []
        assert channel["assignment_method"] == "track_name"


def test_channel_order_matches_the_profile(nes_profile):
    assert CHANNEL_ORDER == nes_profile.channel_ids


def test_splits_come_from_the_directory_name():
    assert split_from_path("nesmdb24_exprsco/train/a.pkl") == "train"
    assert split_from_path("nesmdb24_exprsco/valid/a.pkl") == "valid"
    assert split_from_path("nesmdb24_exprsco/test/a.pkl") == "test"
    assert split_from_path("a.pkl") is None
