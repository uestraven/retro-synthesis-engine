from __future__ import annotations

from rse.data.roles import BY_ORDINAL, BY_TRACK_NAME, SourceVoice, assign_channels


def voice(index: int, name: str | None, *, notes: int = 10, channel: int | None = None):
    return SourceVoice(
        key=(index, channel if channel is not None else index),
        track_index=index,
        midi_channel=channel if channel is not None else index,
        track_name=name,
        instrument_name=None,
        n_notes=notes,
    )


def test_named_tracks_map_to_their_channels_regardless_of_order(nes_profile):
    voices = [voice(0, "NO"), voice(1, "TR"), voice(2, "P2"), voice(3, "P1")]
    result = assign_channels(voices, nes_profile)
    assert [result.channel_id_for(v.key) for v in voices] == ["NO", "TR", "P2", "P1"]
    assert all(result.assignments[v.key].method == BY_TRACK_NAME for v in voices)
    assert result.warnings == []


def test_unnamed_tracks_fall_back_to_profile_order(nes_profile):
    voices = [voice(i, None) for i in range(4)]
    result = assign_channels(voices, nes_profile)
    assert [result.channel_id_for(v.key) for v in voices] == ["P1", "P2", "TR", "NO"]
    assert all(result.assignments[v.key].method == BY_ORDINAL for v in voices)


def test_a_named_track_wins_and_the_rest_fill_in_around_it(nes_profile):
    voices = [voice(0, None), voice(1, "NO"), voice(2, None)]
    result = assign_channels(voices, nes_profile)
    assert result.channel_id_for(voices[1].key) == "NO"
    assert result.channel_id_for(voices[0].key) == "P1"
    assert result.channel_id_for(voices[2].key) == "P2"


def test_duplicate_channel_names_are_reported_not_silently_merged(nes_profile):
    voices = [voice(0, "P1"), voice(1, "P1")]
    result = assign_channels(voices, nes_profile)
    assert result.channel_id_for(voices[0].key) == "P1"
    # The second claimant keeps its notes, on a different channel, and says so.
    assert result.channel_id_for(voices[1].key) == "P2"
    assert any("already taken" in w for w in result.warnings)


def test_voices_beyond_the_channel_count_are_dropped_loudly(nes_profile):
    voices = [voice(i, None) for i in range(6)]
    result = assign_channels(voices, nes_profile)
    assert result.channel_id_for(voices[4].key) is None
    assert result.channel_id_for(voices[5].key) is None
    assert sum("was dropped" in w for w in result.warnings) == 2


def test_silent_tracks_do_not_consume_channels(nes_profile):
    voices = [voice(0, None, notes=0), voice(1, None)]
    result = assign_channels(voices, nes_profile)
    assert result.channel_id_for(voices[0].key) is None
    assert result.channel_id_for(voices[1].key) == "P1"


def test_unused_channels_are_reported(nes_profile):
    result = assign_channels([voice(0, "P1")], nes_profile)
    assert any("no source voice mapped" in w for w in result.warnings)
