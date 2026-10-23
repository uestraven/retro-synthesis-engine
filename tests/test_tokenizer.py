from __future__ import annotations

import pytest

from rse.events.convert import Tolerance, compare_compositions, composition_from_intake
from rse.events.schema import Composition, Noise, Note
from rse.generation.plan import make_plan
from rse.generation.procedural import generate
from rse.tokenizers.grammar import State, TokenGrammar
from rse.tokenizers.tokenizer import Conditioning, RseTokenizer
from rse.tokenizers.vocab import (
    BOS,
    EOS,
    PAD,
    UNK,
    TokenizerConfig,
    Vocabulary,
    build_vocabulary,
)


@pytest.fixture
def tokenizer():
    return RseTokenizer()


@pytest.fixture
def composition(nes_profile):
    generated = generate(nes_profile, make_plan(seed=5, mood="dark", energy="medium", n_bars=8))
    return composition_from_intake(generated.record)


# -- vocabulary ---------------------------------------------------------------


def test_pad_is_index_zero():
    """So padded positions are trivial to mask in a loss."""
    assert build_vocabulary().tokens[0] == PAD
    assert build_vocabulary().pad_id == 0


def test_vocabulary_covers_every_family():
    vocabulary = build_vocabulary()
    for prefix in (
        "Hardware_",
        "Mood_",
        "Energy_",
        "Tempo_",
        "Length_",
        "Position_",
        "Channel_",
        "Pitch_",
        "Noise_",
        "Velocity_",
        "Duration_",
    ):
        assert vocabulary.family(prefix), prefix
    assert "Bar" in vocabulary
    assert "Loop_start" in vocabulary and "Loop_end" in vocabulary


def test_vocabulary_is_compact():
    # A small vocabulary is the point of a channel-aware tokenizer; a general
    # MIDI one would spend hundreds of tokens on programs RSE cannot play.
    assert len(build_vocabulary()) < 300


def test_noise_and_pitch_are_separate_families():
    vocabulary = build_vocabulary()
    assert "Pitch_60" in vocabulary
    assert "Noise_7" in vocabulary
    # The noise channel's range must not be expressible as a pitch token.
    assert set(vocabulary.family("Pitch_")).isdisjoint(vocabulary.family("Noise_"))


def test_unknown_tokens_map_to_unk():
    vocabulary = build_vocabulary()
    assert vocabulary.id_of("Pitch_9999") == vocabulary.id_of(UNK)


def test_config_rejects_nonsense():
    with pytest.raises(ValueError, match="steps_per_bar"):
        TokenizerConfig(steps_per_bar=0)
    with pytest.raises(ValueError, match="pitch_min"):
        TokenizerConfig(pitch_min=100, pitch_max=50)
    with pytest.raises(ValueError, match="velocity_buckets"):
        TokenizerConfig(velocity_buckets=0)
    with pytest.raises(ValueError, match="unpitched channels"):
        TokenizerConfig(unpitched_channels=("DMC",))


def test_controller_tokens_are_explicitly_not_implemented():
    """Reserved in the design, absent from the implementation, and says so."""
    with pytest.raises(NotImplementedError, match="controller tokens"):
        TokenizerConfig(include_controllers=True)


def test_vocabulary_round_trips_through_its_dict():
    original = build_vocabulary()
    assert Vocabulary.from_dict(original.to_dict()).tokens == original.tokens


def test_a_tampered_vocabulary_is_refused():
    payload = build_vocabulary().to_dict()
    payload["tokens"] = payload["tokens"][:-1]
    with pytest.raises(ValueError, match="do not match"):
        Vocabulary.from_dict(payload)


def test_a_future_vocab_version_is_refused():
    payload = build_vocabulary().to_dict()
    payload["config"]["vocab_version"] = "rse-vocab/9.0.0"
    with pytest.raises(ValueError, match="unsupported vocabulary version"):
        Vocabulary.from_dict(payload)


# -- encoding -----------------------------------------------------------------


def test_encoding_starts_with_a_conditioning_header(tokenizer, composition):
    encoded = tokenizer.encode(composition)
    assert encoded.tokens[0] == BOS
    assert encoded.tokens[1] == "Hardware_nes"
    assert encoded.tokens[2].startswith("Mood_")
    assert encoded.tokens[3].startswith("Energy_")
    assert encoded.tokens[4].startswith("Tempo_")
    assert encoded.tokens[5].startswith("Length_")
    assert encoded.tokens[-1] == EOS


def test_conditioning_is_read_from_the_generation_plan(tokenizer, composition):
    encoded = tokenizer.encode(composition)
    assert encoded.conditioning.mood == "dark"
    assert encoded.conditioning.energy == "medium"
    assert "Mood_dark" in encoded.tokens


def test_a_dataset_file_gets_unknown_conditioning(tokenizer, midi_path, nes_profile, source_record):
    from rse.data.midi_io import parse_midi
    from rse.data.normalize import build_intake_record

    record = build_intake_record(parse_midi(midi_path, nes_profile), source_record, nes_profile)
    encoded = tokenizer.encode(composition_from_intake(record))
    # Proposal 6.4: mood conditioning needs labels, so an unlabelled file must
    # say "unknown" rather than have a mood guessed for it.
    assert encoded.conditioning.mood == "unknown"
    assert encoded.conditioning.energy == "unknown"


def test_every_note_is_encoded(tokenizer, composition):
    encoded = tokenizer.encode(composition)
    assert encoded.n_encoded == len(composition.sounding_events())
    assert encoded.n_events == encoded.n_encoded


def test_a_grid_aligned_composition_encodes_without_timing_error(tokenizer, composition):
    encoded = tokenizer.encode(composition)
    assert encoded.max_start_error_ticks == 0


def test_off_grid_notes_report_their_quantization_error(tokenizer):
    """Tokenization is where quantization becomes destructive, so it is measured."""
    off_grid = Composition(
        hardware_profile="nes",
        ticks_per_beat=480,
        events=[Note(channel_id="P1", tick=37, pitch=60, duration=240, velocity=100)],
    )
    encoded = tokenizer.encode(off_grid)
    assert encoded.max_start_error_ticks == 37
    assert encoded.mean_start_error_ticks == 37


def test_a_bar_token_is_emitted_for_every_bar_including_empty_ones(tokenizer):
    sparse = Composition(
        hardware_profile="nes",
        ticks_per_beat=480,
        events=[Note(channel_id="P1", tick=0, pitch=60, duration=240, velocity=100)],
    )
    encoded = tokenizer.encode(sparse, conditioning=Conditioning(n_bars=8))
    assert encoded.tokens.count("Bar") == 8


def test_out_of_vocabulary_values_are_dropped_and_reported(tokenizer):
    bad = Composition(
        hardware_profile="nes",
        ticks_per_beat=480,
        events=[
            Note(channel_id="P1", tick=0, pitch=5, duration=240, velocity=100),
            Noise(channel_id="NO", tick=0, value=99, duration=60, velocity=100),
        ],
    )
    encoded = tokenizer.encode(bad)
    assert encoded.n_out_of_range == 2
    assert encoded.n_encoded == 0
    assert len(encoded.warnings) == 2


def test_overlong_durations_are_clamped_and_counted(tokenizer):
    long_note = Composition(
        hardware_profile="nes",
        ticks_per_beat=480,
        events=[Note(channel_id="P1", tick=0, pitch=60, duration=480 * 40, velocity=100)],
    )
    encoded = tokenizer.encode(long_note)
    assert encoded.n_duration_clamped == 1
    assert "Duration_32" in encoded.tokens


def test_encoding_is_deterministic(tokenizer, composition):
    assert tokenizer.encode(composition).ids == tokenizer.encode(composition).ids


def test_velocity_bucketing_is_monotonic(tokenizer):
    buckets = [tokenizer.velocity_bucket(v) for v in (1, 32, 64, 96, 127)]
    assert buckets == sorted(buckets)
    assert buckets[0] < buckets[-1]


def test_debucketing_stays_inside_the_bucket(tokenizer):
    for bucket in range(tokenizer.config.velocity_buckets):
        assert tokenizer.velocity_bucket(tokenizer.bucket_velocity(bucket)) == bucket


# -- decoding -----------------------------------------------------------------


def test_round_trip_preserves_every_note(tokenizer, composition):
    decoded = tokenizer.decode(tokenizer.encode(composition).ids)
    assert decoded.clean
    assert len(decoded.composition.sounding_events()) == len(composition.sounding_events())


def test_round_trip_is_within_a_stated_tolerance(tokenizer, composition):
    """One grid step of timing, one step of duration, one velocity bucket."""
    decoded = tokenizer.decode(tokenizer.encode(composition).ids)
    tolerance = Tolerance(timing_ticks=60, duration_ticks=120, velocity=16)
    report = compare_compositions(composition, decoded.composition, tolerance=tolerance)
    assert report.ok, report.to_dict()
    assert report.n_matched == len(composition.sounding_events())


def test_noise_decodes_as_noise_not_pitch(tokenizer, composition):
    decoded = tokenizer.decode(tokenizer.encode(composition).ids)
    noise = [e for e in decoded.composition.events if isinstance(e, Noise)]
    assert noise
    assert all(e.channel_id == "NO" for e in noise)
    assert not any(isinstance(e, Note) and e.channel_id == "NO" for e in decoded.composition.events)


def test_tempo_and_loop_survive_the_round_trip(tokenizer, composition):
    decoded = tokenizer.decode(tokenizer.encode(composition).ids)
    assert decoded.composition.loop_start_tick == 0
    assert decoded.composition.loop_end_tick is not None
    original_bpm = 60_000_000 / composition.tempo_map[0].us_per_beat
    decoded_bpm = 60_000_000 / decoded.composition.tempo_map[0].us_per_beat
    assert decoded_bpm == pytest.approx(original_bpm, abs=5)  # tempo is bucketed


def test_decoding_tolerates_garbage(tokenizer, composition):
    """Model output is malformed long before it is musical."""
    ids = tokenizer.encode(composition).ids
    corrupted = ids[:10] + [tokenizer.vocabulary.id_of("Duration_4")] * 5 + ids[10:]
    decoded = tokenizer.decode(corrupted)
    assert decoded.n_tokens_skipped == 5
    assert len(decoded.composition.sounding_events()) > 0


def test_decoding_an_empty_sequence_yields_an_empty_composition(tokenizer):
    decoded = tokenizer.decode([])
    assert decoded.composition.events == []


def test_decode_accepts_token_strings(tokenizer, composition):
    encoded = tokenizer.encode(composition)
    from_strings = tokenizer.decode(encoded.tokens)
    from_ids = tokenizer.decode(encoded.ids)
    assert from_strings.composition.events == from_ids.composition.events


def test_a_coarse_resolution_is_refused(tokenizer):
    with pytest.raises(ValueError, match="too coarse"):
        tokenizer.ticks_per_step(2)


# -- persistence --------------------------------------------------------------


def test_tokenizer_saves_and_loads(tokenizer, tmp_path, composition):
    path = tokenizer.save(tmp_path / "vocab.json")
    restored = RseTokenizer.load(path)
    assert len(restored) == len(tokenizer)
    assert restored.encode(composition).ids == tokenizer.encode(composition).ids


# -- grammar ------------------------------------------------------------------


def test_grammar_accepts_a_real_sequence(tokenizer, composition):
    grammar = TokenGrammar(tokenizer.vocabulary)
    for token in tokenizer.encode(composition).tokens:
        assert grammar.advance(token), token
    assert grammar.finished


def test_grammar_requires_the_header_in_order(tokenizer):
    grammar = TokenGrammar(tokenizer.vocabulary)
    assert not grammar.advance("Bar")  # before BOS
    assert grammar.advance(BOS)
    assert not grammar.advance("Mood_dark")  # hardware must come first
    assert grammar.advance("Hardware_nes")


def test_grammar_demands_a_value_then_velocity_then_duration(tokenizer):
    grammar = TokenGrammar(tokenizer.vocabulary)
    for token in (
        BOS,
        "Hardware_nes",
        "Mood_dark",
        "Energy_low",
        "Tempo_120",
        "Length_8",
        "Bar",
        "Position_0",
        "Channel_P1",
    ):
        assert grammar.advance(token), token
    assert grammar.state is State.EXPECT_VALUE
    assert not grammar.advance("Duration_4")  # cannot skip the pitch
    assert grammar.advance("Pitch_60")
    assert not grammar.advance("Pitch_61")  # cannot have two pitches
    assert grammar.advance("Velocity_4")
    assert grammar.advance("Duration_4")
    assert grammar.state is State.EXPECT_NOTE_OR_NEXT


def test_grammar_requires_noise_values_on_the_noise_channel(tokenizer):
    grammar = TokenGrammar(tokenizer.vocabulary)
    for token in (
        BOS,
        "Hardware_nes",
        "Mood_dark",
        "Energy_low",
        "Tempo_120",
        "Length_8",
        "Bar",
        "Position_0",
        "Channel_NO",
    ):
        grammar.advance(token)
    # The noise channel is unpitched, so a pitch token is illegal here.
    assert not grammar.advance("Pitch_60")
    assert grammar.advance("Noise_7")


def test_grammar_allowed_ids_are_never_empty_mid_sequence(tokenizer, composition):
    grammar = TokenGrammar(tokenizer.vocabulary)
    for token in tokenizer.encode(composition).tokens[:-1]:
        assert grammar.allowed_ids(), grammar.state
        grammar.advance(token)


def test_an_illegal_token_leaves_the_state_alone(tokenizer):
    grammar = TokenGrammar(tokenizer.vocabulary)
    grammar.advance(BOS)
    before = grammar.state
    assert not grammar.advance("Pitch_60")
    assert grammar.state is before


def test_grammar_reset(tokenizer):
    grammar = TokenGrammar(tokenizer.vocabulary)
    grammar.advance(BOS)
    grammar.reset()
    assert grammar.state is State.START
