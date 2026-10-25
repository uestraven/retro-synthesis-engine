from __future__ import annotations

import pytest

from rse.events.convert import composition_from_intake
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


def test_round_trip_preserves_the_music_in_rows(tokenizer, composition):
    """The round-trip contract is musical, not tick-exact.

    Since the grid is each piece's own tracker row, `decode` writes canonical
    ticks and a Tempo that compensates — so ticks legitimately change while the
    music does not. Comparing in rows is what holds, and it is also
    tempo-independent, which matters because the Tempo token is bucketed and a
    2% bucket error is enough to make a comparison in seconds drift apart.
    """
    encoded = tokenizer.encode(composition)
    decoded = tokenizer.decode(encoded.ids)
    fit = tokenizer.fit_grid(composition)

    def rows(comp, step_ticks):
        events = comp.sounding_events()
        first = min(e.tick for e in events)
        return {
            (
                round((e.tick - first) / step_ticks),
                e.channel_id,
                e.pitch if isinstance(e, Note) else e.value,
            )
            for e in events
        }

    original = rows(composition, fit.step_ticks)
    result = rows(decoded.composition, tokenizer.ticks_per_step(decoded.composition.ticks_per_beat))
    # Nothing invented, and nothing lost beyond the intra-row writes that the
    # encoder deliberately folds together.
    assert result <= original, result - original
    assert len(result) >= len(original) - encoded.n_rows_collapsed


def test_round_trip_keeps_velocity_within_its_bucket(tokenizer, composition):
    """Velocity survives to the bucket, which is all the vocabulary carries."""
    decoded = tokenizer.decode(tokenizer.encode(composition).ids)
    buckets: dict[tuple[str, int], set[int]] = {}
    for event in composition.sounding_events():
        key = (event.channel_id, event.pitch if isinstance(event, Note) else event.value)
        buckets.setdefault(key, set()).add(tokenizer.velocity_bucket(event.velocity))
    for event in decoded.composition.sounding_events():
        key = (event.channel_id, event.pitch if isinstance(event, Note) else event.value)
        assert key in buckets
        assert tokenizer.velocity_bucket(event.velocity) in buckets[key]


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
    # Tempo describes the fitted row rate, not the source file's tempo map.
    # NES-MDB's map is useless for this — every file claims 120 BPM — and the
    # row rate is what `decode` needs to turn rows back into seconds. Whether a
    # detected row is a sixteenth or an eighth is not knowable, so the number
    # can differ from the composer's intent by a factor of two while the piece
    # still plays at the right speed.
    fit = tokenizer.fit_grid(composition)
    decoded_bpm = 60_000_000 / decoded.composition.tempo_map[0].us_per_beat
    assert decoded_bpm == pytest.approx(fit.bpm, rel=0.03)  # geometric buckets


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
        "Motif_unspecified",
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
        "Motif_unspecified",
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


# -- section tokens -----------------------------------------------------------


def test_section_tokens_exist_in_the_vocabulary():
    vocabulary = build_vocabulary()
    assert "Section_A" in vocabulary and "Section_F" in vocabulary
    assert len(vocabulary.family("Section_")) == 6


def test_sections_are_emitted_at_the_bar_they_open(tokenizer):
    from rse.events.schema import SectionSpan

    composition = Composition(
        hardware_profile="nes",
        ticks_per_beat=480,
        events=[Note(channel_id="P1", tick=0, pitch=60, duration=240, velocity=100)],
        sections=[SectionSpan(0, "A"), SectionSpan(8, "B")],
    )
    tokens = tokenizer.encode(composition, conditioning=Conditioning(n_bars=16)).tokens
    assert tokens.count("Section_A") == 1
    assert tokens.count("Section_B") == 1
    # Each announces the bar it opens, so Bar follows immediately.
    assert tokens[tokens.index("Section_A") + 1] == "Bar"
    assert tokens[tokens.index("Section_B") + 1] == "Bar"


def test_a_composition_without_sections_emits_none(tokenizer, composition):
    composition.sections = []
    assert not any(t.startswith("Section_") for t in tokenizer.encode(composition).tokens)


def test_sections_round_trip(tokenizer):
    from rse.events.schema import SectionSpan

    original = Composition(
        hardware_profile="nes",
        ticks_per_beat=480,
        events=[
            Note(channel_id="P1", tick=t * 480 * 4, pitch=60, duration=240, velocity=100)
            for t in range(16)
        ],
        sections=[SectionSpan(0, "A"), SectionSpan(4, "B"), SectionSpan(8, "A")],
    )
    decoded = tokenizer.decode(tokenizer.encode(original).ids)
    assert [(s.start_bar, s.label) for s in decoded.composition.sections] == [
        (0, "A"),
        (4, "B"),
        (8, "A"),
    ]


def test_the_grammar_requires_a_bar_after_a_section(tokenizer):
    grammar = TokenGrammar(tokenizer.vocabulary)
    for token in (
        BOS,
        "Hardware_nes",
        "Mood_dark",
        "Energy_low",
        "Tempo_120",
        "Length_8",
        "Motif_unspecified",
    ):
        grammar.advance(token)
    assert grammar.advance("Section_A")
    assert not grammar.advance("Position_0")  # only Bar may follow a Section
    assert not grammar.advance("Section_B")
    assert grammar.advance("Bar")


def test_a_section_may_open_a_later_bar_too(tokenizer):
    grammar = TokenGrammar(tokenizer.vocabulary)
    for token in (
        BOS,
        "Hardware_nes",
        "Mood_dark",
        "Energy_low",
        "Tempo_120",
        "Length_8",
        "Motif_unspecified",
        "Bar",
        "Position_0",
        "Channel_P1",
        "Pitch_60",
        "Velocity_4",
        "Duration_4",
    ):
        assert grammar.advance(token), token
    assert grammar.advance("Section_B")
    assert grammar.advance("Bar")


# -- the tracker row grid -----------------------------------------------------


def row_composition(row_ticks, n_rows, *, ticks_per_beat=480, phase=0):
    """Notes one per row, which is what a tracker produces."""
    return Composition(
        hardware_profile="nes",
        ticks_per_beat=ticks_per_beat,
        events=[
            Note(
                channel_id="P1",
                tick=phase + int(round(i * row_ticks)),
                pitch=60 + i % 7,
                duration=int(row_ticks),
                velocity=100,
            )
            for i in range(n_rows)
        ],
    )


def test_the_grid_is_fitted_to_the_row_not_the_source_frame():
    """NES music advances one row every few video frames; the row is the unit
    the tokenizer should step on, not whatever the source file sampled at."""
    tokenizer = RseTokenizer()
    fit = tokenizer.fit_grid(row_composition(240, 64))
    assert fit.aligned
    assert fit.step_ticks == pytest.approx(240, rel=0.01)


def test_the_tempo_follows_from_the_row_length():
    """Four rows to a beat, so BPM = 15 / row_seconds. A 0.1 s row is 150 BPM,
    which is FamiTracker's default of speed 6 at 60 Hz."""
    tokenizer = RseTokenizer()
    # 480 ticks/beat at 120 BPM is 960 ticks/sec, so 96 ticks is 0.1 s.
    fit = tokenizer.fit_grid(row_composition(96, 80))
    assert fit.period_sec == pytest.approx(0.1, rel=0.02)
    assert fit.bpm == pytest.approx(150.0, rel=0.02)


def test_an_unfittable_composition_falls_back_to_the_old_grid():
    """A single note cannot imply a row, and the encoder must still work."""
    tokenizer = RseTokenizer()
    lonely = Composition(
        hardware_profile="nes",
        ticks_per_beat=480,
        events=[Note(channel_id="P1", tick=0, pitch=60, duration=120, velocity=100)],
    )
    fit = tokenizer.fit_grid(lonely)
    assert not fit.aligned
    assert fit.step_ticks == tokenizer.ticks_per_step(480)


def test_the_fit_can_be_turned_off_for_the_training_comparison():
    from rse.tokenizers.vocab import TokenizerConfig

    tokenizer = RseTokenizer(TokenizerConfig(aligned_grid=False))
    assert not tokenizer.fit_grid(row_composition(240, 64)).aligned


def test_several_events_in_one_row_collapse_to_one_note_per_channel():
    """A tracker row holds one note per channel, so extra events inside a row
    are vibrato or arpeggio writes. Emitting them all would put a chord on a
    monophonic channel, which the hardware cannot play."""
    tokenizer = RseTokenizer()
    composition = row_composition(240, 64)
    # An arpeggio write a third of a row after each of the first four notes.
    for i in range(4):
        composition.events.append(
            Note(channel_id="P1", tick=i * 240 + 80, pitch=72, duration=80, velocity=100)
        )
    encoded = tokenizer.encode(composition)
    assert encoded.n_rows_collapsed == 4
    assert "Pitch_72" not in encoded.tokens


def test_the_first_sounding_row_is_bar_zero_position_zero():
    """Counting from the fitted phase is unstable: the phase is only defined
    modulo the row, so two fits of the same music can differ by one position."""
    tokenizer = RseTokenizer()
    for phase in (0, 97, 239):
        tokens = tokenizer.encode(row_composition(240, 64, phase=phase)).tokens
        assert tokens[tokens.index("Bar") + 1] == "Position_0", phase


def test_encoding_is_stable_when_repeated():
    """The gate that caught both earlier designs: encode, decode, encode again."""
    tokenizer = RseTokenizer()
    composition = row_composition(240, 64)
    first = tokenizer.encode(composition).tokens
    second = tokenizer.encode(tokenizer.decode(first).composition).tokens
    assert first == second


def test_sections_can_be_turned_off_without_changing_the_vocabulary():
    """The ablation must change the data and nothing else.

    Dropping section_labels would shrink the vocabulary and confound the
    comparison with a model-size change, so the tokens stay and simply never
    fire.
    """
    from rse.events.schema import SectionSpan
    from rse.tokenizers.vocab import TokenizerConfig, build_vocabulary

    on = TokenizerConfig()
    off = TokenizerConfig(emit_sections=False)
    assert len(build_vocabulary(on)) == len(build_vocabulary(off))
    assert "Section_A" in build_vocabulary(off).tokens

    composition = row_composition(240, 64)
    composition.sections = [SectionSpan(0, "A"), SectionSpan(2, "B")]
    with_sections = RseTokenizer(on).encode(composition).tokens
    without = RseTokenizer(off).encode(composition).tokens
    assert any(t.startswith("Section_") for t in with_sections)
    assert not any(t.startswith("Section_") for t in without)


def test_turning_sections_off_leaves_the_notes_alone():
    """Only the Section tokens may differ, or the ablation measures two things."""
    from rse.events.schema import SectionSpan
    from rse.tokenizers.vocab import TokenizerConfig

    composition = row_composition(240, 64)
    composition.sections = [SectionSpan(0, "A"), SectionSpan(2, "B")]
    with_sections = [
        t
        for t in RseTokenizer(TokenizerConfig()).encode(composition).tokens
        if not t.startswith("Section_")
    ]
    without = RseTokenizer(TokenizerConfig(emit_sections=False)).encode(composition).tokens
    assert with_sections == without
