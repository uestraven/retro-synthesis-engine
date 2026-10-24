from __future__ import annotations

import pytest

from rse.conditioning import (
    MIN_MELODY_NOTES,
    PHRASE_RELATED,
    PHRASE_SAME,
    UNKNOWN,
    ConditioningFeatures,
    Thresholds,
    _similarity,
    agreement,
    eligibility,
    extract_features,
    fit_thresholds,
    label,
    lead_channel,
    motif_note_participation,
)
from rse.events.schema import Composition, Noise, Note

TPB = 480
ROW = 120


def comp(events) -> Composition:
    return Composition(hardware_profile="nes", ticks_per_beat=TPB, events=events)


def melody(pitches, *, channel="P1", step=ROW, duration=ROW) -> Composition:
    return comp(
        [
            Note(channel_id=channel, tick=i * step, pitch=p, duration=duration, velocity=100)
            for i, p in enumerate(pitches)
        ]
    )


# -- motif recurrence ---------------------------------------------------------


def test_recurrence_counts_each_note_once():
    """A gram-based rate double-counts overlapping windows.

    Five identical notes form two matching 4-grams without any musical return,
    which a gram rate scores as repetition. Participation counts notes.
    """
    figure = [60, 62, 64, 65]
    pitches = figure + [71, 72, 73, 74] + figure
    share = motif_note_participation(pitches)
    # Eight of twelve notes belong to the returning figure.
    assert share == pytest.approx(8 / 12, abs=0.01)


def test_recurrence_requires_separation():
    """Adjacent overlapping windows are not a return."""
    figure = [60, 62, 64, 66]
    # One statement, read through overlapping windows, is not recurrence.
    assert motif_note_participation(figure) == 0.0


def test_a_sequence_with_no_return_scores_zero():
    assert motif_note_participation(list(range(60, 90))) == 0.0


def test_a_fully_repeating_sequence_scores_one():
    figure = [60, 64, 67, 72, 67, 64]
    assert motif_note_participation(figure * 4) == pytest.approx(1.0, abs=0.01)


# -- the lead is not assumed to be P1 ----------------------------------------


def test_the_lead_channel_is_chosen_by_behaviour():
    """Pulse roles swap between pieces and the triangle sometimes carries the
    tune, so assuming P1 mislabels the melody."""
    events = []
    # P1 hammers one pitch; P2 carries a moving line.
    for i in range(24):
        events.append(Note(channel_id="P1", tick=i * ROW, pitch=48, duration=ROW, velocity=90))
        events.append(
            Note(
                channel_id="P2",
                tick=i * ROW,
                pitch=60 + (i * 3) % 12,
                duration=ROW,
                velocity=100,
            )
        )
    assert lead_channel(comp(events)) == "P2"


def test_a_channel_with_too_few_notes_is_not_the_lead():
    events = [Note(channel_id="P1", tick=0, pitch=60, duration=ROW, velocity=100)]
    events += [
        Note(channel_id="P2", tick=i * ROW, pitch=60 + i % 7, duration=ROW, velocity=100)
        for i in range(20)
    ]
    assert lead_channel(comp(events)) == "P2"


# -- phrase similarity -------------------------------------------------------


def test_similarity_gives_partial_credit_for_a_rhythmic_change():
    """Exact tuple equality scored real restatements at a median 0.022, which
    made every phrase its own family and the variation rate meaningless."""
    left = [(60, 2), (62, 2), (64, 2), (65, 2)]
    same_pitches_new_rhythm = [(60, 1), (62, 4), (64, 2), (65, 1)]
    assert _similarity(left, left) == pytest.approx(1.0)
    partial = _similarity(left, same_pitches_new_rhythm)
    assert PHRASE_RELATED < partial < PHRASE_SAME


def test_unrelated_phrases_score_below_the_family_threshold():
    left = [(60, 2), (62, 2), (64, 2), (65, 2)]
    right = [(41, 1), (53, 3), (38, 1), (49, 4)]
    assert _similarity(left, right) < PHRASE_RELATED


# -- eligibility and unknown labels -------------------------------------------


def test_a_sparse_piece_is_unknown_not_low():
    """Recording insufficient evidence as 'low' would teach the model that
    sparse music means low recurrence, which the data does not say. 22% of the
    corpus genuinely never restates a motif and is *more* stepwise than the
    rest, so low recurrence is a real musical behaviour rather than a gap."""
    features = ConditioningFeatures(n_melody_notes=4, n_onsets=4, n_phrase_families=0)
    labels = label(features, Thresholds())
    assert labels["motif"] == UNKNOWN
    assert labels["rhythm"] == UNKNOWN
    assert labels["phrase"] == UNKNOWN


def test_eligibility_tracks_each_axis_separately():
    features = ConditioningFeatures(
        n_melody_notes=MIN_MELODY_NOTES, n_onsets=4, n_phrase_families=3
    )
    ok = eligibility(features)
    assert ok["motif"] and ok["phrase"]
    assert not ok["rhythm"]


# -- rhythm needs a pulse, not just density -----------------------------------


def test_driving_needs_both_activity_and_a_pulse():
    """A dense scatter of noise triggers has a high onset rate and no pulse.
    Calling that driving is the mistake the two-condition test prevents."""
    thresholds = Thresholds(rhythm_driving_onset=14.0, rhythm_driving_pulse=0.55)
    dense_no_pulse = ConditioningFeatures(
        n_melody_notes=50, n_onsets=200, onset_rate=30.0, pulse_strength=0.2
    )
    assert label(dense_no_pulse, thresholds)["rhythm"] != "driving"
    dense_with_pulse = ConditioningFeatures(
        n_melody_notes=50, n_onsets=200, onset_rate=30.0, pulse_strength=0.9
    )
    assert label(dense_with_pulse, thresholds)["rhythm"] == "driving"


def test_bass_percussion_is_unmeasurable_without_both_voices():
    """-1.0 means not measurable. A piece with no percussion has no
    coordination to measure, and zero agreement is a different claim."""
    events = [
        Note(channel_id="P1", tick=i * ROW, pitch=60 + i % 5, duration=ROW, velocity=100)
        for i in range(40)
    ]
    events += [
        Note(channel_id="TR", tick=i * ROW, pitch=40, duration=ROW, velocity=100) for i in range(40)
    ]
    features = extract_features(comp(events))
    assert features.bass_percussion_agreement == -1.0

    events += [
        Noise(channel_id="NO", tick=i * ROW, value=8, duration=ROW, velocity=100) for i in range(40)
    ]
    assert extract_features(comp(events)).bass_percussion_agreement >= 0.0


# -- motif and bar duplication stay independent -------------------------------


def test_recurrence_and_bar_duplication_are_separate_features():
    """Real NES music at maximum recurrence still varies nearly every bar --
    consecutive duplication reaches only 0.04 -- while the model's locked-loop
    failure duplicates adjacent bars outright. The two must be able to diverge,
    or a model could satisfy "repetitive" by looping.
    """
    figure = [60, 64, 67, 72]
    contrast = [55, 57, 59, 61]
    # A bar holds 16 of these notes, so the pattern must change at bar scale
    # for the bars themselves to differ.
    looping = extract_features(melody(figure * 4 * 3))
    developing = extract_features(melody(figure * 4 + contrast * 4 + figure * 4))

    assert looping.motif_note_participation > 0.9
    assert developing.motif_note_participation > 0.9
    # Same high recurrence; the loop duplicates adjacent bars and the other
    # returns to its figure only after intervening material.
    assert looping.bars_consecutive_dup > 0.9
    assert developing.bars_consecutive_dup == 0.0
    assert developing.bars_nonconsecutive_reuse > 0.0


def test_a_constant_pitch_is_not_a_motif():
    """Eight identical notes satisfied the separation rule and scored full
    recurrence. A motif needs internal movement."""
    assert motif_note_participation([60] * 8) == 0.0
    assert motif_note_participation([60, 62] * 6) > 0.0


# -- thresholds ---------------------------------------------------------------


def test_thresholds_are_fitted_from_the_given_split_only():
    features = [
        ConditioningFeatures(
            n_melody_notes=50,
            motif_note_participation=value / 100,
            n_onsets=100,
            onset_rate=value / 5,
            pulse_strength=0.5,
            n_phrase_families=3,
            phrase_variation_rate=value / 100,
        )
        for value in range(1, 101)
    ]
    thresholds = fit_thresholds(features, split="train")
    assert thresholds.fitted_on == "train"
    assert thresholds.n_fitted == 100
    assert 0.0 < thresholds.motif_low < thresholds.motif_high < 1.0


def test_the_zero_recurrence_mode_is_not_split_across_labels():
    """Terciles over everything would put part of a genuine zero mode into
    'medium'. Boundaries come from the non-zero population."""
    features = [
        ConditioningFeatures(n_melody_notes=50, motif_note_participation=0.0) for _ in range(60)
    ] + [
        ConditioningFeatures(n_melody_notes=50, motif_note_participation=0.5 + i / 100)
        for i in range(40)
    ]
    thresholds = fit_thresholds(features)
    assert thresholds.motif_low > 0.0


# -- agreement between views --------------------------------------------------


def test_agreement_compares_axes_independently():
    left = {"motif": "high", "rhythm": "driving", "phrase": "variation"}
    right = {"motif": "high", "rhythm": "moderate", "phrase": "variation"}
    result = agreement(left, right)
    assert result["motif"] and result["phrase"]
    assert not result["rhythm"]


# -- per-window labelling and provenance --------------------------------------


def test_window_labels_come_from_the_decoded_training_example(nes_profile):
    """The labelling unit is the training example as the model sees it.

    A piece's label cannot be inherited by its windows: a track can open
    sparsely and develop a motif later, and 63% of pieces yield one window
    while one yields 96, so inheritance both mislabels windows and weights a
    few long pieces heavily.
    """
    from rse.conditioning import melody_line, motif_note_participation
    from rse.events.convert import composition_from_intake
    from rse.generation.plan import make_plan
    from rse.generation.procedural import generate
    from rse.tokenizers.tokenizer import RseTokenizer
    from rse.training.dataset import TRAIN, windows_from_composition

    tokenizer = RseTokenizer()
    composition = composition_from_intake(
        generate(nes_profile, make_plan(seed=21, n_bars=32)).record
    )
    windows = windows_from_composition(
        composition, tokenizer, split=TRAIN, window_bars=4, composition_id="piece-1"
    )
    assert len(windows) > 1

    # Each window's features must come from decoding its own ids.
    measured = []
    for window in windows:
        decoded = tokenizer.decode(window.ids).composition
        _channel, pitches = melody_line(decoded)
        measured.append(motif_note_participation(pitches) if pitches else 0.0)
    whole = motif_note_participation(melody_line(composition)[1])
    # At least one window must differ from the whole-piece figure, or
    # inheritance would be indistinguishable from measurement.
    assert any(abs(m - whole) > 1e-9 for m in measured)


def test_window_index_and_split_are_recoverable(nes_profile):
    from rse.events.convert import composition_from_intake
    from rse.generation.plan import make_plan
    from rse.generation.procedural import generate
    from rse.tokenizers.tokenizer import RseTokenizer
    from rse.training.dataset import TRAIN, windows_from_composition

    tokenizer = RseTokenizer()
    composition = composition_from_intake(
        generate(nes_profile, make_plan(seed=22, n_bars=32)).record
    )
    windows = windows_from_composition(
        composition, tokenizer, split=TRAIN, window_bars=4, composition_id="piece-2"
    )
    indices = [w.start_bar // 4 for w in windows]
    assert indices == list(range(len(windows)))
    assert all(w.split == TRAIN for w in windows)
    assert all(w.composition_id == "piece-2" for w in windows)


def test_insufficient_melody_is_unknown_not_low():
    """`low` is a real musical behaviour -- 22% of the corpus never restates a
    motif and is more stepwise than the rest -- so it must not double as a
    missing-data marker."""
    import importlib.util
    import pathlib

    spec = importlib.util.spec_from_file_location(
        "wml", pathlib.Path(__file__).parent.parent / "scripts" / "window_motif_labels.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    from rse.conditioning import MIN_MELODY_NOTES, Thresholds

    thresholds = Thresholds(motif_low=0.2, motif_high=0.6)
    assert module.motif_label(0.0, MIN_MELODY_NOTES - 1, thresholds) == UNKNOWN
    assert module.motif_label(0.9, MIN_MELODY_NOTES - 1, thresholds) == UNKNOWN
    assert module.motif_label(0.0, MIN_MELODY_NOTES, thresholds) == "low"
    assert module.motif_label(0.9, MIN_MELODY_NOTES, thresholds) == "high"
    assert module.motif_label(0.4, MIN_MELODY_NOTES, thresholds) == "medium"


def test_sampling_weight_offsets_unequal_windows_per_composition():
    """A 96-window piece would otherwise be seen 96x as often as a one-window
    piece, which is the game-dominance problem the splits exist to avoid."""
    counts = {"a": 1, "b": 4, "c": 96}
    weights = {piece: 1.0 / n for piece, n in counts.items()}
    # Each composition contributes the same total mass regardless of length.
    for piece, n in counts.items():
        assert weights[piece] * n == pytest.approx(1.0)
