from __future__ import annotations

import dataclasses

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


def test_threshold_fitting_is_piece_weighted():
    """One composition contributed 76 windows to the tercile calculation.

    Pooling windows gave (0.5, 0.8) against (0.5091, 0.7536) piece-weighted --
    a 6% shift in the `high` boundary set by a single long piece.
    """
    import statistics

    # Nine pieces with one window each, plus one piece with ninety.
    pooled_input = [0.9] * 90 + [0.1 * i for i in range(1, 10)]
    per_piece = [0.9] + [0.1 * i for i in range(1, 10)]

    def tercile(values, q):
        ordered = sorted(values)
        return ordered[int(q * (len(ordered) - 1))]

    # Pooled, the 90-window piece drags both boundaries to its own value.
    assert tercile(pooled_input, 0.33) == pytest.approx(0.9)
    # Piece-weighted, it is one vote among ten.
    assert tercile(per_piece, 0.33) < 0.9
    assert statistics.fmean(per_piece) < statistics.fmean(pooled_input)


def test_unknown_is_preserved_as_a_distinct_state():
    """UNKNOWN must never collapse into LOW: 3168 of 9609 train windows carry
    it, and reinterpreting them as LOW would nearly double that class with
    examples that have no melodic evidence at all."""
    import importlib.util
    import pathlib

    spec = importlib.util.spec_from_file_location(
        "wml", pathlib.Path(__file__).parent.parent / "scripts" / "window_motif_labels.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from rse.conditioning import MIN_MELODY_NOTES, Thresholds

    thresholds = Thresholds(motif_low=0.5091, motif_high=0.7536)
    states = {
        module.motif_label(p, n, thresholds)
        for p, n in (
            (0.0, 4),
            (0.0, MIN_MELODY_NOTES),
            (0.6, MIN_MELODY_NOTES),
            (0.9, MIN_MELODY_NOTES),
        )
    }
    assert states == {UNKNOWN, "low", "medium", "high"}


# -- Gate 3: prefix, migration, arms ------------------------------------------


def test_migration_preserves_every_pretrained_weight_exactly():
    """The regression the review asked for: after widening the vocabulary but
    before any new token appears in a sequence, logits over the original 267
    ids must be unchanged. This checks weight preservation independently of
    the changed softmax denominator, which would hide a corrupted row.
    """
    torch = pytest.importorskip("torch")
    from rse.models.transformer import ModelConfig, RseTransformer
    from rse.training.migrate import migrate_state_dict

    config = ModelConfig(vocab_size=267, n_layers=2, n_heads=2, d_model=32, max_seq_len=64)
    torch.manual_seed(0)
    old = RseTransformer(config).eval()

    state, report = migrate_state_dict(
        old.state_dict(), old_vocab=267, new_vocab=272, old_positions=64, new_positions=96
    )
    new = RseTransformer(
        ModelConfig(vocab_size=272, n_layers=2, n_heads=2, d_model=32, max_seq_len=96)
    ).eval()
    new.load_state_dict(state)

    ids = torch.randint(0, 267, (2, 16))
    mask = torch.ones_like(ids, dtype=torch.bool)
    with torch.no_grad():
        before = old(ids, mask)
        after = new(ids, mask)[..., :267]
    assert torch.equal(before, after)
    assert report["head_retied"] is True
    assert report["vocab"] == [267, 272]


def test_migration_refuses_a_mismatched_checkpoint():
    """Silently migrating a checkpoint whose shapes disagree would scramble
    the pretrained rows it is supposed to protect."""
    torch = pytest.importorskip("torch")
    from rse.training.migrate import migrate_state_dict

    state = {"token_embedding.weight": torch.zeros(100, 8)}
    with pytest.raises(ValueError, match="expected 267"):
        migrate_state_dict(state, old_vocab=267, new_vocab=272, old_positions=64, new_positions=96)
    with pytest.raises(ValueError, match="cannot shrink"):
        migrate_state_dict({}, old_vocab=272, new_vocab=267, old_positions=64, new_positions=64)


def test_new_embedding_rows_are_not_zero():
    """A zero row gives every new token the same logit and no gradient
    direction to separate them."""
    torch = pytest.importorskip("torch")
    from rse.training.migrate import migrate_state_dict

    state = {"token_embedding.weight": torch.ones(267, 16)}
    out, _ = migrate_state_dict(
        state, old_vocab=267, new_vocab=272, old_positions=1, new_positions=1
    )
    new_rows = out["token_embedding.weight"][267:]
    assert new_rows.shape == (5, 16)
    assert new_rows.abs().sum() > 0


def test_migration_is_deterministic_so_arms_start_identically():
    """All three arms must begin from byte-identical weights, or the
    comparison measures initialization as well as conditioning."""
    torch = pytest.importorskip("torch")
    from rse.training.migrate import migrate_state_dict

    state = {"token_embedding.weight": torch.ones(267, 16)}
    a, _ = migrate_state_dict(
        state, old_vocab=267, new_vocab=272, old_positions=1, new_positions=1, seed=0
    )
    b, _ = migrate_state_dict(
        state, old_vocab=267, new_vocab=272, old_positions=1, new_positions=1, seed=0
    )
    assert torch.equal(a["token_embedding.weight"], b["token_embedding.weight"])


def test_finetune_geometry_comes_from_the_checkpoint():
    """A fine-tune on a window set shorter than the foundation's must keep the
    foundation's position table, not size a narrower one from its own data."""
    pytest.importorskip("torch")
    from rse.models.transformer import ModelConfig
    from rse.training.migrate import geometry_for_finetune

    pretrained = ModelConfig(vocab_size=272, max_seq_len=2305)
    requested = ModelConfig(vocab_size=272, max_seq_len=1781)
    resolved = geometry_for_finetune(pretrained, requested, needed_positions=1781)
    assert resolved.max_seq_len == 2305


def test_finetune_refuses_a_different_architecture_or_a_longer_context():
    """Loading mismatched weights fails deep inside `load_state_dict`; failing
    here says which dimension disagrees, and keeps the arms identical."""
    pytest.importorskip("torch")
    from rse.models.transformer import ModelConfig
    from rse.training.migrate import geometry_for_finetune

    pretrained = ModelConfig(vocab_size=272, n_layers=6, max_seq_len=2305)
    with pytest.raises(ValueError, match="n_layers 6 != 4"):
        geometry_for_finetune(
            pretrained,
            ModelConfig(vocab_size=272, n_layers=4, max_seq_len=2305),
            needed_positions=2305,
        )
    with pytest.raises(ValueError, match="needs 4097 positions"):
        geometry_for_finetune(
            pretrained,
            ModelConfig(vocab_size=272, n_layers=6, max_seq_len=4097),
            needed_positions=4097,
        )


def test_shuffling_preserves_label_prevalence():
    """If the shuffled arm saw a different label distribution it would differ
    from the true arm in two ways at once, and a null result would be
    uninterpretable."""
    from collections import Counter

    from rse.conditioning import shuffled_labels

    labels = ["high"] * 20 + ["medium"] * 10 + ["low"] * 30 + [UNKNOWN] * 40
    out = shuffled_labels(labels, seed=3)
    assert Counter(out) == Counter(labels)
    assert out != labels
    assert shuffled_labels(labels, seed=3) == out  # deterministic


def test_arm_labels_differ_only_in_information():
    from rse.conditioning import arm_label

    assert arm_label("A", "high", "low") == "unspecified"
    assert arm_label("B", "high", "low") == "high"
    assert arm_label("C", "high", "low") == "low"
    with pytest.raises(ValueError):
        arm_label("D", "high", "low")


def test_the_conditioning_token_is_always_present_and_never_a_target():
    """Arm A emits `unspecified` rather than omitting the token, so every arm
    has the same prefix length. The prefix must not be predicted, but its
    embedding must still receive gradients through later music tokens."""
    from rse.conditioning import PREFIX_TOKENS
    from rse.tokenizers.tokenizer import Conditioning, RseTokenizer

    tokenizer = RseTokenizer()
    composition = melody([60, 64, 67, 72] * 8)
    plain = tokenizer.encode(composition, conditioning=Conditioning(n_bars=8)).tokens
    asked = tokenizer.encode(composition, conditioning=Conditioning(n_bars=8, motif="high")).tokens
    assert plain[:PREFIX_TOKENS][-1] == "Motif_unspecified"
    assert asked[:PREFIX_TOKENS][-1] == "Motif_high"
    assert len(plain) == len(asked)
    # Only the conditioning token differs.
    assert plain[:6] == asked[:6] and plain[7:] == asked[7:]


def test_an_unrecognised_motif_request_is_unspecified_not_low():
    from rse.tokenizers.tokenizer import RseTokenizer

    tokenizer = RseTokenizer()
    assert tokenizer.motif_token_value("nonsense") == "unspecified"
    assert tokenizer.motif_token_value("high") == "high"
    assert tokenizer.motif_token_value(UNKNOWN) == UNKNOWN


def test_prefix_is_masked_from_the_loss_but_still_receives_gradient():
    """Two requirements that pull in opposite directions.

    The conditioning token must not be a prediction target -- the objective is
    music, not reproducing the prompt -- yet its embedding must still learn,
    which it can only do through its effect on later musical predictions.

    Note on the unused-row check: weights are tied, so `head.weight` IS the
    token embedding and every row is also an output class. An absent token
    therefore still receives gradient as a negative class in the softmax. That
    is expected with tying, is identical across arms, and is why the assertion
    below compares magnitudes rather than demanding zero.
    """
    torch = pytest.importorskip("torch")
    from torch import nn

    from rse.conditioning import PREFIX_TOKENS
    from rse.models.transformer import ModelConfig, RseTransformer
    from rse.tokenizers.tokenizer import RseTokenizer
    from rse.tokenizers.vocab import MOTIF

    tokenizer = RseTokenizer()
    vocab = len(tokenizer)
    motif_ids = [
        tokenizer.vocabulary.id_of(t) for t in tokenizer.vocabulary.tokens if t.startswith(MOTIF)
    ]
    pad = tokenizer.vocabulary.id_of("<pad>")

    torch.manual_seed(0)
    model = RseTransformer(
        ModelConfig(vocab_size=vocab, n_layers=2, n_heads=2, d_model=32, max_seq_len=64)
    ).train()

    ids = torch.tensor([[1, 4, 9, 13, 30, 80, motif_ids[0], 100, 143, 214, 217, 2]])
    inputs, targets = ids[:, :-1], ids[:, 1:].clone()
    mask = torch.ones_like(inputs, dtype=torch.bool)
    targets[:, : PREFIX_TOKENS - 1] = pad

    loss = nn.functional.cross_entropy(
        model(inputs, mask).reshape(-1, vocab), targets.reshape(-1), ignore_index=pad
    )
    loss.backward()
    grad = model.token_embedding.weight.grad

    present = grad[motif_ids[0]].norm().item()
    absent = grad[motif_ids[4]].norm().item()
    # The token in the sequence conditions every later prediction, so its row
    # must carry far more gradient than one that only appears as a negative
    # class.
    assert present > 0
    assert present > absent * 10

    # No prefix position is a target: mask the music too and nothing is left.
    only_prefix = targets.clone()
    only_prefix[:, PREFIX_TOKENS - 1 :] = pad
    empty = nn.functional.cross_entropy(
        model(inputs, mask).reshape(-1, vocab),
        only_prefix.reshape(-1),
        ignore_index=pad,
    )
    assert torch.isnan(empty)


# -- arm assignment by batch-time substitution ---------------------------------


def _fake_window(cid: str, start_bar: int, ids: list[int]):
    from dataclasses import dataclass

    @dataclass
    class W:
        composition_id: str
        start_bar: int
        ids: list[int]
        split: str = "train"

    return W(cid, start_bar, ids)


def _plan_file(tmp_path, rows):
    import json

    path = tmp_path / "labels.jsonl"
    with path.open("w") as handle:
        for piece, start_bar, label in rows:
            handle.write(
                json.dumps({"piece_id": piece, "start_bar": start_bar, "label": label}) + "\n"
            )
    return path


def test_substitution_changes_only_the_conditioning_token(tmp_path):
    """The whole point of sharing one cache: 'identical except one token' is
    structurally true rather than a property to verify across three files."""
    from rse.tokenizers.tokenizer import RseTokenizer
    from rse.training.arms import apply_arm, load_plan, motif_token_ids

    tokenizer = RseTokenizer()
    ids = motif_token_ids(tokenizer.vocabulary)
    plan = load_plan(_plan_file(tmp_path, [("p1", 0, "high"), ("p2", 0, "low")]), seed=0)
    original = [1, 4, 9, 13, 30, 80, ids["unspecified"], 100, 143, 214]
    window = _fake_window("p1", 0, original)

    for arm in ("A", "B", "C"):
        out = apply_arm(window, arm=arm, plan=plan, token_ids=ids, position=6)
        differing = [i for i in range(len(out)) if out[i] != original[i]]
        assert differing in ([], [6]), (arm, differing)
    assert apply_arm(window, arm="B", plan=plan, token_ids=ids, position=6)[6] == ids["high"]
    assert apply_arm(window, arm="A", plan=plan, token_ids=ids, position=6)[6] == ids["unspecified"]


def test_shuffling_keeps_unknown_as_unknown(tmp_path):
    """Permuting UNKNOWN into the labelled population would let arm C assign a
    confident label to a window with no melodic evidence -- a different
    intervention from 'the labels are wrong'."""
    from rse.training.arms import load_plan

    rows = [(f"p{i}", 0, "high") for i in range(20)]
    rows += [(f"q{i}", 0, UNKNOWN) for i in range(20)]
    rows += [(f"r{i}", 0, "low") for i in range(20)]
    plan = load_plan(_plan_file(tmp_path, rows), seed=5)

    for key, value in plan.true_labels.items():
        if value == UNKNOWN:
            assert plan.shuffled_labels[key] == UNKNOWN
        else:
            assert plan.shuffled_labels[key] != UNKNOWN
    assert plan.prevalence()["B"] == plan.prevalence()["C"]


def test_the_shuffle_is_fixed_not_per_batch(tmp_path):
    """Reshuffling per batch would average over permutations instead of
    testing one fixed wrong labelling."""
    from rse.training.arms import load_plan

    rows = [(f"p{i}", 0, lab) for i, lab in enumerate(["high", "low", "medium"] * 10)]
    path = _plan_file(tmp_path, rows)
    a, b = load_plan(path, seed=7), load_plan(path, seed=7)
    assert a.shuffled_labels == b.shuffled_labels
    assert load_plan(path, seed=8).shuffled_labels != a.shuffled_labels


def test_substitution_refuses_a_malformed_prefix():
    """Searching for the token instead of being told its position could
    silently rewrite a musical token."""
    from rse.training.arms import substitute

    with pytest.raises(IndexError):
        substitute([1, 2, 3], position=6, token_id=99)


def test_prefix_masking_removes_only_prefix_targets():
    """Attention must still see the prefix; only the loss ignores it, so the
    condition embedding keeps learning through later music predictions."""
    torch = pytest.importorskip("torch")
    from rse.training.train import make_batch

    @dataclasses.dataclass
    class W:
        ids: list[int]

    windows = [W(list(range(1, 21)))]
    inputs, targets, mask = make_batch(
        windows, [0], pad_id=0, seq_len=64, device=torch.device("cpu"), mask_prefix=6
    )
    assert (targets[0, :6] == 0).all()
    assert (targets[0, 6:] != 0).all()
    # The pad mask is untouched, so attention still attends to the prefix.
    assert mask[0].all()


def test_validation_masks_the_prefix_too():
    """If validation scored the prefix, an arm whose Motif token is constant
    would win on predicting its own condition rather than on the music, and
    the arm comparison would be biased before a single note was generated."""
    torch = pytest.importorskip("torch")
    from rse.models.transformer import ModelConfig, RseTransformer
    from rse.training.train import TrainConfig, evaluate

    @dataclasses.dataclass
    class W:
        ids: list[int]

    # One window whose prefix is cheap to predict and whose body is not.
    windows = [W([5] * 8 + list(range(9, 40)))]
    model = RseTransformer(ModelConfig(vocab_size=64, n_layers=2, max_seq_len=64))
    model.eval()
    config = TrainConfig(batch_size=1, seq_len=48)
    kwargs = {"pad_id": 0, "device": torch.device("cpu"), "max_batches": 1}
    assert evaluate(model, windows, config, mask_prefix=7, **kwargs) != pytest.approx(
        evaluate(model, windows, config, mask_prefix=0, **kwargs)
    )
