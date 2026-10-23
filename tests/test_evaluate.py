from __future__ import annotations

import pytest

from rse.events.convert import composition_from_intake
from rse.events.schema import Composition, Noise, Note
from rse.generation.plan import make_plan
from rse.generation.procedural import generate
from rse.training.evaluate import (
    MEMORIZATION_THRESHOLD,
    aggregate,
    bar_signatures,
    compare,
    corpus_fingerprints,
    evaluate_composition,
    grid_alignment,
    infer_key,
    melodic_shape,
    melody_pitches,
    ngram_repetition_rate,
    pitch_class_entropy,
    repetition_metrics,
)

C_MAJOR_SCALE = [60, 62, 64, 65, 67, 69, 71, 72]


def composition_of(*events, **kwargs) -> Composition:
    return Composition(hardware_profile="nes", ticks_per_beat=480, events=list(events), **kwargs)


def note(tick, pitch, channel="P1", duration=240):
    return Note(channel_id=channel, tick=tick, pitch=pitch, duration=duration, velocity=100)


# -- harmonic consistency -----------------------------------------------------


def test_key_inference_finds_an_obvious_key():
    name, fraction = infer_key(C_MAJOR_SCALE)
    assert "major" in name
    assert fraction == 1.0


def test_chromatic_noise_fits_no_key_well():
    _name, fraction = infer_key(list(range(60, 72)))
    assert fraction < 0.8


def test_key_inference_on_nothing():
    assert infer_key([]) == ("unknown", 0.0)


def test_pitch_class_entropy_rises_with_variety():
    assert pitch_class_entropy([60] * 8) == pytest.approx(0.0)
    assert pitch_class_entropy(list(range(60, 72))) == pytest.approx(3.585, abs=0.01)
    assert pitch_class_entropy([]) == 0.0


# -- repetition ---------------------------------------------------------------


def test_bar_signatures_group_by_bar():
    composition = composition_of(note(0, 60), note(1920, 60), note(3840, 64))
    assert len(bar_signatures(composition)) == 3


def test_identical_bars_are_detected_as_repetition():
    composition = composition_of(note(0, 60), note(1920, 60))
    repeated, unique = repetition_metrics(bar_signatures(composition))
    assert repeated == pytest.approx(0.5)
    assert unique == pytest.approx(0.5)


def test_all_different_bars_have_no_repetition():
    composition = composition_of(note(0, 60), note(1920, 64), note(3840, 67))
    repeated, unique = repetition_metrics(bar_signatures(composition))
    assert repeated == 0.0
    assert unique == 1.0


def test_repetition_metrics_on_nothing():
    assert repetition_metrics([]) == (0.0, 0.0)


def test_ngram_repetition_is_transposition_invariant():
    """A phrase repeated a fifth up is repetition to a listener."""
    phrase = [60, 62, 64, 65, 67]
    transposed = [p + 7 for p in phrase]
    assert ngram_repetition_rate(phrase + transposed, n=3) > 0.5


def test_ngram_repetition_needs_enough_notes():
    assert ngram_repetition_rate([60, 62], n=4) == 0.0


# -- melodic shape ------------------------------------------------------------


def test_stepwise_motion_is_measured():
    stepwise, leaps = melodic_shape([60, 62, 64, 65])
    assert stepwise == 1.0
    assert leaps == 0.0


def test_leaps_are_measured():
    stepwise, leaps = melodic_shape([60, 72, 60, 72])
    assert stepwise == 0.0
    assert leaps == 1.0


def test_melodic_shape_needs_two_notes():
    assert melodic_shape([60]) == (0.0, 0.0)


# -- technical ----------------------------------------------------------------


def test_grid_alignment():
    aligned = composition_of(note(0, 60), note(120, 62))
    assert grid_alignment(aligned) == 1.0
    off = composition_of(note(0, 60), note(37, 62))
    assert grid_alignment(off) == pytest.approx(0.5)
    assert grid_alignment(composition_of()) == 0.0


def test_melody_pitches_excludes_the_noise_channel():
    """Noise values are hardware periods, never pitches."""
    composition = composition_of(
        note(0, 60),
        Noise(channel_id="NO", tick=0, value=7, duration=60, velocity=100),
    )
    assert melody_pitches(composition) == [60]


def test_melody_pitches_can_select_one_channel():
    composition = composition_of(note(0, 60), note(0, 67, channel="P2"))
    assert melody_pitches(composition, "P2") == [67]


# -- whole-composition evaluation ---------------------------------------------


def test_generated_music_scores_sensibly(nes_profile):
    generated = generate(nes_profile, make_plan(seed=11, mood="heroic", n_bars=16))
    metrics = evaluate_composition(composition_from_intake(generated.record), nes_profile)

    assert metrics.n_notes > 0
    assert metrics.n_bars == 16
    assert metrics.n_channels_used == 4
    assert metrics.hardware_compliant
    assert metrics.loop_ok
    # The generator writes in a key, so most of the lead should fit one.
    assert metrics.in_scale_fraction > 0.85
    assert metrics.grid_aligned_fraction == 1.0
    # And it reuses phrases, so repetition should be present but not total.
    assert metrics.repetition_in_band, metrics.repeated_bar_fraction


def test_empty_music_scores_zero(nes_profile):
    metrics = evaluate_composition(composition_of(), nes_profile)
    assert metrics.n_notes == 0
    assert metrics.key == "unknown"
    assert not metrics.loop_ok


def test_originality_screening_flags_a_copy(nes_profile):
    """MVP criterion 10, applied to output rather than to the dataset."""
    generated = generate(nes_profile, make_plan(seed=21, n_bars=16))
    composition = composition_from_intake(generated.record)
    fingerprints = corpus_fingerprints([composition], nes_profile)

    same = evaluate_composition(composition, nes_profile, corpus_fingerprints=fingerprints)
    assert same.max_corpus_similarity == pytest.approx(1.0)
    assert same.flagged_as_memorized

    other = composition_from_intake(generate(nes_profile, make_plan(seed=22, n_bars=16)).record)
    different = evaluate_composition(other, nes_profile, corpus_fingerprints=fingerprints)
    assert different.max_corpus_similarity < MEMORIZATION_THRESHOLD
    assert not different.flagged_as_memorized


def test_metrics_serialize(nes_profile):
    generated = generate(nes_profile, make_plan(seed=31, n_bars=8))
    payload = evaluate_composition(composition_from_intake(generated.record), nes_profile).to_dict()
    assert payload["n_bars"] == 8
    assert payload["hardware_compliant"] is True
    assert "in_scale_fraction" in payload


# -- aggregation and comparison -----------------------------------------------


def test_aggregate_averages_metrics(nes_profile):
    metrics = [
        evaluate_composition(
            composition_from_intake(generate(nes_profile, make_plan(seed=s, n_bars=8)).record),
            nes_profile,
        )
        for s in (1, 2, 3)
    ]
    summary = aggregate("baseline", metrics)
    assert summary.n == 3
    assert summary.means["n_bars"] == 8
    assert summary.rates["hardware_compliant"] == 1.0


def test_aggregate_of_nothing_is_empty():
    assert aggregate("none", []).n == 0


def test_comparison_reports_deltas_and_the_technical_bar(nes_profile):
    baseline_metrics = [
        evaluate_composition(
            composition_from_intake(generate(nes_profile, make_plan(seed=s, n_bars=8)).record),
            nes_profile,
        )
        for s in (1, 2)
    ]
    baseline = aggregate("rule-based", baseline_metrics)
    # A candidate that produces nothing must fail the technical bar.
    empty = aggregate("model", [evaluate_composition(composition_of(), nes_profile)])

    report = compare(baseline, empty)
    assert report["baseline"]["label"] == "rule-based"
    assert report["candidate"]["label"] == "model"
    assert "mean_deltas" in report and "rate_deltas" in report
    assert report["clears_technical_bar"] is False
    assert report["technical_bar"]["loop_ok"] is False


def test_a_baseline_against_itself_clears_the_bar(nes_profile):
    metrics = [
        evaluate_composition(
            composition_from_intake(generate(nes_profile, make_plan(seed=s, n_bars=8)).record),
            nes_profile,
        )
        for s in (1, 2, 3)
    ]
    summary = aggregate("rule-based", metrics)
    report = compare(summary, summary)
    assert report["clears_technical_bar"] is True
    assert all(delta == 0 for delta in report["mean_deltas"].values())
