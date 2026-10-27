"""What differs between two songs of the same style.

Measured before this existed, across 20 seeds of each style: the melody and
drums were unique every time, but the harmony and bass had only 5-6 variants
(victory: 2), and the form, key and tempo had exactly ONE each. A boss song
was always A harmonic minor at 150.25 BPM, offbeat harmony, bass on all four
beats, in the form A-A'1-A'2-B-A''. `render_harmony` and `render_bass` take no
generator, so the whole backing was a pure function of which of three
progressions got drawn.

The form archetypes are the ones the repertoire used, not a set invented to
look varied: ternary, rounded binary, the 32-bar pop form, the linear
A-B-C-D of boss music, and theme-and-variation. An earlier draft had `abab`
and `arch`, which are not archetypes, and omitted `linear` entirely.
"""

from __future__ import annotations

import pytest

from rse.hardware.constraints import ConstraintEngine
from rse.hardware.profile import load_profile
from rse.hardware.tempo import is_playable
from rse.procedural.compose import STYLES, compose
from rse.procedural.variety import (
    BASS_PATTERNS,
    FORM_RETURNS,
    FORM_TEMPLATES,
    HARMONY_STYLES,
    REORCHESTRATIONS,
    STYLE_BASS,
    STYLE_FORMS,
    STYLE_FRAMES,
    STYLE_HARMONY,
    STYLE_TONICS,
    choose,
    fit_template,
    forms_that_fit,
    natural_length,
    section_bars_for,
)

SEEDS = range(900, 920)


def channel(result, channel_id):
    return tuple(
        sorted(
            (e.tick, getattr(e, "pitch", getattr(e, "value", 0)), e.duration, e.velocity)
            for e in result.composition.sounding_events()
            if e.channel_id == channel_id
        )
    )


# --- v1 untouched ---------------------------------------------------------


@pytest.mark.parametrize("style", sorted(STYLES))
def test_variety_off_is_byte_identical(style: str) -> None:
    signature = lambda r: [  # noqa: E731 - local, single use
        (e.channel_id, e.tick, getattr(e, "pitch", getattr(e, "value", 0)),
         e.duration, e.velocity)
        for e in r.composition.sounding_events()
    ]
    plain = compose(style_name=style, seed=800, target_seconds=60.0)
    explicit = compose(style_name=style, seed=800, target_seconds=60.0, variety=None)
    assert signature(plain) == signature(explicit)


def test_an_unknown_variety_is_refused() -> None:
    with pytest.raises(KeyError):
        compose(style_name="boss", seed=1, variety="lots")


# --- the defect, and the fix ----------------------------------------------


@pytest.mark.parametrize("style", sorted(STYLES))
def test_v1_backing_barely_moves(style: str) -> None:
    """Pinned so the defect cannot silently return. Harmony and bass are pure
    functions of the progression, so they have at most as many variants as
    there are progressions."""
    # `sound_design=None`: this pins a COMPOSITION defect, and the expressive
    # default varies velocity per profile, which would be counted as backing
    # variety it is not.
    harmonies = {
        channel(
            compose(
                style_name=style, seed=s, target_seconds=60.0, sound_design=None
            ),
            "P2",
        )
        for s in SEEDS
    }
    assert len(harmonies) <= 6, style


@pytest.mark.parametrize("style", sorted(STYLES))
def test_variety_moves_the_whole_arrangement(style: str) -> None:
    results = [
        compose(style_name=style, seed=s, target_seconds=60.0, variety="full")
        for s in SEEDS
    ]
    for channel_id, floor in (("P2", 12), ("TR", 12), ("P1", 18), ("NO", 18)):
        unique = {channel(r, channel_id) for r in results}
        assert len(unique) >= floor, f"{style}/{channel_id}: {len(unique)}/20"


@pytest.mark.parametrize("style", sorted(STYLES))
def test_variety_moves_key_tempo_and_form(style: str) -> None:
    results = [
        compose(style_name=style, seed=s, target_seconds=60.0, variety="full")
        for s in SEEDS
    ]
    assert len({r.style.tonic for r in results}) >= 3, style
    assert len({round(r.style.tempo_bpm, 2) for r in results}) >= 2, style
    assert len({tuple(s.label for s in r.sections) for r in results}) >= 2, style


@pytest.mark.parametrize("style", sorted(STYLES))
def test_variety_keeps_every_tempo_playable(style: str) -> None:
    """Variety must not reintroduce tempos the NES cannot produce."""
    for seed in SEEDS:
        result = compose(
            style_name=style, seed=seed, target_seconds=60.0, variety="full"
        )
        assert is_playable(result.style.tempo_bpm), (
            f"{style}/{seed}: {result.style.tempo_bpm}"
        )
        assert result.style.frames_per_row in STYLE_FRAMES[style]


@pytest.mark.parametrize("style", sorted(STYLES))
def test_variety_keeps_the_hardware_valid(style: str) -> None:
    engine = ConstraintEngine(load_profile("nes"))
    for seed in SEEDS:
        result = compose(
            style_name=style, seed=seed, target_seconds=60.0, variety="full"
        )
        assert engine.validate(result.composition) == [], f"{style}/{seed}"


def test_variety_is_deterministic() -> None:
    signature = lambda r: [  # noqa: E731 - local, single use
        (e.channel_id, e.tick, getattr(e, "pitch", getattr(e, "value", 0)))
        for e in r.composition.sounding_events()
    ]
    first = compose(style_name="boss", seed=7, target_seconds=60.0, variety="full")
    second = compose(style_name="boss", seed=7, target_seconds=60.0, variety="full")
    assert signature(first) == signature(second)


# --- the archetypes -------------------------------------------------------


def test_the_archetypes_are_the_attested_ones() -> None:
    assert set(FORM_TEMPLATES) == {
        "ternary", "rounded", "ternary_var", "pop32", "linear", "variation",
    }


def test_linear_does_not_restate_and_the_others_do() -> None:
    """The linear form's return IS the loop point. Forcing a closing
    restatement would turn the one non-repeating archetype into a repeating
    one."""
    assert FORM_RETURNS["linear"] is False
    assert FORM_TEMPLATES["linear"][-1].startswith("contrast")
    for name, template in FORM_TEMPLATES.items():
        if FORM_RETURNS[name]:
            assert template[-1] in ("state", "vary"), name


def test_linear_statements_are_distinct() -> None:
    """A-B-C-D, not A-B-A-B: each letter is its own idea."""
    template = FORM_TEMPLATES["linear"]
    assert len(set(template)) == len(template)


def test_rounded_binary_returns_a_variation_not_a_copy() -> None:
    """The one difference between ternary and rounded binary."""
    assert FORM_TEMPLATES["ternary"][-1] == "state"
    assert FORM_TEMPLATES["rounded"][-1] == "vary"
    assert FORM_RETURNS["rounded"] is True


def test_every_archetype_states_a_theme_first() -> None:
    for name, template in FORM_TEMPLATES.items():
        assert template[0] == "state", name


def test_every_archetype_has_a_contrasting_section() -> None:
    for name, template in FORM_TEMPLATES.items():
        assert any(t.startswith("contrast") for t in template), name


def test_shrinking_never_drops_the_contrast() -> None:
    """pop32 trimmed to three sections once became A-A-A: not a shorter
    pop32, but a form with no contrast at all."""
    for name in FORM_TEMPLATES:
        for count in (2, 3, 4, 5, 6):
            shape = fit_template(name, count)
            assert len(shape) == count, (name, count)
            if count >= 3:
                assert any(t.startswith("contrast") for t in shape), (name, count)


def test_the_archetypes_stay_distinct_at_their_own_length() -> None:
    """Deriving the section count from the duration stretched a three-section
    ternary to five, at which point ternary, ternary_var and variation were
    all the same shape. Each archetype keeps its own length instead."""
    shapes = {
        name: tuple(fit_template(name, natural_length(name)))
        for name in FORM_TEMPLATES
    }
    assert len(set(shapes.values())) == len(shapes), shapes


def test_section_length_absorbs_the_clock() -> None:
    """The form is the musical decision; the bar count fits the duration."""
    seconds_per_bar = 4 * 60.0 / 150.25
    for target in (30.0, 60.0, 120.0):
        for name in FORM_TEMPLATES:
            bars = section_bars_for(
                name, target_seconds=target, seconds_per_bar=seconds_per_bar
            )
            length = natural_length(name) * bars * seconds_per_bar
            assert abs(length - target) <= target * 0.45, (name, target, length)


def test_forms_are_filtered_by_what_fits() -> None:
    """A four-section linear form at tension's tempo is 77 s at best, so it
    is not offered for a 60 s request."""
    slow = 4 * 60.0 / 100.16
    fitting = forms_that_fit(
        "tension", target_seconds=60.0, seconds_per_bar=slow, tolerance=10.0
    )
    assert fitting
    assert set(fitting) <= set(STYLE_FORMS["tension"])
    for name in fitting:
        bars = section_bars_for(
            name, target_seconds=60.0, seconds_per_bar=slow
        )
        assert abs(natural_length(name) * bars * slow - 60.0) <= 10.0


def test_a_style_only_offers_its_own_forms() -> None:
    for style, forms in STYLE_FORMS.items():
        assert set(forms) <= set(FORM_TEMPLATES), style
        assert forms, style


# --- section labels match the archetype -----------------------------------


@pytest.mark.parametrize(
    ("form", "labels"),
    [
        ("ternary", "A-B-A"),
        ("rounded", "A-B-A'"),
        ("ternary_var", "A-A'-B-A"),
        ("pop32", "A-A-B-A"),
        ("linear", "A-B-C-D"),
        ("variation", "A-A'-A''-B-A"),
    ],
)
def test_labels_describe_the_form(form: str, labels: str) -> None:
    """A literal restatement carries the opening's own label; a re-voiced one
    gets a prime. Labelling both A' hid the only difference between ternary
    and rounded binary, and numbering states and varies separately produced
    two sections both called A'."""
    import random

    from rse.procedural.compose import plan_archetype

    plans = plan_archetype(
        random.Random(1), STYLES["boss"], form=form, section_bars=8
    )
    assert "-".join(p.label for p in plans) == labels


# --- re-orchestration -----------------------------------------------------


def test_vary_reorchestrates_rather_than_rewriting() -> None:
    """A' in the repertoire was the same sequence data with the instruments
    reassigned or the track transposed -- new melodic material cost memory
    that competed with the samples. Keeping the motif is also what lets the
    theme get established."""
    result = compose(
        style_name="boss", seed=903, target_seconds=60.0, variety="full"
    )
    varied = [
        v for v in result.variations if v.get("reorchestration") is not None
    ]
    assert varied, "expected at least one re-orchestrated section"
    for entry in varied:
        # Same motif as the opening statement; only the orchestration moved.
        assert entry["motif"] == result.variations[0]["motif"]


def test_every_reorchestration_changes_something() -> None:
    from rse.procedural.variety import Reorchestration

    for spec in REORCHESTRATIONS:
        move = Reorchestration(**spec)
        assert (
            move.transpose
            or move.harmony_style
            or move.bass_pattern
            or move.counter_melody
            or move.thin
        ), spec


# --- the pools ------------------------------------------------------------


def test_every_style_has_pools_for_every_axis() -> None:
    for style in STYLES:
        assert len(STYLE_HARMONY[style]) >= 3, style
        assert len(STYLE_BASS[style]) >= 3, style
        assert len(STYLE_TONICS[style]) >= 4, style
        assert len(STYLE_FRAMES[style]) >= 2, style
        assert len(STYLE_FORMS[style]) >= 3, style


def test_pools_only_name_real_figures() -> None:
    for style, names in STYLE_HARMONY.items():
        assert set(names) <= set(HARMONY_STYLES), style
    for style, names in STYLE_BASS.items():
        assert set(names) <= set(BASS_PATTERNS), style


def test_bass_patterns_stay_inside_a_bar() -> None:
    for name, rows in BASS_PATTERNS.items():
        assert rows == tuple(sorted(rows)), name
        assert rows[0] == 0, f"{name} must attack the downbeat"
        assert max(rows) < 16, name


def test_tonics_stay_in_a_sensible_register() -> None:
    for style, tonics in STYLE_TONICS.items():
        for tonic in tonics:
            assert 48 <= tonic <= 72, (style, tonic)


def test_choose_is_reproducible_and_in_range() -> None:
    for style in STYLES:
        first = choose(
            style, seed=11,
            default_harmony=STYLES[style].harmony_style,
            default_bass=STYLES[style].bass_rows,
            default_tonic=STYLES[style].tonic,
            default_frames=STYLES[style].frames_per_row,
        )
        second = choose(
            style, seed=11,
            default_harmony=STYLES[style].harmony_style,
            default_bass=STYLES[style].bass_rows,
            default_tonic=STYLES[style].tonic,
            default_frames=STYLES[style].frames_per_row,
        )
        assert first == second, style
        assert first.harmony_style in STYLE_HARMONY[style]
        assert first.bass_pattern in STYLE_BASS[style]
        assert first.tonic in STYLE_TONICS[style]
        assert first.frames_per_row in STYLE_FRAMES[style]
        assert first.form in STYLE_FORMS[style]


def test_variety_does_not_disturb_the_melody_stream() -> None:
    """Adding variety must not move the voices that were already varying, or
    an A/B would measure two things at once. With the melody generator
    isolated, the melody is identical with variety off and on except for the
    key and tempo the song chose."""
    from rse.procedural import variety as V

    arrangement = V.choose(
        "boss", seed=55,
        default_harmony=STYLES["boss"].harmony_style,
        default_bass=STYLES["boss"].bass_rows,
        default_tonic=STYLES["boss"].tonic,
        default_frames=STYLES["boss"].frames_per_row,
    )
    assert arrangement.tonic in STYLE_TONICS["boss"]
