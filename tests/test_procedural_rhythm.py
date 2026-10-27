"""Harmonic rhythm changes pacing without changing the frozen baseline.

The 24 pieces the listener called "fantastic" are the thing that worked, so
stage-1 expansion is opt-in: `compose()` with no `harmonic_rhythm` must be
byte-identical to those, and `tests/data/procedural_v1_baseline.json` holds
their event hashes to prove it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from rse.hardware.constraints import ConstraintEngine
from rse.hardware.profile import load_profile
from rse.procedural.compose import (
    ONE_PER_BAR,
    RHYTHMS,
    ROWS_PER_BAR,
    STYLES,
    compose,
)

BASELINE = json.loads(
    (Path(__file__).parent / "data" / "procedural_v1_baseline.json").read_text()
)


def signature(result) -> str:
    events = [
        (
            e.channel_id,
            e.tick,
            getattr(e, "pitch", getattr(e, "value", 0)),
            e.duration,
            e.velocity,
        )
        for e in result.composition.sounding_events()
    ]
    return hashlib.sha256(json.dumps(events, sort_keys=True).encode()).hexdigest()[:16]


@pytest.mark.parametrize("key", sorted(BASELINE))
def test_v1_output_is_unchanged(key: str) -> None:
    """The baseline is the result; expansion must not move it."""
    style, _, seed = key.rpartition("_s")
    # `sound_design=None` on purpose. The baseline's job is to prove the
    # COMPOSITION has not moved, and the signature includes velocity, which
    # is now a performance decision rather than a compositional one. Pinning
    # the expressive render here would make every future sound-design change
    # look like a composition regression.
    result = compose(
        style_name=style, seed=int(seed), target_seconds=60.0, sound_design=None
    )
    assert signature(result) == BASELINE[key]


def test_default_rhythm_is_one_per_bar() -> None:
    result = compose(style_name="boss", seed=1)
    assert result.rhythm is ONE_PER_BAR


def test_default_draws_no_extra_random_number() -> None:
    """An extra draw would shift every melody downstream of it.

    Explicitly requesting one_per_bar must give the same music as the
    default, which is only true if neither consumes the generator.
    """
    assert signature(compose(style_name="boss", seed=3)) == signature(
        compose(style_name="boss", seed=3, harmonic_rhythm="one_per_bar")
    )


@pytest.mark.parametrize("name", sorted(RHYTHMS))
def test_every_rhythm_covers_the_section_exactly(name: str) -> None:
    """The plan must tile the bars with no gap and no overrun."""
    plan = RHYTHMS[name].plan(4, 8)
    assert plan[0][1] == 0
    total = 8 * ROWS_PER_BAR
    covered = 0
    for _index, start, span in plan:
        assert start == covered, f"gap or overlap at row {start}"
        assert span > 0
        covered += span
    assert covered == total


@pytest.mark.parametrize("name", sorted(RHYTHMS))
@pytest.mark.parametrize("style", sorted(STYLES))
def test_every_rhythm_is_playable_in_every_style(style: str, name: str) -> None:
    """Pacing changes must not break the hardware contract."""
    result = compose(style_name=style, seed=42, harmonic_rhythm=name)
    engine = ConstraintEngine(load_profile("nes"))
    assert engine.validate(result.composition) == []
    assert result.composition.sounding_events()


def test_rhythms_produce_different_chord_change_counts() -> None:
    """two_per_bar really is twice as fast as one_per_bar."""
    counts = {}
    for name in ("two_bar_hold", "one_per_bar", "two_per_bar"):
        result = compose(style_name="boss", seed=9, harmonic_rhythm=name)
        counts[name] = result.variations[0]["n_chord_changes"]
    assert counts["two_bar_hold"] < counts["one_per_bar"] < counts["two_per_bar"]


def test_rhythm_changes_the_music() -> None:
    """Different pacing, same progression, different piece."""
    a = signature(compose(style_name="boss", seed=9, harmonic_rhythm="one_per_bar"))
    b = signature(compose(style_name="boss", seed=9, harmonic_rhythm="two_per_bar"))
    assert a != b


def test_pedal_holds_the_bass_on_one_root() -> None:
    """A tonic pedal means the triangle stops following the chords."""
    pedalled = compose(style_name="boss", seed=9, harmonic_rhythm="tonic_pedal")
    moving = compose(style_name="boss", seed=9, harmonic_rhythm="two_per_bar")
    bass = lambda r: {  # noqa: E731 - local, single use
        e.pitch for e in r.composition.sounding_events() if e.channel_id == "TR"
    }
    assert len(bass(pedalled)) < len(bass(moving)), (
        "a pedal should use fewer distinct bass pitches than moving harmony"
    )


def test_melody_notes_never_cross_a_chord_change() -> None:
    """A note held across a chord change is the clash the shared harmonic
    plan exists to prevent."""
    for name in sorted(RHYTHMS):
        result = compose(style_name="action", seed=17, harmonic_rhythm=name)
        plan = RHYTHMS[name].plan(4, 8)
        boundaries = {start for _i, start, _s in plan}
        bar_rows = {b * ROWS_PER_BAR for b in range(40)}
        for event in result.composition.sounding_events():
            if event.channel_id != "P1":
                continue
            start_row = event.tick // 120
            end_row = (event.tick + event.duration) // 120
            crossed = {
                r
                for r in range(start_row + 1, end_row)
                if (r % (8 * ROWS_PER_BAR)) in boundaries and r not in bar_rows
            }
            assert not crossed, f"{name}: note at row {start_row} crosses {crossed}"


def test_auto_picks_from_the_styles_vocabulary() -> None:
    for style, spec in STYLES.items():
        result = compose(style_name=style, seed=77, harmonic_rhythm="auto")
        assert result.rhythm.name in spec.rhythms


def test_unknown_rhythm_is_rejected() -> None:
    with pytest.raises(KeyError, match="unknown harmonic rhythm"):
        compose(style_name="boss", seed=1, harmonic_rhythm="syncopated_waltz")


def test_the_melody_never_attacks_an_odd_row() -> None:
    """The defect the listener found by analysing the MIDI: "just a random
    note added into the melody".

    `vary_motif`'s ornament branch halved the chosen note's length without
    checking it was long enough, so a two-row note became two ONE-row notes
    and the second landed on an odd row -- the only off-eighth attack in an
    otherwise even-grid piece, once per bar, as a ~100 ms blip.

    Every style's melody rhythms contain twos, so it fired on 47% of ornament
    positions and affected 54 of 240 songs across six styles: `action` 23 of
    40 and `boss` 19 of 40, the two styles heard most.
    """
    from rse.procedural.compose import TICKS_PER_BAR, TICKS_PER_ROW

    offenders = []
    for style in sorted(STYLES):
        for seed in range(900, 930):
            result = compose(
                style_name=style, seed=seed, target_seconds=60.0,
                sound_design=None,
            )
            for event in result.composition.sounding_events():
                if event.channel_id != "P1":
                    continue
                row = (event.tick % TICKS_PER_BAR) // TICKS_PER_ROW
                if row % 2:
                    offenders.append((style, seed, row, event.pitch))
    assert offenders == [], f"{len(offenders)} off-grid melody notes"


def test_an_ornament_never_produces_a_one_row_note() -> None:
    """The mechanism, pinned directly."""
    import random

    from rse.procedural.compose import make_motif, vary_motif

    for style in STYLES.values():
        for seed in range(0, 300):
            motif = make_motif(random.Random(seed), style)
            varied = vary_motif(random.Random(seed + 7777), motif)
            assert min(varied.rhythm) >= 2, (
                f"{style.name}/{seed}: {varied.rhythm}"
            )


def test_a_variation_still_varies() -> None:
    """The guard must not turn every ornament into a no-op."""
    import random

    from rse.procedural.compose import make_motif, vary_motif

    style = STYLES["boss"]
    changed = 0
    for seed in range(0, 120):
        motif = make_motif(random.Random(seed), style)
        varied = vary_motif(random.Random(seed + 31), motif)
        if (varied.degrees, varied.rhythm) != (motif.degrees, motif.rhythm):
            changed += 1
    assert changed >= 100, f"only {changed}/120 variations changed anything"
