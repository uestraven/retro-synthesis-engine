"""Instrumentation and expression, and the hardware facts that constrain them.

Audited before this existed: every note of every song carried
`instrument_id=None`, `effects=()`, and a lead velocity of 100 -- every note,
every song, every style. The duty cycle was the renderer's default in all of
them. Forty melodies with forty distinct interval contours were performed by
the same two patches at one fixed volume.

Three hardware facts are pinned here because getting any of them wrong
produces a change that measures well and is inaudible:

1. There are THREE pulse timbres, not four. `DUTY_SEQUENCES[3]` is exactly
   `1 - DUTY_SEQUENCES[1]`.
2. Volume is a 4-bit register, so velocity is quantised to fifteen steps and a
   difference below ~8.5 can change nothing.
3. The triangle has no volume register, so bass dynamics do not exist.
"""

from __future__ import annotations

import numpy as np
import pytest

from rse.procedural.compose import STYLES, compose
from rse.procedural.sound import (
    BASS_VELOCITY,
    COMP_REGISTERS,
    DRY_DUTY,
    LEAD_REGISTERS,
    MAX_REGISTER,
    TIMBRES,
    VOICE_PRESETS,
    choose_voices,
    comp_velocity,
    lead_velocity,
    velocity_for_register,
)
from rse.renderers.nes_apu import DUTY_SEQUENCES, _velocity_to_volume

# --- the hardware facts ---------------------------------------------------


def test_duty_three_is_duty_one_inverted() -> None:
    """So offering it as a fourth instrument would be a choice that changes
    nothing. Phase inversion is audible only against a correlated signal, and
    two independently sequenced channels are not that."""
    assert np.array_equal(DUTY_SEQUENCES[3], 1.0 - DUTY_SEQUENCES[1])


def test_only_three_timbres_are_offered() -> None:
    assert sorted(TIMBRES.values()) == [0, 1, 2]
    assert 3 not in TIMBRES.values()


def test_fifty_percent_duty_has_no_even_harmonics() -> None:
    """What makes it read as hollow, and the reason it is a real third
    timbre rather than a shade of 25%."""
    sequence = DUTY_SEQUENCES[2]
    spectrum = np.abs(np.fft.rfft(sequence - sequence.mean()))
    # Bins 2 and 4 of an 8-point transform are the even harmonics.
    assert spectrum[2] < 1e-6
    assert spectrum[4] < 1e-6
    # 25% keeps them.
    quarter = DUTY_SEQUENCES[1]
    assert np.abs(np.fft.rfft(quarter - quarter.mean()))[2] > 1e-6


def test_velocity_is_quantised_to_fifteen_registers() -> None:
    """The reason expression is authored in register space. A five-unit accent
    does nothing: 98 through 104 all land on 12."""
    assert len({_velocity_to_volume(v) for v in range(1, 128)}) == MAX_REGISTER
    assert _velocity_to_volume(98) == _velocity_to_volume(104)
    assert _velocity_to_volume(100) == _velocity_to_volume(102)


def test_velocity_for_register_round_trips() -> None:
    for register in range(1, MAX_REGISTER + 1):
        assert _velocity_to_volume(velocity_for_register(register)) == register


def test_the_triangle_has_no_dynamics() -> None:
    """Pinned so nobody later "fixes" the bass by giving it velocities that
    the hardware ignores."""
    from rse.hardware.profile import load_profile

    profile = load_profile("nes")
    triangle = next(c for c in profile.channels if c.channel_id == "TR")
    assert triangle.volume_levels == 1
    pulse = next(c for c in profile.channels if c.channel_id == "P1")
    assert pulse.volume_levels == 16


# --- the dynamic plan -----------------------------------------------------


def test_every_lead_level_is_its_own_register() -> None:
    """A plan whose levels collapse onto one register is inaudible."""
    registers = {_velocity_to_volume(velocity_for_register(r))
                 for r in LEAD_REGISTERS.values()}
    assert len(registers) == len(set(LEAD_REGISTERS.values()))


def test_every_comp_level_is_its_own_register() -> None:
    registers = {_velocity_to_volume(velocity_for_register(r))
                 for r in COMP_REGISTERS.values()}
    assert len(registers) == len(set(COMP_REGISTERS.values()))


def test_the_lead_stays_above_the_accompaniment() -> None:
    """A 4-bit register has little range; spending it on comp accents would
    put the comp over the lead."""
    assert min(LEAD_REGISTERS.values()) > max(COMP_REGISTERS.values())


def test_accents_land_on_the_strongest_level() -> None:
    accent = velocity_for_register(LEAD_REGISTERS["accent"])
    for kwargs in (
        {"section_start": True},
        {"cadence": True},
        {"phrase_start": True},
    ):
        assert lead_velocity(row_in_bar=7, **kwargs) == accent


def test_the_beat_hierarchy_descends() -> None:
    downbeat = lead_velocity(row_in_bar=0)
    eighth = lead_velocity(row_in_bar=2)
    sixteenth = lead_velocity(row_in_bar=1)
    assert _velocity_to_volume(downbeat) > _velocity_to_volume(eighth)
    assert _velocity_to_volume(eighth) > _velocity_to_volume(sixteenth)


def test_the_comp_follows_the_beat_only() -> None:
    """It takes no phrase or cadence accent, so it cannot compete with the
    lead for the register range."""
    assert comp_velocity(row_in_bar=0) > comp_velocity(row_in_bar=2)
    assert comp_velocity(row_in_bar=2) > comp_velocity(row_in_bar=1)
    assert max(
        comp_velocity(row_in_bar=r) for r in range(16)
    ) < min(lead_velocity(row_in_bar=r) for r in range(16))


def test_bass_velocity_is_unchanged_from_v1() -> None:
    assert BASS_VELOCITY == 90


# --- instrumentation ------------------------------------------------------


def test_every_preset_is_complementary() -> None:
    """Lead and comp share two monophonic channels of the same waveform. The
    same duty on both is the one combination that muddies the arrangement."""
    for style, presets in VOICE_PRESETS.items():
        for lead, comp in presets:
            assert lead != comp, f"{style}: {lead}/{comp}"
            assert lead in TIMBRES and comp in TIMBRES, style


def test_every_style_has_voice_presets() -> None:
    for style in STYLES:
        assert len(VOICE_PRESETS[style]) >= 3, style


def test_tension_inverts_the_boss_choice() -> None:
    """A thin lead over a rounded bed, rather than the reverse. The styles
    should not all reach for the same instrument."""
    assert all(lead == "bright" or comp == "hollow"
               for lead, comp in VOICE_PRESETS["tension"])


def test_voices_are_deterministic_and_per_song() -> None:
    for style in STYLES:
        first = choose_voices(style, seed=11)
        assert first == choose_voices(style, seed=11)
        assert (first.lead_timbre, first.comp_timbre) in VOICE_PRESETS[style]


def test_voices_vary_across_seeds() -> None:
    for style in STYLES:
        chosen = {
            (choose_voices(style, seed=s).lead_timbre,
             choose_voices(style, seed=s).comp_timbre)
            for s in range(900, 940)
        }
        assert len(chosen) >= 2, style


def test_the_duty_map_names_real_channels() -> None:
    design = choose_voices("boss", seed=1)
    assert set(design.duty_by_channel) == {"P1", "P2"}
    assert all(0 <= d <= 2 for d in design.duty_by_channel.values())


# --- the score is untouched -----------------------------------------------


@pytest.mark.parametrize("style", sorted(STYLES))
def test_sound_design_changes_no_note(style: str) -> None:
    """The whole point: the same score, performed differently. Pitch, onset,
    duration, channel and structure must be identical."""
    def score(result):
        return sorted(
            (e.channel_id, e.tick, getattr(e, "pitch", getattr(e, "value", 0)),
             e.duration)
            for e in result.composition.sounding_events()
        )

    dry = compose(style_name=style, seed=800, target_seconds=60.0, sound_design=None)
    expressive = compose(
        style_name=style, seed=800, target_seconds=60.0, sound_design="expressive"
    )
    assert score(dry) == score(expressive)
    assert [s.label for s in dry.sections] == [s.label for s in expressive.sections]
    assert dry.style.tonic == expressive.style.tonic
    assert dry.style.tempo_bpm == expressive.style.tempo_bpm


@pytest.mark.parametrize("style", sorted(STYLES))
def test_expressive_is_the_default(style: str) -> None:
    """Promoted after the listening test accepted it. The dry render stays
    reachable because the frozen baselines are expressed in it."""
    signature = lambda r: [  # noqa: E731 - local, single use
        (e.channel_id, e.tick, getattr(e, "pitch", getattr(e, "value", 0)),
         e.duration, e.velocity)
        for e in r.composition.sounding_events()
    ]
    default = compose(style_name=style, seed=800, target_seconds=60.0)
    expressive = compose(
        style_name=style, seed=800, target_seconds=60.0, sound_design="expressive"
    )
    dry = compose(style_name=style, seed=800, target_seconds=60.0, sound_design=None)
    assert signature(default) == signature(expressive)
    assert signature(default) != signature(dry), "the two arms must differ"
    # And the dry arm is still exactly v1: one velocity per pitched channel.
    for channel_id, velocity in (("P1", 100), ("TR", 90)):
        assert {
            e.velocity
            for e in dry.composition.sounding_events()
            if e.channel_id == channel_id
        } == {velocity}, channel_id


def test_an_unknown_sound_design_is_refused() -> None:
    with pytest.raises(KeyError):
        compose(style_name="boss", seed=1, sound_design="orchestral")


def test_sound_design_cannot_perturb_composition() -> None:
    """Its generator is seeded separately, so turning it on must not move a
    single compositional decision."""
    dry = compose(style_name="boss", seed=5, target_seconds=60.0, sound_design=None)
    expressive = compose(
        style_name="boss", seed=5, target_seconds=60.0, sound_design="expressive"
    )
    assert dry.motif == expressive.motif
    assert [s.progression for s in dry.sections] == [
        s.progression for s in expressive.sections
    ]


# --- it actually reaches the audio ----------------------------------------


@pytest.mark.parametrize("style", sorted(STYLES))
def test_shaped_velocities_reach_distinct_registers(style: str) -> None:
    """Measured at the APU, not in the event schema. The event schema having a
    velocity field proves nothing about the audio."""
    result = compose(
        style_name=style, seed=800, target_seconds=30.0, sound_design="expressive"
    )
    for channel_id in ("P1", "P2"):
        velocities = sorted(
            {e.velocity for e in result.composition.sounding_events()
             if e.channel_id == channel_id}
        )
        registers = {_velocity_to_volume(v) for v in velocities}
        assert len(registers) == len(velocities), (
            f"{style}/{channel_id}: {velocities} collapse to {registers}"
        )


def test_a_duty_change_changes_the_rendered_stem() -> None:
    """End to end, on the isolated P1 stem. A whole-mix spectral measure is
    too crude here -- it reported 12.5% and 25% as indistinguishable when
    harmonic analysis shows them 0.18 apart."""
    from rse.events.convert import composition_to_intake
    from rse.hardware.constraints import ConstraintEngine
    from rse.hardware.profile import load_profile
    from rse.renderers.base import RenderSettings, notes_from_record
    from rse.renderers.nes_apu import NesApuRenderer

    profile = load_profile("nes")
    result = compose(
        style_name="boss", seed=800, target_seconds=20.0, sound_design=None
    )
    playable, _ = ConstraintEngine(profile).apply(result.composition)
    notes = notes_from_record(composition_to_intake(playable, profile=profile))

    def ratio(duty: int) -> float:
        renderer = NesApuRenderer(profile, duty_by_channel={"P1": duty})
        rendered = renderer.render(notes, RenderSettings(tail_sec=0.0))
        stem = np.asarray(rendered.stems["P1"], dtype=float)
        spectrum = np.abs(np.fft.rfft(stem * np.hanning(len(stem))))
        freqs = np.fft.rfftfreq(len(stem), 1 / rendered.sample_rate)
        low = spectrum[(freqs > 80) & (freqs < 1200)].sum()
        high = spectrum[(freqs > 1200) & (freqs < 6000)].sum()
        return high / (low or 1)

    hollow, classic, bright = ratio(2), ratio(1), ratio(0)
    assert abs(classic - hollow) / hollow > 0.05
    assert abs(bright - hollow) / hollow > 0.05


def test_no_clipping_and_hardware_stays_valid() -> None:
    from rse.hardware.constraints import ConstraintEngine
    from rse.hardware.profile import load_profile

    engine = ConstraintEngine(load_profile("nes"))
    for style in STYLES:
        result = compose(
            style_name=style, seed=800, target_seconds=30.0,
            sound_design="expressive",
        )
        assert engine.validate(result.composition) == [], style
        for event in result.composition.sounding_events():
            assert 1 <= event.velocity <= 127, style
            assert 1 <= _velocity_to_volume(event.velocity) <= MAX_REGISTER


def test_the_dry_duty_matches_the_renderer_default() -> None:
    """So the control arm is reproducible rather than approximated."""
    from rse.renderers.nes_apu import DEFAULT_DUTY

    assert DRY_DUTY == DEFAULT_DUTY


# --- mood was an inert control --------------------------------------------


def test_mood_was_inert_and_is_not_anymore() -> None:
    """The defect: the interpreter read a mood into `key_mode`, which
    `compose` never consults -- it uses the style's own mode -- so "a dark
    theme" and "a bright theme" were byte-identical once a style was chosen.
    """
    from rse.song.generate import generate_song

    def signature(result):
        return [
            (e.channel_id, e.tick, getattr(e, "pitch", getattr(e, "value", 0)),
             e.velocity)
            for e in result.composition.sounding_events()
        ]

    duties = set()
    for mood in ("dark", "mysterious", "bright", "peaceful", "sad"):
        result = generate_song(
            "a theme", seed=7, style="boss", mood=mood, target_seconds=30.0
        )
        duties.add(tuple(sorted(result.meta["duty_by_channel"].items())))
    assert len(duties) >= 3, f"mood still barely moves the sound: {duties}"


def test_mood_leaves_the_notes_alone() -> None:
    """It changes instrumentation, not composition. The mode still comes from
    the style."""
    from rse.song.generate import generate_song

    def score(result):
        return sorted(
            (e.channel_id, e.tick, getattr(e, "pitch", getattr(e, "value", 0)),
             e.duration)
            for e in result.composition.sounding_events()
        )

    dark = generate_song(
        "a theme", seed=7, style="boss", mood="dark", target_seconds=30.0
    )
    bright = generate_song(
        "a theme", seed=7, style="boss", mood="bright", target_seconds=30.0
    )
    assert score(dark) == score(bright)
    assert dark.meta["mode"] == bright.meta["mode"] == "harmonic_minor"


def test_the_report_credits_the_mood_with_what_it_did() -> None:
    """It used to claim "read as minor key", which was true of nothing: the
    key came from the preset and the mood changed no note."""
    from rse.song.generate import finish_report, generate_song

    result = generate_song("a haunting dungeon theme", seed=7, target_seconds=60.0)
    assert result.meta["mode"] == "phrygian"
    report = finish_report(result)
    stated = {
        item["text"]: item["detail"]
        for bucket in report.values()
        for item in bucket
    }
    assert "haunting" in stated
    assert "instrument profile" in stated["haunting"]
    assert "minor key" not in stated["haunting"]


def test_every_mood_maps_to_a_profile_that_exists() -> None:
    from rse.procedural.sound import MOOD_PROFILES, SOUND_PROFILES

    assert set(MOOD_PROFILES.values()) <= set(SOUND_PROFILES)


def test_an_unknown_profile_is_refused() -> None:
    with pytest.raises(KeyError):
        choose_voices("boss", seed=1, profile="orchestral")


def test_an_explicit_profile_outranks_a_mood() -> None:
    design = choose_voices("boss", seed=1, profile="eerie", mood="dark")
    assert design.profile == "eerie"
    assert design.reason == "profile"


def test_an_unmapped_mood_falls_back_to_the_style() -> None:
    """Rather than erroring deep in the composer: the API validates moods, so
    anything reaching here unmapped should degrade, not crash."""
    design = choose_voices("boss", seed=1, mood="peckish")
    assert design.reason == "style"
    assert (design.lead_timbre, design.comp_timbre) in VOICE_PRESETS["boss"]
