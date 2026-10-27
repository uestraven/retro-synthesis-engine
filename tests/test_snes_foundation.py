"""The SNES foundation: profile, role/voice separation, chords and voicings.

Three things are being pinned here.

**The NES path is untouched.** Every one of these abstractions is additive. The
NES composer constructs no `ChordIntent`, allocates no voices, and keeps
writing `channel_id="P1"` directly. Its 24 frozen baselines and its expressive
rendering are asserted elsewhere and must stay green.

**Declaration is not implementation.** `snes.json` describes hardware that the
renderer does not yet support. Its `capabilities.implemented` list is empty on
purpose, and a capability may not move into it until a renderer honours it.

**Harmonic intent is not the notes played.** Cmaj9 is five pitch classes and a
usable SNES voicing of it is four notes. The planner has to shed the right one.
"""

from __future__ import annotations

from itertools import pairwise

import pytest

from rse.hardware.profile import ProfileError, load_profile
from rse.harmony.intent import (
    OMISSION_ORDER,
    ChordIntent,
    Function,
    Quality,
    Seventh,
    Tone,
)
from rse.harmony.voicing import (
    VoicingRequest,
    bass_pitch_for,
    common_tones,
    fits,
    movement,
    plan_progression,
    plan_voicing,
)
from rse.platform.roles import (
    DEFAULT_PRIORITY,
    NES_VOICES,
    PartNote,
    Role,
    VoiceAllocator,
    concurrency,
    nes_voice_for,
)

# --- step 1: the platform profile ----------------------------------------


def test_the_snes_profile_loads_through_the_existing_loader() -> None:
    """No new loader. The audit's central finding was that `ChannelSpec`
    already carries what a sampled voice needs."""
    profile = load_profile("snes")
    assert profile.profile_id == "snes"
    assert len(profile.channels) == 8


def test_the_eight_voices_are_interchangeable() -> None:
    """Not named for musical roles. Assigning a role to a voice is the
    arranger's job, and a voice is reused by different parts over time."""
    profile = load_profile("snes")
    assert [c.channel_id for c in profile.channels] == [
        f"V{i}" for i in range(1, 9)
    ]
    for channel in profile.channels:
        assert channel.role == "sample_voice"
        assert channel.type == "sample"
        assert channel.max_simultaneous_notes == 1


def test_every_voice_declares_the_sample_parameters() -> None:
    profile = load_profile("snes")
    for channel in profile.channels:
        for param in ("sample_id", "root_note", "attack", "release",
                      "volume_left", "volume_right", "echo_send", "priority"):
            assert param in channel.instrument_params, param


def test_nothing_is_advertised_as_working_yet() -> None:
    """The instruction: a capability must not be advertised as functional
    until the renderer supports it."""
    profile = load_profile("snes")
    capabilities = profile.raw["capabilities"]
    assert capabilities["implemented"] == []
    assert "sample_playback" in capabilities["planned"]
    assert profile.raw["renderer"]["implemented"] is False
    assert profile.raw["modes"]["hardware"]["implemented"] is False


def test_the_two_modes_are_distinguished() -> None:
    """Preview is not hardware-exact and must never be labelled as such."""
    modes = load_profile("snes").raw["modes"]
    assert modes["preview"]["hardware_exact"] is False
    assert modes["hardware"]["hardware_exact"] is True


def test_the_aram_budget_is_declared_with_its_geometry() -> None:
    """BRR is 9 bytes per 16 decoded samples, and the echo buffer is charged
    against the same 64 KiB as the samples."""
    aram = load_profile("snes").raw["audio_ram"]
    assert aram["total_bytes"] == 65536
    assert aram["brr_block_bytes"] == 9
    assert aram["brr_samples_per_block"] == 16
    assert aram["echo_buffer_bytes_per_16ms"] > 0


def test_output_is_stereo_at_the_nominal_dsp_rate() -> None:
    """32 kHz is the DSP's output rate, not a required asset rate."""
    output = load_profile("snes").raw["output"]
    assert output["channels"] == 2
    assert output["nominal_sample_rate_hz"] == 32000


def test_percussion_is_sampled_not_a_noise_channel() -> None:
    percussion = load_profile("snes").raw["percussion"]
    assert percussion["sampled"] is True
    assert percussion["voices"] == "shared"


def test_the_nes_profile_still_loads_unchanged() -> None:
    profile = load_profile("nes")
    assert [c.channel_id for c in profile.channels] == ["P1", "P2", "TR", "NO"]


def test_an_unknown_profile_still_fails() -> None:
    with pytest.raises(ProfileError):
        load_profile("gameboy")


# --- step 2: roles independent of voices ---------------------------------


def test_the_nes_mapping_is_exactly_what_it_always_was() -> None:
    """The compatibility path. NES channels are not interchangeable -- the
    triangle is the bass because it cannot be anything else."""
    assert nes_voice_for(Role.LEAD) == "P1"
    assert nes_voice_for(Role.HARMONY) == "P2"
    assert nes_voice_for(Role.BASS) == "TR"
    assert nes_voice_for(Role.PERCUSSION) == "NO"
    assert set(NES_VOICES) == set(Role)


def test_a_voice_is_reused_by_different_parts_over_time() -> None:
    """The property NES does not have and SNES depends on."""
    notes = [
        PartNote(role=Role.LEAD, tick=0, duration=10, value=72),
        PartNote(role=Role.PERCUSSION, tick=10, duration=10, value=1),
        PartNote(role=Role.BASS, tick=20, duration=10, value=40),
    ]
    report = VoiceAllocator(["V1"]).allocate(notes)
    assert len(report.assignments) == 3
    assert {a.voice for a in report.assignments} == {"V1"}
    assert report.peak_voices == 1
    assert report.dropped == 0


def test_release_tails_occupy_the_voice() -> None:
    """A naive allocator counts only `duration` and over-subscribes. Two notes
    whose sounding durations do not overlap still collide if the first has a
    release."""
    notes = [
        PartNote(role=Role.PAD, tick=0, duration=40, value=60, release=60),
        PartNote(role=Role.PAD, tick=50, duration=40, value=62, release=60),
    ]
    assert concurrency(notes) == 2
    report = VoiceAllocator(["V1"]).allocate(notes)
    assert report.stolen + report.dropped == 1, report.summary()


def test_a_part_may_need_several_voices_at_once() -> None:
    """A doubled or chordal part is one role across multiple voices."""
    chord = [
        PartNote(role=Role.HARMONY, tick=0, duration=20, value=pitch)
        for pitch in (60, 64, 67)
    ]
    report = VoiceAllocator(["V1", "V2", "V3"]).allocate(chord)
    assert len({a.voice for a in report.assignments}) == 3
    assert report.peak_voices == 3


def test_eight_voices_is_enforced_and_overflow_is_reported() -> None:
    """Never a silent drop. That is the defect this module exists to make
    impossible."""
    notes = [
        PartNote(role=Role.ARPEGGIO, tick=0, duration=50, value=60 + i)
        for i in range(12)
    ]
    report = VoiceAllocator([f"V{i}" for i in range(1, 9)]).allocate(notes)
    assert report.peak_voices == 8
    assert len(report.assignments) == 8
    assert report.dropped == 4
    assert len(report.displacements) == 4
    for displacement in report.displacements:
        assert displacement.reason == "dropped"
        assert displacement.role is Role.ARPEGGIO


def test_priority_decides_who_survives() -> None:
    """The lead and the harmonic foundation outlive decoration."""
    notes = [
        PartNote(role=role, tick=0, duration=50, value=60)
        for role in (Role.ACCENT, Role.ARPEGGIO, Role.LEAD, Role.BASS)
    ]
    report = VoiceAllocator(["V1", "V2"]).allocate(notes)
    survived = {a.note.role for a in report.assignments}
    assert survived == {Role.LEAD, Role.BASS}
    lost = {d.role for d in report.displacements}
    assert lost == {Role.ACCENT, Role.ARPEGGIO}


def test_a_higher_priority_note_steals_and_the_theft_is_reported() -> None:
    notes = [
        PartNote(role=Role.ACCENT, tick=0, duration=100, value=60),
        PartNote(role=Role.LEAD, tick=10, duration=50, value=72),
    ]
    report = VoiceAllocator(["V1"]).allocate(notes)
    assert report.stolen == 1
    theft = report.displacements[0]
    assert theft.role is Role.ACCENT
    assert theft.taken_by is Role.LEAD
    assert theft.truncated_ticks == 90


def test_allocation_is_deterministic_regardless_of_input_order() -> None:
    notes = [
        PartNote(role=Role.HARMONY, tick=0, duration=20, value=64),
        PartNote(role=Role.LEAD, tick=0, duration=20, value=72),
        PartNote(role=Role.BASS, tick=0, duration=20, value=40),
    ]
    first = VoiceAllocator(["V1", "V2", "V3"]).allocate(notes)
    second = VoiceAllocator(["V1", "V2", "V3"]).allocate(list(reversed(notes)))
    assert [(a.voice, a.note.value) for a in first.assignments] == [
        (a.voice, a.note.value) for a in second.assignments
    ]


def test_voices_can_be_reserved_for_sound_effects() -> None:
    """Game-ready mode: music gets what is left, and the reservation is a
    policy rather than a hardware fact."""
    notes = [
        PartNote(role=Role.ARPEGGIO, tick=0, duration=50, value=60 + i)
        for i in range(8)
    ]
    report = VoiceAllocator(
        [f"V{i}" for i in range(1, 9)], reserved=2
    ).allocate(notes)
    assert report.voice_count == 6
    assert report.peak_voices == 6
    assert report.dropped == 2


def test_reserving_every_voice_is_refused() -> None:
    with pytest.raises(ValueError):
        VoiceAllocator(["V1", "V2"], reserved=2)


def test_the_default_priority_covers_every_role() -> None:
    assert set(DEFAULT_PRIORITY) == set(Role)


# --- step 3: chord intent ------------------------------------------------


def test_the_worked_example_is_five_pitch_classes() -> None:
    chord = ChordIntent(
        root=0, quality=Quality.MAJOR, seventh=Seventh.MAJOR, extensions=(9,)
    )
    assert chord.symbol == "Cmaj9"
    assert sorted(chord.pitch_classes()) == [0, 2, 4, 7, 11]  # C D E G B
    assert chord.tones() == {
        Tone.ROOT: 0, Tone.THIRD: 4, Tone.FIFTH: 7,
        Tone.SEVENTH: 11, Tone.NINTH: 14,
    }


@pytest.mark.parametrize(
    ("kwargs", "symbol"),
    [
        ({"root": 0, "seventh": Seventh.MAJOR}, "Cmaj7"),
        ({"root": 7, "seventh": Seventh.MINOR}, "G7"),
        ({"root": 7, "seventh": Seventh.MINOR, "extensions": (9,)}, "G9"),
        (
            {"root": 2, "quality": Quality.MINOR, "seventh": Seventh.MINOR,
             "extensions": (9, 11)},
            "Dm11",
        ),
        ({"root": 0, "quality": Quality.SUSPENDED_4}, "Csus4"),
        ({"root": 7, "seventh": Seventh.MINOR, "alterations": {5: -1}}, "G7b5"),
        (
            {"root": 0, "seventh": Seventh.MAJOR, "extensions": (9,),
             "inversion": 1},
            "Cmaj9/E",
        ),
    ],
)
def test_chord_symbols(kwargs: dict, symbol: str) -> None:
    assert ChordIntent(**kwargs).symbol == symbol


def test_the_fifth_is_shed_first_and_the_third_last() -> None:
    """The omission order is about function: the third and seventh define the
    quality, the fifth carries no information once a root is sounding."""
    assert OMISSION_ORDER[0] is Tone.FIFTH
    assert OMISSION_ORDER[-1] is Tone.THIRD
    assert OMISSION_ORDER[-2] is Tone.SEVENTH
    # The root is droppable, but only after the genuinely redundant tones.
    assert OMISSION_ORDER.index(Tone.ROOT) > OMISSION_ORDER.index(Tone.FIFTH)
    assert OMISSION_ORDER.index(Tone.ROOT) < OMISSION_ORDER.index(Tone.THIRD)


def test_an_inversion_names_the_bass_member() -> None:
    chord = ChordIntent(root=0, seventh=Seventh.MINOR)
    assert chord.bass_tone() is Tone.ROOT
    assert ChordIntent(root=0, seventh=Seventh.MINOR, inversion=1).bass_tone() is (
        Tone.THIRD
    )
    assert ChordIntent(root=0, seventh=Seventh.MINOR, inversion=3).bass_tone() is (
        Tone.SEVENTH
    )


def test_borrowing_and_function_are_carried_not_inferred() -> None:
    """A chromatic chord with no explanation is unreportable. Minor iv in a
    major key says where it came from."""
    chord = ChordIntent(
        root=5, quality=Quality.MINOR, function=Function.SUBDOMINANT,
        borrowed_from="parallel minor", degree=3,
    )
    assert chord.borrowed_from == "parallel minor"
    assert chord.function is Function.SUBDOMINANT
    assert chord.degree == 3
    assert chord.to_dict()["borrowed_from"] == "parallel minor"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"root": 99},
        {"root": 0, "extensions": (11,)},          # 11th implies a 7th
        {"root": 0, "inversion": 9},
        {"root": 0, "alterations": {7: -1}},        # no seventh to alter
        {"root": 0, "inversion": 3},                # no seventh to invert onto
        {"root": 0, "extensions": (6,)},            # not an extension
    ],
)
def test_impossible_chords_are_refused_at_construction(kwargs: dict) -> None:
    """Validation used to live only in the lazy `tones()`, so a chord with an
    alteration naming a member it did not have constructed happily and failed
    somewhere downstream."""
    with pytest.raises(ValueError):
        ChordIntent(**kwargs)


def test_an_added_ninth_triad_is_legitimate() -> None:
    """Cadd9 has no seventh and is not an error."""
    chord = ChordIntent(root=0, extensions=(9,))
    assert chord.seventh is Seventh.NONE
    assert Tone.NINTH in chord.tones()


# --- step 4: the voicing planner -----------------------------------------


def test_cmaj9_in_three_voices_omits_the_fifth() -> None:
    """The specified behaviour, exactly: bass C, harmony E-B-D, no G."""
    chord = ChordIntent(root=0, seventh=Seventh.MAJOR, extensions=(9,))
    bass = bass_pitch_for(chord)
    voicing = plan_voicing(
        VoicingRequest(intent=chord, max_voices=3, bass=bass)
    )
    assert voicing.omitted == (Tone.FIFTH,)
    assert voicing.tones == (Tone.THIRD, Tone.SEVENTH, Tone.NINTH)
    assert [p % 12 for p in voicing.pitches] == [4, 11, 2]  # E B D
    assert bass % 12 == 0  # C
    assert voicing.voice_count == 3
    assert voicing.keeps_identity


def test_degradation_sheds_colour_before_quality() -> None:
    chord = ChordIntent(root=0, seventh=Seventh.MAJOR, extensions=(9,))
    bass = bass_pitch_for(chord)
    shed: dict[int, tuple[Tone, ...]] = {}
    for voices in (4, 3, 2, 1):
        voicing = plan_voicing(
            VoicingRequest(intent=chord, max_voices=voices, bass=bass)
        )
        shed[voices] = voicing.omitted
        assert voicing.voice_count <= voices
    assert shed[4] == ()
    assert shed[3] == (Tone.FIFTH,)
    assert shed[2] == (Tone.FIFTH, Tone.NINTH)
    # Only at one voice does the chord stop being itself, and it says so.
    assert Tone.SEVENTH in shed[1]


def test_identity_loss_is_reported_not_hidden() -> None:
    chord = ChordIntent(root=0, seventh=Seventh.MAJOR, extensions=(9,))
    voicing = plan_voicing(
        VoicingRequest(intent=chord, max_voices=1, bass=bass_pitch_for(chord))
    )
    assert not voicing.keeps_identity


def test_the_root_is_only_dropped_when_the_bass_has_it() -> None:
    """A rootless voicing with no root anywhere is not a voicing of the
    chord."""
    chord = ChordIntent(root=0, seventh=Seventh.MAJOR, extensions=(9,))
    without = plan_voicing(VoicingRequest(intent=chord, max_voices=3, bass=None))
    assert Tone.ROOT not in without.omitted
    assert 0 in [p % 12 for p in without.pitches], "the root must sound somewhere"


def test_a_triad_whose_root_is_in_the_bass_needs_only_two_voices() -> None:
    chord = ChordIntent(root=9, quality=Quality.MINOR)
    voicing = plan_voicing(
        VoicingRequest(intent=chord, max_voices=3, bass=bass_pitch_for(chord))
    )
    assert voicing.voice_count == 2
    assert set(voicing.tones) == {Tone.THIRD, Tone.FIFTH}


def test_a_suspension_never_loses_its_suspended_tone() -> None:
    """Dropping it leaves a bare fifth, which is a different chord."""
    for quality in (Quality.SUSPENDED_2, Quality.SUSPENDED_4):
        chord = ChordIntent(root=0, quality=quality)
        voicing = plan_voicing(
            VoicingRequest(intent=chord, max_voices=1, bass=bass_pitch_for(chord))
        )
        assert Tone.THIRD not in voicing.omitted, quality


def test_voice_leading_holds_common_tones() -> None:
    """The planner is given the previous chord, so a tone already sounding
    costs nothing to keep."""
    royal = [
        ChordIntent(root=5, seventh=Seventh.MAJOR, function=Function.SUBDOMINANT),
        ChordIntent(root=7, seventh=Seventh.MINOR, function=Function.DOMINANT),
        ChordIntent(root=4, quality=Quality.MINOR, seventh=Seventh.MINOR),
        ChordIntent(root=9, quality=Quality.MINOR),
    ]
    voicings = plan_progression(royal, max_voices=3)
    assert len(voicings) == 4
    # Every step moves a small distance. Three voices moving more than five
    # semitones each on average is not voice leading.
    for first, second in pairwise(voicings):
        assert movement(first, second) <= 15, (
            f"{first.intent.symbol}->{second.intent.symbol}: "
            f"{movement(first, second)} semitones"
        )
    assert sum(
        common_tones(a, b) for a, b in pairwise(voicings)
    ) >= 1


def test_voicings_stay_in_the_harmony_register() -> None:
    royal = [
        ChordIntent(root=r, seventh=Seventh.MAJOR) for r in (5, 7, 0, 2)
    ]
    for voicing in plan_progression(royal, max_voices=3):
        for pitch in voicing.pitches:
            assert 52 <= pitch <= 79, pitch
        for pitch in voicing.bass_pitches:
            assert 28 <= pitch <= 52, pitch


def test_the_harmony_avoids_doubling_the_lead_octave() -> None:
    """Doubling the melody in its own octave blurs it."""
    chord = ChordIntent(root=0, seventh=Seventh.MAJOR, extensions=(9,))
    lead = 64  # E4
    voicing = plan_voicing(
        VoicingRequest(
            intent=chord, max_voices=3, bass=bass_pitch_for(chord), lead=lead
        )
    )
    same_octave = [p for p in voicing.pitches if p == lead]
    assert same_octave == []


def test_fits_checks_identity_not_note_count() -> None:
    """A voicing that fits but has lost its seventh is not the chord asked
    for."""
    seventh = ChordIntent(root=0, seventh=Seventh.MAJOR)
    assert not fits(seventh, 1)
    assert fits(seventh, 2)
    triad = ChordIntent(root=0)
    assert fits(triad, 1)


def test_max_voices_must_be_positive() -> None:
    with pytest.raises(ValueError):
        plan_voicing(VoicingRequest(intent=ChordIntent(root=0), max_voices=0))


# --- the NES path must not notice any of this ----------------------------


def test_the_nes_composer_constructs_no_chord_intent() -> None:
    """Additive, not a migration. The frozen baselines depend on it."""
    from pathlib import Path

    source = Path("rse/procedural/compose.py").read_text()
    assert "ChordIntent" not in source
    assert "rse.harmony" not in source
    assert "VoiceAllocator" not in source


def test_the_nes_triad_model_is_untouched() -> None:
    from rse.procedural.chords import QUALITY_INTERVALS

    assert set(QUALITY_INTERVALS) == {
        "major", "minor", "diminished", "augmented"
    }


def test_nes_composition_still_matches_its_baseline() -> None:
    """The guard that matters most. Rendered dry, because the baseline pins
    composition and velocity is now a performance decision."""
    import hashlib
    import json
    from pathlib import Path

    from rse.procedural.compose import compose

    baseline = json.loads(
        Path("tests/data/procedural_v1_baseline.json").read_text()
    )
    for key in ("boss_s800", "action_s800", "tension_s800"):
        style, _, seed = key.rpartition("_s")
        result = compose(
            style_name=style, seed=int(seed), target_seconds=60.0,
            sound_design=None,
        )
        events = [
            (
                e.channel_id, e.tick,
                getattr(e, "pitch", getattr(e, "value", 0)),
                e.duration, e.velocity,
            )
            for e in result.composition.sounding_events()
        ]
        digest = hashlib.sha256(
            json.dumps(events, sort_keys=True).encode()
        ).hexdigest()[:16]
        assert digest == baseline[key], key


def test_short_pieces_need_section_bars_not_target_seconds() -> None:
    """`target_seconds` cannot make a short piece, and that surprises people.

    The composer lays out whole formal sections, so at the default of 8 bars
    the shortest possible form is two sections -- 16 bars, about 26 seconds --
    whatever duration is requested. Asking for 2.4 seconds silently returns
    26, which is the kind of quiet non-compliance worth pinning.

    `section_bars` is the control that actually shortens a piece. This exists
    because a request for a 2.4-second battle sting went through
    `target_seconds` and came back 11x too long.
    """
    from rse.song.snes_generate import generate_snes

    ignored = generate_snes(style="boss", seed=1, target_seconds=2.4, echo="dry")
    assert ignored.seconds > 20, (
        "target_seconds is expected NOT to shorten the piece; if this now "
        "works, the comment above and the docstring on generate_snes are out "
        "of date"
    )

    short = generate_snes(
        style="boss", seed=1, target_seconds=2.4, section_bars=1, echo="dry"
    )
    assert sum(s.bars for s in short.result.sections) == 2
    assert 3.0 < short.seconds < 5.0, short.seconds

    # The default must be unchanged, since every other caller relies on it.
    default = generate_snes(style="boss", seed=1, target_seconds=60.0, echo="dry")
    assert sum(s.bars for s in default.result.sections) == 40
