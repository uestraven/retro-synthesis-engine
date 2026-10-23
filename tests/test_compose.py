from __future__ import annotations

import numpy as np
import pytest

from rse.compose import MODEL, RULES, ComposeRequest, compose
from rse.exporters.wav import read_wav


def test_request_validates():
    with pytest.raises(ValueError, match="source must be one of"):
        ComposeRequest(source="vibes")
    with pytest.raises(ValueError, match="needs a checkpoint"):
        ComposeRequest(source=MODEL)
    with pytest.raises(ValueError, match="bars must be positive"):
        ComposeRequest(bars=0)


def test_rule_based_compose(nes_profile, tmp_path):
    result = compose(
        ComposeRequest(source=RULES, seed=11, bars=8, mood="dark"),
        out_base=tmp_path / "piece",
        profile=nes_profile,
    )
    assert result.measurements["n_notes"] > 0
    assert result.measurements["n_bars"] == 8
    assert result.duration_sec > 0
    assert result.plan_lines and "dark" in result.plan_lines[0]
    assert len(result.channels) == 4


def test_rule_based_output_needs_no_repair(nes_profile, tmp_path):
    """The generator claims compliance; this is the claim being checked."""
    result = compose(ComposeRequest(seed=3, bars=8), out_base=tmp_path / "p", profile=nes_profile)
    assert result.constraint_report is not None
    assert result.constraint_report.ok, result.constraint_report.counts()
    assert result.measurements["hardware_compliant"] is True


def test_files_are_written(nes_profile, tmp_path):
    result = compose(
        ComposeRequest(seed=5, bars=4, stems=True),
        out_base=tmp_path / "out" / "piece",
        profile=nes_profile,
    )
    assert result.wav.is_file() and result.midi.is_file() and result.json.is_file()
    assert sorted(result.stems) == ["NO", "P1", "P2", "TR"]
    samples, rate = read_wav(result.wav)
    assert rate == 44_100
    assert np.abs(samples).max() > 0.1


def test_nothing_is_written_without_an_out_base(nes_profile):
    result = compose(ComposeRequest(seed=7, bars=4), profile=nes_profile)
    assert result.wav is None and result.midi is None and result.json is None
    assert result.samples.size > 0


def test_a_loop_is_rendered_without_a_silent_tail(nes_profile, tmp_path):
    result = compose(
        ComposeRequest(seed=9, bars=4, loop=True),
        out_base=tmp_path / "loop",
        profile=nes_profile,
    )
    assert result.loop_report is not None
    assert result.loop_report.ok, result.loop_report.issues
    assert result.loop_report.trailing_silence_sec < 0.05


def test_a_one_shot_keeps_its_tail(nes_profile, tmp_path):
    result = compose(
        ComposeRequest(seed=9, bars=4, loop=False),
        out_base=tmp_path / "oneshot",
        profile=nes_profile,
    )
    assert result.loop_report is not None
    assert result.loop_report.trailing_silence_sec > 0.1


def test_compose_is_reproducible(nes_profile, tmp_path):
    first = compose(ComposeRequest(seed=42, bars=8), out_base=tmp_path / "a", profile=nes_profile)
    second = compose(ComposeRequest(seed=42, bars=8), out_base=tmp_path / "b", profile=nes_profile)
    assert np.array_equal(first.samples, second.samples)
    assert first.wav.read_bytes() == second.wav.read_bytes()


def test_duty_overrides_change_the_audio(nes_profile, tmp_path):
    base = compose(
        ComposeRequest(seed=4, bars=4, duty_p1=2), out_base=tmp_path / "d2", profile=nes_profile
    )
    thin = compose(
        ComposeRequest(seed=4, bars=4, duty_p1=0), out_base=tmp_path / "d0", profile=nes_profile
    )
    assert not np.array_equal(base.samples, thin.samples)


def test_summary_is_serializable(nes_profile, tmp_path):
    import json

    result = compose(ComposeRequest(seed=1, bars=4), out_base=tmp_path / "s", profile=nes_profile)
    payload = result.summary()
    assert json.dumps(payload)  # must contain nothing exotic
    assert payload["n_bars"] == 4
    assert payload["loop"]["ok"] is True
    assert payload["files"]["wav"].endswith(".wav")
    assert payload["renderer_id"].startswith("nes_apu/")


# -- model path ---------------------------------------------------------------


def test_model_compose_runs_output_through_the_constraint_engine(nes_profile, tmp_path):
    """A model has no compliance guarantee; the renderer must never find out."""
    pytest.importorskip("torch")
    from rse.models.transformer import ModelConfig
    from rse.tokenizers.tokenizer import RseTokenizer
    from rse.training.dataset import TRAIN, VALID, build_windows
    from rse.training.train import TrainConfig, train

    tokenizer = RseTokenizer()
    from rse.events.convert import composition_from_intake
    from rse.generation.plan import make_plan
    from rse.generation.procedural import generate

    entries = []
    for index in range(4):
        generated = generate(nes_profile, make_plan(seed=200 + index, n_bars=4))
        entries.append(
            (
                composition_from_intake(generated.record),
                TRAIN if index < 3 else VALID,
                f"p{index}",
            )
        )
    windows, _ = build_windows(entries, tokenizer, window_bars=4)
    checkpoint = tmp_path / "tiny.pt"
    train(
        windows,
        tokenizer,
        model_config=ModelConfig(
            vocab_size=len(tokenizer), n_layers=2, n_heads=2, d_model=32, max_seq_len=512
        ),
        config=TrainConfig(
            seed=0,
            max_steps=4,
            batch_size=2,
            seq_len=256,
            eval_every=4,
            eval_batches=1,
            amp=False,
            device="cpu",
            patience=0,
        ),
        checkpoint_path=checkpoint,
    )

    result = compose(
        ComposeRequest(source=MODEL, checkpoint=checkpoint, seed=1, bars=4),
        out_base=tmp_path / "model",
        profile=nes_profile,
    )
    # Whatever the untrained model emitted, what comes out is playable.
    assert result.constraint_report is not None
    assert result.sample_info is not None
    assert result.sample_info["checkpoint"] == str(checkpoint)
    from rse.events.convert import composition_from_intake as to_composition
    from rse.hardware.constraints import ConstraintEngine

    assert ConstraintEngine(nes_profile).validate(to_composition(result.record)) == []


# -- mix and audition ---------------------------------------------------------


def test_channel_selection_validates():
    with pytest.raises(ValueError, match="channels cannot be empty"):
        ComposeRequest(channels=())
    with pytest.raises(ValueError, match="audition_repeats"):
        ComposeRequest(audition_repeats=0)


def test_muting_a_channel_changes_the_audio(nes_profile):
    full = compose(ComposeRequest(seed=3, bars=4), profile=nes_profile)
    without_drums = compose(
        ComposeRequest(seed=3, bars=4, channels=("P1", "P2", "TR")), profile=nes_profile
    )
    assert not np.array_equal(full.samples, without_drums.samples)


def test_muting_re_renders_rather_than_zeroing_a_stem(nes_profile):
    """So the APU's nonlinear mixing still applies to what is left."""
    both = compose(ComposeRequest(seed=3, bars=4, channels=("P1", "P2")), profile=nes_profile)
    one = compose(ComposeRequest(seed=3, bars=4, channels=("P1",)), profile=nes_profile)
    # Two pulses mixed nonlinearly are not twice one pulse, so the solo cannot
    # be a simple scaling of the pair.
    assert not np.allclose(both.samples, one.samples * 2, atol=0.05)


def test_soloing_a_silent_channel_is_an_error(nes_profile):
    with pytest.raises(ValueError, match="no renderable notes"):
        compose(
            ComposeRequest(source=RULES, seed=3, bars=4, channels=("DMC",)),
            profile=nes_profile,
        )


def test_auditioning_repeats_the_audio(nes_profile):
    once = compose(ComposeRequest(seed=5, bars=4), profile=nes_profile)
    thrice = compose(ComposeRequest(seed=5, bars=4, audition_repeats=3), profile=nes_profile)
    assert thrice.samples.size == once.samples.size * 3
    assert np.array_equal(thrice.samples[: once.samples.size], once.samples)


def test_the_loop_report_describes_one_pass_not_the_audition(nes_profile):
    """Otherwise repeating a clean loop would look like a three-times-longer one."""
    once = compose(ComposeRequest(seed=5, bars=4), profile=nes_profile)
    thrice = compose(ComposeRequest(seed=5, bars=4, audition_repeats=3), profile=nes_profile)
    assert once.loop_report.to_dict() == thrice.loop_report.to_dict()
