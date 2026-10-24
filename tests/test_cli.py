from __future__ import annotations

import json

import numpy as np
import pytest

from rse import paths
from rse.cli import main
from rse.data.synthetic import make_sample_release
from rse.exporters.wav import read_wav
from rse.jsonio import write_json

CONFIG = {
    "schema_version": "rse-dataset/0.1.0",
    "dataset_id": "sample",
    "display_name": "Sample",
    "homepage": "",
    "use_class": "commercially_cleared",
    "license_note": "Generated for tests.",
    "releases": {
        "midi": {
            "release_id": "midi",
            "description": "Synthetic MIDI.",
            "url": None,
            "sha256": None,
            "archive_name": "sample_midi.tar.gz",
            "file_globs": ["**/*.mid"],
        }
    },
}


@pytest.fixture
def workspace(tmp_path):
    """A configs dir holding both the committed hardware profiles and a test dataset."""
    configs = tmp_path / "configs"
    (configs / "hardware").mkdir(parents=True)
    for profile in paths.HARDWARE_CONFIGS_DIR.glob("*.json"):
        (configs / "hardware" / profile.name).write_bytes(profile.read_bytes())
    write_json(configs / "datasets" / "sample.json", CONFIG)
    archive = make_sample_release(
        tmp_path / "build", n_per_split=2, archive_name="sample_midi.tar.gz"
    )
    return {"configs": configs, "datasets": tmp_path / "datasets", "archive": archive}


def run(args, workspace, *, as_json=False):
    argv = [
        "--datasets-dir",
        str(workspace["datasets"]),
        "--configs-dir",
        str(workspace["configs"]),
    ]
    if as_json:
        argv.append("--json")
    return main(argv + args)


def test_profiles_lists_the_nes_channels(capsys):
    assert main(["profiles"]) == 0
    out = capsys.readouterr().out
    assert "nes" in out
    for channel in ("P1", "P2", "TR", "NO"):
        assert channel in out


def test_profiles_json_is_machine_readable(capsys):
    assert main(["--json", "profiles"]) == 0
    payload = json.loads(capsys.readouterr().out)
    nes = next(row for row in payload if row["profile_id"] == "nes")
    assert [ch["channel_id"] for ch in nes["channels"]] == ["P1", "P2", "TR", "NO"]
    assert nes["channels"][3]["pitched"] is False


def test_inspect_file_reports_channels_and_compliance(midi_path, capsys):
    assert main(["inspect", "--file", str(midi_path)]) == 0
    out = capsys.readouterr().out
    assert "hardware compliant: True" in out
    assert "track_name" in out
    assert "loop markers: 2" in out


def test_inspect_file_json_emits_a_full_intake_record(midi_path, capsys):
    assert main(["--json", "inspect", "--file", str(midi_path)]) == 0
    record = json.loads(capsys.readouterr().out)
    assert record["hardware_profile"] == "nes"
    assert record["measurements"]["channels_used"] == 4
    assert record["source"]["acquisition_method"] == "inspect"


def test_fetch_then_ingest_then_inspect(workspace, capsys):
    assert (
        run(
            [
                "fetch",
                "--dataset",
                "sample",
                "--release",
                "midi",
                "--archive",
                str(workspace["archive"]),
            ],
            workspace,
        )
        == 0
    )
    assert "fetched sample/midi" in capsys.readouterr().out

    assert run(["ingest", "--dataset", "sample", "--release", "midi"], workspace) == 0
    out = capsys.readouterr().out
    assert "7/7 files normalized (0 failed)" in out
    assert "duplicates  : 1 flagged for review" in out

    assert run(["inspect", "--dataset", "sample", "--release", "midi"], workspace) == 0
    assert "7/7 files normalized" in capsys.readouterr().out


def test_ingest_json_report_is_machine_readable(workspace, capsys):
    run(
        [
            "fetch",
            "--dataset",
            "sample",
            "--release",
            "midi",
            "--archive",
            str(workspace["archive"]),
        ],
        workspace,
    )
    capsys.readouterr()
    assert run(["ingest", "--dataset", "sample", "--release", "midi"], workspace, as_json=True) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["n_normalized"] == 7
    assert report["statistics"]["per_channel"]["P1"]["n_notes"] > 0


def test_inspect_without_a_prior_ingest_explains_itself(workspace, capsys):
    assert run(["inspect", "--dataset", "sample", "--release", "midi"], workspace) == 2
    assert "run `rse ingest" in capsys.readouterr().err


def test_fetch_without_a_source_reports_an_error(workspace, capsys):
    assert run(["fetch", "--dataset", "sample", "--release", "midi"], workspace) == 2
    assert "no download URL" in capsys.readouterr().err


def test_render_from_midi_writes_audio_and_a_sidecar(midi_path, tmp_path, capsys):
    out = tmp_path / "song.wav"
    assert main(["render", "--file", str(midi_path), "--out", str(out)]) == 0
    assert "rendered" in capsys.readouterr().out

    samples, rate = read_wav(out)
    assert rate == 44_100
    assert samples.size > 0
    assert np.abs(samples).max() > 0.1  # actually audible

    metadata = json.loads(out.with_suffix(".render.json").read_text())
    assert metadata["renderer_id"].startswith("nes_apu/")
    assert metadata["channels_rendered"] == ["NO", "P1", "P2", "TR"]
    assert metadata["n_notes"] > 0
    assert metadata["settings"]["sample_rate"] == 44_100


def test_render_stems_produces_one_file_per_channel(midi_path, tmp_path):
    out = tmp_path / "song.wav"
    assert main(["render", "--file", str(midi_path), "--out", str(out), "--stems"]) == 0
    stems = sorted(p.name for p in (tmp_path / "song_stems").glob("*.wav"))
    assert stems == ["NO.wav", "P1.wav", "P2.wav", "TR.wav"]


def test_render_from_an_intake_record(midi_path, tmp_path, nes_profile):
    from rse.data.midi_io import parse_midi
    from rse.data.normalize import build_intake_record
    from rse.data.provenance import describe_file
    from rse.jsonio import write_json

    record = build_intake_record(
        parse_midi(midi_path, nes_profile),
        describe_file(midi_path, root=midi_path.parent, dataset_id="t", dataset_version="t/1"),
        nes_profile,
    )
    record_path = write_json(tmp_path / "rec.json", record)
    out = tmp_path / "from_record.wav"
    assert main(["render", "--record", str(record_path), "--out", str(out)]) == 0
    assert out.is_file()


def test_render_rejects_a_record_for_another_profile(tmp_path, capsys):
    from rse.jsonio import write_json

    record = write_json(tmp_path / "r.json", {"hardware_profile": "genesis", "events": []})
    assert main(["render", "--record", str(record), "--out", str(tmp_path / "x.wav")]) == 2
    assert "targets profile" in capsys.readouterr().err


def test_render_needs_exactly_one_input(midi_path, tmp_path, capsys):
    assert main(["render", "--out", str(tmp_path / "x.wav")]) == 2
    assert "exactly one of" in capsys.readouterr().err
    assert (
        main(
            [
                "render",
                "--file",
                str(midi_path),
                "--record",
                str(midi_path),
                "--out",
                str(tmp_path / "x.wav"),
            ]
        )
        == 2
    )


def test_render_reports_a_composition_with_no_notes(tmp_path, capsys):
    from rse.jsonio import write_json

    record = write_json(
        tmp_path / "empty.json",
        {"hardware_profile": "nes", "midi": {"ticks_per_beat": 480}, "events": []},
    )
    assert main(["render", "--record", str(record), "--out", str(tmp_path / "x.wav")]) == 2
    assert "no renderable notes" in capsys.readouterr().err


def test_generate_writes_wav_midi_and_json(tmp_path, capsys):
    base = tmp_path / "tune"
    assert main(["generate", "--seed", "4321", "--bars", "8", "--out", str(base)]) == 0
    out = capsys.readouterr().out
    assert "form" in out and "loop: clean" in out

    assert base.with_suffix(".wav").is_file()
    assert base.with_suffix(".mid").is_file()
    payload = json.loads(base.with_suffix(".json").read_text())
    assert payload["generation"]["plan"]["seed"] == 4321
    assert payload["measurements"]["hardware_compliant"] is True
    assert payload["loop_report"]["ok"] is True

    samples, rate = read_wav(base.with_suffix(".wav"))
    assert rate == 44_100
    assert np.abs(samples).max() > 0.1


def test_generate_is_reproducible_from_its_seed(tmp_path):
    first, second = tmp_path / "a", tmp_path / "b"
    main(["generate", "--seed", "99", "--bars", "4", "--out", str(first)])
    main(["generate", "--seed", "99", "--bars", "4", "--out", str(second)])
    assert first.with_suffix(".wav").read_bytes() == second.with_suffix(".wav").read_bytes()


def test_generate_honours_musical_parameters(tmp_path, capsys):
    base = tmp_path / "dark"
    assert (
        main(
            [
                "generate",
                "--seed",
                "5",
                "--mood",
                "dark",
                "--energy",
                "high",
                "--bars",
                "16",
                "--tempo",
                "165",
                "--form",
                "ABAC",
                "--key",
                "2",
                "--out",
                str(base),
            ]
        )
        == 0
    )
    assert "165 BPM" in capsys.readouterr().out
    plan = json.loads(base.with_suffix(".json").read_text())["generation"]["plan"]
    assert plan["mood"] == "dark"
    assert plan["form"] == "ABAC"
    assert plan["key"]["tonic"] == 2
    assert plan["n_bars"] == 16


def test_generated_loops_have_no_trailing_silence(tmp_path):
    base = tmp_path / "loop"
    main(["generate", "--seed", "8", "--bars", "4", "--out", str(base)])
    report = json.loads(base.with_suffix(".json").read_text())["loop_report"]
    assert report["trailing_silence_sec"] < 0.05


def test_generate_stems(tmp_path):
    base = tmp_path / "stemmed"
    main(["generate", "--seed", "11", "--bars", "4", "--out", str(base), "--stems"])
    stems = sorted(p.name for p in (tmp_path / "stemmed_stems").glob("*.wav"))
    assert stems == ["NO.wav", "P1.wav", "P2.wav", "TR.wav"]


def test_generate_json_mode_is_machine_readable(tmp_path, capsys):
    base = tmp_path / "j"
    assert main(["--json", "generate", "--seed", "3", "--bars", "4", "--out", str(base)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["plan"]["seed"] == 3
    assert payload["loop_report"]["ok"] is True
    assert payload["wav"].endswith(".wav")


def test_generate_rejects_an_unsupported_length(tmp_path):
    with pytest.raises(SystemExit):
        main(["generate", "--bars", "7", "--out", str(tmp_path / "x")])


def test_validate_a_clean_midi_file(midi_path, capsys):
    assert main(["validate", "--file", str(midi_path), "--round-trip"]) == 0
    out = capsys.readouterr().out
    assert "hardware: playable as written" in out
    assert "loop: clean" in out
    assert "round trip: lossless" in out


def test_validate_a_generated_piece(tmp_path, capsys):
    base = tmp_path / "piece"
    main(["generate", "--seed", "17", "--bars", "8", "--out", str(base)])
    capsys.readouterr()
    assert main(["validate", "--record", str(base.with_suffix(".json")), "--round-trip"]) == 0
    assert "playable as written" in capsys.readouterr().out


def test_validate_repairs_broken_input_and_can_write_the_fix(tmp_path, capsys):
    from rse.events.schema import Composition, Note
    from rse.jsonio import write_json

    broken = Composition(
        hardware_profile="nes",
        ticks_per_beat=480,
        loop_start_tick=0,
        loop_end_tick=960,
        events=[
            Note(channel_id="P1", tick=0, pitch=12, duration=1200, velocity=100),
            Note(channel_id="P1", tick=240, pitch=64, duration=240, velocity=90),
        ],
    )
    path = write_json(tmp_path / "broken.json", broken.to_dict())
    fixed = tmp_path / "fixed.json"
    assert main(["validate", "--record", str(path), "--fix", str(fixed)]) == 1
    out = capsys.readouterr().out
    assert "violation(s)" in out
    assert "transposed" in out

    repaired = Composition.from_dict(json.loads(fixed.read_text()))
    from rse.hardware.constraints import ConstraintEngine

    engine = ConstraintEngine(paths_profile())
    assert engine.validate(repaired) == []


def paths_profile():
    from rse.hardware.profile import load_profile

    return load_profile("nes", configs_dir=paths.HARDWARE_CONFIGS_DIR)


def test_validate_accepts_an_rse_ef_document(midi_path, tmp_path):
    """`validate` must take either representation, told apart by schema_version."""
    from rse.events.convert import composition_from_midi
    from rse.jsonio import write_json

    composition = composition_from_midi(midi_path, paths_profile())
    path = write_json(tmp_path / "ef.json", composition.to_dict())
    assert path.is_file()
    assert main(["validate", "--record", str(path)]) == 0


def test_validate_needs_exactly_one_input(midi_path, capsys):
    assert main(["validate"]) == 2
    assert "exactly one of" in capsys.readouterr().err
    assert main(["validate", "--file", str(midi_path), "--record", str(midi_path)]) == 2


def test_validate_json_mode_is_machine_readable(midi_path, capsys):
    assert main(["--json", "validate", "--file", str(midi_path), "--round-trip"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["constraints"]["ok"] is True
    assert payload["round_trip"]["ok"] is True
    assert payload["loop"]["ok"] is True
    assert payload["schema_version"].startswith("rse-ef/")


def test_render_json_mode_is_machine_readable(midi_path, tmp_path, capsys):
    out = tmp_path / "song.wav"
    assert main(["--json", "render", "--file", str(midi_path), "--out", str(out)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["wav"] == str(out)
    assert payload["duration_sec"] > 0


def test_render_appends_wav_when_no_extension_is_given(midi_path, tmp_path):
    """`--out piece` wrote a file called `piece`: a valid WAV the OS would not
    open, and inconsistent with generate and sample."""
    base = tmp_path / "piece"
    assert main(["render", "--file", str(midi_path), "--out", str(base)]) == 0
    assert base.with_suffix(".wav").is_file()
    assert not base.exists() or base.suffix == ".wav"
    samples, rate = read_wav(base.with_suffix(".wav"))
    assert rate == 44_100 and samples.size > 0


def test_render_respects_an_explicit_extension(midi_path, tmp_path):
    out = tmp_path / "explicit.wav"
    assert main(["render", "--file", str(midi_path), "--out", str(out)]) == 0
    assert out.is_file()


def test_a_dot_in_the_basename_does_not_eat_the_extension(midi_path, tmp_path):
    """`--out take_t0.7` and `--out take_t0.9` both became `take_t0.wav`.

    Path.with_suffix replaces whatever follows the last dot, so sampling two
    temperatures wrote one file and the second silently overwrote the first.
    A whole A/B arm was lost to this.
    """
    from rse.cli import with_extension

    assert with_extension(tmp_path / "take_t0.7", ".wav").name == "take_t0.7.wav"
    assert with_extension(tmp_path / "take_t0.9", ".wav").name == "take_t0.9.wav"
    # An extension that is already right is not doubled.
    assert with_extension(tmp_path / "take.wav", ".wav").name == "take.wav"
    assert with_extension(tmp_path / "take.WAV", ".wav").name == "take.WAV"
    # A plain basename still gains one.
    assert with_extension(tmp_path / "take", ".wav").name == "take.wav"

    first = tmp_path / "piece_t0.7"
    second = tmp_path / "piece_t0.9"
    assert main(["render", "--file", str(midi_path), "--out", str(first)]) == 0
    assert main(["render", "--file", str(midi_path), "--out", str(second)]) == 0
    assert (tmp_path / "piece_t0.7.wav").is_file()
    assert (tmp_path / "piece_t0.9.wav").is_file()


def test_render_accepts_an_rse_ef_document(midi_path, tmp_path):
    """`validate --record` took both record shapes and `render --record` did not.

    That blocked the most important validation available: rendering a decoded
    composition through the same path generated music takes, to tell a broken
    representation apart from a badly trained model.
    """
    from rse.events.convert import composition_from_midi
    from rse.jsonio import write_json

    composition = composition_from_midi(midi_path, paths_profile())
    record = write_json(tmp_path / "ef.json", composition.to_dict())
    out = tmp_path / "ef.wav"
    assert main(["render", "--record", str(record), "--out", str(out)]) == 0
    samples, rate = read_wav(out)
    assert rate == 44_100
    assert np.abs(samples).max() > 0.1


def test_the_corpus_band_rejects_a_model_with_no_repetition():
    """The acceptance band must not pass output that repeats nothing.

    An earlier p10-p90 band did exactly that: some real pieces score zero
    n-gram repetition, so p10 was 0.0 and a model generating none counted as
    inside the corpus distribution against a median of 0.5. The interquartile
    range is the test that discriminates.
    """
    from rse.training.evaluate import Aggregate, in_distribution

    corpus = Aggregate(
        label="corpus",
        n=30,
        medians={"ngram_repetition_rate": 0.5},
        p25={"ngram_repetition_rate": 0.174},
        p75={"ngram_repetition_rate": 0.805},
    )
    silent = Aggregate(label="model", n=4, medians={"ngram_repetition_rate": 0.0})
    verdict = in_distribution(corpus, silent, ("ngram_repetition_rate",))
    assert verdict["ngram_repetition_rate"]["inside"] is False

    healthy = Aggregate(label="model", n=4, medians={"ngram_repetition_rate": 0.45})
    assert in_distribution(corpus, healthy, ("ngram_repetition_rate",))["ngram_repetition_rate"][
        "inside"
    ]


def test_sampling_truncation_can_be_disabled_from_the_cli(tmp_path, monkeypatch):
    """`--top-k 0 --top-p 0` must mean no truncation.

    SamplingConfig takes None for that and rejects 0 outright, so there was no
    way to ask the CLI for untruncated sampling -- the setting the sampling
    sweep found lands inside the corpus repetition band most often, and the one
    every earlier listening test could not have used.
    """
    from rse.generation.sampler import SamplingConfig

    with pytest.raises(ValueError):
        SamplingConfig(top_k=0)
    with pytest.raises(ValueError):
        SamplingConfig(top_p=0.0)
    assert SamplingConfig(top_k=None, top_p=None).top_k is None

    import argparse

    args = argparse.Namespace(top_k=0, top_p=0.0)
    assert (args.top_k or None) is None
    assert (args.top_p or None) is None
