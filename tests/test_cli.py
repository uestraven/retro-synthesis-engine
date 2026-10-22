from __future__ import annotations

import json

import pytest

from rse import paths
from rse.cli import main
from rse.data.synthetic import make_sample_release
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
