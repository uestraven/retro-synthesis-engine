from __future__ import annotations

import pytest

from rse import paths
from rse.data.datasets import acquire
from rse.data.pipeline import IngestConfig, ingest_release, processed_dir
from rse.data.synthetic import make_sample_release
from rse.jsonio import read_json, read_jsonl, write_json

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
def fetched(tmp_path):
    """A fetched sample release: 6 distinct compositions plus one exact copy."""
    configs = tmp_path / "configs"
    write_json(configs / "sample.json", CONFIG)
    archive = make_sample_release(
        tmp_path / "build", n_per_split=2, archive_name="sample_midi.tar.gz"
    )
    datasets_dir = tmp_path / "datasets"
    acquire(
        "sample",
        "midi",
        archive=archive,
        datasets_dir=datasets_dir,
        configs_dir=configs,
    )
    return datasets_dir


@pytest.fixture
def result(fetched):
    return ingest_release(
        "sample",
        "midi",
        config=IngestConfig(profile_id="nes"),
        datasets_dir=fetched,
        configs_dir=paths.CONFIGS_DIR,
        keep_records_in_memory=True,
    )


def test_ingest_requires_a_manifest(tmp_path):
    with pytest.raises(FileNotFoundError, match="run `rse fetch"):
        ingest_release("sample", "midi", datasets_dir=tmp_path / "datasets")


def test_every_source_file_is_normalized(result):
    assert result.report["n_sources"] == 7
    assert result.report["n_normalized"] == 7
    assert result.report["n_failed"] == 0
    assert len(result.records) == 7


def test_artifacts_are_written_where_the_next_phase_expects_them(result, fetched):
    out_dir = processed_dir("sample", "midi", datasets_dir=fetched)
    assert result.index_path == out_dir / "index.jsonl"
    assert (out_dir / "statistics.json").is_file()
    assert (out_dir / "report.json").is_file()
    records = list((out_dir / "records").glob("*.json"))
    assert len(records) == 7
    # Every index row points at a record that exists.
    for row in read_jsonl(result.index_path):
        assert (out_dir / row["record_path"]).is_file()


def test_splits_survive_ingestion(result):
    splits = {row["split"] for row in read_jsonl(result.index_path)}
    assert splits == {"train", "valid", "test"}


def test_the_duplicated_file_is_flagged_and_its_original_is_not(result):
    rows = {row["name"]: row for row in read_jsonl(result.index_path)}
    copy = rows["train/train_000_copy.mid"]
    original = rows["train/train_000.mid"]
    assert copy["duplicate_of"] or original["duplicate_of"]
    # Exactly one of the pair is kept as the representative.
    assert bool(copy["duplicate_of"]) != bool(original["duplicate_of"])
    assert result.report["n_flagged_duplicates"] == 1


def test_statistics_describe_all_four_channels(result):
    stats = result.report["statistics"]
    assert set(stats["per_channel"]) == {"P1", "P2", "TR", "NO"}
    assert stats["n_notes"] > 0
    assert stats["n_controls"] > 0
    assert stats["n_hardware_compliant"] == result.report["n_normalized"]
    assert stats["assignment_methods"]["track_name"] == 7 * 4
    assert stats["n_with_loop_markers"] == 7
    assert stats["bars_per_record"]["min"] == 8
    assert stats["bars_per_record"]["max"] == 16
    # The synthetic files use CC 11 for expression; it must be in the stats.
    assert "11" in stats["controllers_used"]


def test_report_records_the_configuration_that_produced_it(result):
    report = result.report
    assert report["config"]["profile_id"] == "nes"
    assert report["config"]["quantize_grid"] is None
    assert report["hardware_profile"] == "nes"
    assert report["intake_schema_version"].startswith("rse-intake/")
    assert report["rse_version"]


def test_limit_processes_a_prefix_only(fetched):
    result = ingest_release(
        "sample",
        "midi",
        config=IngestConfig(limit=2),
        datasets_dir=fetched,
        configs_dir=paths.CONFIGS_DIR,
    )
    assert result.report["n_sources"] == 2
    assert result.report["n_normalized"] == 2


def test_no_records_mode_still_measures(fetched):
    result = ingest_release(
        "sample",
        "midi",
        config=IngestConfig(write_records=False),
        datasets_dir=fetched,
        configs_dir=paths.CONFIGS_DIR,
    )
    out_dir = processed_dir("sample", "midi", datasets_dir=fetched)
    assert not (out_dir / "records").exists()
    assert result.report["statistics"]["n_notes"] > 0
    for row in read_jsonl(result.index_path):
        assert row["record_path"] is None


def test_unreadable_source_is_reported_and_does_not_stop_the_run(fetched):
    out_dir = processed_dir("sample", "midi", datasets_dir=fetched)
    manifest = list(read_jsonl(out_dir / "manifest.jsonl"))
    broken = fetched / "raw/sample/midi" / manifest[0]["relpath"]
    broken.write_bytes(b"not midi at all")

    result = ingest_release("sample", "midi", datasets_dir=fetched, configs_dir=paths.CONFIGS_DIR)
    assert result.report["n_failed"] == 1
    assert result.report["n_normalized"] == 6
    assert manifest[0]["relpath"] in result.report["failures"][0]["relpath"]


def test_quantization_setting_reaches_the_records(fetched):
    result = ingest_release(
        "sample",
        "midi",
        config=IngestConfig(quantize_grid=0.25),
        datasets_dir=fetched,
        configs_dir=paths.CONFIGS_DIR,
    )
    assert result.report["config"]["quantize_grid"] == 0.25
    out_dir = processed_dir("sample", "midi", datasets_dir=fetched)
    record = read_json(next((out_dir / "records").glob("*.json")))
    assert record["quantize_grid"] == 0.25
    assert all("q_beat" in event for event in record["events"])
