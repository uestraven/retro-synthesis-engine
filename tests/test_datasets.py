from __future__ import annotations

import tarfile
from pathlib import Path

import pytest

from rse.data.datasets import (
    DatasetError,
    acquire,
    build_manifest,
    extract_archive,
    iter_source_files,
    load_release,
)
from rse.data.synthetic import make_sample_release
from rse.jsonio import read_json, read_jsonl, write_json

SAMPLE_CONFIG = {
    "schema_version": "rse-dataset/0.1.0",
    "dataset_id": "sample",
    "display_name": "Sample",
    "homepage": "https://example.invalid/sample",
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
def configs_dir(tmp_path):
    directory = tmp_path / "configs"
    write_json(directory / "sample.json", SAMPLE_CONFIG)
    return directory


@pytest.fixture
def archive(tmp_path):
    # No duplicate copy here: these tests count files, and de-duplication has
    # its own tests.
    return make_sample_release(
        tmp_path / "build",
        n_per_split=2,
        archive_name="sample_midi.tar.gz",
        duplicate_one=False,
    )


def test_load_release_reads_dataset_level_fields(configs_dir):
    release = load_release("sample", "midi", configs_dir=configs_dir)
    assert release.version == "sample/midi"
    assert release.use_class == "commercially_cleared"
    assert release.license_note == "Generated for tests."


def test_missing_dataset_and_release_are_distinct_errors(configs_dir):
    with pytest.raises(DatasetError, match="no dataset definition at"):
        load_release("nope", "midi", configs_dir=configs_dir)
    with pytest.raises(DatasetError, match="has no release"):
        load_release("sample", "vgm", configs_dir=configs_dir)


def test_unknown_use_class_is_refused(tmp_path):
    bad = dict(SAMPLE_CONFIG, use_class="go_wild")
    write_json(tmp_path / "bad.json", bad)
    with pytest.raises(DatasetError, match="unknown use_class"):
        load_release("bad", "midi", configs_dir=tmp_path)


def test_extraction_lands_in_a_fresh_directory(archive, tmp_path):
    dest = tmp_path / "raw"
    extract_archive(archive, dest)
    files = iter_source_files(dest, ("**/*.mid",))
    assert len(files) == 6  # 2 per split, three splits
    assert {p.parent.name for p in files} == {"train", "valid", "test"}


def test_extraction_refuses_to_clobber_an_existing_directory(archive, tmp_path):
    dest = tmp_path / "raw"
    extract_archive(archive, dest)
    with pytest.raises(DatasetError, match="not empty"):
        extract_archive(archive, dest)
    extract_archive(archive, dest, overwrite=True)  # explicit opt-in works


def test_path_traversal_members_are_refused(tmp_path):
    evil = tmp_path / "evil.tar"
    payload = tmp_path / "payload"
    payload.write_text("pwned")
    with tarfile.open(evil, "w") as tf:
        tf.add(payload, arcname="../escaped.txt")
    with pytest.raises(DatasetError, match="escapes the extraction directory"):
        extract_archive(evil, tmp_path / "raw")
    assert not (tmp_path / "escaped.txt").exists()


def test_symlink_members_are_refused(tmp_path):
    evil = tmp_path / "link.tar"
    with tarfile.open(evil, "w") as tf:
        info = tarfile.TarInfo("sneaky")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        tf.addfile(info)
    with pytest.raises(DatasetError, match="non-regular member"):
        extract_archive(evil, tmp_path / "raw")


def test_a_non_archive_is_refused(tmp_path):
    plain = tmp_path / "plain.txt"
    plain.write_text("hello")
    with pytest.raises(DatasetError, match="neither a tar nor a zip"):
        extract_archive(plain, tmp_path / "raw")


def test_manifest_carries_hashes_and_licence_terms(archive, tmp_path, configs_dir):
    release = load_release("sample", "midi", configs_dir=configs_dir)
    raw = tmp_path / "raw"
    extract_archive(archive, raw)
    manifest = build_manifest(release, raw, acquisition_method="local_archive")
    assert len(manifest) == 6
    assert all(len(record.sha256) == 64 for record in manifest)
    assert all(record.use_class == "commercially_cleared" for record in manifest)
    assert all(record.dataset_version == "sample/midi" for record in manifest)
    assert all(record.source_id.startswith("sample:") for record in manifest)
    # Ids must be unique per file, even for byte-identical content.
    assert len({record.source_id for record in manifest}) == 6


def test_empty_match_is_an_error_not_an_empty_manifest(tmp_path, configs_dir):
    release = load_release("sample", "midi", configs_dir=configs_dir)
    with pytest.raises(DatasetError, match="matched"):
        build_manifest(release, tmp_path / "empty", acquisition_method="local_archive")


def test_acquire_from_a_local_archive_writes_manifest_and_provenance(
    archive, tmp_path, configs_dir
):
    result = acquire(
        "sample",
        "midi",
        archive=archive,
        datasets_dir=tmp_path / "datasets",
        configs_dir=configs_dir,
    )
    acq = result["acquisition"]
    assert acq["n_source_files"] == 6
    assert acq["acquisition_method"] == "local_archive"
    assert len(acq["archive_sha256"]) == 64

    rows = list(read_jsonl(tmp_path / "datasets/processed/sample/midi/manifest.jsonl"))
    assert len(rows) == 6
    saved = read_json(tmp_path / "datasets/processed/sample/midi/acquisition.json")
    assert saved["license_note"] == "Generated for tests."


def test_acquire_verifies_a_supplied_checksum(archive, tmp_path, configs_dir):
    with pytest.raises(DatasetError, match="checksum mismatch"):
        acquire(
            "sample",
            "midi",
            archive=archive,
            sha256="0" * 64,
            datasets_dir=tmp_path / "datasets",
            configs_dir=configs_dir,
        )


def test_acquire_without_archive_or_url_explains_what_to_do(tmp_path, configs_dir):
    with pytest.raises(DatasetError, match="no download URL"):
        acquire(
            "sample",
            "midi",
            datasets_dir=tmp_path / "datasets",
            configs_dir=configs_dir,
        )


def test_nesmdb_is_registered_and_research_only():
    from rse import paths

    release = load_release("nesmdb", "midi", configs_dir=paths.DATASET_CONFIGS_DIR)
    assert release.use_class == "research_only"
    assert "rights review" in release.license_note


# --- registering a release whose files are already on disk ----------------


def _config_with_url(tmp_path, url):
    """A config carrying a download URL, which is the normal case."""
    config = {**SAMPLE_CONFIG, "dataset_id": "withurl"}
    config["releases"] = {
        "midi": {**SAMPLE_CONFIG["releases"]["midi"], "url": url}
    }
    directory = tmp_path / "configs_url"
    write_json(directory / "withurl.json", config)
    return directory


def test_from_extracted_writes_a_manifest_without_an_archive(
    tmp_path, configs_dir, archive
):
    """A machine where the raw tree was copied in, not fetched.

    A shared volume, a restored backup, a transferred dataset: the files are
    there and the archive is gone, and re-downloading a research-only dataset
    just to rebuild a manifest is waste.
    """
    datasets_dir = tmp_path / "datasets"
    raw_dir = datasets_dir / "raw" / "sample" / "midi"
    raw_dir.mkdir(parents=True)
    with tarfile.open(archive) as tar:
        tar.extractall(raw_dir, filter="data")

    result = acquire(
        "sample", "midi", datasets_dir=datasets_dir,
        configs_dir=configs_dir, from_extracted=True,
    )
    acq = result["acquisition"]
    assert acq["acquisition_method"] == "extracted"
    assert acq["n_source_files"] > 0
    # No archive means no checksum to verify the tree against, and the record
    # has to say so rather than imply a provenance it does not have.
    assert acq["archive"] is None
    assert acq["archive_sha256"] is None
    # Per-file checksums still describe what is actually on disk.
    rows = list(read_jsonl(Path(result["manifest_path"])))
    assert len(rows) == acq["n_source_files"]
    assert all(row.get("sha256") for row in rows)


def test_from_extracted_beats_a_url_in_the_config(tmp_path, archive):
    """The precedence bug this test exists for.

    `from_extracted` was first checked AFTER the archive and the URL
    branches. Every real release config here carries a URL, so the flag was
    silently ignored: it reported acquisition_method=download for a tree it
    never downloaded, and on a machine with no cached archive it would have
    started a pointless download of data already on disk.

    The URL below is unroutable, so if precedence regresses this test fails
    by trying to use it rather than by asserting on a label.
    """
    configs = _config_with_url(tmp_path, "https://unroutable.invalid/x.tar.gz")
    datasets_dir = tmp_path / "datasets"
    raw_dir = datasets_dir / "raw" / "withurl" / "midi"
    raw_dir.mkdir(parents=True)
    with tarfile.open(archive) as tar:
        tar.extractall(raw_dir, filter="data")

    result = acquire(
        "withurl", "midi", datasets_dir=datasets_dir,
        configs_dir=configs, from_extracted=True,
    )
    assert result["acquisition"]["acquisition_method"] == "extracted"


def test_from_extracted_refuses_when_the_raw_tree_is_missing(
    tmp_path, configs_dir
):
    """Better than writing an empty manifest that looks like a success."""
    with pytest.raises(DatasetError, match="does not exist"):
        acquire(
            "sample", "midi", datasets_dir=tmp_path / "datasets",
            configs_dir=configs_dir, from_extracted=True,
        )


def test_from_extracted_refuses_a_tree_with_no_matching_files(
    tmp_path, configs_dir
):
    """A raw directory of the wrong release is a mistake worth catching: the
    manifest would be empty and ingest would report zero compositions with no
    hint as to why.

    The message comes from `iter_source_files`, which already guards this --
    an extra check in `acquire` was written for it and was dead code. Pinned
    so the guard is not removed as redundant from the other direction.
    """
    datasets_dir = tmp_path / "datasets"
    raw_dir = datasets_dir / "raw" / "sample" / "midi"
    raw_dir.mkdir(parents=True)
    (raw_dir / "notes.txt").write_text("not a midi file")

    with pytest.raises(DatasetError, match="matched"):
        acquire(
            "sample", "midi", datasets_dir=datasets_dir,
            configs_dir=configs_dir, from_extracted=True,
        )
