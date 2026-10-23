"""Dataset acquisition: registry lookup, verified download, safe extraction.

Source archives are untrusted input. Each one is extracted into its own fresh
directory, member paths are refused if they escape that directory, and the raw
tree is never written to again afterwards — normalization reads from it and
writes somewhere else (proposal 5.2, step 7).
"""

from __future__ import annotations

import shutil
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import urlopen

from rse import paths
from rse.data.provenance import (
    USE_UNKNOWN,
    VALID_USE_CLASSES,
    SourceRecord,
    describe_file,
    sha256_file,
    utc_now,
)
from rse.jsonio import read_json, write_json, write_jsonl

#: Readers available for a release's files.
SOURCE_FORMATS = ("midi", "exprsco")


class DatasetError(RuntimeError):
    """Raised for a missing registry entry, failed verification or unsafe archive."""


@dataclass(frozen=True)
class DatasetRelease:
    """One downloadable release of a dataset."""

    dataset_id: str
    release_id: str
    description: str
    url: str | None
    sha256: str | None
    archive_name: str
    file_globs: tuple[str, ...]
    use_class: str
    license_note: str
    homepage: str
    raw: dict[str, Any]
    #: Which reader parses this release's files. See `SOURCE_FORMATS`.
    format: str = "midi"

    @property
    def version(self) -> str:
        return f"{self.dataset_id}/{self.release_id}"


def load_release(
    dataset_id: str,
    release_id: str,
    *,
    configs_dir: Path | None = None,
) -> DatasetRelease:
    """Read one release definition out of configs/datasets/<dataset_id>.json."""
    base = configs_dir or paths.DATASET_CONFIGS_DIR
    path = base / f"{dataset_id}.json"
    if not path.is_file():
        raise DatasetError(f"no dataset definition at {path}")
    data = read_json(path)
    releases = data.get("releases", {})
    if release_id not in releases:
        raise DatasetError(
            f"dataset {dataset_id!r} has no release {release_id!r}; "
            f"known releases: {sorted(releases)}"
        )
    rel = releases[release_id]
    use_class = data.get("use_class", USE_UNKNOWN)
    if use_class not in VALID_USE_CLASSES:
        raise DatasetError(f"dataset {dataset_id!r} declares unknown use_class {use_class!r}")
    source_format = rel.get("format", "midi")
    if source_format not in SOURCE_FORMATS:
        raise DatasetError(
            f"release {dataset_id}/{release_id} declares unknown format "
            f"{source_format!r}; known: {list(SOURCE_FORMATS)}"
        )
    return DatasetRelease(
        dataset_id=data["dataset_id"],
        release_id=rel.get("release_id", release_id),
        description=rel.get("description", ""),
        url=rel.get("url"),
        sha256=rel.get("sha256"),
        archive_name=rel.get("archive_name", f"{dataset_id}_{release_id}.tar.gz"),
        file_globs=tuple(rel.get("file_globs", ("**/*.mid", "**/*.midi"))),
        use_class=use_class,
        license_note=data.get("license_note", ""),
        homepage=data.get("homepage", ""),
        raw=rel,
        format=source_format,
    )


def download(url: str, dest: Path, *, expected_sha256: str | None = None) -> Path:
    """Download `url` to `dest`, verifying the hash when one is known."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and expected_sha256 and sha256_file(dest) == expected_sha256:
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urlopen(url) as response, tmp.open("wb") as fh:  # noqa: S310 - url comes from config
        shutil.copyfileobj(response, fh)
    if expected_sha256:
        actual = sha256_file(tmp)
        if actual != expected_sha256:
            tmp.unlink(missing_ok=True)
            raise DatasetError(
                f"checksum mismatch for {url}: expected {expected_sha256}, got {actual}"
            )
    tmp.replace(dest)
    return dest


def extract_archive(archive: Path, dest_dir: Path, *, overwrite: bool = False) -> Path:
    """Extract an archive into its own empty directory.

    Refuses members whose paths would land outside `dest_dir`, and refuses
    symlinks and device nodes outright — the pipeline only ever wants regular
    files from these archives.
    """
    if dest_dir.exists():
        if not overwrite:
            if any(dest_dir.iterdir()):
                raise DatasetError(
                    f"{dest_dir} already exists and is not empty; pass overwrite=True to replace it"
                )
        else:
            shutil.rmtree(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    resolved_dest = dest_dir.resolve()

    if tarfile.is_tarfile(archive):
        with tarfile.open(archive) as tf:
            for member in tf.getmembers():
                if not (member.isfile() or member.isdir()):
                    raise DatasetError(
                        f"archive {archive.name} contains a non-regular member: {member.name}"
                    )
                _assert_contained(resolved_dest, member.name, archive)
            # `filter="data"` additionally strips ownership, permissions and
            # anything else that could matter outside the extraction directory.
            tf.extractall(dest_dir, filter="data")
    elif zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            for name in zf.namelist():
                _assert_contained(resolved_dest, name, archive)
            zf.extractall(dest_dir)
    else:
        raise DatasetError(f"{archive} is neither a tar nor a zip archive")
    return dest_dir


def _assert_contained(resolved_dest: Path, member_name: str, archive: Path) -> None:
    target = (resolved_dest / member_name).resolve()
    if target != resolved_dest and resolved_dest not in target.parents:
        raise DatasetError(
            f"archive {archive.name} contains a member that escapes the "
            f"extraction directory: {member_name!r}"
        )


def iter_source_files(root: Path, globs: tuple[str, ...]) -> list[Path]:
    """Every file under `root` matching any glob, de-duplicated and sorted."""
    found: set[Path] = set()
    for pattern in globs:
        found.update(p for p in root.glob(pattern) if p.is_file())
    return sorted(found)


def build_manifest(
    release: DatasetRelease,
    raw_dir: Path,
    *,
    source_url: str | None = None,
    acquisition_method: str = "unknown",
) -> list[SourceRecord]:
    """Provenance records for every source file of an extracted release."""
    files = iter_source_files(raw_dir, release.file_globs)
    if not files:
        raise DatasetError(
            f"no files under {raw_dir} matched {release.file_globs}; "
            "check that the archive extracted as expected"
        )
    return [
        describe_file(
            path,
            root=raw_dir,
            dataset_id=release.dataset_id,
            dataset_version=release.version,
            use_class=release.use_class,
            license_note=release.license_note,
            source_url=source_url or release.url,
            acquisition_method=acquisition_method,
        )
        for path in files
    ]


def acquire(
    dataset_id: str,
    release_id: str,
    *,
    archive: Path | None = None,
    url: str | None = None,
    sha256: str | None = None,
    datasets_dir: Path | None = None,
    configs_dir: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Fetch (or register) a release, extract it, and write its manifest.

    Exactly one of `archive` (a file already on disk) or a URL — passed here or
    present in the dataset config — is required.
    """
    release = load_release(dataset_id, release_id, configs_dir=configs_dir)
    root = datasets_dir or paths.DATASETS_DIR
    downloads = root / "downloads" / dataset_id
    raw_dir = root / "raw" / dataset_id / release.release_id

    effective_url = url or release.url
    if archive is not None:
        archive = Path(archive).expanduser().resolve()
        if not archive.is_file():
            raise DatasetError(f"archive {archive} does not exist")
        expected = sha256 or release.sha256
        actual = sha256_file(archive)
        if expected and actual != expected:
            raise DatasetError(
                f"checksum mismatch for {archive}: expected {expected}, got {actual}"
            )
        method = "local_archive"
        archive_sha256 = actual
    elif effective_url:
        archive = download(
            effective_url,
            downloads / release.archive_name,
            expected_sha256=sha256 or release.sha256,
        )
        method = "download"
        archive_sha256 = sha256_file(archive)
    else:
        raise DatasetError(
            f"{release.version} has no download URL in its config. Download it from "
            f"{release.homepage or 'the dataset homepage'} and re-run with "
            f"--archive <path> (and --sha256 <hash> to pin it)."
        )

    extract_archive(archive, raw_dir, overwrite=overwrite)
    manifest = build_manifest(release, raw_dir, source_url=effective_url, acquisition_method=method)

    out_dir = root / "processed" / dataset_id / release.release_id
    manifest_path = write_jsonl(out_dir / "manifest.jsonl", (r.to_dict() for r in manifest))
    acquisition = {
        "dataset_id": release.dataset_id,
        "release_id": release.release_id,
        "dataset_version": release.version,
        "acquired_utc": utc_now(),
        "acquisition_method": method,
        "source_url": effective_url,
        "archive": str(archive),
        "archive_sha256": archive_sha256,
        "raw_dir": str(raw_dir),
        "n_source_files": len(manifest),
        "use_class": release.use_class,
        "license_note": release.license_note,
        "citation": release.raw.get("citation", ""),
        "homepage": release.homepage,
    }
    write_json(out_dir / "acquisition.json", acquisition)
    return {"acquisition": acquisition, "manifest_path": str(manifest_path)}
