"""Provenance records for every file that enters the pipeline.

Proposal 5.5 requires that training sources stay traceable and that research
data stay distinguishable from commercially cleared data. Nothing is allowed
into the processed tree without one of these records attached.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

#: How a source may be used. Deliberately coarse: the point is that a later
#: training run can filter on it without re-deriving the licence question.
USE_RESEARCH_ONLY = "research_only"
USE_CLEARED = "commercially_cleared"
USE_UNKNOWN = "unknown"
VALID_USE_CLASSES = (USE_RESEARCH_ONLY, USE_CLEARED, USE_UNKNOWN)


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def sha256_file(path: Path, *, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass
class SourceRecord:
    """Where one source file came from and what may be done with it."""

    source_id: str
    dataset_id: str
    dataset_version: str
    relpath: str
    sha256: str
    size_bytes: int
    use_class: str = USE_UNKNOWN
    license_note: str = ""
    source_url: str | None = None
    acquisition_method: str = "unknown"
    acquired_utc: str = ""

    def __post_init__(self) -> None:
        if self.use_class not in VALID_USE_CLASSES:
            raise ValueError(f"use_class {self.use_class!r} not one of {VALID_USE_CLASSES}")
        if not self.acquired_utc:
            self.acquired_utc = utc_now()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SourceRecord:
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


def describe_file(
    path: Path,
    *,
    root: Path,
    dataset_id: str,
    dataset_version: str,
    use_class: str = USE_UNKNOWN,
    license_note: str = "",
    source_url: str | None = None,
    acquisition_method: str = "unknown",
) -> SourceRecord:
    """Build a provenance record for one file under `root`."""
    relpath = path.resolve().relative_to(root.resolve()).as_posix()
    digest = sha256_file(path)
    # Hashing path *and* content keeps the id stable across re-extraction to a
    # different directory while still giving two byte-identical files distinct
    # ids — de-duplication is the dedupe module's job, and it cannot flag a
    # pair that shares one identity.
    identity = sha256_bytes(f"{relpath}\n{digest}".encode())
    return SourceRecord(
        source_id=f"{dataset_id}:{identity[:16]}",
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        relpath=relpath,
        sha256=digest,
        size_bytes=path.stat().st_size,
        use_class=use_class,
        license_note=license_note,
        source_url=source_url,
        acquisition_method=acquisition_method,
    )
