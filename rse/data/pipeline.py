"""The Phase 1 ingestion pipeline: manifest in, intake records plus report out.

Source files are only ever read; every derived artifact is written under
`datasets/processed/<dataset>/<release>/` so the raw tree stays a faithful copy
of what was downloaded (proposal 5.2, step 7).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from pickle import UnpicklingError
from typing import Any

from rse import INTAKE_SCHEMA_VERSION, __version__, paths
from rse.data import dedupe
from rse.data.datasets import DatasetError, load_release
from rse.data.exprsco import ExprscoError, load_exprsco, score_to_composition
from rse.data.midi_io import IntakeComposition, MidiParseError, parse_midi
from rse.data.normalize import build_intake_record
from rse.data.provenance import SourceRecord, utc_now
from rse.data.stats import DatasetStats
from rse.hardware.profile import HardwareProfile, load_profile
from rse.jsonio import read_jsonl, write_json, write_jsonl


@dataclass
class IngestConfig:
    """Everything that affects the output of a run, logged into the report."""

    profile_id: str = "nes"
    #: Which reader parses the source files. Normally taken from the release.
    source_format: str = "midi"
    quantize_grid: float | None = None
    duplicate_threshold: float = dedupe.DEFAULT_THRESHOLD
    sketch_size: int | None = dedupe.DEFAULT_SKETCH_SIZE
    write_records: bool = True
    limit: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "source_format": self.source_format,
            "quantize_grid": self.quantize_grid,
            "duplicate_threshold": self.duplicate_threshold,
            "sketch_size": self.sketch_size,
            "write_records": self.write_records,
            "limit": self.limit,
        }


@dataclass
class IngestResult:
    report: dict[str, Any]
    index_path: Path | None
    stats_path: Path | None
    report_path: Path | None
    records: list[dict[str, Any]] = field(default_factory=list)


def read_source(path: Path, profile: HardwareProfile, source_format: str) -> IntakeComposition:
    """Parse one source file with the reader its release calls for."""
    if source_format == "midi":
        return parse_midi(path, profile)
    if source_format == "exprsco":
        return score_to_composition(load_exprsco(path), path, profile)
    raise ValueError(f"no reader for source format {source_format!r}")


def processed_dir(dataset_id: str, release_id: str, *, datasets_dir: Path | None = None) -> Path:
    root = datasets_dir or paths.DATASETS_DIR
    return root / "processed" / dataset_id / release_id


def raw_dir(dataset_id: str, release_id: str, *, datasets_dir: Path | None = None) -> Path:
    root = datasets_dir or paths.DATASETS_DIR
    return root / "raw" / dataset_id / release_id


def ingest_release(
    dataset_id: str,
    release_id: str,
    *,
    config: IngestConfig | None = None,
    datasets_dir: Path | None = None,
    configs_dir: Path | None = None,
    keep_records_in_memory: bool = False,
) -> IngestResult:
    """Normalize every file in a release's manifest."""
    config = config or IngestConfig()
    out_dir = processed_dir(dataset_id, release_id, datasets_dir=datasets_dir)
    manifest_path = out_dir / "manifest.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"no manifest at {manifest_path}; run `rse fetch --dataset {dataset_id} "
            f"--release {release_id}` first"
        )
    sources = [SourceRecord.from_dict(row) for row in read_jsonl(manifest_path)]
    source_root = raw_dir(dataset_id, release_id, datasets_dir=datasets_dir)
    profile = load_profile(config.profile_id, configs_dir=_hardware_dir(configs_dir))

    # The release decides which reader is used, so a caller cannot accidentally
    # parse a score release as MIDI.
    try:
        release = load_release(dataset_id, release_id, configs_dir=_dataset_dir(configs_dir))
    except DatasetError:
        release = None
    if release is not None:
        config.source_format = release.format
    return ingest(
        sources,
        source_root=source_root,
        out_dir=out_dir,
        profile=profile,
        config=config,
        keep_records_in_memory=keep_records_in_memory,
    )


def ingest(
    sources: list[SourceRecord],
    *,
    source_root: Path,
    out_dir: Path,
    profile: HardwareProfile,
    config: IngestConfig | None = None,
    keep_records_in_memory: bool = False,
) -> IngestResult:
    """Parse, normalize, measure and de-duplicate a list of source files."""
    config = config or IngestConfig()
    started = time.monotonic()
    stats = DatasetStats()
    index: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    dup_entries: list[tuple[str, str, set[int]]] = []
    kept: list[dict[str, Any]] = []
    records_dir = out_dir / "records"

    selected = sources[: config.limit] if config.limit else sources
    for source in selected:
        path = source_root / source.relpath
        try:
            comp = read_source(path, profile, config.source_format)
        except (MidiParseError, ExprscoError, UnpicklingError, OSError, ValueError) as exc:
            failures.append(
                {"relpath": source.relpath, "source_id": source.source_id, "error": str(exc)}
            )
            continue

        record = build_intake_record(comp, source, profile, quantize_grid=config.quantize_grid)
        stats.add(record)
        dup_entries.append(
            (
                record["composition_id"],
                dedupe.content_hash(record),
                dedupe.fingerprint(record, sketch_size=config.sketch_size),
            )
        )

        record_path: Path | None = None
        if config.write_records:
            record_path = records_dir / f"{_record_filename(source)}.json"
            write_json(record_path, record, indent=None)
        if keep_records_in_memory:
            kept.append(record)

        measurements = record["measurements"]
        index.append(
            {
                "composition_id": record["composition_id"],
                "name": record["name"],
                "record_path": (
                    record_path.relative_to(out_dir).as_posix() if record_path else None
                ),
                "sha256": source.sha256,
                "use_class": source.use_class,
                "split": _split_from_relpath(source.relpath),
                "n_notes": measurements["n_notes"],
                "n_bars": measurements["n_bars"],
                "end_sec": measurements["end_sec"],
                "channels_used": measurements["channels_used"],
                "hardware_compliant": measurements["hardware_compliant"],
                "n_warnings": len(record["warnings"]),
                "duplicate_of": None,
            }
        )

    duplicates = dedupe.find_duplicates(dup_entries, threshold=config.duplicate_threshold)
    for row in index:
        row["duplicate_of"] = duplicates.duplicate_of.get(row["composition_id"])

    report = {
        "generated_utc": utc_now(),
        "rse_version": __version__,
        "intake_schema_version": INTAKE_SCHEMA_VERSION,
        "hardware_profile": profile.profile_id,
        "hardware_schema_version": profile.schema_version,
        "config": config.to_dict(),
        "n_sources": len(selected),
        "n_normalized": len(index),
        "n_failed": len(failures),
        "n_flagged_duplicates": len(duplicates.duplicate_of),
        "duration_sec": round(time.monotonic() - started, 3),
        "failures": failures[:200],
        "n_failures_truncated": max(0, len(failures) - 200),
        "duplicates": duplicates.to_dict(),
        "statistics": stats.to_dict(),
    }

    index_path = stats_path = report_path = None
    if config.write_records or index:
        index_path = write_jsonl(out_dir / "index.jsonl", index)
        stats_path = write_json(out_dir / "statistics.json", stats.to_dict())
        report_path = write_json(out_dir / "report.json", report)

    return IngestResult(
        report=report,
        index_path=index_path,
        stats_path=stats_path,
        report_path=report_path,
        records=kept,
    )


def _record_filename(source: SourceRecord) -> str:
    """A flat, Windows-safe filename that is stable across runs."""
    stem = Path(source.relpath).stem
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in stem)[:80]
    return f"{safe}_{source.sha256[:12]}"


def _split_from_relpath(relpath: str) -> str | None:
    """Recover a dataset split from its directory name, when present.

    NES-MDB ships composer-disjoint train/valid/test directories; preserving
    which one a file came from is what keeps the Phase 4 evaluation honest.
    """
    parts = Path(relpath).parts
    for part in parts:
        lowered = part.lower()
        if lowered in ("train", "training"):
            return "train"
        if lowered in ("valid", "validation", "val"):
            return "valid"
        if lowered == "test":
            return "test"
    return None


def _hardware_dir(configs_dir: Path | None) -> Path | None:
    """Accept either a configs root or a hardware directory."""
    if configs_dir is None:
        return None
    candidate = configs_dir / "hardware"
    return candidate if candidate.is_dir() else configs_dir


def _dataset_dir(configs_dir: Path | None) -> Path | None:
    """Accept either a configs root or a datasets directory."""
    if configs_dir is None:
        return None
    candidate = configs_dir / "datasets"
    return candidate if candidate.is_dir() else configs_dir
