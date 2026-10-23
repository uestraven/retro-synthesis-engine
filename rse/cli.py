"""Command-line entry point.

Everything the pipeline can do is reachable here without starting a web UI
(proposal 9.1). Each command prints a short human summary by default and a
machine-readable JSON document with `--json`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from rse import __version__, paths
from rse.data.datasets import DatasetError, acquire
from rse.data.midi_io import MidiParseError, parse_midi
from rse.data.normalize import build_intake_record
from rse.data.pipeline import IngestConfig, ingest_release, processed_dir
from rse.data.provenance import USE_UNKNOWN, describe_file
from rse.exporters.midi import write_midi
from rse.exporters.wav import write_stems, write_wav
from rse.generation.loops import validate_loop
from rse.generation.plan import ENERGIES, FORMS, LENGTHS, MOODS, PlanSummary, make_plan
from rse.generation.procedural import generate
from rse.hardware.profile import ProfileError, available_profiles, load_profile
from rse.jsonio import read_json, write_json
from rse.renderers.base import DEFAULT_SAMPLE_RATE, RenderSettings, notes_from_record
from rse.renderers.nes_apu import NesApuRenderer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rse", description=__doc__.splitlines()[0])
    parser.add_argument("--version", action="version", version=f"rse {__version__}")
    parser.add_argument(
        "--datasets-dir",
        type=Path,
        default=None,
        help=f"root for downloads/raw/processed (default: {paths.DATASETS_DIR})",
    )
    parser.add_argument(
        "--configs-dir",
        type=Path,
        default=None,
        help=f"root for hardware/ and datasets/ configs (default: {paths.CONFIGS_DIR})",
    )
    parser.add_argument("--json", action="store_true", help="print JSON instead of a summary")
    sub = parser.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser("fetch", help="acquire a dataset release and write its manifest")
    fetch.add_argument("--dataset", default="nesmdb")
    fetch.add_argument("--release", default="midi")
    fetch.add_argument("--archive", type=Path, default=None, help="use an archive already on disk")
    fetch.add_argument("--url", default=None, help="download URL, overriding the config")
    fetch.add_argument("--sha256", default=None, help="expected archive checksum")
    fetch.add_argument(
        "--overwrite", action="store_true", help="replace an existing extraction directory"
    )
    fetch.set_defaults(func=cmd_fetch)

    ingest = sub.add_parser(
        "ingest", help="normalize a fetched release into intake records, stats and a report"
    )
    ingest.add_argument("--dataset", default="nesmdb")
    ingest.add_argument("--release", default="midi")
    ingest.add_argument("--profile", default="nes", help="hardware profile id")
    ingest.add_argument(
        "--quantize-grid",
        type=float,
        default=None,
        help="add a quantized beat position on this grid, in quarter notes (e.g. 0.25)",
    )
    ingest.add_argument("--duplicate-threshold", type=float, default=None)
    ingest.add_argument("--limit", type=int, default=None, help="process only the first N files")
    ingest.add_argument(
        "--no-records",
        action="store_true",
        help="measure only; do not write per-composition record files",
    )
    ingest.set_defaults(func=cmd_ingest)

    inspect = sub.add_parser("inspect", help="summarize a processed release or a single file")
    inspect.add_argument("--dataset", default="nesmdb")
    inspect.add_argument("--release", default="midi")
    inspect.add_argument("--profile", default="nes")
    inspect.add_argument(
        "--file", type=Path, default=None, help="inspect one MIDI file instead of a release"
    )
    inspect.set_defaults(func=cmd_inspect)

    render = sub.add_parser("render", help="synthesize audio from a MIDI file or intake record")
    render.add_argument("--profile", default="nes")
    render.add_argument("--file", type=Path, default=None, help="a MIDI file to render")
    render.add_argument(
        "--record", type=Path, default=None, help="an intake record written by `rse ingest`"
    )
    render.add_argument("--out", type=Path, default=None, help="output WAV path")
    render.add_argument(
        "--stems", action="store_true", help="also write one WAV per hardware channel"
    )
    render.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    render.add_argument(
        "--oversample",
        type=int,
        default=4,
        help="internal oversampling factor; higher reduces aliasing",
    )
    render.add_argument(
        "--no-filters", action="store_true", help="skip the console's output high-pass"
    )
    render.add_argument("--duty-p1", type=int, choices=(0, 1, 2, 3), default=None)
    render.add_argument("--duty-p2", type=int, choices=(0, 1, 2, 3), default=None)
    render.set_defaults(func=cmd_render)

    generate = sub.add_parser(
        "generate", help="compose an original piece with the rule-based generator"
    )
    generate.add_argument("--profile", default="nes")
    generate.add_argument("--seed", type=int, default=None, help="omit for a random seed")
    generate.add_argument("--mood", default="heroic", choices=list(MOODS))
    generate.add_argument("--energy", default="medium", choices=list(ENERGIES))
    generate.add_argument("--bars", type=int, default=16, choices=list(LENGTHS))
    generate.add_argument("--tempo", type=float, default=None, help="BPM; omit to pick one")
    generate.add_argument("--form", default=None, choices=list(FORMS))
    generate.add_argument("--key", type=int, default=None, help="tonic pitch class, 0=C")
    generate.add_argument("--scale", default=None, help="override the mood's scale")
    generate.add_argument(
        "--out", type=Path, default=None, help="output basename (default: derived from the seed)"
    )
    generate.add_argument("--stems", action="store_true", help="also write per-channel WAVs")
    generate.add_argument(
        "--no-loop", action="store_true", help="do not mark or validate the piece as a loop"
    )
    generate.set_defaults(func=cmd_generate)

    profiles = sub.add_parser("profiles", help="list available hardware profiles")
    profiles.set_defaults(func=cmd_profiles)

    return parser


def cmd_fetch(args: argparse.Namespace) -> int:
    result = acquire(
        args.dataset,
        args.release,
        archive=args.archive,
        url=args.url,
        sha256=args.sha256,
        datasets_dir=args.datasets_dir,
        configs_dir=_dataset_configs(args.configs_dir),
        overwrite=args.overwrite,
    )
    if args.json:
        _print_json(result)
        return 0
    acq = result["acquisition"]
    print(f"fetched {acq['dataset_version']} ({acq['acquisition_method']})")
    print(f"  raw files   : {acq['n_source_files']} under {acq['raw_dir']}")
    print(f"  manifest    : {result['manifest_path']}")
    print(f"  use class   : {acq['use_class']}")
    if acq["license_note"]:
        print(f"  licence     : {acq['license_note']}")
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    config = IngestConfig(
        profile_id=args.profile,
        quantize_grid=args.quantize_grid,
        write_records=not args.no_records,
        limit=args.limit,
    )
    if args.duplicate_threshold is not None:
        config.duplicate_threshold = args.duplicate_threshold
    result = ingest_release(
        args.dataset,
        args.release,
        config=config,
        datasets_dir=args.datasets_dir,
        configs_dir=args.configs_dir,
    )
    if args.json:
        _print_json(result.report)
        return 0
    _print_report(result.report)
    if result.report_path:
        print(f"\nwritten: {result.report_path}")
    return 0 if result.report["n_failed"] == 0 else 1


def cmd_inspect(args: argparse.Namespace) -> int:
    profile = load_profile(args.profile, configs_dir=_hardware_configs(args.configs_dir))
    if args.file is not None:
        return _inspect_file(args, profile)

    out_dir = processed_dir(args.dataset, args.release, datasets_dir=args.datasets_dir)
    report_path = out_dir / "report.json"
    if not report_path.is_file():
        print(
            f"no report at {report_path}; run `rse ingest --dataset {args.dataset} "
            f"--release {args.release}` first",
            file=sys.stderr,
        )
        return 2
    report = read_json(report_path)
    if args.json:
        _print_json(report)
        return 0
    _print_report(report)
    return 0


def _inspect_file(args: argparse.Namespace, profile) -> int:
    path = args.file.expanduser().resolve()
    comp = parse_midi(path, profile)
    source = describe_file(
        path,
        root=path.parent,
        dataset_id="adhoc",
        dataset_version="adhoc/inspect",
        use_class=USE_UNKNOWN,
        acquisition_method="inspect",
    )
    record = build_intake_record(comp, source, profile)
    if args.json:
        _print_json(record)
        return 0

    m = record["measurements"]
    print(f"{path.name}  [{profile.display_name}]")
    print(
        f"  midi format {record['midi']['format']}, {record['midi']['n_tracks']} tracks, "
        f"{record['midi']['ticks_per_beat']} ticks/beat"
    )
    print(
        f"  {m['n_notes']} notes, {m['n_controls']} control events, "
        f"{m['n_bars']} bars, {m['end_sec']:.1f}s"
    )
    tempos = ", ".join(
        "{:g} BPM @ bar {}".format(t["bpm"], t["bar"]) for t in record["tempo_map"][:4]
    )
    print(f"  tempo: {tempos}")
    print(f"  hardware compliant: {m['hardware_compliant']}")
    print(
        f"  {'ch':<4}{'role':<10}{'notes':>7}{'range':>12}{'max_poly':>10}"
        f"{'overlaps':>10}  assigned_by"
    )
    for ch in record["channels"]:
        value_range = f"{ch['value_min']}-{ch['value_max']}" if ch["value_min"] is not None else "-"
        print(
            f"  {ch['channel_id']:<4}{ch['role']:<10}{ch['n_notes']:>7}{value_range:>12}"
            f"{ch['max_simultaneous']:>10}{ch['overlap_count']:>10}  {ch['assignment_method']}"
        )
        if ch["out_of_range_values"]:
            print(f"       out of profile range: {ch['out_of_range_values']}")
        if ch["controllers_used"]:
            print(f"       controllers: {ch['controllers_used']}")
    if record["loop_markers"]:
        print(f"  loop markers: {len(record['loop_markers'])}")
    for warning in record["warnings"][:10]:
        print(f"  ! {warning}")
    extra = len(record["warnings"]) - 10
    if extra > 0:
        print(f"  ! ... and {extra} more warnings")
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    if (args.file is None) == (args.record is None):
        print(
            "error: pass exactly one of --file (MIDI) or --record (intake record)", file=sys.stderr
        )
        return 2

    profile = load_profile(args.profile, configs_dir=_hardware_configs(args.configs_dir))

    if args.record is not None:
        record = read_json(args.record)
        if record.get("hardware_profile") not in (None, profile.profile_id):
            print(
                f"error: record targets profile {record['hardware_profile']!r}, "
                f"not {profile.profile_id!r}",
                file=sys.stderr,
            )
            return 2
        default_stem = args.record.stem
    else:
        path = args.file.expanduser().resolve()
        comp = parse_midi(path, profile)
        source = describe_file(
            path,
            root=path.parent,
            dataset_id="adhoc",
            dataset_version="adhoc/render",
            use_class=USE_UNKNOWN,
            acquisition_method="render",
        )
        record = build_intake_record(comp, source, profile)
        default_stem = path.stem

    duty = {
        cid: value
        for cid, value in (("P1", args.duty_p1), ("P2", args.duty_p2))
        if value is not None
    }
    renderer = NesApuRenderer(profile, duty_by_channel=duty)
    settings = RenderSettings(
        sample_rate=args.sample_rate,
        oversample=args.oversample,
        apply_output_filters=not args.no_filters,
    )
    notes = notes_from_record(record)
    if not notes:
        print("error: that composition contains no renderable notes", file=sys.stderr)
        return 2
    result = renderer.render(notes, settings)

    out_path = args.out or Path(f"{default_stem}.wav")
    write_wav(out_path, result.samples, result.sample_rate)
    # A sidecar so any rendered file can be traced back to what produced it
    # (proposal 9.2).
    metadata = {
        **result.to_dict(),
        "composition_id": record.get("composition_id"),
        "name": record.get("name"),
        "hardware_profile": profile.profile_id,
        "duty_by_channel": renderer.duty_by_channel,
        "n_notes": len(notes),
    }
    sidecar = write_json(out_path.with_suffix(".render.json"), metadata)

    stem_paths: dict[str, Path] = {}
    if args.stems:
        stem_paths = write_stems(
            out_path.parent / f"{out_path.stem}_stems", result.stems, result.sample_rate
        )

    if args.json:
        _print_json(
            {
                **metadata,
                "wav": str(out_path),
                "sidecar": str(sidecar),
                "stems": {k: str(v) for k, v in stem_paths.items()},
            }
        )
        return 0

    print(f"rendered {len(notes)} notes to {out_path}")
    print(
        f"  {result.duration_sec:.2f}s, {result.sample_rate} Hz, "
        f"{renderer.renderer_id}, channels: {', '.join(sorted(result.stems))}"
    )
    print(f"  metadata: {sidecar}")
    for path in stem_paths.values():
        print(f"  stem    : {path}")
    for warning in result.warnings[:10]:
        print(f"  ! {warning}")
    extra = len(result.warnings) - 10
    if extra > 0:
        print(f"  ! ... and {extra} more warnings")
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    profile = load_profile(args.profile, configs_dir=_hardware_configs(args.configs_dir))
    plan = make_plan(
        seed=args.seed,
        mood=args.mood,
        energy=args.energy,
        tempo_bpm=args.tempo,
        n_bars=args.bars,
        form=args.form,
        tonic=args.key,
        scale=args.scale,
        loop=not args.no_loop,
    )
    composition = generate(profile, plan)

    base = args.out or Path(f"rse_{plan.mood}_{plan.seed}")
    base.parent.mkdir(parents=True, exist_ok=True)
    wav_path = base.with_suffix(".wav")
    midi_path = base.with_suffix(".mid")
    json_path = base.with_suffix(".json")

    renderer = NesApuRenderer(profile)
    # A looping piece must not end in silence, or the repeat leaves a gap.
    settings = RenderSettings(tail_sec=0.0 if plan.loop else 0.25)
    result = renderer.render(notes_from_record(composition.record), settings)

    write_wav(wav_path, result.samples, result.sample_rate)
    write_midi(midi_path, composition.record)
    loop_report = validate_loop(
        composition.record, samples=result.samples, sample_rate=result.sample_rate
    )
    write_json(
        json_path,
        {
            **composition.record,
            "render": result.to_dict(),
            "loop_report": loop_report.to_dict(),
        },
    )

    stem_paths: dict[str, Path] = {}
    if args.stems:
        stem_paths = write_stems(
            base.parent / f"{base.name}_stems", result.stems, result.sample_rate
        )

    measurements = composition.record["measurements"]
    if args.json:
        _print_json(
            {
                "composition_id": composition.composition_id,
                "plan": plan.to_dict(),
                "measurements": measurements,
                "render": result.to_dict(),
                "loop_report": loop_report.to_dict(),
                "wav": str(wav_path),
                "midi": str(midi_path),
                "json": str(json_path),
                "stems": {k: str(v) for k, v in stem_paths.items()},
            }
        )
        return 0 if loop_report.ok else 1

    for line in PlanSummary(plan).lines():
        print(line)
    print(
        f"  {measurements['n_notes']} notes, {result.duration_sec:.1f}s, "
        f"hardware compliant: {measurements['hardware_compliant']}"
    )
    for channel in composition.record["channels"]:
        print(
            f"  {channel['channel_id']:<4}{channel['n_notes']:>5} notes  "
            f"range {channel['value_min']}-{channel['value_max']}  "
            f"max poly {channel['max_simultaneous']}"
        )
    print(f"  wav : {wav_path}")
    print(f"  midi: {midi_path}")
    print(f"  json: {json_path}")
    for path in stem_paths.values():
        print(f"  stem: {path}")
    if loop_report.ok:
        print("  loop: clean")
    else:
        for issue in loop_report.issues:
            print(f"  loop! {issue}")
    return 0 if loop_report.ok else 1


def cmd_profiles(args: argparse.Namespace) -> int:
    ids = available_profiles(configs_dir=_hardware_configs(args.configs_dir))
    rows = []
    for profile_id in ids:
        try:
            profile = load_profile(profile_id, configs_dir=_hardware_configs(args.configs_dir))
        except ProfileError as exc:
            rows.append({"profile_id": profile_id, "error": str(exc)})
            continue
        rows.append(
            {
                "profile_id": profile.profile_id,
                "display_name": profile.display_name,
                "schema_version": profile.schema_version,
                "channels": [
                    {
                        "channel_id": ch.channel_id,
                        "role": ch.role,
                        "pitched": ch.pitched,
                        "max_simultaneous_notes": ch.max_simultaneous_notes,
                        "range": [ch.value_min, ch.value_max],
                    }
                    for ch in profile.channels
                ],
            }
        )
    if args.json:
        _print_json(rows)
        return 0
    for row in rows:
        if "error" in row:
            print(f"{row['profile_id']}: ERROR {row['error']}")
            continue
        print(f"{row['profile_id']}  {row['display_name']}  ({row['schema_version']})")
        for ch in row["channels"]:
            kind = "pitch" if ch["pitched"] else "value"
            print(
                f"  {ch['channel_id']:<4}{ch['role']:<10}"
                f"{kind} {ch['range'][0]}-{ch['range'][1]}, "
                f"max {ch['max_simultaneous_notes']} note(s)"
            )
    return 0


def _print_report(report: dict[str, Any]) -> None:
    stats = report.get("statistics", {})
    print(
        f"{report['n_normalized']}/{report['n_sources']} files normalized "
        f"({report['n_failed']} failed) in {report['duration_sec']}s"
    )
    print(
        f"  notes       : {stats.get('n_notes', 0)} "
        f"(median {stats.get('notes_per_record', {}).get('median', 0)} per composition)"
    )
    print(
        f"  controls    : {stats.get('n_controls', 0)}; "
        f"controllers seen: {list(stats.get('controllers_used', {}))[:8]}"
    )
    print(
        f"  bars        : median {stats.get('bars_per_record', {}).get('median', 0)}, "
        f"max {stats.get('bars_per_record', {}).get('max', 0)}"
    )
    print(
        f"  compliant   : {stats.get('n_hardware_compliant', 0)}/{report['n_normalized']} "
        "already satisfy the profile's channel limits"
    )
    print(f"  duplicates  : {report['n_flagged_duplicates']} flagged for review")
    dropped = stats.get("n_dropped_notes", 0)
    if dropped:
        print(f"  dropped     : {dropped} notes had no assignable channel")
    for cid, ch in stats.get("per_channel", {}).items():
        print(
            f"  {cid:<4}{ch['n_notes']:>9} notes  range {ch['value_min']}-{ch['value_max']}  "
            f"max poly {ch['max_simultaneous_seen']}  "
            f"overlapping in {ch['records_with_overlaps']} files"
        )
        if ch["out_of_range_values"]:
            print(f"       outside profile range: {ch['out_of_range_values']}")
    warnings = stats.get("warning_kinds", {})
    if warnings:
        print(f"  warnings    : {warnings}")
    for failure in report.get("failures", [])[:5]:
        print(f"  ! {failure['relpath']}: {failure['error']}")


def _print_json(obj: Any) -> None:
    json.dump(obj, sys.stdout, indent=2)
    sys.stdout.write("\n")


def _dataset_configs(configs_dir: Path | None) -> Path | None:
    if configs_dir is None:
        return None
    candidate = configs_dir / "datasets"
    return candidate if candidate.is_dir() else configs_dir


def _hardware_configs(configs_dir: Path | None) -> Path | None:
    if configs_dir is None:
        return None
    candidate = configs_dir / "hardware"
    return candidate if candidate.is_dir() else configs_dir


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (DatasetError, MidiParseError, ProfileError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
