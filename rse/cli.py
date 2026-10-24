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
from rse.events.convert import (
    composition_from_intake,
    composition_from_midi,
    composition_to_intake,
    round_trip_midi,
)
from rse.events.schema import Composition
from rse.exporters.midi import write_midi
from rse.exporters.wav import write_stems, write_wav
from rse.generation.loops import validate_loop
from rse.generation.plan import ENERGIES, FORMS, LENGTHS, MOODS, PlanSummary, make_plan
from rse.generation.procedural import generate
from rse.hardware.constraints import ConstraintEngine
from rse.hardware.profile import ProfileError, available_profiles, load_profile
from rse.jsonio import read_json, write_json
from rse.renderers.base import DEFAULT_SAMPLE_RATE, RenderSettings, notes_from_record
from rse.renderers.nes_apu import NesApuRenderer
from rse.tokenizers.tokenizer import Conditioning, RseTokenizer
from rse.tokenizers.vocab import TokenizerConfig
from rse.ui.app import DEFAULT_HOST as UI_DEFAULT_HOST
from rse.ui.app import DEFAULT_PORT as UI_DEFAULT_PORT


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
    render.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output path; .wav is appended when no extension is given",
    )
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

    validate = sub.add_parser(
        "validate",
        help="check a composition against the hardware profile and its loop boundary",
    )
    validate.add_argument("--profile", default="nes")
    validate.add_argument("--file", type=Path, default=None, help="a MIDI file to check")
    validate.add_argument(
        "--record", type=Path, default=None, help="an intake record or RSE-EF document"
    )
    validate.add_argument(
        "--fix",
        type=Path,
        default=None,
        help="write a repaired RSE-EF document to this path",
    )
    validate.add_argument(
        "--round-trip",
        action="store_true",
        help="also export to MIDI, read it back, and report any drift",
    )
    validate.set_defaults(func=cmd_validate)

    train = sub.add_parser("train", help="train a composition model (requires the ml extra)")
    train.add_argument("--profile", default="nes")
    train.add_argument(
        "--source",
        default="procedural",
        choices=("procedural", "dataset"),
        help="synthetic corpus from the rule-based generator, or an ingested release",
    )
    train.add_argument("--dataset", default="nesmdb")
    train.add_argument("--release", default="midi")
    train.add_argument("--corpus-size", type=int, default=200, help="procedural pieces to make")
    train.add_argument("--corpus-limit", type=int, default=None, help="cap dataset compositions")
    train.add_argument("--window-bars", type=int, default=16)
    train.add_argument(
        "--no-sections",
        action="store_true",
        help="do not emit Section tokens; the ablation arm for whether they "
        "help, left unresolved by the 32-bar run",
    )
    train.add_argument(
        "--no-aligned-grid",
        action="store_true",
        help="step on one source frame per position instead of fitting each "
        "piece's tracker row; the baseline arm of the grid comparison",
    )

    train.add_argument(
        "--windows",
        type=Path,
        default=None,
        help="cache tokenized windows here; reused on later runs instead of "
        "rebuilding the corpus, which dominates start-up on a large dataset",
    )
    train.add_argument(
        "--rebuild-windows",
        action="store_true",
        help="ignore an existing window cache and rebuild it",
    )
    train.add_argument("--steps", type=int, default=2000)
    train.add_argument("--batch-size", type=int, default=8)
    train.add_argument("--seq-len", type=int, default=512)
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--layers", type=int, default=6)
    train.add_argument("--heads", type=int, default=8)
    train.add_argument("--width", type=int, default=256, help="d_model")
    train.add_argument(
        "--patience",
        type=int,
        default=5,
        help="stop after this many evaluations without a validation "
        "improvement; raise it to compare two runs at an equal step budget, "
        "since an arm that plateaus early otherwise trains far less than its "
        "comparator and looks worse for a reason that is not its encoding",
    )
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--device", default=None, help="cpu, cuda, or omit to detect")
    train.add_argument(
        "--out",
        type=Path,
        default=paths.CHECKPOINTS_DIR / "rse_model.pt",
        help="checkpoint path",
    )
    train.set_defaults(func=cmd_train)

    windows_cmd = sub.add_parser(
        "windows",
        help="tokenize a corpus into training windows, for training elsewhere",
    )
    windows_cmd.add_argument("--profile", default="nes")
    windows_cmd.add_argument("--source", default="dataset", choices=("procedural", "dataset"))
    windows_cmd.add_argument("--dataset", default="nesmdb")
    windows_cmd.add_argument("--release", default="exprsco")
    windows_cmd.add_argument("--corpus-size", type=int, default=200)
    windows_cmd.add_argument("--corpus-limit", type=int, default=None)
    windows_cmd.add_argument("--window-bars", type=int, default=16)
    windows_cmd.add_argument(
        "--no-sections",
        action="store_true",
        help="do not emit Section tokens; the ablation arm for whether they "
        "help, left unresolved by the 32-bar run",
    )
    windows_cmd.add_argument(
        "--no-aligned-grid",
        action="store_true",
        help="step on one source frame per position instead of fitting each "
        "piece's tracker row; the baseline arm of the grid comparison",
    )

    windows_cmd.add_argument("--hop-bars", type=int, default=None)
    windows_cmd.add_argument("--seed", type=int, default=0)
    windows_cmd.add_argument("--out", type=Path, required=True)
    windows_cmd.set_defaults(func=cmd_windows)

    sample = sub.add_parser("sample", help="generate music from a trained checkpoint")
    sample.add_argument("--profile", default="nes")
    sample.add_argument("--checkpoint", type=Path, required=True)
    sample.add_argument("--mood", default="unknown")
    sample.add_argument("--energy", default="unknown")
    sample.add_argument("--bars", type=int, default=16)
    sample.add_argument(
        "--tempo",
        type=float,
        default=None,
        help="BPM; defaults to the tempo the checkpoint was trained on",
    )
    sample.add_argument("--seed", type=int, default=None)
    sample.add_argument("--temperature", type=float, default=1.0)
    sample.add_argument("--top-k", type=int, default=40)
    sample.add_argument(
        "--noise-temperature",
        type=float,
        default=None,
        help="sample the noise channel at its own temperature; percussion "
        "collapses into static at the low values that help the melody",
    )
    sample.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="token budget for one piece; 32 bars at full density needs ~2700",
    )
    sample.add_argument("--top-p", type=float, default=0.95)
    sample.add_argument(
        "--unconstrained",
        action="store_true",
        help="do not mask illegal tokens; shows raw model behaviour",
    )
    sample.add_argument("--out", type=Path, default=None, help="output basename")
    sample.add_argument("--stems", action="store_true")
    sample.set_defaults(func=cmd_sample)

    evaluate = sub.add_parser(
        "evaluate", help="measure a checkpoint against the rule-based baseline"
    )
    evaluate.add_argument("--profile", default="nes")
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--n", type=int, default=16, help="samples to draw from each")
    evaluate.add_argument("--bars", type=int, default=16)
    evaluate.add_argument("--temperature", type=float, default=1.0)
    evaluate.add_argument("--noise-temperature", type=float, default=None)
    evaluate.add_argument("--max-tokens", type=int, default=2048)
    evaluate.add_argument("--seed", type=int, default=0)
    evaluate.add_argument(
        "--corpus-size", type=int, default=64, help="reference pieces for originality screening"
    )
    evaluate.add_argument("--out", type=Path, default=None, help="write the full report here")
    evaluate.set_defaults(func=cmd_evaluate)

    ui = sub.add_parser("ui", help="launch the local web interface (needs the ui extra)")
    ui.add_argument("--profile", default="nes")
    ui.add_argument(
        "--host",
        default=UI_DEFAULT_HOST,
        help=f"bind address; defaults to {UI_DEFAULT_HOST} and refuses anything "
        "non-loopback without --allow-remote",
    )
    ui.add_argument("--port", type=int, default=UI_DEFAULT_PORT)
    ui.add_argument("--out-dir", type=Path, default=None, help="where generated files go")
    ui.add_argument(
        "--allow-remote",
        action="store_true",
        help="permit a non-loopback bind, exposing the interface to the network",
    )
    ui.add_argument("--open", action="store_true", help="open a browser window")
    ui.set_defaults(func=cmd_ui)

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


def with_extension(base: Path, extension: str) -> Path:
    """Append `extension` to a basename, replacing only a matching one.

    `Path.with_suffix` replaces whatever follows the last dot, which silently
    collides on a basename that contains one: `--out take_t0.7` and
    `--out take_t0.9` both became `take_t0.wav`, so sampling two temperatures
    wrote one file and the second overwrote the first with no warning.
    """
    if base.suffix.lower() == extension.lower():
        return base
    return base.with_name(base.name + extension)


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

    out_path = args.out or Path(default_stem)
    # `generate` and `sample` treat --out as a basename and append extensions.
    # This took a literal path, so `--out piece` wrote a file called `piece`
    # with no suffix: a valid WAV the operating system would not open.
    out_path = with_extension(out_path, ".wav")
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
    wav_path = with_extension(base, ".wav")
    midi_path = with_extension(base, ".mid")
    json_path = with_extension(base, ".json")

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


def cmd_validate(args: argparse.Namespace) -> int:
    if (args.file is None) == (args.record is None):
        print(
            "error: pass exactly one of --file (MIDI) or --record (intake record or RSE-EF)",
            file=sys.stderr,
        )
        return 2

    profile = load_profile(args.profile, configs_dir=_hardware_configs(args.configs_dir))
    if args.file is not None:
        composition = composition_from_midi(args.file, profile)
    else:
        data = read_json(args.record)
        # Accept either representation: an RSE-EF document declares its own
        # schema version, an intake record does not.
        if str(data.get("schema_version", "")).startswith("rse-ef/"):
            composition = Composition.from_dict(data)
        else:
            composition = composition_from_intake(data)

    engine = ConstraintEngine(profile)
    repaired, report = engine.apply(composition)
    loop_report = validate_loop(composition_to_intake(composition, profile=profile))

    round_trip = None
    if args.round_trip:
        _, round_trip = round_trip_midi(repaired, profile)

    if args.fix is not None:
        write_json(args.fix, repaired.to_dict())

    payload = {
        "profile": profile.profile_id,
        "schema_version": composition.schema_version,
        "n_events": len(composition.events),
        "constraints": report.to_dict(),
        "loop": loop_report.to_dict(),
        "round_trip": None if round_trip is None else round_trip.to_dict(),
        "fixed": None if args.fix is None else str(args.fix),
    }
    if args.json:
        _print_json(payload)
    else:
        print(f"{len(composition.events)} events, profile {profile.profile_id}")
        if report.ok:
            print("  hardware: playable as written")
        else:
            print(
                f"  hardware: {len(report.violations)} violation(s), "
                f"{len(report.modifications)} change(s) to fix them {report.counts()}"
            )
            for modification in report.modifications[:10]:
                print(
                    f"    {modification.kind:<11}{modification.channel_id:<4}"
                    f"@{modification.tick:<8}{modification.detail}"
                )
            extra = len(report.modifications) - 10
            if extra > 0:
                print(f"    ... and {extra} more")
        print("  loop: " + ("clean" if loop_report.ok else "; ".join(loop_report.issues)))
        if round_trip is not None:
            print(
                f"  round trip: {'lossless' if round_trip.ok else 'LOSSY'} "
                f"({round_trip.n_matched}/{round_trip.n_original} events matched"
                f"{'' if round_trip.ok else f', {len(round_trip.drifted)} drifted'})"
            )
        if args.fix is not None:
            print(f"  repaired: {args.fix}")

    ok = report.ok and loop_report.ok and (round_trip is None or round_trip.ok)
    return 0 if ok else 1


def _dominant_tempo(windows, tokenizer) -> float:
    """The most common Tempo token across a window set, as BPM."""
    from collections import Counter

    from rse.tokenizers.vocab import TEMPO

    counts: Counter[float] = Counter()
    for window in windows:
        for token_id in window.ids[:8]:
            token = tokenizer.vocabulary.token_of(token_id)
            if token.startswith(TEMPO):
                counts[float(token[len(TEMPO) :])] += 1
                break
    return counts.most_common(1)[0][0] if counts else 120.0


def _require_torch() -> None:
    """Fail with instructions rather than a traceback when torch is absent."""
    try:
        import torch  # noqa: F401
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ProfileError(
            'this command needs PyTorch; install it with `pip install -e ".[ml]"`'
        ) from exc


def cmd_train(args: argparse.Namespace) -> int:
    _require_torch()
    from rse.models.transformer import ModelConfig
    from rse.training.corpus import CorpusInfo, corpus_from_procedural, corpus_from_release
    from rse.training.dataset import WindowStats, build_windows, load_windows, save_windows
    from rse.training.train import TrainConfig, train

    profile = load_profile(args.profile, configs_dir=_hardware_configs(args.configs_dir))
    tokenizer = RseTokenizer(
        TokenizerConfig(
            profile_id=profile.profile_id,
            aligned_grid=not getattr(args, "no_aligned_grid", False),
            emit_sections=not getattr(args, "no_sections", False),
        )
    )

    cached = args.windows is not None and args.windows.is_file() and not args.rebuild_windows
    if cached:
        print(f"windows   : reusing cache {args.windows}", flush=True)
        windows = load_windows(args.windows)
        window_stats = WindowStats(n_windows=len(windows))
        for window in windows:
            window_stats.total_tokens += len(window)
            window_stats.max_tokens = max(window_stats.max_tokens, len(window))
            window_stats.per_split[window.split] = window_stats.per_split.get(window.split, 0) + 1
        corpus_info = CorpusInfo(source=f"cache:{args.windows}")
    else:
        if args.source == "procedural":
            entries, corpus_info = corpus_from_procedural(
                profile, n=args.corpus_size, seed=args.seed, n_bars=args.window_bars
            )
        else:
            print("corpus    : loading records...", flush=True)
            entries, corpus_info = corpus_from_release(
                args.dataset,
                args.release,
                datasets_dir=args.datasets_dir,
                limit=args.corpus_limit,
                seed=args.seed,
            )
        print(f"corpus    : {corpus_info.n_compositions} compositions, tokenizing...", flush=True)
        windows, window_stats = build_windows(entries, tokenizer, window_bars=args.window_bars)
        # Release the compositions before training allocates; holding a whole
        # dataset in memory alongside the model is what made the first real run
        # take 7 GB.
        del entries
        if args.windows is not None:
            save_windows(args.windows, windows)
            print(f"windows   : cached to {args.windows}", flush=True)

    if not windows:
        print("error: the corpus produced no training windows", file=sys.stderr)
        return 2

    model_config = ModelConfig(
        vocab_size=len(tokenizer),
        n_layers=args.layers,
        n_heads=args.heads,
        d_model=args.width,
        max_seq_len=max(args.seq_len + 1, window_stats.max_tokens),
    )
    train_config = TrainConfig(
        seed=args.seed,
        max_steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        learning_rate=args.learning_rate,
        patience=args.patience,
        device=args.device,
    )
    _model, history, report = train(
        windows,
        tokenizer,
        model_config=model_config,
        config=train_config,
        checkpoint_path=args.out,
        progress=not args.json,
    )
    report["corpus"] = corpus_info.to_dict()
    report["windows"] = window_stats.to_dict()
    # The tempo the corpus was labelled with, so `rse sample` can default to it
    # instead of to a number that happens to be wrong for this checkpoint.
    report["conditioning_tempo_bpm"] = _dominant_tempo(windows, tokenizer)
    write_json(args.out.with_suffix(".report.json"), report)

    if args.json:
        _print_json(report)
        return 0

    print(f"corpus    : {corpus_info.n_compositions} compositions from {corpus_info.source}")
    if corpus_info.synthetic:
        print("            synthetic: this validates the training pipeline, not musical quality")
    print(f"windows   : {window_stats.n_windows} ({window_stats.per_split})")
    print(f"model     : {report['n_parameters']:,} parameters on {report['device']}")
    print(
        f"best loss : {history.best_valid_loss:.4f} at step {history.best_step}"
        f"{' (stopped early)' if history.stopped_early else ''}"
    )
    print(f"time      : {history.wall_clock_sec:.1f}s")
    print(f"checkpoint: {args.out}")
    return 0


def cmd_windows(args: argparse.Namespace) -> int:
    """Tokenize a corpus to disk. Needs no PyTorch, so it runs anywhere.

    This is the expensive, repeatable half of training set-up, and the only
    part that needs the dataset. Producing it separately is what lets training
    happen on a machine that has a GPU but no copy of NES-MDB.
    """
    from rse.training.corpus import corpus_from_procedural, corpus_from_release
    from rse.training.dataset import build_windows, save_windows

    profile = load_profile(args.profile, configs_dir=_hardware_configs(args.configs_dir))
    tokenizer = RseTokenizer(
        TokenizerConfig(
            profile_id=profile.profile_id,
            aligned_grid=not getattr(args, "no_aligned_grid", False),
            emit_sections=not getattr(args, "no_sections", False),
        )
    )

    if args.source == "procedural":
        entries, corpus_info = corpus_from_procedural(
            profile, n=args.corpus_size, seed=args.seed, n_bars=args.window_bars
        )
    else:
        entries, corpus_info = corpus_from_release(
            args.dataset,
            args.release,
            datasets_dir=args.datasets_dir,
            limit=args.corpus_limit,
            seed=args.seed,
        )
    windows, stats = build_windows(
        entries, tokenizer, window_bars=args.window_bars, hop_bars=args.hop_bars
    )
    if not windows:
        print("error: the corpus produced no windows", file=sys.stderr)
        return 2
    save_windows(args.out, windows)

    sidecar = write_json(
        args.out.with_suffix(".meta.json"),
        {
            "corpus": corpus_info.to_dict(),
            "windows": stats.to_dict(),
            "window_bars": args.window_bars,
            "hop_bars": args.hop_bars,
            "vocab_size": len(tokenizer),
            "tokenizer": tokenizer.config.to_dict(),
            "use_class": "research_only" if args.source == "dataset" else "commercially_cleared",
        },
    )
    if args.json:
        _print_json({"windows": str(args.out), "meta": str(sidecar), "stats": stats.to_dict()})
        return 0
    print(f"{stats.n_windows} windows from {corpus_info.n_compositions} compositions")
    print(f"  per split : {stats.per_split}")
    print(f"  tokens    : mean {stats.mean_tokens:.0f}, max {stats.max_tokens}")
    print(f"  vocabulary: {len(tokenizer)} tokens")
    print(f"  windows   : {args.out}")
    print(f"  metadata  : {sidecar}")
    if args.source == "dataset":
        print(
            "\n  note: these windows are derived from research-only data. Moving\n"
            "  them to a third-party host is a deliberate step, not an incidental\n"
            "  one -- see docs/gpu-training.md."
        )
    return 0


def cmd_sample(args: argparse.Namespace) -> int:
    _require_torch()
    from rse.generation.sampler import SamplingConfig, sample
    from rse.training.train import load_checkpoint

    profile = load_profile(args.profile, configs_dir=_hardware_configs(args.configs_dir))
    model, tokenizer, payload = load_checkpoint(args.checkpoint)

    tempo = args.tempo
    if tempo is None:
        # Default to whatever the checkpoint was trained on. Guessing 120 made
        # exprsco-trained output play at well under half speed.
        tempo = payload.get("report", {}).get("conditioning_tempo_bpm") or 120.0
    conditioning = Conditioning(
        mood=args.mood, energy=args.energy, tempo_bpm=tempo, n_bars=args.bars
    )
    config = SamplingConfig(
        seed=args.seed,
        temperature=args.temperature,
        channel_temperature=({"NO": args.noise_temperature} if args.noise_temperature else {}),
        top_k=args.top_k,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        constrain_to_grammar=not args.unconstrained,
    )
    result = sample(model, tokenizer, conditioning=conditioning, config=config)

    # Model output has no compliance guarantee, so it goes through the
    # constraint engine before anything tries to render it.
    engine = ConstraintEngine(profile)
    playable, constraint_report = engine.apply(result.composition)

    base = args.out or Path(f"rse_sample_{args.seed if args.seed is not None else 'rand'}")
    base.parent.mkdir(parents=True, exist_ok=True)
    record = composition_to_intake(playable, profile=profile)
    renderer = NesApuRenderer(profile)
    render_settings = RenderSettings(tail_sec=0.0)
    notes = notes_from_record(record)
    if not notes:
        print("error: the model produced no renderable notes", file=sys.stderr)
        return 1
    rendered = renderer.render(notes, render_settings)

    write_wav(with_extension(base, ".wav"), rendered.samples, rendered.sample_rate)
    write_midi(with_extension(base, ".mid"), record)
    loop_report = validate_loop(record, samples=rendered.samples, sample_rate=rendered.sample_rate)
    payload_summary = {
        **result.to_dict(),
        "constraints": constraint_report.to_dict(),
        "loop": loop_report.to_dict(),
        "checkpoint": str(args.checkpoint),
        "trained_on": payload.get("report", {}).get("corpus", {}),
        "wav": str(with_extension(base, ".wav")),
        "midi": str(with_extension(base, ".mid")),
    }
    write_json(with_extension(base, ".json"), {**record, "sample": payload_summary})

    stem_paths: dict[str, Path] = {}
    if args.stems:
        stem_paths = write_stems(
            base.parent / f"{base.name}_stems", rendered.stems, rendered.sample_rate
        )

    if args.json:
        _print_json({**payload_summary, "stems": {k: str(v) for k, v in stem_paths.items()}})
        return 0

    print(f"sampled {len(result.ids)} tokens -> {len(playable.events)} events")
    print(f"  {rendered.duration_sec:.1f}s, {len(notes)} notes")
    if result.hit_token_limit:
        print("  ! stopped at the token limit without emitting an end token")
    if constraint_report.ok:
        print("  hardware: playable as sampled")
    else:
        print(f"  hardware: repaired {constraint_report.counts()}")
    print("  loop: " + ("clean" if loop_report.ok else "; ".join(loop_report.issues)))
    print(f"  wav : {with_extension(base, '.wav')}")
    print(f"  midi: {with_extension(base, '.mid')}")
    for path in stem_paths.values():
        print(f"  stem: {path}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    _require_torch()
    from rse.generation.plan import make_plan
    from rse.generation.procedural import generate as generate_procedural
    from rse.generation.sampler import SamplingConfig, sample
    from rse.training.corpus import corpus_from_procedural
    from rse.training.evaluate import (
        aggregate,
        compare,
        corpus_fingerprints,
        evaluate_composition,
    )
    from rse.training.train import load_checkpoint

    profile = load_profile(args.profile, configs_dir=_hardware_configs(args.configs_dir))
    model, tokenizer, _payload = load_checkpoint(args.checkpoint)

    reference, _info = corpus_from_procedural(
        profile, n=args.corpus_size, seed=args.seed, n_bars=args.bars
    )
    fingerprints = corpus_fingerprints([c for c, _, _ in reference], profile)

    model_metrics = []
    for index in range(args.n):
        result = sample(
            model,
            tokenizer,
            conditioning=Conditioning(n_bars=args.bars),
            config=SamplingConfig(
                seed=args.seed + index,
                temperature=args.temperature,
                channel_temperature=(
                    {"NO": args.noise_temperature} if args.noise_temperature else {}
                ),
                max_tokens=args.max_tokens,
            ),
        )
        # Measure what the pipeline actually renders. Raw model output carries
        # no compliance guarantee -- that is what the constraint engine is for
        # -- so judging the raw sample fails the model for something the system
        # repairs by design. Raw compliance is still reported, separately.
        engine = ConstraintEngine(profile)
        raw_compliant = not engine.validate(result.composition)
        playable, _repair = engine.apply(result.composition)
        metrics = evaluate_composition(playable, profile, corpus_fingerprints=fingerprints)
        metrics.raw_hardware_compliant = raw_compliant
        model_metrics.append(metrics)

    baseline_metrics = []
    for index in range(args.n):
        generated = generate_procedural(
            profile, make_plan(seed=900_000 + args.seed + index, n_bars=args.bars)
        )
        baseline_metrics.append(
            evaluate_composition(
                composition_from_intake(generated.record),
                profile,
                corpus_fingerprints=fingerprints,
            )
        )

    comparison = compare(
        aggregate("rule-based baseline", baseline_metrics),
        aggregate("model", model_metrics),
    )
    if args.out:
        write_json(
            args.out,
            {
                "comparison": comparison,
                "model_metrics": [m.to_dict() for m in model_metrics],
                "baseline_metrics": [m.to_dict() for m in baseline_metrics],
            },
        )

    if args.json:
        _print_json(comparison)
        return 0

    baseline = comparison["baseline"]
    candidate = comparison["candidate"]
    print(f"{args.n} samples each, {args.bars} bars\n")
    print(f"  {'metric':<26}{'baseline':>12}{'model':>12}{'delta':>10}")
    for name in sorted(baseline["means"]):
        base_value = baseline["means"].get(name, 0.0)
        model_value = candidate["means"].get(name, 0.0)
        print(
            f"  {name:<26}{base_value:>12.3f}{model_value:>12.3f}{model_value - base_value:>+10.3f}"
        )
    for name in sorted(baseline["rates"]):
        base_value = baseline["rates"].get(name, 0.0)
        model_value = candidate["rates"].get(name, 0.0)
        print(
            f"  {name:<26}{base_value:>11.0%}{model_value:>12.0%}{model_value - base_value:>+10.0%}"
        )
    print(f"\n  technical bar: {comparison['technical_bar']}")
    print(f"  clears it: {comparison['clears_technical_bar']}")
    if args.out:
        print(f"  report: {args.out}")
    return 0 if comparison["clears_technical_bar"] else 1


def cmd_ui(args: argparse.Namespace) -> int:
    from rse.ui.app import DEFAULT_OUTPUT_DIR, UiConfig, UiError, launch

    config = UiConfig(
        host=args.host,
        port=args.port,
        output_dir=args.out_dir or DEFAULT_OUTPUT_DIR,
        configs_dir=_hardware_configs(args.configs_dir),
        profile_id=args.profile,
        allow_remote=args.allow_remote,
        open_browser=args.open,
    )
    try:
        launch(config)
    except UiError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


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
