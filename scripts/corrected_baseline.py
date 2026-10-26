"""The corrected-pipeline listening baseline: 20 fresh eight-bar pieces.

The question is **absolute** quality, not a comparison: how many correctly
arranged Fermata compositions are Best, Good, OK or Bad? Every earlier
listening result was taken through a pipeline that rescaled the musical
timeline by `tempo_bpm / 150`, and in some pieces also shifted everything a
bar late through a dropped opening `<BAR>`. Both are fixed, so this is the
first batch whose audio actually carries the music Fermata composed.

Four pieces each from action, town, exploration, ominous and fast_rhythmic.
`sparse` stays suspended: its lead-collapse diagnosis turned out to be the
timing defect, so there is nothing left to say about it until it is re-examined
on its own.

**Nothing selects.** No corpus-distance ranking -- it failed blind validation
at Kendall tau +0.17 -- no subjective filtering, and no reordering. Metrics
are recorded for later analysis only. A piece that reaches eight valid bars is
in the batch whatever it scores.

**Structural validity is checked, and checked for the things that were
previously invisible.** Hardware validation never looked at bar structure,
which is how a piece spanning nine bars passed every check for three listening
tests. Each piece is now verified for: eight complete bars from the trim, the
declared meter, the requested tempo surviving into the encoded composition,
positions inside the eight-bar window, no negative ticks, the requested
instrumentation, and zero hardware violations. A failure is recorded with its
reason and replaced from the same scenario; the attempt count and failure rate
are reported beside the ratings, because measuring quality only over the
survivors of a silent filter overstates it.

Filenames are assigned by a shuffle seeded independently of scenario, seed and
every metric, and all files are peak-normalised to one level so loudness
carries no information about which scenario produced them.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import json
import random
import zlib
from collections import Counter
from typing import Any

from rse.analysis.restatement import ticks_per_bar
from rse.codec.benchmark import enforce_monophony, ioi_entropy
from rse.conditioning import melody_line, motif_note_participation
from rse.events.schema import Composition
from rse.hardware.constraints import ConstraintEngine
from rse.hardware.profile import load_profile
from rse.tokenizers.frozen import frozen_config
from rse.tokenizers.tokenizer import GridFit, RseTokenizer
from rse.training.evaluate import evaluate_composition
from rse.transfer.arranger import arrange_for_nes, row_seconds
from rse.transfer.ensemble import check_ensemble
from rse.transfer.fermata import RECONSTRUCTION_ID, Vocab, decode, trim_to_bars

ROW_SEC = 0.1
BARS = 8
PEAK_DBFS = -3.0

#: `sparse` is suspended: its lead collapse was the timing defect.
SCENARIOS_IN_BATCH = ("action", "town", "exploration", "ominous", "fast_rhythmic")

CHECKPOINT_REVISION = "18b647098f579ee41db67c0ad4c7f7f39ce0eac1"
CHECKPOINT_SHA256 = "4ca6a5db3a2d1510bad7acde4d313a9ad732fc91dcda4776f044c5b5bd8ea9b7"


def structural_check(
    composition: Composition,
    *,
    score: Any,
    scenario: dict[str, Any],
    trimmed: Any,
    decode_report: Any,
    engine: ConstraintEngine,
    arrange_report: Any,
) -> dict[str, Any]:
    """Everything that should have failed a nine-bar piece and did not."""
    bar_ticks = ticks_per_bar()
    events = composition.sounding_events()
    ticks = [e.tick for e in events]
    encoded_bpm = (
        60_000_000 / composition.tempo_map[0].us_per_beat
        if composition.tempo_map
        else None
    )
    ensemble = check_ensemble(
        scenario=scenario["name"],
        plan=scenario["plan"],
        score=score,
        arranged=composition,
        n_tracks_merged=arrange_report.n_tracks_merged,
    )
    span_bars = (max(ticks) + 1) / bar_ticks if ticks else 0.0
    checks = {
        "eight_complete_bars": bool(trimmed.complete and trimmed.n_bars == BARS),
        "meter_is_declared": score.beats_per_bar == 4,
        "tempo_preserved": encoded_bpm is not None
        and abs(encoded_bpm - float(scenario["tempo"])) / float(scenario["tempo"])
        < 0.02,
        "no_negative_ticks": bool(ticks) and min(ticks) >= 0,
        "onsets_inside_window": bool(ticks) and max(ticks) < BARS * bar_ticks,
        "has_events": bool(events),
        "instrumentation_as_requested": ensemble.honours_request,
        "no_hardware_violations": len(engine.validate(composition)) == 0,
    }
    return {
        "checks": checks,
        "all_passed": all(checks.values()),
        "failed": [name for name, ok in checks.items() if not ok],
        "encoded_bpm": round(encoded_bpm, 3) if encoded_bpm else None,
        "requested_bpm": float(scenario["tempo"]),
        "span_bars": round(span_bars, 3),
        "implicit_first_bar": decode_report.implicit_first_bar,
        "n_events_before_first_bar": decode_report.n_events_before_first_bar,
        "ensemble": ensemble.to_dict(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("/tmp/fermata_dl"))
    parser.add_argument("--per-scenario", type=int, default=4)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=24)
    parser.add_argument("--repetition-penalty", type=float, default=1.12)
    parser.add_argument("--max-new-tokens", type=int, default=2000)
    parser.add_argument("--seed-base", type=int, default=7000)
    parser.add_argument("--shuffle-seed", type=int, default=424242)
    parser.add_argument("--listen-dir", type=Path, default=Path("listen_corrected"))
    parser.add_argument("--out", type=Path, default=Path("corrected_baseline.json"))
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import torch
    from architecture_comparison import normalise_peak
    from codec_comparison import render_window
    from fermata_generate import SCENARIOS, build_header
    from transformers import LlamaForCausalLM
    from transformers.generation.logits_process import LogitsProcessor

    by_name = {s["name"]: s for s in SCENARIOS}
    scenarios = [by_name[n] for n in SCENARIOS_IN_BATCH]

    vocab = Vocab.load(args.model / "tokenizer" / "vocab.json")
    model = LlamaForCausalLM.from_pretrained(args.model, dtype=torch.float32).eval()

    class SuppressBpe(LogitsProcessor):
        def __init__(self, ids: list[int]) -> None:
            self.ids = torch.tensor(ids, dtype=torch.long)

        def __call__(self, _input_ids, scores):
            scores[:, self.ids] = float("-inf")
            return scores

    class StopAfterBars(torch.nn.Module):
        """Halt once the bar after the last requested one opens."""

        def __init__(self, bar_id: int, prompt_len: int, n_bars: int) -> None:
            super().__init__()
            self.bar_id = bar_id
            self.prompt_len = prompt_len
            self.limit = n_bars + 1

        def __call__(self, input_ids, _scores, **_kwargs) -> bool:
            generated = input_ids[0, self.prompt_len :]
            return bool((generated == self.bar_id).sum() >= self.limit)

    processors = [SuppressBpe(vocab.bpe_ids)]
    tokenizer = RseTokenizer(frozen_config())
    profile = load_profile("nes")
    engine = ConstraintEngine(profile)
    audit = args.listen_dir.parent / f"{args.listen_dir.name}_audit"
    audit.mkdir(parents=True, exist_ok=True)

    accepted: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    compositions: dict[str, Composition] = {}

    for scenario in scenarios:
        produced = 0
        attempt = 0
        offset = zlib.crc32(scenario["name"].encode()) % 1000
        while produced < args.per_scenario and attempt < args.per_scenario * args.max_attempts:
            seed = args.seed_base + offset * 10 + attempt
            attempt += 1
            header = build_header(vocab, scenario)
            prompt = torch.tensor([[vocab.id_of(t) for t in header]])
            torch.manual_seed(seed)
            with torch.no_grad():
                out = model.generate(
                    prompt,
                    do_sample=True,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    repetition_penalty=args.repetition_penalty,
                    max_new_tokens=args.max_new_tokens,
                    eos_token_id=vocab.id_of("<EOS>"),
                    pad_token_id=vocab.id_of("<PAD>"),
                    logits_processor=processors,
                    stopping_criteria=[
                        StopAfterBars(vocab.id_of("<BAR>"), prompt.shape[1], BARS)
                    ],
                    use_cache=True,
                )
            raw_ids = out[0].tolist()
            name = f"{scenario['name']}_s{seed}"
            trimmed = trim_to_bars(raw_ids, vocab=vocab, n_bars=BARS)
            if not (trimmed.complete and trimmed.n_bars == BARS):
                failures.append(
                    {
                        "name": name,
                        "scenario": scenario["name"],
                        "seed": seed,
                        "stage": "trim",
                        "reason": trimmed.reason,
                        "n_bars": trimmed.n_bars,
                    }
                )
                continue

            score, decode_report = decode(trimmed.ids, vocab=vocab)
            arranged, arrange_report = arrange_for_nes(
                score, row_sec=ROW_SEC, divisions=tokenizer.config.onset_divisions
            )
            arranged, _trunc = enforce_monophony(arranged)
            # Row length from the score's tempo, or the encode overrides the
            # Tempo token with 150 BPM and undoes the arranger's correction.
            fit = GridFit(
                step_ticks=float(tokenizer.ticks_per_step(480)),
                phase_ticks=0.0,
                aligned=True,
                period_sec=row_seconds(score.tempo_bpm),
            )
            rounded = enforce_monophony(
                tokenizer.decode(tokenizer.encode(arranged, fit=fit).ids).composition
            )[0]
            # Constraint engine LAST, so the D4 round trip cannot reintroduce
            # the polyphony it repairs.
            final, repair = engine.apply(rounded)

            structural = structural_check(
                final,
                score=score,
                scenario=scenario,
                trimmed=trimmed,
                decode_report=decode_report,
                engine=engine,
                arrange_report=arrange_report,
            )
            if not structural["all_passed"]:
                failures.append(
                    {
                        "name": name,
                        "scenario": scenario["name"],
                        "seed": seed,
                        "stage": "structural",
                        "reason": ",".join(structural["failed"]),
                        "structural": structural,
                    }
                )
                (audit / f"{name}.failed.json").write_text(
                    json.dumps({"structural": structural, "raw_ids": raw_ids}, indent=2)
                    + "\n"
                )
                continue

            events = final.sounding_events()
            metrics = evaluate_composition(final, profile)
            _channel, melody = melody_line(final)
            end = final.end_tick or 1
            spt = (final.time.seconds(end) / end) if final.time.seconds(end) else 0.0
            record = {
                "name": name,
                "scenario": scenario["name"],
                "seed": seed,
                "attempt": attempt,
                "termination": trimmed.reason,
                "structural": structural,
                "decode": decode_report.to_dict(),
                "arrangement": arrange_report.to_dict(),
                "hardware_modifications": len(repair)
                if isinstance(repair, list)
                else None,
                # Recorded for later analysis only. Nothing here selects.
                "metrics": {
                    "n_events": len(events),
                    "notes_per_bar": round(metrics.notes_per_bar, 3),
                    "pitch_class_entropy": round(metrics.pitch_class_entropy, 4),
                    "stepwise_fraction": round(metrics.stepwise_fraction, 4),
                    "ngram_repetition_rate": round(metrics.ngram_repetition_rate, 4),
                    "motif_note_participation": round(
                        motif_note_participation(melody), 4
                    ),
                    "ioi_entropy": round(ioi_entropy([e.tick * spt for e in events]), 4),
                    "channels": sorted({e.channel_id for e in events}),
                    "n_percussion": sum(1 for e in events if e.channel_id == "NO"),
                    "duration_sec": round(final.time.seconds(end), 3),
                },
            }
            accepted.append(record)
            compositions[name] = final
            (audit / f"{name}.json").write_text(
                json.dumps(
                    {
                        **record,
                        "header": header,
                        "raw_ids": raw_ids,
                        "trimmed_ids": trimmed.ids,
                    },
                    indent=2,
                )
                + "\n"
            )
            produced += 1
            print(
                f"  {name:<26} {structural['encoded_bpm']:>5.0f}bpm "
                f"span {structural['span_bars']:>5.2f} ev {len(events):>4} "
                f"chan {''.join(c[0] for c in record['metrics']['channels'])} "
                f"{'implicit-bar ' if structural['implicit_first_bar'] else ''}"
                f"(attempt {attempt})",
                flush=True,
            )
        if produced < args.per_scenario:
            print(
                f"  ! {scenario['name']}: only {produced}/{args.per_scenario} "
                f"after {attempt} attempts"
            )

    if not accepted:
        print("error: nothing accepted", file=sys.stderr)
        return 2

    args.listen_dir.mkdir(parents=True, exist_ok=True)
    slots = list(range(1, len(accepted) + 1))
    random.Random(args.shuffle_seed).shuffle(slots)
    key: list[dict[str, Any]] = []
    for slot, record in zip(slots, accepted, strict=True):
        filename = f"take_{slot:02d}.wav"
        destination = args.listen_dir / filename
        render_window(compositions[record["name"]], destination)
        level = normalise_peak(destination, dbfs=PEAK_DBFS)
        record["render"] = {"file": filename, **level}
        key.append(
            {
                "file": filename,
                "name": record["name"],
                "scenario": record["scenario"],
                "seed": record["seed"],
            }
        )
    key.sort(key=lambda row: row["file"])

    manifest = args.listen_dir / "RATINGS.md"
    manifest.write_text(
        "# Corrected-pipeline baseline -- ratings\n\n"
        "Twenty pieces, eight bars each. This is the first batch whose audio\n"
        "carries the music Fermata actually composed: the arranger no longer\n"
        "rescales the timeline by tempo, and a dropped opening bar no longer\n"
        "shifts a piece a bar late.\n\n"
        "Rate each piece **Best / Good / OK / Bad**, judged on its own rather\n"
        "than against the others. Brief comments optional and welcome --\n"
        "melody, direction, pacing, how it ends.\n\n"
        "The question is absolute: how much of this would you use in a game?\n\n"
        "All files are peak-normalised to one level, and filenames carry no\n"
        "information about scenario or seed.\n\n"
        "| file | rating | comment |\n|---|---|---|\n"
        + "".join(f"| {row['file']} |  |  |\n" for row in key)
        + "\n"
    )
    key_path = args.out.parent / f"{args.listen_dir.name}_KEY.json"
    key_path.write_text(json.dumps(key, indent=2) + "\n")

    per_scenario = Counter(r["scenario"] for r in accepted)
    fail_stage = Counter(f["stage"] for f in failures)
    fail_reason = Counter(f["reason"] for f in failures)
    implicit = sum(1 for r in accepted if r["structural"]["implicit_first_bar"])
    attempted = len(accepted) + len(failures)
    payload = {
        "reconstruction": RECONSTRUCTION_ID,
        "checkpoint": {
            "model": "haster/Fermata-0.2B-v1",
            "revision": CHECKPOINT_REVISION,
            "weights_sha256": CHECKPOINT_SHA256,
        },
        "design": {
            "bars": BARS,
            "per_scenario": args.per_scenario,
            "scenarios": list(SCENARIOS_IN_BATCH),
            "sparse": "suspended",
            "temperature": args.temperature,
            "top_k": args.top_k,
            "repetition_penalty": args.repetition_penalty,
            "max_new_tokens": args.max_new_tokens,
            "seed_base": args.seed_base,
            "peak_dbfs": PEAK_DBFS,
            "ranking_used": None,
            "representation": "rse-repr/1.0.0-d4k4-tempo1.01",
            "pipeline": "corrected timing (65609cc) + implicit first bar (d40fa80)",
        },
        "n_attempted": attempted,
        "n_accepted": len(accepted),
        "n_failed": len(failures),
        "failure_rate": round(len(failures) / max(1, attempted), 4),
        "failures_by_stage": dict(fail_stage),
        "failures_by_reason": dict(fail_reason),
        "per_scenario_accepted": dict(per_scenario),
        "n_with_implicit_first_bar": implicit,
        "accepted": accepted,
        "failures": failures,
        "listening_key": key,
    }
    args.out.write_text(json.dumps(payload, indent=2) + "\n")

    print(
        f"\n{len(accepted)} accepted of {attempted} attempted "
        f"(failure rate {payload['failure_rate']:.1%})"
    )
    print(f"per scenario: {dict(per_scenario)}")
    if fail_stage:
        print(f"failures by stage: {dict(fail_stage)}")
        print(f"failures by reason: {dict(fail_reason)}")
    print(f"pieces with a dropped opening bar, now handled: {implicit}/{len(accepted)}")
    print(
        "every accepted piece passed: 8 complete bars, declared meter, tempo "
        "preserved, no negative ticks, onsets inside the window, requested "
        "instrumentation, zero hardware violations"
    )
    print(f"\nrendered {len(key)} files to {args.listen_dir}, peak {PEAK_DBFS} dBFS")
    print(f"rate them in {manifest}")
    print(f"key in {key_path} -- do not open it until the ratings are recorded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
