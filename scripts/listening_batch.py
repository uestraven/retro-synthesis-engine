"""Blind listening batch: balanced scenarios, fixed length, no quality filtering.

The question is how reliably the Fermata-to-NES pipeline produces enjoyable
complete compositions across styles -- not how good its best output can be. So
scenarios are balanced rather than weighted toward the ones that have pleased
before, and **nothing is selected or rejected on any quality measure**. The
previous corpus-distance ranking failed its blind validation (Kendall tau
+0.17, and the best-rated piece scored second from last), so it is not used
here even as a tiebreak.

Two rules that keep the result honest:

**Failures are reported, not hidden.** A piece that cannot reach eight
complete bars is recorded with its reason and a replacement is drawn from the
same scenario. The attempt count and failure rate are reported alongside the
ratings, because measuring quality only over the survivors of a silent filter
would overstate it.

**Filenames carry no information.** They are assigned by a shuffle seeded
independently of scenario, seed, generation order and every metric, and the
key is written outside the listening directory. A listening test whose
filenames leak the condition measures the labels.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Importable when run as a file from a checkout where the package is not
# installed, which is the normal state of a fresh GPU host.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import json
import random
import zlib
from collections import Counter
from typing import Any

from rse.codec.benchmark import enforce_monophony, ioi_entropy
from rse.conditioning import melody_line, motif_note_participation
from rse.hardware.constraints import ConstraintEngine
from rse.hardware.profile import load_profile
from rse.tokenizers.frozen import frozen_config
from rse.tokenizers.tokenizer import GridFit, RseTokenizer
from rse.training.evaluate import evaluate_composition
from rse.transfer.arranger import arrange_for_nes, row_seconds
from rse.transfer.ensemble import check_ensemble
from rse.transfer.fermata import RECONSTRUCTION_ID, Vocab, decode, trim_to_bars

#: One tracker row in seconds: FamiTracker's default speed 6 at 60 Hz.
ROW_SEC = 0.1

#: Pinned so the record is self-contained.
CHECKPOINT_REVISION = "18b647098f579ee41db67c0ad4c7f7f39ce0eac1"
CHECKPOINT_SHA256 = "4ca6a5db3a2d1510bad7acde4d313a9ad732fc91dcda4776f044c5b5bd8ea9b7"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("/tmp/fermata_dl"))
    parser.add_argument("--per-scenario", type=int, default=5)
    parser.add_argument("--bars", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=2000)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=24)
    parser.add_argument("--repetition-penalty", type=float, default=1.12)
    parser.add_argument("--seed-base", type=int, default=2000)
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=4,
        help="replacement draws allowed per slot before giving up on it",
    )
    parser.add_argument("--listen-dir", type=Path, default=Path("listen_batch2"))
    parser.add_argument("--out", type=Path, default=Path("listening_batch2.json"))
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import torch
    from codec_comparison import render_window
    from fermata_generate import SCENARIOS, build_header
    from transformers import LlamaForCausalLM
    from transformers.generation.logits_process import LogitsProcessor

    vocab = Vocab.load(args.model / "tokenizer" / "vocab.json")
    model = LlamaForCausalLM.from_pretrained(args.model, dtype=torch.float32).eval()

    class SuppressBpe(LogitsProcessor):
        """The model trained on the atomic stream; BPE ids are unseen."""

        def __init__(self, ids: list[int]) -> None:
            self.ids = torch.tensor(ids, dtype=torch.long)

        def __call__(self, _input_ids, scores):
            scores[:, self.ids] = float("-inf")
            return scores

    class StopAfterBars(torch.nn.Module):
        """Halt as soon as the bar after the last requested one opens.

        Without this `generate` runs to the token budget on every piece, since
        the model rarely emits `<EOS>` on its own -- which made each sample
        cost the full 2000 tokens regardless of reaching eight bars in a
        fraction of that. Trimming still happens afterwards; this only stops
        paying for tokens that will be discarded.
        """

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
    def fit_for(score) -> GridFit:
        """Row length from the score's own tempo.

        `rse.tokenizers.tokenizer` overrides the Tempo token with `fit.bpm`
        whenever the fit is aligned, so pinning `period_sec` to 0.1 forces
        every piece to 150 BPM and reintroduces the arranger timing defect
        however correctly the notes were placed.
        """
        return GridFit(
            step_ticks=float(tokenizer.ticks_per_step(480)),
            phase_ticks=0.0,
            aligned=True,
            period_sec=row_seconds(score.tempo_bpm),
        )
    audit = args.listen_dir.parent / f"{args.listen_dir.name}_audit"
    audit.mkdir(parents=True, exist_ok=True)

    accepted: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    compositions: dict[str, Any] = {}

    for scenario in SCENARIOS:
        header = build_header(vocab, scenario)
        prompt = torch.tensor([[vocab.id_of(t) for t in header]])
        wanted = args.per_scenario
        attempt = 0
        produced = 0
        while produced < wanted and attempt < wanted * args.max_attempts:
            # A stable per-scenario offset: `hash()` is randomized per
            # process, so using it would make the seeds -- and therefore the
            # whole batch -- irreproducible between runs.
            offset = zlib.crc32(scenario["name"].encode()) % 1000
            seed = args.seed_base + offset * 10 + attempt
            attempt += 1
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
                        StopAfterBars(vocab.id_of("<BAR>"), prompt.shape[1], args.bars)
                    ],
                    use_cache=True,
                )
            raw_ids = out[0].tolist()
            trimmed = trim_to_bars(raw_ids, vocab=vocab, n_bars=args.bars)
            name = f"{scenario['name']}_s{seed}"

            # Exactly the requested bars, reached deliberately. No padding.
            if not (trimmed.complete and trimmed.n_bars == args.bars):
                failures.append(
                    {
                        "name": name,
                        "scenario": scenario["name"],
                        "seed": seed,
                        "reason": trimmed.reason,
                        "n_bars": trimmed.n_bars,
                        "n_raw_tokens": len(raw_ids),
                    }
                )
                (audit / f"{name}.failed.json").write_text(
                    json.dumps({"trim": trimmed.to_dict(), "raw_ids": raw_ids}, indent=2) + "\n"
                )
                continue

            score, decode_report = decode(trimmed.ids, vocab=vocab)
            arranged, arrange_report = arrange_for_nes(
                score, row_sec=ROW_SEC, divisions=tokenizer.config.onset_divisions
            )
            arranged, _trunc = enforce_monophony(arranged)
            # The constraint engine runs LAST, after the D4 round trip.
            # Running it before let the round trip reintroduce violations: two
            # events quantized onto one tick are kept by `enforce_monophony`
            # at length 1 rather than deleted -- right for measurement, since
            # event counts stay comparable, but it hands the renderer two
            # simultaneous noise hits. 2 of 30 pieces were affected.
            rounded = enforce_monophony(
                tokenizer.decode(tokenizer.encode(arranged, fit=fit_for(score)).ids).composition
            )[0]
            final, repair = engine.apply(rounded)
            events = final.sounding_events()
            if not events:
                failures.append(
                    {
                        "name": name,
                        "scenario": scenario["name"],
                        "seed": seed,
                        "reason": "no_events_after_arrangement",
                        "n_bars": trimmed.n_bars,
                    }
                )
                continue

            ensemble = check_ensemble(
                scenario=scenario["name"],
                plan=scenario["plan"],
                score=score,
                arranged=final,
                n_tracks_merged=arrange_report.n_tracks_merged,
            )
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
                "n_bars": trimmed.n_bars,
                "trim": trimmed.to_dict(),
                "decode": decode_report.to_dict(),
                "arrangement": arrange_report.to_dict(),
                "ensemble": ensemble.to_dict(),
                "hardware_modifications": len(repair) if isinstance(repair, list) else None,
                "hardware_violations": len(engine.validate(final)),
                # Recorded for later analysis only. Nothing here selects.
                "metrics": {
                    "n_events": len(events),
                    "notes_per_bar": round(metrics.notes_per_bar, 3),
                    "pitch_class_entropy": round(metrics.pitch_class_entropy, 4),
                    "stepwise_fraction": round(metrics.stepwise_fraction, 4),
                    "ngram_repetition_rate": round(metrics.ngram_repetition_rate, 4),
                    "motif_note_participation": round(motif_note_participation(melody), 4),
                    "ioi_entropy": round(ioi_entropy([e.tick * spt for e in events]), 4),
                    "channels": sorted({e.channel_id for e in events}),
                    "n_percussion": sum(1 for e in events if e.channel_id == "NO"),
                },
            }
            accepted.append(record)
            compositions[name] = final
            # Full audit trail per piece.
            (audit / f"{name}.tokens.json").write_text(
                json.dumps(
                    {
                        "header": header,
                        "raw_ids": raw_ids,
                        "trimmed_ids": trimmed.ids,
                        "tokens": [vocab.id_to_token.get(i, "<UNK>") for i in trimmed.ids],
                    },
                    indent=2,
                )
                + "\n"
            )
            (audit / f"{name}.record.json").write_text(json.dumps(record, indent=2) + "\n")
            render_window(final, audit / f"{name}.wav")
            produced += 1
            print(
                f"  {name:<26} bars {trimmed.n_bars} ev {len(events):>4} "
                f"chan {''.join(c[0] for c in record['metrics']['channels'])} "
                f"ensemble {'ok' if ensemble.honours_request else 'MISMATCH'} "
                f"(attempt {attempt})",
                flush=True,
            )
        if produced < wanted:
            print(f"  ! {scenario['name']}: only {produced}/{wanted} after {attempt} attempts")

    if not accepted:
        print("error: nothing accepted", file=sys.stderr)
        return 2

    # Anonymous filenames, shuffled on a seed unrelated to anything measured.
    args.listen_dir.mkdir(parents=True, exist_ok=True)
    slots = list(range(1, len(accepted) + 1))
    random.Random(args.seed_base * 31 + 7).shuffle(slots)
    key: list[dict[str, Any]] = []
    for slot, record in zip(slots, accepted, strict=True):
        filename = f"piece_{slot:02d}.wav"
        render_window(compositions[record["name"]], args.listen_dir / filename)
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
        "# Listening ratings\n\n"
        "Rate each piece Best / Good / OK / Bad. Optionally note a weakness:\n"
        "repetitive melody, awkward rhythm, poor arrangement, weak ending.\n\n"
        "| file | rating | weakness / comment |\n|---|---|---|\n"
        + "".join(f"| {row['file']} |  |  |\n" for row in key)
        + "\nFilenames carry no information about scenario, seed or any metric.\n"
    )
    key_path = args.out.parent / f"{args.listen_dir.name}_KEY.json"
    key_path.write_text(json.dumps(key, indent=2) + "\n")

    per_scenario = Counter(r["scenario"] for r in accepted)
    fail_by_reason = Counter(f["reason"] for f in failures)
    mismatches = [r["name"] for r in accepted if not r["ensemble"]["honours_request"]]
    payload = {
        "reconstruction": RECONSTRUCTION_ID,
        "checkpoint": {
            "model": "haster/Fermata-0.2B-v1",
            "revision": CHECKPOINT_REVISION,
            "weights_sha256": CHECKPOINT_SHA256,
        },
        "config": {
            "bars": args.bars,
            "per_scenario": args.per_scenario,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "repetition_penalty": args.repetition_penalty,
            "max_new_tokens": args.max_new_tokens,
            "seed_base": args.seed_base,
            "row_sec": ROW_SEC,
            "representation": "rse-repr/1.0.0-d4k4-tempo1.01",
        },
        "n_attempted": len(accepted) + len(failures),
        "n_accepted": len(accepted),
        "n_failed": len(failures),
        "failure_rate": round(len(failures) / max(1, len(accepted) + len(failures)), 4),
        "failures_by_reason": dict(fail_by_reason),
        "per_scenario_accepted": dict(per_scenario),
        "ensemble_mismatches": mismatches,
        "accepted": accepted,
        "failures": failures,
    }
    args.out.write_text(json.dumps(payload, indent=2) + "\n")

    print(
        f"\n{len(accepted)} accepted of {payload['n_attempted']} attempted "
        f"(failure rate {payload['failure_rate']:.1%})"
    )
    print(f"per scenario: {dict(per_scenario)}")
    if fail_by_reason:
        print(f"failures by reason: {dict(fail_by_reason)}")
    print(f"ensemble mismatches: {len(mismatches)}" + (f" -- {mismatches}" if mismatches else ""))
    print(f"\nlisten to {args.listen_dir}/piece_*.wav and fill in {manifest}")
    print(
        f"the key is in {key_path}, outside the listening directory -- "
        f"do not open it until the ratings are recorded"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
