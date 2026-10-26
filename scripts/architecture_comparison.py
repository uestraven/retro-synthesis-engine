"""Blind 8-bar listening comparison: Fermata 0.2B against native RSE 100k.

Considerable effort has gone into the Fermata path on the strength of one
positive listening session and a 30-piece batch with no native arm to compare
against. This establishes whether the pretrained composer is actually better,
under conditions where the only thing differing is the composer.

What is held identical
----------------------
Both arms end as 8-bar, hardware-valid NES compositions, rendered by the same
`render_window` and then **peak-normalised to the same target**. That last
step is not cosmetic: `write_wav` clips rather than normalises, so a denser
arrangement renders louder, and loudness would have told the listener which
arm they were hearing before the first note resolved.

The asymmetries, stated rather than smoothed over
-------------------------------------------------
**Conditioning.** Fermata takes a genre/ensemble/tempo/key/track-plan header.
The native model's prefix has `Mood_`, `Energy_`, `Tempo_`, `Length_` and
`Motif_`, but Gate 3 closed as a null result for learned controllability, so
only `Tempo_` and `Length_` are treated as real controls here. Mood, energy
and motif stay at their neutral values. Tempo is drawn from the same six
values the Fermata scenarios use, so the arms share a tempo *distribution* and
one arm is not uniformly slower than the other. That is the only shared
control, and it is not scenario matching: no attempt is made to give the
native model an equivalent of "ominous" or "town", because it has none.

**Arrangement path.** Fermata composes in its own 24-TPQN grammar and must be
arranged onto four monophonic channels, so its arm runs
`arrange_for_nes` → monophony → D4 round trip → constraint engine. The native
model composes directly in the D4 representation, so its arm is
decode → constraint engine. This difference is intrinsic to the comparison —
it is part of what each architecture *is* — and cannot be removed without
making one of them a different system.

**Failure is not comparable in kind.** Fermata can decide to stop before bar
8, which the trim reports as `natural_eos` and which drops the piece. The
native sampler masks `<eos>` until the requested bar count is reached, so it
*cannot* under-run; it fails instead by filling the bars with nothing, which
shows up as empty trailing bars. Both are reported, separately, and neither is
presented as the other's equivalent.

Nothing is ranked or filtered on quality. The corpus-distance score failed its
blind validation (Kendall tau +0.17) and is not used here even as a tiebreak.
Per-piece metrics are recorded for later analysis only.
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
import wave
import zlib
from collections import Counter
from typing import Any

import numpy as np

from rse.analysis.restatement import ticks_per_bar
from rse.codec.benchmark import enforce_monophony, ioi_entropy
from rse.conditioning import melody_line, motif_note_participation
from rse.hardware.constraints import ConstraintEngine
from rse.hardware.profile import load_profile
from rse.tokenizers.frozen import frozen_config
from rse.tokenizers.tokenizer import GridFit, RseTokenizer
from rse.training.evaluate import evaluate_composition
from rse.transfer.arranger import arrange_for_nes
from rse.transfer.ensemble import check_ensemble
from rse.transfer.fermata import RECONSTRUCTION_ID, Vocab, decode, trim_to_bars

ROW_SEC = 0.1
BARS = 8

#: Peak target for every rendered file, in dBFS. Below 0 so normalisation
#: never has to clip a sample that was already near full scale.
PEAK_DBFS = -3.0

CHECKPOINT_REVISION = "18b647098f579ee41db67c0ad4c7f7f39ce0eac1"
CHECKPOINT_SHA256 = "4ca6a5db3a2d1510bad7acde4d313a9ad732fc91dcda4776f044c5b5bd8ea9b7"


def normalise_peak(path: Path, *, dbfs: float = PEAK_DBFS) -> dict[str, Any]:
    """Rewrite a mono 16-bit WAV so its peak sits at `dbfs`.

    Equalising peak rather than loudness on purpose: a true loudness match
    would alter the perceived balance between a sparse piece and a dense one,
    which is part of what is being judged. Peak matching only removes the
    gross level cue that would identify the arm.
    """
    with wave.open(str(path), "rb") as handle:
        params = handle.getparams()
        frames = handle.readframes(handle.getnframes())
    samples = np.frombuffer(frames, dtype="<i2").astype(np.float64)
    peak = float(np.abs(samples).max()) if samples.size else 0.0
    if peak <= 0:
        return {"peak_before": 0.0, "gain": 1.0, "silent": True}
    target = (10.0 ** (dbfs / 20.0)) * 32767.0
    gain = target / peak
    scaled = np.clip(samples * gain, -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setparams(params)
        handle.writeframes(scaled.tobytes())
    return {
        "peak_before": round(peak / 32767.0, 5),
        "gain": round(gain, 5),
        "silent": False,
    }


def measure(composition: Any, profile: Any, *, bars: int) -> dict[str, Any]:
    """Recorded for later analysis. Nothing here selects or ranks."""
    events = composition.sounding_events()
    metrics = evaluate_composition(composition, profile)
    _channel, melody = melody_line(composition)
    end = composition.end_tick or 1
    spt = (composition.time.seconds(end) / end) if composition.time.seconds(end) else 0.0
    bar_ticks = ticks_per_bar()
    occupied = {int(e.tick // bar_ticks) for e in events}
    trailing = 0
    for bar in range(bars - 1, -1, -1):
        if bar in occupied:
            break
        trailing += 1
    return {
        "n_events": len(events),
        "notes_per_bar": round(metrics.notes_per_bar, 3),
        "pitch_class_entropy": round(metrics.pitch_class_entropy, 4),
        "stepwise_fraction": round(metrics.stepwise_fraction, 4),
        "ngram_repetition_rate": round(metrics.ngram_repetition_rate, 4),
        "motif_note_participation": round(motif_note_participation(melody), 4),
        "ioi_entropy": round(ioi_entropy([e.tick * spt for e in events]), 4),
        "channels": sorted({e.channel_id for e in events}),
        "n_percussion": sum(1 for e in events if e.channel_id == "NO"),
        "n_bars_occupied": len(occupied),
        "empty_trailing_bars": trailing,
        "duration_sec": round(composition.time.seconds(end), 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--native", type=Path, default=Path("ckpt_foundation_d4_100k.pt")
    )
    parser.add_argument(
        "--native-secondary",
        type=Path,
        default=Path("ckpt_foundation_d4_150k.pt"),
        help=(
            "measured but NOT rendered for listening. 100k is the primary "
            "baseline because 150k developed more rhythmic saturation; this "
            "arm uses the same seeds and tempos, so 100k vs 150k is paired."
        ),
    )
    parser.add_argument("--fermata", type=Path, default=Path("/tmp/fermata_dl"))
    parser.add_argument("--per-arm", type=int, default=12)
    parser.add_argument("--native-temperature", type=float, default=0.7)
    parser.add_argument("--native-max-tokens", type=int, default=3072)
    parser.add_argument("--fermata-temperature", type=float, default=0.7)
    parser.add_argument("--fermata-top-k", type=int, default=24)
    parser.add_argument("--fermata-repetition-penalty", type=float, default=1.12)
    parser.add_argument("--fermata-max-new-tokens", type=int, default=2000)
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--seed-base", type=int, default=5000)
    parser.add_argument("--shuffle-seed", type=int, default=777)
    parser.add_argument("--listen-dir", type=Path, default=Path("listen_arch"))
    parser.add_argument("--out", type=Path, default=Path("architecture_comparison.json"))
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import torch
    from codec_comparison import render_window
    from fermata_generate import SCENARIOS, build_header
    from transformers import LlamaForCausalLM
    from transformers.generation.logits_process import LogitsProcessor

    from rse.generation.sampler import Conditioning, SamplingConfig, sample
    from rse.training.train import load_checkpoint

    profile = load_profile("nes")
    engine = ConstraintEngine(profile)
    tokenizer = RseTokenizer(frozen_config())
    audit = args.listen_dir.parent / f"{args.listen_dir.name}_audit"
    audit.mkdir(parents=True, exist_ok=True)

    # Shared tempo distribution: the only control both models implement.
    tempos = [float(s["tempo"]) for s in SCENARIOS]

    pieces: list[dict[str, Any]] = []
    compositions: dict[str, Any] = {}
    failures: list[dict[str, Any]] = []

    # ---- native arm -------------------------------------------------------
    native_model, native_tokenizer, payload = load_checkpoint(args.native, device="cpu")
    history = payload.get("history") or {}
    # `best_valid_loss` and `best_step` are scalars in `history`; the
    # per-eval curve is `valid_loss`. Reading "valid" found neither.
    best_valid = history.get("best_valid_loss")
    best_step = history.get("best_step")
    print(
        f"native: {args.native.name}, "
        f"{sum(p.numel() for p in native_model.parameters()):,} params, "
        f"best valid {best_valid} at step {best_step}",
        flush=True,
    )
    for index in range(args.per_arm):
        seed = args.seed_base + index
        tempo = tempos[index % len(tempos)]
        conditioning = Conditioning(n_bars=BARS, tempo_bpm=tempo)
        result = sample(
            native_model,
            native_tokenizer,
            conditioning=conditioning,
            config=SamplingConfig(
                seed=seed,
                temperature=args.native_temperature,
                max_tokens=args.native_max_tokens,
            ),
            loop=True,
        )
        final, repair = engine.apply(result.composition)
        if not final.sounding_events():
            failures.append(
                {"arm": "native", "seed": seed, "reason": "no_events", "tempo": tempo}
            )
            continue
        name = f"native_s{seed}"
        record = {
            "arm": "native",
            "name": name,
            "seed": seed,
            "tempo_bpm": tempo,
            "n_tokens": len(result.ids),
            "sampling": result.to_dict()["sampling"],
            "hit_token_limit": result.hit_token_limit,
            "overran_length": result.overran_length,
            "underran_length": result.underran_length,
            "n_grammar_rejections": result.n_grammar_rejections,
            "hardware_modifications": len(repair) if isinstance(repair, list) else None,
            "hardware_violations": len(engine.validate(final)),
            "metrics": measure(final, profile, bars=BARS),
        }
        pieces.append(record)
        compositions[name] = final
        (audit / f"{name}.json").write_text(json.dumps(record, indent=2) + "\n")
        print(
            f"  {name:<20} tempo {tempo:>5.0f} ev {record['metrics']['n_events']:>4} "
            f"trailing {record['metrics']['empty_trailing_bars']} "
            f"chan {''.join(c[0] for c in record['metrics']['channels'])}",
            flush=True,
        )

    # ---- native secondary: measured, never rendered -----------------------
    # The 150k run plateaued (final valid 0.57054 against 100k's 0.56938) and
    # showed worse rhythm, which is why it is not the listening baseline. It
    # is still worth measuring: the claim "150k saturates more" has only ever
    # been asserted, and the same seeds make the comparison paired rather than
    # two independent samples.
    secondary: list[dict[str, Any]] = []
    if args.native_secondary and args.native_secondary.exists():
        second_model, second_tokenizer, second_payload = load_checkpoint(
            args.native_secondary, device="cpu"
        )
        second_history = second_payload.get("history") or {}
        print(
            f"native secondary (measured only): {args.native_secondary.name}, "
            f"best valid {second_history.get('best_valid_loss')} "
            f"at step {second_history.get('best_step')}",
            flush=True,
        )
        for index in range(args.per_arm):
            seed = args.seed_base + index
            tempo = tempos[index % len(tempos)]
            result = sample(
                second_model,
                second_tokenizer,
                conditioning=Conditioning(n_bars=BARS, tempo_bpm=tempo),
                config=SamplingConfig(
                    seed=seed,
                    temperature=args.native_temperature,
                    max_tokens=args.native_max_tokens,
                ),
                loop=True,
            )
            final, _repair = engine.apply(result.composition)
            if not final.sounding_events():
                secondary.append(
                    {"seed": seed, "tempo_bpm": tempo, "reason": "no_events"}
                )
                continue
            secondary.append(
                {
                    "name": f"native150k_s{seed}",
                    "seed": seed,
                    "tempo_bpm": tempo,
                    "n_tokens": len(result.ids),
                    "overran_length": result.overran_length,
                    "underran_length": result.underran_length,
                    "hardware_violations": len(engine.validate(final)),
                    "metrics": measure(final, profile, bars=BARS),
                }
            )

    # ---- fermata arm ------------------------------------------------------
    vocab = Vocab.load(args.fermata / "tokenizer" / "vocab.json")
    fermata = LlamaForCausalLM.from_pretrained(args.fermata, dtype=torch.float32).eval()

    class SuppressBpe(LogitsProcessor):
        def __init__(self, ids: list[int]) -> None:
            self.ids = torch.tensor(ids, dtype=torch.long)

        def __call__(self, _input_ids, scores):
            scores[:, self.ids] = float("-inf")
            return scores

    class StopAfterBars(torch.nn.Module):
        def __init__(self, bar_id: int, prompt_len: int, n_bars: int) -> None:
            super().__init__()
            self.bar_id = bar_id
            self.prompt_len = prompt_len
            self.limit = n_bars + 1

        def __call__(self, input_ids, _scores, **_kwargs) -> bool:
            generated = input_ids[0, self.prompt_len :]
            return bool((generated == self.bar_id).sum() >= self.limit)

    processors = [SuppressBpe(vocab.bpe_ids)]
    fit = GridFit(
        step_ticks=float(tokenizer.ticks_per_step(480)),
        phase_ticks=0.0,
        aligned=True,
        period_sec=ROW_SEC,
    )
    print(f"fermata: {args.fermata}", flush=True)

    produced = 0
    attempt = 0
    while produced < args.per_arm and attempt < args.per_arm * args.max_attempts:
        scenario = SCENARIOS[produced % len(SCENARIOS)]
        offset = zlib.crc32(scenario["name"].encode()) % 1000
        seed = args.seed_base + 100000 + offset * 10 + attempt
        attempt += 1
        header = build_header(vocab, scenario)
        prompt = torch.tensor([[vocab.id_of(t) for t in header]])
        torch.manual_seed(seed)
        with torch.no_grad():
            out = fermata.generate(
                prompt,
                do_sample=True,
                temperature=args.fermata_temperature,
                top_k=args.fermata_top_k,
                repetition_penalty=args.fermata_repetition_penalty,
                max_new_tokens=args.fermata_max_new_tokens,
                eos_token_id=vocab.id_of("<EOS>"),
                pad_token_id=vocab.id_of("<PAD>"),
                logits_processor=processors,
                stopping_criteria=[
                    StopAfterBars(vocab.id_of("<BAR>"), prompt.shape[1], BARS)
                ],
                use_cache=True,
            )
        raw_ids = out[0].tolist()
        trimmed = trim_to_bars(raw_ids, vocab=vocab, n_bars=BARS)
        if not (trimmed.complete and trimmed.n_bars == BARS):
            failures.append(
                {
                    "arm": "fermata",
                    "seed": seed,
                    "scenario": scenario["name"],
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
        rounded = enforce_monophony(
            tokenizer.decode(tokenizer.encode(arranged, fit=fit).ids).composition
        )[0]
        final, repair = engine.apply(rounded)
        if not final.sounding_events():
            failures.append(
                {
                    "arm": "fermata",
                    "seed": seed,
                    "scenario": scenario["name"],
                    "reason": "no_events_after_arrangement",
                }
            )
            continue
        name = f"fermata_{scenario['name']}_s{seed}"
        record = {
            "arm": "fermata",
            "name": name,
            "seed": seed,
            "scenario": scenario["name"],
            "tempo_bpm": float(scenario["tempo"]),
            "n_tokens": len(trimmed.ids),
            "termination": trimmed.reason,
            "decode": decode_report.to_dict(),
            "arrangement": arrange_report.to_dict(),
            "ensemble": check_ensemble(
                scenario=scenario["name"],
                plan=scenario["plan"],
                score=score,
                arranged=final,
                n_tracks_merged=arrange_report.n_tracks_merged,
            ).to_dict(),
            "hardware_modifications": len(repair) if isinstance(repair, list) else None,
            "hardware_violations": len(engine.validate(final)),
            "metrics": measure(final, profile, bars=BARS),
        }
        pieces.append(record)
        compositions[name] = final
        (audit / f"{name}.json").write_text(json.dumps(record, indent=2) + "\n")
        print(
            f"  {name:<34} ev {record['metrics']['n_events']:>4} "
            f"trailing {record['metrics']['empty_trailing_bars']} "
            f"chan {''.join(c[0] for c in record['metrics']['channels'])} "
            f"(attempt {attempt})",
            flush=True,
        )
        produced += 1

    if not pieces:
        print("error: nothing generated", file=sys.stderr)
        return 2

    # ---- blind render -----------------------------------------------------
    args.listen_dir.mkdir(parents=True, exist_ok=True)
    slots = list(range(1, len(pieces) + 1))
    random.Random(args.shuffle_seed).shuffle(slots)
    key: list[dict[str, Any]] = []
    for slot, record in zip(slots, pieces, strict=True):
        filename = f"cmp_{slot:02d}.wav"
        destination = args.listen_dir / filename
        render_window(compositions[record["name"]], destination)
        level = normalise_peak(destination)
        record["render"] = {"file": filename, **level}
        key.append(
            {
                "file": filename,
                "arm": record["arm"],
                "name": record["name"],
                "seed": record["seed"],
                "scenario": record.get("scenario"),
            }
        )
    key.sort(key=lambda row: row["file"])

    manifest = args.listen_dir / "RATINGS.md"
    manifest.write_text(
        "# Architecture comparison -- ratings\n\n"
        "Rate every file on the same 1-6 scale, where **1 is the best and 6 is\n"
        "the worst**. Judge each on its own.\n\n"
        "Every file is eight bars, rendered by the same NES renderer and peak-\n"
        "normalised to the same level, so loudness carries no information about\n"
        "which system produced it. Filenames carry none either.\n\n"
        "| file | rating | weakness / comment |\n|---|---|---|\n"
        + "".join(f"| {row['file']} |  |  |\n" for row in key)
        + "\n"
    )
    key_path = args.out.parent / f"{args.listen_dir.name}_KEY.json"
    key_path.write_text(json.dumps(key, indent=2) + "\n")

    by_arm = Counter(r["arm"] for r in pieces)
    fail_by_arm = Counter(f["arm"] for f in failures)
    fail_reasons = Counter(f"{f['arm']}:{f['reason']}" for f in failures)
    payload_out = {
        "reconstruction": RECONSTRUCTION_ID,
        "native_checkpoint": {
            "path": str(args.native),
            "n_parameters": sum(p.numel() for p in native_model.parameters()),
            "best_valid_loss": best_valid,
            "best_step": best_step,
            "max_steps": (payload.get("train_config") or {}).get("max_steps"),
        },
        "fermata_checkpoint": {
            "model": "haster/Fermata-0.2B-v1",
            "revision": CHECKPOINT_REVISION,
            "weights_sha256": CHECKPOINT_SHA256,
        },
        "design": {
            "bars": BARS,
            "per_arm": args.per_arm,
            "peak_dbfs": PEAK_DBFS,
            "shared_controls": ["tempo", "length"],
            "native_conditioning": "mood/energy/motif neutral; Gate 3 was a null",
            "native_sampling": {
                "temperature": args.native_temperature,
                "max_tokens": args.native_max_tokens,
            },
            "fermata_sampling": {
                "temperature": args.fermata_temperature,
                "top_k": args.fermata_top_k,
                "repetition_penalty": args.fermata_repetition_penalty,
                "max_new_tokens": args.fermata_max_new_tokens,
            },
            "ranking_used": None,
            "representation": "rse-repr/1.0.0-d4k4-tempo1.01",
        },
        "n_pieces": len(pieces),
        "per_arm_produced": dict(by_arm),
        "failures_by_arm": dict(fail_by_arm),
        "failures_by_reason": dict(fail_reasons),
        "native_attempts": args.per_arm,
        "fermata_attempts": attempt,
        "pieces": pieces,
        "failures": failures,
        "listening_key": key,
        "native_secondary": {
            "path": str(args.native_secondary),
            "rendered_for_listening": False,
            "note": (
                "same seeds and tempos as the primary native arm, so the "
                "100k vs 150k comparison is paired"
            ),
            "pieces": secondary,
        },
    }
    args.out.write_text(json.dumps(payload_out, indent=2) + "\n")

    print(f"\n{len(pieces)} pieces: {dict(by_arm)}")
    print(
        f"yield -- native {by_arm['native']}/{args.per_arm} attempts; "
        f"fermata {by_arm['fermata']}/{attempt} attempts"
    )
    if fail_reasons:
        print(f"failures: {dict(fail_reasons)}")
    print(
        "note: the two failure modes are not equivalent. The native sampler "
        "masks <eos> until the target bar count, so it cannot under-run; it "
        "fails by leaving bars empty instead."
    )
    for arm in ("native", "fermata"):
        rows = [r for r in pieces if r["arm"] == arm]
        if not rows:
            continue
        trailing = [r["metrics"]["empty_trailing_bars"] for r in rows]
        print(
            f"  {arm:<8} empty trailing bars: "
            f"{sum(1 for t in trailing if t)}/{len(rows)} pieces, max {max(trailing)}"
        )
    measured = [row for row in secondary if "metrics" in row]
    if measured:
        import statistics as _st

        native_rows = [r for r in pieces if r["arm"] == "native"]
        print(f"\n150k secondary (paired on seed, not rendered), n={len(measured)}:")
        print(f"  {'metric':<24}{'100k':>10}{'150k':>10}")
        for metric in ("notes_per_bar", "n_events", "empty_trailing_bars", "ioi_entropy"):
            primary_values = [r["metrics"][metric] for r in native_rows]
            second_values = [r["metrics"][metric] for r in measured]
            if not primary_values or not second_values:
                continue
            print(
                f"  {metric:<24}{_st.median(primary_values):>10.3f}"
                f"{_st.median(second_values):>10.3f}"
            )

    print(f"\nrendered {len(key)} files to {args.listen_dir}, peak-normalised to {PEAK_DBFS} dBFS")
    print(f"key in {key_path} -- do not open it until the ratings are recorded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
