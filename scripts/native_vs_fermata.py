"""Compare the native foundation with Fermata on identical musical measures.

Deliberately not a loss comparison. The two models have different vocabularies
over different alphabets, so their cross-entropies measure different tasks and
putting them side by side would be meaningless. What can be compared is the
music, through the same renderer at the same length.

The metric set here is byte-identical to the one `fermata_to_nes.py` applies to
Fermata's arranged output, computed by the same functions on the same code
path. That matters more than it sounds: a comparison assembled from two
separately written metric implementations would differ in definitions nobody
had written down.

Fermata's figures come from its results JSON rather than being recomputed, so
whatever is reported for it here is exactly what Phase 4 reported.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path
from typing import Any

from rse.codec.benchmark import enforce_monophony, ioi_entropy
from rse.conditioning import melody_line, motif_note_participation
from rse.hardware.constraints import ConstraintEngine
from rse.hardware.profile import load_profile
from rse.training.evaluate import bar_signatures, evaluate_composition, repetition_metrics

#: Metrics reported for both models, in the order the table prints them.
COLUMNS = (
    "n_events",
    "n_bars",
    "notes_per_bar",
    "pitch_class_entropy",
    "ngram_repetition_rate",
    "stepwise_fraction",
    "motif_note_participation",
    "ioi_entropy",
    "n_percussion",
    "n_channels_used",
)


def _seconds_per_tick(composition) -> float:
    end = composition.end_tick or 1
    total = composition.time.seconds(end)
    return total / end if total else 0.0


def measure(composition, *, profile, row_ticks: int) -> dict[str, Any]:
    """The shared metric set, on a decoded and hardware-valid composition."""
    events = composition.sounding_events()
    metrics = evaluate_composition(composition, profile)
    _channel, melody = melody_line(composition)
    spt = _seconds_per_tick(composition)
    repeated, unique = repetition_metrics(bar_signatures(composition, quantize_ticks=row_ticks))
    return {
        "n_events": len(events),
        "n_bars": metrics.n_bars,
        "notes_per_bar": round(metrics.notes_per_bar, 3),
        "pitch_class_entropy": round(metrics.pitch_class_entropy, 4),
        "ngram_repetition_rate": round(metrics.ngram_repetition_rate, 4),
        "stepwise_fraction": round(metrics.stepwise_fraction, 4),
        "motif_note_participation": round(motif_note_participation(melody), 4),
        "ioi_entropy": round(ioi_entropy([e.tick * spt for e in events]), 4),
        "n_percussion": sum(1 for e in events if e.channel_id == "NO"),
        "n_channels_used": len({e.channel_id for e in events}),
        # Row-tolerant, so a bar repeated a sub-row slot later still counts.
        "repeated_bar_fraction": round(repeated, 4),
        "unique_bar_fraction": round(unique, 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fermata-results", type=Path, default=Path("fermata_nes_results.json"))
    parser.add_argument("--bars", type=int, default=8)
    parser.add_argument("--temperatures", type=float, nargs="+", default=[0.7, 0.85, 1.0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--max-tokens", type=int, default=3072)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--render-dir", type=Path, default=Path("native_d4_audio"))
    parser.add_argument("--render", type=int, default=6)
    parser.add_argument("--out", type=Path, default=Path("native_vs_fermata.json"))
    args = parser.parse_args()

    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from codec_comparison import render_window

    from rse.generation.sampler import SamplingConfig, sample
    from rse.tokenizers.tokenizer import Conditioning
    from rse.training.train import load_checkpoint

    model, tokenizer, payload = load_checkpoint(args.checkpoint, device=args.device)
    profile = load_profile("nes")
    engine = ConstraintEngine(profile)
    row_ticks = tokenizer.ticks_per_step(480)
    tempo = payload.get("report", {}).get("conditioning_tempo_bpm") or 150.0
    print(
        f"checkpoint: {args.checkpoint.name}, vocab {len(tokenizer)}, "
        f"codec {tokenizer.config.codec}{tokenizer.config.sub_row_divisions}, "
        f"tempo {tempo}",
        flush=True,
    )

    rows: list[dict[str, Any]] = []
    rendered = 0
    for temperature in args.temperatures:
        for seed in args.seeds:
            result = sample(
                model,
                tokenizer,
                conditioning=Conditioning(n_bars=args.bars, tempo_bpm=tempo),
                config=SamplingConfig(
                    seed=seed,
                    temperature=temperature,
                    top_k=None,
                    top_p=None,
                    max_tokens=args.max_tokens,
                ),
            )
            playable, _repair = engine.apply(result.composition)
            final, _t = enforce_monophony(playable)
            row = {
                "model": "native_d4",
                "temperature": temperature,
                "seed": seed,
                "n_tokens": len(result.ids),
                **measure(final, profile=profile, row_ticks=row_ticks),
            }
            rows.append(row)
            print(
                f"  t={temperature} seed {seed}: events {row['n_events']:>4} "
                f"bars {row['n_bars']:>3} pcent {row['pitch_class_entropy']:.2f} "
                f"ngram {row['ngram_repetition_rate']:.3f} "
                f"perc {row['n_percussion']:>4} chan {row['n_channels_used']}",
                flush=True,
            )
            if rendered < args.render:
                render_window(final, args.render_dir / f"native_t{temperature}_s{seed}.wav")
                rendered += 1

    fermata: list[dict[str, Any]] = []
    if args.fermata_results.is_file():
        data = json.loads(args.fermata_results.read_text())
        for entry in data["generations"]:
            post = entry["post_arrangement"]
            fermata.append(
                {
                    "model": "fermata_arranged",
                    "scenario": entry["scenario"],
                    "seed": entry["seed"],
                    **{k: post.get(k) for k in COLUMNS if k != "n_channels_used"},
                    "n_channels_used": len(post["channels_used"]),
                }
            )
    else:
        print(f"note: {args.fermata_results} missing; reporting the native model alone")

    def summarize(group: list[dict[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {"n": len(group)}
        for column in COLUMNS:
            values = [g[column] for g in group if isinstance(g.get(column), (int, float))]
            if values:
                out[column] = {
                    "median": round(st.median(values), 4),
                    "min": round(min(values), 4),
                    "max": round(max(values), 4),
                }
        return out

    payload_out = {
        "checkpoint": str(args.checkpoint),
        "bars": args.bars,
        "note": (
            "Loss values are NOT compared: different vocabularies over different "
            "alphabets measure different tasks."
        ),
        "native": {"rows": rows, "summary": summarize(rows)},
        "fermata": {"rows": fermata, "summary": summarize(fermata)} if fermata else None,
    }
    args.out.write_text(json.dumps(payload_out, indent=2) + "\n")

    print(f"\n{'metric':<26}{'native median':>15}{'fermata median':>16}")
    print("-" * 57)
    native_summary = summarize(rows)
    fermata_summary = summarize(fermata) if fermata else {}
    for column in COLUMNS:
        left = native_summary.get(column, {}).get("median")
        right = fermata_summary.get(column, {}).get("median")
        print(
            f"{column:<26}"
            f"{('-' if left is None else f'{left:.3f}'):>15}"
            f"{('-' if right is None else f'{right:.3f}'):>16}"
        )
    print(f"\nwrote {args.out}; rendered {rendered} to {args.render_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
