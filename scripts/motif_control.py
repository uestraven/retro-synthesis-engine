"""Does the Motif token steer generated motif recurrence?

Gate 3's question is controllability, and validation loss cannot answer it.
The Motif token is one of ~2,300 positions, so even conditioning that steers
sampling usefully moves mean cross-entropy by thousandths. This measures the
thing the design doc actually asked for: generate with `Motif_high` and with
`Motif_low`, then measure `motif_note_participation` on the decoded output.

What makes the result interpretable is the arm comparison, not the absolute
numbers. Arm B was trained on true labels and arm C on the same labels
permuted, so:

- B separates high from low and C does not  -> the labels carry steering
  information and conditioning works;
- neither separates                          -> the labels do not;
- both separate                              -> the separation comes from
  something other than label content, and the measurement is confounded.

Arm A is the floor: its Motif token was always `unspecified` during training,
so any separation it shows is noise in the measurement, and sets the scale
against which B's separation has to be judged.

Only windows with at least `MIN_MELODY_NOTES` melodic notes are scored.
Below that the label extractor emits `unknown` rather than a recurrence
class, so a participation figure computed there describes too little music to
mean anything -- and a sample that generates almost no melody would otherwise
contribute a spurious 0.0 to whichever condition produced it.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path
from typing import Any

from rse.conditioning import MIN_MELODY_NOTES, melody_line, motif_note_participation
from rse.hardware.constraints import ConstraintEngine
from rse.hardware.profile import load_profile
from rse.tokenizers.tokenizer import Conditioning


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--motifs", nargs="+", default=["high", "medium", "low", "unspecified"])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(16)))
    parser.add_argument("--bars", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0, help="0 disables")
    parser.add_argument("--top-p", type=float, default=0.0, help="0 disables")
    parser.add_argument("--max-tokens", type=int, default=2304)
    parser.add_argument("--profile", default="nes")
    parser.add_argument("--thresholds", type=Path, default=Path("motif_thresholds_frozen.json"))
    parser.add_argument("--out", type=Path, default=Path("motif_control.json"))
    args = parser.parse_args()

    from rse.generation.sampler import SamplingConfig, sample
    from rse.training.train import load_checkpoint

    profile = load_profile(args.profile)
    engine = ConstraintEngine(profile)

    thresholds: dict[str, float] = {}
    if args.thresholds.exists():
        frozen = json.loads(args.thresholds.read_text())
        thresholds = frozen.get("primary", frozen)

    rows: list[dict[str, Any]] = []
    for checkpoint in args.checkpoints:
        model, tokenizer, _payload = load_checkpoint(checkpoint)
        for motif in args.motifs:
            for seed in args.seeds:
                result = sample(
                    model,
                    tokenizer,
                    conditioning=Conditioning(n_bars=args.bars, motif=motif),
                    config=SamplingConfig(
                        seed=seed,
                        temperature=args.temperature,
                        top_k=args.top_k or None,
                        top_p=args.top_p or None,
                        max_tokens=args.max_tokens,
                    ),
                )
                playable, _repair = engine.apply(result.composition)
                channel, pitches = melody_line(playable)
                rows.append(
                    {
                        "arm": checkpoint.stem,
                        "motif_requested": motif,
                        "seed": seed,
                        "lead_channel": channel,
                        "n_melody_notes": len(pitches),
                        "scorable": len(pitches) >= MIN_MELODY_NOTES,
                        "motif_note_participation": motif_note_participation(pitches),
                        "n_notes": len(playable.sounding_events()),
                        # The label is not a pure recurrence label: on train
                        # windows `high` carries a median 64 melodic notes
                        # against `low`'s 42, and 25% triangle leads against
                        # 13%. So a token effect could show up as density or
                        # register while recurrence stays flat, and reporting
                        # only recurrence would miss a working channel with
                        # the wrong label on it.
                        "mean_pitch": st.mean(pitches) if pitches else None,
                        "pitch_range": (max(pitches) - min(pitches)) if pitches else None,
                    }
                )

    summary: dict[str, Any] = {}
    for checkpoint in args.checkpoints:
        arm = checkpoint.stem
        per_motif: dict[str, Any] = {}
        for motif in args.motifs:
            scored = [
                r["motif_note_participation"]
                for r in rows
                if r["arm"] == arm and r["motif_requested"] == motif and r["scorable"]
            ]
            attempted = sum(1 for r in rows if r["arm"] == arm and r["motif_requested"] == motif)
            cells = [
                r
                for r in rows
                if r["arm"] == arm and r["motif_requested"] == motif and r["scorable"]
            ]
            per_motif[motif] = {
                "n_scorable": len(scored),
                "n_attempted": attempted,
                "mean": st.mean(scored) if scored else None,
                "median": st.median(scored) if scored else None,
                "stdev": st.stdev(scored) if len(scored) > 1 else None,
                # Reported beside the mean because a difference smaller than
                # the standard error is not a difference, and with 16 seeds
                # these cells are noisy.
                "stderr": (st.stdev(scored) / len(scored) ** 0.5) if len(scored) > 1 else None,
                "mean_pitch": st.mean([c["mean_pitch"] for c in cells]) if cells else None,
                "mean_melody_notes": st.mean([c["n_melody_notes"] for c in cells])
                if cells
                else None,
            }
        high, low = per_motif.get("high", {}), per_motif.get("low", {})
        separation = (
            high["mean"] - low["mean"]
            if high.get("mean") is not None and low.get("mean") is not None
            else None
        )
        # Pooled standard error of the difference, so `high - low` can be read
        # against what 16 seeds can actually resolve.
        pooled = None
        if high.get("stderr") is not None and low.get("stderr") is not None:
            pooled = (high["stderr"] ** 2 + low["stderr"] ** 2) ** 0.5
        summary[arm] = {
            "per_motif": per_motif,
            "high_minus_low": separation,
            "high_minus_low_stderr": pooled,
            "high_minus_low_in_stderrs": (
                abs(separation) / pooled if separation is not None and pooled else None
            ),
            "pitch_high_minus_low": (
                high["mean_pitch"] - low["mean_pitch"]
                if high.get("mean_pitch") is not None and low.get("mean_pitch") is not None
                else None
            ),
        }

    payload = {
        "thresholds": thresholds,
        "sampling": {
            "temperature": args.temperature,
            "top_k": args.top_k or None,
            "top_p": args.top_p or None,
            "bars": args.bars,
            "n_seeds": len(args.seeds),
        },
        "min_melody_notes": MIN_MELODY_NOTES,
        "summary": summary,
        "rows": rows,
    }
    args.out.write_text(json.dumps(payload, indent=2) + "\n")

    width = max(len(c.stem) for c in args.checkpoints)
    print(f"{'arm':<{width}}  " + "  ".join(f"{m:>14}" for m in args.motifs) + "   high-low")
    for arm, entry in summary.items():
        cells = []
        for motif in args.motifs:
            stats = entry["per_motif"][motif]
            if stats["mean"] is None:
                cells.append(f"{'n/a':>14}")
            else:
                cell = f"{stats['mean']:.3f} ({stats['n_scorable']}/{stats['n_attempted']})"
                cells.append(f"{cell:>14}")
        sep, se = entry["high_minus_low"], entry["high_minus_low_stderr"]
        if sep is None:
            tail = "   n/a"
        elif se:
            tail = f"   {sep:+.3f} +/-{se:.3f} ({abs(sep) / se:.1f} SE)"
        else:
            tail = f"   {sep:+.3f}"
        pitch = entry["pitch_high_minus_low"]
        if pitch is not None:
            tail += f"   pitch {pitch:+.1f}st"
        print(f"{arm:<{width}}  " + "  ".join(cells) + tail)
    if thresholds:
        print(
            f"\nfrozen thresholds: low<{thresholds.get('motif_low')} "
            f"high>={thresholds.get('motif_high')}"
        )
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
