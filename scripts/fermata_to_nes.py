"""Take Fermata's generations through the NES pipeline, and measure the cost.

For each generation: arrange for four voices, encode with the **frozen** D4
codec, validate against the hardware profile, render, and evaluate. Both the
pre-arrangement and post-arrangement forms are kept, because the point is to
separate a failure of the composer from a failure of the arrangement. A single
"after" number cannot do that.

The comparison with the native model is deliberately *not* made on loss. Two
tokenizers measuring different alphabets produce incomparable cross-entropies;
what can be compared is the music, through the same renderer at the same
length.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Importable when run as a file from a checkout where the package is not
# installed, which is the normal state of a fresh GPU host: `python
# scripts/x.py` puts `scripts/` on sys.path, not the repository root.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import json
from pathlib import Path
from typing import Any

from rse.codec.benchmark import enforce_monophony, ioi_entropy, overlapping_events
from rse.conditioning import melody_line, motif_note_participation
from rse.events.classify import classify_events
from rse.hardware.constraints import ConstraintEngine
from rse.hardware.profile import load_profile
from rse.tokenizers.frozen import frozen_config
from rse.tokenizers.tokenizer import GridFit, RseTokenizer
from rse.training.evaluate import evaluate_composition
from rse.transfer.arranger import arrange_for_nes
from rse.transfer.fermata import RECONSTRUCTION_ID, Vocab, decode

#: One tracker row, in seconds. 0.1 s is the FamiTracker default (speed 6 at
#: 60 Hz), which puts the arrangement on a grid real NES music uses.
ROW_SEC = 0.1


def _seconds_per_tick(composition) -> float:
    end = composition.end_tick or 1
    total = composition.time.seconds(end)
    return total / end if total else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", type=Path, default=Path("fermata_generations"))
    parser.add_argument("--model", type=Path, default=Path("/tmp/fermata_dl"))
    parser.add_argument("--row-sec", type=float, default=ROW_SEC)
    parser.add_argument("--render-dir", type=Path, default=Path("fermata_nes_audio"))
    parser.add_argument("--render", type=int, default=6)
    parser.add_argument("--out", type=Path, default=Path("fermata_nes_results.json"))
    args = parser.parse_args()

    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from codec_comparison import render_window

    vocab = Vocab.load(args.model / "tokenizer" / "vocab.json")
    tokenizer = RseTokenizer(frozen_config())
    profile = load_profile("nes")
    engine = ConstraintEngine(profile)
    fit = GridFit(
        step_ticks=float(tokenizer.ticks_per_step(480)),
        phase_ticks=0.0,
        aligned=True,
        period_sec=args.row_sec,
    )

    token_files = sorted(args.generations.glob("*.tokens.json"))
    if not token_files:
        print(f"error: no *.tokens.json under {args.generations}")
        return 2

    rows: list[dict[str, Any]] = []
    rendered = 0
    for path in token_files:
        payload = json.loads(path.read_text())
        ids = payload["ids"]
        stem = path.name.replace(".tokens.json", "")
        score, decode_report = decode(ids, vocab=vocab)

        # Before arrangement: what the composer actually wrote.
        pre_notes = sum(len(t.notes) for t in score.tracks)
        pre_perc = sum(len(t.percussion) for t in score.tracks)
        pre_pitches = {n.pitch for t in score.tracks for n in t.notes}

        arranged, arrange_report = arrange_for_nes(score, row_sec=args.row_sec)
        overlaps_before = len(overlapping_events(arranged))
        arranged, truncated = enforce_monophony(arranged)

        playable, repair = engine.apply(arranged)
        violations = engine.validate(playable)

        # Through the frozen codec, unchanged.
        encoded = tokenizer.encode(playable, fit=fit)
        d4 = tokenizer.decode(encoded.ids)
        final, _t = enforce_monophony(d4.composition)
        events = final.sounding_events()

        metrics = evaluate_composition(final, profile)
        _channel, melody = melody_line(final)
        spt = _seconds_per_tick(final)
        classes = (
            classify_events(final, row_ticks=max(1.0, fit.step_ticks), events=events).labels
            if events
            else []
        )

        row: dict[str, Any] = {
            "name": stem,
            "scenario": payload["scenario"]["name"],
            "seed": payload["seed"],
            "pre_arrangement": {
                "n_tracks": len(score.tracks),
                "n_notes": pre_notes,
                "n_percussion": pre_perc,
                "n_distinct_pitches": len(pre_pitches),
                "n_bars": decode_report.n_bars,
                "hit_eos": decode_report.hit_eos,
            },
            "arrangement": arrange_report.to_dict(),
            "overlaps_before_monophony": overlaps_before,
            "n_truncated_for_monophony": truncated,
            "hardware": {
                "modifications": len(repair) if isinstance(repair, list) else None,
                "violations_after_repair": len(violations),
                "compliant": not violations,
            },
            "post_arrangement": {
                "n_events": len(events),
                "n_tokens_d4": len(encoded.ids),
                "n_bars": metrics.n_bars,
                "channels_used": sorted({e.channel_id for e in events}),
                "n_percussion": sum(1 for e in events if e.channel_id == "NO"),
                "pitch_class_entropy": round(metrics.pitch_class_entropy, 4),
                "ngram_repetition_rate": round(metrics.ngram_repetition_rate, 4),
                "unique_bar_fraction": round(metrics.unique_bar_fraction, 4),
                "notes_per_bar": round(metrics.notes_per_bar, 2),
                "stepwise_fraction": round(metrics.stepwise_fraction, 4),
                "motif_note_participation": round(motif_note_participation(melody), 4),
                "ioi_entropy": round(ioi_entropy([e.tick * spt for e in events]), 4),
                "class_counts": {name: classes.count(name) for name in sorted(set(classes))},
            },
        }
        rows.append(row)
        print(
            f"{stem:<18} pre {pre_notes:>4}n/{pre_perc:>4}p -> post "
            f"{len(events):>4}ev  retention {arrange_report.retention:.3f}  "
            f"poly-drop {arrange_report.n_polyphony_dropped:>4}  "
            f"chan {''.join(c[0] for c in row['post_arrangement']['channels_used'])}  "
            f"pcent {metrics.pitch_class_entropy:.2f}  "
            f"hw {'ok' if not violations else 'BAD'}",
            flush=True,
        )

        if rendered < args.render:
            render_window(final, args.render_dir / f"{stem}_nes.wav")
            rendered += 1

    payload = {
        "reconstruction": RECONSTRUCTION_ID,
        "row_sec": args.row_sec,
        "frozen_representation": frozen_config().to_dict()["vocab_version"],
        "n_generations": len(rows),
        "generations": rows,
    }
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {args.out}; rendered {rendered} to {args.render_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
