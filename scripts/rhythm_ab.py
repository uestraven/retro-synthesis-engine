"""24 matched pairs: procedural v1 against harmonic-rhythm auto.

Same style, same seed, same progression, same motif. The only intended
difference is the rate at which chords change, plus the two things that
follow from it: the bass holds a pedal under `tonic_pedal`, and a melody note
longer than the chord it sits on is truncated.

That truncation is an **experimental constraint, not a theory requirement**.
A common tone held from C major into A minor moves from third to fifth with no
conflict, and a suspension that resolves is a device. It is measured per piece
rather than defended: v1 truncates nothing, and `tension` under `tonic_pedal`
loses 32 notes and 256 rows -- 16 bars of melody duration out of 24. If the
treatment loses on that style, truncation is the first suspect.

Presentation order is balanced: the treatment sits in A for half the pairs,
chosen by an independent generator so knowing one pair says nothing about the
next.
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
from collections import Counter
from typing import Any

from rse.hardware.constraints import ConstraintEngine
from rse.hardware.profile import load_profile
from rse.procedural.analysis import analyse_progression
from rse.procedural.compose import STYLES, compose
from rse.song.loopcheck import check as loop_check
from rse.song.loopcheck import pad_to_loop_length

PEAK_DBFS = -3.0
BAR_TICKS = 1920


def style_rhythms(name: str) -> tuple[str, ...]:
    return STYLES[name].rhythms


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="800,801,802,803")
    parser.add_argument("--target-seconds", type=float, default=60.0)
    parser.add_argument("--order-seed", type=int, default=24601)
    parser.add_argument("--shuffle-seed", type=int, default=31337)
    parser.add_argument("--listen-dir", type=Path, default=Path("listen_rhythm"))
    parser.add_argument("--out", type=Path, default=Path("rhythm_ab.json"))
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from architecture_comparison import normalise_peak
    from codec_comparison import render_window

    engine = ConstraintEngine(load_profile("nes"))
    seeds = [int(s) for s in args.seeds.split(",")]
    order_rng = random.Random(args.order_seed)

    pairs: list[dict[str, Any]] = []
    built: dict[str, Any] = {}
    for style_name in STYLES:
        for seed in seeds:
            control = compose(
                style_name=style_name, seed=seed, target_seconds=args.target_seconds
            )
            # `auto` includes one_per_bar, which IS the control -- it chose
            # it for 15 of 24 pairs on the first run, so those pairs were
            # byte-identical and tested nothing. The treatment therefore
            # draws from the style's alternatives only. `auto` keeps its
            # production meaning; this is an experiment-design choice.
            alternatives = [r for r in style_rhythms(style_name) if r != "one_per_bar"]
            # Round-robin rather than random, so every alternative a style
            # declares is actually heard. A random draw left truncation
            # unrepresented in all 24 pairs, which would have made the
            # truncation-versus-preference question unanswerable -- and
            # truncation only bites on tension + tonic_pedal, so it has to be
            # covered deliberately or not at all.
            picked = alternatives[seeds.index(seed) % len(alternatives)]
            treatment = compose(
                style_name=style_name,
                seed=seed,
                target_seconds=args.target_seconds,
                harmonic_rhythm=picked,
            )
            style = control.style
            analysis = analyse_progression(
                style.tonic, style.mode, control.sections[0].progression
            )
            name = f"{style_name}_s{seed}"

            def describe(result: Any) -> dict[str, Any]:
                composition = result.composition
                end = composition.loop_end_tick or 0
                melody = [
                    e
                    for e in composition.sounding_events()
                    if e.channel_id == "P1"
                ]
                return {
                    "rhythm": result.rhythm.name,
                    "pedal": result.rhythm.pedal,
                    "chord_changes_per_section": result.variations[0][
                        "n_chord_changes"
                    ],
                    "bars": end // BAR_TICKS,
                    "duration_sec": round(composition.time.seconds(end), 3),
                    "n_events": len(composition.sounding_events()),
                    "n_melody_notes": len(melody),
                    "notes_truncated": result.truncation["notes_truncated"],
                    "rows_removed": result.truncation["rows_removed"],
                    "hardware_violations": len(engine.validate(composition)),
                }

            # Treatment in A for half the pairs, drawn independently.
            treatment_is_a = order_rng.random() < 0.5
            pairs.append(
                {
                    "name": name,
                    "style": style_name,
                    "seed": seed,
                    "tempo_bpm": style.tempo_bpm,
                    "mode": style.mode,
                    "key": analysis.tonic,
                    "progression_numerals": analysis.numerals,
                    "progression_chords": analysis.chord_names,
                    "motif": control.motif.to_dict(),
                    "form": [s.label for s in control.sections],
                    "treatment_is_a": treatment_is_a,
                    "control": describe(control),
                    "treatment": describe(treatment),
                }
            )
            built[f"{name}/control"] = control.composition
            built[f"{name}/treatment"] = treatment.composition
            print(
                f"  {name:<18}{style.tempo_bpm:>4}bpm  "
                f"control one_per_bar -> treatment {treatment.rhythm.name:<13}"
                f"trunc {treatment.truncation['notes_truncated']:>3} notes / "
                f"{treatment.truncation['rows_removed']:>4} rows"
            )

    args.listen_dir.mkdir(parents=True, exist_ok=True)
    slots = list(range(1, len(pairs) + 1))
    random.Random(args.shuffle_seed).shuffle(slots)
    key: list[dict[str, Any]] = []
    for slot, record in zip(slots, pairs, strict=True):
        label = f"pair_{slot:02d}"
        sides = {
            "A": "treatment" if record["treatment_is_a"] else "control",
            "B": "control" if record["treatment_is_a"] else "treatment",
        }
        for side, which in sides.items():
            composition = built[f"{record['name']}/{which}"]
            destination = args.listen_dir / f"{label}_{side}.wav"
            render_window(composition, destination)
            pad_to_loop_length(
                destination,
                loop_seconds=composition.time.seconds(
                    composition.loop_end_tick or 0
                ),
            )
            normalise_peak(destination, dbfs=PEAK_DBFS)
            record.setdefault("loop", {})[which] = loop_check(
                composition, bar_ticks=BAR_TICKS, wav_path=destination
            ).to_dict()
        record["pair"] = label
        key.append(
            {
                "pair": label,
                "A": sides["A"],
                "B": sides["B"],
                "name": record["name"],
                "style": record["style"],
                "seed": record["seed"],
                "treatment_rhythm": record["treatment"]["rhythm"],
            }
        )
    key.sort(key=lambda row: row["pair"])

    (args.listen_dir / "RATINGS.md").write_text(
        "# Harmonic rhythm -- 24 matched pairs\n\n"
        "Each pair is **one composition, two harmonic pacings.** Same style,\n"
        "seed, key, tempo, progression and motif. One version changes chords\n"
        "once per bar; the other changes them at some other rate. Nothing\n"
        "tells you which is which, and A/B is shuffled per pair.\n\n"
        "The original version of these pieces is the batch you called\n"
        "\"fantastic\", so matching that is not an improvement -- the new\n"
        "pacing has to earn a preference.\n\n"
        "Three answers per pair:\n\n"
        "1. **Would A pass as NES game music?** yes / no\n"
        "2. **Would B pass?** yes / no\n"
        "3. **Which is better?** `A`, `B`, or `=`\n\n"
        "Comments are the most useful column. Worth noting especially:\n"
        "pacing, whether the melody sounds cut short, harmonic interest,\n"
        "and abrupt phrase endings.\n\n"
        "| pair | A passes? | B passes? | better | comment |\n"
        "|---|---|---|---|---|\n"
        + "".join(f"| {row['pair']} |  |  |  |  |\n" for row in key)
        + "\n"
    )
    key_path = args.out.parent / f"{args.listen_dir.name}_KEY.json"
    key_path.write_text(json.dumps(key, indent=2) + "\n")

    chosen = Counter(p["treatment"]["rhythm"] for p in pairs)
    trunc = [p["treatment"]["notes_truncated"] for p in pairs]
    args.out.write_text(
        json.dumps(
            {
                "n_pairs": len(pairs),
                "seeds": seeds,
                "treatment_in_a": sum(1 for p in pairs if p["treatment_is_a"]),
                "rhythms_chosen": dict(chosen),
                "truncation": {
                    "pairs_with_any": sum(1 for t in trunc if t),
                    "total_notes": sum(trunc),
                    "total_rows": sum(p["treatment"]["rows_removed"] for p in pairs),
                },
                "control_violations": sum(
                    p["control"]["hardware_violations"] for p in pairs
                ),
                "treatment_violations": sum(
                    p["treatment"]["hardware_violations"] for p in pairs
                ),
                "pairs": pairs,
                "listening_key": key,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\n{len(pairs)} pairs; treatment in A for "
          f"{sum(1 for p in pairs if p['treatment_is_a'])}/{len(pairs)}")
    print(f"rhythms auto chose: {dict(chosen)}")
    print(f"truncation: {sum(1 for t in trunc if t)}/{len(pairs)} pairs affected, "
          f"{sum(trunc)} notes, "
          f"{sum(p['treatment']['rows_removed'] for p in pairs)} rows total")
    print(f"violations: control {sum(p['control']['hardware_violations'] for p in pairs)}, "
          f"treatment {sum(p['treatment']['hardware_violations'] for p in pairs)}")
    print(f"\nrate them in {args.listen_dir / 'RATINGS.md'}")
    print(f"key in {key_path} -- do not open it until the answers are recorded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
