"""Render a procedural batch for the same question the Fermata baseline answered.

Twenty pieces, so the result is directly comparable with the corrected
Fermata baseline's 12/20 -- 60% -- passing as NES game music. Same renderer,
same peak normalisation, same blind presentation, same question.

Nothing is selected or filtered. Seeds are fixed per style and every piece
that composes is in the batch, because a procedural generator can be sampled
cheaply and quietly keeping the good ones would make the comparison
meaningless.
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
from rse.procedural.compose import STYLES, compose
from rse.song.loopcheck import check as loop_check
from rse.song.loopcheck import pad_to_loop_length

PEAK_DBFS = -3.0
BAR_TICKS = 1920


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-style", type=int, default=4)
    parser.add_argument("--seed-base", type=int, default=800)
    parser.add_argument("--target-seconds", type=float, default=60.0)
    parser.add_argument("--shuffle-seed", type=int, default=13131)
    parser.add_argument("--listen-dir", type=Path, default=Path("listen_procedural"))
    parser.add_argument("--out", type=Path, default=Path("procedural_batch.json"))
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from architecture_comparison import normalise_peak
    from codec_comparison import render_window

    engine = ConstraintEngine(load_profile("nes"))
    pieces: list[dict[str, Any]] = []
    built: dict[str, Any] = {}

    for style_name in STYLES:
        for index in range(args.per_style):
            seed = args.seed_base + index
            result = compose(
                style_name=style_name, seed=seed, target_seconds=args.target_seconds
            )
            composition = result.composition
            violations = engine.validate(composition)
            end = composition.loop_end_tick or 0
            name = f"{style_name}_s{seed}"
            pieces.append(
                {
                    "name": name,
                    "style": style_name,
                    "seed": seed,
                    "bars": end // BAR_TICKS,
                    "duration_sec": round(composition.time.seconds(end), 3),
                    "n_events": len(composition.sounding_events()),
                    "hardware_violations": len(violations),
                    "plan": result.to_dict(),
                }
            )
            built[name] = composition
            print(
                f"  {name:<18}{result.style.tempo_bpm:>5}bpm "
                f"{end // BAR_TICKS:>3} bars {pieces[-1]['duration_sec']:>6.1f}s "
                f"{pieces[-1]['n_events']:>5} ev  {len(violations)} viol"
            )

    args.listen_dir.mkdir(parents=True, exist_ok=True)
    slots = list(range(1, len(pieces) + 1))
    random.Random(args.shuffle_seed).shuffle(slots)
    key = []
    for slot, record in zip(slots, pieces, strict=True):
        filename = f"proc_{slot:02d}.wav"
        destination = args.listen_dir / filename
        render_window(built[record["name"]], destination)
        composition = built[record["name"]]
        pad_to_loop_length(
            destination,
            loop_seconds=composition.time.seconds(composition.loop_end_tick or 0),
        )
        normalise_peak(destination, dbfs=PEAK_DBFS)
        diagnostic = loop_check(
            composition, bar_ticks=BAR_TICKS, wav_path=destination
        )
        record["render"] = {"file": filename}
        record["loop_diagnostic"] = diagnostic.to_dict()
        key.append(
            {
                "file": filename,
                "name": record["name"],
                "style": record["style"],
                "seed": record["seed"],
            }
        )
    key.sort(key=lambda row: row["file"])

    (args.listen_dir / "RATINGS.md").write_text(
        "# Procedural composer -- would this pass as NES game music?\n\n"
        f"{len(key)} pieces, about a minute each. These were written by rules,\n"
        "not by a model: a chord progression is chosen first and the melody,\n"
        "bass, accompaniment and drums are all derived from it.\n\n"
        "**One question per piece: would this pass as NES game music?**\n"
        "Write `yes` or `no`.\n\n"
        "The Fermata pipeline scores 12/20 -- 60% -- on this exact question,\n"
        "so that is what this is measured against.\n\n"
        "Optional second column if you have the patience: `BEST / GOOD / OK /\n"
        "BAD`, the stricter standard, where Fermata scores 5/20.\n\n"
        "| file | passes? yes/no | good? BEST/GOOD/OK/BAD | comment |\n"
        "|---|---|---|---|\n"
        + "".join(f"| {row['file']} |  |  |  |\n" for row in key)
        + "\nFilenames carry no information about style or seed.\n"
    )
    key_path = args.out.parent / f"{args.listen_dir.name}_KEY.json"
    key_path.write_text(json.dumps(key, indent=2) + "\n")
    args.out.write_text(
        json.dumps(
            {
                "n_pieces": len(pieces),
                "per_style": args.per_style,
                "styles": sorted(STYLES),
                "benchmark": {
                    "fermata_pass_rate": "12/20 = 60%",
                    "fermata_best_good": "5/20 = 25%",
                },
                "total_hardware_violations": sum(
                    p["hardware_violations"] for p in pieces
                ),
                "loops_timing_clean": sum(
                    1 for p in pieces if p["loop_diagnostic"]["timing_continuous"]
                ),
                "pieces": pieces,
                "listening_key": key,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\n{len(pieces)} pieces, "
          f"{sum(p['hardware_violations'] for p in pieces)} total violations, "
          f"{sum(1 for p in pieces if p['loop_diagnostic']['timing_continuous'])}"
          f"/{len(pieces)} loops timing-clean")
    print(f"durations: {Counter(round(p['duration_sec']/10)*10 for p in pieces)}")
    print(f"rate them in {args.listen_dir / 'RATINGS.md'}")
    print(f"key in {key_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
