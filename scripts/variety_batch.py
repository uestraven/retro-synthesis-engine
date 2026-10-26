"""Six songs per arm, so sameness is audible as sameness.

A matched pair cannot answer "do these feel too similar" -- that question is
about the spread across a batch, not about one song against another. So this
writes two arms of six songs each, from the same six seeds:

    arm A: v1, as it ships
    arm B: variety=full

Arm A is expected to sound like six versions of one song, because measurably
it nearly is: across 20 seeds a style had 5-6 harmonies, 5-6 bass lines, and
exactly one form, one key and one tempo.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import json

from rse.procedural.compose import compose
from rse.song.audio import normalise_peak, render_window

#: One style per song, so the batch spans the product rather than one preset.
STYLES_IN_ORDER = ("boss", "action", "adventure", "town", "tension", "victory")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("listen_variety"))
    parser.add_argument("--seed", type=int, default=5200)
    parser.add_argument("--seconds", type=float, default=60.0)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] = {}
    for index, style in enumerate(STYLES_IN_ORDER, start=1):
        seed = args.seed + index
        for arm, variety in (("A_v1", None), ("B_variety", "full")):
            result = compose(
                style_name=style,
                seed=seed,
                target_seconds=args.seconds,
                variety=variety,
            )
            end = result.composition.loop_end_tick
            name = f"{index:02d}_{style}_{arm}"
            path = args.out_dir / f"{name}.wav"
            render_window(result.composition, path)
            normalise_peak(path)
            manifest[name] = {
                "style": style,
                "seed": seed,
                "arm": arm,
                "tempo_bpm": round(result.style.tempo_bpm, 2),
                "frames_per_row": result.style.frames_per_row,
                "tonic": result.style.tonic,
                "form": [s.label for s in result.sections],
                "section_bars": result.sections[0].bars,
                "seconds": round(result.composition.time.seconds(end), 2),
            }
            info = manifest[name]
            print(
                f"{name:26} {info['tempo_bpm']:>7.2f}BPM tonic {info['tonic']:>3} "
                f"{'-'.join(info['form']):14} {info['section_bars']:>2}bars "
                f"{info['seconds']:>5.1f}s"
            )

    (args.out_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(f"\nwrote {len(manifest)} songs to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
