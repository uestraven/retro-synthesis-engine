"""Paired listening test: v1's bar-by-bar melody against phrase structure.

One composition per pair, two melodies. Same style, seed, tempo, key,
progression and motif; the only difference is whether the melody is planned in
four-bar phrases with cadences and a breath, or restated every bar as v1 does.

Blind and shuffled per pair, because the thing being measured is a preference
and I know which arm I wrote.

Both arms isolate the melody's randomness. The first build of this test did
not, and the backing was different in 12 of 12 pairs: the melody draws once
per note, the phrased melody has fewer notes, so the drums and every later
motif shifted. It would have measured phrasing plus a different arrangement.

What v1 measured, section A of every style: one distinct attack pattern across
eight bars, zero rest, and bar endings on the tonic once or twice in eight by
accident. The phrased arm has two or three attack patterns, 6-9% rest, and a
closed cadence on the tonic at the end of the period.
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

from rse.procedural.compose import compose
from rse.song.audio import normalise_peak, render_window

#: Weighted toward the styles the listener hears most, as in the harmony
#: pilots: the melody change should show up first where the line is busiest.
PLAN = (
    ("boss", 4),
    ("action", 4),
    ("adventure", 2),
    ("town", 1),
    ("victory", 1),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("listen_phrase"))
    parser.add_argument("--seed", type=int, default=4100)
    parser.add_argument("--seconds", type=float, default=60.0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    shuffler = random.Random(args.seed)
    key: dict[str, object] = {}
    pair_index = 0

    for style, count in PLAN:
        for _offset in range(count):
            pair_index += 1
            seed = args.seed + pair_index
            arms = {}
            for arm, phrasing in (("v1", None), ("phrased", "period")):
                result = compose(
                    style_name=style,
                    seed=seed,
                    target_seconds=args.seconds,
                    phrasing=phrasing,
                    # BOTH arms, so the only difference between them is
                    # phrasing. Without it the arms' melodies consume
                    # different numbers of draws and the drums diverge too.
                    isolate_melody_rng=True,
                )
                end = result.composition.loop_end_tick
                arms[arm] = {
                    "composition": result.composition,
                    "bars": end // 1920,
                    "seconds": round(result.composition.time.seconds(end), 2),
                    "sections": len(result.sections),
                }
            # Shuffle which arm is A, so the file name carries no hint.
            order = ["v1", "phrased"]
            shuffler.shuffle(order)
            pair = f"pair_{pair_index:02d}"
            for letter, arm in zip("AB", order, strict=True):
                path = args.out_dir / f"{pair}_{letter}.wav"
                render_window(arms[arm]["composition"], path)
                normalise_peak(path)
            key[pair] = {
                "style": style,
                "seed": seed,
                "A": order[0],
                "B": order[1],
                "bars": {a: arms[a]["bars"] for a in arms},
                "seconds": {a: arms[a]["seconds"] for a in arms},
            }
            print(
                f"{pair}  {style:10} seed {seed}  "
                f"A={order[0]:7} B={order[1]:7}  "
                f"{arms['v1']['seconds']}s / {arms['phrased']['seconds']}s"
            )

    (args.out_dir.parent / f"{args.out_dir.name}_KEY.json").write_text(
        json.dumps(key, indent=2) + "\n"
    )
    print(f"\nwrote {pair_index} pairs to {args.out_dir}")
    print(f"key: {args.out_dir.name}_KEY.json (do not read it before rating)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
