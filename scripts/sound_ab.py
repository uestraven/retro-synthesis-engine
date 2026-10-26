"""Dry NES against Expressive NES, from identical note events.

The score is frozen. Pitch, onset, duration, channel, progression and section
structure are byte-identical between arms -- asserted here, not assumed -- so
the only variables are the pulse duty cycles and the note velocities.

Control  : the renderer's default duty (P1 50%, P2 25%), uniform velocity.
Treatment: style-specific complementary duties, position-shaped velocity.

Loudness is matched by peak normalisation, with no dynamics compression: a
compressor would flatten the very thing under test.

A/B order is BALANCED, not shuffled per pair. Shuffling each pair
independently put the treatment in slot A nine times out of twelve, which
leaves an order bias in a test whose whole purpose is to remove one.

Some styles draw `hollow/classic` = (50%, 25%), which IS the dry default. Those
pairs therefore test velocity shaping alone, and the manifest records which is
which -- a natural control for separating expression from instrumentation.
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

from rse.procedural.compose import STYLES, compose
from rse.procedural.sound import DRY_DUTY
from rse.song.audio import normalise_peak, render_window


def score_of(result):
    return sorted(
        (e.channel_id, e.tick, getattr(e, "pitch", getattr(e, "value", 0)), e.duration)
        for e in result.composition.sounding_events()
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("listen_sound"))
    parser.add_argument("--seed", type=int, default=6300)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--per-style", type=int, default=2)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    key: dict[str, object] = {}
    # Exactly half the pairs lead with each arm, then the assignment order is
    # shuffled so the listener cannot predict it from the pair number.
    total = len(STYLES) * args.per_style
    assignments = [["dry", "expressive"]] * (total // 2) + [
        ["expressive", "dry"]
    ] * (total - total // 2)
    random.Random(args.seed).shuffle(assignments)
    index = 0
    for style in sorted(STYLES):
        for offset in range(args.per_style):
            index += 1
            seed = args.seed + index + offset
            dry = compose(style_name=style, seed=seed, target_seconds=args.seconds)
            wet = compose(
                style_name=style, seed=seed, target_seconds=args.seconds,
                sound_design="expressive",
            )
            # The claim this experiment rests on, checked every time.
            assert score_of(dry) == score_of(wet), f"{style}/{seed}: score moved"

            arms = {
                "dry": (dry, DRY_DUTY),
                "expressive": (wet, wet.sound.duty_by_channel),
            }
            order = assignments[index - 1]
            pair = f"pair_{index:02d}"
            for letter, arm in zip("AB", order, strict=True):
                result, duty = arms[arm]
                path = args.out_dir / f"{pair}_{letter}.wav"
                render_window(result.composition, path, duty_by_channel=duty)
                normalise_peak(path)
            timbre_changed = wet.sound.duty_by_channel != DRY_DUTY
            key[pair] = {
                "style": style,
                "seed": seed,
                "A": order[0],
                "B": order[1],
                "lead_timbre": wet.sound.lead_timbre,
                "comp_timbre": wet.sound.comp_timbre,
                "duty": wet.sound.duty_by_channel,
                "timbre_changed": timbre_changed,
                "tests": "velocity + timbre" if timbre_changed else "velocity only",
            }
            print(
                f"{pair}  {style:10} seed {seed}  A={order[0]:10} B={order[1]:10}  "
                f"{wet.sound.lead_timbre:8}/{wet.sound.comp_timbre:8}  "
                f"{key[pair]['tests']}"
            )

    (args.out_dir.parent / f"{args.out_dir.name}_KEY.json").write_text(
        json.dumps(key, indent=2) + "\n"
    )
    changed = sum(1 for v in key.values() if v["timbre_changed"])
    print(f"\n{index} pairs: {changed} test velocity + timbre, "
          f"{index - changed} test velocity alone")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
