"""Two small diagnostics in one batch, deliberately not a research cycle.

**A. Velocity in isolation** -- three songs rendered twice, both arms forced to
the same duty cycle, so the ONLY difference is per-note velocity. Verified
before rendering: identical pitches, onsets and durations, and velocity
reaching two or three distinct APU registers where the dry arm has exactly
one.

Measured budget: 1.34-2.09 dB of total range. That is near the audibility
threshold in a four-voice mix, and it is a property of the hardware plus the
need to keep the lead above the accompaniment, not of the shaping rules. Read
a null here as "2 dB is too little", not as "velocity does not work".

**B. Style identity** -- boss, town and tension, dry against expressive, with
the composition fixed. The question is whether the profiles make each style
more recognisably itself, not whether they are brighter or louder.
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
from rse.procedural.sound import DRY_DUTY
from rse.song.audio import normalise_peak, render_window

#: The velocity-only controls, recorded in the accepted test's key.
VELOCITY_PAIRS = (("boss", 6307), ("town", 6309), ("victory", 6311))

#: Three contrasting styles for the identity question, each with the profile
#: that is meant to CHARACTERISE it rather than whatever the seed drew.
#:
#: The question is whether the profiles make a style recognisable, so testing
#: a random draw answers a different and weaker question -- and `tension` drew
#: `full`, whose duty is the dry default, which would have produced an
#: identity pair with no timbre change in it at all.
#:
#: boss     -> aggressive: thin cutting lead, assertive
#: town     -> warm:       rounded lead over a glassy comp, gentle
#: tension  -> eerie:      thin lead over a hollow bed, restrained
IDENTITY_STYLES = (
    ("boss", 7101, "aggressive"),
    ("town", 7102, "warm"),
    ("tension", 7103, "eerie"),
)


def score(result):
    return sorted(
        (e.channel_id, e.tick, getattr(e, "pitch", getattr(e, "value", 0)), e.duration)
        for e in result.composition.sounding_events()
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("listen_v1"))
    parser.add_argument("--seconds", type=float, default=60.0)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    key: dict[str, object] = {}

    # -- A. velocity only, both arms at the dry duty ----------------------
    order = [["dry", "shaped"], ["shaped", "dry"], ["dry", "shaped"]]
    random.Random(7700).shuffle(order)
    for index, ((style, seed), arms) in enumerate(
        zip(VELOCITY_PAIRS, order, strict=True), start=1
    ):
        flat = compose(
            style_name=style, seed=seed, target_seconds=args.seconds,
            sound_design=None,
        )
        shaped = compose(
            style_name=style, seed=seed, target_seconds=args.seconds,
            sound_design="expressive",
        )
        # The control is enforced at RENDER time: both arms get DRY_DUTY
        # regardless of which profile the song drew. Requiring the draw to
        # coincide with the dry duty was an unnecessary constraint, and it
        # broke the moment the style pools were corrected.
        assert score(flat) == score(shaped), f"{style}/{seed}: score moved"
        by_arm = {"dry": flat, "shaped": shaped}
        name = f"vel_{index:02d}"
        for letter, arm in zip("AB", arms, strict=True):
            path = args.out_dir / f"{name}_{letter}.wav"
            # SAME duty for both arms. Velocity is the only variable.
            render_window(by_arm[arm].composition, path, duty_by_channel=DRY_DUTY)
            normalise_peak(path)
        key[name] = {
            "test": "velocity only",
            "style": style,
            "seed": seed,
            "A": arms[0],
            "B": arms[1],
            "duty": DRY_DUTY,
            "dry_lead_velocities": sorted(
                {e.velocity for e in flat.composition.sounding_events()
                 if e.channel_id == "P1"}
            ),
            "shaped_lead_velocities": sorted(
                {e.velocity for e in shaped.composition.sounding_events()
                 if e.channel_id == "P1"}
            ),
        }
        print(f"{name}  {style:8} seed {seed}  A={arms[0]:6} B={arms[1]:6}  "
              f"velocity only, duty held at {DRY_DUTY}")

    # -- B. style identity, dry against expressive ------------------------
    for style, seed, profile in IDENTITY_STYLES:
        flat = compose(
            style_name=style, seed=seed, target_seconds=args.seconds,
            sound_design=None,
        )
        wet = compose(
            style_name=style, seed=seed, target_seconds=args.seconds,
            sound_design="expressive", sound_profile=profile,
        )
        assert wet.sound.profile == profile
        assert wet.sound.duty_by_channel != DRY_DUTY, (
            f"{style}: {profile} is the dry duty, so this pair tests nothing"
        )
        assert score(flat) == score(wet)
        for arm, result, duty in (
            ("dry", flat, DRY_DUTY),
            ("expressive", wet, wet.sound.duty_by_channel),
        ):
            path = args.out_dir / f"id_{style}_{arm}.wav"
            render_window(result.composition, path, duty_by_channel=duty)
            normalise_peak(path)
        key[f"id_{style}"] = {
            "test": "style identity",
            "style": style,
            "seed": seed,
            "profile": wet.sound.profile,
            "duty": wet.sound.duty_by_channel,
            "lead_timbre": wet.sound.lead_timbre,
            "comp_timbre": wet.sound.comp_timbre,
        }
        print(f"id_{style:10} seed {seed}  profile {wet.sound.profile:11} "
              f"{wet.sound.lead_timbre}/{wet.sound.comp_timbre}")

    (args.out_dir.parent / f"{args.out_dir.name}_KEY.json").write_text(
        json.dumps(key, indent=2) + "\n"
    )
    print(f"\nwrote {len(key)} entries to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
