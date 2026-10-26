"""Stage 2C: weighted blind pilot for the expanded harmonic vocabulary.

24 pairs, weighted 12 action / 8 boss / 1 each elsewhere. Action is the only
style exercising the borrowed major dominant, and boss the only one with
several new diatonic families, so an even split would spend most of the
listening budget where the musical change is smallest.

Three safeguards, all enforced before anything is rendered.

**Seeds are preselected on chord-sequence difference alone.** A pair whose two
arms resolve to the same progression tells us nothing, so those seeds are
rejected -- and the rejections are recorded, so selection cannot have been
steered by how the audio sounds. For action, the seed must additionally put
the borrowed V major in the piece.

**Arm matching is verified, not assumed.** Identical seeds do not guarantee
identical random decisions: the altered-chord branch of the melody renderer
once drew from a different range and desynchronised the percussion. Each pair
reports whether drums and melody onsets match, so a confounded pair is visible
rather than silent.

**A/B order is exactly balanced** -- treatment first in 12 pairs and second in
12, balanced within action and boss independently. The harmonic-rhythm test
had the treatment in A for 15 of 24, and nearly every decisive judgement came
from the other 9.

The b9 is left alone, per instruction. In C minor the borrowed G major
dominant admits Ab from the scale, a flat 9 against it. Every occurrence is
logged with its duration, velocity and what follows it, so the listening
comments can be checked against where it actually happens.
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
from rse.procedural.chords import resolve
from rse.procedural.compose import compose
from rse.song.loopcheck import check as loop_check
from rse.song.loopcheck import pad_to_loop_length

PEAK_DBFS = -3.0
BAR_TICKS = 1920
ALLOCATION = {
    "action": 12,
    "boss": 8,
    "adventure": 1,
    "town": 1,
    "tension": 1,
    "victory": 1,
}


def resolved(result: Any) -> list[tuple[int, ...]]:
    """Each section's progression as concrete pitch-class tuples."""
    style = result.style
    out = []
    for section in result.sections:
        out.append(
            tuple(
                tuple(
                    sorted(
                        (style.tonic + s) % 12
                        for s in resolve(
                            spec, tonic=style.tonic, mode=style.mode
                        ).semitones
                    )
                )
                for spec in section.progression
            )
        )
    return out


def has_borrowed(result: Any) -> bool:
    style = result.style
    return any(
        not resolve(spec, tonic=style.tonic, mode=style.mode).diatonic
        for section in result.sections
        for spec in section.progression
    )


def channel(result: Any, name: str) -> list[tuple[int, int, int, int]]:
    """(tick, voice, duration, velocity) for one channel."""
    return [
        (
            e.tick,
            getattr(e, "pitch", getattr(e, "value", 0)),
            e.duration,
            e.velocity,
        )
        for e in result.composition.sounding_events()
        if e.channel_id == name
    ]


def rhythm_only(events: list[tuple[int, int, int, int]]) -> list[tuple[int, int]]:
    """Onsets and durations, dropping pitch.

    The melody's PITCHES are expected to differ -- a new chord requires
    different notes. What must match is where the notes fall and how long
    they last, which is the motif's contour and rhythm surviving intact.
    """
    return [(tick, duration) for tick, _voice, duration, _velocity in events]


def flat_nine_incidents(result: Any) -> list[dict[str, Any]]:
    """Melody notes a semitone above a borrowed chord's root, with context."""
    style = result.style
    melody = sorted(
        (
            e
            for e in result.composition.sounding_events()
            if e.channel_id == "P1"
        ),
        key=lambda e: e.tick,
    )
    out: list[dict[str, Any]] = []
    offset = 0
    for section in result.sections:
        for bar in range(section.bars):
            spec = section.progression[bar % len(section.progression)]
            voicing = resolve(spec, tonic=style.tonic, mode=style.mode)
            if voicing.diatonic:
                continue
            root = (style.tonic + voicing.semitones[0]) % 12
            index = offset + bar
            low, high = index * BAR_TICKS, (index + 1) * BAR_TICKS
            for position, event in enumerate(melody):
                if not (low <= event.tick < high):
                    continue
                if (event.pitch % 12 - root) % 12 != 1:
                    continue
                following = melody[position + 1] if position + 1 < len(melody) else None
                out.append(
                    {
                        "bar": index,
                        "tick": event.tick,
                        "pitch_class": event.pitch % 12,
                        "chord_root_class": root,
                        "duration_rows": event.duration // 120,
                        "velocity": event.velocity,
                        "resolves_by": (
                            following.pitch - event.pitch if following else None
                        ),
                    }
                )
        offset += section.bars
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-seconds", type=float, default=60.0)
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--seed-tries", type=int, default=400)
    parser.add_argument("--shuffle-seed", type=int, default=777001)
    parser.add_argument("--listen-dir", type=Path, default=Path("listen_vocab"))
    parser.add_argument("--out", type=Path, default=Path("vocabulary_ab.json"))
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from architecture_comparison import normalise_peak
    from codec_comparison import render_window

    engine = ConstraintEngine(load_profile("nes"))
    selected: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    built: dict[str, Any] = {}

    for style_name, wanted in ALLOCATION.items():
        found = 0
        seed = args.seed_start
        tried = 0
        while found < wanted and tried < args.seed_tries:
            tried += 1
            seed += 1
            control = compose(
                style_name=style_name, seed=seed, target_seconds=args.target_seconds
            )
            treatment = compose(
                style_name=style_name,
                seed=seed,
                target_seconds=args.target_seconds,
                vocabulary="extra",
            )
            # Safeguard 1: the harmony must actually differ.
            if resolved(control) == resolved(treatment):
                rejected.append(
                    {"style": style_name, "seed": seed, "reason": "same_progression"}
                )
                continue
            if style_name == "action" and not has_borrowed(treatment):
                rejected.append(
                    {"style": style_name, "seed": seed, "reason": "no_borrowed_chord"}
                )
                continue

            # Safeguard 2: report arm matching rather than assume it.
            matched = {
                "drums": channel(control, "NO") == channel(treatment, "NO"),
                "melody_onsets": rhythm_only(channel(control, "P1"))
                == rhythm_only(channel(treatment, "P1")),
                "n_events_equal": len(control.composition.sounding_events())
                == len(treatment.composition.sounding_events()),
            }
            style = control.style
            selected.append(
                {
                    "style": style_name,
                    "seed": seed,
                    "tempo_bpm": style.tempo_bpm,
                    "mode": style.mode,
                    "matched": matched,
                    "control_progressions": [
                        analyse_progression(
                            style.tonic, style.mode, s.progression
                        ).numerals
                        if all(isinstance(x, int) for x in s.progression)
                        else "mixed"
                        for s in control.sections
                    ],
                    "treatment_has_borrowed": has_borrowed(treatment),
                    "flat_nine": flat_nine_incidents(treatment),
                    "control_violations": len(engine.validate(control.composition)),
                    "treatment_violations": len(
                        engine.validate(treatment.composition)
                    ),
                }
            )
            built[f"{style_name}_s{seed}/control"] = control.composition
            built[f"{style_name}_s{seed}/treatment"] = treatment.composition
            found += 1
        if found < wanted:
            print(f"  ! {style_name}: only {found}/{wanted} after {tried} seeds")

    # Safeguard 3: exact order balance, within action and boss separately.
    order_rng = random.Random(args.shuffle_seed)
    for group in ("action", "boss", "other"):
        members = [
            s
            for s in selected
            if (s["style"] == group if group != "other" else s["style"] not in ("action", "boss"))
        ]
        flags = [True] * (len(members) // 2) + [False] * (len(members) - len(members) // 2)
        order_rng.shuffle(flags)
        for member, flag in zip(members, flags, strict=True):
            member["treatment_is_a"] = flag

    args.listen_dir.mkdir(parents=True, exist_ok=True)
    slots = list(range(1, len(selected) + 1))
    random.Random(args.shuffle_seed + 1).shuffle(slots)
    key: list[dict[str, Any]] = []
    for slot, record in zip(slots, selected, strict=True):
        label = f"pair_{slot:02d}"
        sides = {
            "A": "treatment" if record["treatment_is_a"] else "control",
            "B": "control" if record["treatment_is_a"] else "treatment",
        }
        stem = f"{record['style']}_s{record['seed']}"
        for side, which in sides.items():
            composition = built[f"{stem}/{which}"]
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
            ).to_dict()["timing_continuous"]
        record["pair"] = label
        key.append(
            {
                "pair": label,
                "A": sides["A"],
                "B": sides["B"],
                "style": record["style"],
                "seed": record["seed"],
                "has_borrowed": record["treatment_has_borrowed"],
                "flat_nine_count": len(record["flat_nine"]),
            }
        )
    key.sort(key=lambda row: row["pair"])

    (args.listen_dir / "RATINGS.md").write_text(
        "# Harmonic vocabulary -- 24 matched pairs\n\n"
        "Each pair is **one composition with two chord vocabularies.** Same\n"
        "style, seed, key, tempo, motif and chord-change rate. Only the\n"
        "progressions differ. A/B is shuffled and exactly balanced.\n\n"
        "The melody has to change where a new chord requires different\n"
        "pitches -- that is expected, not a fault.\n\n"
        "Three answers per pair:\n\n"
        "1. **Would A pass as NES game music?** yes / no\n"
        "2. **Would B pass?** yes / no\n"
        "3. **Which is better?** `A`, `B`, or `=`\n\n"
        "Comments matter most here. Worth noting: whether the harmony is more\n"
        "interesting, whether tension resolves satisfyingly, and anything\n"
        "that sounds like a wrong note.\n\n"
        "| pair | A passes? | B passes? | better | comment |\n"
        "|---|---|---|---|---|\n"
        + "".join(f"| {row['pair']} |  |  |  |  |\n" for row in key)
        + "\n"
    )
    key_path = args.out.parent / f"{args.listen_dir.name}_KEY.json"
    key_path.write_text(json.dumps(key, indent=2) + "\n")

    nines = sum(len(s["flat_nine"]) for s in selected)
    args.out.write_text(
        json.dumps(
            {
                "allocation": ALLOCATION,
                "n_pairs": len(selected),
                "treatment_in_a": sum(1 for s in selected if s["treatment_is_a"]),
                "selection": {
                    "rejected": len(rejected),
                    "by_reason": dict(Counter(r["reason"] for r in rejected)),
                    "detail": rejected,
                },
                "arm_matching": {
                    "drums": sum(1 for s in selected if s["matched"]["drums"]),
                    "melody_onsets": sum(
                        1 for s in selected if s["matched"]["melody_onsets"]
                    ),
                    "event_count": sum(
                        1 for s in selected if s["matched"]["n_events_equal"]
                    ),
                },
                "flat_nine_total": nines,
                "violations": {
                    "control": sum(s["control_violations"] for s in selected),
                    "treatment": sum(s["treatment_violations"] for s in selected),
                },
                "pairs": selected,
                "listening_key": key,
            },
            indent=2,
        )
        + "\n"
    )

    print(f"\n{len(selected)} pairs: {dict(Counter(s['style'] for s in selected))}")
    print(f"treatment in A: {sum(1 for s in selected if s['treatment_is_a'])}/{len(selected)}")
    for group in ("action", "boss"):
        members = [s for s in selected if s["style"] == group]
        print(f"  {group}: treatment in A for "
              f"{sum(1 for s in members if s['treatment_is_a'])}/{len(members)}")
    print(f"seeds rejected: {len(rejected)} {dict(Counter(r['reason'] for r in rejected))}")
    print(f"arm matching -- drums "
          f"{sum(1 for s in selected if s['matched']['drums'])}/{len(selected)}, "
          f"melody onsets "
          f"{sum(1 for s in selected if s['matched']['melody_onsets'])}/{len(selected)}")
    print(f"action pairs with the borrowed V: "
          f"{sum(1 for s in selected if s['style']=='action' and s['treatment_has_borrowed'])}"
          f"/{sum(1 for s in selected if s['style']=='action')}")
    print(f"flat-9 incidents logged: {nines}")
    print(f"violations: control {sum(s['control_violations'] for s in selected)}, "
          f"treatment {sum(s['treatment_violations'] for s in selected)}")
    print(f"\nrate them in {args.listen_dir / 'RATINGS.md'}")
    print(f"key in {key_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
