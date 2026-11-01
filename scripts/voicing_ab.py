#!/usr/bin/env python3
"""Render the same SNES song twice, with and without the voicing fix.

Run this where you can actually play the result. Two WAVs come out, identical
in seed, tempo, form and length, differing only in how each chord is voiced:

    python scripts/voicing_ab.py --out listen_voicing

    listen_voicing/battle_before.wav    the harmony as it was
    listen_voicing/battle_after.wav     the harmony with the fix

## What the difference is

Handing a chord's root to the bass is correct orchestration -- the harmony
should not double it in unison. The voicer did that and then never used the
voice it had freed, so a three-voice budget produced two notes. Measured on
this exact song: 34 of 40 bars were rootless two-note dyads, and `max_voices`
had no effect at all above 2.

A rootless third and fifth has no harmonic identity. C-E-G minus the C could
be C major, A minor or F major, so the chord rested entirely on the bass note
and a listener heard padding rather than harmony.

## What to listen for

Whether the chords now have a quality you can name -- major or minor -- which
is the specific thing a rootless dyad cannot tell you. That makes it the
clearest test of whether this was the real problem.

Also whether it is now too thick. Three sustained string notes per bar under
a lead, a bass and drums is a fuller texture than two. If it sounds muddy the
lever is `harmony_voices` in rse/snes/arrange.py, not reverting this.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rse.procedural.compose import compose  # noqa: E402
from rse.snes.arrange import arrange  # noqa: E402
from rse.song.snes_generate import generate_snes, write_snes_artifacts  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="listen_voicing", help="output directory")
    parser.add_argument("--style", default="boss")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--section-bars", type=int, default=8)
    parser.add_argument("--tempo", type=int, default=160)
    parser.add_argument("--echo", default="room")
    parser.add_argument(
        "--reserved", type=int, default=0,
        help="voices held back for sound effects. 0 so the comparison is not "
             "confounded by reservation, which halves the harmony on its own",
    )
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    shared = dict(
        style=args.style, seed=args.seed, target_seconds=args.seconds,
        section_bars=args.section_bars, echo=args.echo,
        reserved_voices=args.reserved, tempo_bpm=args.tempo,
    )

    for tag, spend in (("before", False), ("after", True)):
        song = generate_snes(**shared, spend_freed_voice=spend)
        write_snes_artifacts(song, out_dir=out, name=f"battle_{tag}")

        # Measure the same arrangement so the numbers describe the audio.
        result = compose(
            style_name=args.style, seed=args.seed, target_seconds=args.seconds,
            section_bars=args.section_bars, tempo_bpm=args.tempo,
        )
        voicings = arrange(result, spend_freed_voice=spend).voicings
        sizes = Counter(len(v.pitches) for v in voicings)
        rootless = sum(
            1 for v in voicings
            if (v.intent.root % 12) not in {p % 12 for p in v.pitches}
        )
        harmony = sum(
            1 for a in song.allocation.assignments
            if a.note.role.value == "harmony"
        )
        bars = sum(s.bars for s in result.sections)

        print(f"{tag}:  battle_{tag}.wav  ({song.seconds:.2f}s)")
        print(f"   pitches per chord   {dict(sorted(sizes.items()))}")
        print(f"   rootless chords     {rootless}/{len(voicings)}")
        print(f"   harmony notes       {harmony} ({harmony / bars:.2f} per bar)")
        print(f"   voices              peak {song.allocation.peak_voices}/8, "
              f"{song.allocation.dropped} dropped")
        print("   first four chords   "
              + "  ".join(
                  f"{v.intent.symbol}{v.pitches}" for v in voicings[:4]
              ))
        print()

    print(f"Both files are in {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
