"""The first SNES-style boss theme: shared composer -> arranger -> sampler.

A thin front end. The generation path lives in `rse.song.snes_generate` so
this script and the website's `console: "snes"` branch cannot drift apart
about what a SNES song is -- the same discipline the NES path follows, and for
the same reason: a silence fix once went into the library while the script
kept its own copy of the function.

SNES-INSPIRED PREVIEW. Samples are really BRR-encoded and the ARAM figures are
measured, but interpolation is a 4-point Catmull-Rom kernel plus a documented
low-pass rather than the S-DSP's own table, envelopes are floating point, and
there is no SPC export or emulator verification.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse

from rse.song.snes_generate import (
    ECHO_PRESETS,
    generate_snes,
    summary,
    write_snes_artifacts,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("listen_snes"))
    parser.add_argument("--name", default="snes_boss")
    parser.add_argument("--style", default="boss")
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--reserved", type=int, default=0)
    parser.add_argument(
        "--echo", default=None,
        help=f"preset ({', '.join(sorted(ECHO_PRESETS))}) or milliseconds. "
             "MEMORY ONLY -- does not change the rendered audio. It reserves "
             "the DSP's echo buffer against the same 64 KiB as the samples, "
             "so it shows what a reverb would cost in instrument memory; "
             "there is no echo processor",
    )
    parser.add_argument(
        "--no-brr", action="store_true",
        help="skip BRR quantisation (the earlier clean-PCM preview)",
    )
    args = parser.parse_args()

    echo: str | int | None = args.echo
    if echo is not None and echo.isdigit():
        echo = int(echo)

    song = generate_snes(
        style=args.style,
        seed=args.seed,
        target_seconds=args.seconds,
        echo=echo,
        reserved_voices=args.reserved,
        brr=not args.no_brr,
    )
    payload = write_snes_artifacts(song, out_dir=args.out_dir, name=args.name)

    print(summary(song))
    if song.warnings:
        print("\nwarnings:")
        for warning in song.warnings[:6]:
            print(f"   {warning}")
    print()
    for path in payload["files"].values():
        if path:
            print(f"wrote {path}")
    print(f"wrote {args.out_dir / (args.name + '.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
