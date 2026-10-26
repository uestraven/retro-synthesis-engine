"""Blind A/B pairs: historical timing against corrected, same twelve scores.

Each pair holds one composition arranged two ways. Nothing about the music
differs — same model, same seed, same symbolic score, same arranger roles,
same hardware repair. Only the time conversion differs.

Within each pair the two files are labelled A and B in a randomised order, so
knowing that one pair put the correction in A says nothing about the next.
Pair order is randomised too, and the mapping is written outside the listening
directory.

**This cannot fully conceal the intervention, and does not claim to.** The
corrected version of a 76 BPM piece lasts three times as long as the
historical one, which is audible immediately. The question being asked is
which version the listener prefers, not whether they can tell them apart. The
ratings sheet says so explicitly, so the judgement is made with that known
rather than against a pretence.

Loudness is equalised: both members of every pair are peak-normalised to the
same target, as the architecture comparison's files were. Level is the one cue
that would be an artefact of rendering rather than of the music.
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
import shutil
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report", type=Path, default=Path("timing_fix_report.json")
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=Path("listen_timing_pairs_audit"),
        help="where timing_fix_report.py wrote the per-piece renders",
    )
    parser.add_argument("--listen-dir", type=Path, default=Path("listen_timing"))
    parser.add_argument("--shuffle-seed", type=int, default=31337)
    parser.add_argument(
        "--require-fidelity",
        action="store_true",
        default=True,
        help="refuse to build pairs unless the historical branch reproduced "
        "the archived audio exactly",
    )
    parser.add_argument(
        "--allow-fidelity-mismatch",
        dest="require_fidelity",
        action="store_false",
        help="build pairs anyway, recording the mismatch in the key",
    )
    parser.add_argument("--out", type=Path, default=Path("listen_timing_manifest.json"))
    args = parser.parse_args()

    report = json.loads(args.report.read_text())
    pieces = report["pieces"]
    exact = report.get("n_reproduced_archived_audio_exactly")
    if args.require_fidelity and exact != len(pieces):
        differing = [
            p["name"] for p in pieces if not p["fidelity"].get("bytes_identical")
        ]
        print(
            f"error: the historical branch reproduced only {exact}/{len(pieces)} "
            f"of the archived files exactly: {differing}\n"
            "A pair built on an unfaithful historical branch answers a "
            "different question than the one asked. Investigate, or pass "
            "--allow-fidelity-mismatch to proceed deliberately.",
            file=sys.stderr,
        )
        return 2

    rng = random.Random(args.shuffle_seed)
    order = list(range(len(pieces)))
    rng.shuffle(order)

    args.listen_dir.mkdir(parents=True, exist_ok=True)
    key: list[dict[str, Any]] = []
    for slot, index in enumerate(order, 1):
        piece = pieces[index]
        # Independent coin per pair: a listener who works out one pair learns
        # nothing about the rest.
        corrected_is_a = rng.random() < 0.5
        sides = {
            "A": "corrected" if corrected_is_a else "historical",
            "B": "historical" if corrected_is_a else "corrected",
        }
        entry: dict[str, Any] = {
            "pair": f"pair_{slot:02d}",
            "name": piece["name"],
            "scenario": piece["scenario"],
            "seed": piece["seed"],
            "intended_bpm": piece["intended_bpm"],
            "A": sides["A"],
            "B": sides["B"],
            "historical_encoded_bpm": piece["historical"]["encoded_bpm"],
            "corrected_encoded_bpm": piece["corrected"]["encoded_bpm"],
            "historical_bars": piece["historical"]["bars_spanned"],
            "corrected_bars": piece["corrected"]["bars_spanned"],
            "historical_lead_half_ratio": piece["historical"]["lead_half_ratio"],
            "corrected_lead_half_ratio": piece["corrected"]["lead_half_ratio"],
            "reproduced_archived_exactly": piece["fidelity"].get("bytes_identical"),
        }
        for side, which in sides.items():
            source = args.audit / f"{piece['name']}.{which}.wav"
            if not source.exists():
                print(f"error: missing {source}", file=sys.stderr)
                return 2
            shutil.copyfile(source, args.listen_dir / f"pair_{slot:02d}_{side}.wav")
        key.append(entry)

    manifest = args.listen_dir / "RATINGS.md"
    manifest.write_text(
        "# Timing correction -- A/B listening\n\n"
        "Twelve pairs. Each pair is **one composition, arranged two ways**.\n"
        "The model, the seed, the notes, the channel assignment and the\n"
        "hardware repair are identical within a pair. Only the conversion\n"
        "from the composer's timeline to the NES timeline differs.\n\n"
        "For each pair, play `A` then `B` and record **A**, **B**, or **=**\n"
        "for no meaningful preference. Comments on pacing, melody, rhythm\n"
        "and how the piece ends are the most useful part.\n\n"
        "Two things stated up front rather than hidden:\n\n"
        "- **You will often hear which is which.** One version of a slow\n"
        "  piece can last three times as long as the other. This test asks\n"
        "  which you prefer, not whether you can tell them apart.\n"
        "- **A and B are shuffled independently per pair.** Working out one\n"
        "  pair tells you nothing about the next.\n\n"
        "Both files in a pair are peak-normalised to the same level, so\n"
        "loudness carries no information.\n\n"
        "| pair | preference (A / B / =) | comment |\n|---|---|---|\n"
        + "".join(f"| pair_{slot:02d} |  |  |\n" for slot in range(1, len(key) + 1))
        + "\nFilenames carry no information about scenario, seed or treatment.\n"
    )
    key_path = args.out.parent / f"{args.listen_dir.name}_KEY.json"
    key_path.write_text(json.dumps(key, indent=2) + "\n")
    args.out.write_text(
        json.dumps(
            {
                "report": str(args.report),
                "n_pairs": len(key),
                "shuffle_seed": args.shuffle_seed,
                "fidelity_enforced": args.require_fidelity,
                "n_reproduced_archived_audio_exactly": exact,
                "listen_dir": str(args.listen_dir),
                "pairs": [
                    {k: v for k, v in entry.items() if k not in ("A", "B")}
                    for entry in key
                ],
            },
            indent=2,
        )
        + "\n"
    )

    corrected_in_a = sum(1 for entry in key if entry["A"] == "corrected")
    print(f"built {len(key)} pairs in {args.listen_dir}")
    print(f"corrected sits in A for {corrected_in_a} of {len(key)} pairs")
    print(f"ratings sheet: {manifest}")
    print(f"key: {key_path} -- do not open it until the judgements are recorded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
