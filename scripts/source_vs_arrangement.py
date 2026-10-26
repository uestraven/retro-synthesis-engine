"""Source-versus-arrangement audit: is the failure Fermata's or the arranger's?

Six pieces from the corrected baseline, each delivered three ways:

- `*.source.mid` — Fermata's own symbolic score, before any NES conversion,
  at its own tempo with its declared GM programs (80 Lead 1 square, 81 Lead 2,
  38 Synth Bass) and drums on the GM percussion channel.
- `*.lead.mid` — the declared lead track alone, for when a melody is hard to
  pick out of the full texture.
- `*.nes.wav` — the existing corrected-pipeline rendition, copied unchanged.

Nothing is regenerated: the scores are rebuilt from the token streams stored
in `listen_corrected_audit/`, so these are the same compositions that were
rated. No cleanup, no transposition, no quantisation.

**This is a diagnostic, not a preference test.** A GM piano and an NES pulse
channel sound nothing alike, and asking which is nicer measures the soundfont.
The questions are narrower: can a melody be followed, does the harmony hold
together, and does the piece go somewhere — asked of each version separately.

Rendering the MIDI needs a General MIDI synth, and there is none on this
machine (no fluidsynth, no timidity, no soundfont). Windows Media Player's
built-in GM synth plays all six consistently, which is exactly what is wanted
here — the same property that previously made it a hazard, when `.mid` files
got played instead of the `.wav` renders.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import json
import re
import shutil
from typing import Any

from rse.transfer.arranger import assign_roles
from rse.transfer.fermata import SUBPOS, Vocab, decode, trim_to_bars
from rse.transfer.score import MELODY, Score

#: MIDI resolution for the exported files. A multiple of Fermata's 24 TPQN, so
#: every onset lands on an exact tick rather than being rounded.
EXPORT_TPQ = 480

#: GM percussion channel, zero-indexed.
DRUM_CHANNEL = 9

#: Fermata's declared lead.
INTENDED_LEAD_PROGRAM = 80

RATING_ORDER = {"BEST": 0, "GOOD": 1, "OK": 2, "BAD": 3}


def read_ratings(path: Path) -> dict[str, tuple[str, str]]:
    out: dict[str, tuple[str, str]] = {}
    for line in path.read_text().splitlines():
        match = re.match(
            r"\|\s*(take_\d+\.wav)\s*\|\s*(BEST|GOOD|OK|BAD)\s*\|\s*(.*?)\s*\|", line
        )
        if match:
            out[match.group(1)] = (match.group(2), match.group(3))
    return out


def score_to_midi(
    score: Score, *, only_lead: bool = False, only_index: int | None = None
) -> Any:
    """Write Fermata's score as a GM MIDI file, unmodified.

    Pitches, onsets, durations, tempo and programs are the score's own. The
    only change is tick resolution, scaled by an exact integer so no onset
    moves.
    """
    import mido

    scale = EXPORT_TPQ // SUBPOS
    midi = mido.MidiFile(type=1, ticks_per_beat=EXPORT_TPQ)

    meta = mido.MidiTrack()
    meta.append(
        mido.MetaMessage("set_tempo", tempo=int(round(60_000_000 / score.tempo_bpm)), time=0)
    )
    meta.append(
        mido.MetaMessage(
            "time_signature", numerator=score.beats_per_bar, denominator=4, time=0
        )
    )
    midi.tracks.append(meta)

    lead_index = next(
        (
            i
            for i, t in enumerate(score.tracks)
            if t.program == INTENDED_LEAD_PROGRAM and not t.percussion
        ),
        None,
    )
    if lead_index is None:
        # No track declared program 80 at all. Falling back to the first
        # pitched track matches what the rest of the pipeline does, and
        # without it `only_lead` matched nothing and wrote a silent file.
        lead_index = next(
            (i for i, t in enumerate(score.tracks) if t.notes), None
        )

    channel = 0
    for index, track in enumerate(score.tracks):
        if only_index is not None and index != only_index:
            continue
        if only_lead and index != lead_index:
            continue
        is_drums = bool(track.percussion) and not track.notes
        out = mido.MidiTrack()
        use_channel = DRUM_CHANNEL if is_drums else channel
        if not is_drums:
            out.append(
                mido.Message(
                    "program_change",
                    channel=use_channel,
                    program=int(track.program or 0),
                    time=0,
                )
            )
            channel += 1
            if channel == DRUM_CHANNEL:
                channel += 1

        # Absolute (tick, type, note, velocity) then one pass to delta times,
        # so overlapping notes cannot corrupt the ordering.
        points: list[tuple[int, int, int, int]] = []
        for note in track.notes:
            start = note.tick * scale
            points.append((start, 1, note.pitch, note.velocity))
            points.append((start + max(1, note.duration * scale), 0, note.pitch, 0))
        for hit in track.percussion:
            start = hit.tick * scale
            points.append((start, 1, hit.key, hit.velocity))
            points.append((start + max(1, hit.duration * scale), 0, hit.key, 0))
        points.sort(key=lambda p: (p[0], p[1]))

        previous = 0
        for tick, kind, note, velocity in points:
            out.append(
                mido.Message(
                    "note_on" if kind else "note_off",
                    channel=use_channel,
                    note=max(0, min(127, note)),
                    # Fermata velocities are small integers; scale into MIDI
                    # range without changing their relative dynamics.
                    velocity=max(1, min(127, velocity * 8)) if kind else 0,
                    time=tick - previous,
                )
            )
            previous = tick
        if len(out) > 1:
            midi.tracks.append(out)
    return midi


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("/tmp/fermata_dl"))
    parser.add_argument("--baseline", type=Path, default=Path("corrected_baseline.json"))
    parser.add_argument("--audit", type=Path, default=Path("listen_corrected_audit"))
    parser.add_argument("--ratings", type=Path, default=Path("listen_corrected/RATINGS.md"))
    parser.add_argument("--nes-dir", type=Path, default=Path("listen_corrected"))
    parser.add_argument("--out-dir", type=Path, default=Path("audit_source_vs_nes"))
    parser.add_argument("--bars", type=int, default=8)
    parser.add_argument("--label-seed", type=int, default=606)
    parser.add_argument("--out", type=Path, default=Path("source_vs_arrangement.json"))
    args = parser.parse_args()

    baseline = json.loads(args.baseline.read_text())
    by_file = {r["render"]["file"]: r for r in baseline["accepted"]}
    ratings = read_ratings(args.ratings)
    vocab = Vocab.load(args.model / "tokenizer" / "vocab.json")

    # Two Best/Good, two OK, two Bad. Within each band, pick the extremes of
    # arranger intervention: the piece the arranger changed most and the one
    # it changed least. That makes the comparison maximally informative about
    # whether arrangement is implicated -- including take_03, which the
    # arranger left untouched and which was rated "harmonies make no sense".
    chosen: list[dict[str, Any]] = []
    for band in (("BEST", "GOOD"), ("OK",), ("BAD",)):
        pool = []
        for file, (rating, note) in ratings.items():
            if rating not in band:
                continue
            record = by_file[file]
            arrangement = record["arrangement"]
            pool.append(
                {
                    "file": file,
                    "rating": rating,
                    "note": note,
                    "record": record,
                    "arp_pct": round(
                        100
                        * arrangement["n_polyphony_arpeggiated"]
                        / max(1, arrangement["n_events_in"]),
                        2,
                    ),
                }
            )
        pool.sort(key=lambda p: p["arp_pct"])
        picks = [pool[0], pool[-1]] if len(pool) > 1 else pool[:1]
        for pick, why in zip(picks, ("least arranged", "most arranged"), strict=False):
            pick["selected_because"] = f"{'/'.join(band)}, {why} in its band"
            chosen.append(pick)

    # Neutral labels in a shuffled order. The first version of this put the
    # existing rating in the filename, which hands the listener an
    # expectation before they hear anything -- especially unhelpful here,
    # since a prediction about the two Bad pieces was already on record. The
    # mapping lives in a key outside the audit directory.
    import random

    labels = [f"piece_{chr(ord('A') + i)}" for i in range(len(chosen))]
    random.Random(args.label_seed).shuffle(labels)
    for pick, label in zip(chosen, labels, strict=True):
        pick["stem"] = label

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for pick in chosen:
        record = pick["record"]
        name = record["name"]
        stored = json.loads((args.audit / f"{name}.json").read_text())
        trimmed = trim_to_bars(stored["raw_ids"], vocab=vocab, n_bars=args.bars)
        score, decode_report = decode(trimmed.ids, vocab=vocab)
        roles = assign_roles(score)
        lead_index = next(
            (
                i
                for i, t in enumerate(score.tracks)
                if t.program == INTENDED_LEAD_PROGRAM and not t.percussion
            ),
            None,
        )

        stem = pick["stem"]
        source_path = args.out_dir / f"{stem}.source.mid"
        lead_path = args.out_dir / f"{stem}.lead.mid"
        score_to_midi(score).save(str(source_path))
        score_to_midi(score, only_lead=True).save(str(lead_path))
        # The declared lead carries under 17% of the notes in most of these
        # pieces, so the voice the arranger picked behaviourally is often the
        # one actually carrying the tune. Export both rather than assuming.
        chosen = next((i for i, r in roles.items() if r == MELODY), None)
        busiest_path = None
        if chosen is not None and chosen != lead_index:
            busiest_path = args.out_dir / f"{stem}.busiest.mid"
            score_to_midi(score, only_index=chosen).save(str(busiest_path))
        nes_source = args.nes_dir / pick["file"]
        nes_path = args.out_dir / f"{stem}.nes.wav"
        if nes_source.exists():
            shutil.copyfile(nes_source, nes_path)

        arrangement = record["arrangement"]
        rows.append(
            {
                "stem": stem,
                "name": name,
                "scenario": record["scenario"],
                "rating": pick["rating"],
                "comment": pick["note"],
                "selected_because": pick["selected_because"],
                "listening_file": pick["file"],
                "tempo_bpm": score.tempo_bpm,
                "n_source_events": sum(
                    len(t.notes) + len(t.percussion) for t in score.tracks
                ),
                "intended_lead_index": lead_index,
                "arranger_lead_index": next(
                    (i for i, r in roles.items() if r == MELODY), None
                ),
                "arranger_agrees_with_plan": lead_index is not None
                and roles.get(lead_index) == MELODY,
                "implicit_first_bar": decode_report.implicit_first_bar,
                "arrangement": arrangement,
                "arp_pct": pick["arp_pct"],
                "stolen_pct": round(
                    100
                    * arrangement["n_stolen_truncated"]
                    / max(1, arrangement["n_events_in"]),
                    2,
                ),
                "events_lost": arrangement["n_events_in"] - arrangement["n_events_out"],
                "files": {
                    "source_midi": source_path.name,
                    "lead_midi": lead_path.name,
                    "busiest_midi": busiest_path.name if busiest_path else None,
                    "nes_wav": nes_path.name if nes_source.exists() else None,
                },
            }
        )
        agrees = "plan" if rows[-1]["arranger_agrees_with_plan"] else "OVERRIDDEN"
        print(
            f"  {stem:<10}arp {pick['arp_pct']:>5.1f}%  "
            f"lead trk {lead_index} -> {agrees}",
            flush=True,
        )

    sheet = args.out_dir / "AUDIT.md"
    sheet.write_text(
        "# Source versus arrangement\n\n"
        "Six compositions, each in three forms. Same music throughout -- the\n"
        "scores are rebuilt from the stored token streams, with no cleanup,\n"
        "transposition or requantisation.\n\n"
        "- `*.source.mid` -- Fermata's own score, before NES conversion, at\n"
        "  its own tempo with its declared GM instruments.\n"
        "- `*.lead.mid` -- the declared lead track alone.\n"
        "- `*.nes.wav` -- the NES rendition you already rated.\n\n"
        "**Play the `.mid` files through Windows' built-in GM synth** (just\n"
        "open them) and the `.wav` with the player. The timbres are not\n"
        "comparable and are not meant to be.\n\n"
        "For each piece, answer the three questions **separately for the MIDI\n"
        "and the NES version** -- that contrast is the whole point:\n\n"
        "1. **Melody** -- can you follow a tune? (yes / barely / no)\n"
        "2. **Harmony** -- does it hold together? (yes / barely / no)\n"
        "3. **Direction** -- does it go somewhere? (yes / barely / no)\n\n"
        "| piece | MIDI: mel / harm / dir | NES: mel / harm / dir | notes |\n"
        "|---|---|---|---|\n"
        + "".join(f"| {row['stem']} |  |  |  |\n" for row in rows)
        + "\nLabels are neutral and shuffled. They carry no information about\n"
        "which piece this is, what it scored before, or how much the arranger\n"
        "changed it -- the mapping is held outside this directory until your\n"
        "answers are recorded.\n"
    )
    key_path = args.out.parent / f"{args.out_dir.name}_KEY.json"
    key_path.write_text(json.dumps({"pieces": rows}, indent=2) + "\n")
    # The public record withholds the prior rating and the arrangement
    # figures, so reading it cannot set an expectation either.
    private_fields = (
        "rating",
        "comment",
        "selected_because",
        "listening_file",
        "name",
    )
    public = [
        {k: v for k, v in row.items() if k not in private_fields} for row in rows
    ]
    args.out.write_text(
        json.dumps(
            {
                "baseline": str(args.baseline),
                "n_pieces": len(rows),
                "export_tpq": EXPORT_TPQ,
                "note": "scores rebuilt from stored token streams; nothing regenerated",
                "key": str(key_path),
                "pieces": public,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"private key: {key_path} -- do not open it until the answers are in")
    print(f"\nwrote {len(rows)} pieces x 3 files to {args.out_dir}")
    print(f"audit sheet: {sheet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
