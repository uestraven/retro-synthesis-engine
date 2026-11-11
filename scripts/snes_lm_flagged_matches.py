#!/usr/bin/env python3
"""Review of flagged verbatim matches, as the registered criterion requires.

The Originality criterion is "no identified near-exact copying of substantial
training passages, judged by REVIEW of flagged matches against a corpus
control". The coverage ladder says how much overlap exists; this says what the
overlapping material actually is and which training track it came from.

Inverted index: the generated excerpts are tiny (~60k tokens) and the training
corpus is large (~10M), so the excerpts' n-grams are indexed and the corpus is
scanned once. Searching each excerpt against all 1,384 tracks instead is
quadratic and does not finish.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rse.snes_lm import representation as rep  # noqa: E402
from rse.snes_lm.release import Release  # noqa: E402


def body(tokens, vocab) -> bytes:
    return bytes(t for t in tokens
                 if t not in (vocab.PAD, vocab.BOS, vocab.EOS))


def track_name(track) -> str:
    for attribute in ("name", "track_id", "path", "title"):
        value = getattr(track, attribute, None)
        if value:
            return str(value)
    return str(track)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab", default="/tmp/snes_vocab.json")
    parser.add_argument("--release", default="~/snesmdb-releases/snesmdb-50game-rc1")
    parser.add_argument("--tokens",
                        default="/tmp/snes_level2_raw_aligned/model_tokens.json")
    parser.add_argument("--out", default="/tmp/snes_flagged_matches.json")
    #: Longest window with any match in the coverage ladder was 16 tokens.
    parser.add_argument("--windows", default="24,20,16")
    args = parser.parse_args()

    vocab = rep.read_vocab(args.vocab)
    release = Release(args.release)
    rows = [r for r in json.loads(Path(args.tokens).read_text()) if r["valid"]]
    excerpts = {r["seed"]: body(r["tokens"], vocab) for r in rows}
    print(f"  {len(excerpts)} valid excerpts, "
          f"{sum(len(s) for s in excerpts.values()):,} tokens", flush=True)

    train = release.tracks(split="train")
    report: dict[str, list] = {}
    for n in [int(x) for x in args.windows.split(",")]:
        index: dict[bytes, list[int]] = {}
        for seed, stream in excerpts.items():
            for i in range(len(stream) - n + 1):
                index.setdefault(stream[i:i + n], []).append(seed)
        hits: dict[tuple[int, bytes], set[str]] = {}
        for track in train:
            name = track_name(track)
            stream = body(rep.encode(release.events(track), vocab).tokens, vocab)
            for i in range(len(stream) - n + 1):
                fragment = stream[i:i + n]
                seeds = index.get(fragment)
                if seeds:
                    for seed in seeds:
                        hits.setdefault((seed, fragment), set()).add(name)
        print(f"  n={n} ({n // 4} events): {len(hits)} distinct flagged matches",
              flush=True)
        entries = []
        for (seed, fragment), tracks in sorted(hits.items(), key=lambda k: k[0][0]):
            events = rep.decode([vocab.BOS] + list(fragment) + [vocab.EOS], vocab)
            entries.append({
                "seed": seed,
                "n_tokens": n,
                "n_events": n // 4,
                "n_training_tracks_containing": len(tracks),
                "example_source_track": sorted(tracks)[0],
                "events": [
                    {"voice": e["voice"],
                     "onset_sec": float(e["onset_sec"]),
                     "duration_sec": float(e["duration_sec"])
                     if e.get("duration_sec") is not None else None,
                     "midi_note": e.get("midi_note")}
                    for e in events
                ],
            })
        report[str(n)] = entries
        for entry in entries[:3]:
            print(f"    seed {entry['seed']} <- {entry['example_source_track']} "
                  f"(in {entry['n_training_tracks_containing']} track(s))",
                  flush=True)
            for e in entry["events"]:
                print(f"       v{e['voice']} +{e['onset_sec']:.3f}s "
                      f"dur {e['duration_sec']} note {e['midi_note']}", flush=True)
        if entries:
            break

    Path(args.out).write_text(json.dumps({
        "schema": "snes_lm.flagged_matches.v1",
        "note": "verbatim n-gram matches between generated excerpts and "
                "training tracks, for the registered originality review",
        "matches": report,
    }, indent=1))
    print(f"\n  written to {args.out}")


if __name__ == "__main__":
    main()
