#!/usr/bin/env python3
"""Step 5: memorization versus novelty, against the TRAINING population.

The question is whether the generators reproduce training material verbatim.
Measured exactly, not by similarity heuristic: tokens are ids below 256, so a
window packs into `bytes` and an n-gram either appears in training or does not.

**The control carries the result.** Real music the model never saw -- the
held-out split -- is measured by the identical procedure, so "how much does a
generated excerpt share with training" can be read against "how much does
genuine unseen SNES music share with training". Without that, any overlap
figure is unreadable: a shared 8-token run may simply be what this
representation makes of a common cadence.

The comparator is included deliberately. `SlotNGram` is an order-10 model, so
it copies up to 10-token contexts BY CONSTRUCTION. It therefore sets the scale
for "a generator that is definitely copying", and if the Transformer's figures
approach it, that is evidence of memorization rather than composition.

Matches never span track boundaries: n-grams are collected per track.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rse.snes_lm import representation as rep  # noqa: E402
from rse.snes_lm.release import Release  # noqa: E402

#: Reported in tokens; 4 tokens is one event under the stride-4 serialization.
LADDER = (4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512)


def stream(tokens: list[int], vocab: rep.Vocab) -> bytes:
    body = [t for t in tokens if t not in (vocab.PAD, vocab.BOS, vocab.EOS)]
    return bytes(body)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab", default="/tmp/snes_vocab.json")
    parser.add_argument("--release", default="~/snesmdb-releases/snesmdb-50game-rc1")
    parser.add_argument("--model-tokens",
                        default="/tmp/snes_level2_raw_aligned/model_tokens.json")
    parser.add_argument("--baseline-tokens",
                        default="/tmp/snes_level2_raw/baseline_tokens.json")
    parser.add_argument("--out", default="/tmp/snes_originality.json")
    args = parser.parse_args()

    vocab = rep.read_vocab(args.vocab)
    release = Release(args.release)

    print("  encoding training population...", flush=True)
    train = [stream(rep.encode(release.events(t), vocab).tokens, vocab)
             for t in release.tracks(split="train")]
    print(f"    {len(train)} tracks, {sum(len(s) for s in train):,} tokens",
          flush=True)

    queries: dict[str, list[bytes]] = {}
    for name, path in (("transformer", args.model_tokens),
                       ("slotngram", args.baseline_tokens)):
        rows = json.loads(Path(path).read_text())
        queries[name] = [stream(r["tokens"], vocab) for r in rows if r["valid"]]

    # The control: real music never trained on, through the identical pipeline.
    queries["held_out_real"] = [
        stream(rep.encode(release.events(t), vocab).tokens, vocab)
        for t in release.tracks(split="test")
    ]

    for name, items in queries.items():
        print(f"  {name}: {len(items)} items, "
              f"{sum(len(s) for s in items):,} tokens", flush=True)

    results: dict[str, dict] = {name: {"n_items": len(items), "coverage": {},
                                       "max_verbatim_tokens": 0}
                                for name, items in queries.items()}
    for n in LADDER:
        grams = set()
        for s in train:
            for i in range(len(s) - n + 1):
                grams.add(s[i:i + n])
        any_match = False
        line = [f"  n={n:4} ({n // 4:3} events)  training {len(grams):>9,}"]
        for name, items in queries.items():
            fractions = []
            for s in items:
                positions = len(s) - n + 1
                if positions <= 0:
                    continue
                hits = sum(1 for i in range(positions) if s[i:i + n] in grams)
                fractions.append(hits / positions)
            if not fractions:
                line.append(f"{name} n/a")
                continue
            median = st.median(fractions)
            results[name]["coverage"][str(n)] = {
                "median_fraction_matched": median,
                "max_fraction_matched": max(fractions),
                "n_items_measured": len(fractions),
            }
            if max(fractions) > 0:
                any_match = True
                results[name]["max_verbatim_tokens"] = n
            line.append(f"{name} med {100 * median:6.2f}% max {100 * max(fractions):6.2f}%")
        print("  |  ".join(line), flush=True)
        del grams
        if not any_match:
            print(f"    no population matches at n={n}; stopping ladder",
                  flush=True)
            break

    Path(args.out).write_text(json.dumps({
        "schema": "snes_lm.originality.v1",
        "ladder": list(LADDER),
        "note": "coverage = fraction of an item's n-token windows appearing "
                "verbatim in ANY training track; matches never span tracks",
        "results": results,
    }, indent=1))
    print(f"\n  written to {args.out}")
    for name, r in results.items():
        print(f"    {name:16} longest verbatim match >= {r['max_verbatim_tokens']} "
              f"tokens ({r['max_verbatim_tokens'] // 4} events)")


if __name__ == "__main__":
    main()
