#!/usr/bin/env python3
"""Select sampling parameters by the declared distributional criterion.

Registered procedure: choose, for each arm independently, the sampling
parameters whose generated output best matches the TRAINING corpus on three
statistics — attacks per second, mean simultaneous voices, and the melodic
interval distribution — scored as the sum of absolute standardised
differences (`generate.distance`).

**No audio is heard and no test data is touched.** Development seeds are
disjoint from the 24 registered generation seeds, so the selection cannot be
contaminated by the batch it will produce.

The two arms may land on different numerical settings. That is acceptable and
was registered as such: the selection PROCEDURE is symmetric — same grid, same
budget, same seeds, same objective — even though the models need different
temperatures to reach comparable distributional behaviour.

Tie-breaking, declared in advance: lowest temperature, then highest top_p,
then lowest seed order. Ties are broken deterministically so the choice does
not depend on dictionary ordering.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rse.snes_lm import baselines as bl  # noqa: E402
from rse.snes_lm import generate as gen  # noqa: E402
from rse.snes_lm import representation as rep  # noqa: E402
from rse.snes_lm.release import Release  # noqa: E402
from rse.snes_lm.train import Config, Decoder  # noqa: E402

#: The declared grid. Fixed before any excerpt was generated.
TEMPERATURES = (0.8, 0.9, 1.0, 1.1, 1.2)
TOP_PS = (1.0, 0.95)

#: Development seeds, disjoint from the registered generation seeds 0..23.
DEV_SEEDS = (1000, 1001, 1002, 1003)
DEV_SECONDS = 25.0


def reference_statistics(release: Release, vocab: rep.Vocab, n_tracks: int):
    """The corpus target, from TRAIN only, through the same tokenizer.

    Round-tripped rather than taken as published: the generated arms are
    decoded token streams, so the reference must be too, or the comparison
    would charge the arms for the tokenizer's own quantisation.
    """
    tracks = release.tracks(split="train")
    random.Random(0).shuffle(tracks)
    per_track = [
        rep.decode(rep.encode(release.events(track), vocab).tokens, vocab)
        for track in tracks[:n_tracks]
    ]
    return gen.statistics(per_track)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("tmp/best.pt"))
    parser.add_argument("--vocab", default="/tmp/snes_vocab.json")
    parser.add_argument("--release", default="~/snesmdb-releases/snesmdb-50game-rc1")
    parser.add_argument("--ref-tracks", type=int, default=60)
    parser.add_argument("--out", default="/tmp/snes_sampling_selection.json")
    args = parser.parse_args()

    vocab = rep.read_vocab(args.vocab)
    release = Release(args.release)
    reference = reference_statistics(release, vocab, args.ref_tracks)
    print("  reference (train, round-tripped):")
    print(f"    {reference.attacks_per_second:.2f} attacks/s, "
          f"{reference.mean_simultaneous_voices:.3f} voices/onset, "
          f"{len(reference.interval_histogram)} interval bins", flush=True)

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = Config(**{**state["config"], "device": "cpu"})
    model = Decoder(vocab.size, cfg)
    model.load_state_dict(state["model"])
    model.eval()

    train = [rep.encode(release.events(t), vocab).tokens
             for t in release.tracks(split="train")]
    comparator = bl.SlotNGram(vocab, order=10, discount=64.0,
                              use_event_prefix=False).fit(train)

    results: dict[str, list[dict]] = {"model": [], "baseline": []}
    for arm in ("baseline", "model"):
        print(f"\n  === {arm} ===", flush=True)
        for temperature in TEMPERATURES:
            for top_p in TOP_PS:
                started = time.time()
                per_excerpt: list[list[dict]] = []
                valid = 0
                for seed in DEV_SEEDS:
                    config = gen.SamplingConfig(
                        temperature=temperature, top_p=top_p, seed=seed
                    )
                    excerpt = (
                        gen.generate_model(model, vocab, config, seconds=DEV_SECONDS)
                        if arm == "model"
                        else gen.generate_baseline(
                            comparator, vocab, config, seconds=DEV_SECONDS
                        )
                    )
                    valid += bool(excerpt.valid)
                    per_excerpt.append(excerpt.events)
                statistics = gen.statistics(per_excerpt)
                score = gen.distance(statistics, reference)
                results[arm].append({
                    "temperature": temperature, "top_p": top_p,
                    "distance": score, "valid": valid, "n": len(DEV_SEEDS),
                    "attacks_per_second": statistics.attacks_per_second,
                    "mean_simultaneous_voices": statistics.mean_simultaneous_voices,
                    "seconds": round(time.time() - started, 1),
                })
                print(f"    T={temperature:<4} top_p={top_p:<5} "
                      f"dist={score:7.4f}  {statistics.attacks_per_second:6.2f} att/s  "
                      f"{statistics.mean_simultaneous_voices:5.3f} v/onset  "
                      f"valid {valid}/{len(DEV_SEEDS)}  ({time.time()-started:.0f}s)",
                      flush=True)

    selected = {}
    for arm, rows in results.items():
        # declared tie-break: distance, then lowest temperature, then highest top_p
        rows.sort(key=lambda r: (round(r["distance"], 6), r["temperature"], -r["top_p"]))
        selected[arm] = {k: rows[0][k] for k in
                         ("temperature", "top_p", "distance", "valid")}
    payload = {
        "schema": "snes_lm.sampling_selection.v1",
        "objective": "sum of |relative difference| on attacks/s and mean "
                     "simultaneity, plus total variation on the interval "
                     "histogram; lower is closer to the training corpus",
        "grid": {"temperature": list(TEMPERATURES), "top_p": list(TOP_PS)},
        "dev_seeds": list(DEV_SEEDS),
        "dev_seconds": DEV_SECONDS,
        "tie_break": "distance, then lowest temperature, then highest top_p",
        "reference": reference.to_dict(),
        "results": results,
        "selected": selected,
    }
    Path(args.out).write_text(json.dumps(payload, indent=1))
    print("\n  SELECTED (validation-free, listening-free):")
    for arm, choice in selected.items():
        print(f"    {arm:9} temperature {choice['temperature']}, "
              f"top_p {choice['top_p']}  (distance {choice['distance']:.4f})")
    print(f"  written to {args.out}")


if __name__ == "__main__":
    main()
