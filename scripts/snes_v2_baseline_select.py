#!/usr/bin/env python3
"""Select the matched G3 baseline's order on VALIDATION. Test stays sealed.

Order candidates were declared before selection: `baseline.ORDER_CANDIDATES`.
Selection is on validation bits per **scored target**, using the registered
mask that excludes padding and snapshot positions -- the same denominator the
Transformer's loss uses. Training likelihood is never used to choose.

The baseline is fitted on the **exported training windows**, so both arms see
exactly the same token streams, including snapshots. Nothing about the
Transformer is required to run this.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import random

from rse.snes_v2 import baseline as bl  # noqa: E402
from rse.snes_v2 import delta as dx  # noqa: E402
from rse.snes_v2 import grammar as gx  # noqa: E402

#: Constrained scoring needs the legal set at each target, whose median size is
#: 1,746, so scoring all 1,356,836 validation targets is ~2e9 distribution
#: evaluations -- out of reach in Python. A fixed random subsample is used
#: instead, IDENTICAL across every candidate and family, with its standard
#: error reported. The subsample is declared here, not chosen after seeing
#: results.
SUBSAMPLE = 8000
SUBSAMPLE_SEED = 90210


def load(window_dir: Path, split: str, limit: int | None = None):
    data = np.load(window_dir / f"{split}.npz")
    tokens, mask, lengths = data["tokens"], data["mask"], data["lengths"]
    if limit:
        tokens, mask, lengths = tokens[:limit], mask[:limit], lengths[:limit]
    streams = [tokens[i, :lengths[i]].astype(int).tolist()
               for i in range(len(lengths))]
    masks = [mask[i, :lengths[i]].tolist() for i in range(len(lengths))]
    return streams, masks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", default=str(
        Path.home() / "workplace/v2_artifacts/windows"))
    parser.add_argument("--out", default=str(
        Path.home() / "workplace/v2_artifacts/baseline_selection.json"))
    parser.add_argument("--train-limit", type=int, default=0,
                        help="0 = all training windows")
    args = parser.parse_args()

    window_dir = Path(args.windows)
    manifest = json.loads((window_dir / "MANIFEST.json").read_text())
    vocab_size = manifest["pad_id"] + 1

    train_streams, _ = load(window_dir, "train", args.train_limit or None)
    valid_streams, valid_masks = load(window_dir, "valid")
    print(f"  vocab {vocab_size:,}  train windows {len(train_streams):,}  "
          f"valid windows {len(valid_streams):,}", flush=True)
    print(f"  order candidates {bl.ORDER_CANDIDATES} (declared before "
          f"selection)\n", flush=True)

    # The grammar mask comes from the same causal state for every model.
    raw_vocab = json.loads((Path.home() /
        "workplace/v2_artifacts/delta_vocab_k512.json").read_text())
    gvocab = gx.Vocab(delta=dx.DeltaCodec(
        direct={int(k): v for k, v in raw_vocab["direct"].items()},
        n_observed=raw_vocab["n_distinct_train"]))
    # Enumerate maskable positions WITHOUT materialising their legal sets,
    # subsample, and only then build the sets that are actually needed. A
    # legal set holds about 1,746 tokens, so one per validation position would
    # be over 100 GB -- the run was killed at 19 GB of 30 GB climbing.
    candidates_pos = [(w, i)
                      for w, (stream, mask) in enumerate(
                          zip(valid_streams, valid_masks))
                      for i in range(1, len(stream)) if mask[i]]
    rng = random.Random(SUBSAMPLE_SEED)
    picked = sorted(rng.sample(range(len(candidates_pos)),
                               min(SUBSAMPLE, len(candidates_pos))))
    wanted: dict[int, set[int]] = {}
    for index in picked:
        w, i = candidates_pos[index]
        wanted.setdefault(w, set()).add(i)
    print(f"  building legal sets for {len(picked):,} sampled targets across "
          f"{len(wanted):,} windows ...", flush=True)
    sets_by_position: dict[tuple[int, int], frozenset[int]] = {}
    for w, indices in wanted.items():
        sets = gx.legal_at(valid_streams[w], gvocab)
        for i in indices:
            if sets[i]:
                sets_by_position[(w, i)] = sets[i]
        del sets
    positions = [candidates_pos[i] for i in picked
                 if candidates_pos[i] in sets_by_position]
    legal_sets = [sets_by_position[p] for p in positions]
    print(f"  {len(candidates_pos):,} maskable validation targets; "
          f"scoring a fixed subsample of {len(positions):,}\n", flush=True)
    del sets_by_position

    results = []
    families = (("witten_bell", bl.G3NGram), ("kneser_ney", bl.G3KneserNey))
    for family, cls in families:
        for order in bl.ORDER_CANDIDATES:
            t0 = time.time()
            model = cls(vocab_size=vocab_size, order=order)
            model.fit(train_streams)
            fit_sec = time.time() - t0
            t0 = time.time()
            scored = bl.score_constrained(model, valid_streams, positions,
                                          legal_sets)
            row = {"family": family, "order": order, **scored,
                   "fit_sec": round(fit_sec, 1),
                   "score_sec": round(time.time() - t0, 1),
                   "contexts": model.memory_estimate()}
            results.append(row)
            print(f"  {family:12} order {order}: "
                  f"{scored['bits_per_scored_target']:.4f} "
                  f"+/- {scored['standard_error']:.4f} bits (constrained)   "
                  f"unconstrained {scored['unconstrained_bits']:.4f}   "
                  f"(fit {fit_sec:.0f}s, score {row['score_sec']:.0f}s)",
                  flush=True)
            del model

    best = min(results, key=lambda r: r["bits_per_scored_target"])
    payload = {
        "schema": "snes_v2.baseline_selection.v1",
        "smoothing": "Witten-Bell, classic interpolated form",
        "order_candidates": list(bl.ORDER_CANDIDATES),
        "selection_criterion": "validation bits per scored target, registered "
                               "mask excluding padding and snapshot positions",
        "training_likelihood_used": False,
        "test_split_touched": False,
        "fit_population": "exported training windows, identical to the "
                          "Transformer's",
        "results": results,
        "amendment": "Modified Kneser-Ney added as a prespecified alternative "
                     "AFTER the Witten-Bell results were seen on the "
                     "development population. Prospective relative to "
                     "Transformer training and the sealed test, NOT relative "
                     "to baseline validation. Disclosed as such.",
        "scoring_convention": "grammar-constrained renormalisation, "
                              "P(y|c,legal), same causal mask for all models",
        "subsample": {"n": SUBSAMPLE, "seed": SUBSAMPLE_SEED,
                      "reason": "median legal set 1,746 makes full-population "
                                "constrained scoring intractable in Python",
                      "identical_across_candidates": True},
        "selected_family": best["family"],
        "selected_order": best["order"],
        "selected_bits": best["bits_per_scored_target"],
    }
    Path(args.out).write_text(json.dumps(payload, indent=1))
    print(f"\n  SELECTED {best['family']} order {best['order']} at "
          f"{best['bits_per_scored_target']:.4f} +/- "
          f"{best['standard_error']:.4f} bits (constrained)")
    print(f"  written to {args.out}")


if __name__ == "__main__":
    main()
