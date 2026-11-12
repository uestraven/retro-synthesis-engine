#!/usr/bin/env python3
"""Matched predictive comparison: Transformer vs KN baseline, same population.

**Secondary and descriptive.** It cannot change the selected checkpoint or the
generation procedure; it is not another selection stage.

The checkpoint's training-time validation loss of 6.1251 bits must NOT be
compared against the baseline's 7.2184: that figure is unconstrained and
measured over all 1,356,836 validation targets, while the baseline's is
grammar-constrained over an 8,000-target subsample. Mixing them would mix both
the convention and the denominator -- the error class this project has hit
three times.

So both arms are scored here on the identical 8,000 targets -- same seed, same
enumeration order -- under the identical constrained convention

    P(y | c, legal) = P(y | c) / sum over legal of P(v | c)

with the mask built from the same causal state. The target count is asserted,
not assumed.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics as st
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rse.snes_lm.train import Config, Decoder  # noqa: E402
from rse.snes_v2 import baseline as bl  # noqa: E402
from rse.snes_v2 import delta as dx  # noqa: E402
from rse.snes_v2 import grammar as gx  # noqa: E402
from rse.snes_v2.accounting import Accounting  # noqa: E402

#: Must match scripts/snes_v2_baseline_select.py exactly.
SUBSAMPLE = 8000
SUBSAMPLE_SEED = 90210


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", default=str(
        Path.home() / "workplace/v2_artifacts/windows"))
    parser.add_argument("--checkpoint", default=str(ROOT / "tmp/run/best.pt"))
    parser.add_argument("--out", default=str(
        Path.home() / "workplace/v2_artifacts/matched_comparison.json"))
    args = parser.parse_args()

    window_dir = Path(args.windows)
    manifest = json.loads((window_dir / "MANIFEST.json").read_text())
    vocab_size = manifest["pad_id"] + 1
    raw = json.loads((Path.home() /
        "workplace/v2_artifacts/delta_vocab_k512.json").read_text())
    gvocab = gx.Vocab(delta=dx.DeltaCodec(
        direct={int(k): v for k, v in raw["direct"].items()},
        n_observed=raw["n_distinct_train"]))

    data = np.load(window_dir / "valid.npz")
    tokens, mask, lengths = data["tokens"], data["mask"], data["lengths"]
    streams = [tokens[i, :lengths[i]].astype(int).tolist()
               for i in range(len(lengths))]
    masks = [mask[i, :lengths[i]].tolist() for i in range(len(lengths))]

    # The identical subsample: same enumeration, same seed.
    candidates = [(w, i) for w, (s, m) in enumerate(zip(streams, masks))
                  for i in range(1, len(s)) if m[i]]
    rng = random.Random(SUBSAMPLE_SEED)
    picked = sorted(rng.sample(range(len(candidates)),
                               min(SUBSAMPLE, len(candidates))))
    wanted: dict[int, set[int]] = {}
    for index in picked:
        w, i = candidates[index]
        wanted.setdefault(w, set()).add(i)
    legal_of: dict[tuple[int, int], frozenset[int]] = {}
    for w, indices in wanted.items():
        sets = gx.legal_at(streams[w], gvocab)
        for i in indices:
            if sets[i]:
                legal_of[(w, i)] = sets[i]
        del sets
    positions = [candidates[i] for i in picked if candidates[i] in legal_of]
    legal_sets = [legal_of[p] for p in positions]

    acc = Accounting(stage="matched_comparison")
    acc.check("subsample_size", observed=len(positions), intended=SUBSAMPLE,
              population="registered 8,000-target validation subsample")
    acc.exit_if_failed()
    print(f"  {len(positions):,} matched targets, constrained convention\n",
          flush=True)

    # --- Transformer ---
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = Config(**{**state["config"], "device": "cpu"})
    model = Decoder(state["vocab_size"], cfg)
    model.load_state_dict(state["model"])
    model.eval()
    acc.check("checkpoint_step", observed=state["step"], intended=5400,
              population="selected checkpoint")
    acc.check("n_parameters", observed=sum(p.numel() for p in model.parameters()),
              intended=5_646_336, population="declared parameter count")

    by_window: dict[int, list[int]] = {}
    for index, (w, i) in enumerate(positions):
        by_window.setdefault(w, []).append(index)
    transformer_bits: list[float] = [0.0] * len(positions)
    with torch.no_grad():
        for count, (w, indices) in enumerate(by_window.items()):
            row = torch.tensor([tokens[w].astype(np.int64)], dtype=torch.long)
            logits = model(row[:, :-1])[0]
            probs = F.softmax(logits.float(), dim=-1)
            for index in indices:
                _, i = positions[index]
                dist = probs[i - 1]
                target = streams[w][i]
                legal = list(legal_sets[index])
                raw_p = float(dist[target])
                mass = float(dist[torch.tensor(legal)].sum())
                transformer_bits[index] = -math.log2(
                    max(raw_p / max(mass, 1e-30), 1e-300))
            if (count + 1) % 400 == 0:
                print(f"    transformer {count + 1}/{len(by_window)}", flush=True)

    # --- baseline, refitted identically ---
    train = np.load(window_dir / "train.npz")
    train_streams = [train["tokens"][i, :train["lengths"][i]].astype(int).tolist()
                     for i in range(len(train["lengths"]))]
    print("  refitting the frozen KN order-1 baseline ...", flush=True)
    kn = bl.G3KneserNey(vocab_size=vocab_size, order=1).fit(train_streams)
    kn_scored = bl.score_constrained(kn, streams, positions, legal_sets)

    t_mean = st.mean(transformer_bits)
    t_se = (st.variance(transformer_bits) / len(transformer_bits)) ** 0.5
    b_mean = kn_scored["bits_per_scored_target"]
    paired = [b - t for b, t in zip(
        [x for x in _per_target(kn, streams, positions, legal_sets)],
        transformer_bits)]
    paired_mean = st.mean(paired)
    paired_se = (st.variance(paired) / len(paired)) ** 0.5

    report = {
        "schema": "snes_v2.matched_comparison.v1",
        "status": "secondary descriptive result; cannot change the checkpoint "
                  "or the generation procedure",
        "population": {"n_targets": len(positions), "seed": SUBSAMPLE_SEED,
                       "split": "validation",
                       "convention": "grammar-constrained P(y|c,legal)"},
        "transformer": {"checkpoint_step": state["step"],
                        "bits_per_scored_target": t_mean,
                        "standard_error": t_se},
        "baseline": {"family": "kneser_ney", "order": 1,
                     "bits_per_scored_target": b_mean,
                     "standard_error": kn_scored["standard_error"]},
        "relative_improvement": (b_mean - t_mean) / b_mean,
        "paired_difference_bits": {"mean": paired_mean,
                                   "standard_error": paired_se,
                                   "note": "baseline minus transformer, per "
                                           "target; paired because both arms "
                                           "scored the identical targets"},
        "do_not_compare": "the checkpoint's training-time 6.1251 bits is "
                          "unconstrained over all 1,356,836 validation "
                          "targets and is NOT comparable to either figure here",
    }
    Path(args.out).write_text(json.dumps(report, indent=1))
    acc.note("transformer_bits", t_mean)
    acc.note("baseline_bits", b_mean)
    acc.write(Path(args.out).with_name("matched_comparison_accounting.json"))

    print(f"\n  transformer   {t_mean:.4f} +/- {t_se:.4f} bits")
    print(f"  KN order 1    {b_mean:.4f} +/- {kn_scored['standard_error']:.4f} bits")
    print(f"  relative improvement {100 * report['relative_improvement']:.2f}%")
    print(f"  paired difference    {paired_mean:+.4f} +/- {paired_se:.4f} bits")
    acc.exit_if_failed()


def _per_target(model, windows, positions, legal_sets):
    for (w, i), legal in zip(positions, legal_sets):
        stream = windows[w]
        dist = model.distribution(stream[max(0, i - model.order):i])
        raw_p = max(dist[stream[i]], 1e-300)
        mass = sum(dist[v] for v in legal) or 1e-300
        yield -math.log2(max(raw_p / mass, 1e-300))


if __name__ == "__main__":
    main()
