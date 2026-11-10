#!/usr/bin/env python3
"""Run the sealed test evaluation. ONCE.

Protocol: snesmdb/docs/PROTOCOL_FREEZE.md. No tuning or checkpoint selection
may follow inspection of these numbers.

The target accounting matters more than it looks. The model is trained and
evaluated on WINDOWS: for each 1024-token window it predicts positions
1..1023, so position 0 of every window after the first is context only and is
never a target. Scoring the n-gram on whole tracks instead would give it 1,137
extra targets on the test split -- 1,313 windows against 176 tracks -- and the
two models would not be performing the same task.

So the baseline is scored on the identical windows, with history confined to
each window, exactly as the model sees it.

Two consequences, both stated rather than hidden:

* The baseline's validation figure of 3.3839 was computed on whole-track
  sequences and is therefore NOT directly comparable to the windowed test
  number. This is a further reason the registered criterion is measured
  against the baseline re-measured on test.
* The baseline is still FITTED on whole train tracks, which gives it slightly
  more context than the model had. That favours the baseline, and the
  conservative direction is the right one for a criterion the model has to
  clear.

1024 is divisible by the stride of 4, so a window's position 0 always falls on
a TIME slot and the per-slot convention holds unchanged within every window.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rse.snes_lm import baselines as bl  # noqa: E402
from rse.snes_lm import representation as rep  # noqa: E402
from rse.snes_lm.evaluate_test import (  # noqa: E402
    SLOT_NAMES,
    TrackScore,
    aggregate,
    level1_verdict,
    paired_bootstrap,
    write_report,
)
from rse.snes_lm.release import Release  # noqa: E402
from rse.snes_lm.train import Config, Decoder  # noqa: E402

_LN2 = math.log(2.0)


def score_model(checkpoint: Path, windows, meta, vocab) -> list[TrackScore]:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = Config(**{**state["config"], "device": "cpu"})
    model = Decoder(vocab.size, cfg)
    model.load_state_dict(state["model"])
    model.eval()

    by_track: dict[str, TrackScore] = {}
    with torch.no_grad():
        for start in range(0, windows.shape[0], 8):
            batch = windows[start : start + 8]
            logits = model(batch)
            targets = batch[:, 1:]
            keep = targets != vocab.PAD
            flat = (
                F.cross_entropy(
                    logits[:, :-1, :].reshape(-1, logits.shape[-1]),
                    targets.reshape(-1),
                    reduction="none",
                ).view_as(targets)
                * keep
            )
            positions = torch.arange(targets.shape[1]) % rep.STRIDE
            for row in range(batch.shape[0]):
                info = meta[start + row]
                score = by_track.setdefault(
                    info["track_id"],
                    TrackScore(info["track_id"], info["game"], 0.0, 0),
                )
                score.bits += float(flat[row].sum()) / _LN2
                score.n_tokens += int(keep[row].sum())
                for slot in range(rep.STRIDE):
                    mask = (positions == slot) & keep[row]
                    score.slot_bits[slot] += float((flat[row] * mask).sum()) / _LN2
                    score.slot_counts[slot] += int(mask.sum())
    return [by_track[k] for k in sorted(by_track)]


def score_baseline(model, windows, meta, vocab) -> list[TrackScore]:
    """The baseline on the IDENTICAL windows, history confined to each one."""
    by_track: dict[str, TrackScore] = {}
    for index in range(windows.shape[0]):
        body = [int(t) for t in windows[index].tolist() if t != vocab.PAD]
        if len(body) < 2:
            continue
        info = meta[index]
        score = by_track.setdefault(
            info["track_id"], TrackScore(info["track_id"], info["game"], 0.0, 0)
        )
        for offset, value in enumerate(model.stream(body)):
            bits = -value / _LN2
            slot = bl._slot(offset + 1)
            score.bits += bits
            score.n_tokens += 1
            score.slot_bits[slot] += bits
            score.slot_counts[slot] += 1
    return [by_track[k] for k in sorted(by_track)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--release", default="~/snesmdb-releases/snesmdb-50game-rc1")
    parser.add_argument("--vocab", default="/tmp/snes_vocab.json")
    parser.add_argument("--test-windows", default="/tmp/snes_test_windows.pt")
    parser.add_argument("--out", default="/tmp/snes_sealed_test.json")
    parser.add_argument("--resamples", type=int, default=10_000)
    args = parser.parse_args()

    bundle = torch.load(args.test_windows, weights_only=False)
    windows, meta = bundle["windows"], bundle["meta"]
    vocab = rep.Vocab.from_dict(bundle["vocab"])
    print(f"  test: {windows.shape[0]} windows / "
          f"{len({m['track_id'] for m in meta})} tracks / "
          f"{len({m['game'] for m in meta})} games", flush=True)

    print("  fitting the frozen comparator on train...", flush=True)
    release = Release(args.release)
    train_seqs = [
        rep.encode(release.events(t), vocab).tokens
        for t in release.tracks(split="train")
    ]
    comparator = bl.SlotNGram(vocab, order=10, discount=64.0,
                              use_event_prefix=False).fit(train_seqs)

    t0 = time.time()
    print("  scoring the comparator on test windows...", flush=True)
    base_scores = score_baseline(comparator, windows, meta, vocab)
    print(f"    {time.time()-t0:.0f}s", flush=True)

    t0 = time.time()
    print("  scoring the model on test windows...", flush=True)
    model_scores = score_model(args.checkpoint, windows, meta, vocab)
    print(f"    {time.time()-t0:.0f}s", flush=True)

    model_agg = aggregate(model_scores)
    base_agg = aggregate(base_scores)
    assert model_agg["n_tokens"] == base_agg["n_tokens"], (
        f"target mismatch: model {model_agg['n_tokens']} vs "
        f"baseline {base_agg['n_tokens']} -- the two are not scoring the "
        f"same task and the comparison is invalid"
    )

    boot = paired_bootstrap(model_scores, base_scores,
                            n_resamples=args.resamples, seed=0)
    verdict = level1_verdict(model_agg["bits_per_token"],
                             base_agg["bits_per_token"], boot)

    per_game_delta = {
        game: (base_agg["per_game_bits"][game] - model_agg["per_game_bits"][game])
        / base_agg["per_game_bits"][game]
        for game in sorted(base_agg["per_game_bits"])
    }

    report = {
        "schema": "snes_lm.sealed_test.v1",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": __import__("hashlib").sha256(
            args.checkpoint.read_bytes()
        ).hexdigest(),
        "model": model_agg,
        "baseline": base_agg,
        "bootstrap": boot,
        "verdict": verdict,
        "per_game_relative_improvement": per_game_delta,
        "n_games_improved": sum(1 for v in per_game_delta.values() if v > 0),
        "accounting_note": (
            "Both models scored on identical windowed targets with history "
            "confined to each window. The baseline is fitted on whole train "
            "tracks, which favours it."
        ),
    }
    write_report(args.out, report)

    print(f"\n  model    {model_agg['bits_per_token']:.4f} bits/token")
    print(f"  baseline {base_agg['bits_per_token']:.4f} bits/token")
    print(f"  relative improvement {verdict['relative_improvement']*100:.2f}%  "
          f"(threshold 5%)")
    print(f"  bootstrap 95% CI [{boot['ci_lower']*100:.2f}%, "
          f"{boot['ci_upper']*100:.2f}%]")
    print(f"\n  per slot:")
    for slot in SLOT_NAMES:
        m = model_agg["per_slot_bits"][slot]
        b = base_agg["per_slot_bits"][slot]
        print(f"    {slot:6} model {m:6.3f}  baseline {b:6.3f}  "
              f"{100*(b-m)/b:+6.1f}%")
    print(f"\n  per game ({report['n_games_improved']}/{len(per_game_delta)} improved):")
    for game, delta in sorted(per_game_delta.items(), key=lambda kv: -kv[1]):
        print(f"    {game:22} {delta*100:+6.2f}%")
    print(f"\n  {verdict['verdict']}")
    print(f"  report written to {args.out}")


if __name__ == "__main__":
    main()
