"""The sealed test evaluation for the registered Level 1 experiment.

Runs ONCE, after the model, checkpoint, configuration, tokenizer, baseline
configuration and this code are frozen. Protocol:
`snesmdb/docs/PROTOCOL_FREEZE.md`.

The registered criterion is measured against the baseline RE-MEASURED ON TEST,
not against the validation reference:

    pass  iff  test_bits(transformer)  <=  0.95 x test_bits(frozen_slotngram)

That matters because the splits are documented as non-interchangeable --
pitched share runs 77.5 / 81.8 / 85.5% across train/valid/test -- so a
validation-derived threshold would be the wrong bar.

No hyperparameter tuning or checkpoint selection may follow inspection of
these numbers. A poor result is reported as a poor result.

The test set deliberately never left this machine: only train and validation
tensors were sent to the rented GPU host.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from rse.snes_lm.representation import STRIDE, Vocab

SLOT_NAMES = ("TIME", "VOICE", "PITCH", "DUR")
_LN2 = math.log(2.0)


@dataclass
class TrackScore:
    """One track's loss under one model, kept per-track for the bootstrap."""

    track_id: str
    game: str
    bits: float
    n_tokens: int
    slot_bits: list[float] = field(default_factory=lambda: [0.0] * STRIDE)
    slot_counts: list[int] = field(default_factory=lambda: [0] * STRIDE)

    @property
    def bits_per_token(self) -> float:
        return self.bits / self.n_tokens if self.n_tokens else float("nan")


def aggregate(scores: Sequence[TrackScore]) -> dict[str, Any]:
    """Token-weighted overall and per-slot figures, plus per-game."""
    total_bits = sum(s.bits for s in scores)
    total_tokens = sum(s.n_tokens for s in scores)
    slot_bits = [sum(s.slot_bits[i] for s in scores) for i in range(STRIDE)]
    slot_counts = [sum(s.slot_counts[i] for s in scores) for i in range(STRIDE)]
    by_game: dict[str, list[TrackScore]] = {}
    for score in scores:
        by_game.setdefault(score.game, []).append(score)
    return {
        "bits_per_token": total_bits / total_tokens,
        "n_tokens": total_tokens,
        "n_tracks": len(scores),
        "per_slot_bits": {
            SLOT_NAMES[i]: slot_bits[i] / slot_counts[i]
            for i in range(STRIDE)
            if slot_counts[i]
        },
        "per_game_bits": {
            game: sum(s.bits for s in group) / sum(s.n_tokens for s in group)
            for game, group in sorted(by_game.items())
        },
    }


def paired_bootstrap(
    model: Sequence[TrackScore],
    baseline: Sequence[TrackScore],
    *,
    n_resamples: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Bootstrap the relative improvement, resampling TRACKS together.

    Paired and over tracks, both deliberately. Paired because the two models
    see identical tracks, so resampling them independently would discard the
    pairing and inflate the interval. Over tracks rather than tokens because
    tokens within a track are not independent -- a 57,510-token track is one
    piece of music, not 57,510 observations.

    The statistic is the relative reduction in token-weighted bits/token,
    which is exactly the quantity the 5% criterion is stated in.
    """
    assert len(model) == len(baseline)
    pairs = []
    for m, b in zip(model, baseline):
        assert m.track_id == b.track_id, (m.track_id, b.track_id)
        pairs.append((m.bits, m.n_tokens, b.bits, b.n_tokens))

    def relative(sample: Sequence[tuple[float, int, float, int]]) -> float:
        mb = sum(p[0] for p in sample) / sum(p[1] for p in sample)
        bb = sum(p[2] for p in sample) / sum(p[3] for p in sample)
        return (bb - mb) / bb

    observed = relative(pairs)
    rng = random.Random(seed)
    n = len(pairs)
    draws = []
    for _ in range(n_resamples):
        sample = [pairs[rng.randrange(n)] for _ in range(n)]
        draws.append(relative(sample))
    draws.sort()
    low = draws[int(alpha / 2 * n_resamples)]
    high = draws[int((1 - alpha / 2) * n_resamples)]
    return {
        "observed_relative_improvement": observed,
        "ci_lower": low,
        "ci_upper": high,
        "confidence": 1 - alpha,
        "n_resamples": n_resamples,
        "n_tracks": n,
        # The registered threshold is 5%; an interval whose lower bound clears
        # it is the evidence the protocol asks for.
        "ci_excludes_zero": low > 0.0,
        "ci_lower_exceeds_5pct": low > 0.05,
        "fraction_of_resamples_above_5pct": sum(1 for d in draws if d > 0.05)
        / n_resamples,
    }


def level1_verdict(
    model_bits: float, baseline_bits: float, bootstrap: dict[str, Any]
) -> dict[str, Any]:
    """The registered criterion, applied without reinterpretation."""
    threshold = 0.95 * baseline_bits
    relative = (baseline_bits - model_bits) / baseline_bits
    return {
        "model_test_bits": model_bits,
        "baseline_test_bits": baseline_bits,
        "threshold_bits": threshold,
        "relative_improvement": relative,
        "meets_5pct_threshold": model_bits <= threshold,
        "bootstrap_ci_lower": bootstrap["ci_lower"],
        "bootstrap_supports": bootstrap["ci_lower_exceeds_5pct"],
        "verdict": (
            "LEVEL 1 LEARNING CRITERION MET"
            if model_bits <= threshold
            else "LEVEL 1 LEARNING CRITERION NOT MET"
        ),
        "note": (
            "Level 1 also requires reproducibility: fixed split, configuration, "
            "evaluation code and recorded seeds. This function judges the "
            "likelihood criterion only. Nothing here speaks to Level 2, which "
            "requires the separate multi-listener study."
        ),
    }


def write_report(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, indent=1))


def clustered_bootstrap(
    model: Sequence[TrackScore],
    baseline: Sequence[TrackScore],
    *,
    n_resamples: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Bootstrap resampling GAMES, not tracks.

    The track-level bootstrap answers "how uncertain is this improvement over
    these tracks", and that is the registered statistic. It does NOT answer
    "how uncertain is it over unseen games", because tracks within a game are
    correlated -- they share a composer, a driver, an instrument bank and an
    idiom, so 176 tracks are nowhere near 176 independent observations of
    game-level performance.

    Resampling whole games with replacement treats the game as the unit. With
    only 8 games the interval is necessarily wide, and that width is the
    honest expression of what eight games can support. It is reported
    alongside the registered track-level interval, not instead of it.
    """
    by_game: dict[str, list[tuple[float, int, float, int]]] = {}
    for m, b in zip(model, baseline):
        assert m.track_id == b.track_id, (m.track_id, b.track_id)
        by_game.setdefault(m.game, []).append((m.bits, m.n_tokens, b.bits, b.n_tokens))
    games = sorted(by_game)

    def relative(selected: Sequence[str]) -> float:
        rows = [row for game in selected for row in by_game[game]]
        mb = sum(r[0] for r in rows) / sum(r[1] for r in rows)
        bb = sum(r[2] for r in rows) / sum(r[3] for r in rows)
        return (bb - mb) / bb

    observed = relative(games)
    rng = random.Random(seed)
    draws = []
    for _ in range(n_resamples):
        sample = [games[rng.randrange(len(games))] for _ in range(len(games))]
        draws.append(relative(sample))
    draws.sort()
    low = draws[int(alpha / 2 * n_resamples)]
    high = draws[int((1 - alpha / 2) * n_resamples)]
    return {
        "unit": "game",
        "n_games": len(games),
        "observed_relative_improvement": observed,
        "ci_lower": low,
        "ci_upper": high,
        "confidence": 1 - alpha,
        "ci_lower_exceeds_5pct": low > 0.05,
        "note": (
            "Resamples whole games. With 8 clusters this interval is wide by "
            "construction; that width is what eight games support. Reported "
            "alongside the registered track-level interval, not instead of it."
        ),
    }
