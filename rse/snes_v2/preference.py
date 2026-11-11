"""Clustered bootstrap for the registered paired-preference endpoint.

Registered analysis, fixed before any listener response exists:

- **adjusted preference** = `(v2 wins + 0.5 x ties) / valid comparisons`
- **practical threshold** 60%
- **statistical criterion** clustered 95% CI lower bound strictly above 50%
- **both** must pass to claim demonstrated preference

Each listener judges several pairs and each pair is judged by several
listeners, so responses are correlated along two axes at once. Treating 480
judgements as 480 independent observations would understate the interval
badly -- the same error as the Level 1 track-level bootstrap before it was
reclustered by game, where the interval moved from [22.63, 24.68] to
[21.93, 25.92].

**Two-way clustering is handled by resampling listeners and pairs jointly**:
each replicate draws listeners with replacement AND pairs with replacement,
keeping only the judgements at their intersection. That is conservative
relative to one-way clustering, which is the correct direction when the
sample is 20 listeners.

Ties are **data, not missing values**. "No discernible difference" is a valid
registered response and contributes half a win to each arm.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Sequence

#: A single judgement. `choice` is from the listener's point of view after
#: un-randomising the A/B side, so "v2" always means the v2 excerpt won.
Choice = Literal["v2", "baseline", "tie"]

#: Registered constants. Changing any of these after responses exist would
#: invalidate the endpoint.
PRACTICAL_THRESHOLD = 0.60
CONFIDENCE = 0.95
N_REPLICATES = 10_000
BOOTSTRAP_SEED = 20261111
#: Percentile method, fixed in advance.
CI_METHOD = "percentile"
#: A listener missing more than this share of their assignment is excluded.
MAX_MISSING_SHARE = 0.50


@dataclass(frozen=True)
class Judgement:
    listener: str
    pair: str
    choice: Choice | None     # None = missing or unparseable


@dataclass
class PreferenceResult:
    adjusted_preference: float
    ci_low: float
    ci_high: float
    n_judgements: int
    n_valid: int
    n_ties: int
    n_missing: int
    n_listeners: int
    n_pairs: int
    excluded_listeners: list[str]
    judgements_per_pair: dict[str, int]
    threshold_met: bool
    superiority_established: bool

    @property
    def verdict(self) -> str:
        if self.threshold_met and self.superiority_established:
            return "demonstrated preference"
        if self.threshold_met:
            return "threshold met, superiority not established"
        return "threshold not met"

    def to_dict(self) -> dict[str, Any]:
        return {
            "adjusted_preference": self.adjusted_preference,
            "ci_low": self.ci_low, "ci_high": self.ci_high,
            "confidence": CONFIDENCE, "n_replicates": N_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED, "ci_method": CI_METHOD,
            "practical_threshold": PRACTICAL_THRESHOLD,
            "n_judgements": self.n_judgements, "n_valid": self.n_valid,
            "n_ties": self.n_ties, "n_missing": self.n_missing,
            "n_listeners": self.n_listeners, "n_pairs": self.n_pairs,
            "excluded_listeners": self.excluded_listeners,
            "judgements_per_pair": self.judgements_per_pair,
            "threshold_met": self.threshold_met,
            "superiority_established": self.superiority_established,
            "verdict": self.verdict,
        }


def _score(choice: Choice) -> float:
    return {"v2": 1.0, "baseline": 0.0, "tie": 0.5}[choice]


def _apply_missing_data_rule(
    judgements: Sequence[Judgement],
) -> tuple[list[Judgement], list[str], int]:
    """The registered rule, applied before any estimate is formed.

    A missing response is dropped from its pair's denominator; the pair stays
    in the study. A listener missing more than half their assignment is
    excluded entirely and reported. Neither decision is retrospective.
    """
    assigned: dict[str, int] = {}
    missing: dict[str, int] = {}
    for j in judgements:
        assigned[j.listener] = assigned.get(j.listener, 0) + 1
        if j.choice is None:
            missing[j.listener] = missing.get(j.listener, 0) + 1
    excluded = sorted(
        listener for listener, total in assigned.items()
        if missing.get(listener, 0) / total > MAX_MISSING_SHARE)
    kept = [j for j in judgements
            if j.listener not in excluded and j.choice is not None]
    n_missing = sum(1 for j in judgements
                    if j.listener not in excluded and j.choice is None)
    return kept, excluded, n_missing


def analyse(judgements: Sequence[Judgement], *,
            n_replicates: int = N_REPLICATES,
            seed: int = BOOTSTRAP_SEED) -> PreferenceResult:
    """Adjusted preference with a two-way clustered percentile interval."""
    kept, excluded, n_missing = _apply_missing_data_rule(judgements)
    if not kept:
        raise ValueError("no valid judgements after the missing-data rule")

    by_listener: dict[str, list[Judgement]] = {}
    by_pair: dict[str, list[Judgement]] = {}
    for j in kept:
        by_listener.setdefault(j.listener, []).append(j)
        by_pair.setdefault(j.pair, []).append(j)

    point = sum(_score(j.choice) for j in kept) / len(kept)

    listeners = sorted(by_listener)
    pairs = sorted(by_pair)
    # Index judgements by (listener, pair) so a replicate can take the
    # intersection of its resampled listeners and pairs.
    cell: dict[tuple[str, str], list[Judgement]] = {}
    for j in kept:
        cell.setdefault((j.listener, j.pair), []).append(j)

    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(n_replicates):
        drawn_listeners = [rng.choice(listeners) for _ in listeners]
        drawn_pairs = [rng.choice(pairs) for _ in pairs]
        total = 0.0
        count = 0
        for listener in drawn_listeners:
            for pair in drawn_pairs:
                for j in cell.get((listener, pair), ()):
                    total += _score(j.choice)
                    count += 1
        if count:
            estimates.append(total / count)
    estimates.sort()
    if not estimates:
        raise ValueError("every bootstrap replicate was empty")
    # SYMMETRIC tail allocation: exclude the same COUNT from each end.
    # Indexing the upper bound as int((1-alpha)*n) excluded one fewer value
    # above than below, so the interval was not mirrored under A/B label
    # reversal -- a 0.18 pp asymmetry that the symmetry test caught.
    alpha = (1.0 - CONFIDENCE) / 2.0
    tail = int(alpha * len(estimates))
    low = estimates[tail]
    high = estimates[len(estimates) - 1 - tail]

    return PreferenceResult(
        adjusted_preference=point, ci_low=low, ci_high=high,
        n_judgements=len(judgements), n_valid=len(kept),
        n_ties=sum(1 for j in kept if j.choice == "tie"),
        n_missing=n_missing,
        n_listeners=len(listeners), n_pairs=len(pairs),
        excluded_listeners=excluded,
        judgements_per_pair={p: len(v) for p, v in sorted(by_pair.items())},
        threshold_met=point >= PRACTICAL_THRESHOLD,
        superiority_established=low > 0.50,
    )


def reverse_labels(judgements: Iterable[Judgement]) -> list[Judgement]:
    """Swap v2 and baseline. Used to verify label symmetry."""
    flip = {"v2": "baseline", "baseline": "v2", "tie": "tie"}
    return [Judgement(j.listener, j.pair,
                      None if j.choice is None else flip[j.choice])
            for j in judgements]
