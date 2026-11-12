"""Sampling selection under the frozen procedure and Amendment A1.

Separated from the driver script so the eligibility rule is unit-testable
without a checkpoint, a GPU, or 40 generations.

The rule being enforced (Amendment A1, registered 2026-11-12, before any
development excerpt existed):

- each candidate produces **exactly four** outcomes, one per registered
  development seed;
- a candidate is **eligible only if all four** decode into performances for
  which the frozen metrics are defined;
- an ineligible candidate is **not penalised and not scored over its
  survivors** -- it simply cannot win;
- if no candidate is eligible, selection is **declared unsuccessful**.

Scoring over survivors is the specific failure this prevents: a candidate
succeeding on one seed of four would otherwise be ranked on its single
success, rewarded *because* its failures were excluded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import statistics as st
from typing import Any, Sequence

from . import statistics as sx

#: The registered outcome categories. Every attempted outcome falls in
#: exactly one, and the tally always reports all four, zeros included --
#: a category silently absent from a report is how a rate loses its
#: denominator.
OUTCOME_CATEGORIES = ("complete", "short", "budget_exhausted", "invalid")

#: §2 of the frozen procedure. Disjoint from the primary evaluation seeds.
DEVELOPMENT_SEEDS = (1000, 1001, 1002, 1003)

#: §1: 5 temperatures x 2 top-p values.
TEMPERATURES = (0.8, 0.9, 1.0, 1.1, 1.2)
TOP_P_VALUES = (1.0, 0.95)

#: 10 candidates x 4 seeds. Asserted, never assumed.
EXPECTED_ATTEMPTED = len(TEMPERATURES) * len(TOP_P_VALUES) * len(DEVELOPMENT_SEEDS)


class SelectionFailure(RuntimeError):
    """No candidate was eligible. Not recoverable by choosing a fallback."""


@dataclass
class Outcome:
    """One retained outcome from one registered seed."""

    seed: int
    outcome: str
    #: Per-excerpt frozen statistics, or None when they are undefined --
    #: an undecodable stream or a zero-note performance. This is the
    #: mechanism of Amendment A1: None cannot be scored.
    statistics: dict[str, Any] | None
    reason: str = ""
    #: Reported, never scored. See `duration_summary`.
    duration_sec: float = 0.0
    exceeded_hard_cap: bool = False

    def __post_init__(self) -> None:
        if self.outcome not in OUTCOME_CATEGORIES:
            raise ValueError(f"unregistered outcome {self.outcome!r}")

    @property
    def scorable(self) -> bool:
        return self.statistics is not None


@dataclass
class Candidate:
    temperature: float
    top_p: float
    outcomes: list[Outcome] = field(default_factory=list)

    @property
    def key(self) -> tuple[float, float]:
        return (self.temperature, self.top_p)

    def eligibility(self) -> tuple[bool, list[str]]:
        """(eligible, reasons). All four seeds must yield defined metrics."""
        reasons: list[str] = []
        seeds = [o.seed for o in self.outcomes]
        if sorted(seeds) != sorted(DEVELOPMENT_SEEDS):
            reasons.append(
                f"expected seeds {list(DEVELOPMENT_SEEDS)}, got {sorted(seeds)}")
        if len(seeds) != len(set(seeds)):
            reasons.append("duplicate seed")
        for outcome in self.outcomes:
            if not outcome.scorable:
                reasons.append(
                    f"seed {outcome.seed}: {outcome.outcome} with no defined "
                    f"metrics{(' -- ' + outcome.reason) if outcome.reason else ''}")
        return (not reasons, reasons)


#: §5's target window. Used for REPORTING ONLY.
TARGET_MIN_SEC = 45.0
TARGET_MAX_SEC = 60.0


def duration_summary(outcomes: Sequence[Outcome]) -> dict[str, Any]:
    """Durations beside the objective, never inside it.

    The frozen §4 objective scores density, simultaneity and the interval
    histogram. **None of the three measures duration**, so a candidate whose
    excerpts barely play can score well: the baseline's winning candidate
    scored 0.6969 with only 1 of 40 excerpts in the 45-60 s target and 12
    under 5 seconds, the shortest 0.05 s.

    §4 stays frozen -- that was the registered decision -- so the remedy is
    visibility rather than a changed formula. Every candidate carries this
    block, so a winner that barely plays cannot be read as an in-target
    result. Nothing here feeds `objective()` or `tie_break_key()`.
    """
    durations = sorted(o.duration_sec for o in outcomes)
    if not durations:
        return {"n": 0, "note": "no outcomes"}
    return {
        "n": len(durations),
        "min_sec": durations[0],
        "median_sec": st.median(durations),
        "max_sec": durations[-1],
        "n_in_target": sum(1 for d in durations
                           if TARGET_MIN_SEC <= d <= TARGET_MAX_SEC),
        "n_under_5_sec": sum(1 for d in durations if d < 5.0),
        "n_exceeded_hard_cap": sum(1 for o in outcomes if o.exceeded_hard_cap),
        "target_sec": [TARGET_MIN_SEC, TARGET_MAX_SEC],
        "scored": False,
        "note": "reported beside the frozen §4 objective, never inside it",
    }


@dataclass
class SelectionResult:
    successful: bool
    reference: dict[str, Any]
    rows: list[dict[str, Any]]
    #: Denominators, reported separately and never conflated.
    n_attempted_outcomes: int
    n_candidates: int
    n_eligible_candidates: int
    outcome_tally: dict[str, int]
    selected: dict[str, Any] | None = None
    failure_reason: str = ""


def select(candidates: Sequence[Candidate],
           reference: dict[str, Any]) -> SelectionResult:
    """Apply the frozen objective to the eligible candidates only.

    Ineligible candidates appear in `rows` with their failure reasons and no
    objective. They are never ranked, so no combination of surviving-seed
    values can make one win.
    """
    tally = {category: 0 for category in OUTCOME_CATEGORIES}
    attempted = 0
    for candidate in candidates:
        for outcome in candidate.outcomes:
            tally[outcome.outcome] += 1
            attempted += 1
    if sum(tally.values()) != attempted:            # defensive; see test
        raise AssertionError("outcome tally does not cover every attempt")

    rows: list[dict[str, Any]] = []
    scorable_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        eligible, reasons = candidate.eligibility()
        row: dict[str, Any] = {
            "temperature": candidate.temperature,
            "top_p": candidate.top_p,
            "eligible": eligible,
            "n_outcomes": len(candidate.outcomes),
            "outcomes": [{"seed": o.seed, "outcome": o.outcome,
                          "scorable": o.scorable, "reason": o.reason}
                         for o in sorted(candidate.outcomes,
                                         key=lambda o: o.seed)],
            "ineligibility_reasons": reasons,
            # Beside the objective, never inside it.
            "duration": duration_summary(candidate.outcomes),
        }
        if eligible:
            aggregated = sx.aggregate([o.statistics for o in candidate.outcomes])
            row["aggregate"] = aggregated
            row["objective"] = sx.objective(aggregated, reference)
            scorable_rows.append(row)
        else:
            # No aggregate and no objective: an ineligible candidate is not
            # scored over its survivors, so there is nothing to rank.
            row["aggregate"] = None
            row["objective"] = None
        rows.append(row)

    result = SelectionResult(
        successful=bool(scorable_rows),
        reference=reference,
        rows=rows,
        n_attempted_outcomes=attempted,
        n_candidates=len(candidates),
        n_eligible_candidates=len(scorable_rows),
        outcome_tally=tally,
    )
    if not scorable_rows:
        result.failure_reason = (
            "no candidate had all four development seeds decode into a "
            "performance with defined metrics; per Amendment A1 §6 sampling "
            "selection is declared unsuccessful, no fallback is chosen, the "
            "grid is not changed and seeds are not replaced")
        return result

    best = min(scorable_rows, key=sx.tie_break_key)
    result.selected = {
        "temperature": best["temperature"],
        "top_p": best["top_p"],
        "objective": best["objective"],
        "aggregate": best["aggregate"],
        "duration": best["duration"],
        "tie_break": "lowest distance, then lowest temperature, then highest "
                     "top-p; distances compared at 6 decimal places",
    }
    return result
