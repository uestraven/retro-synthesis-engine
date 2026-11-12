"""G3 autoregressive generation with the frozen mask, state and rollover.

Order of operations, fixed: **mask to legal tokens, renormalise, then apply
temperature and nucleus.** Masking after sampling would let an illegal token
be drawn and silently replaced; masking before renormalisation would leave the
distribution un-normalised and make top-p meaningless.

Everything stateful comes from `grammar.GenerationState` -- the same engine
`legal_at` wraps -- so **no separate inference implementation exists**. Level
2's positional-phase defect arose precisely from training and inference
building their contexts by different code.

Rollover reconstructs the model input exactly as `build_windows` would at the
same musical instant: `BOS`, the snapshot of currently sounding notes, then
the groups that follow. A snapshot carries handle, current pitch, status and
age -- never a note's eventual end, which a generator could not know.

Nothing is retried. A seed that exhausts its budget, stops early, or breaks
the grammar is a **retained outcome**, because a rate without its failures is
not a rate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from rse.snes_v2 import grammar as gx

GRID_HZ = 4000.0
SOFT_STOP_SEC = 45.0
HARD_CAP_SEC = 60.0
TOKEN_BUDGET = 6000
CONTEXT = 1024


@dataclass
class SamplingConfig:
    temperature: float = 1.0
    top_p: float = 1.0
    seed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {"temperature": self.temperature, "top_p": self.top_p,
                "seed": self.seed}


@dataclass
class Excerpt:
    seed: int
    arm: str
    tokens: list[int]
    #: complete | short | budget_exhausted | invalid
    outcome: str
    termination_reason: str
    duration_sec: float
    n_groups: int
    n_rollovers: int
    #: Probability mass the model placed on ILLEGAL tokens before masking.
    #: The diagnostic of what it learned; the mask itself earns it no credit.
    illegal_mass_mean: float
    illegal_mass_max: float
    forced_closures: int
    allocator_steals: int
    invalid_reason: str = ""
    snapshots: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in (
            "seed", "arm", "outcome", "termination_reason", "duration_sec",
            "n_groups", "n_rollovers", "illegal_mass_mean", "illegal_mass_max",
            "forced_closures", "allocator_steals", "invalid_reason")} | {
            "n_tokens": len(self.tokens), "snapshot_sizes": self.snapshots}


def _apply_sampling(probs: list[float], legal: Sequence[int],
                    config: SamplingConfig) -> tuple[list[int], list[float]]:
    """Mask, renormalise, then temperature and nucleus. Order is load-bearing."""
    kept = [(t, probs[t]) for t in legal]
    mass = sum(p for _, p in kept)
    if mass <= 0:
        kept = [(t, 1.0) for t, _ in kept]
        mass = float(len(kept)) or 1.0
    kept = [(t, p / mass) for t, p in kept]
    if config.temperature != 1.0:
        kept = [(t, p ** (1.0 / config.temperature)) for t, p in kept]
        total = sum(p for _, p in kept) or 1.0
        kept = [(t, p / total) for t, p in kept]
    if config.top_p < 1.0:
        kept.sort(key=lambda tp: -tp[1])
        cumulative = 0.0
        cut = len(kept)
        for index, (_, p) in enumerate(kept):
            cumulative += p
            if cumulative >= config.top_p:
                cut = index + 1
                break
        kept = kept[:cut]
        total = sum(p for _, p in kept) or 1.0
        kept = [(t, p / total) for t, p in kept]
    return [t for t, _ in kept], [p for _, p in kept]


def generate(
    next_distribution: Callable[[Sequence[int]], list[float]],
    vocab: gx.Vocab,
    config: SamplingConfig,
    *,
    arm: str = "model",
    soft_stop_sec: float = SOFT_STOP_SEC,
    hard_cap_sec: float = HARD_CAP_SEC,
    token_budget: int = TOKEN_BUDGET,
    context: int = CONTEXT,
) -> Excerpt:
    """Generate one excerpt. `next_distribution` maps a context to probabilities.

    The callable abstraction lets the Transformer and the n-gram baseline share
    this one harness, and lets tests drive it with a scripted model whose
    failures are reproducible.
    """
    import random

    rng = random.Random(config.seed)
    state = gx.GenerationState(vocab)
    emitted: list[int] = [vocab.BOS]
    window: list[int] = [vocab.BOS]
    illegal: list[float] = []
    snapshots: list[int] = []
    rollovers = 0
    n_groups = 0
    outcome = reason = ""
    invalid_reason = ""

    while True:
        if len(emitted) - 1 >= token_budget:
            outcome, reason = "budget_exhausted", (
                f"token budget {token_budget} reached"
                + (" mid-action" if state.mid_action else " at a boundary"))
            break
        # Rollover only at a group boundary, so no action or escape is split.
        if len(window) >= context - 8 and state.at_group_boundary:
            snapshot = state.snapshot()
            window = [vocab.BOS]
            if snapshot:
                window += gx.encode_snapshot(snapshot, vocab)
            snapshots.append(len(snapshot))
            rollovers += 1

        probs = next_distribution(window)
        legal = sorted(state.legal())
        if not legal:
            outcome, reason = "invalid", "no legal token available"
            invalid_reason = reason
            break
        illegal.append(max(0.0, 1.0 - sum(probs[t] for t in legal)))
        tokens, weights = _apply_sampling(probs, legal, config)
        token = rng.choices(tokens, weights=weights, k=1)[0]

        try:
            before_boundary = state.at_group_boundary
            state.advance(token)
        except gx.GrammarError as exc:
            outcome, reason = "invalid", str(exc)
            invalid_reason = str(exc)
            break
        emitted.append(token)
        window.append(token)
        if not before_boundary and state.at_group_boundary:
            n_groups += 1

        if token == vocab.EOS:
            seconds = state.seconds(GRID_HZ)
            if seconds >= soft_stop_sec:
                outcome, reason = "complete", f"EOS at {seconds:.2f}s"
            else:
                outcome, reason = "short", f"EOS at {seconds:.2f}s, before soft stop"
            break

        seconds = state.seconds(GRID_HZ)
        if state.at_group_boundary and seconds >= hard_cap_sec:
            emitted.append(vocab.EOS)
            outcome, reason = "complete", f"hard cap at {seconds:.2f}s"
            break
        if state.at_group_boundary and seconds >= soft_stop_sec:
            emitted.append(vocab.EOS)
            outcome, reason = "complete", f"soft stop at {seconds:.2f}s"
            break

    return Excerpt(
        seed=config.seed, arm=arm, tokens=emitted, outcome=outcome,
        termination_reason=reason, duration_sec=state.seconds(GRID_HZ),
        n_groups=n_groups, n_rollovers=rollovers,
        illegal_mass_mean=sum(illegal) / len(illegal) if illegal else 0.0,
        illegal_mass_max=max(illegal) if illegal else 0.0,
        forced_closures=state.forced_closures,
        allocator_steals=state.steals,
        invalid_reason=invalid_reason, snapshots=snapshots)
