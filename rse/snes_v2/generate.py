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

**Termination is grammatical, not appended.** A delta is the PREFIX of the
group it belongs to, so a boundary just after a delta has an OPEN group with
no actions, and `EOS` closes an open group -- a group with nothing in it has
nothing to close. An earlier version appended `EOS` unconditionally at the
stopping boundary, which made every soft-stop and hard-cap excerpt
unparseable: the harness, not the model, invalidated them. See
`_terminate()`.
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
    #: Handles released to close the terminal group legally. An intervention,
    #: so it is counted rather than folded into the note content.
    terminal_releases: int = 0
    #: True when the group boundary the stop fired at was already past the
    #: 60 s cap, because a single sampled delta can jump over it. Section 5
    #: stops at the NEXT boundary, so this is conformant but outside the
    #: 45-60 s target and must not be read as an in-target excerpt.
    exceeded_hard_cap: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in (
            "seed", "arm", "outcome", "termination_reason", "duration_sec",
            "n_groups", "n_rollovers", "illegal_mass_mean", "illegal_mass_max",
            "forced_closures", "allocator_steals", "invalid_reason",
            "terminal_releases", "exceeded_hard_cap")} | {
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


def _terminate(state: "gx.GenerationState", emitted: list[int],
               vocab: gx.Vocab, pre_delta_len: int) -> tuple[int, bool, str]:
    """Close the open group legally, then EOS. Returns (releases, ok, note).

    Two cases, and neither invents musical material the model did not
    produce:

    1. Notes are sounding. Each active handle **releases** -- the ending a
       composition actually has -- which gives the terminal group its
       actions, after which EOS is legal. Counted, not hidden.
    2. Nothing is sounding, or the stop fell mid-action. There is no action
       to close the group with, so the **incomplete terminal group is
       dropped**: `emitted` is truncated back to before the delta that
       opened it, and EOS follows the previous group's actions -- a delta
       closes the PRECEDING group, so that group keeps everything it had.
       The clock reverts with it.

    `ok` is False only when there is no earlier group to fall back on: a
    stop inside the very first group, where every candidate termination
    would have to invent an action. The outcome is then retained and
    reported as not decodable, which is the honest answer, not a repair.
    """
    releases = 0
    while vocab.EOS not in state.legal():
        ends = sorted(t for t in state.legal() if vocab.is_end(t))
        if not ends:
            if pre_delta_len <= 1:
                return releases, False, (
                    "no completed group to terminate after; closing the "
                    "stream would require inventing an action")
            del emitted[pre_delta_len:]
            emitted.append(vocab.EOS)
            return releases, True, ("incomplete terminal group dropped; the "
                                    "clock reverts to the previous boundary")
        state.advance(ends[0])
        emitted.append(ends[0])
        releases += 1
    state.advance(vocab.EOS)
    emitted.append(vocab.EOS)
    return releases, True, ""


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
    pre_delta_len = 1
    terminal_releases = 0
    termination_note = ""
    outcome = reason = ""
    invalid_reason = ""

    while True:
        if len(emitted) - 1 >= token_budget:
            where = " mid-action" if state.mid_action else " at a boundary"
            # Amendment A1 calls `budget_exhausted` a DECODABLE outcome, so
            # the stream must actually be closed. Left unterminated it has no
            # EOS, `parse` rejects it, and the candidate would be ineligible
            # for the harness's reason. The releases push a few tokens past
            # the budget; that is disclosed, not hidden.
            terminal_releases, closed, termination_note = _terminate(
                state, emitted, vocab, pre_delta_len)
            outcome = "budget_exhausted"
            reason = f"token budget {token_budget} reached{where}"
            if terminal_releases:
                reason += f"; {terminal_releases} terminal release(s)"
            if termination_note:
                reason += f"; {termination_note}"
            if not closed:
                invalid_reason = termination_note
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
            # Where this group's delta prefix begins, so an empty terminal
            # group can be dropped rather than closed with invented notes.
            if token >= vocab.delta_base and not state.mid_action:
                pre_delta_len = len(emitted)
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
        if state.at_group_boundary and seconds >= soft_stop_sec:
            which = "hard cap" if seconds >= hard_cap_sec else "soft stop"
            terminal_releases, closed, termination_note = _terminate(
                state, emitted, vocab, pre_delta_len)
            seconds = state.seconds(GRID_HZ)
            outcome = "complete"
            reason = f"{which} at {seconds:.2f}s"
            if terminal_releases:
                reason += f"; {terminal_releases} terminal release(s)"
            if termination_note:
                reason += f"; {termination_note}"
            if not closed:
                invalid_reason = termination_note
            break

    return Excerpt(
        seed=config.seed, arm=arm, tokens=emitted, outcome=outcome,
        termination_reason=reason, duration_sec=state.seconds(GRID_HZ),
        n_groups=n_groups, n_rollovers=rollovers,
        illegal_mass_mean=sum(illegal) / len(illegal) if illegal else 0.0,
        illegal_mass_max=max(illegal) if illegal else 0.0,
        forced_closures=state.forced_closures,
        allocator_steals=state.steals,
        invalid_reason=invalid_reason, snapshots=snapshots,
        terminal_releases=terminal_releases,
        exceeded_hard_cap=state.seconds(GRID_HZ) > hard_cap_sec)
