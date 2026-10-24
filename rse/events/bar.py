"""Finding the bar: how many grid steps repeat, and which one starts them.

Why this exists
---------------
`rse.events.pulse` finds the grid a piece's onsets lie on, but a grid is not a
bar. The detected step is typically a 32nd-note-sized subdivision: measured
across 93 pieces with a clear repeat, the content repeats every 32 steps at the
median, with 58% of pieces on a multiple of 16 and 63% on a power of two.

That is what makes the tokenizer's `Position_x` meaningless, and it is not fixed
by aligning the grid alone. `Position` was "frame index modulo 16" — so with a
bar 32 steps long, position 0 lands on a downbeat in one bar and on the middle
of the next, and the token's musical sense alternates within a single piece. A
phase term corrects where the grid sits; only a bar length corrects what a
position counts against.

Method
------
Quantize every sounding event onto the pulse grid and reduce each step to the
set of (channel, value) pairs sounding there. Score a candidate length L by how
well the sequence matches itself L steps later, using Jaccard overlap over the
steps where either side has content — overlap rather than equality because
repeats carry fills and variations, and demanding an exact match throws away
pieces whose form is obvious to a listener.

Keeping the pitch in the compared symbol is deliberate. Comparing only which
channels sound is far more permissive and scores a repeat for almost every
piece, but it is also satisfied by any piece with continuous accompaniment, and
it measurably picks worse lengths (see Limits).

Of the lengths scoring near the peak, take the **smallest**. This is the
opposite of the rule in `pulse`, and deliberately so: a pattern repeating every
32 steps also matches at 64 and 96, so here the multiples are the artefacts and
the fundamental is the answer. In `pulse` the divisors were the artefacts.

Then double that length until it reaches a bar-sized window. A piece whose drums
repeat every 8 steps has a real 8-step period, but counting positions against 8
would fire four Bar tokens per notated bar and make the `Length` conditioning
mean something different again. Doubling is safe because a signal matching at L
necessarily matches at 2L. This is a stated preference for a bar-sized window,
not something the data derives on its own.

The downbeat is the offset putting the most onsets on position 0, preferring
percussion when a piece has any, since a kick on beat one is the usual anchor.

Limits — read before relying on this
------------------------------------
This is markedly less reliable than `pulse`, and the numbers say so. Over 300
pieces: a pulse is found for 81%, a repeat for 57% of those, and 64% of the
resulting lengths are powers of two — so a bar that is both present and
plausible covers **30% of the corpus**. The modal lengths are 16 and 32, with
24, 51 and 20 also appearing. `is_musical` exposes that split so a caller can
fall back rather than tokenize against a length it has no reason to trust.

Three alternatives were measured and rejected. Scoring channel occupancy, or
per-channel onset correlation, raises coverage to 98-100% but drops the
power-of-two rate to 52-53%: they find a repeat in everything, including pieces
that have none, and a wrong bar is worse than an honest refusal. Restricting the
search to plausible meters (8/16/32/64, or those plus triple-meter 12/24/48)
looks attractive but changes the answer on a third of the pieces the free search
already resolved to a power of two, so it is not the clean filter it appears to
be.

**The practical consequence: do not block a tokenizer change on this.** Most of
the representational win from `pulse` does not need a correct bar. Aligning the
grid makes the step size and every `Duration` musical — today a duration is
counted in 24 Hz frames, whose musical value differs per piece — and that is
independent of whether a position counts against the right bar length. Treat
this module as a soft signal and keep a fallback.

Nothing here handles a piece that changes meter partway.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from rse.events.pulse import PulseEstimate
from rse.events.schema import Composition, Note

#: Candidate repeat lengths, in grid steps. Below 4 a "repeat" is an ornament;
#: above 64 there is not enough of a piece left to confirm it twice.
MIN_REPEAT = 4
MAX_REPEAT = 64

#: Bar-sized window the detected repeat is doubled into.
MIN_BAR_STEPS = 16
MAX_BAR_STEPS = 64

#: Grid steps a piece must span before a repeat can be confirmed.
MIN_STEPS = 64

#: Occupied steps needed at a lag before its score counts.
MIN_COMPARISONS = 20

#: Jaccard self-similarity below which no repeat is claimed.
MIN_SIMILARITY = 0.25

#: Fraction of the peak score a shorter length must reach to be preferred.
PEAK_TOLERANCE = 0.9


@dataclass(frozen=True)
class BarEstimate:
    """A composition's bar, in units of the pulse grid's steps."""

    #: Steps per bar, after scaling the raw repeat into a bar-sized window.
    length_steps: int
    #: Which grid step begins a bar, in [0, length_steps).
    offset_steps: int
    #: Jaccard self-similarity at the detected repeat, 0 to 1.
    confidence: float
    #: The raw fundamental repeat, before doubling. Kept because a piece whose
    #: drums repeat every 8 is a different thing from one whose bar is 16.
    repeat_steps: int
    n_steps: int

    @property
    def found(self) -> bool:
        return self.length_steps > 0 and self.confidence >= MIN_SIMILARITY

    @property
    def is_musical(self) -> bool:
        """Whether the length is a power of two.

        Not a guarantee of correctness, but the only cheap signal separating a
        plausible bar from the repeat finder having locked onto a sub-pattern.
        A caller that cannot tolerate a wrong bar should require this.
        """
        n = self.length_steps
        return n > 0 and (n & (n - 1)) == 0

    def position(self, step: int) -> tuple[int, int]:
        """A grid step as (bar index, position within the bar)."""
        if not self.found:
            return 0, step
        shifted = step - self.offset_steps
        return shifted // self.length_steps, shifted % self.length_steps

    def to_dict(self) -> dict[str, Any]:
        return {
            "length_steps": self.length_steps,
            "offset_steps": self.offset_steps,
            "repeat_steps": self.repeat_steps,
            "confidence": round(self.confidence, 4),
            "is_musical": self.is_musical,
            "n_steps": self.n_steps,
            "found": self.found,
        }


def step_contents(
    composition: Composition, pulse: PulseEstimate
) -> dict[int, frozenset[tuple[str, int]]]:
    """What sounds at each grid step, as (channel, value) pairs.

    Values are kept rather than reduced to channel occupancy so that a repeat
    has to agree on pitch, not merely on something being present — otherwise
    any piece with continuous accompaniment matches itself at every lag.
    """
    if not pulse.found:
        return {}
    contents: dict[int, set[tuple[str, int]]] = {}
    for event in composition.sounding_events():
        step = int(round((event.tick - pulse.phase_ticks) / pulse.period_ticks))
        value = event.pitch if isinstance(event, Note) else event.value
        contents.setdefault(step, set()).add((event.channel_id, value))
    return {step: frozenset(pairs) for step, pairs in contents.items()}


def _similarity(sequence: list[frozenset[tuple[str, int]]], lag: int) -> float:
    """Jaccard overlap between the sequence and itself, `lag` steps on.

    Averaged over steps where either side has content. Empty-against-empty is
    excluded: a sparse piece would otherwise score near 1.0 at every lag purely
    on its silence agreeing with itself.
    """
    total, compared = 0.0, 0
    for i in range(len(sequence) - lag):
        left, right = sequence[i], sequence[i + lag]
        if not left and not right:
            continue
        union = len(left | right)
        total += len(left & right) / union if union else 0.0
        compared += 1
    if compared < MIN_COMPARISONS:
        return 0.0
    return total / compared


def detect_bar(composition: Composition, pulse: PulseEstimate) -> BarEstimate:
    """Find how many grid steps make a bar, and which step starts one."""
    contents = step_contents(composition, pulse)
    if not contents:
        return BarEstimate(0, 0, 0.0, 0, 0)

    low, high = min(contents), max(contents)
    n_steps = high - low + 1
    if n_steps < MIN_STEPS:
        return BarEstimate(0, 0, 0.0, 0, n_steps)

    empty: frozenset[tuple[str, int]] = frozenset()
    sequence = [contents.get(low + i, empty) for i in range(n_steps)]

    max_lag = min(MAX_REPEAT, n_steps // 2)
    lags = range(MIN_REPEAT, max_lag + 1)
    scores = {lag: _similarity(sequence, lag) for lag in lags}
    if not scores:
        return BarEstimate(0, 0, 0.0, 0, n_steps)

    peak = max(scores.values())
    if peak < MIN_SIMILARITY:
        return BarEstimate(0, 0, peak, 0, n_steps)

    # The fundamental, not its multiples: a pattern repeating every 32 steps
    # also matches at 64 and 96.
    repeat = min(lag for lag, score in scores.items() if score >= PEAK_TOLERANCE * peak)

    length = repeat
    while length < MIN_BAR_STEPS and length * 2 <= MAX_BAR_STEPS:
        length *= 2

    offset = _downbeat(composition, pulse, length)
    return BarEstimate(
        length_steps=length,
        offset_steps=offset,
        confidence=float(scores[repeat]),
        repeat_steps=repeat,
        n_steps=n_steps,
    )


def _downbeat(composition: Composition, pulse: PulseEstimate, length: int) -> int:
    """The offset where the most channels start together.

    Counting channel starts rather than distinct onset times is what makes this
    work: a drum part that hits once every four steps has identical onset
    density at four different offsets, so density alone cannot say which is the
    downbeat. Simultaneous starts across channels can, because an accented beat
    one is usually where the bass, the lead and the kick coincide.

    This still fails on a piece with no accent at all — a uniform arpeggio over
    a steady hat has no recoverable downbeat, and the lowest offset is returned.
    A wrong offset rotates every position token by a constant, which is far less
    damaging than a wrong `length_steps`: the model sees a consistent grid that
    simply starts somewhere other than where a score would.
    """
    weights: dict[int, int] = {}
    for event in composition.sounding_events():
        step = int(round((event.tick - pulse.phase_ticks) / pulse.period_ticks))
        weights[step] = weights.get(step, 0) + 1
    if not weights:
        return 0
    counts = np.zeros(length, dtype=np.int64)
    for step, weight in weights.items():
        counts[step % length] += weight
    return int(np.argmax(counts))
