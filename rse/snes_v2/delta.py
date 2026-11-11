"""Exact delta encoding for the RSE v2 token layer.

The 4000 Hz codec certifies onset error at 0.125 ms. **The token layer must not
spend that**, so every delta encoding here is exact: a delta of 24,585 ticks
decodes to exactly 24,585 ticks. No bucketing, no coarse escape for long rests.

The Level 1 representation is the cautionary case. Its duration buckets decoded
to upper edges, so one note in 32 became 141 seconds, and the defect survived
until a round-trip overlap measurement exposed it. A lossy token layer sitting
on a lossless codec would reproduce that class of bug exactly.

Two designs, both exact:

- **C1, exact vocabulary** — one token per delta observed in training.
- **C2, exact hybrid** — the `k` most frequent deltas get one token each; the
  rest are an escape plus radix-256 digits.

**Both need the escape**, because a vocabulary fitted on training data cannot
cover validation, test, or generated performances. An unseen delta must encode
exactly rather than fall back to a nearest neighbour. So the designs differ
only in `k` and in how often the escape fires.

Escape framing is **length-prefixed**: `ESCAPE_n` followed by exactly `n`
digit tokens, most significant first. That makes an incomplete escape
detectable -- a generated stream ending mid-escape raises rather than
silently decoding to zero or to an arbitrary duration, which matters because
Level 2's generator did stop mid-event and the harness had to notice.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence

#: Radix for escape digits. 256 keeps two digits sufficient for any delta up
#: to 65,535 ticks, which is 16.4 seconds at 4000 Hz.
RADIX = 256

#: Maximum escape length supported. Three digits reach 16,777,215 ticks, about
#: 70 minutes at 4000 Hz -- longer than any track in the corpus.
MAX_ESCAPE_DIGITS = 3


class NonPositiveDelta(ValueError):
    """An escape decoded to zero or less.

    **This is the clock-stagnation guard at the token layer.** The grouped-event
    codec removed `TIME_SAME` so that stalling is unrepresentable, and a
    variable-length escape must not reintroduce it: `ESCAPE_1` followed by
    digit 0 is a well-formed token pair that would otherwise decode to a delta
    of zero and let a generator emit groups forever without advancing time.
    Rejected rather than clamped to 1, so a generator that produces it is
    reported as invalid instead of silently corrected.
    """


class IncompleteDelta(ValueError):
    """A delta escape ran off the end of the stream.

    Raised rather than returning a default. A truncated escape is missing
    information, and inventing a duration for it would be the same failure as
    decoding a duration bucket to its upper edge.
    """


@dataclass(frozen=True)
class DeltaCodec:
    """Fitted on TRAINING deltas only. Unseen values still encode exactly."""

    #: Delta value -> direct token index, for the frequent head.
    direct: dict[int, int]
    #: Number of distinct deltas seen in training, for reporting.
    n_observed: int

    @property
    def n_direct(self) -> int:
        return len(self.direct)

    @property
    def vocab_size(self) -> int:
        """Direct tokens + one escape per length + the digit alphabet."""
        return self.n_direct + MAX_ESCAPE_DIGITS + RADIX

    def encode(self, delta: int) -> list[int]:
        """Token indices for one delta. Exact for every positive integer."""
        if delta <= 0:
            raise ValueError(f"delta must be positive, got {delta}")
        token = self.direct.get(delta)
        if token is not None:
            return [token]
        digits: list[int] = []
        value = delta
        while value:
            digits.append(value % RADIX)
            value //= RADIX
        digits.reverse()
        if len(digits) > MAX_ESCAPE_DIGITS:
            raise ValueError(f"delta {delta} exceeds {MAX_ESCAPE_DIGITS} digits")
        escape = self.n_direct + (len(digits) - 1)
        base = self.n_direct + MAX_ESCAPE_DIGITS
        return [escape] + [base + d for d in digits]

    def decode(self, tokens: Sequence[int], index: int = 0) -> tuple[int, int]:
        """Return (delta, next_index). Raises on a truncated escape."""
        if index >= len(tokens):
            raise IncompleteDelta("no tokens")
        head = tokens[index]
        if head < self.n_direct:
            for value, token in self.direct.items():
                if token == head:
                    return value, index + 1
            raise ValueError(f"token {head} is not a known direct delta")
        if head >= self.n_direct + MAX_ESCAPE_DIGITS:
            raise ValueError(f"token {head} is a digit, not a delta start")
        n_digits = head - self.n_direct + 1
        base = self.n_direct + MAX_ESCAPE_DIGITS
        if index + n_digits >= len(tokens) + 0:
            if index + n_digits > len(tokens) - 1:
                raise IncompleteDelta(
                    f"escape wants {n_digits} digits, {len(tokens) - index - 1} present")
        value = 0
        for offset in range(1, n_digits + 1):
            digit = tokens[index + offset]
            if digit < base or digit >= base + RADIX:
                raise IncompleteDelta(
                    f"expected a digit at {index + offset}, got token {digit}")
            value = value * RADIX + (digit - base)
        if value <= 0:
            raise NonPositiveDelta(
                f"escape decoded to {value}; a delta must advance time")
        return value, index + n_digits + 1

    def is_canonical(self, tokens: Sequence[int]) -> bool:
        """Is this the encoding `encode` would produce?

        A value can be written with leading zero digits -- `ESCAPE_2 [0, 5]`
        and `ESCAPE_1 [5]` both decode to 5 -- so decoding is unambiguous but
        the mapping is not bijective. Non-canonical forms decode correctly and
        are NOT rejected; they are detectable so a generator emitting them is
        visible rather than silently normalised.
        """
        value, consumed = self.decode(tokens, 0)
        return list(tokens[:consumed]) == self.encode(value)

    #: Worst-case tokens for one delta: the escape plus its digit budget. The
    #: budget is finite BY CONSTRUCTION, so no delta can consume unbounded
    #: tokens without advancing time.
    MAX_TOKENS_PER_DELTA = 1 + MAX_ESCAPE_DIGITS

    def n_tokens(self, delta: int) -> int:
        """Token cost without materialising the encoding."""
        if delta in self.direct:
            return 1
        digits = 0
        value = delta
        while value:
            digits += 1
            value //= RADIX
        return 1 + digits


def fit(deltas: Iterable[int], *, k: int | None = None) -> DeltaCodec:
    """Fit on training deltas.

    `k=None` gives C1: every observed delta gets a direct token. An integer
    `k` gives C2: only the `k` most frequent do, the tail escaping exactly.
    Ties are broken by delta value so the fit is deterministic.
    """
    counts = Counter(deltas)
    ordered = sorted(counts, key=lambda d: (-counts[d], d))
    chosen = ordered if k is None else ordered[:k]
    return DeltaCodec(direct={d: i for i, d in enumerate(chosen)},
                      n_observed=len(counts))
