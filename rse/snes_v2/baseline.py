"""The matched G3 baseline: interpolated n-gram with Witten-Bell smoothing.

Matched means **identical in everything except the model**: the same frozen G3
vocabulary, the same 1,384 training tracks, the same exported windows, the
same causal snapshot and rollover semantics, the same grammar mask, the same
sampling procedure, the same duration targets and the same renderer.

**No position-based slot assumptions.** Level 2's comparator exploited a fixed
stride-4 cycle and was handed the slot from absolute position, which made its
structural validity look like learning. G3 is variable-length and stateful, so
that shortcut does not exist here and is not reintroduced.

Witten-Bell, in its classic form, interpolating order `k` with order `k-1`:

    lambda(h) = N1+(h) / (c(h) + N1+(h))
    P(w|h)    = (1 - lambda(h)) * c(h,w)/c(h) + lambda(h) * P(w|h')

where `N1+(h)` is the number of distinct tokens observed after context `h`.
A context never seen falls straight through to the shorter one, so no
probability is lost and no zero mass can arise -- the failure that made a
Level 2 n-gram score worse than its own backoff.

Order is selected on **validation** scored targets with the registered mask,
never on training likelihood.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

#: Candidate orders, declared before selection.
ORDER_CANDIDATES = (1, 2, 3, 4, 5)


@dataclass
class G3NGram:
    """Counts for every order up to `order`, for interpolated backoff."""

    vocab_size: int
    order: int
    #: context tuple -> {token: count}
    counts: list[dict[tuple[int, ...], dict[int, int]]] = field(
        default_factory=list)
    #: context tuple -> total count
    totals: list[dict[tuple[int, ...], int]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.counts:
            self.counts = [defaultdict(dict) for _ in range(self.order + 1)]
            self.totals = [defaultdict(int) for _ in range(self.order + 1)]

    def fit(self, streams: Iterable[Sequence[int]]) -> "G3NGram":
        for stream in streams:
            for index, token in enumerate(stream):
                for k in range(self.order + 1):
                    if index < k:
                        continue
                    context = tuple(stream[index - k:index])
                    bucket = self.counts[k].setdefault(context, {})
                    bucket[token] = bucket.get(token, 0) + 1
                    self.totals[k][context] += 1
        return self

    def prob(self, context: Sequence[int], token: int) -> float:
        """Interpolated Witten-Bell probability, recursing down the orders."""
        # Order 0: uniform over the vocabulary, so nothing is ever zero.
        probability = 1.0 / self.vocab_size
        for k in range(0, self.order + 1):
            if k > len(context):
                break
            key = tuple(context[len(context) - k:]) if k else ()
            total = self.totals[k].get(key, 0)
            if not total:
                continue
            distinct = len(self.counts[k].get(key, {}))
            weight = distinct / (total + distinct)
            observed = self.counts[k].get(key, {}).get(token, 0) / total
            probability = (1.0 - weight) * observed + weight * probability
        return probability

    def distribution(self, context: Sequence[int]) -> list[float]:
        """Full distribution over the vocabulary. Sums to 1 by construction."""
        out = [0.0] * self.vocab_size
        uniform = 1.0 / self.vocab_size
        for token in range(self.vocab_size):
            out[token] = uniform
        for k in range(0, self.order + 1):
            if k > len(context):
                break
            key = tuple(context[len(context) - k:]) if k else ()
            total = self.totals[k].get(key, 0)
            if not total:
                continue
            bucket = self.counts[k].get(key, {})
            distinct = len(bucket)
            weight = distinct / (total + distinct)
            for token in range(self.vocab_size):
                observed = bucket.get(token, 0) / total
                out[token] = (1.0 - weight) * observed + weight * out[token]
        return out

    def memory_estimate(self) -> dict[str, int]:
        return {f"order_{k}_contexts": len(self.totals[k])
                for k in range(self.order + 1)}


def score(model: G3NGram, windows: Sequence[Sequence[int]],
          masks: Sequence[Sequence[bool]]) -> dict[str, Any]:
    """Bits per SCORED target, with the registered mask.

    The mask excludes padding and snapshot positions, exactly as the
    Transformer's loss does, so the two arms are scored on the same
    denominator. Level 1 showed how quietly a denominator can shift.
    """
    total_bits = 0.0
    n = 0
    for stream, mask in zip(windows, masks):
        for index in range(1, len(stream)):
            if not mask[index]:
                continue
            context = stream[max(0, index - model.order):index]
            probability = model.prob(context, stream[index])
            total_bits -= math.log2(max(probability, 1e-300))
            n += 1
    return {"bits_per_scored_target": total_bits / n if n else float("inf"),
            "n_scored": n}


# --- interpolated modified Kneser-Ney ------------------------------------

@dataclass
class G3KneserNey:
    """Interpolated modified Kneser-Ney over G3 tokens.

    Added by a prospective amendment after the Witten-Bell results were seen
    on the DEVELOPMENT population -- disclosed as such, because it is not
    prospective relative to baseline validation. The motivation is structural
    rather than cosmetic: Witten-Bell's interpolation weight for a singleton
    context is `1/(1+1) = 0.5`, which places half the mass on one observed
    continuation, and the order-5 model has 4,156,583 distinct contexts over
    about 9.1M positions. Modified KN instead *discounts* observed
    continuations and builds lower-order distributions from **continuation
    counts**, which is the standard remedy for exactly that failure.

    It is not guaranteed to win. Selection remains lowest validation loss
    across the two prespecified families.
    """

    vocab_size: int
    order: int
    raw: list[dict[tuple[int, ...], dict[int, int]]] = field(default_factory=list)
    totals: list[dict[tuple[int, ...], int]] = field(default_factory=list)
    #: Discounts D1, D2, D3+ per order, from training count-of-counts.
    discounts: list[tuple[float, float, float]] = field(default_factory=list)
    count_of_counts: list[dict[int, int]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.raw:
            self.raw = [dict() for _ in range(self.order + 1)]
            self.totals = [dict() for _ in range(self.order + 1)]

    def fit(self, streams: Iterable[Sequence[int]]) -> "G3KneserNey":
        # Raw counts for the highest order; every lower order is rebuilt from
        # CONTINUATION counts, which is what distinguishes KN from Witten-Bell.
        top = self.order
        for stream in streams:
            for index in range(top, len(stream)):
                context = tuple(stream[index - top:index])
                bucket = self.raw[top].setdefault(context, {})
                bucket[stream[index]] = bucket.get(stream[index], 0) + 1
        for k in range(top - 1, -1, -1):
            for context, bucket in self.raw[k + 1].items():
                shorter = context[1:]
                target = self.raw[k].setdefault(shorter, {})
                for token in bucket:
                    # Count DISTINCT predecessors, not occurrences.
                    target[token] = target.get(token, 0) + 1
        for k in range(top + 1):
            for context, bucket in self.raw[k].items():
                self.totals[k][context] = sum(bucket.values())
        # Discounts from count-of-counts, training data only.
        self.count_of_counts = []
        self.discounts = []
        for k in range(top + 1):
            coc: dict[int, int] = {}
            for bucket in self.raw[k].values():
                for value in bucket.values():
                    if value <= 4:
                        coc[value] = coc.get(value, 0) + 1
            self.count_of_counts.append(coc)
            n1, n2, n3, n4 = (coc.get(i, 0) for i in (1, 2, 3, 4))
            if n1 and n2:
                y = n1 / (n1 + 2 * n2)
                d1 = max(0.0, min(1.0, 1 - 2 * y * n2 / n1))
                d2 = max(0.0, min(2.0, 2 - 3 * y * n3 / n2)) if n3 else 0.75
                d3 = max(0.0, min(3.0, 3 - 4 * y * n4 / n3)) if n3 and n4 else 0.75
            else:
                d1 = d2 = d3 = 0.75          # standard fallback
            self.discounts.append((d1, d2, d3))
        return self

    def _discount(self, k: int, count: int) -> float:
        d1, d2, d3 = self.discounts[k]
        return d1 if count == 1 else d2 if count == 2 else d3

    def distribution(self, context: Sequence[int]) -> list[float]:
        out = [1.0 / self.vocab_size] * self.vocab_size
        for k in range(0, self.order + 1):
            if k > len(context):
                break
            key = tuple(context[len(context) - k:]) if k else ()
            total = self.totals[k].get(key, 0)
            if not total:
                continue
            bucket = self.raw[k].get(key, {})
            gamma = sum(self._discount(k, c) for c in bucket.values()) / total
            lower = out
            out = [gamma * lower[t] for t in range(self.vocab_size)]
            for token, count in bucket.items():
                out[token] += max(count - self._discount(k, count), 0.0) / total
        return out

    def prob(self, context: Sequence[int], token: int) -> float:
        return self.distribution(context)[token]

    def memory_estimate(self) -> dict[str, int]:
        return {f"order_{k}_contexts": len(self.totals[k])
                for k in range(self.order + 1)}


def score_constrained(model, windows: Sequence[Sequence[int]],
                      positions: Sequence[tuple[int, int]],
                      legal_sets: Sequence[frozenset[int]]) -> dict[str, Any]:
    """NLL under the GRAMMAR-CONSTRAINED distribution used for generation.

        P(y | c, legal) = P(y | c) / sum over legal of P(v | c)

    The same causal state supplies the mask for every model, so no arm is
    advantaged by the harness. Unconstrained loss is reported alongside for
    reference but selection uses this convention.
    """
    bits_c = bits_u = 0.0
    n = 0
    per_target: list[float] = []
    for (window_index, position), legal in zip(positions, legal_sets):
        stream = windows[window_index]
        target = stream[position]
        context = stream[max(0, position - model.order):position]
        dist = model.distribution(context)
        raw = max(dist[target], 1e-300)
        mass = sum(dist[v] for v in legal) or 1e-300
        value = -math.log2(max(raw / mass, 1e-300))
        per_target.append(value)
        bits_c += value
        bits_u += -math.log2(raw)
        n += 1
    mean = bits_c / n if n else float("inf")
    variance = sum((x - mean) ** 2 for x in per_target) / max(1, n - 1)
    return {"bits_per_scored_target": mean,
            "standard_error": (variance / n) ** 0.5 if n else None,
            "unconstrained_bits": bits_u / n if n else None,
            "n_scored": n,
            "median_legal_set": sorted(len(s) for s in legal_sets)[n // 2]
                                if n else None}
