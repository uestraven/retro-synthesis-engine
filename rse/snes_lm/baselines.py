"""Matched statistical baselines, in the units the Transformer optimises.

Every baseline here predicts the SAME registered target tokens under the SAME
masking policy as the model, and reports per-token cross-entropy in
bits/token. That is the point: a baseline measured in different units cannot
establish the 5% threshold the experiment registered.

The existing 27-game interval n-gram is NOT one of these. It is per-interval
over a melody-only population, so it is useful context and not a comparator.

**Deterministic heuristics need a probabilistic formulation.** A rule that
names one token predicts it with probability 1 and incurs infinite
cross-entropy the moment it is wrong, which makes it unscorable rather than
bad. Each heuristic here is therefore mixed with a backoff distribution,

    P(x) = lam * P_heuristic(x) + (1 - lam) * P_backoff(x)

with `lam` selected on VALIDATION ONLY. That is a proper distribution, finite
everywhere, and it measures what the heuristic contributes over the backoff
rather than how often it is exactly right.

**The backoff is a stride-aware unigram, not a flat one.** The token stream has
a fixed TIME, VOICE, PITCH, DUR cycle, so knowing that position 2 mod 4 must
be a pitch is structural information the model gets for free from the grammar.
A baseline denied it would flatter the model by an amount that has nothing to
do with music. Both are reported: the naive unigram shows what the structure
alone is worth.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from rse.snes_lm.representation import STRIDE, Vocab

#: Natural log of 2, for converting nats to bits once rather than per token.
_LN2 = math.log(2.0)


@dataclass
class Score:
    """A cross-entropy measurement over a population."""

    name: str
    bits_per_token: float
    n_tokens: int
    #: Per-track bits/token, so uncertainty can be bootstrapped over tracks.
    per_track: list[float] = field(default_factory=list)
    #: Per-game bits/token, for the descriptive per-game report.
    per_game: dict[str, float] = field(default_factory=dict)
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "bits_per_token": self.bits_per_token,
            "n_tokens": self.n_tokens,
            "n_tracks": len(self.per_track),
            "per_game": dict(sorted(self.per_game.items())),
            "detail": self.detail,
        }


class Model:
    """A scorable next-token distribution over the registered token stream."""

    name = "model"

    def logprob(self, history: Sequence[int], target: int) -> float:
        """Natural log probability of `target` given `history`. Must be finite."""
        raise NotImplementedError

    def stream(self, body: Sequence[int]) -> list[float]:
        """Log probabilities for every target in one track, in order.

        Exists so a model can keep incremental state. The default re-derives
        history each step, which is fine for models whose `logprob` only reads
        the last few tokens; `IntervalReversal` overrides it because rebuilding
        its pitch history per token is quadratic in track length, and tracks
        reach 57,510 tokens.
        """
        return [
            self.logprob(body[:index], body[index]) for index in range(1, len(body))
        ]


def _slot(index: int) -> int:
    """Which stride position a target at `index` occupies.

    `index` is the position in the body INCLUDING the leading BOS, which is
    how `score` indexes targets. So the first real token sits at index 1 and
    must map to slot 0 (TIME).

    This convention is load-bearing. An earlier version counted slots from a
    BOS-excluded body while scoring a BOS-included one, rotating every slot by
    one. The stride-aware unigram then predicted the wrong token group at
    every position and scored 18.0 bits/token against the flat unigram's
    6.2 -- impossible, since it has strictly more information, which is how the
    bug surfaced. Any change here must keep `fit` and `logprob` on the same
    convention.
    """
    return (index - 1) % STRIDE


class StrideUnigram(Model):
    """Token frequency conditioned on the stride slot. The honest floor.

    Also serves as the backoff for every heuristic, so the mixtures measure a
    musical contribution rather than re-measuring token frequency.
    """

    name = "unigram_stride_aware"

    def __init__(self, vocab: Vocab, alpha: float = 0.5) -> None:
        self.vocab = vocab
        self.alpha = alpha
        self.counts: list[dict[int, int]] = [defaultdict(int) for _ in range(STRIDE)]
        self.totals = [0] * STRIDE
        self.eos: dict[int, int] = defaultdict(int)
        self.eos_total = 0
        self._support = self._slot_support(vocab)

    @staticmethod
    def _slot_support(vocab: Vocab) -> list[int]:
        """How many token ids are legal in each slot, for the smoothing mass."""
        return [vocab.n_time, 8, vocab.n_pitch, vocab.n_dur]

    def fit(self, sequences: Iterable[Sequence[int]]) -> "StrideUnigram":
        for tokens in sequences:
            body = [t for t in tokens if t != self.vocab.PAD]
            for index in range(1, len(body)):  # same indexing as `score`
                token = body[index]
                slot = _slot(index)
                if token == self.vocab.EOS:
                    self.eos[slot] += 1
                    self.eos_total += 1
                    continue
                self.counts[slot][token] += 1
                self.totals[slot] += 1
        return self

    def logprob(self, history: Sequence[int], target: int) -> float:
        index = len(history)
        slot = _slot(index)
        support = self._support[slot] + 1  # +1 for EOS, legal at any slot
        if target == self.vocab.EOS:
            numerator = self.eos[slot] + self.alpha
        else:
            numerator = self.counts[slot].get(target, 0) + self.alpha
        denominator = self.totals[slot] + self.eos[slot] + self.alpha * support
        return math.log(numerator / denominator)


class FlatUnigram(StrideUnigram):
    """Token frequency ignoring stride position.

    Reported to show what the grammar alone is worth: the gap between this and
    the stride-aware unigram is structure, not music.
    """

    name = "unigram_flat"

    def fit(self, sequences: Iterable[Sequence[int]]) -> "FlatUnigram":
        for tokens in sequences:
            body = [t for t in tokens if t != self.vocab.PAD]
            for index in range(1, len(body)):
                token = body[index]
                if token == self.vocab.EOS:
                    for slot in range(STRIDE):
                        self.eos[slot] += 1
                    self.eos_total += 1
                    continue
                for slot in range(STRIDE):
                    self.counts[slot][token] += 1
                    self.totals[slot] += 1
        return self

    @staticmethod
    def _slot_support(vocab: Vocab) -> list[int]:
        return [vocab.size] * STRIDE


class NGram(Model):
    """Interpolated n-gram over the token stream.

    Orders are interpolated rather than backed off, with a single weight per
    order selected on validation. Counts never cross a track boundary, matching
    the dataloader policy: a model should not learn a transition between
    unrelated pieces, and neither should its baseline.
    """

    def __init__(
        self, vocab: Vocab, order: int, alpha: float = 0.5, discount: float = 1.0
    ) -> None:
        self.vocab = vocab
        self.order = order
        self.alpha = alpha
        #: Witten-Bell discount; selected on validation.
        self.discount = discount
        self.name = f"ngram_order{order}_d{discount}"
        # context tuple -> {token: count}
        self.tables: list[dict[tuple[int, ...], dict[int, int]]] = [
            defaultdict(lambda: defaultdict(int)) for _ in range(order + 1)
        ]
        self.unigram = StrideUnigram(vocab, alpha=alpha)

    def fit(self, sequences: Iterable[Sequence[int]]) -> "NGram":
        seqs = list(sequences)
        self.unigram.fit(seqs)
        for tokens in seqs:
            body = [t for t in tokens if t != self.vocab.PAD]
            for index in range(1, len(body)):
                target = body[index]
                for k in range(1, self.order + 1):
                    if index - k < 0:
                        break
                    context = tuple(body[index - k : index])
                    self.tables[k][context][target] += 1
        return self

    def logprob(self, history: Sequence[int], target: int) -> float:
        """Witten-Bell interpolation from the stride-aware unigram upward.

        The weight on each order is `total / (total + distinct)` -- how
        concentrated that context's continuations are -- so a context seen
        many times with few distinct followers is trusted and a sparse one is
        not. Crucially the lower order ALWAYS retains mass, so an order that
        has never seen this target cannot drive the probability to zero.

        An earlier version used `total / (total + alpha*100)`, which trusted
        any frequently-seen context regardless of how many continuations it
        had. When such a context assigned the target zero count the estimate
        collapsed onto 9% of the backoff, and the order-3 n-gram scored 6.274
        bits/token against its own backoff's 4.209 -- impossible for an
        interpolated model, which is how the bug surfaced.
        """
        probability = math.exp(self.unigram.logprob(history, target))
        for k in range(1, self.order + 1):
            if len(history) < k:
                break
            context = tuple(history[-k:])
            table = self.tables[k].get(context)
            if not table:
                continue
            total = sum(table.values())
            distinct = len(table)
            weight = total / (total + self.discount * distinct)
            probability = weight * (table.get(target, 0) / total) + (
                1.0 - weight
            ) * probability
        return math.log(max(probability, 1e-12))


class Heuristic(Model):
    """A deterministic rule, made scorable by mixing with a backoff.

    `predict` returns the token the rule names, or None when it does not
    apply. The mixture weight is selected on validation.
    """

    def __init__(self, vocab: Vocab, backoff: Model, lam: float, name: str) -> None:
        self.vocab = vocab
        self.backoff = backoff
        self.lam = lam
        self.name = name

    def predict(self, history: Sequence[int]) -> int | None:
        raise NotImplementedError

    def logprob(self, history: Sequence[int], target: int) -> float:
        base = math.exp(self.backoff.logprob(history, target))
        guess = self.predict(history)
        if guess is None:
            return math.log(max(base, 1e-12))
        point = 1.0 if guess == target else 0.0
        mixed = self.lam * point + (1.0 - self.lam) * base
        return math.log(max(mixed, 1e-12))


class CopyPreviousEvent(Heuristic):
    """Predict the token four positions back: the same slot, previous event.

    This is the repetition control. Game music repeats heavily, and a model
    that only learns "the next bar looks like the last one" should not be
    mistaken for one that learned structure.
    """

    def predict(self, history: Sequence[int]) -> int | None:
        if len(history) < STRIDE:
            return None
        return history[-STRIDE]


class CopyAtLag(Heuristic):
    """Predict the token `lag` positions back, with `lag` chosen on validation.

    A lag that is a multiple of the stride copies a whole earlier event, so a
    well-chosen lag captures bar-level or phrase-level repetition.
    """

    def __init__(self, vocab, backoff, lam, lag: int) -> None:
        super().__init__(vocab, backoff, lam, name=f"copy_lag{lag}")
        self.lag = lag

    def predict(self, history: Sequence[int]) -> int | None:
        if len(history) < self.lag:
            return None
        return history[-self.lag]


class IntervalReversal(Heuristic):
    """For pitch slots, predict the previous pitch minus the previous interval.

    The music-specific heuristic: melodies turn around. Scored 36.72% in the
    27-game interval experiment, well above the frequency floor, which is why
    it belongs here -- beating a frequency baseline is not evidence of
    learning structure.

    Applies only at pitch slots and only when the two previous pitches were
    both real semitones; it declines elsewhere and the backoff carries it.
    """

    name = "interval_reversal"

    def __init__(self, vocab, backoff, lam) -> None:
        super().__init__(vocab, backoff, lam, name="interval_reversal")

    def _guess(self, previous: int | None, before: int | None) -> int | None:
        if previous is None or before is None:
            return None
        guess = previous - (previous - before)
        return guess if self.vocab.pitch_base <= guess < self.vocab.pitch_none else None

    def predict(self, history: Sequence[int]) -> int | None:
        if _slot(len(history)) != 2:
            return None
        pitches = [
            t for t in history if self.vocab.pitch_base <= t < self.vocab.pitch_none
        ]
        if len(pitches) < 2:
            return None
        return self._guess(pitches[-1], pitches[-2])

    def stream(self, body: Sequence[int]) -> list[float]:
        """Incremental: carry the last two real pitches instead of rescanning."""
        out: list[float] = []
        previous = before = None
        for index in range(1, len(body)):
            target = body[index]
            base = math.exp(self.backoff.logprob(body[:index], target))
            guess = self._guess(previous, before) if _slot(index) == 2 else None
            if guess is None:
                out.append(math.log(max(base, 1e-12)))
            else:
                mixed = self.lam * (1.0 if guess == target else 0.0) + (1.0 - self.lam) * base
                out.append(math.log(max(mixed, 1e-12)))
            token = body[index]
            if self.vocab.pitch_base <= token < self.vocab.pitch_none:
                before, previous = previous, token
        return out


def score(
    model: Model,
    sequences: Iterable[tuple[str, str, Sequence[int]]],
    vocab: Vocab,
) -> Score:
    """Cross-entropy in bits/token over the registered targets.

    Targets are every non-pad token after BOS, which is the same accounting the
    model's masked loss uses. Per-track and per-game figures are retained so
    uncertainty can be bootstrapped over tracks and the eight test games can be
    reported descriptively.
    """
    total_bits = 0.0
    total_tokens = 0
    per_track: list[float] = []
    game_bits: dict[str, float] = defaultdict(float)
    game_tokens: dict[str, int] = defaultdict(int)

    for track_id, game, tokens in sequences:
        body = [t for t in tokens if t != vocab.PAD]
        logps = model.stream(body)
        bits = -sum(logps) / _LN2
        count = len(logps)
        if count:
            total_bits += bits
            total_tokens += count
            per_track.append(bits / count)
            game_bits[game] += bits
            game_tokens[game] += count

    return Score(
        name=model.name,
        bits_per_token=total_bits / total_tokens if total_tokens else float("inf"),
        n_tokens=total_tokens,
        per_track=per_track,
        per_game={g: game_bits[g] / game_tokens[g] for g in game_bits},
    )


class SlotNGram(Model):
    """An n-gram conditioned WITHIN slot, plus the current event's prefix.

    Why this exists. A naive n-gram over the raw token stream conditions on
    whatever happens to precede a position, and on a fixed-stride TIME VOICE
    PITCH DUR serialisation that is usually the wrong thing: the order-1
    context for a pitch is the voice token beside it. Measured on validation,
    every order of the naive n-gram came out WORSE than its own stride-aware
    unigram backoff, monotonically with order -- 4.29 at order 1 rising to 6.11
    at order 4, against the backoff's 4.23. The model was being asked to
    predict a pitch from a voice id.

    The 27-game interval n-gram that beat its frequency baseline by 10.9
    accuracy points ran over a pure interval sequence, where consecutive
    elements are musically adjacent. This recovers that property: the context
    for a target is the previous `order` tokens OF THE SAME SLOT -- positions
    -4, -8, -12 -- optionally preceded by the tokens already emitted for the
    current event, which are genuinely informative (a voice constrains which
    pitch follows).

    That the interleaving defeats a naive n-gram is worth knowing for the
    Transformer too: it has to learn to attend four positions back, and a
    fixed stride at least makes that a learnable constant offset.
    """

    def __init__(
        self,
        vocab: Vocab,
        order: int = 2,
        discount: float = 1.0,
        use_event_prefix: bool = True,
    ) -> None:
        self.vocab = vocab
        self.order = order
        self.discount = discount
        self.use_event_prefix = use_event_prefix
        self.name = (
            f"slot_ngram_order{order}_d{discount}"
            f"{'_prefix' if use_event_prefix else ''}"
        )
        self.tables: list[dict[tuple[int, ...], dict[int, int]]] = [
            defaultdict(lambda: defaultdict(int)) for _ in range(order + 1)
        ]
        self.unigram = StrideUnigram(vocab, alpha=0.5)

    def _contexts(self, body: Sequence[int], index: int) -> list[tuple[int, ...]]:
        """Progressively longer contexts for the target at `index`."""
        slot = _slot(index)
        same_slot: list[int] = []
        position = index - STRIDE
        while position >= 1 and len(same_slot) < self.order:
            same_slot.append(body[position])
            position -= STRIDE
        prefix: tuple[int, ...] = ()
        if self.use_event_prefix and slot > 0:
            start = index - slot
            if start >= 1:
                prefix = tuple(body[start:index])
        out: list[tuple[int, ...]] = []
        for k in range(1, len(same_slot) + 1):
            out.append((slot,) + tuple(reversed(same_slot[:k])) + prefix)
        return out

    def fit(self, sequences: Iterable[Sequence[int]]) -> "SlotNGram":
        seqs = list(sequences)
        self.unigram.fit(seqs)
        for tokens in seqs:
            body = [t for t in tokens if t != self.vocab.PAD]
            for index in range(1, len(body)):
                target = body[index]
                for k, context in enumerate(self._contexts(body, index), start=1):
                    self.tables[k][context][target] += 1
        return self

    def stream(self, body: Sequence[int]) -> list[float]:
        out: list[float] = []
        for index in range(1, len(body)):
            target = body[index]
            probability = math.exp(self.unigram.logprob(body[:index], target))
            for k, context in enumerate(self._contexts(body, index), start=1):
                table = self.tables[k].get(context)
                if not table:
                    continue
                total = sum(table.values())
                weight = total / (total + self.discount * len(table))
                probability = weight * (table.get(target, 0) / total) + (
                    1.0 - weight
                ) * probability
            out.append(math.log(max(probability, 1e-12)))
        return out

    def logprob(self, history: Sequence[int], target: int) -> float:
        return self.stream(list(history) + [target])[-1]
