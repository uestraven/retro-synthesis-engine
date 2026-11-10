"""Autoregressive generation from the locked checkpoint, and from the comparator.

Level 2 of the pre-registered experiment. The model is NOT retrained and the
Level 1 conclusion is not revisited.

**The central discipline: sampling parameters are fixed WITHOUT LISTENING.**
A generator tuned by ear against its own output is a generator tuned to the
thing Level 2 is supposed to measure. So the selection criterion is
distributional, declared in advance, and computed on a development subset of
the TRAINING split:

    choose the sampling parameters whose generated output best matches the
    training corpus on three registered statistics -- attacks per second,
    mean simultaneous voices, and the distribution of melodic intervals --
    scored as the sum of absolute standardised differences.

No test data is used. No audio is heard. The chosen parameters are recorded
before any excerpt is rendered.

**Validity is measured, not enforced.** The registered criterion asks for
>=99% structurally valid sequences, which is a measurement of what the models
produce. Constraining sampling to the legal slot at each position would make
the criterion vacuous and would also flatter whichever model had learned the
grammar less well. Generation is therefore unconstrained and validity is
reported.

**Both arms are generated identically**: same durations, same seeds, same
number of excerpts, same unconstrained sampling, same rendering. Excerpts are
taken as generated, never selected for sounding better.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from rse.snes_lm import representation as rep
from rse.snes_lm.baselines import Model as BaselineModel
from rse.snes_lm.baselines import _slot

#: Registered excerpt length. Seconds rather than bars: the representation does
#: not guarantee a universal meter, and the release's own onset-grid scan found
#: a plausible row grid for only about half of tracks, so "16 bars" is not
#: well defined corpus-wide.
EXCERPT_SECONDS = 25.0

#: Hard cap on generated events, so a degenerate sample cannot run away.
MAX_EVENTS = 4_000


@dataclass
class SamplingConfig:
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    seed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "seed": self.seed,
        }


@dataclass
class Excerpt:
    """One generated excerpt, with everything needed to judge it."""

    arm: str
    seed: int
    tokens: list[int]
    events: list[dict[str, Any]] = field(default_factory=list)
    valid: bool = False
    invalid_reason: str = ""
    duration_sec: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "seed": self.seed,
            "n_tokens": len(self.tokens),
            "n_events": len(self.events),
            "valid": self.valid,
            "invalid_reason": self.invalid_reason,
            "duration_sec": self.duration_sec,
        }


def _filter_logits(
    logits: torch.Tensor, config: SamplingConfig
) -> torch.Tensor:
    """Temperature, then top-k, then nucleus. Order matters and is fixed."""
    if config.temperature <= 0:
        raise ValueError("temperature must be positive")
    logits = logits / config.temperature
    if config.top_k and config.top_k < logits.shape[-1]:
        threshold = torch.topk(logits, config.top_k).values[..., -1, None]
        logits = logits.masked_fill(logits < threshold, float("-inf"))
    if config.top_p < 1.0:
        ordered, index = torch.sort(logits, descending=True)
        cumulative = torch.cumsum(F.softmax(ordered, dim=-1), dim=-1)
        remove = cumulative - F.softmax(ordered, dim=-1) > config.top_p
        remove[..., 0] = False  # always keep the most likely token
        ordered = ordered.masked_fill(remove, float("-inf"))
        logits = torch.empty_like(logits).scatter_(-1, index, ordered)
    return logits


@torch.no_grad()
def generate_model(
    model: torch.nn.Module,
    vocab: rep.Vocab,
    config: SamplingConfig,
    *,
    seconds: float = EXCERPT_SECONDS,
    max_events: int = MAX_EVENTS,
    context: int = 1024,
) -> Excerpt:
    """Sample from the Transformer, starting from BOS alone.

    Unconditioned: the excerpt is the model's own idea of how a piece starts,
    not a continuation of real music. A continuation would make originality
    and coherence far harder to attribute.
    """
    model.eval()
    generator = torch.Generator().manual_seed(config.seed)
    tokens = [vocab.BOS]
    clock = 0.0
    events_emitted = 0

    stop = False
    while not stop and events_emitted < max_events:
        window = tokens[-context:]
        logits = model(torch.tensor([window], dtype=torch.long))[0, -1, :]
        probabilities = F.softmax(_filter_logits(logits, config), dim=-1)
        token = int(torch.multinomial(probabilities, 1, generator=generator))
        if token == vocab.EOS:
            break
        tokens.append(token)
        position = len(tokens) - 1
        if _slot(position) == 0 and vocab.group(token) == "time":
            offset = token - vocab.time_base
            clock += 0.0 if offset == 0 else vocab.time_value(offset - 1)
        # STOP ONLY AT AN EVENT BOUNDARY. Cutting the moment the clock passes
        # the target truncated mid-event and produced sequences that failed the
        # structural-validity criterion for the HARNESS's reason rather than
        # the model's -- a 241-token body, not a multiple of the stride. The
        # criterion measures what the model emits, so the harness must not be
        # the thing that breaks it.
        if _slot(position) == 3:
            events_emitted += 1
            stop = clock >= seconds

    tokens.append(vocab.EOS)
    return _finish("model", config.seed, tokens, vocab)


def generate_baseline(
    model: BaselineModel,
    vocab: rep.Vocab,
    config: SamplingConfig,
    *,
    seconds: float = EXCERPT_SECONDS,
    max_events: int = MAX_EVENTS,
) -> Excerpt:
    """Sample from the comparator, by the same procedure.

    The comparator exposes log-probabilities rather than logits, so its
    distribution is built by evaluating every token id. That is 240 evaluations
    per step, which is slow and exactly correct -- the alternative would be a
    different sampling procedure for the two arms.
    """
    rng = random.Random(config.seed)
    tokens = [vocab.BOS]
    clock = 0.0
    events_emitted = 0

    stop = False
    while not stop and events_emitted < max_events:
        logprobs = [model.logprob(tokens, t) for t in range(vocab.size)]
        scaled = [lp / config.temperature for lp in logprobs]
        peak = max(scaled)
        weights = [math.exp(value - peak) for value in scaled]
        token = rng.choices(range(vocab.size), weights=weights, k=1)[0]
        if token in (vocab.EOS, vocab.PAD):
            break
        tokens.append(token)
        position = len(tokens) - 1
        if _slot(position) == 0 and vocab.group(token) == "time":
            offset = token - vocab.time_base
            clock += 0.0 if offset == 0 else vocab.time_value(offset - 1)
        if _slot(position) == 3:  # event boundary; see generate_model
            events_emitted += 1
            stop = clock >= seconds

    tokens.append(vocab.EOS)
    return _finish("baseline", config.seed, tokens, vocab)


def _finish(arm: str, seed: int, tokens: list[int], vocab: rep.Vocab) -> Excerpt:
    valid, reason = rep.structurally_valid(tokens, vocab)
    events = rep.decode(tokens, vocab) if valid else []
    duration = max((e["onset_sec"] for e in events), default=0.0)
    return Excerpt(
        arm=arm, seed=seed, tokens=tokens, events=events,
        valid=valid, invalid_reason="" if valid else reason,
        duration_sec=duration,
    )


# --- the declared, listening-free parameter selection ---------------------


@dataclass
class CorpusStatistics:
    """The three registered statistics, computed identically everywhere."""

    attacks_per_second: float
    mean_simultaneous_voices: float
    interval_histogram: dict[int, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "attacks_per_second": self.attacks_per_second,
            "mean_simultaneous_voices": self.mean_simultaneous_voices,
            "interval_histogram": {str(k): v for k, v in
                                   sorted(self.interval_histogram.items())},
        }


def statistics(events: Sequence[dict[str, Any]], *, clip: int = 12) -> CorpusStatistics:
    """Attacks/sec, mean simultaneity, and the melodic interval distribution.

    Simultaneity is counted per distinct onset, which the corpus licenses: a
    full-population check found zero same-voice retriggers in 1,527,546 ticks,
    so events sharing an onset are always distinct voices.

    Intervals are per voice, between consecutive pitched attacks, clipped to
    +/- `clip` semitones. Clipping keeps a handful of octave-error outliers
    from dominating a histogram comparison.
    """
    if not events:
        return CorpusStatistics(0.0, 0.0, {})
    span = max(e["onset_sec"] for e in events) or 1e-9
    onsets: dict[float, int] = {}
    for event in events:
        onsets[round(float(event["onset_sec"]), 6)] = (
            onsets.get(round(float(event["onset_sec"]), 6), 0) + 1
        )
    by_voice: dict[int, list[float]] = {}
    for event in events:
        note = event.get("midi_note")
        if note is None:
            continue
        by_voice.setdefault(event["voice"], []).append(float(note))
    counts: dict[int, int] = {}
    total = 0
    for pitches in by_voice.values():
        for a, b in zip(pitches, pitches[1:]):
            step = max(-clip, min(clip, int(round(b - a))))
            counts[step] = counts.get(step, 0) + 1
            total += 1
    histogram = {k: v / total for k, v in counts.items()} if total else {}
    return CorpusStatistics(
        attacks_per_second=len(events) / span,
        mean_simultaneous_voices=sum(onsets.values()) / len(onsets),
        interval_histogram=histogram,
    )


def distance(candidate: CorpusStatistics, reference: CorpusStatistics) -> float:
    """Sum of absolute standardised differences on the three statistics.

    The declared selection criterion. Lower is a closer match to the corpus.
    The interval term is total variation distance between the histograms,
    which is already on a 0-1 scale and needs no standardisation.
    """
    if reference.attacks_per_second <= 0:
        return float("inf")
    density = abs(
        candidate.attacks_per_second - reference.attacks_per_second
    ) / reference.attacks_per_second
    simultaneity = abs(
        candidate.mean_simultaneous_voices - reference.mean_simultaneous_voices
    ) / max(reference.mean_simultaneous_voices, 1e-9)
    keys = set(candidate.interval_histogram) | set(reference.interval_histogram)
    variation = 0.5 * sum(
        abs(
            candidate.interval_histogram.get(k, 0.0)
            - reference.interval_histogram.get(k, 0.0)
        )
        for k in keys
    )
    return density + simultaneity + variation
