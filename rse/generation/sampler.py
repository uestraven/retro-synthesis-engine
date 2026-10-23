"""Sampling from a trained model.

Proposal 6.6 asks for temperature, top-k/top-p, a seed, target length, tempo
and hardware profile at inference time. These are sampling controls, not
training hyperparameters, and they live here rather than in the trainer.

The sampler is grammar-constrained: illegal tokens are masked out before
sampling. For a small model this is the difference between music and noise,
and it is the cheapest form of the hardware awareness proposal 3.2 wants
during generation rather than after it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from rse.events.schema import Composition
from rse.models.transformer import RseTransformer
from rse.tokenizers.grammar import TokenGrammar
from rse.tokenizers.tokenizer import Conditioning, RseTokenizer
from rse.tokenizers.vocab import (
    BAR,
    BOS,
    ENERGY,
    HW,
    LENGTH,
    LOOP_START,
    MOOD,
    TEMPO,
)


@dataclass
class SamplingConfig:
    """Inference controls (proposal 6.6)."""

    seed: int | None = None
    temperature: float = 1.0
    #: Per-channel temperature overrides. The collapse this exists for is
    #: channel-specific: at temperature 0.7 the pitched channels produced
    #: baseline-matching repetition and 99.8% in-key notes, while the noise
    #: channel emitted one duration 99.4% of the time and filled over 100% of
    #: frames against 54% in the training data -- continuous static. Sampling
    #: percussion hotter than melody treats the actual problem rather than
    #: trading the whole piece's coherence away to fix the drums.
    channel_temperature: dict[str, float] = field(default_factory=dict)
    top_k: int | None = 40
    top_p: float | None = 0.95
    max_tokens: int = 2048
    #: Mask out tokens the grammar forbids before sampling.
    constrain_to_grammar: bool = True

    def __post_init__(self) -> None:
        if self.temperature <= 0:
            raise ValueError(f"temperature must be positive, got {self.temperature}")
        for channel_id, value in self.channel_temperature.items():
            if value <= 0:
                raise ValueError(
                    f"temperature for channel {channel_id!r} must be positive, got {value}"
                )
        if self.top_k is not None and self.top_k < 1:
            raise ValueError(f"top_k must be at least 1, got {self.top_k}")
        if self.top_p is not None and not 0 < self.top_p <= 1:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")
        if self.max_tokens < 2:
            raise ValueError("max_tokens must be at least 2")

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "temperature": self.temperature,
            "channel_temperature": dict(self.channel_temperature),
            "top_k": self.top_k,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "constrain_to_grammar": self.constrain_to_grammar,
        }


@dataclass
class SampledComposition:
    """What a sampling run produced, and how."""

    composition: Composition
    tokens: list[str]
    ids: list[int]
    conditioning: Conditioning
    config: SamplingConfig
    n_grammar_rejections: int = 0
    hit_token_limit: bool = False
    #: True when the model tried to run past the requested bar count and was
    #: held to it.
    overran_length: bool = False
    #: True when the model tried to stop short of the requested bar count and
    #: was held to it.
    underran_length: bool = False
    decode_warnings: list[str] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_tokens": len(self.ids),
            "n_events": len(self.composition.events),
            "conditioning": self.conditioning.to_dict(),
            "sampling": self.config.to_dict(),
            "n_grammar_rejections": self.n_grammar_rejections,
            "hit_token_limit": self.hit_token_limit,
            "overran_length": self.overran_length,
            "underran_length": self.underran_length,
            "decode_warnings": list(self.decode_warnings)[:20],
        }


def build_prompt(
    tokenizer: RseTokenizer, conditioning: Conditioning, *, loop: bool = True
) -> list[str]:
    """The conditioning header a sampling run starts from."""
    prompt = [
        BOS,
        f"{HW}{tokenizer.config.profile_id}",
        f"{MOOD}{conditioning.mood}",
        f"{ENERGY}{conditioning.energy}",
        f"{TEMPO}{tokenizer.tempo_token_value(conditioning.tempo_bpm)}",
        f"{LENGTH}{tokenizer.length_token_value(conditioning.n_bars)}",
    ]
    if loop:
        prompt.append(LOOP_START)
    return prompt


def filter_logits(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    allowed: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Apply the grammar mask, temperature, top-k and top-p, in that order.

    The mask comes first so that top-k and top-p select among *legal* tokens.
    Filtering first and masking afterwards can leave nothing to sample from.
    """
    logits = logits.clone()
    if allowed is not None:
        mask = torch.full_like(logits, float("-inf"))
        index = torch.tensor(allowed, dtype=torch.long, device=logits.device)
        mask[index] = logits[index]
        logits = mask

    logits = logits / temperature

    if top_k is not None:
        k = min(top_k, int((logits > float("-inf")).sum().item()) or 1)
        threshold = torch.topk(logits, k).values[-1]
        logits = torch.where(logits < threshold, torch.full_like(logits, float("-inf")), logits)

    if top_p is not None and top_p < 1.0:
        ordered, order = torch.sort(logits, descending=True)
        probabilities = torch.softmax(ordered, dim=-1)
        cumulative = torch.cumsum(probabilities, dim=-1)
        # Keep the first token above the threshold, so top_p can never empty
        # the candidate set.
        drop = cumulative - probabilities > top_p
        drop[0] = False
        ordered = ordered.masked_fill(drop, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(0, order, ordered)

    return logits


@torch.no_grad()
def sample(
    model: RseTransformer,
    tokenizer: RseTokenizer,
    *,
    conditioning: Conditioning | None = None,
    config: SamplingConfig | None = None,
    loop: bool = True,
    device: torch.device | str | None = None,
) -> SampledComposition:
    """Generate one composition from a trained model."""
    config = config or SamplingConfig()
    conditioning = conditioning or Conditioning()
    resolved = torch.device(device) if device else next(model.parameters()).device
    model.eval()

    generator = None
    if config.seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(config.seed)

    tokens = build_prompt(tokenizer, conditioning, loop=loop)
    grammar = TokenGrammar(tokenizer.vocabulary)
    for token in tokens:
        grammar.advance(token)
    ids = [tokenizer.vocabulary.id_of(token) for token in tokens]

    eos_id = tokenizer.vocabulary.eos_id
    bar_id = tokenizer.vocabulary.id_of(BAR)
    target_bars = max(1, conditioning.n_bars)
    bars_emitted = 0
    rejections = 0
    hit_limit = True
    overran_length = False
    underran_length = False

    while len(ids) < config.max_tokens:
        window = ids[-model.config.max_seq_len :]
        inputs = torch.tensor([window], dtype=torch.long, device=resolved)
        logits = model(inputs)[0, -1, :].float().cpu()

        allowed = grammar.allowed_ids() if config.constrain_to_grammar else None
        if allowed is not None and bars_emitted < target_bars and len(allowed) > 1:
            # Length is a control in both directions. Without this the model
            # ends early, leaving the last bars empty, which both breaks the
            # requested length and makes the loop die away across the seam.
            without_end = tuple(i for i in allowed if i != eos_id)
            if without_end:
                allowed = without_end
                if eos_id in grammar.allowed_ids():
                    underran_length = True
        if allowed is not None and bars_emitted >= target_bars and bar_id in allowed:
            # The requested length is a control, not a suggestion. Once the
            # last bar is open, another Bar token is removed from the options
            # so the model has to finish rather than run on; an undertrained
            # model will otherwise emit bars forever, since the grammar alone
            # has no reason to stop it.
            allowed = tuple(i for i in allowed if i != bar_id)
            overran_length = True
        if allowed is not None and not allowed:
            break
        # While the grammar is inside a channel's note, that channel's own
        # temperature applies.
        temperature = config.temperature
        if grammar.channel is not None:
            temperature = config.channel_temperature.get(grammar.channel, temperature)
        filtered = filter_logits(
            logits,
            temperature=temperature,
            top_k=config.top_k,
            top_p=config.top_p,
            allowed=allowed,
        )
        probabilities = torch.softmax(filtered, dim=-1)
        if not torch.isfinite(probabilities).all() or float(probabilities.sum()) <= 0:
            break
        next_id = int(torch.multinomial(probabilities, 1, generator=generator).item())

        token = tokenizer.vocabulary.token_of(next_id)
        if not grammar.advance(token):
            # Only reachable with the grammar mask off; count it and stop
            # rather than appending something the decoder cannot read.
            rejections += 1
            if not config.constrain_to_grammar:
                ids.append(next_id)
                tokens.append(token)
                continue
            break

        ids.append(next_id)
        tokens.append(token)
        if next_id == bar_id:
            bars_emitted += 1
        if next_id == eos_id:
            hit_limit = False
            break

    decoded = tokenizer.decode(ids)
    return SampledComposition(
        composition=decoded.composition,
        tokens=tokens,
        ids=ids,
        conditioning=conditioning,
        config=config,
        n_grammar_rejections=rejections,
        hit_token_limit=hit_limit,
        overran_length=overran_length,
        underran_length=underran_length,
        decode_warnings=decoded.warnings,
    )
