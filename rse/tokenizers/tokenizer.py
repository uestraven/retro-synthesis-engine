"""Encoding RSE-EF to tokens and back.

Tokenization is where quantization finally becomes destructive, and that is a
deliberate change of contract. The intake stage keeps source ticks untouched
and treats quantization as additive (proposal 5.3); here, note starts and
durations are snapped to the grid because a model needs a finite vocabulary.
`encode` therefore reports the error it introduced, so the cost is visible
rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rse.data.timing import TempoChange, TimeSignatureChange
from rse.events.schema import Composition, Noise, Note, SectionSpan
from rse.jsonio import read_json, write_json
from rse.tokenizers.grammar import TokenGrammar
from rse.tokenizers.vocab import (
    BAR,
    BOS,
    CHANNEL,
    DURATION,
    ENERGY,
    EOS,
    HW,
    LENGTH,
    LOOP_END,
    LOOP_START,
    MOOD,
    NOISE,
    PITCH,
    POSITION,
    SECTION,
    TEMPO,
    VELOCITY,
    TokenizerConfig,
    Vocabulary,
    build_vocabulary,
)

DEFAULT_TICKS_PER_BEAT = 480


@dataclass
class Conditioning:
    """The musical intent prefixed to a sequence (proposal 6.4)."""

    mood: str = "unknown"
    energy: str = "unknown"
    tempo_bpm: float = 120.0
    n_bars: int = 16

    def to_dict(self) -> dict[str, Any]:
        return {
            "mood": self.mood,
            "energy": self.energy,
            "tempo_bpm": self.tempo_bpm,
            "n_bars": self.n_bars,
        }


@dataclass
class EncodedComposition:
    """A token sequence plus what encoding it cost."""

    ids: list[int]
    tokens: list[str]
    conditioning: Conditioning
    n_events: int = 0
    n_encoded: int = 0
    max_start_error_ticks: int = 0
    total_start_error_ticks: int = 0
    n_duration_clamped: int = 0
    n_out_of_range: int = 0
    warnings: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.ids)

    @property
    def mean_start_error_ticks(self) -> float:
        return self.total_start_error_ticks / self.n_encoded if self.n_encoded else 0.0

    def stats(self) -> dict[str, Any]:
        return {
            "n_tokens": len(self.ids),
            "n_events": self.n_events,
            "n_encoded": self.n_encoded,
            "n_dropped": self.n_events - self.n_encoded,
            "max_start_error_ticks": self.max_start_error_ticks,
            "mean_start_error_ticks": round(self.mean_start_error_ticks, 3),
            "n_duration_clamped": self.n_duration_clamped,
            "n_out_of_range": self.n_out_of_range,
            "conditioning": self.conditioning.to_dict(),
            "warnings": self.warnings[:20],
        }


@dataclass
class DecodedComposition:
    composition: Composition
    n_tokens_read: int = 0
    n_tokens_skipped: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return self.n_tokens_skipped == 0


class RseTokenizer:
    """REMI-inspired tokenizer with RSE's channel and noise semantics."""

    def __init__(
        self, config: TokenizerConfig | None = None, vocabulary: Vocabulary | None = None
    ) -> None:
        if vocabulary is not None:
            self.vocabulary = vocabulary
            self.config = vocabulary.config
        else:
            self.config = config or TokenizerConfig()
            self.vocabulary = build_vocabulary(self.config)

    def __len__(self) -> int:
        return len(self.vocabulary)

    # -- grid helpers -----------------------------------------------------

    def ticks_per_step(self, ticks_per_beat: int) -> int:
        """Ticks in one grid step."""
        per_bar = ticks_per_beat * self.config.beats_per_bar
        step = per_bar // self.config.steps_per_bar
        if step <= 0:
            raise ValueError(
                f"resolution {ticks_per_beat} ticks/beat is too coarse for "
                f"{self.config.steps_per_bar} steps per bar"
            )
        return step

    def velocity_bucket(self, velocity: int) -> int:
        buckets = self.config.velocity_buckets
        return max(0, min(buckets - 1, velocity * buckets // 128))

    def bucket_velocity(self, bucket: int) -> int:
        buckets = self.config.velocity_buckets
        return max(1, min(127, int((bucket + 0.5) * 128 / buckets)))

    def tempo_token_value(self, bpm: float) -> int:
        values = self.config.tempo_values
        return min(values, key=lambda value: abs(value - bpm))

    def length_token_value(self, n_bars: int) -> int:
        return min(self.config.lengths, key=lambda length: abs(length - n_bars))

    # -- encode -----------------------------------------------------------

    def encode(
        self,
        composition: Composition,
        *,
        conditioning: Conditioning | None = None,
    ) -> EncodedComposition:
        """Turn a composition into a token sequence."""
        ticks_per_beat = composition.ticks_per_beat
        step_ticks = self.ticks_per_step(ticks_per_beat)
        steps_per_bar = self.config.steps_per_bar
        sounding = composition.sounding_events()

        if conditioning is None:
            conditioning = conditioning_from_composition(composition, self.config)

        result = EncodedComposition(
            ids=[], tokens=[], conditioning=conditioning, n_events=len(sounding)
        )

        # Group events onto the grid, keeping channel order stable so the same
        # music always produces the same sequence.
        grid: dict[tuple[int, int], list[tuple[int, Note | Noise]]] = {}
        channel_order = {cid: i for i, cid in enumerate(self.config.channels)}
        for event in sounding:
            if event.channel_id not in channel_order:
                result.warnings.append(
                    f"channel {event.channel_id!r} is not in the vocabulary; dropped"
                )
                continue
            value = event.pitch if isinstance(event, Note) else event.value
            if not self._value_in_vocab(event.channel_id, value):
                result.n_out_of_range += 1
                result.warnings.append(
                    f"{event.channel_id} value {value} is outside the vocabulary; dropped"
                )
                continue
            step = int(round(event.tick / step_ticks))
            error = abs(event.tick - step * step_ticks)
            result.max_start_error_ticks = max(result.max_start_error_ticks, error)
            result.total_start_error_ticks += error
            key = (step // steps_per_bar, step % steps_per_bar)
            grid.setdefault(key, []).append((channel_order[event.channel_id], event))

        tokens: list[str] = [
            BOS,
            f"{HW}{self.config.profile_id}",
            f"{MOOD}{conditioning.mood}",
            f"{ENERGY}{conditioning.energy}",
            f"{TEMPO}{self.tempo_token_value(conditioning.tempo_bpm)}",
            f"{LENGTH}{self.length_token_value(conditioning.n_bars)}",
        ]
        if composition.loop_start_tick is not None:
            tokens.append(LOOP_START)

        n_bars = max((bar for bar, _ in grid), default=-1) + 1
        n_bars = max(n_bars, conditioning.n_bars)
        # Sections are announced at the bar they open. Empty means the form is
        # unknown, and the stream simply carries no Section tokens.
        section_at_bar = {
            span.start_bar: span.label
            for span in composition.sections
            if span.label in self.config.section_labels
        }
        for bar in range(n_bars):
            if bar in section_at_bar:
                tokens.append(f"{SECTION}{section_at_bar[bar]}")
            # A Bar token for every bar, including empty ones, so bar count and
            # phrase position stay recoverable from the sequence alone.
            tokens.append(BAR)
            for position in range(steps_per_bar):
                events = grid.get((bar, position))
                if not events:
                    continue
                tokens.append(f"{POSITION}{position}")
                for _order, event in sorted(events, key=lambda pair: pair[0]):
                    duration_steps = int(round(event.duration / step_ticks))
                    if duration_steps < 1:
                        duration_steps = 1
                    if duration_steps > self.config.max_duration_steps:
                        duration_steps = self.config.max_duration_steps
                        result.n_duration_clamped += 1
                    value = event.pitch if isinstance(event, Note) else event.value
                    prefix = PITCH if isinstance(event, Note) else NOISE
                    tokens.extend(
                        (
                            f"{CHANNEL}{event.channel_id}",
                            f"{prefix}{value}",
                            f"{VELOCITY}{self.velocity_bucket(event.velocity)}",
                            f"{DURATION}{duration_steps}",
                        )
                    )
                    result.n_encoded += 1

        if composition.loop_end_tick is not None:
            tokens.append(LOOP_END)
        tokens.append(EOS)

        result.tokens = tokens
        result.ids = [self.vocabulary.id_of(token) for token in tokens]
        return result

    def _value_in_vocab(self, channel_id: str, value: int) -> bool:
        if self.config.is_pitched(channel_id):
            return self.config.pitch_min <= value <= self.config.pitch_max
        return self.config.noise_min <= value <= self.config.noise_max

    # -- decode -----------------------------------------------------------

    def decode(
        self,
        ids: list[int] | list[str],
        *,
        ticks_per_beat: int = DEFAULT_TICKS_PER_BEAT,
    ) -> DecodedComposition:
        """Rebuild a composition from tokens.

        Forgiving by necessity: model output is malformed long before it is
        musical, and a decoder that raised would make early checkpoints
        impossible to listen to. Illegal tokens are skipped and counted.
        """
        tokens = [
            token if isinstance(token, str) else self.vocabulary.token_of(token) for token in ids
        ]
        step_ticks = self.ticks_per_step(ticks_per_beat)
        grammar = TokenGrammar(self.vocabulary)
        result = DecodedComposition(
            composition=Composition(
                hardware_profile=self.config.profile_id, ticks_per_beat=ticks_per_beat
            )
        )

        bar = -1
        pending_section: str | None = None
        sections: list[SectionSpan] = []
        position = 0
        channel: str | None = None
        value: int | None = None
        velocity = 96
        tempo_bpm: float | None = None
        has_loop_start = False
        has_loop_end = False

        for token in tokens:
            if not grammar.advance(token):
                result.n_tokens_skipped += 1
                continue
            result.n_tokens_read += 1

            if token == BAR:
                bar += 1
                position = 0
                if pending_section is not None:
                    sections.append(SectionSpan(start_bar=bar, label=pending_section))
                    pending_section = None
            elif token.startswith(SECTION):
                pending_section = token[len(SECTION) :]
            elif token == LOOP_START:
                has_loop_start = True
            elif token == LOOP_END:
                has_loop_end = True
            elif token == EOS:
                break
            elif token.startswith(TEMPO):
                tempo_bpm = float(token[len(TEMPO) :])
            elif token.startswith(POSITION):
                position = int(token[len(POSITION) :])
            elif token.startswith(CHANNEL):
                channel = token[len(CHANNEL) :]
                value = None
            elif token.startswith(PITCH) or token.startswith(NOISE):
                value = int(token.split("_", 1)[1])
            elif token.startswith(VELOCITY):
                velocity = self.bucket_velocity(int(token[len(VELOCITY) :]))
            elif token.startswith(DURATION):
                steps = int(token[len(DURATION) :])
                if channel is None or value is None or bar < 0:
                    result.warnings.append(
                        f"a duration arrived with no note to attach it to at bar {bar}"
                    )
                    continue
                tick = (bar * self.config.steps_per_bar + position) * step_ticks
                duration = steps * step_ticks
                if self.config.is_pitched(channel):
                    event = Note(
                        channel_id=channel,
                        tick=tick,
                        pitch=value,
                        duration=duration,
                        velocity=velocity,
                    )
                else:
                    event = Noise(
                        channel_id=channel,
                        tick=tick,
                        value=value,
                        duration=duration,
                        velocity=velocity,
                    )
                result.composition.events.append(event)
                value = None

        composition = result.composition
        composition.sections = sections
        composition.tempo_map = [TempoChange(0, int(round(60_000_000 / (tempo_bpm or 120.0))))]
        composition.time_signatures = [TimeSignatureChange(0, self.config.beats_per_bar, 4)]
        if has_loop_start:
            composition.loop_start_tick = 0
        if has_loop_end:
            composition.loop_end_tick = max(bar + 1, 1) * self.config.steps_per_bar * step_ticks
        composition.normalize()
        return result

    # -- persistence ------------------------------------------------------

    def save(self, path: Path) -> Path:
        return write_json(Path(path), self.vocabulary.to_dict())

    @classmethod
    def load(cls, path: Path) -> RseTokenizer:
        return cls(vocabulary=Vocabulary.from_dict(read_json(Path(path))))


def conditioning_from_composition(
    composition: Composition, config: TokenizerConfig
) -> Conditioning:
    """Read conditioning out of a composition's own metadata.

    Generated pieces carry their plan, so mood and energy are known exactly.
    Dataset files do not, and get `unknown` rather than a guess — proposal 6.4
    is explicit that mood conditioning needs labels before it means anything.
    """
    plan = (composition.source_metadata.get("generation") or {}).get("plan") or {}
    tempo_bpm = 120.0
    if composition.tempo_map:
        tempo_bpm = 60_000_000 / composition.tempo_map[0].us_per_beat

    ticks_per_bar = composition.ticks_per_beat * config.beats_per_bar
    n_bars = plan.get("n_bars") or max(
        1, -(-composition.end_tick // ticks_per_bar) if composition.end_tick else 1
    )
    return Conditioning(
        mood=plan.get("mood", "unknown"),
        energy=plan.get("energy", "unknown"),
        tempo_bpm=plan.get("tempo_bpm", tempo_bpm),
        n_bars=int(n_bars),
    )
