"""Token vocabulary construction.

A REMI-inspired layout (proposal 6.2) with the token families RSE needs that
general-purpose MIDI tokenizers do not have: a hardware channel is not a
General MIDI program, and an unpitched noise period is not a pitch.

The vocabulary is built from a config, versioned, and saved beside a
checkpoint. A model is only meaningful alongside the exact vocabulary it was
trained on, so the two travel together.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: Bump when the vocabulary layout changes in a way that invalidates
#: checkpoints trained against it.
VOCAB_VERSION = "rse-vocab/1.3.0"

# Reserved tokens. PAD must be index 0 so padded positions are easy to mask.
PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"
UNK = "<unk>"
SPECIAL_TOKENS = (PAD, BOS, EOS, UNK)

#: Conditioning values the first model is trained to accept. Tempo and length
#: are reliable immediately; mood and energy only become meaningful once the
#: training data is labelled for them (proposal 6.4).
MOODS = ("heroic", "dark", "mysterious", "melancholic", "unknown")
ENERGIES = ("low", "medium", "high", "unknown")


@dataclass(frozen=True)
class TokenizerConfig:
    """Everything that determines the vocabulary and the encoding."""

    profile_id: str = "nes"
    #: Positions per bar. Fixed at 16 because that is the tracker
    #: convention NES music is authored under: 4 rows per beat, 4 beats per
    #: bar. This is a convention, not something to measure -- see
    #: `RseTokenizer.fit_grid`.
    steps_per_bar: int = 16
    beats_per_bar: int = 4
    #: Step on each piece's own tracker row, found by `detect_pulse`,
    #: rather than treating one source frame as one step. False reproduces
    #: the old encoding, which is what the training A/B compares against.
    aligned_grid: bool = True
    #: Pitch range covered by PITCH tokens, inclusive.
    pitch_min: int = 21
    pitch_max: int = 108
    #: Unpitched value range covered by NOISE tokens, inclusive.
    noise_min: int = 1
    noise_max: int = 16
    #: Longest note length expressible, in grid steps.
    max_duration_steps: int = 32
    #: Velocity is bucketed; the APU only has 16 volume levels anyway.
    velocity_buckets: int = 8
    #: Tempo buckets, in BPM.
    tempo_min: int = 40
    #: Reaches 360 because NES-MDB's exprsco grid is one frame per sixteenth at
    #: 24 Hz, which is exactly 360 BPM. Capping at a musical 220 silently
    #: relabelled every exprsco window as 220 and made sampled audio play at
    #: the wrong speed.
    tempo_max: int = 360
    #: Tempo buckets are geometric, not linear. A fitted row rate can imply
    #: anything from ~50 to ~300 BPM, and a fixed 5 BPM step is a 10% error
    #: at 50 BPM while being needlessly fine at 300 — one fixture's 52 BPM
    #: row rounded to 50 and played the piece 4% slow. A constant ratio
    #: spends tokens where the ear notices, bounding the error at ~1.5%
    #: everywhere.
    tempo_ratio: float = 1.03
    #: Bar counts offered as a conditioning token.
    lengths: tuple[int, ...] = (4, 8, 16, 32)
    #: Channels, in a fixed order. Defaults to the NES profile's channels.
    channels: tuple[str, ...] = ("P1", "P2", "TR", "NO")
    #: Which channels are unpitched, and so use NOISE tokens.
    unpitched_channels: tuple[str, ...] = ("NO",)
    #: Section letters available as structure tokens. This is the family
    #: proposal 6.2 reserved for phrase boundaries: a `Section_A` before a bar
    #: says "the form returns here", which is the one thing the first model
    #: could not learn from `Bar` tokens alone.
    section_labels: tuple[str, ...] = ("A", "B", "C", "D", "E", "F")
    #: Reserved: controller events are preserved in the data but not yet
    #: modelled. See docs/tokenizer.md for why.
    include_controllers: bool = False
    vocab_version: str = VOCAB_VERSION

    def __post_init__(self) -> None:
        if self.steps_per_bar <= 0:
            raise ValueError(f"steps_per_bar must be positive, got {self.steps_per_bar}")
        if self.pitch_min > self.pitch_max:
            raise ValueError("pitch_min must not exceed pitch_max")
        if self.noise_min > self.noise_max:
            raise ValueError("noise_min must not exceed noise_max")
        if self.velocity_buckets < 1:
            raise ValueError("velocity_buckets must be at least 1")
        if self.max_duration_steps < 1:
            raise ValueError("max_duration_steps must be at least 1")
        if self.include_controllers:
            raise NotImplementedError(
                "controller tokens are not implemented yet; the data preserves "
                "controllers, the first model simply does not read them"
            )
        unknown = set(self.unpitched_channels) - set(self.channels)
        if unknown:
            raise ValueError(f"unpitched channels not in channels: {sorted(unknown)}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "steps_per_bar": self.steps_per_bar,
            "aligned_grid": self.aligned_grid,
            "beats_per_bar": self.beats_per_bar,
            "pitch_min": self.pitch_min,
            "pitch_max": self.pitch_max,
            "noise_min": self.noise_min,
            "noise_max": self.noise_max,
            "max_duration_steps": self.max_duration_steps,
            "velocity_buckets": self.velocity_buckets,
            "tempo_min": self.tempo_min,
            "tempo_max": self.tempo_max,
            "tempo_ratio": self.tempo_ratio,
            "lengths": list(self.lengths),
            "channels": list(self.channels),
            "unpitched_channels": list(self.unpitched_channels),
            "section_labels": list(self.section_labels),
            "include_controllers": self.include_controllers,
            "vocab_version": self.vocab_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TokenizerConfig:
        version = str(data.get("vocab_version", ""))
        if version and not version.startswith("rse-vocab/1"):
            raise ValueError(
                f"unsupported vocabulary version {version!r}; this build reads rse-vocab/1.x"
            )
        return cls(
            profile_id=data.get("profile_id", "nes"),
            steps_per_bar=int(data.get("steps_per_bar", 16)),
            aligned_grid=bool(data.get("aligned_grid", True)),
            beats_per_bar=int(data.get("beats_per_bar", 4)),
            pitch_min=int(data.get("pitch_min", 21)),
            pitch_max=int(data.get("pitch_max", 108)),
            noise_min=int(data.get("noise_min", 1)),
            noise_max=int(data.get("noise_max", 16)),
            max_duration_steps=int(data.get("max_duration_steps", 32)),
            velocity_buckets=int(data.get("velocity_buckets", 8)),
            tempo_min=int(data.get("tempo_min", 40)),
            tempo_max=int(data.get("tempo_max", 360)),
            tempo_ratio=float(data.get("tempo_ratio", 1.03)),
            lengths=tuple(data.get("lengths", (4, 8, 16, 32))),
            channels=tuple(data.get("channels", ("P1", "P2", "TR", "NO"))),
            unpitched_channels=tuple(data.get("unpitched_channels", ("NO",))),
            section_labels=tuple(data.get("section_labels", ("A", "B", "C", "D", "E", "F"))),
            include_controllers=bool(data.get("include_controllers", False)),
            vocab_version=version or VOCAB_VERSION,
        )

    @property
    def tempo_values(self) -> tuple[int, ...]:
        values, value = [], float(self.tempo_min)
        while value <= self.tempo_max:
            values.append(int(round(value)))
            value *= self.tempo_ratio
        if values and values[-1] != self.tempo_max:
            values.append(self.tempo_max)
        return tuple(dict.fromkeys(values))

    def is_pitched(self, channel_id: str) -> bool:
        return channel_id not in self.unpitched_channels


@dataclass
class Vocabulary:
    """A fixed token list with lookups in both directions."""

    tokens: tuple[str, ...]
    config: TokenizerConfig
    _index: dict[str, int] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._index = {token: i for i, token in enumerate(self.tokens)}
        if len(self._index) != len(self.tokens):
            raise ValueError("vocabulary contains duplicate tokens")

    def __len__(self) -> int:
        return len(self.tokens)

    def __contains__(self, token: str) -> bool:
        return token in self._index

    def id_of(self, token: str) -> int:
        """Token id, falling back to `<unk>` for anything unrepresentable."""
        return self._index.get(token, self._index[UNK])

    def token_of(self, token_id: int) -> str:
        return self.tokens[token_id]

    @property
    def pad_id(self) -> int:
        return self._index[PAD]

    @property
    def bos_id(self) -> int:
        return self._index[BOS]

    @property
    def eos_id(self) -> int:
        return self._index[EOS]

    @property
    def unk_id(self) -> int:
        return self._index[UNK]

    def family(self, prefix: str) -> tuple[int, ...]:
        """Ids of every token in one family, e.g. all `Pitch_*` tokens."""
        return tuple(i for token, i in self._index.items() if token.startswith(prefix))

    def to_dict(self) -> dict[str, Any]:
        return {"config": self.config.to_dict(), "tokens": list(self.tokens)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Vocabulary:
        config = TokenizerConfig.from_dict(data["config"])
        tokens = tuple(data["tokens"])
        rebuilt = build_vocabulary(config)
        if tokens != rebuilt.tokens:
            raise ValueError(
                "saved tokens do not match the vocabulary rebuilt from the saved "
                "config; the checkpoint and this build disagree"
            )
        return rebuilt


# Token name prefixes. Kept as constants because encode, decode and the
# grammar-constrained sampler all have to agree on them exactly.
HW = "Hardware_"
MOOD = "Mood_"
ENERGY = "Energy_"
TEMPO = "Tempo_"
LENGTH = "Length_"
BAR = "Bar"
POSITION = "Position_"
CHANNEL = "Channel_"
PITCH = "Pitch_"
NOISE = "Noise_"
VELOCITY = "Velocity_"
DURATION = "Duration_"
LOOP_START = "Loop_start"
LOOP_END = "Loop_end"
SECTION = "Section_"


def build_vocabulary(config: TokenizerConfig | None = None) -> Vocabulary:
    """Construct the full token list for a config, in a stable order."""
    config = config or TokenizerConfig()
    tokens: list[str] = list(SPECIAL_TOKENS)

    tokens.append(f"{HW}{config.profile_id}")
    tokens.extend(f"{MOOD}{mood}" for mood in MOODS)
    tokens.extend(f"{ENERGY}{energy}" for energy in ENERGIES)
    tokens.extend(f"{TEMPO}{tempo}" for tempo in config.tempo_values)
    tokens.extend(f"{LENGTH}{length}" for length in config.lengths)

    tokens.append(BAR)
    tokens.extend(f"{POSITION}{step}" for step in range(config.steps_per_bar))
    tokens.extend(f"{CHANNEL}{channel}" for channel in config.channels)
    tokens.extend(f"{PITCH}{pitch}" for pitch in range(config.pitch_min, config.pitch_max + 1))
    tokens.extend(f"{NOISE}{value}" for value in range(config.noise_min, config.noise_max + 1))
    tokens.extend(f"{VELOCITY}{bucket}" for bucket in range(config.velocity_buckets))
    tokens.extend(f"{DURATION}{steps}" for steps in range(1, config.max_duration_steps + 1))
    tokens.extend(f"{SECTION}{label}" for label in config.section_labels)
    tokens.extend((LOOP_START, LOOP_END))

    return Vocabulary(tokens=tuple(tokens), config=config)
