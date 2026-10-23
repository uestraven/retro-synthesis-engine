"""The token grammar, as one state machine used by two callers.

The decoder uses it to know what it is looking at, and the sampler uses it to
mask out illegal tokens before sampling. Keeping both on one implementation is
the point: a sampler that can emit sequences the decoder cannot read produces
silent corruption, and two hand-written state machines drift apart.

Masking is also the cheapest form of the hardware awareness proposal 3.2 asks
for. A small model will happily emit two pitches in a row; refusing to sample
one costs nothing and removes a whole class of garbage.
"""

from __future__ import annotations

from enum import Enum, auto

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
    TEMPO,
    VELOCITY,
    Vocabulary,
)


class State(Enum):
    """Where the parser is in the sequence."""

    START = auto()
    EXPECT_HARDWARE = auto()
    EXPECT_MOOD = auto()
    EXPECT_ENERGY = auto()
    EXPECT_TEMPO = auto()
    EXPECT_LENGTH = auto()
    EXPECT_FIRST_BAR = auto()
    #: After a Bar: a Position, another Bar, or the end.
    EXPECT_POSITION = auto()
    #: After a Position: a Channel must follow.
    EXPECT_CHANNEL = auto()
    #: After a Channel: the pitch or noise value for that channel.
    EXPECT_VALUE = auto()
    EXPECT_VELOCITY = auto()
    EXPECT_DURATION = auto()
    #: After a complete note: another note here, a new Position, Bar, or end.
    EXPECT_NOTE_OR_NEXT = auto()
    DONE = auto()


class TokenGrammar:
    """Tracks parser state and answers what may come next."""

    def __init__(self, vocabulary: Vocabulary) -> None:
        self.vocabulary = vocabulary
        self.config = vocabulary.config
        self.state = State.START
        #: The channel named by the most recent Channel token, which decides
        #: whether a pitch or a noise value is legal next.
        self.channel: str | None = None

    def reset(self) -> None:
        self.state = State.START
        self.channel = None

    # -- what may come next -----------------------------------------------

    def allowed_prefixes(self) -> tuple[str, ...]:
        """Token name prefixes legal in the current state."""
        if self.state is State.START:
            return (BOS,)
        if self.state is State.EXPECT_HARDWARE:
            return (HW,)
        if self.state is State.EXPECT_MOOD:
            return (MOOD,)
        if self.state is State.EXPECT_ENERGY:
            return (ENERGY,)
        if self.state is State.EXPECT_TEMPO:
            return (TEMPO,)
        if self.state is State.EXPECT_LENGTH:
            return (LENGTH,)
        if self.state is State.EXPECT_FIRST_BAR:
            return (LOOP_START, BAR)
        if self.state is State.EXPECT_POSITION:
            return (POSITION, BAR, LOOP_END, EOS)
        if self.state is State.EXPECT_CHANNEL:
            return (CHANNEL,)
        if self.state is State.EXPECT_VALUE:
            if self.channel is not None and not self.config.is_pitched(self.channel):
                return (NOISE,)
            return (PITCH,)
        if self.state is State.EXPECT_VELOCITY:
            return (VELOCITY,)
        if self.state is State.EXPECT_DURATION:
            return (DURATION,)
        if self.state is State.EXPECT_NOTE_OR_NEXT:
            return (CHANNEL, POSITION, BAR, LOOP_END, EOS)
        return ()

    def allowed_ids(self) -> tuple[int, ...]:
        """Vocabulary ids legal in the current state."""
        prefixes = self.allowed_prefixes()
        if not prefixes:
            return ()
        allowed: set[int] = set()
        for prefix in prefixes:
            if prefix in (BOS, EOS, BAR, LOOP_START, LOOP_END):
                # Exact tokens, not families; `Bar` would otherwise also match
                # nothing and `Loop_start` would match `Loop_start` only.
                allowed.add(self.vocabulary.id_of(prefix))
            else:
                allowed.update(self.vocabulary.family(prefix))
        return tuple(sorted(allowed))

    def permits(self, token: str) -> bool:
        for prefix in self.allowed_prefixes():
            if prefix in (BOS, EOS, BAR, LOOP_START, LOOP_END):
                if token == prefix:
                    return True
            elif token.startswith(prefix):
                return True
        return False

    # -- consuming tokens -------------------------------------------------

    def advance(self, token: str) -> bool:
        """Consume `token`. Returns whether it was legal.

        An illegal token leaves the state untouched, so a caller reading model
        output can skip it and carry on rather than losing the rest of the
        sequence.
        """
        if not self.permits(token):
            return False

        if self.state is State.START:
            self.state = State.EXPECT_HARDWARE
        elif self.state is State.EXPECT_HARDWARE:
            self.state = State.EXPECT_MOOD
        elif self.state is State.EXPECT_MOOD:
            self.state = State.EXPECT_ENERGY
        elif self.state is State.EXPECT_ENERGY:
            self.state = State.EXPECT_TEMPO
        elif self.state is State.EXPECT_TEMPO:
            self.state = State.EXPECT_LENGTH
        elif self.state is State.EXPECT_LENGTH:
            self.state = State.EXPECT_FIRST_BAR
        elif self.state is State.EXPECT_FIRST_BAR:
            self.state = State.EXPECT_POSITION if token == BAR else State.EXPECT_FIRST_BAR
        elif self.state in (State.EXPECT_POSITION, State.EXPECT_NOTE_OR_NEXT):
            if token == EOS:
                self.state = State.DONE
            elif token == BAR or token == LOOP_END:
                self.state = State.EXPECT_POSITION
            elif token.startswith(POSITION):
                self.state = State.EXPECT_CHANNEL
            elif token.startswith(CHANNEL):
                self.channel = token[len(CHANNEL) :]
                self.state = State.EXPECT_VALUE
        elif self.state is State.EXPECT_CHANNEL:
            self.channel = token[len(CHANNEL) :]
            self.state = State.EXPECT_VALUE
        elif self.state is State.EXPECT_VALUE:
            self.state = State.EXPECT_VELOCITY
        elif self.state is State.EXPECT_VELOCITY:
            self.state = State.EXPECT_DURATION
        elif self.state is State.EXPECT_DURATION:
            self.state = State.EXPECT_NOTE_OR_NEXT
        return True

    @property
    def finished(self) -> bool:
        return self.state is State.DONE
