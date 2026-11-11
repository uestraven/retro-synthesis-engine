"""The RSE v2 G3 token grammar: model-facing serialization of grouped events.

Sits on the frozen 4000 Hz codec and the frozen C2 k=512 delta vocabulary.
Neither is changed here.

    BOS [SNAPSHOT] END_h* PITCH* (CHANGE_h PITCH)* DELTA ... EOS

`DELTA` is the group separator: a *complete* delta closes the previous group
and advances the clock. `EOS` closes the final group. There is no
`GROUP_START`/`GROUP_END`; G0's explicit framing cost more than half of Level
1's musical context (6.6 s against 13.7 s per 1024 tokens) and G3 recovers
16.8 s without a factored prediction head.

**One pitch family, two syntactic positions.** The fixed ends -> starts ->
changes order makes a bare `PITCH` unambiguously a note start, while a `PITCH`
following `CHANGE_h` is that change's value. Separate `START_pitch` and
`PITCH` families would double the vocabulary and halve each token's training
support for no expressive gain.

**Pitch carries its own status.** No status token is emitted, because status is
recoverable from the pitch token alone: across all 1,740 tracks `estimated` is
always pitched while `unrecoverable`, `noise` and `pitch_modulated` are always
unpitched, and no status occurs in both cases. Each unpitched status therefore
gets a DEDICATED symbol -- a single generic missing-pitch token would destroy a
source distinction.

**Scope of the losslessness claim.** This grammar is lossless with respect to
the declared tokenizable action stream, apart from the disclosed 10-cent pitch
quantisation (max 5 cents). It is NOT lossless with respect to the full
`events.v4` record: source `event_id`, `provenance_voice` and the audit
metadata the Milestone 1 legato verification relied on are not representable
and need a sidecar. The Milestone 1 reconstruction claim and any claim about
this layer are separate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Iterable, Sequence

from rse.snes_v2 import codec as cx
from rse.snes_v2.delta import DeltaCodec, IncompleteDelta, NonPositiveDelta

#: Bins per semitone for the pitch grid. 10 gives a 5-cent maximum error,
#: verified at mean 2.41 c / max 5.00 c over 2,130,666 train pitches.
BINS_PER_SEMITONE = 10

#: Pitch grid span, from the corpus range -36.74..131.09 MIDI with margin.
PITCH_MIN_BIN = -400
PITCH_MAX_BIN = 1340

#: Unpitched statuses, each with its own symbol so the category survives.
UNPITCHED_STATUSES = ("unrecoverable", "noise", "pitch_modulated")

#: Structural ceiling on actions in one group: ends, starts and changes are
#: each bounded by the eight slots. The observed maximum is 16 (8 ends + 8
#: starts); the limit is the derivable 24 so a valid-but-unseen structure is
#: not made unrepresentable by a corpus artefact.
MAX_ACTIONS_PER_GROUP = 3 * cx.N_HARDWARE_VOICES


class GrammarError(ValueError):
    """The token stream is not a valid completed performance."""


class IncompleteStream(GrammarError):
    """The stream ends without a valid terminal boundary.

    **A valid prefix is not a valid completed performance.** Sampling that
    reaches its token budget mid-group must not be silently finalised: Level
    2's generator stopped mid-event and only the harness noticing kept that
    from being reported as the model's output.
    """


class State(Enum):
    """Parser position. Also drives the generation mask."""

    START = "start"                 # expect BOS
    SNAPSHOT_ENTRY = "snap_entry"   # expect HANDLE_h or SNAPSHOT_END
    SNAPSHOT_PITCH = "snap_pitch"   # expect PITCH
    SNAPSHOT_AGE = "snap_age"       # expect a delta (ticks since note start)
    GROUP_ENDS = "ends"             # END_h | PITCH | CHANGE_h | DELTA | EOS
    GROUP_STARTS = "starts"         # PITCH | CHANGE_h | DELTA | EOS
    GROUP_CHANGES = "changes"       # CHANGE_h | DELTA | EOS
    CHANGE_PITCH = "change_pitch"   # expect PITCH
    DELTA_DIGITS = "delta_digits"   # expect an escape digit
    DONE = "done"


@dataclass(frozen=True)
class Vocab:
    """Token layout. Built from the frozen delta codec, never refitted here."""

    delta: DeltaCodec

    BOS: int = 0
    EOS: int = 1
    SNAPSHOT_START: int = 2
    SNAPSHOT_END: int = 3

    @property
    def end_base(self) -> int:
        return 4

    @property
    def change_base(self) -> int:
        return self.end_base + cx.N_HARDWARE_VOICES

    @property
    def handle_base(self) -> int:
        """Snapshot-only handle markers."""
        return self.change_base + cx.N_HARDWARE_VOICES

    @property
    def pitch_base(self) -> int:
        return self.handle_base + cx.N_HARDWARE_VOICES

    @property
    def n_pitch(self) -> int:
        return (PITCH_MAX_BIN - PITCH_MIN_BIN + 1) + len(UNPITCHED_STATUSES)

    @property
    def delta_base(self) -> int:
        return self.pitch_base + self.n_pitch

    @property
    def size(self) -> int:
        return self.delta_base + self.delta.vocab_size

    # --- pitch field ----------------------------------------------------

    def pitch_token(self, midi_note: float | None, status: str | None) -> int:
        if midi_note is None:
            try:
                offset = UNPITCHED_STATUSES.index(status or "unrecoverable")
            except ValueError as exc:
                raise GrammarError(f"unknown unpitched status {status!r}") from exc
            return self.pitch_base + (PITCH_MAX_BIN - PITCH_MIN_BIN + 1) + offset
        scaled = Decimal(str(midi_note)) * BINS_PER_SEMITONE
        bin_index = int(scaled.to_integral_value(rounding=ROUND_HALF_UP))
        if not (PITCH_MIN_BIN <= bin_index <= PITCH_MAX_BIN):
            raise GrammarError(f"pitch {midi_note} outside the grid")
        return self.pitch_base + (bin_index - PITCH_MIN_BIN)

    def read_pitch(self, token: int) -> tuple[float | None, str]:
        """Inverse of `pitch_token`. Status recovers from the token alone."""
        offset = token - self.pitch_base
        n_bins = PITCH_MAX_BIN - PITCH_MIN_BIN + 1
        if not (0 <= offset < self.n_pitch):
            raise GrammarError(f"token {token} is not a pitch")
        if offset >= n_bins:
            return None, UNPITCHED_STATUSES[offset - n_bins]
        return (offset + PITCH_MIN_BIN) / BINS_PER_SEMITONE, "estimated"

    def is_pitch(self, token: int) -> bool:
        return self.pitch_base <= token < self.pitch_base + self.n_pitch

    def is_end(self, token: int) -> bool:
        return self.end_base <= token < self.end_base + cx.N_HARDWARE_VOICES

    def is_change(self, token: int) -> bool:
        return self.change_base <= token < self.change_base + cx.N_HARDWARE_VOICES

    def is_handle(self, token: int) -> bool:
        return self.handle_base <= token < self.handle_base + cx.N_HARDWARE_VOICES

    def encode_delta(self, ticks: int) -> list[int]:
        return [self.delta_base + t for t in self.delta.encode(ticks)]

    def decode_delta(self, tokens: Sequence[int], index: int) -> tuple[int, int]:
        # Only the tokens a delta can occupy. Slicing the whole remaining
        # stream here made parsing quadratic: 6.8 s for an 8,390-token track.
        window = tokens[index:index + self.delta.MAX_TOKENS_PER_DELTA]
        shifted = [t - self.delta_base for t in window]
        value, consumed = self.delta.decode(shifted, 0)
        return value, index + consumed


# --- serialization --------------------------------------------------------

def serialize(groups: Sequence[cx.Group], vocab: Vocab) -> list[int]:
    """Grouped events -> G3 tokens. Handles are the codec's allocator slots.

    The first group's time is carried as a leading delta when it is non-zero,
    so an initial rest is represented rather than lost.
    """
    tokens = [vocab.BOS]
    slot_of: dict[int, int] = {}
    free = list(range(cx.N_HARDWARE_VOICES))
    for index, group in enumerate(groups):
        if index == 0:
            if group.delta:
                tokens += vocab.encode_delta(group.delta)
        else:
            tokens += vocab.encode_delta(group.delta)
        actions = group.ordered()
        if len(actions) > MAX_ACTIONS_PER_GROUP:
            raise GrammarError(
                f"group {index} has {len(actions)} actions, limit "
                f"{MAX_ACTIONS_PER_GROUP}")
        for action in actions:
            if action.kind == "end":
                slot = slot_of.pop(action.note_id, None)
                if slot is None:
                    continue          # end for a note never started here
                free.append(slot); free.sort()
                tokens.append(vocab.end_base + slot)
            elif action.kind == "start":
                if not free:
                    raise GrammarError("start with no free slot; the codec's "
                                       "encode-time clamp should prevent this")
                slot = free.pop(0)
                slot_of[action.note_id] = slot
                tokens.append(vocab.pitch_token(action.midi_note,
                                                action.pitch_status))
            else:
                slot = slot_of.get(action.note_id)
                if slot is None:
                    continue          # change for a note not active
                tokens.append(vocab.change_base + slot)
                tokens.append(vocab.pitch_token(action.midi_note,
                                                action.pitch_status))
    tokens.append(vocab.EOS)
    return tokens


@dataclass
class ParseReport:
    n_groups: int = 0
    n_ends: int = 0
    n_starts: int = 0
    n_changes: int = 0
    n_actions_max_in_group: int = 0
    #: Forced group closures at MAX_ACTIONS_PER_GROUP. An intervention, never
    #: a choice the model made.
    n_forced_group_closures: int = 0
    total_ticks: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in (
            "n_groups", "n_ends", "n_starts", "n_changes",
            "n_actions_max_in_group", "n_forced_group_closures", "total_ticks")}


@dataclass
class ParsedGroup:
    """One decoded group: slot-referenced actions at one musical instant."""

    delta: int
    ends: list[int] = field(default_factory=list)
    starts: list[tuple[float | None, str]] = field(default_factory=list)
    changes: list[tuple[int, float | None, str]] = field(default_factory=list)


def legal(state: State, *, active: frozenset[int], n_actions: int,
          vocab: Vocab, started: bool,
          changed: frozenset[int] = frozenset()) -> set[int]:
    """Tokens admissible now. Drives both validation and the generation mask.

    Handle validity is part of this: an `END_h` or `CHANGE_h` naming an
    inactive slot is excluded. Probability mass the model placed on those
    BEFORE masking is the diagnostic of what it learned; the mask itself is an
    engineering guarantee and earns the model no credit.
    """
    if state is State.START:
        return {vocab.BOS}
    if state is State.DONE:
        return set()
    if state is State.SNAPSHOT_ENTRY:
        return {vocab.SNAPSHOT_END} | {
            vocab.handle_base + h for h in range(cx.N_HARDWARE_VOICES)}
    if state in (State.SNAPSHOT_PITCH, State.CHANGE_PITCH):
        return {vocab.pitch_base + i for i in range(vocab.n_pitch)}
    if state is State.DELTA_DIGITS:
        base = vocab.delta_base + vocab.delta.n_direct + 3
        return {base + d for d in range(256)}

    out: set[int] = set()
    room = n_actions < MAX_ACTIONS_PER_GROUP
    if state is State.GROUP_ENDS and room:
        out |= {vocab.end_base + h for h in active}
    if state in (State.GROUP_ENDS, State.GROUP_STARTS) and room:
        if len(active) < cx.N_HARDWARE_VOICES:
            out |= {vocab.pitch_base + i for i in range(vocab.n_pitch)}
    if state in (State.GROUP_ENDS, State.GROUP_STARTS,
                 State.GROUP_CHANGES) and room:
        # At most ONE change per handle per group. Without this the action
        # count is unbounded (8 ends + 8 starts + arbitrarily many changes) and
        # MAX_ACTIONS_PER_GROUP would be a arbitrary bound above the observed
        # 16 rather than a structural maximum. A second change at the same
        # instant also has no meaning: it would simply supersede the first.
        out |= {vocab.change_base + h for h in active - changed}
    # A delta always closes the group; EOS closes the final one. Both are
    # legal whenever a group is open, which is what makes the stream
    # terminable at a valid boundary.
    if started:
        out |= {vocab.delta_base + t for t in range(vocab.delta.n_direct + 3)}
        out.add(vocab.EOS)
    return out


def parse(tokens: Sequence[int], vocab: Vocab) -> tuple[list[ParsedGroup], ParseReport]:
    """G3 tokens -> groups. STRICT: a prefix is not a completed performance.

    Rejects, rather than repairing: a truncated delta escape, an unfinished
    `CHANGE_h` awaiting its pitch, a stream with no terminal `EOS`, an action
    naming an inactive slot, a start with no free slot, and a group exceeding
    the action limit. Empty groups are rejected too -- the codec never emits
    one, so a group with no actions is a generator artefact.
    """
    if not tokens or tokens[0] != vocab.BOS:
        raise GrammarError("stream does not begin with BOS")

    report = ParseReport()
    groups: list[ParsedGroup] = []
    active: set[int] = set()
    index = 1
    pending_delta = 0
    current = ParsedGroup(delta=0)
    state = State.GROUP_ENDS
    started = False
    changed: set[int] = set()

    def close_group(delta_for_next: int) -> None:
        nonlocal current, state, started
        n = len(current.ends) + len(current.starts) + len(current.changes)
        if n == 0:
            raise GrammarError(f"empty group before token {index}")
        report.n_groups += 1
        report.n_ends += len(current.ends)
        report.n_starts += len(current.starts)
        report.n_changes += len(current.changes)
        report.n_actions_max_in_group = max(report.n_actions_max_in_group, n)
        groups.append(current)
        current = ParsedGroup(delta=delta_for_next)
        state = State.GROUP_ENDS
        started = False
        changed.clear()

    while index < len(tokens):
        token = tokens[index]
        if token == vocab.EOS:
            close_group(0)
            report.total_ticks = sum(g.delta for g in groups)
            state = State.DONE
            if index != len(tokens) - 1:
                raise GrammarError("tokens follow EOS")
            return groups, report

        n_actions = len(current.ends) + len(current.starts) + len(current.changes)
        if n_actions >= MAX_ACTIONS_PER_GROUP and not (
                token >= vocab.delta_base):
            raise GrammarError(
                f"group exceeds {MAX_ACTIONS_PER_GROUP} actions at token {index}")

        if vocab.is_end(token):
            if state is not State.GROUP_ENDS:
                raise GrammarError(f"END after starts/changes at {index}")
            slot = token - vocab.end_base
            if slot not in active:
                raise GrammarError(f"END on inactive slot {slot} at {index}")
            active.discard(slot)
            current.ends.append(slot)
            started = True
            index += 1
        elif vocab.is_pitch(token):
            if state not in (State.GROUP_ENDS, State.GROUP_STARTS):
                raise GrammarError(f"start after changes began at {index}")
            if len(active) >= cx.N_HARDWARE_VOICES:
                raise GrammarError(f"start with no free slot at {index}")
            midi, status = vocab.read_pitch(token)
            slot = min(set(range(cx.N_HARDWARE_VOICES)) - active)
            active.add(slot)
            current.starts.append((midi, status))
            state = State.GROUP_STARTS
            started = True
            index += 1
        elif vocab.is_change(token):
            slot = token - vocab.change_base
            if slot not in active:
                raise GrammarError(f"CHANGE on inactive slot {slot} at {index}")
            if slot in changed:
                raise GrammarError(
                    f"second CHANGE on slot {slot} in one group at {index}; "
                    f"this is what makes {MAX_ACTIONS_PER_GROUP} structural")
            changed.add(slot)
            if index + 1 >= len(tokens):
                raise IncompleteStream(f"CHANGE_{slot} without its pitch")
            nxt = tokens[index + 1]
            if not vocab.is_pitch(nxt):
                raise GrammarError(f"CHANGE_{slot} not followed by a pitch")
            midi, status = vocab.read_pitch(nxt)
            current.changes.append((slot, midi, status))
            state = State.GROUP_CHANGES
            started = True
            index += 2
        elif token >= vocab.delta_base:
            try:
                value, nxt = vocab.decode_delta(tokens, index)
            except (IncompleteDelta, NonPositiveDelta) as exc:
                raise IncompleteStream(str(exc)) from exc
            if not started:
                # A leading delta before any action is the initial rest.
                if groups or current.ends or current.starts or current.changes:
                    raise GrammarError(f"delta closes an empty group at {index}")
                current.delta = value
                index = nxt
                continue
            close_group(value)
            index = nxt
        else:
            raise GrammarError(f"token {token} illegal at {index}")

    raise IncompleteStream("stream ends without EOS; a valid prefix is not a "
                           "completed performance")


# --- window state: the Level 2 positional-phase defect, prevented ---------

@dataclass(frozen=True)
class ActiveNote:
    """One sounding note at a window boundary.

    **No eventual ending.** A training window must not expose a note's
    duration, which is available in the published data but NOT available to a
    generator at that instant. Including it would leak future information into
    the conditioning context.
    """

    handle: int
    midi_note: float | None
    status: str
    #: Ticks elapsed since this note started, at the boundary.
    age_ticks: int


@dataclass
class Window:
    """A training-ready window. ONE convention for all three cases.

    Every window begins immediately before a complete group's actions, with
    the decoder ready for actions. The snapshot describes the state at that
    instant, and the delta that advanced the clock to it is NOT repeated
    inside the window -- so initial windows, mid-track windows and inference
    rollover are constructed identically rather than by three rules.

    `start_tick` is metadata, not tokens: a window's absolute position is
    unavailable to a generator and is never conditioned on.
    """

    tokens: list[int]
    snapshot: tuple[ActiveNote, ...]
    first_group: int
    last_group: int
    start_tick: int
    n_snapshot_tokens: int

    @property
    def n_tokens(self) -> int:
        return len(self.tokens)


def encode_snapshot(notes: Sequence[ActiveNote], vocab: Vocab) -> list[int]:
    """SNAPSHOT_START (HANDLE_h PITCH <delta age>)* SNAPSHOT_END.

    Handles are emitted in ascending order so the encoding is deterministic.
    An age of zero -- a note that started exactly at the boundary -- is
    representable because the delta codec refuses zero, so age+1 is encoded
    and decremented on read.
    """
    tokens = [vocab.SNAPSHOT_START]
    for note in sorted(notes, key=lambda n: n.handle):
        tokens.append(vocab.handle_base + note.handle)
        tokens.append(vocab.pitch_token(note.midi_note, note.status))
        tokens += vocab.encode_delta(note.age_ticks + 1)
    tokens.append(vocab.SNAPSHOT_END)
    return tokens


def decode_snapshot(tokens: Sequence[int], index: int,
                    vocab: Vocab) -> tuple[list[ActiveNote], int]:
    if index >= len(tokens) or tokens[index] != vocab.SNAPSHOT_START:
        raise GrammarError("expected SNAPSHOT_START")
    index += 1
    out: list[ActiveNote] = []
    while index < len(tokens) and tokens[index] != vocab.SNAPSHOT_END:
        if not vocab.is_handle(tokens[index]):
            raise GrammarError(f"expected a handle at {index}")
        handle = tokens[index] - vocab.handle_base
        if index + 1 >= len(tokens) or not vocab.is_pitch(tokens[index + 1]):
            raise IncompleteStream("snapshot entry without a pitch")
        midi, status = vocab.read_pitch(tokens[index + 1])
        try:
            age, index = vocab.decode_delta(tokens, index + 2)
        except (IncompleteDelta, NonPositiveDelta) as exc:
            raise IncompleteStream(f"snapshot age: {exc}") from exc
        out.append(ActiveNote(handle, midi, status, age - 1))
    if index >= len(tokens):
        raise IncompleteStream("snapshot without SNAPSHOT_END")
    return out, index + 1


class StreamState:
    """Active-note state advanced group by group.

    **Both the training and the inference path use this one class.** Level 2's
    positional-phase defect arose because training and generation constructed
    their contexts by different code; here they cannot diverge.
    """

    def __init__(self, vocab: Vocab) -> None:
        self.vocab = vocab
        self.tick = 0
        self.free = list(range(cx.N_HARDWARE_VOICES))
        #: handle -> (midi_note, status, start_tick)
        self.notes: dict[int, tuple[float | None, str, int]] = {}
        self.slot_of: dict[int, int] = {}

    def snapshot(self) -> tuple[ActiveNote, ...]:
        return tuple(
            ActiveNote(handle, midi, status, self.tick - start)
            for handle, (midi, status, start) in sorted(self.notes.items()))

    def emit(self, group: cx.Group, *, first: bool = False) -> list[int]:
        """Advance the state AND return this group's action tokens.

        One operation, deliberately. Emitting tokens and advancing state in
        separate functions is how they drift apart: an earlier split version
        read slot assignments before the group was applied, so a pitch change
        on a note started in the same group was silently dropped. The leading
        delta is NOT included -- the caller owns group separation, which is
        what lets a window begin at a boundary without repeating it.
        """
        if first:
            if group.delta:
                self.tick += group.delta
        else:
            self.tick += group.delta

        tokens: list[int] = []
        changed: set[int] = set()
        for action in group.ordered():
            if action.kind == "end":
                slot = self.slot_of.pop(action.note_id, None)
                if slot is None:
                    continue
                self.notes.pop(slot, None)
                self.free.append(slot)
                self.free.sort()
                tokens.append(self.vocab.end_base + slot)
            elif action.kind == "start":
                if not self.free:
                    continue
                slot = self.free.pop(0)
                self.slot_of[action.note_id] = slot
                self.notes[slot] = (action.midi_note,
                                    action.pitch_status or "estimated",
                                    self.tick)
                tokens.append(self.vocab.pitch_token(action.midi_note,
                                                     action.pitch_status))
            else:
                slot = self.slot_of.get(action.note_id)
                if slot is None or slot in changed:
                    continue
                changed.add(slot)
                _, _, began = self.notes[slot]
                self.notes[slot] = (action.midi_note,
                                    action.pitch_status or "estimated", began)
                tokens.append(self.vocab.change_base + slot)
                tokens.append(self.vocab.pitch_token(action.midi_note,
                                                     action.pitch_status))
        return tokens


def build_windows(groups: Sequence[cx.Group], vocab: Vocab, *,
                  context: int = 1024) -> list[Window]:
    """Group-aligned training windows with state snapshots.

    A window never splits a group, an action, or a delta escape, and every
    window -- initial, mid-track or inference rollover -- is built by the one
    convention: begin immediately before a group's actions, with the snapshot
    describing the state at that instant.
    """
    windows: list[Window] = []
    state = StreamState(vocab)
    index = 0
    while index < len(groups):
        snapshot = state.snapshot()
        start_tick = state.tick
        head = [vocab.BOS]
        if snapshot:
            head += encode_snapshot(snapshot, vocab)
        tokens = list(head)
        first_group = index
        # Replay from a copy so the window's own emission does not advance the
        # outer state past what this window covers.
        local = StreamState(vocab)
        local.tick = state.tick
        local.free = list(state.free)
        local.notes = dict(state.notes)
        local.slot_of = dict(state.slot_of)
        last = index - 1
        while index < len(groups):
            group = groups[index]
            piece: list[int] = []
            if index != first_group:
                piece += vocab.encode_delta(group.delta) if group.delta else []
                if not group.delta:
                    break      # a non-leading group must advance time
            probe = StreamState(vocab)
            probe.tick = local.tick
            probe.free = list(local.free)
            probe.notes = dict(local.notes)
            probe.slot_of = dict(local.slot_of)
            piece += probe.emit(group, first=(index == first_group))
            if len(tokens) + len(piece) + 1 > context:     # +1 for EOS
                break
            tokens += piece
            local = probe
            last = index
            index += 1
        if last < first_group:
            # One group alone exceeds the budget; emit it rather than stall.
            group = groups[first_group]
            tokens = list(head) + local.emit(group, first=True)
            last = first_group
            index = first_group + 1
        tokens.append(vocab.EOS)
        windows.append(Window(tokens=tokens, snapshot=snapshot,
                              first_group=first_group, last_group=last,
                              start_tick=start_tick,
                              n_snapshot_tokens=len(head) - 1))
        # Advance the outer state across exactly the groups this window covered.
        for position in range(first_group, last + 1):
            state.emit(groups[position], first=(position == 0))
    return windows
