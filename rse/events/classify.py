"""What kind of thing each source event is.

NES-MDB is a register-write transcription, so a single musical note can appear
as many events: an arpeggio chord is one note per frame cycling through three
pitches, a vibrato is an alternation a semitone or two wide, and a tremolo is
the same pitch retriggered. Any statement of the form "the codec deletes N% of
notes" is meaningless without this distinction, because deleting the second,
third and fourth frames of an arpeggio costs a timbre, while deleting a genuine
attack costs a note.

The grouping threshold is structural rather than fitted. Two events closer
together than half a tracker row cannot occupy different rows, so they are
within-row effects by construction -- no percentile of any distribution is
involved, and nothing here is calibrated against validation or test data.

The taxonomy deliberately keeps `passage` separate from the effect classes.
A rapid run of unrelated pitches might be a genuine fast melodic line, and
folding it into `arpeggio` would let a codec claim credit for discarding real
music.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from statistics import median

from rse.events.schema import Composition, Noise, Note

#: A genuine new onset: the first event of any group, and every isolated event.
ATTACK = "attack"
#: Rapid cycle through three or more pitches, or a wide two-pitch alternation.
#: On the NES this is how a chord is played on one monophonic channel.
ARPEGGIO = "arpeggio"
#: Rapid narrow alternation -- pitch modulation, not separate notes.
VIBRATO = "vibrato"
#: The same value retriggered rapidly: tremolo, echo, or a held note rewritten.
RETRIGGER = "retrigger"
#: Rapid succession that matches none of the above. Possibly real fast melody,
#: so it is never counted as an effect.
PASSAGE = "passage"

CLASSES = (ATTACK, ARPEGGIO, VIBRATO, RETRIGGER, PASSAGE)

#: Events within this fraction of a row of each other belong to one group.
#: Half a row is the structural boundary: anything closer cannot be a separate
#: tracker row.
RAPID_ROW_FRACTION = 0.5

#: Alternations up to this wide are pitch modulation rather than harmony.
VIBRATO_SEMITONES = 2

#: Intervals at least this wide are harmonic. The earlier measurement of
#: within-row events found a median separation of 9 semitones with 81% at 3 or
#: more, which is what makes 3 the boundary rather than a guess.
ARPEGGIO_SEMITONES = 3


@dataclass
class ClassifiedEvents:
    """Per-event classes, in the order the events were given."""

    labels: list[str] = field(default_factory=list)
    #: Index of the group each event belongs to, so a caller can ask which
    #: events a codec collapsed into one another.
    group_of: list[int] = field(default_factory=list)
    rapid_ticks: float = 0.0

    def counts(self) -> dict[str, int]:
        counter = Counter(self.labels)
        return {name: counter.get(name, 0) for name in CLASSES}

    def attack_indices(self) -> list[int]:
        return [i for i, label in enumerate(self.labels) if label == ATTACK]

    def to_dict(self) -> dict[str, object]:
        return {
            "counts": self.counts(),
            "n_events": len(self.labels),
            "rapid_ticks": self.rapid_ticks,
            "thresholds": {
                "rapid_row_fraction": RAPID_ROW_FRACTION,
                "vibrato_semitones": VIBRATO_SEMITONES,
                "arpeggio_semitones": ARPEGGIO_SEMITONES,
            },
        }


def _classify_group(events: list[Note | Noise]) -> list[str]:
    """Label one group of rapid events. The first is always an attack."""
    if len(events) == 1:
        return [ATTACK]

    values = [e.pitch if isinstance(e, Note) else e.value for e in events]
    distinct = set(values)
    if len(distinct) == 1:
        rest = RETRIGGER
    elif any(isinstance(e, Noise) for e in events):
        # A noise period is not a pitch, so an interval between two of them is
        # not an interval. Calling a noise run a vibrato or an arpeggio would
        # be a category error, and it would inflate whichever effect class the
        # percussion channel happened to land in.
        rest = PASSAGE
    else:
        steps = [abs(b - a) for a, b in zip(values, values[1:], strict=False) if b != a]
        typical = median(steps) if steps else 0
        widest = max(steps) if steps else 0
        if widest <= VIBRATO_SEMITONES:
            rest = VIBRATO
        elif typical >= ARPEGGIO_SEMITONES:
            rest = ARPEGGIO
        else:
            rest = PASSAGE
    return [ATTACK] + [rest] * (len(events) - 1)


def classify_events(
    composition: Composition,
    *,
    row_ticks: float,
    events: list[Note | Noise] | None = None,
) -> ClassifiedEvents:
    """Classify every sounding event against the piece's own row.

    `row_ticks` is the fitted tracker row from `detect_pulse`, not a nominal
    grid step: the whole point is to ask whether two events could be separate
    rows in the music as written.
    """
    if row_ticks <= 0:
        raise ValueError(f"row_ticks must be positive, got {row_ticks}")
    source = composition.sounding_events() if events is None else events
    rapid = row_ticks * RAPID_ROW_FRACTION

    order = sorted(range(len(source)), key=lambda i: (source[i].channel_id, source[i].tick))
    labels: list[str] = [ATTACK] * len(source)
    group_of: list[int] = [-1] * len(source)

    group: list[int] = []
    next_group = 0
    for position, index in enumerate(order):
        event = source[index]
        if group:
            previous = source[group[-1]]
            same_channel = previous.channel_id == event.channel_id
            if not same_channel or event.tick - previous.tick > rapid:
                for slot, member in zip(
                    _classify_group([source[i] for i in group]), group, strict=True
                ):
                    labels[member] = slot
                    group_of[member] = next_group
                next_group += 1
                group = []
        group.append(index)
        if position == len(order) - 1:
            for slot, member in zip(
                _classify_group([source[i] for i in group]), group, strict=True
            ):
                labels[member] = slot
                group_of[member] = next_group
            next_group += 1

    return ClassifiedEvents(labels=labels, group_of=group_of, rapid_ticks=rapid)
