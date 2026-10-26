"""Phrase structure: the thing the melody has been missing.

Measured on the v1 composer, section A of every style:

    style      bars  distinct rhythms  rest%  bar-ends-on-tonic
    action        8                 1    0.0                1/8
    boss          8                 1    0.0                1/8
    tension       8                 1    0.0                2/8
    town          8                 1    0.0                2/8

One distinct rhythm across eight bars. Every bar of `boss` attacks rows
0, 2, 4, 8, 10, 12 and no others. No rest anywhere, so the line never breathes.
Bar endings land on the tonic once or twice in eight, by accident rather than
by cadence.

That is not a melody with a weak phrase structure; it is a melody with no
phrase structure. The motif is stated, re-anchored on a random chord tone, and
restated every bar until the section ends.

What a phrase needs, and what this module supplies:

**A unit larger than the bar.** Four bars group into a phrase, two phrases into
an eight-bar period. The bar stops being the unit of repetition.

**An arc.** Within a phrase the motif is stated, stated again, pushed up to a
registral peak, and then resolved. The peak lands in bar 3 of 4 rather than
wherever the random walk happened to go.

**A cadence.** The phrase ends on a long note chosen for its harmonic function:
the antecedent leaves the question open on a chord tone that is not the tonic,
the consequent closes on the tonic. This is the difference between a line that
arrives and one that merely stops.

**A breath.** The cadence note is followed by silence. Rests are what make the
next phrase sound like a new thought rather than more of the same.

Nothing here generates pitches or picks rhythms -- `render_melody` still does
that. This module only says what each bar is *for*, which is the decision that
was missing.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Bars to a phrase. Four is the default because the eight-bar period built
#: from two of them is the commonest shape in the corpus this targets, and
#: because the section length is already eight.
PHRASE_BARS = 4

#: What a bar does inside its phrase.
#:
#: `state`   -- play the motif as it stands.
#: `vary`    -- play it altered, so the repeat is not literal.
#: `climb`   -- push it up the scale to the phrase's registral peak.
#: `cadence` -- abandon the motif's rhythm for a long arrival note, then rest.
ROLES = ("state", "vary", "climb", "cadence")


@dataclass(frozen=True)
class BarRole:
    """What one bar of the melody is for."""

    #: Index of the bar within the section.
    bar: int
    #: Index of the phrase within the section, and of the bar within it.
    phrase: int
    position: int
    role: str
    #: `None` outside a cadence bar; otherwise "open" or "closed".
    cadence: str | None = None
    #: Rows of silence to leave at the end of the bar. The breath.
    rest_rows: int = 0
    #: Degrees to lift the motif by, for the climb.
    lift: int = 0

    @property
    def is_cadence(self) -> bool:
        return self.role == "cadence"

    def to_dict(self) -> dict[str, object]:
        return {
            "bar": self.bar,
            "phrase": self.phrase,
            "position": self.position,
            "role": self.role,
            "cadence": self.cadence,
            "rest_rows": self.rest_rows,
            "lift": self.lift,
        }


def _phrase_roles(
    phrase_bars: int, *, final: bool, cadence_rest: int, lift: int
) -> list[tuple[str, str | None, int, int]]:
    """Roles for one phrase as (role, cadence, rest_rows, lift).

    A four-bar phrase is state / state / climb / cadence: the repeat makes the
    motif memorable, the climb gives the phrase somewhere to go, and the
    cadence gives it somewhere to land. Shorter phrases drop the middle.
    """
    closing = "closed" if final else "open"
    if phrase_bars <= 1:
        return [("cadence", closing, cadence_rest, 0)]
    if phrase_bars == 2:
        return [("state", None, 0, 0), ("cadence", closing, cadence_rest, 0)]
    if phrase_bars == 3:
        return [
            ("state", None, 0, 0),
            ("climb", None, 0, lift),
            ("cadence", closing, cadence_rest, 0),
        ]
    # Four or more: state, then vary for any middle bars, then climb, cadence.
    roles: list[tuple[str, str | None, int, int]] = [("state", None, 0, 0)]
    for _ in range(phrase_bars - 3):
        roles.append(("vary", None, 0, 0))
    roles.append(("climb", None, 0, lift))
    roles.append(("cadence", closing, cadence_rest, 0))
    return roles


def plan_phrases(
    n_bars: int,
    *,
    phrase_bars: int = PHRASE_BARS,
    cadence_rest: int = 4,
    lift: int = 2,
    final_phrase_closes: bool = True,
) -> list[BarRole]:
    """Assign a role to every bar of a section.

    The LAST phrase of the section closes on the tonic and the others stay
    open, which is what makes an eight-bar period sound like one sentence
    rather than two unrelated halves. A section whose bar count is not a
    multiple of `phrase_bars` gets a short final phrase rather than a truncated
    cadence -- the arrival matters more than the symmetry.
    """
    if n_bars <= 0:
        return []
    phrase_bars = max(1, phrase_bars)

    # Split into phrases, giving any remainder to the last one so the cadence
    # is never cut off mid-arrival.
    spans: list[int] = []
    remaining = n_bars
    while remaining > 0:
        if remaining < 2 * phrase_bars:
            spans.append(remaining)
            break
        spans.append(phrase_bars)
        remaining -= phrase_bars

    roles: list[BarRole] = []
    bar = 0
    for phrase_index, span in enumerate(spans):
        final = phrase_index == len(spans) - 1 and final_phrase_closes
        for position, (role, cadence, rest, bar_lift) in enumerate(
            _phrase_roles(span, final=final, cadence_rest=cadence_rest, lift=lift)
        ):
            roles.append(
                BarRole(
                    bar=bar,
                    phrase=phrase_index,
                    position=position,
                    role=role,
                    cadence=cadence,
                    rest_rows=rest,
                    lift=bar_lift,
                )
            )
            bar += 1
    return roles


def describe(roles: list[BarRole]) -> str:
    """One line per phrase, for a report."""
    if not roles:
        return "(no bars)"
    lines = []
    for phrase in sorted({r.phrase for r in roles}):
        bars = [r for r in roles if r.phrase == phrase]
        shape = " ".join(
            f"{r.role}{'(' + r.cadence + ')' if r.cadence else ''}" for r in bars
        )
        lines.append(f"phrase {phrase}: {shape}")
    return "\n".join(lines)
