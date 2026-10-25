"""Splitting an over-long window at bar boundaries, losing nothing.

About 1.3% of 8-bar windows encode to more than 3072 tokens under the frozen
representation. Dropping them is not a neutral choice: those windows carry a
median 1150 events against 268 for the ones that fit, so excluding them would
systematically remove the densest music -- the arpeggio and passage material
the codec was chosen to preserve.

So they are split instead. The rules that make a split safe:

**Fragments are cut only at bar boundaries.** A bar is the smallest unit whose
start a `Bar` token can name, so cutting inside one would produce a fragment
whose first event is not at a position the grammar can express.

**Events belong to the fragment containing their onset, and to exactly one.**
A note whose duration runs past its fragment's last bar keeps its full
duration: duration is a token, not an end position, so decode reproduces the
audible continuation without the note having to be clipped or duplicated. That
is what lets the union of fragments hold every original event exactly once.

**Labels are not inherited.** A fragment is shorter than the window its
conditioning label described, and Gate 3A measured that cropping changes the
label: at 1024 tokens, 1114 of 3900 cropped windows changed motif class. Every
fragment is marked so a label join cannot silently match it to its parent's
label.

**A bar that will not fit on its own is an error.** Truncating it silently is
the failure mode this module exists to prevent, so it raises with the bar
index and its token count instead.

The splitter is greedy and deterministic: take as many complete consecutive
bars as fit, emit, continue. Greedy maximizes bars per fragment, which keeps
the count of fragments down and each fragment as close to the original musical
unit as possible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rse.events.schema import Composition, Noise, Note
from rse.tokenizers.tokenizer import GridFit, RseTokenizer


class FragmentError(RuntimeError):
    """A window cannot be split to fit the context."""


@dataclass(frozen=True)
class Fragment:
    """One bar-aligned piece of an over-long window."""

    #: Bars relative to the parent window.
    start_bar: int
    n_bars: int
    ids: list[int] = field(default_factory=list)
    #: Provenance: which window this came from and where it sat.
    composition_id: str = ""
    source_start_bar: int = 0
    source_n_bars: int = 0
    fragment_index: int = 0
    n_fragments: int = 1
    split: str = ""

    @property
    def is_fragment(self) -> bool:
        """True when this is part of a split window, not a whole one."""
        return self.n_fragments > 1

    @property
    def absolute_start_bar(self) -> int:
        """Where this fragment starts in the parent composition."""
        return self.source_start_bar + self.start_bar

    def to_dict(self) -> dict[str, Any]:
        return {
            "ids": self.ids,
            "split": self.split,
            "composition_id": self.composition_id,
            # The key a cache row is identified by. For a fragment this is its
            # own absolute start, never the parent's, so a label manifest built
            # for whole windows cannot match it by accident.
            "start_bar": self.absolute_start_bar,
            "n_bars": self.n_bars,
            "source_start_bar": self.source_start_bar,
            "source_n_bars": self.source_n_bars,
            "fragment_index": self.fragment_index,
            "n_fragments": self.n_fragments,
            "is_fragment": self.is_fragment,
            "labels_inherited": False,
        }


def bar_of(
    tick: int,
    *,
    fit: GridFit,
    rows_per_bar: int,
    divisions: int,
    n_bars: int | None = None,
) -> int:
    """Which bar a tick falls in, on this window's own grid.

    `slot_of` rounds to the nearest slot, so an event sitting just inside the
    window's final bar can round *up* past it. Without `n_bars` to clamp
    against, such an event lands in a bar no fragment covers and disappears --
    `verify_split` caught exactly one event going missing out of 1504 this
    way. It belongs to the last bar, because it is inside the window.
    """
    slot = fit.slot_of(tick, divisions)
    bar = slot // (rows_per_bar * divisions)
    if n_bars is not None:
        bar = max(0, min(bar, n_bars - 1))
    return bar


def events_by_bar(
    composition: Composition,
    *,
    fit: GridFit,
    rows_per_bar: int,
    divisions: int,
    n_bars: int | None = None,
) -> dict[int, list[Note | Noise]]:
    """Group events by the bar their onset falls in.

    By onset, so every event lands in exactly one bar however long it sounds.
    `n_bars` clamps a rounding overshoot into the final bar; see `bar_of`.
    """
    out: dict[int, list[Note | Noise]] = {}
    for event in composition.sounding_events():
        bar = bar_of(
            event.tick,
            fit=fit,
            rows_per_bar=rows_per_bar,
            divisions=divisions,
            n_bars=n_bars,
        )
        out.setdefault(bar, []).append(event)
    return out


def _slice(
    composition: Composition,
    events: list[Note | Noise],
    *,
    shift_ticks: int,
) -> Composition:
    """A composition holding `events`, rebased so its first bar is bar 0."""
    moved = [type(e)(**{**vars(e), "tick": e.tick - shift_ticks}) for e in events]
    return Composition(
        hardware_profile=composition.hardware_profile,
        ticks_per_beat=composition.ticks_per_beat,
        events=moved,
        tempo_map=list(composition.tempo_map),
        time_signatures=list(composition.time_signatures),
    )


def split_window(
    composition: Composition,
    tokenizer: RseTokenizer,
    *,
    fit: GridFit,
    context: int,
    window_bars: int,
    composition_id: str = "",
    source_start_bar: int = 0,
    split: str = "",
) -> list[Fragment]:
    """Cut `composition` into bar-aligned fragments that each fit `context`.

    Returns a single whole-window fragment when the window already fits, so a
    caller can use this unconditionally.
    """
    rows_per_bar = tokenizer.config.steps_per_bar
    divisions = tokenizer.config.onset_divisions
    bar_ticks = fit.step_ticks * rows_per_bar

    whole = tokenizer.encode(composition, fit=fit)
    if len(whole.ids) <= context:
        return [
            Fragment(
                start_bar=0,
                n_bars=window_bars,
                ids=list(whole.ids),
                composition_id=composition_id,
                source_start_bar=source_start_bar,
                source_n_bars=window_bars,
                fragment_index=0,
                n_fragments=1,
                split=split,
            )
        ]

    grouped = events_by_bar(
        composition,
        fit=fit,
        rows_per_bar=rows_per_bar,
        divisions=divisions,
        n_bars=window_bars,
    )

    pieces: list[tuple[int, int, list[int]]] = []
    bar = 0
    while bar < window_bars:
        taken = 0
        accepted: list[int] | None = None
        # Grow the fragment one bar at a time, keeping the last encoding that
        # fit. Encoding per step rather than estimating, because token count is
        # not linear in events -- a Position token is shared by the notes under
        # it.
        while bar + taken < window_bars:
            candidate_bars = taken + 1
            events: list[Note | Noise] = []
            for offset in range(candidate_bars):
                events.extend(grouped.get(bar + offset, ()))
            if not events:
                taken = candidate_bars
                accepted = accepted if accepted is not None else []
                continue
            shift = int(round(bar * bar_ticks))
            encoded = tokenizer.encode(_slice(composition, events, shift_ticks=shift), fit=fit)
            if len(encoded.ids) > context:
                if candidate_bars == 1:
                    raise FragmentError(
                        f"bar {source_start_bar + bar} alone encodes to "
                        f"{len(encoded.ids)} tokens, over the {context} context; "
                        f"refusing to truncate it"
                    )
                break
            taken = candidate_bars
            accepted = list(encoded.ids)

        if taken == 0 or accepted is None:
            raise FragmentError(f"made no progress splitting bars {source_start_bar + bar} onward")
        pieces.append((bar, taken, accepted))
        bar += taken

    return [
        Fragment(
            start_bar=start,
            n_bars=n,
            ids=ids,
            composition_id=composition_id,
            source_start_bar=source_start_bar,
            source_n_bars=window_bars,
            fragment_index=index,
            n_fragments=len(pieces),
            split=split,
        )
        for index, (start, n, ids) in enumerate(pieces)
    ]


def verify_split(
    composition: Composition,
    fragments: list[Fragment],
    *,
    fit: GridFit,
    tokenizer: RseTokenizer,
    window_bars: int,
) -> dict[str, Any]:
    """Assert the split lost nothing, duplicated nothing and covers every bar.

    Raises `FragmentError` on any violation. Called by the cache builder rather
    than left to a test, because the failure it guards against is silent.
    """
    rows_per_bar = tokenizer.config.steps_per_bar
    divisions = tokenizer.config.onset_divisions
    grouped = events_by_bar(
        composition,
        fit=fit,
        rows_per_bar=rows_per_bar,
        divisions=divisions,
        n_bars=window_bars,
    )

    covered: list[int] = []
    for fragment in sorted(fragments, key=lambda f: f.start_bar):
        covered.extend(range(fragment.start_bar, fragment.start_bar + fragment.n_bars))

    expected = list(range(window_bars))
    if covered != expected:
        duplicated = sorted({b for b in covered if covered.count(b) > 1})
        missing = sorted(set(expected) - set(covered))
        raise FragmentError(
            f"fragments cover bars {covered[:12]}... expected {expected[:12]}...; "
            f"duplicated {duplicated}, missing {missing}"
        )

    # Events are grouped by onset bar, so bar coverage being exact means event
    # coverage is exact. Counting anyway: the grouping is the thing that would
    # be wrong if `bar_of` and the splitter disagreed about the grid.
    n_events = sum(len(v) for v in grouped.values())
    n_in_fragments = sum(
        len(grouped.get(b, ()))
        for fragment in fragments
        for b in range(fragment.start_bar, fragment.start_bar + fragment.n_bars)
    )
    if n_in_fragments != n_events:
        raise FragmentError(f"fragments hold {n_in_fragments} events, the window has {n_events}")

    over = [f for f in fragments if not f.ids]
    if over:
        raise FragmentError(f"{len(over)} fragments encoded to nothing")

    return {
        "n_fragments": len(fragments),
        "n_events": n_events,
        "bars_covered": len(covered),
        "max_tokens": max(len(f.ids) for f in fragments),
    }
