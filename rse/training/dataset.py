"""Turning a corpus of compositions into training windows.

Proposal 5.4 asks for 8-16 bar windows that preserve phrase and loop
boundaries. So windows are cut on *bar* lines, never mid-bar, and a window
carries its own conditioning header — a model conditioned on "16 bars" must be
trained on sequences that really are 16 bars.

Splits come from the corpus, not from a random shuffle here. NES-MDB ships
composer-disjoint splits, and reshuffling them would quietly invite the
memorization the originality screening is meant to catch.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from rse.events.schema import Composition, Noise, Note
from rse.events.structure import SEGMENT_BARS, derive_sections
from rse.jsonio import read_jsonl, write_jsonl
from rse.tokenizers.tokenizer import Conditioning, RseTokenizer, conditioning_from_composition

#: Split names used throughout training.
TRAIN = "train"
VALID = "valid"
TEST = "test"
SPLITS = (TRAIN, VALID, TEST)


@dataclass
class Window:
    """One training example: a token sequence and where it came from."""

    ids: list[int]
    split: str
    composition_id: str
    start_bar: int
    n_bars: int
    #: Set when this window is one piece of an over-long window that was split
    #: at bar boundaries. A fragment is shorter than the window a conditioning
    #: label was computed for, so a label join must exclude it rather than
    #: inherit the parent's label -- Gate 3A measured that cropping changes
    #: motif class for 1114 of 3900 windows.
    is_fragment: bool = False
    source_start_bar: int | None = None
    source_n_bars: int | None = None
    fragment_index: int = 0
    n_fragments: int = 1

    def __len__(self) -> int:
        return len(self.ids)

    def to_dict(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "ids": self.ids,
            "split": self.split,
            "composition_id": self.composition_id,
            "start_bar": self.start_bar,
            "n_bars": self.n_bars,
        }
        if self.is_fragment:
            row.update(
                {
                    "is_fragment": True,
                    "source_start_bar": self.source_start_bar,
                    "source_n_bars": self.source_n_bars,
                    "fragment_index": self.fragment_index,
                    "n_fragments": self.n_fragments,
                    "labels_inherited": False,
                }
            )
        return row

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Window:
        return cls(
            ids=list(data["ids"]),
            split=data["split"],
            composition_id=data["composition_id"],
            start_bar=int(data["start_bar"]),
            n_bars=int(data["n_bars"]),
            is_fragment=bool(data.get("is_fragment", False)),
            source_start_bar=(
                int(data["source_start_bar"]) if data.get("source_start_bar") is not None else None
            ),
            source_n_bars=(
                int(data["source_n_bars"]) if data.get("source_n_bars") is not None else None
            ),
            fragment_index=int(data.get("fragment_index", 0)),
            n_fragments=int(data.get("n_fragments", 1)),
        )


@dataclass
class WindowStats:
    n_compositions: int = 0
    n_windows: int = 0
    n_skipped_empty: int = 0
    max_tokens: int = 0
    total_tokens: int = 0
    per_split: dict[str, int] = field(default_factory=dict)

    @property
    def mean_tokens(self) -> float:
        return self.total_tokens / self.n_windows if self.n_windows else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_compositions": self.n_compositions,
            "n_windows": self.n_windows,
            "n_skipped_empty": self.n_skipped_empty,
            "max_tokens": self.max_tokens,
            "mean_tokens": round(self.mean_tokens, 1),
            "per_split": dict(sorted(self.per_split.items())),
        }


def slice_bars(
    composition: Composition,
    start_bar: int,
    n_bars: int,
    *,
    beats_per_bar: int = 4,
    ticks_per_bar: float | None = None,
) -> Composition:
    """Extract `n_bars` bars starting at `start_bar`, rebased to tick zero.

    Notes are cut on bar lines rather than being allowed to hang over the end,
    so every window is self-contained and can be rendered on its own.
    """
    per_bar = ticks_per_bar or composition.ticks_per_beat * beats_per_bar
    start_tick = int(round(start_bar * per_bar))
    end_tick = int(round(start_tick + n_bars * per_bar))

    events = []
    for event in composition.sorted_events():
        if event.tick < start_tick or event.tick >= end_tick:
            continue
        shifted_tick = event.tick - start_tick
        if isinstance(event, Note | Noise):
            duration = min(event.duration, end_tick - event.tick)
            if duration <= 0:
                continue
            from dataclasses import replace

            events.append(replace(event, tick=shifted_tick, duration=duration))
        else:
            from dataclasses import replace

            events.append(replace(event, tick=shifted_tick))

    window = Composition(
        hardware_profile=composition.hardware_profile,
        ticks_per_beat=composition.ticks_per_beat,
        events=events,
        tempo_map=list(composition.tempo_map[:1]),
        time_signatures=list(composition.time_signatures[:1]),
        # A slice of a loop is not itself a loop unless it is the whole thing.
        loop_start_tick=0 if start_bar == 0 and composition.loop_start_tick is not None else None,
        loop_end_tick=(
            int(round(n_bars * per_bar))
            if composition.loop_end_tick is not None
            and composition.loop_end_tick <= end_tick
            and start_bar == 0
            else None
        ),
        source_metadata=dict(composition.source_metadata),
    )
    window.normalize()
    return window


def bar_count(
    composition: Composition, *, beats_per_bar: int = 4, ticks_per_bar: float | None = None
) -> int:
    """Bars in a composition.

    `ticks_per_bar` comes from the fitted row when the caller has one. The
    nominal fallback, `ticks_per_beat * beats_per_bar`, is not a musical bar
    for a sampled source: on NES-MDB's exprsco release it is 16 frames, or
    0.67 s, against roughly 4 s for a real bar.
    """
    if not composition.events:
        return 0
    per_bar = ticks_per_bar or composition.ticks_per_beat * beats_per_bar
    return max(1, int(-(-composition.end_tick // per_bar)))


def windows_from_composition(
    composition: Composition,
    tokenizer: RseTokenizer,
    *,
    split: str,
    window_bars: int = 16,
    hop_bars: int | None = None,
    composition_id: str | None = None,
    context_tokens: int = 0,
) -> list[Window]:
    """Cut one composition into overlapping bar-aligned windows.

    `context_tokens` enables bar-boundary splitting of any window that would
    not fit. Zero leaves windows whole at any length, which is the historical
    behaviour; a positive value guarantees no window in the result exceeds it,
    with the overflow split rather than truncated or dropped.
    """
    beats_per_bar = tokenizer.config.beats_per_bar
    # Cut windows on the piece's own bars. Slicing on the nominal
    # ticks_per_beat * 4 put roughly 2.7 fitted bars of music in a window
    # labelled Length_16, and the encoder padded the rest with empty bars.
    fit = tokenizer.fit_grid(composition)
    ticks_per_bar = fit.step_ticks * tokenizer.config.steps_per_bar
    total_bars = bar_count(composition, beats_per_bar=beats_per_bar, ticks_per_bar=ticks_per_bar)
    if total_bars == 0:
        return []
    hop = hop_bars or window_bars
    identifier = composition_id or str(composition.source_metadata.get("composition_id", "unknown"))

    out: list[Window] = []
    # A composition shorter than one window still yields that window, padded
    # out by the bar tokens the tokenizer emits for empty bars.
    starts = list(range(0, max(1, total_bars - window_bars + 1), hop))
    for start in starts:
        sliced = slice_bars(
            composition,
            start,
            window_bars,
            beats_per_bar=beats_per_bar,
            ticks_per_bar=ticks_per_bar,
        )
        if not sliced.sounding_events():
            continue
        # NES-MDB has no form annotation, so the window's sections are inferred
        # from its own repetition. Derived per window rather than per
        # composition, because a window's form is what its token stream claims.
        if not sliced.sections and tokenizer.config.emit_sections:
            # Four segments per window. The form analysis sets its
            # threshold from the median pairwise similarity and needs at
            # least three comparisons to have a baseline, so three
            # segments is the floor and two produce a single comparison
            # and no form at all. With the default 8-bar segment an 8-bar
            # window held one segment and every window emitted a lone
            # Section_A, making the token a constant and its ablation
            # meaningless.
            sliced.sections = derive_sections(
                sliced,
                segment_bars=max(1, min(SEGMENT_BARS, window_bars // 4)),
                beats_per_bar=beats_per_bar,
                n_bars=window_bars,
            )
        # Pass the fit: without it the tempo falls back to the source file's
        # own value while positions and durations are counted in fitted rows,
        # so the Tempo token describes a different grid from the one the
        # sequence uses and `decode` plays the window at the wrong speed.
        base = conditioning_from_composition(composition, tokenizer.config, fit=fit)
        conditioning = Conditioning(
            mood=base.mood,
            energy=base.energy,
            tempo_bpm=base.tempo_bpm,
            n_bars=window_bars,
        )
        encoded = tokenizer.encode(
            sliced, conditioning=conditioning, fit=replace(fit, phase_ticks=0.0)
        )
        if context_tokens and len(encoded.ids) > context_tokens:
            # Over-long windows carry roughly three times the events of those
            # that fit, so dropping them would clip the densest music out of
            # the corpus. Split instead, at bar boundaries, losing nothing.
            from rse.training.fragments import split_window, verify_split

            window_fit = replace(fit, phase_ticks=0.0)
            fragments = split_window(
                sliced,
                tokenizer,
                fit=window_fit,
                context=context_tokens,
                window_bars=window_bars,
                composition_id=identifier,
                source_start_bar=start,
                split=split,
            )
            verify_split(
                sliced,
                fragments,
                fit=window_fit,
                tokenizer=tokenizer,
                window_bars=window_bars,
            )
            out.extend(
                Window(
                    ids=fragment.ids,
                    split=split,
                    composition_id=identifier,
                    start_bar=fragment.absolute_start_bar,
                    n_bars=fragment.n_bars,
                    is_fragment=True,
                    source_start_bar=start,
                    source_n_bars=window_bars,
                    fragment_index=fragment.fragment_index,
                    n_fragments=fragment.n_fragments,
                )
                for fragment in fragments
            )
            continue
        out.append(
            Window(
                ids=encoded.ids,
                split=split,
                composition_id=identifier,
                start_bar=start,
                n_bars=window_bars,
            )
        )
    return out


def build_windows(
    compositions: list[tuple[Composition, str, str]],
    tokenizer: RseTokenizer,
    *,
    window_bars: int = 16,
    hop_bars: int | None = None,
    context_tokens: int = 0,
) -> tuple[list[Window], WindowStats]:
    """Cut a corpus of `(composition, split, id)` triples into windows."""
    stats = WindowStats()
    windows: list[Window] = []
    for composition, split, identifier in compositions:
        stats.n_compositions += 1
        produced = windows_from_composition(
            composition,
            tokenizer,
            split=split,
            window_bars=window_bars,
            hop_bars=hop_bars,
            composition_id=identifier,
            context_tokens=context_tokens,
        )
        if not produced:
            stats.n_skipped_empty += 1
            continue
        for window in produced:
            stats.n_windows += 1
            stats.total_tokens += len(window)
            stats.max_tokens = max(stats.max_tokens, len(window))
            stats.per_split[split] = stats.per_split.get(split, 0) + 1
        windows.extend(produced)
    return windows, stats


def save_windows(path: Path, windows: list[Window]) -> Path:
    return write_jsonl(Path(path), (w.to_dict() for w in windows))


def load_windows(path: Path) -> list[Window]:
    return [Window.from_dict(row) for row in read_jsonl(Path(path))]


def split_windows(windows: list[Window]) -> dict[str, list[Window]]:
    out: dict[str, list[Window]] = {split: [] for split in SPLITS}
    for window in windows:
        out.setdefault(window.split, []).append(window)
    return out


def assign_splits(
    identifiers: list[str],
    *,
    seed: int = 0,
    valid_fraction: float = 0.1,
    test_fraction: float = 0.1,
) -> dict[str, str]:
    """Split a corpus by composition, never by window.

    Windows from one composition overlap, so splitting at window level would
    put near-identical sequences in train and validation and make the
    validation loss meaningless.
    """
    if valid_fraction + test_fraction >= 1.0:
        raise ValueError("valid and test fractions must leave something for training")
    unique = sorted(set(identifiers))
    rng = random.Random(seed)
    rng.shuffle(unique)
    n_valid = max(1, int(len(unique) * valid_fraction)) if len(unique) > 2 else 0
    n_test = max(1, int(len(unique) * test_fraction)) if len(unique) > 2 else 0
    assignment: dict[str, str] = {}
    for index, identifier in enumerate(unique):
        if index < n_valid:
            assignment[identifier] = VALID
        elif index < n_valid + n_test:
            assignment[identifier] = TEST
        else:
            assignment[identifier] = TRAIN
    return assignment
