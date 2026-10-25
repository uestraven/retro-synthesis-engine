"""Branch B: which representation preserves the most useful information?

Deterministic throughout. Every figure is a property of encode and decode on
the same source pieces, so the whole ranking reproduces from the corpus with no
training and no sampling.

What is compared, and against what
----------------------------------
Each codec is measured against the **original MIDI event stream** of the same
piece, cut at the same window boundaries, with the official splits untouched.
Train selects, validation confirms, and the test split is reserved for a single
frozen candidate -- so this script refuses to touch test unless explicitly
asked, and records which split produced every number.

Codec A is the baseline the current foundation was trained on. B and C buy
sub-row onsets in two different ways and must agree on fidelity, differing only
in token cost; D additionally refines durations. Reading B against C isolates
*cost*, and B/C against D isolates *articulation*.

Reading the table
-----------------
Event recall is reported per class from `rse.events.classify`, because "kept
77% of notes" means different things depending on whether the missing quarter
was arpeggio frames or attacks. A codec gets credit for preserving any event,
including `passage` events whose musical status is uncertain: the classes
interpret what survived, they do not decide what counts.

Onset and duration errors are in **seconds** on matched survivors only. The two
streams do not share a tick scale -- `decode` rebuilds time on a nominal grid
and recovers the row rate from the Tempo token -- so comparing raw ticks would
report a tempo-label difference as a timing error.

Token cost is the actual measured distribution, with the share of windows over
2,304 tokens, since a codec that cannot fit a window in context is not a
candidate however faithful it is.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any

from rse.codec.benchmark import (
    FidelityMetrics,
    enforce_monophony,
    measure_fidelity,
    overlapping_events,
    percentile,
    summarize,
)
from rse.events.classify import CLASSES, classify_events
from rse.events.pulse import detect_pulse
from rse.events.schema import Composition, Noise, Note
from rse.tokenizers.tokenizer import RseTokenizer
from rse.tokenizers.vocab import TokenizerConfig
from rse.training.corpus import corpus_from_release

#: The context the current architecture provides.
CONTEXT_LIMIT = 2304


@dataclass
class CodecSpec:
    """One codec configuration under test."""

    codec: str
    k: int

    @property
    def label(self) -> str:
        return self.codec if self.codec == "A" else f"{self.codec}{self.k}"

    def tokenizer(self) -> RseTokenizer:
        if self.codec == "A":
            return RseTokenizer(TokenizerConfig())
        return RseTokenizer(TokenizerConfig(codec=self.codec, sub_row_divisions=self.k))


@dataclass
class Accumulator:
    """Per-codec totals across pieces."""

    n_pieces: int = 0
    n_original: int = 0
    n_matched: int = 0
    n_missing: int = 0
    n_spurious: int = 0
    n_truncated: int = 0
    kept_by_class: Counter[str] = field(default_factory=Counter)
    total_by_class: Counter[str] = field(default_factory=Counter)
    onset_mae: list[float] = field(default_factory=list)
    onset_p95: list[float] = field(default_factory=list)
    duration_mae: list[float] = field(default_factory=list)
    ioi_retained: list[float] = field(default_factory=list)
    cv_original: list[float] = field(default_factory=list)
    cv_reconstructed: list[float] = field(default_factory=list)
    tokens: list[int] = field(default_factory=list)
    density_original: list[float] = field(default_factory=list)
    density_reconstructed: list[float] = field(default_factory=list)
    overlaps_before_repair: int = 0

    def add(self, metrics: FidelityMetrics, *, overlaps: int) -> None:
        self.n_pieces += 1
        self.n_original += metrics.n_original
        self.n_matched += metrics.n_matched
        self.n_missing += metrics.n_missing
        self.n_spurious += metrics.n_spurious
        self.n_truncated += metrics.n_truncated_for_monophony
        self.overlaps_before_repair += overlaps
        for name, count in metrics.n_by_class.items():
            self.total_by_class[name] += count
            self.kept_by_class[name] += round(metrics.recall_by_class.get(name, 0.0) * count)
        self.onset_mae.append(metrics.onset_mae_sec)
        self.onset_p95.append(metrics.onset_p95_sec)
        self.duration_mae.append(metrics.duration_mae_sec)
        self.ioi_retained.append(metrics.ioi_entropy_retained)
        self.cv_original.append(metrics.note_length_cv_original)
        self.cv_reconstructed.append(metrics.note_length_cv_reconstructed)
        self.tokens.append(metrics.n_tokens)

    def to_dict(self) -> dict[str, Any]:
        over = sum(1 for t in self.tokens if t > CONTEXT_LIMIT)
        return {
            "n_pieces": self.n_pieces,
            "n_original_events": self.n_original,
            "event_recall": round(self.n_matched / self.n_original, 5) if self.n_original else 0.0,
            "n_missing": self.n_missing,
            "n_spurious": self.n_spurious,
            "spurious_rate": round(self.n_spurious / self.n_original, 5)
            if self.n_original
            else 0.0,
            "n_truncated_for_monophony": self.n_truncated,
            "overlaps_before_repair": self.overlaps_before_repair,
            "recall_by_class": {
                name: round(self.kept_by_class[name] / self.total_by_class[name], 5)
                for name in CLASSES
                if self.total_by_class[name]
            },
            "n_by_class": {name: self.total_by_class[name] for name in CLASSES},
            "onset_mae_sec": summarize(self.onset_mae),
            "onset_p95_sec": summarize(self.onset_p95),
            "duration_mae_sec": summarize(self.duration_mae),
            "ioi_entropy_retained": summarize(self.ioi_retained),
            "note_length_cv": {
                "original_median": round(median(self.cv_original), 4) if self.cv_original else 0.0,
                "reconstructed_median": round(median(self.cv_reconstructed), 4)
                if self.cv_reconstructed
                else 0.0,
            },
            "tokens": {
                "median": round(median(self.tokens)) if self.tokens else 0,
                "p90": round(percentile([float(t) for t in self.tokens], 0.90)),
                "max": max(self.tokens) if self.tokens else 0,
                "over_context": over,
                "over_context_share": round(over / len(self.tokens), 5) if self.tokens else 0.0,
            },
            "melodic_density_per_8_bars": {
                "original_median": round(median(self.density_original), 2)
                if self.density_original
                else 0.0,
                "reconstructed_median": round(median(self.density_reconstructed), 2)
                if self.density_reconstructed
                else 0.0,
            },
        }


def _sec_per_tick(composition: Composition) -> float:
    """Seconds per tick from the composition's own tempo map."""
    end = composition.end_tick or 1
    seconds = composition.time.seconds(end)
    return seconds / end if seconds else 0.0


def _window(composition: Composition, start_tick: int, end_tick: int) -> Composition:
    """The events of one window, with time rebased to the window start."""
    events: list[Note | Noise] = []
    for event in composition.sounding_events():
        if start_tick <= event.tick < end_tick:
            events.append(type(event)(**{**vars(event), "tick": event.tick - start_tick}))
    out = Composition(
        hardware_profile=composition.hardware_profile,
        ticks_per_beat=composition.ticks_per_beat,
        events=list(events),
        tempo_map=list(composition.tempo_map),
        time_signatures=list(composition.time_signatures),
    )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="nesmdb")
    parser.add_argument("--release", default="midi")
    parser.add_argument(
        "--split",
        default="train",
        choices=("train", "valid", "test"),
        help="train selects, valid confirms; test is reserved for one frozen candidate",
    )
    parser.add_argument(
        "--allow-test",
        action="store_true",
        help="required to touch the reserved test split",
    )
    parser.add_argument("--limit", type=int, default=120)
    parser.add_argument("--window-bars", type=int, default=8)
    parser.add_argument("--k", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--codecs", nargs="+", default=["A", "B", "C", "D"])
    parser.add_argument("--out", type=Path, default=Path("codec_comparison.json"))
    args = parser.parse_args()

    if args.split == "test" and not args.allow_test:
        print(
            "error: the test split is reserved for the final frozen codec; "
            "pass --allow-test to override",
            flush=True,
        )
        return 2

    specs: list[CodecSpec] = []
    for codec in args.codecs:
        if codec == "A":
            specs.append(CodecSpec("A", 1))
        else:
            specs.extend(CodecSpec(codec, k) for k in args.k)

    print(f"corpus    : loading {args.split} split...", flush=True)
    entries, info = corpus_from_release(
        args.dataset, args.release, limit=args.limit, splits=(args.split,)
    )
    print(f"corpus    : {len(entries)} pieces", flush=True)

    tokenizers = {spec.label: spec.tokenizer() for spec in specs}
    accumulators = {spec.label: Accumulator() for spec in specs}
    n_declined = 0

    for index, (composition, _split, _piece_id) in enumerate(entries, 1):
        pulse = detect_pulse(composition)
        if not pulse.found or pulse.period_ticks <= 0:
            n_declined += 1
            continue
        row_ticks = pulse.period_ticks
        bar_ticks = row_ticks * 16
        window_ticks = bar_ticks * args.window_bars
        end = composition.end_tick
        starts = [int(round(s * window_ticks)) for s in range(max(1, int(end // window_ticks)))]
        original_spt = _sec_per_tick(composition)

        for start in starts:
            source = _window(composition, start, start + int(round(window_ticks)))
            events = source.sounding_events()
            if len(events) < 8:
                continue
            classes = classify_events(source, row_ticks=row_ticks, events=events).labels
            # The window's own grid, taken from the parent piece so every
            # codec and every window agrees about where the rows are.
            for spec in specs:
                tokenizer = tokenizers[spec.label]
                fit = type(tokenizer.fit_grid(source))(
                    step_ticks=row_ticks,
                    phase_ticks=0.0,
                    aligned=True,
                    period_sec=pulse.period_sec,
                )
                encoded = tokenizer.encode(source, fit=fit)
                decoded = tokenizer.decode(encoded.ids)
                overlaps = len(overlapping_events(decoded.composition))
                repaired, truncated = enforce_monophony(decoded.composition)
                reconstructed = repaired.sounding_events()
                recon_spt = _sec_per_tick(repaired)
                metrics = measure_fidelity(
                    events,
                    reconstructed,
                    classes=classes,
                    original_sec_per_tick=original_spt,
                    reconstructed_sec_per_tick=recon_spt,
                    row_sec=pulse.period_sec,
                    n_truncated=truncated,
                    n_tokens=len(encoded.ids),
                )
                acc = accumulators[spec.label]
                acc.add(metrics, overlaps=overlaps)
                acc.density_original.append(
                    sum(1 for e in events if isinstance(e, Note) and e.channel_id != "NO")
                )
                acc.density_reconstructed.append(
                    sum(1 for e in reconstructed if isinstance(e, Note) and e.channel_id != "NO")
                )
        if index % 20 == 0:
            print(f"          : {index}/{len(entries)} pieces", flush=True)

    payload = {
        "corpus": info.to_dict(),
        "split": args.split,
        "window_bars": args.window_bars,
        "context_limit": CONTEXT_LIMIT,
        "n_pieces_declined_no_row": n_declined,
        "vocab_size": {label: len(t) for label, t in tokenizers.items()},
        "codecs": {label: acc.to_dict() for label, acc in accumulators.items()},
    }
    args.out.write_text(json.dumps(payload, indent=2) + "\n")

    header = (
        f"{'codec':<6}{'vocab':>7}{'recall':>8}{'spur':>7}{'onsMAE':>9}{'onsP95':>9}"
        f"{'durMAE':>9}{'IOI%':>7}{'tokens':>8}{'>ctx':>7}"
    )
    print(f"\n{args.split} split, {args.window_bars}-bar windows\n")
    print(header)
    print("-" * len(header))
    for label, acc in accumulators.items():
        d = acc.to_dict()
        print(
            f"{label:<6}{payload['vocab_size'][label]:>7}"
            f"{d['event_recall']:>8.3f}{d['spurious_rate']:>7.3f}"
            f"{d['onset_mae_sec']['median'] * 1000:>8.1f}m"
            f"{d['onset_p95_sec']['median'] * 1000:>8.1f}m"
            f"{d['duration_mae_sec']['median'] * 1000:>8.1f}m"
            f"{d['ioi_entropy_retained']['median'] * 100:>6.0f}%"
            f"{d['tokens']['median']:>8}{d['tokens']['over_context_share']:>7.2f}"
        )
    print("\nevent recall by class")
    classes_header = f"{'codec':<6}" + "".join(f"{c:>11}" for c in CLASSES)
    print(classes_header)
    print("-" * len(classes_header))
    for label, acc in accumulators.items():
        by_class = acc.to_dict()["recall_by_class"]
        print(f"{label:<6}" + "".join(f"{by_class.get(c, 0.0):>11.3f}" for c in CLASSES))
    print(f"\nwrote {args.out}")
    if n_declined:
        print(f"note: {n_declined} pieces had no fitted row and were skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
