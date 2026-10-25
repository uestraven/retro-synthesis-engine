"""How much of the absolute onset error is tempo quantization?

Branch B measured grid onset error falling from 15.60 ms to 2.91 ms while
absolute onset error stayed flat near 36.5 ms for every codec. The candidate
explanation was the `Tempo_` token: its geometric buckets mislabel a fitted row
rate, and a rate error accumulates across a window. That is a hypothesis, not a
result -- a 0.34% median rate error over 128 rows accounts for roughly 19 ms,
which is half of 36.5 ms, so something else may be involved.

This isolates it with a 2x2 plus an oracle, holding source pieces, window
boundaries, event matching and monophony repair constant:

    A/1.03   historical baseline
    A/1.01   tempo improvement alone, no codec change
    D4/1.03  codec improvement alone, historical tempo
    D4/1.01  both
    D4/exact oracle: the true fitted row rate, unquantized

The oracle is a diagnostic ceiling, not a deployable configuration: a decoder
has only the Tempo token to rebuild time from, so an unquantized rate is not
something a trained model could emit. Its purpose is to say how much error
survives after tempo is removed as a source -- and therefore whether a third
mechanism is at work.

Drift is reported against elapsed time within the window. A rate error produces
error growing linearly with elapsed time; a constant offset or a quantization
error does not. The slope is the discriminating evidence, so it is reported in
ms per second alongside the bin means rather than inferred from the totals.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean, median
from typing import Any

from rse.codec.benchmark import (
    MATCH_TOLERANCE_ROWS,
    coefficient_of_variation,
    enforce_monophony,
    ioi_entropy,
    match_events,
    percentile,
)
from rse.events.classify import CLASSES, classify_events
from rse.events.pulse import detect_pulse
from rse.events.schema import Composition, Noise, Note
from rse.tokenizers.tokenizer import DEFAULT_TICKS_PER_BEAT, GridFit, RseTokenizer
from rse.tokenizers.vocab import TokenizerConfig
from rse.training.corpus import corpus_from_release


def _fit_line(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Least-squares intercept and slope, so offset and drift separate."""
    n = len(xs)
    if n < 2:
        return (ys[0] if ys else 0.0), 0.0
    mean_x, mean_y = fmean(xs), fmean(ys)
    den = sum((x - mean_x) ** 2 for x in xs)
    if den == 0:
        return mean_y, 0.0
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True)) / den
    return mean_y - slope * mean_x, slope


#: Elapsed-time bins for the drift profile, in seconds.
DRIFT_BINS = (0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 9.0, 1e9)


@dataclass
class Config:
    label: str
    codec: str
    k: int
    tempo_ratio: float
    #: Use the exact fitted row rate instead of the Tempo token's value.
    oracle: bool = False

    def tokenizer(self) -> RseTokenizer:
        if self.codec == "A":
            return RseTokenizer(TokenizerConfig(tempo_ratio=self.tempo_ratio))
        return RseTokenizer(
            TokenizerConfig(
                codec=self.codec,
                sub_row_divisions=self.k,
                tempo_ratio=self.tempo_ratio,
            )
        )


@dataclass
class Totals:
    n_windows: int = 0
    n_original: int = 0
    n_matched: int = 0
    n_spurious: int = 0
    kept_by_class: dict[str, int] = field(default_factory=dict)
    total_by_class: dict[str, int] = field(default_factory=dict)
    grid_onset: list[float] = field(default_factory=list)
    abs_onset: list[float] = field(default_factory=list)
    abs_onset_p95: list[float] = field(default_factory=list)
    duration: list[float] = field(default_factory=list)
    ioi_retained: list[float] = field(default_factory=list)
    cv_reconstructed: list[float] = field(default_factory=list)
    tokens: list[int] = field(default_factory=list)
    tempo_rel_error: list[float] = field(default_factory=list)
    #: (elapsed seconds, signed error seconds) for the drift profile.
    drift: list[tuple[float, float]] = field(default_factory=list)
    #: Fitted per window, never pooled. Pooling mixes pieces at different
    #: tempos and window start times, so one global line through all of them
    #: describes nothing -- it produced a "jitter" larger than the total error
    #: it was supposed to be a component of.
    offsets: list[float] = field(default_factory=list)
    slopes: list[float] = field(default_factory=list)
    jitters: list[float] = field(default_factory=list)

    def drift_profile(self) -> dict[str, Any]:
        bins: list[dict[str, Any]] = []
        for low, high in zip(DRIFT_BINS, DRIFT_BINS[1:], strict=False):
            inside = [e for t, e in self.drift if low <= t < high]
            if inside:
                bins.append(
                    {
                        "from_sec": low,
                        "to_sec": None if high > 1e8 else high,
                        "n": len(inside),
                        "mean_signed_ms": round(fmean(inside) * 1000, 3),
                        "mean_abs_ms": round(fmean(abs(e) for e in inside) * 1000, 3),
                    }
                )
        # Least-squares slope through the origin: a pure rate error gives a
        # straight line from zero, so a slope fitted with an intercept would
        # absorb part of the very thing being measured.
        num = sum(t * e for t, e in self.drift)
        den = sum(t * t for t, e in self.drift)
        slope = (num / den) if den else 0.0

        # Absolute onset error decomposes into three things that matter very
        # differently to a listener, so they are reported apart:
        #
        #   offset  every onset shifted equally. The encoder anchors the first
        #           sounding row at bar 0, so a piece starts a few ms early.
        #           Inaudible: nothing about the music changes.
        #   drift   error growing with elapsed time, from a mislabelled global
        #           row rate. Perceptually a playback-speed change of a
        #           fraction of a percent, not a rhythmic defect.
        #   jitter  what remains once both are removed: note-to-note timing
        #           error. This is the part a listener hears as sloppy timing,
        #           and the part a finer grid actually fixes.
        #
        # Reporting only the total would credit a tempo fix with repairing
        # rhythm, which it does not do.
        return {
            "bins": bins,
            "pooled_slope_ms_per_sec": round(slope * 1000, 4),
            "offset_ms": round(median(self.offsets) * 1000, 3) if self.offsets else 0.0,
            "slope_ms_per_sec": round(median(self.slopes) * 1000, 4) if self.slopes else 0.0,
            "jitter_mae_ms": round(median(self.jitters) * 1000, 3) if self.jitters else 0.0,
            "jitter_p95_ms": round(percentile(self.jitters, 0.95) * 1000, 3)
            if self.jitters
            else 0.0,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_windows": self.n_windows,
            "n_original_events": self.n_original,
            "event_recall": round(self.n_matched / self.n_original, 5) if self.n_original else 0.0,
            "spurious_rate": round(self.n_spurious / self.n_original, 5)
            if self.n_original
            else 0.0,
            "recall_by_class": {
                name: round(self.kept_by_class.get(name, 0) / self.total_by_class[name], 5)
                for name in CLASSES
                if self.total_by_class.get(name)
            },
            "grid_onset_mae_ms": round(median(self.grid_onset) * 1000, 3)
            if self.grid_onset
            else 0.0,
            "abs_onset_mae_ms": round(median(self.abs_onset) * 1000, 3) if self.abs_onset else 0.0,
            "abs_onset_p95_ms": round(median(self.abs_onset_p95) * 1000, 3)
            if self.abs_onset_p95
            else 0.0,
            "duration_mae_ms": round(median(self.duration) * 1000, 3) if self.duration else 0.0,
            "ioi_entropy_retained": round(median(self.ioi_retained), 4)
            if self.ioi_retained
            else 0.0,
            "note_length_cv": round(median(self.cv_reconstructed), 4)
            if self.cv_reconstructed
            else 0.0,
            "tempo_rel_error_median_pct": round(median(self.tempo_rel_error) * 100, 4)
            if self.tempo_rel_error
            else 0.0,
            "tokens_median": round(median(self.tokens)) if self.tokens else 0,
            "tokens_p90": round(percentile([float(t) for t in self.tokens], 0.90))
            if self.tokens
            else 0,
            "drift": self.drift_profile(),
        }


def _window(composition: Composition, start: int, end: int) -> Composition:
    events: list[Note | Noise] = [
        type(e)(**{**vars(e), "tick": e.tick - start})
        for e in composition.sounding_events()
        if start <= e.tick < end
    ]
    return Composition(
        hardware_profile=composition.hardware_profile,
        ticks_per_beat=composition.ticks_per_beat,
        events=events,
        tempo_map=list(composition.tempo_map),
        time_signatures=list(composition.time_signatures),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="nesmdb")
    parser.add_argument("--release", default="midi")
    parser.add_argument("--split", default="train", choices=("train", "valid"))
    parser.add_argument("--limit", type=int, default=400)
    parser.add_argument("--window-bars", type=int, default=8)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path("tempo_precision.json"))
    args = parser.parse_args()

    configs = [
        Config("A/1.03", "A", 1, 1.03),
        Config("A/1.01", "A", 1, 1.01),
        Config(f"D{args.k}/1.03", "D", args.k, 1.03),
        Config(f"D{args.k}/1.01", "D", args.k, 1.01),
        Config("A/exact", "A", 1, 1.01, oracle=True),
        Config(f"D{args.k}/exact", "D", args.k, 1.01, oracle=True),
    ]
    tokenizers = {c.label: c.tokenizer() for c in configs}
    totals = {c.label: Totals() for c in configs}

    print(f"corpus    : loading {args.split}...", flush=True)
    entries, info = corpus_from_release(
        args.dataset, args.release, limit=args.limit, splits=(args.split,)
    )
    print(f"corpus    : {len(entries)} pieces", flush=True)

    # One nominal row in decode's tick space, which is what a reconstructed
    # onset is counted in. Taken from a tokenizer rather than hardcoded so it
    # cannot fall out of step with `ticks_per_step`.
    nominal_row = next(iter(tokenizers.values())).ticks_per_step(DEFAULT_TICKS_PER_BEAT)
    declined = 0

    for index, (composition, _split, _pid) in enumerate(entries, 1):
        pulse = detect_pulse(composition)
        if not pulse.found or pulse.period_ticks <= 0 or pulse.period_sec <= 0:
            declined += 1
            continue
        row_ticks = pulse.period_ticks
        window_ticks = int(round(row_ticks * 16 * args.window_bars))
        if window_ticks <= 0:
            continue
        end = composition.end_tick
        original_spt = (
            composition.time.seconds(end) / end if end and composition.time.seconds(end) else 0.0
        )
        if original_spt <= 0:
            continue

        for start in range(0, max(1, end - window_ticks), window_ticks):
            source = _window(composition, start, start + window_ticks)
            events = source.sounding_events()
            if len(events) < 8:
                continue
            classes = classify_events(source, row_ticks=row_ticks, events=events).labels
            fit = GridFit(
                step_ticks=row_ticks,
                phase_ticks=0.0,
                aligned=True,
                period_sec=pulse.period_sec,
            )

            for config in configs:
                tokenizer = tokenizers[config.label]
                encoded = tokenizer.encode(source, fit=fit)
                decoded = tokenizer.decode(encoded.ids)
                repaired, _trunc = enforce_monophony(decoded.composition)
                reconstructed = repaired.sounding_events()

                labelled_bpm = tokenizer.tempo_token_value(fit.bpm)
                if config.oracle:
                    recon_row_sec = pulse.period_sec
                else:
                    # 4 rows to a beat, so a row is 15/BPM seconds.
                    recon_row_sec = 15.0 / labelled_bpm
                recon_spt = recon_row_sec / nominal_row

                result = match_events(
                    events,
                    reconstructed,
                    tolerance=pulse.period_sec * MATCH_TOLERANCE_ROWS,
                    original_scale=original_spt,
                    reconstructed_scale=recon_spt,
                )
                t = totals[config.label]
                t.n_windows += 1
                t.n_original += len(events)
                t.n_matched += len(result.pairs)
                t.n_spurious += len(result.spurious_reconstructed)
                for name in classes:
                    t.total_by_class[name] = t.total_by_class.get(name, 0) + 1
                for oi, _ri in result.pairs:
                    name = classes[oi]
                    t.kept_by_class[name] = t.kept_by_class.get(name, 0) + 1

                signed = []
                durations = []
                for oi, ri in result.pairs:
                    at = events[oi].tick * original_spt
                    delta = reconstructed[ri].tick * recon_spt - at
                    signed.append((at, delta))
                    durations.append(
                        abs(
                            events[oi].duration * original_spt
                            - reconstructed[ri].duration * recon_spt
                        )
                    )
                if signed:
                    errs = [abs(d) for _a, d in signed]
                    t.abs_onset.append(fmean(errs))
                    t.abs_onset_p95.append(percentile(errs, 0.95))
                    t.drift.extend(signed)
                    # One fit per window: offset, rate error, and what is left.
                    inter, slope_w = _fit_line([a for a, _d in signed], [d for _a, d in signed])
                    t.offsets.append(inter)
                    t.slopes.append(slope_w)
                    t.jitters.append(fmean(abs(d - (inter + slope_w * a)) for a, d in signed))
                if durations:
                    t.duration.append(fmean(durations))
                if encoded.n_encoded:
                    t.grid_onset.append(
                        encoded.total_start_error_ticks / encoded.n_encoded * original_spt
                    )
                original_ioi = ioi_entropy([e.tick * original_spt for e in events])
                recon_ioi = ioi_entropy([e.tick * recon_spt for e in reconstructed])
                t.ioi_retained.append(recon_ioi / original_ioi if original_ioi > 0 else 0.0)
                t.cv_reconstructed.append(
                    coefficient_of_variation([e.duration * recon_spt for e in reconstructed])
                )
                t.tokens.append(len(encoded.ids))
                t.tempo_rel_error.append(abs(labelled_bpm / fit.bpm - 1.0) if fit.bpm else 0.0)

        if index % 50 == 0:
            print(f"          : {index}/{len(entries)}", flush=True)

    payload = {
        "corpus": info.to_dict(),
        "split": args.split,
        "window_bars": args.window_bars,
        "k": args.k,
        "n_declined_no_row": declined,
        "vocab_size": {label: len(t) for label, t in tokenizers.items()},
        "configs": {label: t.to_dict() for label, t in totals.items()},
    }
    args.out.write_text(json.dumps(payload, indent=2) + "\n")

    hdr = (
        f"{'config':<12}{'vocab':>7}{'recall':>8}{'gridMAE':>10}{'absMAE':>9}"
        f"{'absP95':>9}{'durMAE':>9}{'IOI%':>7}{'tempoErr':>10}{'drift':>12}"
    )
    print(f"\n{args.split}: {args.window_bars}-bar windows, k={args.k}\n")
    print(hdr)
    print("-" * len(hdr))
    for label, t in totals.items():
        d = t.to_dict()
        print(
            f"{label:<12}{payload['vocab_size'][label]:>7}{d['event_recall']:>8.3f}"
            f"{d['grid_onset_mae_ms']:>9.2f}m{d['abs_onset_mae_ms']:>8.2f}m"
            f"{d['abs_onset_p95_ms']:>8.2f}m{d['duration_mae_ms']:>8.2f}m"
            f"{d['drift']['jitter_mae_ms']:>8.2f}m{d['drift']['offset_ms']:>8.2f}m"
            f"{d['drift']['slope_ms_per_sec']:>8.2f}ms/s"
            f"{d['ioi_entropy_retained'] * 100:>6.0f}%"
        )
    print("\nsigned onset error against elapsed time (ms)")
    labels = list(totals)
    print(f"{'elapsed':<12}" + "".join(f"{label:>14}" for label in labels))
    rows = {
        label: {b["from_sec"]: b for b in totals[label].drift_profile()["bins"]} for label in labels
    }
    for low in DRIFT_BINS[:-1]:
        if not any(low in rows[label] for label in labels):
            continue
        cells = []
        for label in labels:
            bin_row = rows[label].get(low)
            cells.append(f"{bin_row['mean_signed_ms']:>14.2f}" if bin_row else f"{'-':>14}")
        print(f"{low:>5.0f}s+      " + "".join(cells))
    print(f"\nwrote {args.out}")
    if declined:
        print(f"note: {declined} pieces had no fitted row")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
