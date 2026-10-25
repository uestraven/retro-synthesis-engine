"""Phase 2: what survives a round trip through a pretrained model's grid?

Measures NES-MDB -> model-native score -> reconstructed events, deterministically
and with no GPU. The reference is the **D4-decoded composition**, not the raw
MIDI: D4 is what RSE actually holds and renders, so the loss measured here is
the external representation's alone.

Three resolutions are compared, and the third is a control rather than a
proposal:

- **24 TPQN** -- Fermata `tokenizer_v2`'s actual resolution.
- **8 TPQN** -- a common MidiTok default, for scale.
- **48 TPQN** -- a hypothetical that *is* a multiple of RSE's beat/16 grid, so
  it isolates how much of the loss is quantization rather than anything else.
  The released Fermata checkpoint does not support it.

Losses are kept apart on purpose. Quantization is what the tick grid costs;
expressiveness is what the representation cannot say at any resolution
(monophony, noise-as-timbre); arrangement is what collapsing back onto four
voices costs. Reporting a single number would make the result useless for
choosing between fine-tuning and a composer-plus-arranger architecture.

Event classes come from `rse.events.classify`, so recall is reported for real
attacks separately from arpeggio frames, vibrato, retriggers and ambiguous
fast passages. A codec that drops a quarter of its events is a very different
proposition depending on which quarter.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Importable when run as a file from a checkout where the package is not
# installed, which is the normal state of a fresh GPU host: `python
# scripts/x.py` puts `scripts/` on sys.path, not the repository root.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import json
from collections import Counter
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
from rse.tokenizers.frozen import frozen_config
from rse.tokenizers.tokenizer import GridFit, RseTokenizer
from rse.training.corpus import corpus_from_release
from rse.transfer.converters import (
    ROWS_PER_BEAT,
    composition_to_score,
    percussion_round_trip,
    score_to_composition,
    seconds_per_tick,
)

RESOLUTIONS = (8, 24, 48)


@dataclass
class Totals:
    n_windows: int = 0
    n_original: int = 0
    n_matched: int = 0
    n_spurious: int = 0
    kept_by_class: Counter[str] = field(default_factory=Counter)
    total_by_class: Counter[str] = field(default_factory=Counter)
    kept_by_channel: Counter[str] = field(default_factory=Counter)
    total_by_channel: Counter[str] = field(default_factory=Counter)
    onset: list[float] = field(default_factory=list)
    onset_p95: list[float] = field(default_factory=list)
    duration: list[float] = field(default_factory=list)
    ioi_retained: list[float] = field(default_factory=list)
    cv: list[float] = field(default_factory=list)
    density_in: list[int] = field(default_factory=list)
    density_out: list[int] = field(default_factory=list)
    collisions: int = 0
    percussion_lost: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_windows": self.n_windows,
            "n_original_events": self.n_original,
            "event_recall": round(self.n_matched / self.n_original, 5) if self.n_original else 0.0,
            "spurious_rate": round(self.n_spurious / self.n_original, 5)
            if self.n_original
            else 0.0,
            "recall_by_class": {
                name: round(self.kept_by_class[name] / self.total_by_class[name], 5)
                for name in CLASSES
                if self.total_by_class[name]
            },
            "recall_by_channel": {
                name: round(self.kept_by_channel[name] / self.total_by_channel[name], 5)
                for name in sorted(self.total_by_channel)
            },
            "n_by_channel": dict(self.total_by_channel),
            "onset_mae_ms": round(median(self.onset) * 1000, 3) if self.onset else 0.0,
            "onset_p95_ms": round(median(self.onset_p95) * 1000, 3) if self.onset_p95 else 0.0,
            "duration_mae_ms": round(median(self.duration) * 1000, 3) if self.duration else 0.0,
            "ioi_entropy_retained": round(median(self.ioi_retained), 4)
            if self.ioi_retained
            else 0.0,
            "note_length_cv": round(median(self.cv), 4) if self.cv else 0.0,
            "melodic_density_in": round(median(self.density_in), 1) if self.density_in else 0,
            "melodic_density_out": round(median(self.density_out), 1) if self.density_out else 0,
            "onset_collisions": self.collisions,
            "percussion_lost_in_conversion": self.percussion_lost,
        }


def _window(composition: Composition, start: int, end: int) -> Composition:
    events: list[Note | Noise] = [
        type(e)(**{**vars(e), "tick": e.tick - start})
        for e in composition.sounding_events()
        if start <= e.tick < end
    ]
    out = Composition(
        hardware_profile=composition.hardware_profile,
        ticks_per_beat=composition.ticks_per_beat,
        events=events,
        tempo_map=list(composition.tempo_map),
        time_signatures=list(composition.time_signatures),
    )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="nesmdb")
    parser.add_argument("--release", default="midi")
    parser.add_argument("--split", default="train", choices=("train", "valid"))
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--window-bars", type=int, default=8)
    parser.add_argument(
        "--drum-key-shift",
        type=int,
        default=0,
        help="model Fermata's documented +1 GM drum-key defect",
    )
    parser.add_argument("--out", type=Path, default=Path("transfer_audit.json"))
    args = parser.parse_args()

    tokenizer = RseTokenizer(frozen_config())
    totals = {tpqn: Totals() for tpqn in RESOLUTIONS}
    declined = 0

    print(f"corpus    : loading {args.split}...", flush=True)
    entries, info = corpus_from_release(
        args.dataset, args.release, limit=args.limit, splits=(args.split,)
    )
    print(f"corpus    : {len(entries)} pieces", flush=True)

    for index, (composition, _split, _pid) in enumerate(entries, 1):
        pulse = detect_pulse(composition)
        if not pulse.found or pulse.period_sec <= 0:
            declined += 1
            continue
        row_ticks, row_sec = pulse.period_ticks, pulse.period_sec
        window_ticks = int(round(row_ticks * 16 * args.window_bars))
        if window_ticks <= 0:
            continue
        fit = GridFit(step_ticks=row_ticks, phase_ticks=0.0, aligned=True, period_sec=row_sec)
        end = composition.end_tick

        for start in range(0, max(1, end - window_ticks), window_ticks):
            source = _window(composition, start, start + window_ticks)
            if len(source.sounding_events()) < 8:
                continue
            # The reference is what RSE actually holds: D4 encoded and decoded.
            encoded = tokenizer.encode(source, fit=fit)
            d4 = tokenizer.decode(encoded.ids).composition
            d4, _trunc = enforce_monophony(d4)
            reference = d4.sounding_events()
            if len(reference) < 8:
                continue
            ref_spt = seconds_per_tick(d4)
            classes = classify_events(d4, row_ticks=row_ticks, events=reference).labels
            tempo_bpm = 60.0 / (row_sec * ROWS_PER_BEAT)

            for tpqn in RESOLUTIONS:
                score, out_loss = composition_to_score(
                    d4, ticks_per_quarter=tpqn, tempo_bpm=tempo_bpm, row_sec=row_sec
                )
                rebuilt, back_loss = score_to_composition(
                    score, row_sec=row_sec, drum_key_shift=args.drum_key_shift
                )
                rebuilt, _t = enforce_monophony(rebuilt)
                got = rebuilt.sounding_events()
                got_spt = seconds_per_tick(rebuilt)

                result = match_events(
                    reference,
                    got,
                    tolerance=row_sec * MATCH_TOLERANCE_ROWS,
                    original_scale=ref_spt,
                    reconstructed_scale=got_spt,
                )
                t = totals[tpqn]
                t.n_windows += 1
                t.n_original += len(reference)
                t.n_matched += len(result.pairs)
                t.n_spurious += len(result.spurious_reconstructed)
                t.collisions += out_loss.n_onset_collisions
                t.percussion_lost += (
                    back_loss.n_percussion_shifted_lost + out_loss.n_percussion_unmapped
                )
                for i, name in enumerate(classes):
                    t.total_by_class[name] += 1
                    t.total_by_channel[reference[i].channel_id] += 1
                for oi, _ri in result.pairs:
                    t.kept_by_class[classes[oi]] += 1
                    t.kept_by_channel[reference[oi].channel_id] += 1

                errs, durs = [], []
                for oi, ri in result.pairs:
                    a = reference[oi].tick * ref_spt
                    b = got[ri].tick * got_spt
                    errs.append(abs(b - a))
                    durs.append(abs(reference[oi].duration * ref_spt - got[ri].duration * got_spt))
                if errs:
                    t.onset.append(fmean(errs))
                    t.onset_p95.append(percentile(errs, 0.95))
                if durs:
                    t.duration.append(fmean(durs))
                ref_ioi = ioi_entropy([e.tick * ref_spt for e in reference])
                got_ioi = ioi_entropy([e.tick * got_spt for e in got])
                t.ioi_retained.append(got_ioi / ref_ioi if ref_ioi > 0 else 0.0)
                t.cv.append(coefficient_of_variation([e.duration * got_spt for e in got]))
                t.density_in.append(sum(1 for e in reference if isinstance(e, Note)))
                t.density_out.append(sum(1 for e in got if isinstance(e, Note)))

        if index % 50 == 0:
            print(f"          : {index}/{len(entries)}", flush=True)

    payload = {
        "corpus": info.to_dict(),
        "split": args.split,
        "window_bars": args.window_bars,
        "drum_key_shift": args.drum_key_shift,
        "n_declined_no_row": declined,
        "rse_onset_unit": f"beat/{ROWS_PER_BEAT * frozen_config().sub_row_divisions}",
        "percussion_round_trip": {
            "no_shift": percussion_round_trip(0),
            "fermata_plus_one": percussion_round_trip(1),
        },
        "resolutions": {str(k): v.to_dict() for k, v in totals.items()},
    }
    args.out.write_text(json.dumps(payload, indent=2) + "\n")

    hdr = (
        f"{'TPQN':>5}{'exact':>7}{'recall':>8}{'spur':>7}{'onsMAE':>9}{'onsP95':>9}"
        f"{'durMAE':>9}{'IOI':>6}{'collide':>9}{'percLost':>10}"
    )
    unit = ROWS_PER_BEAT * frozen_config().sub_row_divisions
    print(f"\n{args.split}: RSE onsets are on beat/{unit}; drum shift {args.drum_key_shift}\n")
    print(hdr)
    print("-" * len(hdr))
    for tpqn in RESOLUTIONS:
        d = totals[tpqn].to_dict()
        print(
            f"{tpqn:>5}{str(tpqn % unit == 0):>7}{d['event_recall']:>8.3f}"
            f"{d['spurious_rate']:>7.3f}{d['onset_mae_ms']:>8.2f}m{d['onset_p95_ms']:>8.2f}m"
            f"{d['duration_mae_ms']:>8.2f}m{d['ioi_entropy_retained'] * 100:>5.0f}%"
            f"{d['onset_collisions']:>9}{d['percussion_lost_in_conversion']:>10}"
        )
    print("\nrecall by event class")
    h2 = f"{'TPQN':>5}" + "".join(f"{c:>11}" for c in CLASSES)
    print(h2)
    print("-" * len(h2))
    for tpqn in RESOLUTIONS:
        by = totals[tpqn].to_dict()["recall_by_class"]
        print(f"{tpqn:>5}" + "".join(f"{by.get(c, 0.0):>11.3f}" for c in CLASSES))
    print("\nrecall by channel")
    channels = sorted(totals[RESOLUTIONS[0]].total_by_channel)
    h3 = f"{'TPQN':>5}" + "".join(f"{c:>9}" for c in channels)
    print(h3)
    print("-" * len(h3))
    for tpqn in RESOLUTIONS:
        by = totals[tpqn].to_dict()["recall_by_channel"]
        print(f"{tpqn:>5}" + "".join(f"{by.get(c, 0.0):>9.3f}" for c in channels))
    print(f"\nwrote {args.out}")
    if declined:
        print(f"note: {declined} pieces had no fitted row")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
