#!/usr/bin/env python3
"""The matched hardware-consistency aggregation table.

Every cell here is computed by ONE set of definitions, applied identically to
the real-music controls and to both generated arms. This exists because the
same quantity was previously reported under two aggregations -- a pooled
event-weighted rate and a per-excerpt median -- whose ratio to a control
differs about twofold. Mixing a pooled numerator with a median denominator
produces a wrong multiplier, so both aggregations are computed for every
population and are never crossed.

Definitions, fixed here and used everywhere:

- **Eligible same-voice pair**: consecutive events on one voice, ordered by
  onset, where the earlier event has a non-null duration. This is the
  denominator of the overlap rate.
- **Overlapping pair**: an eligible pair whose earlier event's nominal end
  exceeds the next onset by more than the documented 1 microsecond rounding
  tolerance. Sub-tolerance crossings are the release's own independent
  rounding of onset and duration, not musical overlap.
- **Overlap rate, per excerpt**: overlapping / eligible within one excerpt.
  The reported figure is the MEDIAN over excerpts. Primary statistic.
- **Overlap rate, pooled**: total overlapping / total eligible across the
  population. Event-weighted, so excerpts contribute in proportion to size.
- **Voice-stealing rate**: events whose duration is clamped by the
  eight-voice policy, over all events. Reported as per-excerpt median
  (primary) and pooled.
- **Peak raw note instances**: maximum concurrent note intervals ignoring
  voice. MAY exceed eight; that is the diagnostic.
- **Peak occupancy after stealing**: maximum concurrent intervals once each
  note is clamped at the next key-on on its voice, in exact Decimal with
  half-open intervals. Bounded by eight.
- **Peak occupied hardware voices**: distinct voices sounding at once.
  Bounded by eight by construction. This is NOT the same quantity as the
  previous two and is reported separately so they cannot be conflated.

The two runaway excerpts (seeds 6 and 19) are RETAINED in the full
population. A sensitivity analysis excluding them is reported alongside and
clearly labelled; it does not replace the registered population.

The cross-game rendering control (one track per game) and the earlier
Actraiser-only diagnostic set are kept as separate populations. They are not
interchangeable: the Actraiser-only set contains actr-05, whose published
events require 1.7% voice stealing, whereas the cross-game control's
published events require 0.0%.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rse.snes_lm import hardware_checks as hc  # noqa: E402
from rse.snes_lm import representation as rep  # noqa: E402
from rse.snes_lm.release import Release  # noqa: E402

N_CONTROL_TRACKS = 60
CONTROL_SHUFFLE_SEED = 0


def measure_population(name: str, excerpts: list[list[dict]], note: str = "") -> dict:
    """All cells for one population, under the definitions in the docstring."""
    rows = []
    pooled_overlapping = pooled_eligible = 0
    pooled_clamped = pooled_events = 0
    for events in excerpts:
        if not events:
            continue
        report = hc.same_voice_overlap(events)
        _, n_clamped = hc.apply_voice_stealing(events)
        occ = hc.occupancy(events)
        pooled_overlapping += report.n_overlapping
        pooled_eligible += report.n_pairs
        pooled_clamped += n_clamped
        pooled_events += len(events)
        rows.append({
            "n_events": len(events),
            "eligible_pairs": report.n_pairs,
            "overlapping_pairs": report.n_overlapping,
            "overlap_rate": report.fraction,
            "stealing_rate": n_clamped / len(events) if events else 0.0,
            "peak_raw_notes": occ.concurrent_notes,
            "peak_after_stealing": occ.concurrent_after_stealing,
            "peak_occupied_voices": occ.occupied_voices,
        })
    if not rows:
        return {"population": name, "n_excerpts": 0, "note": note}

    def med(key):
        return st.median(r[key] for r in rows)

    return {
        "population": name,
        "note": note,
        "n_excerpts": len(rows),
        "n_events_total": pooled_events,
        "n_eligible_pairs_total": pooled_eligible,
        "overlap_rate_median_per_excerpt": med("overlap_rate"),
        "overlap_rate_pooled_event_weighted": (
            pooled_overlapping / pooled_eligible if pooled_eligible else 0.0),
        "n_overlapping_pairs_total": pooled_overlapping,
        "stealing_rate_median_per_excerpt": med("stealing_rate"),
        "stealing_rate_pooled_event_weighted": (
            pooled_clamped / pooled_events if pooled_events else 0.0),
        "n_clamped_total": pooled_clamped,
        "peak_raw_notes_max": max(r["peak_raw_notes"] for r in rows),
        "peak_raw_notes_median": med("peak_raw_notes"),
        "peak_after_stealing_max": max(r["peak_after_stealing"] for r in rows),
        "peak_occupied_voices_max": max(r["peak_occupied_voices"] for r in rows),
        "per_excerpt": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab", default="/tmp/snes_vocab.json")
    parser.add_argument("--release", default="~/snesmdb-releases/snesmdb-50game-rc1")
    parser.add_argument("--model-tokens",
                        default=str(Path.home() /
                                    "workplace/level2_artifacts/model_tokens_corrected.json"))
    parser.add_argument("--baseline-tokens",
                        default=str(Path.home() /
                                    "workplace/level2_artifacts/baseline_tokens.json"))
    parser.add_argument("--out", default=str(Path.home() /
                        "workplace/level2_artifacts/aggregation_table.json"))
    args = parser.parse_args()

    vocab = rep.read_vocab(args.vocab)
    release = Release(args.release)
    populations: list[dict] = []

    # --- real-music controls -------------------------------------------------
    tracks = release.tracks(split="train")
    random.Random(CONTROL_SHUFFLE_SEED).shuffle(tracks)
    control = tracks[:N_CONTROL_TRACKS]
    # A PUBLISHED control must be restricted to key_on events to be matched.
    #
    # The published tables also carry `legato_step` events -- pitch changes
    # within an already sounding voice -- and `rep.attacks()` drops them, so no
    # generator can emit one. Including them makes a published control that is
    # not comparable to anything the generators produce: a legato step shares
    # its voice with its parent attack and so registers as same-voice overlap
    # by construction. Over the 60-track control that inflates pooled overlap
    # from 0.256% to 4.166%, with 82% of it coming from 3 of 60 tracks.
    #
    # Restricting to key_on also makes the denominators agree exactly: 131,713
    # eligible pairs for both the published and round-tripped control.
    def key_on(events):
        return [e for e in events if e.get("origin") == "key_on"]

    populations.append(measure_population(
        "real, as published, KEY-ON ONLY (matched published control)",
        [key_on(release.events(t)) for t in control],
        f"{N_CONTROL_TRACKS} train music tracks, shuffle seed "
        f"{CONTROL_SHUFFLE_SEED}. The matched upper bound of symbolic "
        f"fidelity: legato_step events excluded because the representation "
        f"drops them and no generator can emit one."))
    populations.append(measure_population(
        "real, as published, all events incl. legato (UNMATCHED)",
        [release.events(t) for t in control],
        "Reported for transparency only. NOT comparable to the generated "
        "arms: legato_step events overlap their parent attack by "
        "construction. Do not use this row for any multiplier."))
    populations.append(measure_population(
        "real, tokenized round-trip (THE registered control)",
        [rep.decode(rep.encode(release.events(t), vocab).tokens, vocab)
         for t in control],
        "Same tracks through encode then decode. This is the fair comparison "
        "for the generators, which can only emit round-tripped material. "
        "Supersedes the previously quoted 6.638% pooled figure, whose "
        "population could not be verified."))

    # --- the two rendering controls, kept DISTINCT ---------------------------
    #
    # These are 25-SECOND WINDOWS, the same unit as a generated excerpt, and so
    # are the directly matched comparison. The 60-track control above is whole
    # tracks and is a different unit; both are reported, labelled.
    #
    # The cross-game set and the Actraiser-only set are NOT interchangeable.
    # The Actraiser-only set contains actr-05, whose published events need 1.7%
    # voice stealing; the cross-game set's published events need 0.0%. The
    # 0.0% figure belongs to the cross-game control and must be cited as such.
    def pick(one_per_game: bool, limit: int = 5):
        chosen, seen = [], set()
        for track in release.tracks(split="test"):
            events = release.events(track)
            if not events:
                continue
            window = [e for e in events
                      if float(e["onset_sec"]) < 25.0]
            if len(window) < 40:
                continue
            pitched = sum(1 for e in window if e.get("midi_note") is not None)
            if pitched / len(window) < 0.5:
                continue
            name = (getattr(track, "name", None)
                    or getattr(track, "track_id", None) or "")
            game = str(name).split("-")[0]
            if one_per_game:
                if game in seen:
                    continue
                seen.add(game)
            chosen.append((track, window))
            if len(chosen) >= limit:
                break
        return chosen

    for one_per_game, label in ((True, "cross-game"), (False, "Actraiser-only")):
        chosen = pick(one_per_game)
        names = [str(getattr(t, "name", None) or getattr(t, "track_id", ""))
                 for t, _ in chosen]
        populations.append(measure_population(
            f"rendering control, as published ({label}, 25 s, key-on only)",
            [[e for e in w if e.get("origin") == "key_on"] for _, w in chosen],
            f"Tracks: {', '.join(names)}."
            + (" The 0.0% published-event stealing figure belongs HERE."
               if one_per_game else
               " Contains actr-05 at 1.7%; NOT the source of the 0.0% figure.")))
        populations.append(measure_population(
            f"rendering control, round-tripped ({label}, 25 s)",
            [[e for e in rep.decode(
                rep.encode(release.events(t), vocab).tokens, vocab)
              if float(e["onset_sec"]) < 25.0] for t, _ in chosen],
            f"Same {label} tracks through encode then decode."))

    # --- generated arms ------------------------------------------------------
    baseline_rows = json.loads(Path(args.baseline_tokens).read_text())
    populations.append(measure_population(
        "SlotNGram (order 10, T=1.2, top_p=0.95)",
        [rep.decode(r["tokens"], vocab) for r in baseline_rows if r["valid"]],
        "All 24 excerpts structurally valid."))

    model_rows = [r for r in json.loads(Path(args.model_tokens).read_text())
                  if r["valid"]]
    populations.append(measure_population(
        "Transformer (T=1.0, top_p=0.95) -- FULL registered population",
        [rep.decode(r["tokens"], vocab) for r in model_rows],
        "23 of 24 structurally valid. RETAINS the two runaway excerpts "
        "(seeds 6 and 19). This is the registered population."))
    populations.append(measure_population(
        "Transformer -- SENSITIVITY, runaways excluded",
        [rep.decode(r["tokens"], vocab) for r in model_rows
         if r["seed"] not in (6, 19)],
        "Seeds 6 and 19 removed. SENSITIVITY ANALYSIS ONLY; does not replace "
        "the registered population above."))

    payload = {
        "schema": "snes_lm.aggregation_table.v1",
        "definitions": __doc__,
        "rounding_tolerance_sec": hc.ROUNDING_TOLERANCE_SEC,
        "n_hardware_voices": hc.N_HARDWARE_VOICES,
        "warning": "Never divide a pooled numerator by a per-excerpt-median "
                   "denominator. Compare median to median, pooled to pooled.",
        "populations": populations,
    }
    Path(args.out).write_text(json.dumps(payload, indent=1))

    hdr = (f"  {'population':54} {'n':>3} {'events':>8} {'pairs':>8} "
           f"{'ovl med':>8} {'ovl pool':>9} {'steal med':>10} {'steal pool':>11} "
           f"{'raw':>5} {'stl':>4} {'vox':>4}")
    print(hdr)
    for p in populations:
        if not p.get("n_excerpts"):
            continue
        print(f"  {p['population'][:54]:54} {p['n_excerpts']:3} "
              f"{p['n_events_total']:8,} {p['n_eligible_pairs_total']:8,} "
              f"{100*p['overlap_rate_median_per_excerpt']:7.2f}% "
              f"{100*p['overlap_rate_pooled_event_weighted']:8.2f}% "
              f"{100*p['stealing_rate_median_per_excerpt']:9.2f}% "
              f"{100*p['stealing_rate_pooled_event_weighted']:10.2f}% "
              f"{p['peak_raw_notes_max']:5} {p['peak_after_stealing_max']:4} "
              f"{p['peak_occupied_voices_max']:4}")
    print(f"\n  written to {args.out}")


if __name__ == "__main__":
    main()
