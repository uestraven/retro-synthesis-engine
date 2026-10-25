"""Fixed-seed generation battery, with a rhythm and variance diagnosis.

Thirty-six samples per checkpoint by default -- twelve seeds at each of three
temperatures -- because six samples cannot separate a capability change from
sampling noise. At 100k, identical-temperature seeds differed by 4x in event
count and by three channels, so the per-seed outcomes are reported and not
only the medians.

Why the token distributions are here
------------------------------------
The native model's inter-onset-interval entropy is 0.179 against a like-for-like
corpus reference of 1.479 (D4 round-trip, 8 bars; the often-quoted 2.358 is
full songs of original MIDI and is not comparable). The corpus is not the
constraint: measured over 2,500 train windows its `Position_` distribution is
nearly uniform, entropy 5.62 bits of a possible 6.00, and 42% of onsets sit
off-row. Its `Duration_` distribution, by contrast, is mode-dominated --
`Duration_1` alone is 38.6% of all duration tokens, entropy 3.02 of 6.97.

So the question is which of those two shapes the model reproduces. If it
matches the corpus on Position and collapses on Duration, the rhythm problem
is note length. If it collapses on Position too, it is onset placement, and
the near-uniform corpus distribution says that is a learning failure rather
than a property of the data. This script measures both against the same cache
the model trained on.

Termination is classified rather than summarized: early EOS, token-budget
exhaustion and grammar stalls are different failures with different fixes, and
"did not reach eight bars" conflates them.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Importable when run as a file from a checkout where the package is not
# installed, which is the normal state of a fresh GPU host.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import json
import math
import statistics as st
from collections import Counter
from typing import Any

from rse.codec.benchmark import enforce_monophony, ioi_entropy
from rse.conditioning import melody_line, motif_note_participation
from rse.hardware.constraints import ConstraintEngine
from rse.hardware.profile import load_profile
from rse.training.evaluate import bar_signatures, evaluate_composition, repetition_metrics

#: Token families whose generated distribution is compared with the corpus.
FAMILIES = ("Position_", "Duration_", "Channel_", "Pitch_", "Velocity_")


def entropy(counter: Counter) -> float:
    total = sum(counter.values())
    if not total:
        return 0.0
    return -sum((n / total) * math.log2(n / total) for n in counter.values() if n)


def kl_divergence(p: Counter, q: Counter) -> float:
    """KL(generated || corpus), in bits, with the corpus smoothed.

    Directional on purpose: it answers "how surprising is the generated
    distribution under the corpus", which is the question when the corpus is
    the target.
    """
    keys = set(p) | set(q)
    total_p, total_q = sum(p.values()), sum(q.values())
    if not total_p or not total_q:
        return 0.0
    out = 0.0
    for key in keys:
        pi = p.get(key, 0) / total_p
        qi = (q.get(key, 0) + 1) / (total_q + len(keys))
        if pi > 0:
            out += pi * math.log2(pi / qi)
    return out


def corpus_distributions(cache: Path, tokenizer, *, split: str, limit: int) -> dict[str, Counter]:
    out: dict[str, Counter] = {family: Counter() for family in FAMILIES}
    seen = 0
    with cache.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row["split"] != split:
                continue
            seen += 1
            if seen > limit:
                break
            for token_id in row["ids"]:
                token = tokenizer.vocabulary.token_of(token_id)
                for family in FAMILIES:
                    if token.startswith(family):
                        out[family][token[len(family) :]] += 1
                        break
    return out


def bar_occupancy(composition, *, ticks_per_bar: int, n_bars: int) -> list[int]:
    """Events in each requested bar, by index."""
    counts = [0] * max(1, n_bars)
    for event in composition.sounding_events():
        index = event.tick // ticks_per_bar if ticks_per_bar else 0
        if 0 <= index < len(counts):
            counts[index] += 1
    return counts


def classify_termination(
    result,
    *,
    max_tokens: int,
    bars_requested: int,
    bar_tokens: int,
    occupancy: list[int],
    ends_with_eos: bool,
) -> str:
    """Why the piece is the length it is.

    The previous version called anything short of the requested bars
    `early_eos_short`, which was wrong and misled a whole round of analysis.
    The sampler **masks `<eos>` out of the legal set until the requested bar
    count is emitted**, so the model cannot stop early. A short piece is one
    whose later bars were emitted but left *empty* -- a different failure with
    a different fix, and one EOS handling cannot touch.
    """
    if len(result.ids) >= max_tokens:
        return "token_budget_exhausted"
    trailing_empty = 0
    for count in reversed(occupancy):
        if count:
            break
        trailing_empty += 1
    if bar_tokens < bars_requested:
        return "fewer_bar_tokens_than_requested"
    if trailing_empty:
        return "empty_trailing_bars"
    if any(count == 0 for count in occupancy):
        return "empty_interior_bars"
    return "complete_all_bars_occupied" if ends_with_eos else "complete_no_eos"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path("caches/d4_full.jsonl"))
    parser.add_argument("--corpus-split", default="valid")
    parser.add_argument("--corpus-limit", type=int, default=2000)
    parser.add_argument("--temperatures", type=float, nargs="+", default=[0.7, 0.85, 1.0])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(12)))
    parser.add_argument("--bars", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=3072)
    parser.add_argument(
        "--structural-temperatures",
        nargs="+",
        default=["none"],
        help="sweep the structural decision temperature. `none` omits the "
        "option entirely, which is the true baseline arm; `0` is greedy",
    )
    parser.add_argument(
        "--isolate-eos",
        action="store_true",
        help="hold P(stop) at its baseline while heating the rhythmic "
        "categories; without it a hot structural temperature also raises the "
        "chance of ending early",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--render-dir", type=Path, default=Path("battery_audio"))
    parser.add_argument("--render", type=int, default=6)
    parser.add_argument("--out", type=Path, default=Path("generation_battery.json"))
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from codec_comparison import render_window

    from rse.generation.sampler import SamplingConfig, sample
    from rse.tokenizers.tokenizer import Conditioning
    from rse.training.train import load_checkpoint

    model, tokenizer, payload = load_checkpoint(args.checkpoint, device=args.device)
    profile = load_profile("nes")
    engine = ConstraintEngine(profile)
    row_ticks = tokenizer.ticks_per_step(480)
    tempo = payload.get("report", {}).get("conditioning_tempo_bpm") or 150.0
    best = payload.get("history", {}).get("best_valid_loss")
    print(
        f"checkpoint {args.checkpoint.name}: vocab {len(tokenizer)}, "
        f"best valid {best}, tempo {tempo}",
        flush=True,
    )

    corpus = (
        corpus_distributions(
            args.cache, tokenizer, split=args.corpus_split, limit=args.corpus_limit
        )
        if args.cache.is_file()
        else {family: Counter() for family in FAMILIES}
    )
    if not args.cache.is_file():
        print(f"note: {args.cache} missing; token comparison skipped", flush=True)

    rows: list[dict[str, Any]] = []
    generated: dict[str, Counter] = {family: Counter() for family in FAMILIES}
    rendered = 0

    structural_settings: list[float | None] = [
        None if str(v).lower() == "none" else float(v) for v in args.structural_temperatures
    ]

    for structural in structural_settings:
        for temperature in args.temperatures:
            for seed in args.seeds:
                result = sample(
                    model,
                    tokenizer,
                    conditioning=Conditioning(n_bars=args.bars, tempo_bpm=tempo),
                    config=SamplingConfig(
                        seed=seed,
                        temperature=temperature,
                        structural_temperature=structural,
                        structural_isolate_eos=args.isolate_eos,
                        top_k=None,
                        top_p=None,
                        max_tokens=args.max_tokens,
                    ),
                )
                for token_id in result.ids:
                    token = tokenizer.vocabulary.token_of(token_id)
                    for family in FAMILIES:
                        if token.startswith(family):
                            generated[family][token[len(family) :]] += 1
                            break

                playable, _repair = engine.apply(result.composition)
                final, _t = enforce_monophony(playable)
                events = final.sounding_events()
                metrics = evaluate_composition(final, profile)
                _channel, melody = melody_line(final)
                end = final.end_tick or 1
                spt = (final.time.seconds(end) / end) if final.time.seconds(end) else 0.0
                repeated_exact, unique_exact = repetition_metrics(bar_signatures(final))
                repeated_row, unique_row = repetition_metrics(
                    bar_signatures(final, quantize_ticks=row_ticks)
                )
                tokens = [tokenizer.vocabulary.token_of(i) for i in result.ids]
                bar_tokens = sum(1 for t in tokens if t == "Bar")
                ends_with_eos = bool(tokens) and tokens[-1] == "<eos>"
                ticks_per_bar = row_ticks * tokenizer.config.steps_per_bar
                occupancy = bar_occupancy(final, ticks_per_bar=int(ticks_per_bar), n_bars=args.bars)
                trailing_empty = 0
                for count in reversed(occupancy):
                    if count:
                        break
                    trailing_empty += 1
                by_channel = Counter(e.channel_id for e in events)
                durations = Counter(e.duration for e in events)
                onsets = sorted({e.tick for e in events})
                iois = [b - a for a, b in zip(onsets, onsets[1:], strict=False)]

                row = {
                    "temperature": temperature,
                    "structural_temperature": structural,
                    "seed": seed,
                    "n_tokens": len(result.ids),
                    "termination": classify_termination(
                        result,
                        max_tokens=args.max_tokens,
                        bars_requested=args.bars,
                        bar_tokens=bar_tokens,
                        occupancy=occupancy,
                        ends_with_eos=ends_with_eos,
                    ),
                    "n_bars": metrics.n_bars,
                    "completed_8_bars": sum(1 for c in occupancy if c) >= args.bars,
                    # The sampler masks <eos> until the requested bars are
                    # emitted, so a short piece is one with empty late bars,
                    # not one that stopped early. Recording both separately
                    # because conflating them misdirected a round of analysis.
                    "n_bar_tokens": bar_tokens,
                    "ends_with_eos": ends_with_eos,
                    "bars_with_content": sum(1 for c in occupancy if c),
                    "empty_trailing_bars": trailing_empty,
                    "bar_occupancy": occupancy,
                    "min_bar_events": min(occupancy) if occupancy else 0,
                    "held_from_stopping": bool(result.underran_length),
                    "held_from_continuing": bool(result.overran_length),
                    "n_events": len(events),
                    "channels": {c: by_channel.get(c, 0) for c in ("P1", "P2", "TR", "NO")},
                    "n_channels_used": sum(1 for c in by_channel.values() if c),
                    "n_percussion": by_channel.get("NO", 0),
                    "notes_per_bar": round(metrics.notes_per_bar, 3),
                    "pitch_class_entropy": round(metrics.pitch_class_entropy, 4),
                    "stepwise_fraction": round(metrics.stepwise_fraction, 4),
                    "ngram_repetition_rate": round(metrics.ngram_repetition_rate, 4),
                    "motif_note_participation": round(motif_note_participation(melody), 4),
                    "repeated_bar_fraction_exact": round(repeated_exact, 4),
                    "repeated_bar_fraction_row_tolerant": round(repeated_row, 4),
                    "unique_bar_fraction_row_tolerant": round(unique_row, 4),
                    "ioi_entropy": round(ioi_entropy([e.tick * spt for e in events]), 4),
                    "n_distinct_iois": len(set(iois)),
                    "n_distinct_durations": len(durations),
                    "top_duration_share": (
                        round(durations.most_common(1)[0][1] / len(events), 4) if events else 0.0
                    ),
                    # A hammered single pitch is the 10k failure; it must be
                    # detectable from the row alone.
                    "top_pitch_share": (
                        round(
                            Counter(e.pitch for e in events if hasattr(e, "pitch")).most_common(1)[
                                0
                            ][1]
                            / max(1, sum(1 for e in events if hasattr(e, "pitch"))),
                            4,
                        )
                        if any(hasattr(e, "pitch") for e in events)
                        else 0.0
                    ),
                }
                rows.append(row)
                if rendered < args.render:
                    render_window(final, args.render_dir / f"t{temperature}_s{seed}.wav")
                    rendered += 1
                print(
                    f"  t={temperature} s={seed:<2} {row['termination']:<24} "
                    f"bars {row['n_bars']:>2} ev {row['n_events']:>4} "
                    f"chan {row['n_channels_used']} perc {row['n_percussion']:>4} "
                    f"pcent {row['pitch_class_entropy']:.2f} "
                    f"ioiH {row['ioi_entropy']:.2f} "
                    f"ngram {row['ngram_repetition_rate']:.3f}",
                    flush=True,
                )

    def dist(key: str) -> dict[str, Any]:
        values = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
        if not values:
            return {}
        return {
            "median": round(st.median(values), 4),
            "mean": round(st.fmean(values), 4),
            "min": round(min(values), 4),
            "max": round(max(values), 4),
            "stdev": round(st.pstdev(values), 4) if len(values) > 1 else 0.0,
        }

    token_report = {
        family: {
            "generated_entropy_bits": round(entropy(generated[family]), 4),
            "corpus_entropy_bits": round(entropy(corpus[family]), 4),
            "kl_generated_vs_corpus_bits": round(
                kl_divergence(generated[family], corpus[family]), 4
            ),
            "generated_distinct": len(generated[family]),
            "corpus_distinct": len(corpus[family]),
            "generated_top3": generated[family].most_common(3),
            "corpus_top3": corpus[family].most_common(3),
        }
        for family in FAMILIES
    }

    payload_out = {
        "checkpoint": str(args.checkpoint),
        "best_valid_loss": best,
        "n_samples": len(rows),
        "bars_requested": args.bars,
        "temperatures": args.temperatures,
        "isolate_eos": args.isolate_eos,
        "structural_temperatures": ["none" if v is None else v for v in structural_settings],
        "seeds": args.seeds,
        "summary": {
            key: dist(key)
            for key in (
                "n_events",
                "n_bars",
                "n_channels_used",
                "n_percussion",
                "notes_per_bar",
                "pitch_class_entropy",
                "stepwise_fraction",
                "ngram_repetition_rate",
                "motif_note_participation",
                "ioi_entropy",
                "n_distinct_iois",
                "n_distinct_durations",
                "top_duration_share",
                "top_pitch_share",
                "repeated_bar_fraction_row_tolerant",
            )
        },
        "completion_rate": round(sum(1 for r in rows if r["completed_8_bars"]) / len(rows), 4),
        "termination_reasons": dict(Counter(r["termination"] for r in rows)),
        "all_four_channels_rate": round(
            sum(1 for r in rows if r["n_channels_used"] == 4) / len(rows), 4
        ),
        "percussion_present_rate": round(
            sum(1 for r in rows if r["n_percussion"] > 0) / len(rows), 4
        ),
        "token_distributions": token_report,
        "rows": rows,
    }
    args.out.write_text(json.dumps(payload_out, indent=2) + "\n")

    print(f"\n{len(rows)} samples, isolate_eos={args.isolate_eos}")
    arms = sorted({r["structural_temperature"] for r in rows}, key=lambda v: (v is None, v))
    if len(arms) > 1 or True:
        hdr = (
            f"\n{'structT':<10}{'n':>4}{'full%':>7}{'barsOK':>8}{'emptyT':>8}"
            f"{'ev':>6}{'n/bar':>7}{'4chan':>7}{'perc%':>7}{'ioiH':>7}{'pitchH':>8}"
        )
        print(hdr)
        print("-" * (len(hdr) - 1))
        for arm in arms:
            group = [r for r in rows if r["structural_temperature"] == arm]
            done = [r for r in group if r["completed_8_bars"]]
            label = "none" if arm is None else f"{arm}"
            print(
                f"{label:<10}{len(group):>4}"
                f"{len(done) / len(group):>7.0%}"
                f"{st.median([r['bars_with_content'] for r in group]):>8.1f}"
                f"{st.median([r['empty_trailing_bars'] for r in group]):>8.1f}"
                f"{st.median([r['n_events'] for r in group]):>6.0f}"
                f"{st.median([r['notes_per_bar'] for r in group]):>7.1f}"
                f"{sum(1 for r in group if r['n_channels_used'] == 4) / len(group):>7.0%}"
                f"{sum(1 for r in group if r['n_percussion'] > 0) / len(group):>7.0%}"
                f"{st.median([r['ioi_entropy'] for r in group]):>7.2f}"
                f"{st.median([r['pitch_class_entropy'] for r in group]):>8.2f}"
            )
        # Completed-only, so premature endings cannot flatter the rhythm.
        print("\n  completed 8-bar samples only:")
        for arm in arms:
            done = [r for r in rows if r["structural_temperature"] == arm and r["completed_8_bars"]]
            if not done:
                continue
            label = "none" if arm is None else f"{arm}"
            print(
                f"  {label:<10}{len(done):>4}"
                f"{st.median([r['notes_per_bar'] for r in done]):>9.1f}"
                f"{st.median([r['ioi_entropy'] for r in done]):>8.2f}"
                f"{st.median([r['pitch_class_entropy'] for r in done]):>8.2f}"
            )
    print(f"  8-bar completion       {payload_out['completion_rate']:.1%}")
    print(f"  all four channels      {payload_out['all_four_channels_rate']:.1%}")
    print(f"  percussion present     {payload_out['percussion_present_rate']:.1%}")
    print(f"  termination            {payload_out['termination_reasons']}")
    print(f"\n{'metric':<36}{'median':>9}{'stdev':>9}{'min':>9}{'max':>9}")
    print("-" * 72)
    for key, value in payload_out["summary"].items():
        if value:
            print(
                f"{key:<36}{value['median']:>9.3f}{value['stdev']:>9.3f}"
                f"{value['min']:>9.3f}{value['max']:>9.3f}"
            )
    print(f"\n{'token family':<14}{'gen H':>9}{'corpus H':>10}{'KL':>9}{'gen/corp distinct':>20}")
    print("-" * 62)
    for family, entry in token_report.items():
        print(
            f"{family:<14}{entry['generated_entropy_bits']:>9.3f}"
            f"{entry['corpus_entropy_bits']:>10.3f}"
            f"{entry['kl_generated_vs_corpus_bits']:>9.3f}"
            f"{entry['generated_distinct']:>10}/{entry['corpus_distinct']:<9}"
        )
    print(f"\nwrote {args.out}; rendered {rendered} to {args.render_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
