"""Token-length distribution for a codec, and a context policy with no silent
truncation.

Silent truncation is not an acceptable default. A cropped window loses music,
and Gate 3A measured a second cost: at 1024 tokens, 1114 of 3900 cropped
windows changed their motif label, so the model would train against labels
describing music outside its context. Any policy here has to say explicitly
what happens to a window that does not fit.

Three policies are costed from the measured distribution:

- **fit**: pick a context that holds every window. No loss, maximum attention
  cost, and the cost is set by the single longest window in the corpus.
- **exclude**: keep a smaller context and drop the windows that overflow,
  reporting exactly how many and whether they are a biased subset. Honest but
  lossy, and the bias matters -- if the excluded windows are the dense ones,
  the model is trained on a corpus whose density distribution has been clipped.
- **split**: cut an overlong window into shorter ones at bar boundaries, so no
  music is lost and labels are re-derived per piece. Costs more windows.

The script reports what each would cost. It does not choose: that is a
modelling decision with a compute budget attached.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import fmean, median
from typing import Any

from rse.codec.benchmark import percentile
from rse.events.pulse import detect_pulse
from rse.tokenizers.tokenizer import GridFit, RseTokenizer
from rse.tokenizers.vocab import TokenizerConfig
from rse.training.corpus import corpus_from_release

CANDIDATE_CONTEXTS = (1024, 1536, 2048, 2304, 2560, 3072, 4096)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="nesmdb")
    parser.add_argument("--release", default="midi")
    parser.add_argument("--split", default="train", choices=("train", "valid"))
    parser.add_argument("--limit", type=int, default=600)
    parser.add_argument("--window-bars", type=int, default=8)
    parser.add_argument("--codec", default="D")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--tempo-ratio", type=float, default=1.01)
    parser.add_argument("--out", type=Path, default=Path("context_policy.json"))
    args = parser.parse_args()

    config = (
        TokenizerConfig(tempo_ratio=args.tempo_ratio)
        if args.codec == "A"
        else TokenizerConfig(
            codec=args.codec, sub_row_divisions=args.k, tempo_ratio=args.tempo_ratio
        )
    )
    tokenizer = RseTokenizer(config)
    label = args.codec if args.codec == "A" else f"{args.codec}{args.k}"

    print(f"corpus    : loading {args.split}...", flush=True)
    entries, info = corpus_from_release(
        args.dataset, args.release, limit=args.limit, splits=(args.split,)
    )
    print(f"corpus    : {len(entries)} pieces, codec {label}/{args.tempo_ratio}", flush=True)

    lengths: list[int] = []
    # Density of each window, so an exclusion policy can be checked for bias
    # rather than assumed harmless.
    densities: list[int] = []
    declined = 0

    for index, (composition, _split, _pid) in enumerate(entries, 1):
        pulse = detect_pulse(composition)
        if not pulse.found or pulse.period_ticks <= 0:
            declined += 1
            continue
        row = pulse.period_ticks
        window_ticks = int(round(row * 16 * args.window_bars))
        if window_ticks <= 0:
            continue
        fit = GridFit(step_ticks=row, phase_ticks=0.0, aligned=True, period_sec=pulse.period_sec)
        end = composition.end_tick
        for start in range(0, max(1, end - window_ticks), window_ticks):
            from rse.events.schema import Composition

            events = [
                type(e)(**{**vars(e), "tick": e.tick - start})
                for e in composition.sounding_events()
                if start <= e.tick < start + window_ticks
            ]
            if len(events) < 8:
                continue
            window = Composition(
                hardware_profile=composition.hardware_profile,
                ticks_per_beat=composition.ticks_per_beat,
                events=events,
                tempo_map=list(composition.tempo_map),
                time_signatures=list(composition.time_signatures),
            )
            encoded = tokenizer.encode(window, fit=fit)
            lengths.append(len(encoded.ids))
            densities.append(len(events))
        if index % 100 == 0:
            print(f"          : {index}/{len(entries)}", flush=True)

    if not lengths:
        print("error: no windows produced", flush=True)
        return 2

    policies: dict[str, Any] = {}
    for context in CANDIDATE_CONTEXTS:
        over = [i for i, n in enumerate(lengths) if n > context]
        inside_density = [densities[i] for i, n in enumerate(lengths) if n <= context]
        over_density = [densities[i] for i in over]
        policies[str(context)] = {
            "n_over": len(over),
            "share_over": round(len(over) / len(lengths), 5),
            "median_density_inside": round(median(inside_density), 1) if inside_density else 0,
            "median_density_excluded": round(median(over_density), 1) if over_density else 0,
            # If the overflowing windows are the dense ones, excluding them
            # clips the corpus density distribution the model learns from.
            "density_bias": round(
                (median(over_density) / median(inside_density))
                if inside_density and over_density
                else 1.0,
                3,
            ),
            # Splitting an 8-bar window in half roughly halves its length, so
            # the extra windows are those that still would not fit at half.
            "windows_if_split_in_half": sum(1 for n in lengths if n > context),
        }

    payload = {
        "corpus": info.to_dict(),
        "split": args.split,
        "codec": label,
        "tempo_ratio": args.tempo_ratio,
        "window_bars": args.window_bars,
        "vocab_size": len(tokenizer),
        "n_windows": len(lengths),
        "n_declined_no_row": declined,
        "tokens": {
            "mean": round(fmean(lengths), 1),
            "median": round(median(lengths)),
            "p75": round(percentile([float(n) for n in lengths], 0.75)),
            "p90": round(percentile([float(n) for n in lengths], 0.90)),
            "p95": round(percentile([float(n) for n in lengths], 0.95)),
            "p99": round(percentile([float(n) for n in lengths], 0.99)),
            "max": max(lengths),
        },
        "policies": policies,
    }
    args.out.write_text(json.dumps(payload, indent=2) + "\n")

    t = payload["tokens"]
    print(f"\ncodec {label}/{args.tempo_ratio}, {len(lengths)} windows, vocab {len(tokenizer)}")
    print(
        f"tokens: median {t['median']}  p75 {t['p75']}  p90 {t['p90']}  "
        f"p95 {t['p95']}  p99 {t['p99']}  max {t['max']}"
    )
    print(f"\n{'context':>9}{'over':>8}{'share':>9}{'dens in':>10}{'dens out':>10}{'bias':>7}")
    print("-" * 53)
    for context, row in policies.items():
        print(
            f"{context:>9}{row['n_over']:>8}{row['share_over']:>9.4f}"
            f"{row['median_density_inside']:>10.1f}{row['median_density_excluded']:>10.1f}"
            f"{row['density_bias']:>7.2f}"
        )
    print(f"\nwrote {args.out}")
    print(f"note: {declined} pieces had no fitted row")
    histogram = dict(sorted(Counter(n // 500 * 500 for n in lengths).items()))
    print(f"length histogram (500-token buckets): {histogram}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
