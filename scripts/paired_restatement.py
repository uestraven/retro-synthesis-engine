"""Re-measure the paired run's 16-bar arms against the corpus control.

The paired experiment recorded exact bar restatement and nothing else, which
turned out to be the wrong measure: the corpus restates exact bars *less* than
the generated pieces do, so "median 0.115, one exact bar in 24" was never
evidence of a defect. What distinguishes real NES music over sixteen bars is
contour recurrence -- the same melodic shape coming back transposed, or
rhythmically altered, or reharmonised -- and the paired run never computed it.

So this rebuilds each long arm from its stored token stream and runs it
through `rse.analysis.restatement`, the same module the corpus reference uses.
Rebuilding rather than re-generating keeps the comparison exact: the audit
holds the raw ids, so no sampling happens here and the result is deterministic.

The pipeline is reproduced verbatim from `paired_length.py`, including the
constraint engine running last, because a different arrangement path would
change the composition being measured and the two sides would no longer be
comparable.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import json
import statistics as st
from typing import Any

from rse.analysis.restatement import NGRAM, analyse_halves, ticks_per_bar
from rse.codec.benchmark import enforce_monophony
from rse.hardware.constraints import ConstraintEngine
from rse.hardware.profile import load_profile
from rse.tokenizers.frozen import frozen_config
from rse.tokenizers.tokenizer import GridFit, RseTokenizer
from rse.transfer.arranger import arrange_for_nes
from rse.transfer.fermata import Vocab, decode, trim_to_bars

ROW_SEC = 0.1

METRICS = (
    "exact_restatement_mean",
    "contour_recurrence",
    "interval_recurrence",
    "pitch_overlap",
    "new_pitches",
    "lost_pitches",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("/tmp/fermata_dl"))
    parser.add_argument("--audit", type=Path, default=Path("listen_paired_audit"))
    parser.add_argument("--paired", type=Path, default=Path("paired_length.json"))
    parser.add_argument(
        "--reference", type=Path, default=Path("restatement_reference.json")
    )
    parser.add_argument("--bars", type=int, default=16)
    parser.add_argument("--ngram", type=int, default=NGRAM)
    parser.add_argument("--out", type=Path, default=Path("paired_restatement.json"))
    args = parser.parse_args()

    vocab = Vocab.load(args.model / "tokenizer" / "vocab.json")
    tokenizer = RseTokenizer(frozen_config())
    profile = load_profile("nes")
    engine = ConstraintEngine(profile)
    fit = GridFit(
        step_ticks=float(tokenizer.ticks_per_step(480)),
        phase_ticks=0.0,
        aligned=True,
        period_sec=ROW_SEC,
    )
    bar_ticks = ticks_per_bar()

    pairs = json.loads(args.paired.read_text())["pairs"]
    usable = [p for p in pairs if p["usable"]]
    print(f"{len(usable)} usable pairs in {args.paired}")

    rows: list[dict[str, Any]] = []
    for pair in usable:
        audit = args.audit / f"{pair['name']}.pair.json"
        if not audit.exists():
            print(f"  ! no audit for {pair['name']}")
            continue
        raw_ids = json.loads(audit.read_text())["raw_ids"]
        trimmed = trim_to_bars(raw_ids, vocab=vocab, n_bars=args.bars)
        if not (trimmed.complete and trimmed.n_bars == args.bars):
            print(f"  ! {pair['name']} no longer trims to {args.bars} bars")
            continue
        score, _report = decode(trimmed.ids, vocab=vocab)
        arranged, _arrange = arrange_for_nes(
            score, row_sec=ROW_SEC, divisions=tokenizer.config.onset_divisions
        )
        arranged, _trunc = enforce_monophony(arranged)
        rounded = enforce_monophony(
            tokenizer.decode(tokenizer.encode(arranged, fit=fit).ids).composition
        )[0]
        final, _repair = engine.apply(rounded)
        report = analyse_halves(
            final, n_bars=args.bars, bar_ticks=bar_ticks, ngram=args.ngram
        )
        rows.append(
            {"name": pair["name"], "scenario": pair["scenario"], **report.to_dict()}
        )
        def show(value: float | None) -> str:
            """`None` means the melody was too short to hold an n-gram, which
            is a different outcome from zero recurrence."""
            return "n/a" if value is None else f"{value:.3f}"

        print(
            f"  {pair['name']:<24} exact {report.exact_restatement_mean:.3f}  "
            f"contour {show(report.contour_recurrence):>6}  "
            f"interval {show(report.interval_recurrence):>6}  "
            f"lead {report.lead_channel}  "
            f"melody {report.n_melody_first}->{report.n_melody_second}",
            flush=True,
        )

    if not rows:
        print("error: nothing re-measured", file=sys.stderr)
        return 2

    reference = (
        json.loads(args.reference.read_text()) if args.reference.exists() else {}
    )
    corpus = reference.get("corpus", {})

    summary: dict[str, Any] = {}
    print(
        f"\n{'metric':<24}{'gen median':>12}{'corpus med':>12}"
        f"{'corpus p25':>12}{'corpus p75':>12}  verdict"
    )
    print("-" * 86)
    for metric in METRICS:
        values = [r[metric] for r in rows if r.get(metric) is not None]
        if not values:
            continue
        generated = st.median(values)
        band = corpus.get(metric) or {}
        entry: dict[str, Any] = {
            "generated_median": round(generated, 4),
            "generated_n": len(values),
            "corpus_median": band.get("median"),
            "corpus_p25": band.get("p25"),
            "corpus_p75": band.get("p75"),
        }
        verdict = "no corpus reference"
        if band:
            low, high = band["p25"], band["p75"]
            if generated < low:
                verdict = "BELOW corpus p25"
            elif generated > high:
                verdict = "above corpus p75"
            else:
                verdict = "within corpus IQR"
            entry["in_corpus_iqr"] = low <= generated <= high
        summary[metric] = entry
        print(
            f"  {metric:<22}{generated:>12.3f}"
            f"{(band.get('median') if band else float('nan')):>12.3f}"
            f"{(band.get('p25') if band else float('nan')):>12.3f}"
            f"{(band.get('p75') if band else float('nan')):>12.3f}  {verdict}"
        )

    args.out.write_text(
        json.dumps(
            {
                "n_measured": len(rows),
                "ngram": args.ngram,
                "reference": str(args.reference),
                "reference_split": reference.get("split"),
                "reference_n_spans": reference.get("n_spans_measured"),
                "summary": summary,
                "pieces": rows,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
