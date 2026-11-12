#!/usr/bin/env python3
"""Generate the primary evaluation batch: seeds 0-23, one arm, fixed settings.

The settings are **not** arguments and **not** hardcoded guesses: they are
read from that arm's selection report and asserted against the registered
values below. A typo in a flag therefore cannot quietly produce a batch at
settings nobody chose -- the failure this guards against is the one §9
forbids, a sampling parameter changed after selection.

Seeds 0-23 are disjoint from the development seeds 1000-1003, so this batch
is untouched by anything selection saw.

Every seed yields **one retained outcome**. Nothing is re-rolled: §9 forbids
regenerating an unflattering excerpt, and a rate without its failures is not
a rate.

Durations and 20.0 s window analysability (Addendum D2) are recorded here but
pairing is NOT done here -- it needs both arms, and is a separate step so
that one arm's result cannot silently shape the other's.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rse.snes_v2 import delta as dx  # noqa: E402
from rse.snes_v2 import generate as gen  # noqa: E402
from rse.snes_v2 import grammar as gx  # noqa: E402
from rse.snes_v2 import runners as rx  # noqa: E402
from rse.snes_v2 import statistics as sx  # noqa: E402
from rse.snes_v2.accounting import Accounting  # noqa: E402

#: §2 of the frozen procedure: the primary evaluation seeds.
EVALUATION_SEEDS = tuple(range(24))

#: The registered count, as a LITERAL. `intended=len(EVALUATION_SEEDS)` would
#: be tautological -- it moves with whatever the list happens to be, so a
#: truncated seed list would reconcile against itself and pass. Checking
#: against this literal is what makes the population assertion able to fail.
N_EVALUATION_SEEDS = 24

#: Addendum D2: the common listening window.
WINDOW_SEC = 20.0

#: The registered outputs of sampling selection. Asserted against each arm's
#: report, so the batch cannot run at settings nobody selected.
REGISTERED_SETTINGS = {
    "model": {"temperature": 1.0, "top_p": 1.0},
    "baseline": {"temperature": 0.8, "top_p": 0.95},
}

GRID_HZ = 4000.0
EXPECTED_STEP = 5400
EXPECTED_PARAMETERS = 5_646_336


def identifier(arm: str, seed: int) -> str:
    return f"eval_{arm}_s{seed:02d}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("model", "baseline"), required=True)
    parser.add_argument("--selection-report", required=True,
                        help="that arm's sampling_selection_<arm>.json")
    parser.add_argument("--checkpoint", default=str(ROOT / "best.pt"))
    parser.add_argument("--windows", default=str(ROOT / "windows"))
    parser.add_argument("--delta-vocab", default=str(ROOT / "delta_vocab_k512.json"))
    parser.add_argument("--out", default=str(ROOT / "eval_out"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    acc = Accounting(stage=f"evaluation_batch_{args.arm}")

    # --- the settings, read from the selection report, then asserted -------
    report = json.loads(Path(args.selection_report).read_text())
    if report.get("arm") != args.arm:
        raise SystemExit(f"selection report is for arm {report.get('arm')!r}, "
                         f"not {args.arm!r}")
    if not report.get("successful"):
        raise SystemExit("selection for this arm was declared unsuccessful; "
                         "there is no registered setting to generate at")
    chosen = report["selected"]
    expected = REGISTERED_SETTINGS[args.arm]
    for key, want in expected.items():
        if chosen[key] != want:
            raise SystemExit(
                f"{args.arm}: report says {key}={chosen[key]}, registered "
                f"value is {want}. Refusing to generate at an unregistered "
                f"setting (§9).")
    config_of = lambda seed: gen.SamplingConfig(                # noqa: E731
        temperature=chosen["temperature"], top_p=chosen["top_p"], seed=seed)

    raw = json.loads(Path(args.delta_vocab).read_text())
    vocab = gx.Vocab(delta=dx.DeltaCodec(
        direct={int(k): v for k, v in raw["direct"].items()},
        n_observed=raw["n_distinct_train"]))

    # --- the model, from the SAME library loaders selection used ------
    if args.arm == "model":
        distribution, meta = rx.transformer_distribution(
            Path(args.checkpoint), args.device)
        acc.check("checkpoint_step", observed=meta["step"],
                  intended=EXPECTED_STEP, population="locked checkpoint",
                  strict=False)
        acc.check("n_parameters", observed=meta["n_parameters"],
                  intended=EXPECTED_PARAMETERS,
                  population="declared parameter count", strict=False)
    else:
        manifest = json.loads((Path(args.windows) / "MANIFEST.json").read_text())
        distribution, meta = rx.baseline_distribution(
            Path(args.windows), manifest["pad_id"] + 1)
    acc.exit_if_failed()

    print(f"  arm {args.arm}  temperature {chosen['temperature']}  "
          f"top-p {chosen['top_p']}  (from the selection report, asserted)",
          flush=True)
    print(f"  {len(EVALUATION_SEEDS)} evaluation seeds, disjoint from "
          f"development; window {WINDOW_SEC}s\n", flush=True)

    records: list[dict] = []
    started = time.time()
    for seed in EVALUATION_SEEDS:
        excerpt = gen.generate(distribution, vocab, config_of(seed),
                               arm=args.arm)
        statistics, note = rx.measure(excerpt, vocab)
        reaches = excerpt.duration_sec >= WINDOW_SEC
        record = excerpt.to_dict() | {
            "identifier": identifier(args.arm, seed),
            "temperature": chosen["temperature"], "top_p": chosen["top_p"],
            "decodable": statistics is not None,
            "measurement_note": note,
            "statistics": statistics,
            "reaches_window": reaches,
            "window_sec": WINDOW_SEC,
        }
        records.append(record)
        (out_dir / f"{record['identifier']}.json").write_text(
            json.dumps({"tokens": excerpt.tokens} | record, indent=1))
        print(f"    seed {seed:2d}  {excerpt.outcome:17s} "
              f"{excerpt.duration_sec:7.2f}s  "
              f"{'reaches 20s' if reaches else 'SHORT OF 20s':12s}  "
              f"illegal {excerpt.illegal_mass_mean:.4f}  "
              f"{'decodable' if statistics else 'NOT DECODABLE: ' + note}",
              flush=True)

    tally: dict[str, int] = {c: 0 for c in
                             ("complete", "short", "budget_exhausted", "invalid")}
    for record in records:
        tally[record["outcome"]] += 1
    acc.check("registered_seed_count", observed=len(EVALUATION_SEEDS),
              intended=N_EVALUATION_SEEDS,
              population="§2 primary evaluation seeds 0-23", strict=False)
    acc.check("attempted_outcomes", observed=len(records),
              intended=N_EVALUATION_SEEDS,
              population="one retained outcome per evaluation seed",
              strict=False)
    acc.check("distinct_identifiers",
              observed=len({r["identifier"] for r in records}),
              intended=N_EVALUATION_SEEDS,
              population="predetermined output identifiers", strict=False)
    acc.note("outcome_tally", tally)
    acc.note("n_reaching_window", sum(1 for r in records if r["reaches_window"]))
    acc.note("n_decodable", sum(1 for r in records if r["decodable"]))

    decodable = [r["statistics"] for r in records if r["decodable"]]
    payload = {
        "schema": "snes_v2.evaluation_batch.v1",
        "arm": args.arm,
        "model": meta,
        "settings": {"temperature": chosen["temperature"],
                     "top_p": chosen["top_p"],
                     "provenance": "read from the selection report and "
                                   "asserted against the registered value"},
        "seeds": list(EVALUATION_SEEDS),
        "seeds_note": "disjoint from the development seeds 1000-1003",
        "no_rerolls": "every seed yields one retained outcome; §9 forbids "
                      "regenerating an unflattering excerpt",
        "window_sec": WINDOW_SEC,
        "denominators": {
            "attempted_outcomes": len(records),
            "expected_attempted_outcomes": N_EVALUATION_SEEDS,
            "decodable": sum(1 for r in records if r["decodable"]),
            "reaching_window": sum(1 for r in records if r["reaches_window"]),
            "note": "reported separately; pairing needs both arms and is a "
                    "separate step",
        },
        "outcome_tally": tally,
        "aggregate_over_decodable": sx.aggregate(decodable) if decodable else None,
        "excerpts": records,
        "wall_clock_sec": time.time() - started,
    }
    (out_dir / f"evaluation_batch_{args.arm}.json").write_text(
        json.dumps(payload, indent=1))
    acc.write(out_dir / f"evaluation_batch_{args.arm}_accounting.json")

    durations = sorted(r["duration_sec"] for r in records)
    print(f"\n  attempted outcomes  {len(records)} / {N_EVALUATION_SEEDS}")
    print(f"  outcome tally       {tally}")
    print(f"  decodable           {payload['denominators']['decodable']}/"
          f"{len(records)}")
    print(f"  reaching {WINDOW_SEC}s      "
          f"{payload['denominators']['reaching_window']}/{len(records)}"
          f"   <- analysable pairs need BOTH arms to reach it")
    print(f"  durations           min {durations[0]:.2f}s  "
          f"median {durations[len(durations) // 2]:.2f}s  "
          f"max {durations[-1]:.2f}s")
    print(f"  {payload['wall_clock_sec']:.1f}s; written to {out_dir}")
    acc.exit_if_failed()


if __name__ == "__main__":
    main()
