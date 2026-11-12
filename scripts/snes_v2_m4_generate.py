#!/usr/bin/env python3
"""Milestone 4: generate the 36 frozen seeds and audit eligibility.

Endpoint A (generation reliability) is produced here and is
**unconditional**: every one of the 36 registered seeds counts, including
decode failures, premature stops and budget exhaustion. Nothing is re-rolled
and nothing is substituted.

Endpoint B's stimulus set is *selected* here but not evaluated: the first 24
eligible excerpts in **ascending seed order**, a rule fixed in the protocol
so no excerpt is chosen for how it sounds. **No audio is rendered and nothing
is heard.**

The script is deliberately separate from the Milestone 3 batch script rather
than a parameterisation of it, so Milestone 3's frozen seed population and
artifacts cannot be disturbed.

Operational condition B of Addendum M4-1: if the frozen procedure cannot
supply 24 primary comparisons, the study does not proceed as a confirmatory
listening experiment. This script reports that verdict; it does not decide to
relax anything.
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
from rse.snes_v2.binomial import clopper_pearson  # noqa: E402

#: Frozen in V2_MILESTONE4_PROTOCOL.md §2. Disjoint from Milestone 3's
#: evaluation seeds 0-23 and the development seeds 1000-1003.
M4_SEEDS = tuple(range(3000, 3036))
#: A LITERAL, so the population check cannot reconcile against itself.
N_M4_SEEDS = 36

#: §4: the common listening window. §2: the required primary comparisons.
WINDOW_SEC = 20.0
TARGET_SEC = 45.0
REQUIRED_PRIMARY_PAIRS = 24

#: The registered Milestone 3 selection, asserted against the report.
REGISTERED_TEMPERATURE = 1.0
REGISTERED_TOP_P = 1.0
EXPECTED_STEP = 5400
EXPECTED_PARAMETERS = 5_646_336

#: Seeds that must NOT appear: already examined for duration.
FORBIDDEN_SEEDS = frozenset(range(24)) | {1000, 1001, 1002, 1003}


def identifier(seed: int) -> str:
    return f"m4_model_s{seed}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-report", required=True)
    parser.add_argument("--checkpoint", default=str(ROOT / "best.pt"))
    parser.add_argument("--delta-vocab", default=str(ROOT / "delta_vocab_k512.json"))
    parser.add_argument("--out", default=str(ROOT / "m4_out"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    acc = Accounting(stage="m4_generation")

    overlap = set(M4_SEEDS) & FORBIDDEN_SEEDS
    if overlap:
        raise SystemExit(f"seed population overlaps already-examined seeds: "
                         f"{sorted(overlap)}")

    report = json.loads(Path(args.selection_report).read_text())
    if report.get("arm") != "model" or not report.get("successful"):
        raise SystemExit("selection report must be the successful model arm")
    chosen = report["selected"]
    if (chosen["temperature"] != REGISTERED_TEMPERATURE
            or chosen["top_p"] != REGISTERED_TOP_P):
        raise SystemExit(
            f"report says t={chosen['temperature']} p={chosen['top_p']}; "
            f"registered is t={REGISTERED_TEMPERATURE} "
            f"p={REGISTERED_TOP_P}. Refusing to generate (§9).")

    raw = json.loads(Path(args.delta_vocab).read_text())
    vocab = gx.Vocab(delta=dx.DeltaCodec(
        direct={int(k): v for k, v in raw["direct"].items()},
        n_observed=raw["n_distinct_train"]))

    distribution, meta = rx.transformer_distribution(
        Path(args.checkpoint), args.device)
    acc.check("checkpoint_step", observed=meta["step"], intended=EXPECTED_STEP,
              population="locked Milestone 3 checkpoint", strict=False)
    acc.check("n_parameters", observed=meta["n_parameters"],
              intended=EXPECTED_PARAMETERS,
              population="declared parameter count", strict=False)
    acc.exit_if_failed()

    print(f"  Milestone 4 generation: seeds {M4_SEEDS[0]}-{M4_SEEDS[-1]} "
          f"({N_M4_SEEDS})")
    print(f"  t={chosen['temperature']} p={chosen['top_p']} (asserted); "
          f"window {WINDOW_SEC}s; target {TARGET_SEC}s")
    print("  NO AUDIO IS RENDERED AND NOTHING IS HEARD\n", flush=True)

    records: list[dict] = []
    started = time.time()
    for seed in M4_SEEDS:
        excerpt = gen.generate(
            distribution, vocab,
            gen.SamplingConfig(temperature=chosen["temperature"],
                               top_p=chosen["top_p"], seed=seed),
            arm="model")
        statistics, note = rx.measure(excerpt, vocab)
        decodable = statistics is not None
        if excerpt.seed != seed:
            raise SystemExit(
                f"generator returned seed {excerpt.seed} for requested seed "
                f"{seed}; every excerpt's provenance would be mis-attributed")
        record = excerpt.to_dict() | {
            # From the LOOP, not from the excerpt: the identifier and the
            # seed must come from one source, or an audit can attribute an
            # excerpt to the wrong seed without any error.
            "seed": seed,
            "identifier": identifier(seed),
            "temperature": chosen["temperature"], "top_p": chosen["top_p"],
            "decodable": decodable,
            "measurement_note": note,
            "statistics": statistics,
            "reaches_window": excerpt.duration_sec >= WINDOW_SEC,
            "reaches_target": excerpt.duration_sec >= TARGET_SEC,
            # Endpoint B eligibility needs BOTH: a decodable performance and
            # enough audio. A 25 s stream that will not parse is not a
            # stimulus.
            "eligible_for_endpoint_b": decodable
            and excerpt.duration_sec >= WINDOW_SEC,
        }
        records.append(record)
        (out_dir / f"{record['identifier']}.json").write_text(
            json.dumps({"tokens": excerpt.tokens} | record, indent=1))
        flags = []
        if record["reaches_target"]:
            flags.append("45s")
        elif record["reaches_window"]:
            flags.append("20s")
        else:
            flags.append("SHORT")
        if not decodable:
            flags.append("UNDECODABLE")
        print(f"    seed {seed}  {excerpt.outcome:17s} "
              f"{excerpt.duration_sec:7.2f}s  {'/'.join(flags):14s} "
              f"illegal {excerpt.illegal_mass_mean:.4f}", flush=True)

    # --- Endpoint A: unconditional, over ALL registered seeds -------------
    n = len(records)
    reaching_window = sum(1 for r in records if r["reaches_window"])
    reaching_target = sum(1 for r in records if r["reaches_target"])
    decodable_count = sum(1 for r in records if r["decodable"])
    eligible = [r for r in records if r["eligible_for_endpoint_b"]]

    tally = {c: 0 for c in ("complete", "short", "budget_exhausted", "invalid")}
    for r in records:
        tally[r["outcome"]] += 1

    acc.check("registered_seed_count", observed=len(M4_SEEDS),
              intended=N_M4_SEEDS, population="§2 Milestone 4 seeds 3000-3035",
              strict=False)
    acc.check("attempted_outcomes", observed=n, intended=N_M4_SEEDS,
              population="one retained outcome per registered seed",
              strict=False)
    acc.check("distinct_identifiers",
              observed=len({r["identifier"] for r in records}),
              intended=N_M4_SEEDS, population="predetermined identifiers",
              strict=False)

    # --- Endpoint B stimulus set: first 24 eligible, by seed order --------
    selected = sorted(eligible, key=lambda r: r["seed"])[:REQUIRED_PRIMARY_PAIRS]
    feasible = len(selected) >= REQUIRED_PRIMARY_PAIRS

    endpoint_a = {
        "denominator": n,
        "unconditional": True,
        "decodable": {"count": decodable_count,
                      "rate": decodable_count / n,
                      "ci95": clopper_pearson(decodable_count, n)},
        "reaching_20s": {"count": reaching_window,
                         "rate": reaching_window / n,
                         "ci95": clopper_pearson(reaching_window, n)},
        "reaching_45s": {"count": reaching_target,
                         "rate": reaching_target / n,
                         "ci95": clopper_pearson(reaching_target, n)},
        "interval_method": "Clopper-Pearson exact",
    }
    payload = {
        "schema": "snes_v2.m4_generation.v1",
        "protocol": "V2_MILESTONE4_PROTOCOL.md, frozen 2026-11-12, "
                    "with Addendum M4-1",
        "no_audio_rendered": True,
        "nothing_heard": True,
        "model": meta,
        "settings": {"temperature": chosen["temperature"],
                     "top_p": chosen["top_p"],
                     "provenance": "read from the Milestone 3 selection "
                                   "report and asserted against the "
                                   "registered value"},
        "seeds": list(M4_SEEDS),
        "seeds_note": "disjoint from 0-23 and 1000-1003, both already "
                      "examined for duration",
        "outcome_tally": tally,
        "endpoint_a": endpoint_a,
        "endpoint_b_stimulus_set": {
            "window_sec": WINDOW_SEC,
            "required_pairs": REQUIRED_PRIMARY_PAIRS,
            "eligible_count": len(eligible),
            "selection_rule": "first 24 eligible in ascending seed order; "
                              "no excerpt chosen for how it sounds",
            "selected_seeds": [r["seed"] for r in selected],
            "feasible": feasible,
            "verdict": (
                "feasible: the frozen procedure supplies the 24 required "
                "primary comparisons"
                if feasible else
                f"NOT FEASIBLE: {len(eligible)} eligible of "
                f"{REQUIRED_PRIMARY_PAIRS} required. Per Addendum M4-1 "
                f"operational condition B the study does not proceed as a "
                f"confirmatory listening experiment; results are preserved "
                f"and the limitation reported without redefining success."),
        },
        "excerpts": records,
        "wall_clock_sec": time.time() - started,
    }
    (out_dir / "m4_generation.json").write_text(json.dumps(payload, indent=1))
    acc.note("eligible_for_endpoint_b", len(eligible))
    acc.note("endpoint_b_feasible", feasible)
    acc.write(out_dir / "m4_generation_accounting.json")

    print(f"\n  ENDPOINT A (unconditional, denominator {n})")
    for name, key in (("decodable", "decodable"),
                      ("reaching 20s", "reaching_20s"),
                      ("reaching 45s", "reaching_45s")):
        block = endpoint_a[key]
        low, high = block["ci95"]
        print(f"    {name:14s} {block['count']:2d}/{n}  "
              f"{block['rate']:6.1%}  95% CI [{low:.1%}, {high:.1%}]")
    print(f"    outcome tally  {tally}")
    print(f"\n  ENDPOINT B stimulus set")
    print(f"    eligible       {len(eligible)}/{n} "
          f"(decodable AND >= {WINDOW_SEC}s)")
    print(f"    required       {REQUIRED_PRIMARY_PAIRS}")
    print(f"    selected seeds {[r['seed'] for r in selected]}")
    print(f"\n  {'FEASIBLE' if feasible else 'NOT FEASIBLE'}: "
          f"{payload['endpoint_b_stimulus_set']['verdict']}")
    print(f"  {payload['wall_clock_sec']:.1f}s; written to {out_dir}")

    acc.exit_if_failed()
    if not feasible:
        sys.exit(4)


if __name__ == "__main__":
    main()
