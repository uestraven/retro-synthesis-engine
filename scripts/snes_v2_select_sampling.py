#!/usr/bin/env python3
"""Run the frozen sampling-selection grid for one arm. No audio is heard.

Executes §§1-7 of `V2_GENERATION_PROCEDURE.md` with Amendment A1 eligibility:
10 candidates x 4 development seeds = 40 attempted outcomes, every one
retained, candidates eligible only when all four decode into performances
with defined metrics.

The selection logic itself lives in `rse.snes_v2.selection` and is tested
there. This script is the driver: it loads the locked checkpoint, calls the
frozen generator, measures with the frozen statistics, and writes a manifest.

Exits nonzero -- so a pipeline cannot read a partial result as a success --
when the attempted population is not 40, when any seed produced no retained
outcome, or when selection is declared unsuccessful under A1 §6.

§8: the baseline comparator gets this identical procedure. `--arm baseline`
runs it with the same grid, seeds, objective, tie-break, stopping rules,
rollover and mask.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rse.snes_v2 import delta as dx  # noqa: E402
from rse.snes_v2 import generate as gen  # noqa: E402
from rse.snes_v2 import grammar as gx  # noqa: E402
from rse.snes_v2 import selection as sel  # noqa: E402
from rse.snes_v2 import statistics as sx  # noqa: E402
from rse.snes_v2.accounting import Accounting  # noqa: E402

GRID_HZ = 4000.0
#: The locked checkpoint. Asserted, not trusted.
EXPECTED_STEP = 5400
EXPECTED_PARAMETERS = 5_646_336


def transformer_distribution(checkpoint: Path, device: str):
    """Returns (callable, metadata). The callable maps a context to probs."""
    import torch
    import torch.nn.functional as F

    from rse.snes_lm.train import Config, Decoder

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = Config(**{**state["config"], "device": device})
    model = Decoder(state["vocab_size"], cfg)
    model.load_state_dict(state["model"])
    model.eval()
    model.to(device)
    n_parameters = sum(p.numel() for p in model.parameters())

    def distribution(context):
        with torch.no_grad():
            row = torch.tensor([list(context)], dtype=torch.long, device=device)
            logits = model(row)[0, -1, :]
            return F.softmax(logits.float(), dim=-1).cpu().tolist()

    return distribution, {"step": state["step"], "n_parameters": n_parameters,
                          "vocab_size": state["vocab_size"]}


def baseline_distribution(window_dir: Path, vocab_size: int):
    from rse.snes_v2 import baseline as bl

    train = np.load(window_dir / "train.npz")
    streams = [train["tokens"][i, :train["lengths"][i]].astype(int).tolist()
               for i in range(len(train["lengths"]))]
    model = bl.G3KneserNey(vocab_size=vocab_size, order=1).fit(streams)

    def distribution(context):
        return model.distribution(list(context)[-model.order:])

    return distribution, {"family": "kneser_ney", "order": 1,
                          "n_train_streams": len(streams)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("model", "baseline"), default="model")
    parser.add_argument("--checkpoint", default=str(ROOT / "tmp/run/best.pt"))
    parser.add_argument("--windows", default=str(ROOT / "tmp/windows"))
    parser.add_argument("--delta-vocab", default=str(ROOT / "tmp/delta_vocab_k512.json"))
    parser.add_argument("--reference", default=str(ROOT / "tmp/reference_stats.json"))
    parser.add_argument("--out", default=str(ROOT / "tmp/out"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = json.loads(Path(args.delta_vocab).read_text())
    vocab = gx.Vocab(delta=dx.DeltaCodec(
        direct={int(k): v for k, v in raw["direct"].items()},
        n_observed=raw["n_distinct_train"]))
    reference = json.loads(Path(args.reference).read_text())["reference"]

    acc = Accounting(stage=f"sampling_selection_{args.arm}")

    if args.arm == "model":
        distribution, meta = transformer_distribution(
            Path(args.checkpoint), args.device)
        acc.check("checkpoint_step", observed=meta["step"],
                  intended=EXPECTED_STEP, population="locked checkpoint", strict=False)
        acc.check("n_parameters", observed=meta["n_parameters"],
                  intended=EXPECTED_PARAMETERS,
                  population="declared parameter count", strict=False)
        acc.check("vocab_size", observed=meta["vocab_size"],
                  intended=vocab.size + 1,
                  population="G3 vocabulary plus PAD", strict=False)
    else:
        manifest = json.loads(
            (Path(args.windows) / "MANIFEST.json").read_text())
        distribution, meta = baseline_distribution(
            Path(args.windows), manifest["pad_id"] + 1)
    acc.exit_if_failed()

    print(f"  arm {args.arm}; vocabulary {vocab.size:,}; "
          f"reference {reference['attacks_per_sec']:.4f} attacks/s, "
          f"{reference['mean_simultaneity']:.4f} simultaneity", flush=True)

    candidates: list[sel.Candidate] = []
    excerpt_records: list[dict] = []
    started = time.time()
    for temperature in sel.TEMPERATURES:
        for top_p in sel.TOP_P_VALUES:
            candidate = sel.Candidate(temperature=temperature, top_p=top_p)
            for seed in sel.DEVELOPMENT_SEEDS:
                config = gen.SamplingConfig(temperature=temperature,
                                            top_p=top_p, seed=seed)
                excerpt = gen.generate(distribution, vocab, config,
                                       arm=args.arm)
                statistics, note = measure(excerpt, vocab)
                candidate.outcomes.append(sel.Outcome(
                    seed=seed, outcome=excerpt.outcome,
                    statistics=statistics, reason=note))
                record = excerpt.to_dict() | {
                    "temperature": temperature, "top_p": top_p,
                    "scorable": statistics is not None,
                    "measurement_note": note,
                    "statistics": statistics,
                    #: Predetermined identifier, so the 40 outputs are named
                    #: before they exist and none can be quietly dropped.
                    "identifier": identifier(args.arm, temperature, top_p, seed),
                }
                excerpt_records.append(record)
                (out_dir / f"{record['identifier']}.json").write_text(
                    json.dumps({"tokens": excerpt.tokens} | record, indent=1))
                print(f"    t={temperature} p={top_p} seed={seed}  "
                      f"{excerpt.outcome:17s} {excerpt.duration_sec:6.2f}s  "
                      f"illegal {excerpt.illegal_mass_mean:.4f}  "
                      f"{'scorable' if statistics else 'NOT SCORABLE: ' + note}",
                      flush=True)
            candidates.append(candidate)

    result = sel.select(candidates, reference)
    elapsed = time.time() - started

    acc.check("attempted_outcomes", observed=result.n_attempted_outcomes,
              intended=sel.EXPECTED_ATTEMPTED,
              population="10 candidates x 4 development seeds",
              strict=False)
    acc.check("candidates", observed=result.n_candidates,
              intended=len(sel.TEMPERATURES) * len(sel.TOP_P_VALUES),
              population="registered sampling grid", strict=False)
    acc.check("retained_records", observed=len(excerpt_records),
              intended=sel.EXPECTED_ATTEMPTED,
              population="one retained outcome per registered seed",
              strict=False)
    acc.note("eligible_candidates", result.n_eligible_candidates)
    acc.note("outcome_tally", result.outcome_tally)
    acc.note("selection_successful", result.successful)

    payload = {
        "schema": "snes_v2.sampling_selection.v1",
        "arm": args.arm,
        "model": meta,
        "procedure": "V2_GENERATION_PROCEDURE.md sections 1-7, with "
                     "Amendment A1 development-candidate eligibility",
        "no_audio_heard": True,
        "no_test_data_touched": True,
        "grid": {"temperatures": list(sel.TEMPERATURES),
                 "top_p": list(sel.TOP_P_VALUES),
                 "development_seeds": list(sel.DEVELOPMENT_SEEDS)},
        "denominators": {
            "attempted_outcomes": result.n_attempted_outcomes,
            "expected_attempted_outcomes": sel.EXPECTED_ATTEMPTED,
            "candidates": result.n_candidates,
            "eligible_candidates": result.n_eligible_candidates,
            "note": "eligible candidates and attempted outcomes are separate "
                    "populations and are never conflated",
        },
        "outcome_tally": result.outcome_tally,
        "reference": reference,
        "successful": result.successful,
        "selected": result.selected,
        "failure_reason": result.failure_reason,
        "candidates": result.rows,
        "excerpts": excerpt_records,
        "wall_clock_sec": elapsed,
    }
    (out_dir / f"sampling_selection_{args.arm}.json").write_text(
        json.dumps(payload, indent=1))
    acc.write(out_dir / f"sampling_selection_{args.arm}_accounting.json")

    print(f"\n  attempted outcomes   {result.n_attempted_outcomes} / "
          f"{sel.EXPECTED_ATTEMPTED}")
    print(f"  outcome tally        {result.outcome_tally}")
    print(f"  eligible candidates  {result.n_eligible_candidates} / "
          f"{result.n_candidates}")
    for row in sorted(result.rows, key=lambda r: (r["objective"] is None,
                                                  r["objective"] or 0.0)):
        if row["objective"] is None:
            print(f"    t={row['temperature']} p={row['top_p']}  INELIGIBLE  "
                  f"{row['ineligibility_reasons'][0]}")
        else:
            print(f"    t={row['temperature']} p={row['top_p']}  "
                  f"objective {row['objective']:.6f}  "
                  f"{row['aggregate']['attacks_per_sec']:7.3f} attacks/s  "
                  f"{row['aggregate']['mean_simultaneity']:.3f} simultaneity")
    if result.successful:
        print(f"\n  SELECTED  temperature {result.selected['temperature']}  "
              f"top-p {result.selected['top_p']}  "
              f"objective {result.selected['objective']:.6f}")
    else:
        print(f"\n  SELECTION UNSUCCESSFUL: {result.failure_reason}")
    print(f"  {elapsed:.1f}s; written to {out_dir}")

    acc.exit_if_failed()
    if not result.successful:
        sys.exit(3)


def identifier(arm: str, temperature: float, top_p: float, seed: int) -> str:
    return (f"{arm}_t{str(temperature).replace('.', '')}"
            f"_p{str(top_p).replace('.', '')}_s{seed}")


def measure(excerpt, vocab) -> tuple[dict | None, str]:
    """Frozen statistics for one retained outcome, or None with the reason.

    None is Amendment A1's mechanism: the metrics are undefined, so the
    outcome is not scorable and its candidate is ineligible. The outcome is
    still retained and reported.
    """
    if excerpt.outcome == "invalid":
        return None, f"grammar failure: {excerpt.invalid_reason}"
    try:
        parsed, _ = gx.parse(excerpt.tokens, vocab)
    except Exception as exc:                       # noqa: BLE001
        return None, f"undecodable: {type(exc).__name__}: {exc}"
    notes = gx.notes_from_groups(parsed)
    statistics = sx.excerpt_statistics(notes, grid_hz=GRID_HZ)
    if statistics is None:
        return None, (f"decoded to {len(notes)} notes; the frozen metrics are "
                      f"undefined without a positive span")
    return statistics, ""


if __name__ == "__main__":
    main()
