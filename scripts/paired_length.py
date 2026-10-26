"""Paired 8-bar vs 16-bar test: same prompts, same seeds, one token stream.

The question is whether letting the composer develop for sixteen bars produces
better music than stopping it at eight, on the pieces that already worked at
eight. Two listener comments from the 30-piece batch motivated it: "needs to
be longer to assess the entire loop" and "needs a bit more complexity", both
on pieces rated near the top.

**Both arms come from a single generation.** The naive design runs the model
twice per seed, once with an 8-bar stop and once with a 16-bar stop, and
relies on the RNG stream being identical so the shorter piece is a prefix of
the longer one. That holds here -- same prompt, same logits, same draws -- but
relying on it is unnecessary when the 16-bar stream *contains* the 8-bar one
by construction. So each seed is sampled once to sixteen bars and
`trim_to_bars` cuts the same stream at 8 and at 16. The 8-bar arm is then
provably the 16-bar arm's prefix rather than probably it, the compute halves,
and the only difference between the two pieces is where the composer was cut
off.

That also makes the 8-bar arm a regression check: for the seeds carried over
from `listen_batch2`, it must reproduce those pieces token for token. If it
does not, something downstream of the sampler has changed and the comparison
is invalid. `--verify-against` enforces this and fails closed.

Two things deliberately left alone:

**`repetition_penalty` stays at 1.12.** Over a 16-bar history the penalty
accumulates across twice as many tokens, which biases the model *away* from
restating earlier material -- arguably against the development we are looking
for. Lowering it for the long arm would change the prefix and destroy the
pairing, so it stays identical and the confound is reported instead.

**Nothing is selected on quality.** Seeds come from the previous batch in seed
order and every complete pair is rendered, so the listening set is not
conditioned on how good either arm sounds.
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
import random
from collections import Counter
from typing import Any

from rse.codec.benchmark import enforce_monophony, ioi_entropy
from rse.conditioning import melody_line, motif_note_participation
from rse.events.schema import Composition
from rse.hardware.constraints import ConstraintEngine
from rse.hardware.profile import load_profile
from rse.tokenizers.frozen import frozen_config
from rse.tokenizers.tokenizer import GridFit, RseTokenizer
from rse.training.evaluate import evaluate_composition
from rse.transfer.arranger import arrange_for_nes
from rse.transfer.ensemble import check_ensemble
from rse.transfer.fermata import RECONSTRUCTION_ID, Vocab, decode, trim_to_bars

#: One tracker row in seconds: FamiTracker's default speed 6 at 60 Hz.
ROW_SEC = 0.1

#: Pinned so the record is self-contained.
CHECKPOINT_REVISION = "18b647098f579ee41db67c0ad4c7f7f39ce0eac1"
CHECKPOINT_SHA256 = "4ca6a5db3a2d1510bad7acde4d313a9ad732fc91dcda4776f044c5b5bd8ea9b7"

def half_development(composition) -> dict:
    """Delegates to `rse.analysis.restatement`, which is also what the corpus
    control uses.

    This function previously derived bar boundaries as `end_tick / n_bars`.
    That is wrong whenever the last bar ends early: every boundary shifts and
    bar *i* gets compared against the wrong bar. It reported a median exact
    restatement of 0.115 where the grid-aligned measure gives 0.000. Keeping
    one implementation, shared with the corpus reference, is the only way the
    two sides stay comparable.
    """
    from rse.analysis.restatement import analyse_halves, ticks_per_bar

    report = analyse_halves(composition, n_bars=16, bar_ticks=ticks_per_bar())
    payload = report.to_dict()
    # Field names the original run wrote, kept so stored records stay readable.
    payload["restatement_mean"] = payload["exact_restatement_mean"]
    payload["restatement_exact_bars"] = payload["exact_restatement_bars"]
    return payload


def build_arm(
    ids: list[int],
    *,
    n_bars: int,
    vocab: Vocab,
    tokenizer: RseTokenizer,
    fit: GridFit,
    engine: ConstraintEngine,
    profile: Any,
    scenario: dict[str, Any],
) -> tuple[dict[str, Any], Composition | None]:
    """One arm of a pair: trim the shared stream, then the full NES pipeline.

    Identical to the batch-2 pipeline, including the constraint engine running
    LAST so the D4 round trip cannot reintroduce the polyphony it repairs.
    Only `n_bars` differs between the two arms.
    """
    trimmed = trim_to_bars(ids, vocab=vocab, n_bars=n_bars)
    record: dict[str, Any] = {
        "n_bars_requested": n_bars,
        "termination": trimmed.reason,
        "n_bars": trimmed.n_bars,
        "complete": bool(trimmed.complete and trimmed.n_bars == n_bars),
        "trim": trimmed.to_dict(),
        "n_trimmed_tokens": len(trimmed.ids),
    }
    if not record["complete"]:
        return record, None

    score, decode_report = decode(trimmed.ids, vocab=vocab)
    arranged, arrange_report = arrange_for_nes(
        score, row_sec=ROW_SEC, divisions=tokenizer.config.onset_divisions
    )
    arranged, _trunc = enforce_monophony(arranged)
    rounded = enforce_monophony(
        tokenizer.decode(tokenizer.encode(arranged, fit=fit).ids).composition
    )[0]
    final, repair = engine.apply(rounded)
    events = final.sounding_events()
    if not events:
        record["complete"] = False
        record["termination"] = "no_events_after_arrangement"
        return record, None

    ensemble = check_ensemble(
        scenario=scenario["name"],
        plan=scenario["plan"],
        score=score,
        arranged=final,
        n_tracks_merged=arrange_report.n_tracks_merged,
    )
    metrics = evaluate_composition(final, profile)
    _channel, melody = melody_line(final)
    end = final.end_tick or 1
    spt = (final.time.seconds(end) / end) if final.time.seconds(end) else 0.0
    record.update(
        {
            "decode": decode_report.to_dict(),
            "arrangement": arrange_report.to_dict(),
            "ensemble": ensemble.to_dict(),
            "hardware_modifications": len(repair) if isinstance(repair, list) else None,
            "hardware_violations": len(engine.validate(final)),
            "duration_sec": round(final.time.seconds(end), 3),
            "metrics": {
                "n_events": len(events),
                "notes_per_bar": round(metrics.notes_per_bar, 3),
                "pitch_class_entropy": round(metrics.pitch_class_entropy, 4),
                "stepwise_fraction": round(metrics.stepwise_fraction, 4),
                "ngram_repetition_rate": round(metrics.ngram_repetition_rate, 4),
                "motif_note_participation": round(motif_note_participation(melody), 4),
                "ioi_entropy": round(ioi_entropy([e.tick * spt for e in events]), 4),
                "channels": sorted({e.channel_id for e in events}),
                "n_percussion": sum(1 for e in events if e.channel_id == "NO"),
            },
        }
    )
    if n_bars == 16:
        record["development"] = half_development(final)
    return record, final


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("/tmp/fermata_dl"))
    parser.add_argument(
        "--seeds-from",
        type=Path,
        default=Path("listen_batch2_KEY.json"),
        help="reuse the previous batch's scenarios and seeds verbatim",
    )
    parser.add_argument(
        "--verify-against",
        type=Path,
        default=Path("listen_batch2_audit"),
        help="audit dir whose 8-bar token streams the short arm must reproduce",
    )
    parser.add_argument("--short-bars", type=int, default=8)
    parser.add_argument("--long-bars", type=int, default=16)
    # 8 bars needed 280-1900 tokens (median 780). Double the worst case, with
    # headroom, so TOKEN_BUDGET failures stay rare enough to be reportable
    # rather than the dominant outcome.
    parser.add_argument("--max-new-tokens", type=int, default=4600)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=24)
    parser.add_argument("--repetition-penalty", type=float, default=1.12)
    parser.add_argument(
        "--listen-scenarios",
        default="action,exploration,ominous,town,fast_rhythmic",
        help="scenarios rendered for listening; others are measured only",
    )
    parser.add_argument(
        "--listen-pairs-per-scenario",
        type=int,
        default=3,
        help="pairs per scenario in the listening set, taken in seed order",
    )
    parser.add_argument("--shuffle-seed", type=int, default=90210)
    parser.add_argument("--listen-dir", type=Path, default=Path("listen_paired"))
    parser.add_argument("--out", type=Path, default=Path("paired_length.json"))
    parser.add_argument("--limit", type=int, default=0, help="smoke test: first N seeds")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import torch
    from codec_comparison import render_window
    from fermata_generate import SCENARIOS, build_header
    from transformers import LlamaForCausalLM
    from transformers.generation.logits_process import LogitsProcessor

    by_name = {s["name"]: s for s in SCENARIOS}
    requested = [
        (row["scenario"], int(row["seed"]))
        for row in json.loads(args.seeds_from.read_text())
    ]
    requested.sort()
    if args.limit:
        requested = requested[: args.limit]
    print(f"{len(requested)} (scenario, seed) pairs from {args.seeds_from}", flush=True)

    vocab = Vocab.load(args.model / "tokenizer" / "vocab.json")
    model = LlamaForCausalLM.from_pretrained(args.model, dtype=torch.float32).eval()

    class SuppressBpe(LogitsProcessor):
        """The model trained on the atomic stream; BPE ids are unseen."""

        def __init__(self, ids: list[int]) -> None:
            self.ids = torch.tensor(ids, dtype=torch.long)

        def __call__(self, _input_ids, scores):
            scores[:, self.ids] = float("-inf")
            return scores

    class StopAfterBars(torch.nn.Module):
        """Halt as soon as the bar after the last requested one opens."""

        def __init__(self, bar_id: int, prompt_len: int, n_bars: int) -> None:
            super().__init__()
            self.bar_id = bar_id
            self.prompt_len = prompt_len
            self.limit = n_bars + 1

        def __call__(self, input_ids, _scores, **_kwargs) -> bool:
            generated = input_ids[0, self.prompt_len :]
            return bool((generated == self.bar_id).sum() >= self.limit)

    processors = [SuppressBpe(vocab.bpe_ids)]
    tokenizer = RseTokenizer(frozen_config())
    profile = load_profile("nes")
    engine = ConstraintEngine(profile)
    fit = GridFit(
        step_ticks=float(tokenizer.ticks_per_step(480)),
        phase_ticks=0.0,
        aligned=True,
        period_sec=ROW_SEC,
    )
    audit = args.listen_dir.parent / f"{args.listen_dir.name}_audit"
    audit.mkdir(parents=True, exist_ok=True)

    pairs: list[dict[str, Any]] = []
    compositions: dict[str, Composition] = {}
    verify_failures: list[dict[str, Any]] = []

    for scenario_name, seed in requested:
        scenario = by_name[scenario_name]
        header = build_header(vocab, scenario)
        prompt = torch.tensor([[vocab.id_of(t) for t in header]])
        torch.manual_seed(seed)
        with torch.no_grad():
            out = model.generate(
                prompt,
                do_sample=True,
                temperature=args.temperature,
                top_k=args.top_k,
                repetition_penalty=args.repetition_penalty,
                max_new_tokens=args.max_new_tokens,
                eos_token_id=vocab.id_of("<EOS>"),
                pad_token_id=vocab.id_of("<PAD>"),
                logits_processor=processors,
                stopping_criteria=[
                    StopAfterBars(vocab.id_of("<BAR>"), prompt.shape[1], args.long_bars)
                ],
                use_cache=True,
            )
        raw_ids = out[0].tolist()
        name = f"{scenario_name}_s{seed}"

        short, short_comp = build_arm(
            raw_ids,
            n_bars=args.short_bars,
            vocab=vocab,
            tokenizer=tokenizer,
            fit=fit,
            engine=engine,
            profile=profile,
            scenario=scenario,
        )
        long, long_comp = build_arm(
            raw_ids,
            n_bars=args.long_bars,
            vocab=vocab,
            tokenizer=tokenizer,
            fit=fit,
            engine=engine,
            profile=profile,
            scenario=scenario,
        )

        # The short arm must reproduce the previous batch token for token.
        # Anything else means the sampler or the trim changed underneath us.
        verified: bool | None = None
        previous = args.verify_against / f"{name}.tokens.json"
        if previous.exists():
            want = json.loads(previous.read_text())["trimmed_ids"]
            got = trim_to_bars(raw_ids, vocab=vocab, n_bars=args.short_bars).ids
            verified = want == got
            if not verified:
                verify_failures.append(
                    {"name": name, "n_previous": len(want), "n_now": len(got)}
                )

        record = {
            "name": name,
            "scenario": scenario_name,
            "seed": seed,
            "n_raw_tokens": len(raw_ids),
            "reproduces_batch2": verified,
            "short": short,
            "long": long,
            "usable": bool(short["complete"] and long["complete"]),
        }
        pairs.append(record)
        if short_comp is not None:
            compositions[f"{name}__8"] = short_comp
        if long_comp is not None:
            compositions[f"{name}__16"] = long_comp
        (audit / f"{name}.pair.json").write_text(
            json.dumps({**record, "raw_ids": raw_ids}, indent=2) + "\n"
        )
        print(
            f"  {name:<24} raw {len(raw_ids):>5} "
            f"8bar {short['termination']:<13} 16bar {long['termination']:<13} "
            f"{'pair ok' if record['usable'] else 'PAIR LOST'}"
            + ("" if verified is None else f" repro {'ok' if verified else 'FAIL'}"),
            flush=True,
        )

    if verify_failures:
        print(
            f"\nerror: {len(verify_failures)} short arms did not reproduce "
            f"{args.verify_against}: {[f['name'] for f in verify_failures]}",
            file=sys.stderr,
        )
        return 2

    # Listening set: complete pairs, scenarios chosen up front, seed order.
    # `sparse` is measured but not rendered -- 4 of its 9 draws already failed
    # to reach 8 bars and all 5 of its rated pieces landed in the bottom two
    # bands, so spending listening budget on it buys nothing the objective
    # completion rate does not already say.
    wanted_scenarios = [s for s in args.listen_scenarios.split(",") if s]
    chosen: list[tuple[str, str]] = []
    for scenario_name in wanted_scenarios:
        usable = [
            p for p in pairs if p["scenario"] == scenario_name and p["usable"]
        ][: args.listen_pairs_per_scenario]
        for pair in usable:
            chosen.append((pair["name"], "8"))
            chosen.append((pair["name"], "16"))

    args.listen_dir.mkdir(parents=True, exist_ok=True)
    slots = list(range(1, len(chosen) + 1))
    random.Random(args.shuffle_seed).shuffle(slots)
    key: list[dict[str, Any]] = []
    for slot, (name, arm) in zip(slots, chosen, strict=True):
        filename = f"take_{slot:02d}.wav"
        render_window(compositions[f"{name}__{arm}"], args.listen_dir / filename)
        key.append({"file": filename, "name": name, "arm_bars": int(arm)})
    key.sort(key=lambda row: row["file"])

    manifest = args.listen_dir / "RATINGS.md"
    manifest.write_text(
        "# Paired length test -- ratings\n\n"
        "Rate every take on the same 1-6 scale you used before, where **1 is\n"
        "the best and 6 is the worst**. Judge each take on its own; do not try\n"
        "to work out which takes belong together.\n\n"
        "One thing this test cannot hide: half the takes are twice as long as\n"
        "the other half, so the length of a file does tell you which arm it\n"
        "is in. What stays hidden is the scenario, the seed, and which long\n"
        "take extends which short one. Rate what you hear; a longer piece\n"
        "that goes nowhere should score worse than a short one that works.\n\n"
        "| file | rating | weakness / comment |\n|---|---|---|\n"
        + "".join(f"| {row['file']} |  |  |\n" for row in key)
        + "\nFilenames carry no information about scenario, seed or arm.\n"
    )
    key_path = args.out.parent / f"{args.listen_dir.name}_KEY.json"
    key_path.write_text(json.dumps(key, indent=2) + "\n")

    usable = [p for p in pairs if p["usable"]]
    lost_reason = Counter(
        p["long"]["termination"] for p in pairs if not p["long"]["complete"]
    )
    payload = {
        "reconstruction": RECONSTRUCTION_ID,
        "checkpoint": {
            "model": "haster/Fermata-0.2B-v1",
            "revision": CHECKPOINT_REVISION,
            "weights_sha256": CHECKPOINT_SHA256,
        },
        "design": {
            "short_bars": args.short_bars,
            "long_bars": args.long_bars,
            "one_stream_per_seed": True,
            "seeds_from": str(args.seeds_from),
            "verified_against": str(args.verify_against),
            "temperature": args.temperature,
            "top_k": args.top_k,
            "repetition_penalty": args.repetition_penalty,
            "max_new_tokens": args.max_new_tokens,
            "row_sec": ROW_SEC,
            "representation": "rse-repr/1.0.0-d4k4-tempo1.01",
            "listen_scenarios": wanted_scenarios,
            "listen_pairs_per_scenario": args.listen_pairs_per_scenario,
        },
        "n_pairs_attempted": len(pairs),
        "n_pairs_usable": len(usable),
        "long_arm_failures_by_reason": dict(lost_reason),
        "n_reproduced_batch2": sum(1 for p in pairs if p["reproduces_batch2"]),
        "pairs": pairs,
        "listening_set": key,
    }
    args.out.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"\n{len(usable)}/{len(pairs)} pairs complete at both lengths")
    if lost_reason:
        print(f"16-bar arm failures: {dict(lost_reason)}")
    print(f"short arms reproducing batch 2: {payload['n_reproduced_batch2']}")
    print(f"rendered {len(key)} takes to {args.listen_dir}")
    print(f"key in {key_path} -- do not open it until the ratings are recorded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
