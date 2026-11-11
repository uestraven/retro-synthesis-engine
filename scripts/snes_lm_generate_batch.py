#!/usr/bin/env python3
"""The registered Level 2 generation batches: 24 x 25 s per arm.

Seeds 0..23, disjoint from the development seeds (1000..1003) used for
sampling selection. Raw token sequences are preserved before any rendering or
voice stealing, so the four hardware-consistency measurements are taken on
what the generators actually produced.

Excerpts are taken AS GENERATED. None is selected for sounding better, and
none is regenerated because its statistics were unflattering.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rse.snes_lm import baselines as bl  # noqa: E402
from rse.snes_lm import generate as gen  # noqa: E402
from rse.snes_lm import hardware_checks as hc  # noqa: E402
from rse.snes_lm import representation as rep  # noqa: E402
from rse.snes_lm.release import Release  # noqa: E402
from rse.snes_lm.train import Config, Decoder  # noqa: E402

SEEDS = tuple(range(24))
SECONDS = 25.0


def measure(excerpts: list[gen.Excerpt]) -> dict:
    """The four registered measurements, identically defined for both arms."""
    valid = [e for e in excerpts if e.valid]
    pairs = overlapping = clamped = notes = 0
    worst = 0.0
    peak_before = peak_after = 0
    for excerpt in valid:
        report = hc.same_voice_overlap(excerpt.events)
        pairs += report.n_pairs
        overlapping += report.n_overlapping
        worst = max(worst, report.max_overlap_sec)
        peak_before = max(peak_before, hc.max_simultaneous_voices(excerpt.events))
        stolen, n = hc.apply_voice_stealing(excerpt.events)
        clamped += n
        notes += len(excerpt.events)
        peak_after = max(peak_after, hc.max_simultaneous_voices(stolen))
    stats = gen.statistics([e.events for e in valid])
    return {
        "n_excerpts": len(excerpts),
        "n_valid": len(valid),
        "validity_rate": len(valid) / len(excerpts) if excerpts else 0.0,
        "invalid_reasons": [e.invalid_reason for e in excerpts if not e.valid],
        "raw_same_voice_overlap_rate": overlapping / pairs if pairs else 0.0,
        "n_overlapping_pairs": overlapping,
        "n_same_voice_pairs": pairs,
        "max_overlap_sec": worst,
        "voice_stealing_rate": clamped / notes if notes else 0.0,
        "n_clamped": clamped,
        "n_notes": notes,
        "peak_concurrent_before_stealing": peak_before,
        "peak_voices_after_stealing": peak_after,
        "attacks_per_second": stats.attacks_per_second,
        "mean_simultaneous_voices": stats.mean_simultaneous_voices,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("tmp/best.pt"))
    parser.add_argument("--vocab", default="/tmp/snes_vocab.json")
    parser.add_argument("--release", default="~/snesmdb-releases/snesmdb-50game-rc1")
    parser.add_argument("--selection", default="/tmp/snes_sampling_selection.json")
    parser.add_argument("--out", default="/tmp/snes_level2_batch.json")
    parser.add_argument("--raw-dir", default="/tmp/snes_level2_raw")
    parser.add_argument("--arm", choices=("model", "baseline", "both"), default="both")
    args = parser.parse_args()

    vocab = rep.read_vocab(args.vocab)
    selection = json.loads(Path(args.selection).read_text())["selected"]
    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    arms: dict[str, list[gen.Excerpt]] = {}

    if args.arm in ("baseline", "both"):
        release = Release(args.release)
        print("  fitting the frozen comparator...", flush=True)
        train = [rep.encode(release.events(t), vocab).tokens
                 for t in release.tracks(split="train")]
        comparator = bl.SlotNGram(vocab, order=10, discount=64.0,
                                  use_event_prefix=False).fit(train)
        choice = selection["baseline"]
        print(f"  baseline T={choice['temperature']} top_p={choice['top_p']}", flush=True)
        out = []
        t0 = time.time()
        for seed in SEEDS:
            config = gen.SamplingConfig(temperature=choice["temperature"],
                                        top_p=choice["top_p"], seed=seed)
            out.append(gen.generate_baseline(comparator, vocab, config,
                                            seconds=SECONDS))
        print(f"    {time.time()-t0:.0f}s", flush=True)
        arms["baseline"] = out

    if args.arm in ("model", "both"):
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        cfg = Config(**{**state["config"], "device": "cpu"})
        model = Decoder(vocab.size, cfg)
        model.load_state_dict(state["model"])
        model.eval()
        choice = selection["model"]
        print(f"  model T={choice['temperature']} top_p={choice['top_p']}", flush=True)
        out = []
        t0 = time.time()
        for seed in SEEDS:
            config = gen.SamplingConfig(temperature=choice["temperature"],
                                        top_p=choice["top_p"], seed=seed)
            excerpt = gen.generate_model(model, vocab, config, seconds=SECONDS)
            out.append(excerpt)
            print(f"    seed {seed:2}: {len(excerpt.tokens):5} tokens, "
                  f"{len(excerpt.events):4} events, valid={excerpt.valid}",
                  flush=True)
        print(f"    {time.time()-t0:.0f}s", flush=True)
        arms["model"] = out

    report = {
        "schema": "snes_lm.level2_batch.v1",
        "seeds": list(SEEDS),
        "seconds": SECONDS,
        "selection": selection,
        "control_tokenized_real_overlap": 0.06638,
        "measurements": {arm: measure(ex) for arm, ex in arms.items()},
    }
    for arm, excerpts in arms.items():
        (raw_dir / f"{arm}_tokens.json").write_text(json.dumps(
            [{"seed": e.seed, "tokens": e.tokens, "valid": e.valid,
              "invalid_reason": e.invalid_reason} for e in excerpts]))
    Path(args.out).write_text(json.dumps(report, indent=1))

    print()
    for arm, m in report["measurements"].items():
        print(f"  === {arm} ===")
        print(f"    valid                        {m['n_valid']}/{m['n_excerpts']} "
              f"({100*m['validity_rate']:.1f}%)")
        print(f"    raw same-voice overlap       {100*m['raw_same_voice_overlap_rate']:.2f}%"
              f"  ({m['n_overlapping_pairs']}/{m['n_same_voice_pairs']}), "
              f"worst {m['max_overlap_sec']:.2f} s")
        print(f"    voice-stealing rate          {100*m['voice_stealing_rate']:.1f}% "
              f"({m['n_clamped']}/{m['n_notes']} notes)")
        print(f"    peak concurrent before       {m['peak_concurrent_before_stealing']}")
        print(f"    peak voices after stealing   {m['peak_voices_after_stealing']}")
        print(f"    attacks/s                    {m['attacks_per_second']:.2f}")
    print(f"\n  control (real tracks, same tokenizer): 6.638% overlap")


if __name__ == "__main__":
    main()
