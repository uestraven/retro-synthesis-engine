"""Generate a large Fermata batch, rank by corpus distance, render both ends.

The experiment: does ranking by closeness to the NES-MDB distribution predict
which pieces a listener prefers? Two preferred pieces out of twelve suggested
it does -- they sat near the corpus median on density, IOI entropy and
stepwise motion while the rest overshot, and they scored far higher on motif
participation. That is a hypothesis built on n=2.

So this renders the **top and the bottom** of the ranking under neutral names
(`pick_01`, `pick_02`, ...), with the mapping written to a separate file. If
the ranking works, the listener's preferences cluster at one end. If it does
not, the ranking is wrong and should be discarded rather than tuned until it
agrees.

Nothing here maximizes a metric. Every rhythmic and melodic measure is a
two-sided distance from the corpus, in units of the corpus's own IQR.
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
import statistics as st
from typing import Any

from rse.codec.benchmark import enforce_monophony, ioi_entropy
from rse.conditioning import melody_line, motif_note_participation
from rse.hardware.constraints import ConstraintEngine
from rse.hardware.profile import load_profile
from rse.tokenizers.frozen import frozen_config
from rse.tokenizers.tokenizer import GridFit, RseTokenizer
from rse.training.evaluate import evaluate_composition
from rse.transfer.arranger import arrange_for_nes
from rse.transfer.fermata import RECONSTRUCTION_ID, Vocab, decode
from rse.transfer.select import WEIGHTS, Reference, rank

#: One tracker row in seconds: FamiTracker's default speed 6 at 60 Hz.
ROW_SEC = 0.1

#: Scenarios, weighted by the evidence. `ominous` and `town` produced the two
#: pieces a listener picked, so they get more draws -- but not all of them,
#: because two preferred pieces is not grounds for abandoning the others.
SCENARIO_WEIGHTS = {
    "ominous": 3,
    "town": 3,
    "exploration": 2,
    "action": 2,
    "fast_rhythmic": 1,
    "sparse": 1,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("/tmp/fermata_dl"))
    parser.add_argument("--reference", type=Path, default=Path("rhythm_reference.json"))
    parser.add_argument("--n", type=int, default=60, help="pieces to generate")
    parser.add_argument("--max-new-tokens", type=int, default=1400)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=24)
    parser.add_argument("--repetition-penalty", type=float, default=1.12)
    parser.add_argument("--seed-base", type=int, default=1000)
    parser.add_argument("--render-top", type=int, default=5)
    parser.add_argument("--render-bottom", type=int, default=5)
    parser.add_argument("--render-dir", type=Path, default=Path("listen_blind"))
    parser.add_argument("--out", type=Path, default=Path("fermata_selection.json"))
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import torch
    from codec_comparison import render_window
    from fermata_generate import SCENARIOS, build_header
    from transformers import LlamaForCausalLM
    from transformers.generation.logits_process import LogitsProcessor

    reference = Reference.from_rhythm_reference(json.loads(args.reference.read_text()))
    print(f"reference metrics: {sorted(reference.median)}", flush=True)

    vocab = Vocab.load(args.model / "tokenizer" / "vocab.json")
    model = LlamaForCausalLM.from_pretrained(args.model, dtype=torch.float32).eval()

    class SuppressBpe(LogitsProcessor):
        def __init__(self, ids: list[int]) -> None:
            self.ids = torch.tensor(ids, dtype=torch.long)

        def __call__(self, _input_ids, scores):
            scores[:, self.ids] = float("-inf")
            return scores

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

    by_name = {s["name"]: s for s in SCENARIOS}
    pool: list[str] = []
    for name, weight in SCENARIO_WEIGHTS.items():
        if name in by_name:
            pool.extend([name] * weight)
    chooser = random.Random(args.seed_base)

    pieces: list[dict[str, Any]] = []
    compositions: dict[str, Any] = {}

    for index in range(args.n):
        scenario = by_name[chooser.choice(pool)]
        seed = args.seed_base + index
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
                use_cache=True,
            )
        ids = out[0].tolist()
        score, _report = decode(ids, vocab=vocab)
        arranged, arrange_report = arrange_for_nes(
            score, row_sec=ROW_SEC, divisions=tokenizer.config.onset_divisions
        )
        arranged, _trunc = enforce_monophony(arranged)
        playable, _repair = engine.apply(arranged)
        final, _t = enforce_monophony(
            tokenizer.decode(tokenizer.encode(playable, fit=fit).ids).composition
        )
        events = final.sounding_events()
        if not events:
            continue
        metrics = evaluate_composition(final, profile)
        _channel, melody = melody_line(final)
        end = final.end_tick or 1
        spt = (final.time.seconds(end) / end) if final.time.seconds(end) else 0.0
        name = f"{scenario['name']}_s{seed}"
        pieces.append(
            {
                "name": name,
                "scenario": scenario["name"],
                "seed": seed,
                "n_events": len(events),
                "n_channels_used": len({e.channel_id for e in events}),
                "n_percussion": sum(1 for e in events if e.channel_id == "NO"),
                "n_bars": metrics.n_bars,
                "notes_per_bar": round(metrics.notes_per_bar, 3),
                "ioi_entropy": round(ioi_entropy([e.tick * spt for e in events]), 4),
                "pitch_class_entropy": round(metrics.pitch_class_entropy, 4),
                "stepwise_fraction": round(metrics.stepwise_fraction, 4),
                "ngram_repetition_rate": round(metrics.ngram_repetition_rate, 4),
                "motif_note_participation": round(motif_note_participation(melody), 4),
                "retention": arrange_report.retention,
            }
        )
        compositions[name] = final
        if (index + 1) % 10 == 0:
            print(f"  generated {index + 1}/{args.n}", flush=True)

    if not pieces:
        print("error: nothing generated", file=sys.stderr)
        return 2

    ranked = rank(pieces, reference)
    eligible = [s for s in ranked if s.eligible]
    print(f"\n{len(pieces)} generated, {len(eligible)} eligible")

    head = f"{'rank':>5}{'score':>8}  {'piece':<24}{'n/bar':>7}{'ioiH':>7}{'motif':>7}{'chan':>6}"
    print(f"\nclosest to the corpus:\n{head}")
    print("-" * len(head))
    for position, scored in enumerate(eligible[: args.render_top], 1):
        m = scored.metrics
        print(
            f"{position:>5}{scored.score:>8.3f}  {scored.name:<24}"
            f"{m['notes_per_bar']:>7.1f}{m['ioi_entropy']:>7.2f}"
            f"{m['motif_note_participation']:>7.2f}{m['n_channels_used']:>6}"
        )
    print(f"\nfurthest from the corpus:\n{head}")
    print("-" * len(head))
    for position, scored in enumerate(eligible[-args.render_bottom :], 1):
        m = scored.metrics
        print(
            f"{position:>5}{scored.score:>8.3f}  {scored.name:<24}"
            f"{m['notes_per_bar']:>7.1f}{m['ioi_entropy']:>7.2f}"
            f"{m['motif_note_participation']:>7.2f}{m['n_channels_used']:>6}"
        )

    # Render both ends under neutral names, shuffled, so the listening test is
    # blind. A ranking validated against labelled audio validates nothing.
    chosen = eligible[: args.render_top] + eligible[-args.render_bottom :]
    labels = list(enumerate(chosen, 1))
    random.Random(args.seed_base + 7).shuffle(labels)
    key: list[dict[str, Any]] = []
    for slot, (_position, scored) in enumerate(labels, 1):
        out_name = f"pick_{slot:02d}.wav"
        render_window(compositions[scored.name], args.render_dir / out_name)
        is_top = scored in eligible[: args.render_top]
        key.append(
            {
                "file": out_name,
                "piece": scored.name,
                "end": "top" if is_top else "bottom",
                "score": round(scored.score, 4),
            }
        )

    args.out.write_text(
        json.dumps(
            {
                "reconstruction": RECONSTRUCTION_ID,
                "n_generated": len(pieces),
                "n_eligible": len(eligible),
                "weights": dict(WEIGHTS),
                "sampling": {
                    "temperature": args.temperature,
                    "top_k": args.top_k,
                    "repetition_penalty": args.repetition_penalty,
                    "max_new_tokens": args.max_new_tokens,
                },
                "ranked": [s.to_dict() for s in ranked],
                "blind_key": key,
            },
            indent=2,
        )
        + "\n"
    )
    key_path = args.render_dir / "BLIND_KEY.json"
    key_path.write_text(json.dumps(key, indent=2) + "\n")
    print(f"\nrendered {len(key)} files to {args.render_dir} under neutral names")
    print(f"the mapping is in {key_path} -- do not read it before listening")
    print(f"wrote {args.out}")
    print(
        f"\nmedian score, top {args.render_top}: "
        f"{st.median([s.score for s in eligible[: args.render_top]]):.3f}; "
        f"bottom {args.render_bottom}: "
        f"{st.median([s.score for s in eligible[-args.render_bottom :]]):.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
