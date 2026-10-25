"""Generate with the unmodified Fermata checkpoint and arrange for the NES.

Reproducible: every scenario, sampling parameter and seed is recorded in the
output JSON alongside the checkpoint revision and weight digest.

Two things this script is careful about.

**It uses a KV cache.** A first attempt re-ran the full forward pass per token,
which is quadratic and did not finish 900 tokens on CPU inside twenty minutes.
`generate()` handles the cache, so the cost is linear.

**It suppresses the dead BPE tokens.** 7,348 of 9,507 vocabulary entries are
`<BPE_*>`, and the model card records that the merges were never applied to
the training shards, so the model never saw them. Sampling one would be
sampling an id outside the trained distribution.

Early termination and invalid grammar are recorded, never repaired: the point
is to find out what the checkpoint actually does, and a script that forced
every output to eight bars would hide exactly the failure mode the native
model showed.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Importable when run as a file from a checkout where the package is not
# installed, which is the normal state of a fresh GPU host: `python
# scripts/x.py` puts `scripts/` on sys.path, not the repository root.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rse.transfer.fermata import RECONSTRUCTION_ID, Vocab, decode

#: Scenarios phrased in the controls Fermata actually has: a genre label, an
#: ensemble label, a tempo, a key and a track plan. These are not text
#: prompts -- the model has no free-text conditioning.
SCENARIOS: tuple[dict[str, Any], ...] = (
    {
        "name": "action",
        "genre": "soundtrack",
        "ensemble": "band",
        "tempo": 168,
        "key": "0:minor",
        "plan": ["<TRACK_80>", "<TRACK_81>", "<TRACK_38>", "<DRUM_TRACK>"],
    },
    {
        "name": "exploration",
        "genre": "soundtrack",
        "ensemble": "chamber",
        "tempo": 120,
        "key": "0:major",
        "plan": ["<TRACK_80>", "<TRACK_81>", "<TRACK_38>", "<DRUM_TRACK>"],
    },
    {
        "name": "ominous",
        "genre": "soundtrack",
        "ensemble": "orchestral",
        "tempo": 92,
        "key": "-3:minor",
        "plan": ["<TRACK_80>", "<TRACK_81>", "<TRACK_38>", "<DRUM_TRACK>"],
    },
    {
        "name": "town",
        "genre": "folk",
        "ensemble": "band",
        "tempo": 132,
        "key": "2:major",
        "plan": ["<TRACK_80>", "<TRACK_81>", "<TRACK_38>", "<DRUM_TRACK>"],
    },
    {
        "name": "fast_rhythmic",
        "genre": "electronic",
        "ensemble": "band",
        "tempo": 180,
        "key": "0:minor",
        "plan": ["<TRACK_80>", "<TRACK_81>", "<TRACK_38>", "<DRUM_TRACK>"],
    },
    {
        "name": "sparse",
        "genre": "classical",
        "ensemble": "solo",
        "tempo": 76,
        "key": "0:major",
        "plan": ["<TRACK_80>", "<TRACK_38>"],
    },
)


@dataclass
class Sampling:
    """The card's recommended defaults."""

    temperature: float = 0.7
    top_k: int = 24
    repetition_penalty: float = 1.12
    max_new_tokens: int = 1200

    def to_dict(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "max_new_tokens": self.max_new_tokens,
        }


def build_header(vocab: Vocab, scenario: dict[str, Any]) -> list[str]:
    tempo = min(vocab.tempos, key=lambda t: abs(t - scenario["tempo"]))
    key = f"<KEY_{scenario['key']}>"
    header = [
        "<BOS>",
        f"<GENRE_{scenario['genre']}>",
        f"<ENSEMBLE_{scenario['ensemble']}>",
        f"<TEMPO_{tempo}>",
        key if vocab.has(key) else "<KEY_0:none>",
        "<METER_4/4>",
        "<PLAN>",
        *scenario["plan"],
        "<PLAN_END>",
    ]
    missing = [t for t in header if not vocab.has(t)]
    if missing:
        raise KeyError(f"scenario {scenario['name']}: unknown tokens {missing}")
    return header


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("/tmp/fermata_dl"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--scenarios", nargs="+", default=None)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=24)
    parser.add_argument("--repetition-penalty", type=float, default=1.12)
    parser.add_argument("--max-new-tokens", type=int, default=1200)
    parser.add_argument("--out", type=Path, default=Path("fermata_generations"))
    args = parser.parse_args()

    import torch
    from transformers import LlamaForCausalLM
    from transformers.generation.logits_process import LogitsProcessor

    vocab = Vocab.load(args.model / "tokenizer" / "vocab.json")
    model = LlamaForCausalLM.from_pretrained(args.model, dtype=torch.float32).eval()

    class SuppressBpe(LogitsProcessor):
        """The model was trained on the atomic stream; BPE ids are unseen."""

        def __init__(self, ids: list[int]) -> None:
            self.ids = torch.tensor(ids, dtype=torch.long)

        def __call__(self, _input_ids, scores):
            scores[:, self.ids] = float("-inf")
            return scores

    processors = [SuppressBpe(vocab.bpe_ids)]
    sampling = Sampling(
        temperature=args.temperature,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        max_new_tokens=args.max_new_tokens,
    )

    wanted = set(args.scenarios) if args.scenarios else None
    scenarios = [s for s in SCENARIOS if wanted is None or s["name"] in wanted]
    args.out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for scenario in scenarios:
        header = build_header(vocab, scenario)
        prompt = torch.tensor([[vocab.id_of(t) for t in header]])
        for seed in args.seeds:
            torch.manual_seed(seed)
            started = time.time()
            with torch.no_grad():
                out = model.generate(
                    prompt,
                    do_sample=True,
                    temperature=sampling.temperature,
                    top_k=sampling.top_k,
                    repetition_penalty=sampling.repetition_penalty,
                    max_new_tokens=sampling.max_new_tokens,
                    eos_token_id=vocab.id_of("<EOS>"),
                    pad_token_id=vocab.id_of("<PAD>"),
                    logits_processor=processors,
                    use_cache=True,
                )
            elapsed = time.time() - started
            ids = out[0].tolist()
            score, report = decode(ids, vocab=vocab)

            stem = f"{scenario['name']}_s{seed}"
            (args.out / f"{stem}.tokens.json").write_text(
                json.dumps(
                    {
                        "reconstruction": RECONSTRUCTION_ID,
                        "scenario": scenario,
                        "seed": seed,
                        "sampling": sampling.to_dict(),
                        "header": header,
                        "ids": ids,
                        "tokens": [vocab.id_to_token.get(i, "<UNK>") for i in ids],
                    },
                    indent=2,
                )
                + "\n"
            )
            (args.out / f"{stem}.score.json").write_text(
                json.dumps({"score": score.to_dict(), "decode": report.to_dict()}, indent=2) + "\n"
            )

            notes = sum(len(t.notes) for t in score.tracks)
            perc = sum(len(t.percussion) for t in score.tracks)
            pitches = {n.pitch for t in score.tracks for n in t.notes}
            row = {
                "scenario": scenario["name"],
                "seed": seed,
                "n_tokens": len(ids),
                "hit_eos": report.hit_eos,
                "n_bars": report.n_bars,
                "n_tokens_skipped": report.n_skipped,
                "n_tracks": len(score.tracks),
                "n_notes": notes,
                "n_percussion": perc,
                "n_distinct_pitches": len(pitches),
                "tempo_bpm": score.tempo_bpm,
                "seconds_to_generate": round(elapsed, 1),
                "tokens_per_second": round(len(ids) / elapsed, 2) if elapsed else 0.0,
            }
            rows.append(row)
            print(
                f"{row['scenario']:<14} seed {seed}  {row['n_tokens']:>5} tok  "
                f"bars {row['n_bars']:>3}  notes {notes:>5}  perc {perc:>4}  "
                f"pitches {len(pitches):>3}  eos {report.hit_eos}  "
                f"{row['tokens_per_second']:.1f} tok/s",
                flush=True,
            )

    summary = {
        "reconstruction": RECONSTRUCTION_ID,
        "model": str(args.model),
        "sampling": sampling.to_dict(),
        "seeds": args.seeds,
        "generations": rows,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nwrote {args.out}/summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
