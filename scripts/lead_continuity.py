"""Where does the melody get lost -- in Fermata, or in RSE's arranger?

The 16-bar analysis measured second-half lead activity at 78.8% of the first
half against roughly 100% in the corpus. But that figure came from the
completed pipeline, so it cannot say whether Fermata wrote a weak lead or the
arranger weakened a good one. Those need opposite fixes, so the ambiguity is
resolved before anything is changed.

Three measurement points per piece, across the two boundaries that matter:

1. **Fermata's own score**, before any NES conversion. The intended lead is
   the track declared `<TRACK_80>` -- GM program 80, Lead 1 (square) -- which
   is plan index 0 in every scenario. Taking the plan's word for it rather
   than guessing is the point: `rse.transfer.arranger.assign_roles` chooses
   the lead *behaviourally* (lowest register becomes bass, then the most
   active of the rest becomes melody), and the two can disagree.
2. **The role assignment**, where that disagreement becomes visible. If the
   arranger designates some other track as melody, Fermata's intended tune is
   demoted to accompaniment and an accompaniment figure is promoted. Since
   `_track_movement` is mean absolute interval, a fast arpeggio outscores a
   stepwise melody -- and "chaotic arpeggios" is exactly what the listening
   test reported.
3. **The final NES composition**, where surviving note counts per channel say
   how much of the intended lead actually reached the hardware.

Nothing is regenerated with new randomness: each piece is reproduced from its
recorded seed and scenario, and the reproduction is checked against the event
count the comparison recorded. A mismatch means the pipeline moved and the
diagnosis would be describing different music.

Eight-bar and sixteen-bar results are kept in separate files on purpose. The
78.8% figure is a 16-bar measurement and this batch is 8 bars; pooling them
would hide whichever one carries the effect.
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

from rse.analysis.restatement import NGRAM, contour, ngrams, recurrence
from rse.codec.benchmark import enforce_monophony
from rse.hardware.constraints import ConstraintEngine
from rse.hardware.profile import load_profile
from rse.tokenizers.frozen import frozen_config
from rse.tokenizers.tokenizer import GridFit, RseTokenizer
from rse.transfer.arranger import (
    arrange_for_nes,
    assign_roles,
)
from rse.transfer.fermata import Vocab, decode, trim_to_bars
from rse.transfer.score import MELODY, PERCUSSION, Score

ROW_SEC = 0.1

#: GM program of the track every scenario declares first: Lead 1 (square).
#: This is the composer's stated intent for which voice carries the tune.
INTENDED_LEAD_PROGRAM = 80


def intended_lead_index(score: Score) -> int | None:
    """Plan index of the track Fermata was told to use as the lead.

    Falls back to the first pitched track only when no program matches, which
    would mean the header was not honoured at all -- worth knowing, so it is
    reported rather than silently patched.
    """
    for index, track in enumerate(score.tracks):
        if track.program == INTENDED_LEAD_PROGRAM and not track.percussion:
            return index
    for index, track in enumerate(score.tracks):
        if track.notes and not track.percussion:
            return index
    return None


def per_bar_counts(
    ticks: list[int], *, n_bars: int, bar_ticks: float
) -> list[int]:
    counts = [0] * n_bars
    for tick in ticks:
        bar = int(tick // bar_ticks)
        if 0 <= bar < n_bars:
            counts[bar] += 1
    return counts


def longest_run(values: list[int], *, equal_to: int = 0) -> int:
    """Longest consecutive stretch of bars holding `equal_to` notes."""
    best = run = 0
    for value in values:
        run = run + 1 if value == equal_to else 0
        best = max(best, run)
    return best


def sounding_fraction(
    spans: list[tuple[int, int]], *, total_ticks: float
) -> float:
    """Union of note durations over the span length, so overlaps don't
    double-count a voice as more present than it is."""
    if not spans or total_ticks <= 0:
        return 0.0
    merged: list[list[int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return min(1.0, sum(e - s for s, e in merged) / total_ticks)


def halves(counts: list[int]) -> tuple[int, int, float | None]:
    half = len(counts) // 2
    first, second = sum(counts[:half]), sum(counts[half:])
    return first, second, (second / first) if first else None


def analyse_score_side(
    score: Score, *, n_bars: int, ngram: int
) -> dict[str, Any]:
    """Fermata's intended lead, measured before any NES conversion."""
    bar_ticks = score.ticks_per_quarter * score.beats_per_bar
    total = bar_ticks * n_bars
    index = intended_lead_index(score)
    roles = assign_roles(score)
    arranger_lead = next(
        (i for i, role in roles.items() if role == MELODY), None
    )

    payload: dict[str, Any] = {
        "ticks_per_quarter": score.ticks_per_quarter,
        "beats_per_bar": score.beats_per_bar,
        "n_tracks": len(score.tracks),
        "intended_lead_index": index,
        "intended_lead_program": (
            score.tracks[index].program if index is not None else None
        ),
        "arranger_lead_index": arranger_lead,
        "arranger_agrees_with_plan": (
            index is not None and arranger_lead == index
        ),
        "role_assignment": {str(k): v for k, v in sorted(roles.items())},
        "tracks": [
            {
                "index": i,
                "program": t.program,
                "role_declared": t.role,
                "percussion": bool(t.percussion),
                "n_notes": len(t.notes),
                "n_percussion": len(t.percussion),
                "mean_pitch": round(
                    st.fmean([n.pitch for n in t.notes]), 2
                )
                if t.notes
                else None,
                "assigned_role": roles.get(i),
            }
            for i, t in enumerate(score.tracks)
        ],
    }
    if index is None:
        return payload

    lead = score.tracks[index]
    counts = per_bar_counts(
        [n.tick for n in lead.notes], n_bars=n_bars, bar_ticks=bar_ticks
    )
    first, second, ratio = halves(counts)
    pitches = [n.pitch for n in sorted(lead.notes, key=lambda n: n.tick)]
    split = (n_bars // 2) * bar_ticks
    early = [n.pitch for n in sorted(lead.notes, key=lambda n: n.tick) if n.tick < split]
    late = [n.pitch for n in sorted(lead.notes, key=lambda n: n.tick) if n.tick >= split]
    payload.update(
        {
            "lead_notes_total": len(lead.notes),
            "lead_notes_per_bar": counts,
            "lead_silent_bars": sum(1 for c in counts if c == 0),
            "lead_longest_silent_bars": longest_run(counts),
            "lead_first_half": first,
            "lead_second_half": second,
            "lead_half_ratio": None if ratio is None else round(ratio, 4),
            "lead_sounding_fraction": round(
                sounding_fraction(
                    [(n.tick, n.tick + n.duration) for n in lead.notes],
                    total_ticks=total,
                ),
                4,
            ),
            "lead_contour_recurrence": recurrence(
                contour(early), contour(late), n=ngram
            ),
            "lead_distinct_contour_ngrams": len(ngrams(contour(pitches), ngram)),
        }
    )
    return payload


def analyse_nes_side(
    composition: Any, *, n_bars: int, ngram: int, bar_ticks: int
) -> dict[str, Any]:
    """What reached the hardware, per channel, after arrangement and repair."""
    from rse.events.schema import Note

    events = composition.sounding_events()
    per_channel: dict[str, dict[str, Any]] = {}
    total = bar_ticks * n_bars
    for channel in ("P1", "P2", "TR", "NO"):
        rows = [e for e in events if e.channel_id == channel]
        if not rows:
            continue
        counts = per_bar_counts(
            [e.tick for e in rows], n_bars=n_bars, bar_ticks=bar_ticks
        )
        first, second, ratio = halves(counts)
        pitched = sorted(
            (e for e in rows if isinstance(e, Note)), key=lambda e: e.tick
        )
        split = (n_bars // 2) * bar_ticks
        early = [e.pitch for e in pitched if e.tick < split]
        late = [e.pitch for e in pitched if e.tick >= split]
        per_channel[channel] = {
            "n_events": len(rows),
            "per_bar": counts,
            "silent_bars": sum(1 for c in counts if c == 0),
            "longest_silent_bars": longest_run(counts),
            "first_half": first,
            "second_half": second,
            "half_ratio": None if ratio is None else round(ratio, 4),
            "sounding_fraction": round(
                sounding_fraction(
                    [(e.tick, e.tick + getattr(e, "duration", 0)) for e in rows],
                    total_ticks=total,
                ),
                4,
            ),
            "contour_recurrence": recurrence(
                contour(early), contour(late), n=ngram
            ),
        }
    return per_channel


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("/tmp/fermata_dl"))
    parser.add_argument(
        "--comparison", type=Path, default=Path("architecture_comparison.json")
    )
    parser.add_argument("--bars", type=int, default=8)
    parser.add_argument("--ngram", type=int, default=NGRAM)
    parser.add_argument("--out", type=Path, default=Path("lead_continuity_8bar.json"))
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import torch
    from fermata_generate import SCENARIOS, build_header
    from transformers import LlamaForCausalLM
    from transformers.generation.logits_process import LogitsProcessor

    comparison = json.loads(args.comparison.read_text())
    design = comparison["design"]["fermata_sampling"]
    targets = [p for p in comparison["pieces"] if p["arm"] == "fermata"]
    print(f"{len(targets)} fermata pieces to diagnose at {args.bars} bars")

    by_name = {s["name"]: s for s in SCENARIOS}
    vocab = Vocab.load(args.model / "tokenizer" / "vocab.json")
    model = LlamaForCausalLM.from_pretrained(args.model, dtype=torch.float32).eval()

    class SuppressBpe(LogitsProcessor):
        def __init__(self, ids: list[int]) -> None:
            self.ids = torch.tensor(ids, dtype=torch.long)

        def __call__(self, _input_ids, scores):
            scores[:, self.ids] = float("-inf")
            return scores

    class StopAfterBars(torch.nn.Module):
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
    nes_bar_ticks = int(tokenizer.ticks_per_step(480)) * tokenizer.config.steps_per_bar

    rows: list[dict[str, Any]] = []
    mismatches: list[str] = []
    for target in targets:
        scenario = by_name[target["scenario"]]
        seed = target["seed"]
        header = build_header(vocab, scenario)
        prompt = torch.tensor([[vocab.id_of(t) for t in header]])
        torch.manual_seed(seed)
        with torch.no_grad():
            out = model.generate(
                prompt,
                do_sample=True,
                temperature=design["temperature"],
                top_k=design["top_k"],
                repetition_penalty=design["repetition_penalty"],
                max_new_tokens=design["max_new_tokens"],
                eos_token_id=vocab.id_of("<EOS>"),
                pad_token_id=vocab.id_of("<PAD>"),
                logits_processor=processors,
                stopping_criteria=[
                    StopAfterBars(vocab.id_of("<BAR>"), prompt.shape[1], args.bars)
                ],
                use_cache=True,
            )
        trimmed = trim_to_bars(out[0].tolist(), vocab=vocab, n_bars=args.bars)
        score, _decode_report = decode(trimmed.ids, vocab=vocab)

        before = analyse_score_side(score, n_bars=args.bars, ngram=args.ngram)

        arranged, arrange_report = arrange_for_nes(
            score, row_sec=ROW_SEC, divisions=tokenizer.config.onset_divisions
        )
        arranged, _trunc = enforce_monophony(arranged)
        rounded = enforce_monophony(
            tokenizer.decode(tokenizer.encode(arranged, fit=fit).ids).composition
        )[0]
        final, _repair = engine.apply(rounded)
        after = analyse_nes_side(
            final, n_bars=args.bars, ngram=args.ngram, bar_ticks=nes_bar_ticks
        )

        # Reproduction check: the diagnosis is only about these pieces if the
        # rebuild matches what the comparison measured.
        reproduced = len(final.sounding_events()) == target["metrics"]["n_events"]
        if not reproduced:
            mismatches.append(target["name"])

        lead_index = before.get("intended_lead_index")
        lead_channel = None
        if lead_index is not None:
            assignment = before["role_assignment"].get(str(lead_index))
            lead_channel = assignment
        row = {
            "name": target["name"],
            "scenario": target["scenario"],
            "seed": seed,
            "reproduced": reproduced,
            "n_events_recorded": target["metrics"]["n_events"],
            "n_events_rebuilt": len(final.sounding_events()),
            "intended_lead_role": lead_channel,
            "arrangement": arrange_report.to_dict(),
            "before": before,
            "after": after,
        }
        rows.append(row)
        ratio = before.get("lead_half_ratio")
        print(
            f"  {target['name']:<34} lead trk {lead_index} "
            f"({before.get('intended_lead_program')}) "
            f"notes {before.get('lead_notes_total')} "
            f"half-ratio {'n/a' if ratio is None else f'{ratio:.2f}'} "
            f"silent {before.get('lead_silent_bars')}/{args.bars} "
            f"plan-agrees {before.get('arranger_agrees_with_plan')} "
            f"{'' if reproduced else 'REPRO MISMATCH'}",
            flush=True,
        )

    payload = {
        "bars": args.bars,
        "ngram": args.ngram,
        "source": str(args.comparison),
        "intended_lead_program": INTENDED_LEAD_PROGRAM,
        "n_pieces": len(rows),
        "n_reproduced": sum(1 for r in rows if r["reproduced"]),
        "reproduction_mismatches": mismatches,
        "pieces": rows,
    }
    args.out.write_text(json.dumps(payload, indent=2) + "\n")

    agree = sum(1 for r in rows if r["before"].get("arranger_agrees_with_plan"))
    ratios = [
        r["before"]["lead_half_ratio"]
        for r in rows
        if r["before"].get("lead_half_ratio") is not None
    ]
    print(f"\nreproduced {payload['n_reproduced']}/{len(rows)}")
    if mismatches:
        print(f"MISMATCHES: {mismatches}")
    print(f"arranger lead == plan's intended lead: {agree}/{len(rows)}")
    if ratios:
        print(
            f"pre-arrangement lead half-ratio: median {st.median(ratios):.3f} "
            f"(corpus reference ~1.000), min {min(ratios):.3f}, max {max(ratios):.3f}"
        )
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
