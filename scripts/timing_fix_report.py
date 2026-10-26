"""Historical versus corrected arrangement timing, on the same twelve scores.

No new generation, no new model, no sampling change. Each of the twelve
Fermata pieces from the blind architecture comparison is reproduced from its
recorded seed, and the identical symbolic score is arranged twice.

**The historical branch runs the real pre-fix code**, extracted from commit
`5a6c456` with `git show` and loaded as a module. An earlier version of this
script reimplemented the two wrong lines by hand, which was not faithful: it
omitted arpeggio expansion, and the archived pieces carry 15 to 48
arpeggiated events each. A reconstruction that differs from the audio the
listener actually judged cannot answer whether the fix improves that audio.

Both historical defects are reproduced together, because either alone gives
the wrong answer:

1. the arranger scaling positions and durations by `tempo_bpm / 150`;
2. `GridFit(period_sec=0.1)`, which makes the D4 encode override the Tempo
   token with 150 BPM — `tokenizer.py` does this whenever the fit is aligned.

Fidelity is then *verified* rather than assumed: the historical render is
peak-normalised exactly as the comparison did and compared byte-for-byte with
the archived WAV. Rendering is deterministic, so equality is the right
standard; anything less is reported per piece.

The intended lead is followed by GM program (80, Lead 1 square) into whichever
channel each branch assigned it. Keying on P1 would have measured the wrong
voice on the six pieces where `assign_roles` demotes the declared lead to
accompaniment.

Archived audio in `listen_arch/` is never written to. Corrected renders go to
a separate directory.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import hashlib
import importlib.util
import json
import statistics as st
import subprocess
from typing import Any

from rse.analysis.restatement import contour, recurrence
from rse.codec.benchmark import enforce_monophony
from rse.events.schema import Composition, Note
from rse.hardware.constraints import ConstraintEngine
from rse.hardware.profile import load_profile
from rse.tokenizers.frozen import frozen_config
from rse.tokenizers.tokenizer import GridFit, RseTokenizer
from rse.transfer.arranger import arrange_for_nes, assign_roles, row_seconds
from rse.transfer.fermata import Vocab, decode, trim_to_bars
from rse.transfer.score import Score

#: The row length every historical caller passed, which is 150 BPM.
HISTORICAL_ROW_SEC = 0.1
TICKS_PER_BEAT = 480

#: Commit holding the arranger as it was when the archived audio was made.
HISTORICAL_COMMIT = "5a6c456"

#: GM program of the track every scenario declares first: Lead 1 (square).
INTENDED_LEAD_PROGRAM = 80

CHANNEL_OF_ROLE = {
    "melody": "P1",
    "accompaniment": "P2",
    "bass": "TR",
    "percussion": "NO",
}


def load_historical_arranger(commit: str, *, workdir: Path) -> Any:
    """Import `rse/transfer/arranger.py` as it was at `commit`."""
    workdir.mkdir(parents=True, exist_ok=True)
    source = workdir / "arranger_pre_fix.py"
    text = subprocess.run(
        ["git", "show", f"{commit}:rse/transfer/arranger.py"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if "scale = tick_sec / out_spt" not in text:
        raise SystemExit(
            f"{commit} does not hold the defective arranger; refusing to "
            "compare against a branch that is not the historical one"
        )
    source.write_text(text)
    spec = importlib.util.spec_from_file_location("rse_arranger_pre_fix", source)
    if spec is None or spec.loader is None:
        raise SystemExit(f"could not load {source}")
    module = importlib.util.module_from_spec(spec)
    # Register before executing: `@dataclass` resolves annotations through
    # `sys.modules[cls.__module__]`, so a module that is not yet registered
    # fails with AttributeError on the first dataclass in the file.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def intended_lead_channel(score: Score) -> str:
    """Channel the declared lead (program 80) was assigned, by role."""
    roles = assign_roles(score)
    index = next(
        (
            i
            for i, t in enumerate(score.tracks)
            if t.program == INTENDED_LEAD_PROGRAM and not t.percussion
        ),
        None,
    )
    if index is None:
        index = next((i for i, t in enumerate(score.tracks) if t.notes), None)
    role = roles.get(index) if index is not None else None
    return CHANNEL_OF_ROLE.get(role or "melody", "P1")


def finish(
    arranged: Composition,
    *,
    tokenizer: RseTokenizer,
    engine: ConstraintEngine,
    period_sec: float,
) -> Composition:
    """The shared tail: monophony, D4 round trip, hardware repair."""
    arranged, _trunc = enforce_monophony(arranged)
    fit = GridFit(
        step_ticks=float(tokenizer.ticks_per_step(TICKS_PER_BEAT)),
        phase_ticks=0.0,
        aligned=True,
        period_sec=period_sec,
    )
    rounded = enforce_monophony(
        tokenizer.decode(tokenizer.encode(arranged, fit=fit).ids).composition
    )[0]
    final, _repair = engine.apply(rounded)
    return final


def describe(
    composition: Composition,
    *,
    n_bars: int,
    lead_channel: str,
    engine: ConstraintEngine,
) -> dict[str, Any]:
    events = composition.sounding_events()
    bar_ticks = TICKS_PER_BEAT * 4
    end = composition.end_tick or 0
    per_bar = [0] * n_bars
    beyond = 0
    for event in events:
        bar = int(event.tick // bar_ticks)
        if 0 <= bar < n_bars:
            per_bar[bar] += 1
        else:
            beyond += 1
    lead = sorted(
        (e for e in events if e.channel_id == lead_channel and isinstance(e, Note)),
        key=lambda e: e.tick,
    )
    half = n_bars // 2
    split = half * bar_ticks
    early = [e.pitch for e in lead if e.tick < split]
    late = [e.pitch for e in lead if e.tick >= split]
    encoded_bpm = (
        60_000_000 / composition.tempo_map[0].us_per_beat
        if composition.tempo_map
        else None
    )
    return {
        "n_events": len(events),
        "encoded_bpm": round(encoded_bpm, 3) if encoded_bpm else None,
        "bars_spanned": round(end / bar_ticks, 3) if bar_ticks else 0.0,
        "duration_sec": round(composition.time.seconds(end), 3) if end else 0.0,
        "per_bar": per_bar,
        "events_beyond_window": beyond,
        "empty_bars_in_window": sum(1 for c in per_bar if c == 0),
        "lead_channel": lead_channel,
        "lead_notes": len(lead),
        "lead_first_half": len(early),
        "lead_second_half": len(late),
        "lead_half_ratio": round(len(late) / len(early), 4) if early else None,
        "lead_contour_recurrence": recurrence(contour(early), contour(late)),
        "lead_distinct_pitches": len({e.pitch for e in lead}),
        "hardware_violations": len(engine.validate(composition)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("/tmp/fermata_dl"))
    parser.add_argument(
        "--comparison", type=Path, default=Path("architecture_comparison.json")
    )
    parser.add_argument("--archived", type=Path, default=Path("listen_arch"))
    parser.add_argument(
        "--archived-key", type=Path, default=Path("listen_arch_KEY.json")
    )
    parser.add_argument("--historical-commit", default=HISTORICAL_COMMIT)
    parser.add_argument("--workdir", type=Path, default=Path("/tmp/rse_historical"))
    parser.add_argument("--bars", type=int, default=8)
    parser.add_argument(
        "--render-dir", type=Path, default=Path("listen_timing_pairs_audit")
    )
    parser.add_argument("--out", type=Path, default=Path("timing_fix_report.json"))
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import torch
    from architecture_comparison import normalise_peak
    from codec_comparison import render_window
    from fermata_generate import SCENARIOS, build_header
    from transformers import LlamaForCausalLM
    from transformers.generation.logits_process import LogitsProcessor

    historical = load_historical_arranger(
        args.historical_commit, workdir=args.workdir
    )
    print(f"historical arranger loaded from {args.historical_commit}")

    comparison = json.loads(args.comparison.read_text())
    sampling = comparison["design"]["fermata_sampling"]
    targets = [p for p in comparison["pieces"] if p["arm"] == "fermata"]
    archived_file = {
        row["name"]: row["file"]
        for row in json.loads(args.archived_key.read_text())
        if row["arm"] == "fermata"
    }
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
    audit = args.render_dir
    audit.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for target in targets:
        scenario = by_name[target["scenario"]]
        header = build_header(vocab, scenario)
        prompt = torch.tensor([[vocab.id_of(t) for t in header]])
        torch.manual_seed(target["seed"])
        with torch.no_grad():
            out = model.generate(
                prompt,
                do_sample=True,
                temperature=sampling["temperature"],
                top_k=sampling["top_k"],
                repetition_penalty=sampling["repetition_penalty"],
                max_new_tokens=sampling["max_new_tokens"],
                eos_token_id=vocab.id_of("<EOS>"),
                pad_token_id=vocab.id_of("<PAD>"),
                logits_processor=processors,
                stopping_criteria=[
                    StopAfterBars(vocab.id_of("<BAR>"), prompt.shape[1], args.bars)
                ],
                use_cache=True,
            )
        trimmed = trim_to_bars(out[0].tolist(), vocab=vocab, n_bars=args.bars)
        score, _report = decode(trimmed.ids, vocab=vocab)
        lead_channel = intended_lead_channel(score)

        # Historical: real pre-fix arranger AND the 150 BPM encode override.
        old_arranged, old_report = historical.arrange_for_nes(
            score,
            row_sec=HISTORICAL_ROW_SEC,
            divisions=tokenizer.config.onset_divisions,
        )
        old = finish(
            old_arranged,
            tokenizer=tokenizer,
            engine=engine,
            period_sec=HISTORICAL_ROW_SEC,
        )
        # Corrected: musical positions AND a tempo-matched fit.
        new_arranged, new_report = arrange_for_nes(
            score,
            row_sec=HISTORICAL_ROW_SEC,
            divisions=tokenizer.config.onset_divisions,
        )
        new = finish(
            new_arranged,
            tokenizer=tokenizer,
            engine=engine,
            period_sec=row_seconds(score.tempo_bpm),
        )

        # Fidelity: does the historical branch reproduce the archived audio?
        old_wav = audit / f"{target['name']}.historical.wav"
        new_wav = audit / f"{target['name']}.corrected.wav"
        render_window(old, old_wav)
        normalise_peak(old_wav)
        render_window(new, new_wav)
        normalise_peak(new_wav)
        archived = args.archived / archived_file[target["name"]]
        fidelity: dict[str, Any] = {"archived": str(archived)}
        if archived.exists():
            want = hashlib.sha256(archived.read_bytes()).hexdigest()
            got = hashlib.sha256(old_wav.read_bytes()).hexdigest()
            fidelity.update(
                {
                    "archived_sha256": want,
                    "historical_sha256": got,
                    "bytes_identical": want == got,
                    "archived_bytes": archived.stat().st_size,
                    "historical_bytes": old_wav.stat().st_size,
                }
            )
        else:
            fidelity["bytes_identical"] = None

        row = {
            "name": target["name"],
            "scenario": target["scenario"],
            "seed": target["seed"],
            "intended_bpm": score.tempo_bpm,
            "source_bars": args.bars,
            "source_events": sum(
                len(t.notes) + len(t.percussion) for t in score.tracks
            ),
            "fidelity": fidelity,
            "historical_arrangement": old_report.to_dict(),
            "corrected_arrangement": new_report.to_dict(),
            "historical": describe(
                old, n_bars=args.bars, lead_channel=lead_channel, engine=engine
            ),
            "corrected": describe(
                new, n_bars=args.bars, lead_channel=lead_channel, engine=engine
            ),
        }
        rows.append(row)
        mark = (
            "exact"
            if fidelity.get("bytes_identical")
            else ("DIFFERS" if fidelity.get("bytes_identical") is False else "n/a")
        )
        print(
            f"  {target['name']:<34}{score.tempo_bpm:>5.0f}bpm "
            f"bars {row['historical']['bars_spanned']:>5.2f}->{row['corrected']['bars_spanned']:>5.2f} "
            f"bpm {row['historical']['encoded_bpm']:>6.1f}->{row['corrected']['encoded_bpm']:>6.1f} "
            f"lead({lead_channel}) {row['historical']['lead_half_ratio']}->"
            f"{row['corrected']['lead_half_ratio']}  repro {mark}",
            flush=True,
        )

    def med(side: str, field: str) -> float | None:
        values = [r[side][field] for r in rows if r[side][field] is not None]
        return st.median(values) if values else None

    fields = (
        "encoded_bpm",
        "bars_spanned",
        "duration_sec",
        "n_events",
        "empty_bars_in_window",
        "events_beyond_window",
        "lead_notes",
        "lead_half_ratio",
        "lead_contour_recurrence",
        "lead_distinct_pitches",
        "hardware_violations",
    )
    summary = {
        field: {"historical": med("historical", field), "corrected": med("corrected", field)}
        for field in fields
    }
    exact = sum(1 for r in rows if r["fidelity"].get("bytes_identical"))
    args.out.write_text(
        json.dumps(
            {
                "bars": args.bars,
                "historical_commit": args.historical_commit,
                "n_pieces": len(rows),
                "n_reproduced_archived_audio_exactly": exact,
                "archived_audio_dir": str(args.archived),
                "render_dir": str(args.render_dir),
                "summary_medians": summary,
                "pieces": rows,
            },
            indent=2,
        )
        + "\n"
    )

    print(f"\nhistorical branch reproduces archived audio exactly: {exact}/{len(rows)}")
    if exact != len(rows):
        print(
            "  differing: "
            + str([r["name"] for r in rows if not r["fidelity"].get("bytes_identical")])
        )
    print(f"\n{'measure':<28}{'historical':>12}{'corrected':>12}")
    print("-" * 52)
    for field, pair in summary.items():
        left = "n/a" if pair["historical"] is None else f"{pair['historical']:.3f}"
        right = "n/a" if pair["corrected"] is None else f"{pair['corrected']:.3f}"
        print(f"  {field:<26}{left:>12}{right:>12}")
    off_hist = sum(
        1 for r in rows if abs(r["historical"]["bars_spanned"] - args.bars) > 0.5
    )
    off_corr = sum(
        1 for r in rows if abs(r["corrected"]["bars_spanned"] - args.bars) > 0.5
    )
    print(
        f"\nspan not ~{args.bars} bars: historical {off_hist}/{len(rows)}, "
        f"corrected {off_corr}/{len(rows)}"
    )
    print(
        f"hardware violations: historical "
        f"{sum(r['historical']['hardware_violations'] for r in rows)}, corrected "
        f"{sum(r['corrected']['hardware_violations'] for r in rows)}"
    )
    print(f"\nwrote {args.out}; paired audio in {args.render_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
