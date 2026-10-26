"""Phase 3 and 4: does a supplied opening improve what Fermata writes next?

Two arms, both ending as corrected-pipeline NES audio:

- **guided** — 5 four-bar openings x 3 seeds. Bars 1-4 are supplied from the
  original token stream; bars 5-8 are generated.
- **control** — 15 fresh eight-bar generations, matched to the guided arm's
  scenario and tempo distribution. Fresh seeds, not the five pieces the
  openings came from: comparing new material against pieces already selected
  for being good would be no comparison at all.

The measurement that matters is **bars 5-8 only**, and both arms are cut the
same way. A guided piece whose first four bars are known-good would win any
whole-piece comparison without generating anything worth hearing, so the
primary listening view contains only the last four bars of every piece, guided
and control alike.

Thematic development needs the opening, so it is a second view over the whole
eight bars and a separate judgement. Exact bar repetition is not scored as
development: the corpus restates exact bars in only 9.1% of its own 16-bar
spans, so that metric was retired.

Nothing is ranked, filtered or post-selected. Failures are recorded with their
reason and replaced from the same condition, and the attempt counts are
reported beside the ratings.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import dataclasses
import json
import random
from collections import Counter
from typing import Any

from rse.analysis.restatement import contour, recurrence, ticks_per_bar
from rse.codec.benchmark import enforce_monophony, ioi_entropy
from rse.conditioning import melody_line, motif_note_participation
from rse.events.schema import Composition, Note
from rse.hardware.constraints import ConstraintEngine
from rse.hardware.profile import load_profile
from rse.tokenizers.frozen import frozen_config
from rse.tokenizers.tokenizer import GridFit, RseTokenizer
from rse.training.evaluate import evaluate_composition
from rse.transfer.arranger import arrange_for_nes, row_seconds
from rse.transfer.fermata import Vocab, decode, trim_to_bars

ROW_SEC = 0.1
THEME_BARS = 4
TARGET_BARS = 8
PEAK_DBFS = -3.0


def slice_bars(
    composition: Composition, *, start_bar: int, end_bar: int, bar_ticks: int
) -> Composition:
    """A new composition holding only `[start_bar, end_bar)`, shifted to zero.

    Used to render bars 5-8 on their own. Events are kept whole: a note that
    begins inside the window keeps its full duration even if it rings past the
    end, because shortening it would change the music rather than the excerpt.
    A note that began earlier and is still sounding is dropped -- it belongs to
    the material being withheld.
    """
    low, high = start_bar * bar_ticks, end_bar * bar_ticks
    events = []
    for event in composition.events:
        if low <= event.tick < high:
            events.append(dataclasses.replace(event, tick=event.tick - low))
    sliced = Composition(
        hardware_profile=composition.hardware_profile,
        ticks_per_beat=composition.ticks_per_beat,
        events=events,
    )
    sliced.tempo_map = list(composition.tempo_map)
    sliced.time_signatures = list(composition.time_signatures)
    sliced.normalize()
    return sliced


def develop_metrics(
    composition: Composition, *, theme_bars: int, target_bars: int, bar_ticks: int
) -> dict[str, Any]:
    """How the later material relates to the opening, without scoring copying.

    `contour_recurrence` is transposition- and rhythm-invariant, so a figure
    that returns transposed or rhythmically altered counts, while a verbatim
    paste is reported separately as `identical_bars` so the two can be told
    apart.
    """
    split = theme_bars * bar_ticks
    channel, melody = melody_line(composition)
    early = [p for tick, p in _ticked(composition, channel) if tick < split]
    late = [p for tick, p in _ticked(composition, channel) if tick >= split]

    def fingerprint(bar: int) -> frozenset[tuple[Any, int, Any]]:
        return frozenset(
            (
                e.channel_id,
                e.tick - bar * bar_ticks,
                getattr(e, "pitch", None) if isinstance(e, Note) else getattr(e, "value", None),
            )
            for e in composition.sounding_events()
            if bar * bar_ticks <= e.tick < (bar + 1) * bar_ticks
        )

    opening = [fingerprint(b) for b in range(theme_bars)]
    later = [fingerprint(b) for b in range(theme_bars, target_bars)]
    identical = sum(
        1 for b in later if any(b == a and b for a in opening)
    )
    return {
        "lead_channel": channel,
        "n_lead_opening": len(early),
        "n_lead_later": len(late),
        "lead_ratio": round(len(late) / len(early), 4) if early else None,
        "contour_recurrence": recurrence(contour(early), contour(late)),
        "identical_bars": identical,
        "melody_notes": len(melody),
    }


def _ticked(composition: Composition, channel: str) -> list[tuple[int, int]]:
    if not channel:
        return []
    return [
        (e.tick, e.pitch)
        for e in sorted(composition.sounding_events(channel), key=lambda e: e.tick)
        if isinstance(e, Note)
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("/tmp/fermata_dl"))
    parser.add_argument("--bank", type=Path, default=Path("theme_bank_4bar.json"))
    parser.add_argument("--baseline", type=Path, default=Path("corrected_baseline.json"))
    parser.add_argument("--per-theme", type=int, default=3)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--theme-bars", type=int, default=THEME_BARS)
    parser.add_argument("--target-bars", type=int, default=TARGET_BARS)
    parser.add_argument("--seed-base", type=int, default=31000)
    parser.add_argument("--shuffle-seed", type=int, default=515151)
    parser.add_argument("--listen-dir", type=Path, default=Path("listen_guided"))
    parser.add_argument("--out", type=Path, default=Path("theme_guided.json"))
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import torch
    from architecture_comparison import normalise_peak
    from codec_comparison import render_window
    from fermata_generate import SCENARIOS, build_header
    from transformers import LlamaForCausalLM
    from transformers.generation.logits_process import LogitsProcessor

    bank = json.loads(args.bank.read_text())
    sampling = json.loads(args.baseline.read_text())["design"]
    by_name = {s["name"]: s for s in SCENARIOS}
    themes = bank["themes"]
    print(f"{len(themes)} openings x {args.per_theme} seeds = "
          f"{len(themes) * args.per_theme} guided; same count control")

    vocab = Vocab.load(args.model / "tokenizer" / "vocab.json")
    model = LlamaForCausalLM.from_pretrained(args.model, dtype=torch.float32).eval()

    class SuppressBpe(LogitsProcessor):
        def __init__(self, ids: list[int]) -> None:
            self.ids = torch.tensor(ids, dtype=torch.long)

        def __call__(self, _input_ids, scores):
            scores[:, self.ids] = float("-inf")
            return scores

    class StopAfterBars(torch.nn.Module):
        """Counts `<BAR>` over the whole sequence, prefix included."""

        def __init__(self, bar_id: int, offset: int, n_bars: int) -> None:
            super().__init__()
            self.bar_id = bar_id
            self.offset = offset
            self.limit = n_bars + 1

        def __call__(self, input_ids, _scores, **_kwargs) -> bool:
            return bool((input_ids[0] == self.bar_id).sum() + self.offset >= self.limit)

    processors = [SuppressBpe(vocab.bpe_ids)]
    tokenizer = RseTokenizer(frozen_config())
    profile = load_profile("nes")
    engine = ConstraintEngine(profile)
    bar_ticks = ticks_per_bar()
    audit = args.listen_dir.parent / f"{args.listen_dir.name}_audit"
    audit.mkdir(parents=True, exist_ok=True)

    def finish(ids: list[int]) -> tuple[Composition, dict[str, Any]] | None:
        """Token stream to hardware-valid NES composition, or None."""
        trimmed = trim_to_bars(ids, vocab=vocab, n_bars=args.target_bars)
        if not (trimmed.complete and trimmed.n_bars == args.target_bars):
            return None
        score, decode_report = decode(trimmed.ids, vocab=vocab)
        arranged, arrange_report = arrange_for_nes(
            score, row_sec=ROW_SEC, divisions=tokenizer.config.onset_divisions
        )
        arranged, _t = enforce_monophony(arranged)
        fit = GridFit(
            step_ticks=float(tokenizer.ticks_per_step(480)),
            phase_ticks=0.0,
            aligned=True,
            period_sec=row_seconds(score.tempo_bpm),
        )
        rounded = enforce_monophony(
            tokenizer.decode(tokenizer.encode(arranged, fit=fit).ids).composition
        )[0]
        final, _repair = engine.apply(rounded)
        if not final.sounding_events():
            return None
        ticks = [e.tick for e in final.sounding_events()]
        if min(ticks) < 0 or max(ticks) >= args.target_bars * bar_ticks:
            return None
        return final, {
            "termination": trimmed.reason,
            "decode": decode_report.to_dict(),
            "arrangement": arrange_report.to_dict(),
            "tempo_bpm": score.tempo_bpm,
            "hardware_violations": len(engine.validate(final)),
        }

    def measure(final: Composition) -> dict[str, Any]:
        events = final.sounding_events()
        metrics = evaluate_composition(final, profile)
        _c, melody = melody_line(final)
        end = final.end_tick or 1
        spt = (final.time.seconds(end) / end) if final.time.seconds(end) else 0.0
        return {
            "n_events": len(events),
            "notes_per_bar": round(metrics.notes_per_bar, 3),
            "pitch_class_entropy": round(metrics.pitch_class_entropy, 4),
            "stepwise_fraction": round(metrics.stepwise_fraction, 4),
            "motif_note_participation": round(motif_note_participation(melody), 4),
            "ioi_entropy": round(ioi_entropy([e.tick * spt for e in events]), 4),
            "channels": sorted({e.channel_id for e in events}),
            "duration_sec": round(final.time.seconds(end), 3),
        }

    pieces: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    full: dict[str, Composition] = {}
    tail: dict[str, Composition] = {}

    # ---- guided ----------------------------------------------------------
    for theme in themes:
        produced = attempt = 0
        prefix = theme["prefix_ids"]
        offset = 1 if theme["implicit_first_bar"] else 0
        while produced < args.per_theme and attempt < args.per_theme * args.max_attempts:
            seed = args.seed_base + theme["theme_id"] * 100 + attempt
            attempt += 1
            torch.manual_seed(seed)
            with torch.no_grad():
                out = model.generate(
                    torch.tensor([prefix]),
                    do_sample=True,
                    temperature=sampling["temperature"],
                    top_k=sampling["top_k"],
                    repetition_penalty=sampling["repetition_penalty"],
                    max_new_tokens=sampling["max_new_tokens"],
                    eos_token_id=vocab.id_of("<EOS>"),
                    pad_token_id=vocab.id_of("<PAD>"),
                    logits_processor=processors,
                    stopping_criteria=[
                        StopAfterBars(vocab.id_of("<BAR>"), offset, args.target_bars)
                    ],
                    use_cache=True,
                )
            ids = out[0].tolist()
            name = f"guided_t{theme['theme_id']}_s{seed}"
            if ids[: len(prefix)] != prefix:
                failures.append({"name": name, "arm": "guided", "reason": "prefix_altered"})
                continue
            built = finish(ids)
            if built is None:
                failures.append(
                    {"name": name, "arm": "guided", "theme_id": theme["theme_id"],
                     "seed": seed, "reason": "incomplete_or_invalid"}
                )
                continue
            final, report = built
            pieces.append(
                {
                    "name": name,
                    "arm": "guided",
                    "theme_id": theme["theme_id"],
                    "theme_source": theme["source_name"],
                    "theme_idea": None,
                    "scenario": theme["scenario"],
                    "seed": seed,
                    "attempt": attempt,
                    "n_prefix_tokens": len(prefix),
                    "n_generated_tokens": len(ids) - len(prefix),
                    **report,
                    "metrics": measure(final),
                    "development": develop_metrics(
                        final, theme_bars=args.theme_bars,
                        target_bars=args.target_bars, bar_ticks=bar_ticks
                    ),
                }
            )
            full[name] = final
            tail[name] = slice_bars(
                final, start_bar=args.theme_bars, end_bar=args.target_bars,
                bar_ticks=bar_ticks
            )
            produced += 1
            print(
                f"  {name:<28} theme {theme['theme_id']} "
                f"gen {len(ids) - len(prefix):>4} tok  "
                f"ev {report['arrangement']['n_events_out']:>4}",
                flush=True,
            )
        if produced < args.per_theme:
            print(f"  ! theme {theme['theme_id']}: {produced}/{args.per_theme}")

    # ---- control: matched scenario distribution --------------------------
    wanted = Counter(t["scenario"] for t in themes)
    plan = [s for s, n in wanted.items() for _ in range(n * args.per_theme)]
    print(f"control scenario plan: {dict(Counter(plan))}")
    counts: Counter[str] = Counter()
    attempt = 0
    for scenario_name in plan:
        scenario = by_name[scenario_name]
        produced = False
        tries = 0
        while not produced and tries < args.max_attempts:
            seed = args.seed_base + 50000 + attempt * 10 + tries
            tries += 1
            attempt += 1
            prompt = torch.tensor([[vocab.id_of(t) for t in build_header(vocab, scenario)]])
            torch.manual_seed(seed)
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
                        StopAfterBars(vocab.id_of("<BAR>"), 0, args.target_bars)
                    ],
                    use_cache=True,
                )
            ids = out[0].tolist()
            counts[scenario_name] += 1
            name = f"control_{scenario_name}_s{seed}"
            built = finish(ids)
            if built is None:
                failures.append(
                    {"name": name, "arm": "control", "scenario": scenario_name,
                     "seed": seed, "reason": "incomplete_or_invalid"}
                )
                continue
            final, report = built
            pieces.append(
                {
                    "name": name,
                    "arm": "control",
                    "theme_id": None,
                    "scenario": scenario_name,
                    "seed": seed,
                    "attempt": tries,
                    "n_prefix_tokens": 0,
                    "n_generated_tokens": len(ids),
                    **report,
                    "metrics": measure(final),
                    "development": develop_metrics(
                        final, theme_bars=args.theme_bars,
                        target_bars=args.target_bars, bar_ticks=bar_ticks
                    ),
                }
            )
            full[name] = final
            tail[name] = slice_bars(
                final, start_bar=args.theme_bars, end_bar=args.target_bars,
                bar_ticks=bar_ticks
            )
            produced = True
            print(f"  {name:<34} ev {report['arrangement']['n_events_out']:>4}", flush=True)

    if not pieces:
        print("error: nothing generated", file=sys.stderr)
        return 2

    # ---- two blind views -------------------------------------------------
    rng = random.Random(args.shuffle_seed)
    order = list(range(len(pieces)))
    # Shuffle, then separate same-theme continuations so three versions of one
    # opening never land next to each other.
    for _ in range(200):
        rng.shuffle(order)
        themes_seq = [pieces[i].get("theme_id") for i in order]
        if not any(
            a is not None and a == b for a, b in zip(themes_seq, themes_seq[1:], strict=False)
        ):
            break

    view1 = args.listen_dir / "view1_continuation"
    view2 = args.listen_dir / "view2_full"
    view1.mkdir(parents=True, exist_ok=True)
    view2.mkdir(parents=True, exist_ok=True)
    key: list[dict[str, Any]] = []
    for slot, index in enumerate(order, 1):
        record = pieces[index]
        clip = f"clip_{slot:02d}.wav"
        whole = f"piece_{slot:02d}.wav"
        if tail[record["name"]].sounding_events():
            render_window(tail[record["name"]], view1 / clip)
            normalise_peak(view1 / clip, dbfs=PEAK_DBFS)
        render_window(full[record["name"]], view2 / whole)
        normalise_peak(view2 / whole, dbfs=PEAK_DBFS)
        record["render"] = {"clip": clip, "piece": whole}
        key.append(
            {
                "clip": clip,
                "piece": whole,
                "arm": record["arm"],
                "name": record["name"],
                "theme_id": record["theme_id"],
                "scenario": record["scenario"],
                "seed": record["seed"],
            }
        )
        (audit / f"{record['name']}.json").write_text(json.dumps(record, indent=2) + "\n")

    (view1 / "RATINGS.md").write_text(
        "# View 1 -- the last four bars only\n\n"
        f"{len(key)} clips. Each is **bars 5-8 of a piece, with bars 1-4 "
        "removed.**\n"
        "You are not hearing any opening. Some of these pieces were given a\n"
        "four-bar opening before this material was written and some were not,\n"
        "and nothing here tells you which.\n\n"
        "This is the main measurement. Rate the music you hear:\n"
        "**BEST / GOOD / OK / BAD**.\n\n"
        "Judge it on its own -- melody, harmony, whether it goes anywhere.\n"
        "A clip starting mid-piece will feel like it starts abruptly; that is\n"
        "true of every clip here, so do not hold it against any of them.\n\n"
        "| clip | rating | comment |\n|---|---|---|\n"
        + "".join(f"| {row['clip']} |  |  |\n" for row in key)
        + "\n"
    )
    (view2 / "RATINGS.md").write_text(
        "# View 2 -- whole pieces, does the later material develop the opening?\n\n"
        f"{len(key)} pieces, eight bars each. Same pieces as view 1, whole.\n\n"
        "**Do this view after view 1.** It needs the opening, which view 1\n"
        "deliberately withholds.\n\n"
        "One question: **does the second half relate musically to the first?**\n"
        "Write `STRONG`, `PARTIAL`, or `ABSENT`.\n\n"
        "`STRONG` -- the opening idea comes back, answered, varied or\n"
        "developed. It does not need to be repeated exactly; a figure that\n"
        "returns transposed or with changed rhythm counts fully.\n"
        "`PARTIAL` -- some connection, or the same texture without the idea.\n"
        "`ABSENT` -- the second half could belong to a different piece.\n\n"
        "A piece that simply copies its opening bars verbatim is `PARTIAL`,\n"
        "not `STRONG` -- copying is not development.\n\n"
        "| piece | connection | comment |\n|---|---|---|\n"
        + "".join(f"| {row['piece']} |  |  |\n" for row in key)
        + "\n"
    )
    key_path = args.out.parent / f"{args.listen_dir.name}_KEY.json"
    key_path.write_text(json.dumps(key, indent=2) + "\n")

    by_arm = Counter(r["arm"] for r in pieces)
    payload = {
        "design": {
            "theme_bars": args.theme_bars,
            "target_bars": args.target_bars,
            "per_theme": args.per_theme,
            "sampling": {
                k: sampling[k]
                for k in ("temperature", "top_k", "repetition_penalty", "max_new_tokens")
            },
            "primary_measure": "view 1, bars 5-8 only, both arms cut identically",
            "secondary_measure": "view 2, whole piece, thematic connection",
            "ranking_used": None,
            "pipeline": "corrected timing + implicit first bar",
        },
        "n_pieces": len(pieces),
        "per_arm": dict(by_arm),
        "n_failed": len(failures),
        "failures_by_reason": dict(Counter(f["reason"] for f in failures)),
        "control_attempts": sum(counts.values()),
        "pieces": pieces,
        "failures": failures,
        "listening_key": key,
    }
    args.out.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"\n{len(pieces)} pieces: {dict(by_arm)}; {len(failures)} failures")
    if failures:
        print(f"  by reason: {payload['failures_by_reason']}")
    print(f"view 1 (bars 5-8): {view1}")
    print(f"view 2 (full):     {view2}")
    print(f"key: {key_path} -- do not open it until both views are rated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
