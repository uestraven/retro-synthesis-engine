"""Phase 1 and 2: build the opening-theme bank and prove continuation works.

The bank is the first two complete bars of the five Best/Good pieces from the
corrected 20-piece baseline, taken from their **original Fermata token
streams** stored in `listen_corrected_audit/`. Not from the NES arrangement,
and not from MIDI reconstructed out of it: the prefix has to be something the
model can actually continue, which means its own grammar, with the scenario
header, tempo, meter, declared plan and every track intact.

Two bars are extracted as a *prefix*, not as a trimmed piece. `trim_to_bars`
appends `<EOS>` to close a stream, which is the opposite of what is wanted
here -- the cut stops just before the token that would open bar 3, so the
model's next move is its own. Whether it then emits `<BAR>` is a test of the
model rather than of the splice, which is why the boundary token is withheld
rather than supplied.

The seven structural checks from the brief are run per opening, and the script
fails closed: if continuation does not resume cleanly at bar 3, the batch
should not be generated.

**Phase 1 also needs a human answer before Phase 3 runs.** Each opening is
rendered as audio so its two bars can be judged for whether they contain a
recognisable musical idea at all. A piece can become good after a weak
opening, so "these five pieces were rated Best or Good" does not establish
that their first two bars are strong themes. Weak openings stay in the bank as
diagnostic cases; replacing them quietly would turn the experiment into a
search for openings that make it succeed.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import json
import re
from typing import Any

from rse.codec.benchmark import enforce_monophony
from rse.events.schema import Composition
from rse.hardware.constraints import ConstraintEngine
from rse.hardware.profile import load_profile
from rse.tokenizers.frozen import frozen_config
from rse.tokenizers.tokenizer import GridFit, RseTokenizer
from rse.transfer.arranger import arrange_for_nes, row_seconds
from rse.transfer.fermata import Vocab, decode, trim_to_bars

ROW_SEC = 0.1
THEME_BARS = 2
TARGET_BARS = 8


def read_ratings(path: Path) -> dict[str, tuple[str, str]]:
    out: dict[str, tuple[str, str]] = {}
    for line in path.read_text().splitlines():
        match = re.match(
            r"\|\s*(take_\d+\.wav)\s*\|\s*(BEST|GOOD|OK|BAD)\s*\|\s*(.*?)\s*\|", line
        )
        if match:
            out[match.group(1)] = (match.group(2), match.group(3))
    return out


def bar_prefix(ids: list[int], *, vocab: Vocab, n_bars: int) -> dict[str, Any]:
    """Cut a stream just before the token that opens bar `n_bars + 1`.

    Unlike `trim_to_bars` this appends nothing: the result is a prefix for the
    model to continue, so no `<EOS>` and no `<BAR>` for the next bar. An
    implicit opening bar counts as the first, matching `decode`.
    """
    bar_id = vocab.id_of("<BAR>")
    implicit = False
    for token_id in ids:
        if token_id == bar_id:
            break
        if vocab.id_to_token.get(token_id, "").startswith(("<PITCH_", "<DRUM_NOTE_")):
            implicit = True
            break

    seen = 1 if implicit else 0
    for index, token_id in enumerate(ids):
        if token_id == bar_id:
            seen += 1
            if seen == n_bars + 1:
                return {
                    "ids": ids[:index],
                    "n_bars": n_bars,
                    "implicit_first_bar": implicit,
                    "cut_at": index,
                    "complete": True,
                }
    return {
        "ids": list(ids),
        "n_bars": seen,
        "implicit_first_bar": implicit,
        "cut_at": len(ids),
        "complete": False,
    }


def structural_checks(
    *,
    prefix: list[int],
    completed: list[int],
    vocab: Vocab,
    theme_bars: int,
    target_bars: int,
    original: list[int],
) -> dict[str, Any]:
    """The seven checks from the brief, each answered separately."""
    eos_id = vocab.id_of("<EOS>")
    tokens = [vocab.id_to_token.get(i, "<UNK>") for i in completed]
    generated = completed[len(prefix) :]
    gen_tokens = [vocab.id_to_token.get(i, "<UNK>") for i in generated]

    # Where does the first generated bar boundary fall, and is it the first
    # thing generated? Anything emitted before it belongs to bar 2.
    first_bar_at = next(
        (i for i, t in enumerate(gen_tokens) if t == "<BAR>"), None
    )
    notes_before_boundary = sum(
        1
        for t in gen_tokens[: first_bar_at or 0]
        if t.startswith(("<PITCH_", "<DRUM_NOTE_"))
    )

    trimmed = trim_to_bars(completed, vocab=vocab, n_bars=target_bars)
    _score, report = decode(trimmed.ids, vocab=vocab)

    # The prefix must survive token-identically in the completed stream, and
    # must not already contain material from the original bars 3+.
    prefix_intact = completed[: len(prefix)] == prefix
    prefix_is_original = original[: len(prefix)] == prefix

    header_families = (
        "<BOS>",
        "<TEMPO_",
        "<METER_",
        "<PLAN>",
        "<TRACK_",
        "<DRUM_TRACK>",
        "<PLAN_END>",
    )
    prefix_tokens = [vocab.id_to_token.get(i, "") for i in prefix]
    has_context = all(
        any(t.startswith(f) or t == f for t in prefix_tokens) for f in header_families
    )

    checks = {
        "prefix_ends_at_bar_boundary": bool(prefix) and eos_id not in prefix,
        "resumes_at_bar_3": first_bar_at is not None and notes_before_boundary == 0,
        "no_duplicate_or_skipped_boundary": tokens.count("<BAR>")
        >= target_bars - (1 if report.implicit_first_bar else 0),
        "prefix_token_identical": prefix_intact and prefix_is_original,
        "full_musical_context_present": has_context,
        "no_leakage_from_original_tail": len(prefix) < len(original),
        "eight_valid_bars": bool(trimmed.complete and trimmed.n_bars == target_bars),
    }
    return {
        "checks": checks,
        "all_passed": all(checks.values()),
        "failed": [k for k, v in checks.items() if not v],
        "n_prefix_tokens": len(prefix),
        "n_generated_tokens": len(generated),
        "first_generated_bar_at": first_bar_at,
        "notes_before_first_boundary": notes_before_boundary,
        "n_bar_tokens_total": tokens.count("<BAR>"),
        "implicit_first_bar": report.implicit_first_bar,
        "decoded_bars": report.n_bars,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("/tmp/fermata_dl"))
    parser.add_argument("--baseline", type=Path, default=Path("corrected_baseline.json"))
    parser.add_argument("--audit", type=Path, default=Path("listen_corrected_audit"))
    parser.add_argument("--ratings", type=Path, default=Path("listen_corrected/RATINGS.md"))
    parser.add_argument("--theme-bars", type=int, default=THEME_BARS)
    parser.add_argument("--target-bars", type=int, default=TARGET_BARS)
    parser.add_argument("--validate-seed", type=int, default=99)
    parser.add_argument("--theme-dir", type=Path, default=Path("listen_themes"))
    parser.add_argument("--out", type=Path, default=Path("theme_bank.json"))
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import torch
    from architecture_comparison import normalise_peak
    from codec_comparison import render_window
    from transformers import LlamaForCausalLM
    from transformers.generation.logits_process import LogitsProcessor

    baseline = json.loads(args.baseline.read_text())
    by_file = {r["render"]["file"]: r for r in baseline["accepted"]}
    ratings = read_ratings(args.ratings)
    sampling = baseline["design"]

    bank = [
        {"file": f, "rating": r, "comment": note, "record": by_file[f]}
        for f, (r, note) in ratings.items()
        if r in ("BEST", "GOOD")
    ]
    bank.sort(key=lambda b: ({"BEST": 0, "GOOD": 1}[b["rating"]], b["record"]["name"]))
    print(f"opening bank: {len(bank)} Best/Good pieces")

    vocab = Vocab.load(args.model / "tokenizer" / "vocab.json")
    model = LlamaForCausalLM.from_pretrained(args.model, dtype=torch.float32).eval()

    class SuppressBpe(LogitsProcessor):
        def __init__(self, ids: list[int]) -> None:
            self.ids = torch.tensor(ids, dtype=torch.long)

        def __call__(self, _input_ids, scores):
            scores[:, self.ids] = float("-inf")
            return scores

    class StopAfterBars(torch.nn.Module):
        """Counts bars over the whole sequence, prefix included."""

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
    args.theme_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for entry in bank:
        name = entry["record"]["name"]
        stored = json.loads((args.audit / f"{name}.json").read_text())
        original = stored["raw_ids"]
        cut = bar_prefix(original, vocab=vocab, n_bars=args.theme_bars)
        if not cut["complete"]:
            print(f"  ! {name}: could not cut {args.theme_bars} bars")
            continue
        prefix = cut["ids"]

        # Render the opening alone so its musical content can be judged.
        theme_trim = trim_to_bars(original, vocab=vocab, n_bars=args.theme_bars)
        theme_score, theme_report = decode(theme_trim.ids, vocab=vocab)
        arranged, _ar = arrange_for_nes(
            theme_score, row_sec=ROW_SEC, divisions=tokenizer.config.onset_divisions
        )
        arranged, _t = enforce_monophony(arranged)
        fit = GridFit(
            step_ticks=float(tokenizer.ticks_per_step(480)),
            phase_ticks=0.0,
            aligned=True,
            period_sec=row_seconds(theme_score.tempo_bpm),
        )
        rounded = enforce_monophony(
            tokenizer.decode(tokenizer.encode(arranged, fit=fit).ids).composition
        )[0]
        theme_final, _rep = engine.apply(rounded)
        theme_wav = args.theme_dir / f"theme_{len(rows) + 1:02d}.wav"
        render_window(theme_final, theme_wav)
        normalise_peak(theme_wav)

        # Drums off. "Does a pitched idea get established" is the question,
        # and a busy drum part can carry an opening that has no tune in it --
        # which is how three of the two-bar openings read as having content.
        pitched_only = Composition(
            hardware_profile=theme_final.hardware_profile,
            ticks_per_beat=theme_final.ticks_per_beat,
            events=[e for e in theme_final.events if e.channel_id != "NO"],
        )
        pitched_only.tempo_map = list(theme_final.tempo_map)
        pitched_only.time_signatures = list(theme_final.time_signatures)
        pitched_only.normalize()
        pitched_wav = None
        if pitched_only.sounding_events():
            pitched_wav = args.theme_dir / f"theme_{len(rows) + 1:02d}.pitched.wav"
            render_window(pitched_only, pitched_wav)
            normalise_peak(pitched_wav)

        # Phase 2: does continuation actually work from this prefix?
        prompt = torch.tensor([prefix])
        torch.manual_seed(args.validate_seed)
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
                    StopAfterBars(
                        vocab.id_of("<BAR>"),
                        1 if cut["implicit_first_bar"] else 0,
                        args.target_bars,
                    )
                ],
                use_cache=True,
            )
        completed = out[0].tolist()
        checks = structural_checks(
            prefix=prefix,
            completed=completed,
            vocab=vocab,
            theme_bars=args.theme_bars,
            target_bars=args.target_bars,
            original=original,
        )
        n_pitched = sum(len(t.notes) for t in theme_score.tracks)
        n_drums = sum(len(t.percussion) for t in theme_score.tracks)
        theme_events = n_pitched + n_drums
        pitched_programs = sorted(
            {t.program for t in theme_score.tracks if t.notes and t.program is not None}
        )
        row = {
            "theme_id": len(rows) + 1,
            "theme_wav": theme_wav.name,
            "source_name": name,
            "source_rating": entry["rating"],
            "source_comment": entry["comment"],
            "scenario": entry["record"]["scenario"],
            "tempo_bpm": theme_score.tempo_bpm,
            "n_prefix_tokens": len(prefix),
            "n_theme_events": theme_events,
            "n_pitched_notes": n_pitched,
            "n_drum_hits": n_drums,
            "pitched_programs": pitched_programs,
            "pitched_wav": pitched_wav.name if pitched_wav else None,
            "theme_tracks_with_notes": sum(
                1 for t in theme_score.tracks if t.notes or t.percussion
            ),
            "implicit_first_bar": cut["implicit_first_bar"],
            "theme_duration_sec": round(
                theme_final.time.seconds(theme_final.end_tick or 0), 3
            ),
            "prefix_ids": prefix,
            "validation": checks,
        }
        rows.append(row)
        print(
            f"  theme {row['theme_id']} {name:<24} {entry['rating']:<5} "
            f"{n_pitched:>3} pitched + {n_drums:>3} drums  progs {pitched_programs}  "
            f"prefix {len(prefix):>4} tok  "
            f"{'VALID' if checks['all_passed'] else 'FAILED: ' + ','.join(checks['failed'])}",
            flush=True,
        )

    sheet = args.theme_dir / "THEMES.md"
    sheet.write_text(
        f"# Opening themes -- do these {args.theme_bars} bars establish an idea?\n\n"
        f"Five {args.theme_bars}-bar openings from the five pieces you rated\n"
        "Best or Good in the corrected baseline.\n\n"
        "**The question is whether a PITCHED idea gets established** -- a\n"
        "melody, a bass riff, any pitched figure. Drums alone do not count:\n"
        "this experiment is about melodic development, and a busy drum part\n"
        "can make an opening sound like it has content when it has no tune in\n"
        "it. That is what happened at two bars.\n\n"
        "Each theme has two files. Play `theme_NN.wav` first; if the drums\n"
        "make it hard to tell, play `theme_NN.pitched.wav`, which is the same\n"
        "music with the drum channel muted.\n\n"
        "For each: **is there a musical idea here?** Write `YES`, `WEAK`, or\n"
        "`NO`.\n\n"
        "`YES` -- a pitched figure you could recognise again if it came back.\n"
        "`WEAK` -- something pitched is there but it is not distinctive.\n"
        "`NO` -- no pitched idea. Drums only, or just notes starting.\n\n"
        "| theme | idea? YES / WEAK / NO | notes |\n|---|---|---|\n"
        + "".join(f"| theme_{row['theme_id']:02d} |  |  |\n" for row in rows)
        + "\nWeak or absent openings stay in the experiment as diagnostic\n"
        "cases. They are not replaced -- if most openings are weak, that is\n"
        "itself the finding.\n"
    )

    passed = sum(1 for r in rows if r["validation"]["all_passed"])
    args.out.write_text(
        json.dumps(
            {
                "theme_bars": args.theme_bars,
                "target_bars": args.target_bars,
                "n_themes": len(rows),
                "n_validated": passed,
                "sampling": {
                    k: sampling[k]
                    for k in ("temperature", "top_k", "repetition_penalty", "max_new_tokens")
                },
                "validate_seed": args.validate_seed,
                "themes": rows,
            },
            indent=2,
        )
        + "\n"
    )

    print(f"\ncontinuation validated on {passed}/{len(rows)} openings")
    if passed != len(rows):
        print("  check failures:")
        for r in rows:
            if not r["validation"]["all_passed"]:
                print(f"    theme {r['theme_id']}: {r['validation']['failed']}")
        print("\nNOT safe to generate the batch until continuation resumes cleanly.")
    print(f"\nwrote {args.out}; theme audio in {args.theme_dir}")
    print(f"answer {sheet} before Phase 3")
    return 0 if passed == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
