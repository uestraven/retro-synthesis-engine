"""Does planning for NES hardware before composing produce better music?

Two full-length tracks from one prompt, same composer, same renderer:

- **arm A, current pipeline** — scenario-style conditioning, and the arranger
  infers channel roles from note activity after the fact.
- **arm B, composition-aware** — an NES composition profile fixes the four
  channel roles, the form, and a per-section key plan *before* generation, and
  the arranger honours those roles instead of guessing them.

**What is held equal, and why that matters.** Both arms run at the profile's
tempo, meter, duration and track plan. The obvious version of this experiment
lets arm A take its tempo from the prompt wording ("driving" reads as 170 BPM)
and arm B from the profile (128), which would make one track a third shorter
than the other and confound the thing being tested with pacing. Here the only
differences are the ones under test:

1. declared channel roles versus inferred ones;
2. an explicit A-A'-B-A'' form with one contrast section, versus the form the
   prompt interpreter derives;
3. a per-section key plan versus one key throughout.

Difference 1 is the substantive one. `assign_roles` picks the lead by mean
absolute interval, so a fast arpeggio outscores a stepwise tune -- and on 12
measured pieces it overrode the composer's own declared lead 6 times.

Nothing is ranked or selected on a metric. Both tracks are rendered, labelled
neutrally, and shuffled; the mapping is written outside the listening
directory. Stricter limits are not assumed to produce better music, which is
the whole reason for listening rather than asserting.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import json
import random
from typing import Any

from rse.hardware.constraints import ConstraintEngine
from rse.hardware.profile import load_profile
from rse.prompt.interpret import interpret
from rse.prompt.profile import NesCompositionProfile, Section, load
from rse.song.build import SongBuilder, bar_prefix
from rse.tokenizers.frozen import frozen_config
from rse.tokenizers.tokenizer import RseTokenizer
from rse.transfer.fermata import Vocab

PEAK_DBFS = -3.0

ACCEPTANCE_PROMPT = (
    "Compose a 60-second NES-era JRPG battle theme. Dark and ominous but "
    "energetic, with a memorable opening melody, driving percussion, a "
    "contrasting middle section and a return to the main theme. Make it "
    "seamlessly loopable."
)


def current_pipeline_profile(base: NesCompositionProfile, prompt: str) -> NesCompositionProfile:
    """Arm A: the shape the pipeline produced before profiles existed.

    Tempo, meter, duration and instrumentation are taken from the profile so
    the arms are comparable on length and pacing. What is NOT taken is the
    planning layer: no declared roles, no contrast key, and the form the
    prompt interpreter derives rather than an explicit A-A'-B-A''.
    """
    plan = interpret(prompt)
    arm = NesCompositionProfile(
        name="current_pipeline",
        duration_seconds=base.duration_seconds,
        tempo_bpm=base.tempo_bpm,
        beats_per_bar=base.beats_per_bar,
        genre=plan.genre,
        ensemble=plan.ensemble,
        key_root=plan.key_root,
        key_mode=plan.key_mode,
        roles=dict(base.roles),
        theme_bars=base.theme_bars,
    )
    # The interpreter's form: sections continuing from A, no keyed contrast.
    section_bars = max(2, base.theme_bars * 2)
    count = base.total_bars() // section_bars
    arm.sections = [Section("A", section_bars, "fresh")] + [
        Section(f"A{i}", section_bars, "continue_from", continue_from="A")
        for i in range(1, count)
    ]
    return arm


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default=ACCEPTANCE_PROMPT)
    parser.add_argument("--model", type=Path, default=Path("/tmp/fermata_dl"))
    parser.add_argument(
        "--profile", type=Path, default=Path("profiles/nes_battle.yaml")
    )
    parser.add_argument("--seed", type=int, default=606060)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=24)
    parser.add_argument("--repetition-penalty", type=float, default=1.12)
    parser.add_argument("--max-new-tokens", type=int, default=2000)
    parser.add_argument("--shuffle-seed", type=int, default=9090)
    parser.add_argument("--listen-dir", type=Path, default=Path("listen_profile_ab"))
    parser.add_argument("--out", type=Path, default=Path("profile_ab.json"))
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import torch
    from architecture_comparison import normalise_peak
    from codec_comparison import render_window
    from transformers import LlamaForCausalLM
    from transformers.generation.logits_process import LogitsProcessor

    profile_b = load(args.profile)
    profile_a = current_pipeline_profile(profile_b, args.prompt)
    print(f"prompt: {args.prompt}\n")
    for label, prof in (("A current", profile_a), ("B profile", profile_b)):
        print(
            f"  arm {label}: {prof.tempo_bpm} BPM, key {prof.key_root}:"
            f"{prof.key_mode}, form {'-'.join(s.label for s in prof.sections)}, "
            f"{prof.total_bars()} bars = {prof.planned_seconds():.1f} s"
        )
    print()

    vocab = Vocab.load(args.model / "tokenizer" / "vocab.json")
    model = LlamaForCausalLM.from_pretrained(args.model, dtype=torch.float32).eval()

    class SuppressBpe(LogitsProcessor):
        def __init__(self, ids: list[int]) -> None:
            self.ids = torch.tensor(ids, dtype=torch.long)

        def __call__(self, _input_ids, scores):
            scores[:, self.ids] = float("-inf")
            return scores

    class StopAfterBars(torch.nn.Module):
        def __init__(self, bar_id: int, offset: int, n_bars: int) -> None:
            super().__init__()
            self.bar_id = bar_id
            self.offset = offset
            self.limit = n_bars + 1

        def __call__(self, input_ids, _scores, **_kwargs) -> bool:
            return bool((input_ids[0] == self.bar_id).sum() + self.offset >= self.limit)

    processors = [SuppressBpe(vocab.bpe_ids)]
    bar_id = vocab.id_of("<BAR>")

    def generate(prompt_ids, *, seed, n_bars, bar_offset):
        torch.manual_seed(seed)
        with torch.no_grad():
            out = model.generate(
                torch.tensor([prompt_ids]),
                do_sample=True,
                temperature=args.temperature,
                top_k=args.top_k,
                repetition_penalty=args.repetition_penalty,
                max_new_tokens=args.max_new_tokens,
                eos_token_id=vocab.id_of("<EOS>"),
                pad_token_id=vocab.id_of("<PAD>"),
                logits_processor=processors,
                stopping_criteria=[StopAfterBars(bar_id, bar_offset, n_bars)],
                use_cache=True,
            )
        return out[0].tolist()

    tokenizer = RseTokenizer(frozen_config())
    hardware = load_profile("nes")
    engine = ConstraintEngine(hardware)

    arms: dict[str, dict[str, Any]] = {}
    tracks: dict[str, Any] = {}
    for label, prof, roles in (
        ("current", profile_a, None),
        ("profile", profile_b, profile_b.role_for_program()),
    ):
        builder = SongBuilder(
            vocab=vocab,
            tokenizer=tokenizer,
            engine=engine,
            generate=generate,
            role_for_program=roles,
        )
        prefixes: dict[str, list[int]] = {}
        results = []
        print(f"arm {label}:")
        for index, section in enumerate(prof.sections):
            result = builder.build_section(
                section,
                profile=prof,
                seed=args.seed + index * 1000,
                prefixes=prefixes,
            )
            if not result.ok:
                print(f"  {section.label}: FAILED")
                results.append(result)
                continue
            if section.source == "fresh" and section.label not in prefixes:
                cut = bar_prefix(
                    result.token_ids, vocab=vocab, n_bars=prof.theme_bars
                )
                if cut:
                    prefixes[section.label] = cut
            results.append(result)
            print(
                f"  {section.label:<5} {result.report['n_events']:>4} events, "
                f"roles {result.report['role_assignment']}, "
                f"{result.attempts} attempt(s)"
            )
        track, assembly = builder.assemble(results, profile=prof)
        tracks[label] = track
        arms[label] = {
            "profile": prof.to_dict(),
            "reachability": prof.reachability(),
            "sections": [r.to_dict() for r in results],
            "assembly": assembly,
            "duration_sec": round(
                track.time.seconds(track.loop_end_tick or track.end_tick or 0), 3
            ),
        }
        print(
            f"  -> {assembly['total_bars']} bars, "
            f"{arms[label]['duration_sec']:.1f} s, "
            f"{assembly['hardware_violations']} violations, "
            f"{assembly['seam_repairs']} seam repairs\n"
        )

    # Blind: neutral names, shuffled.
    args.listen_dir.mkdir(parents=True, exist_ok=True)
    labels = list(arms)
    random.Random(args.shuffle_seed).shuffle(labels)
    key = []
    for slot, label in enumerate(labels, 1):
        filename = f"track_{slot}.wav"
        render_window(tracks[label], args.listen_dir / filename)
        normalise_peak(args.listen_dir / filename, dbfs=PEAK_DBFS)
        key.append({"file": filename, "arm": label})
    key.sort(key=lambda row: row["file"])

    (args.listen_dir / "RATINGS.md").write_text(
        "# Two full tracks from the same prompt\n\n"
        "Same composer, same renderer, same tempo and length. One was "
        "generated with an NES composition plan fixing the four channel "
        "roles, the form and a key change for the middle section before "
        "anything was composed. The other was not. Nothing tells you which.\n\n"
        "Play each twice -- the second time tells you whether the loop joins.\n\n"
        "For each track:\n\n"
        "1. **Would this pass as NES game music?** yes / no\n"
        "2. **Is the melody recognisable when it comes back?** yes / partly / no\n"
        "3. **Does the accompaniment support the melody or compete with it?** "
        "supports / competes / neither\n"
        "4. **Do the sections transition well?** yes / rough / no\n"
        "5. **Does the loop join?** yes / audible seam / no\n\n"
        "Then: **which track is the better song overall?** `1`, `2`, or `=`\n\n"
        "| track | passes? | melody returns? | accompaniment | transitions | loop |\n"
        "|---|---|---|---|---|---|\n"
        + "".join(f"| {row['file']} |  |  |  |  |  |\n" for row in key)
        + "\n**Better song overall:** \n\n**Notes:** \n"
    )
    key_path = args.out.parent / f"{args.listen_dir.name}_KEY.json"
    key_path.write_text(json.dumps(key, indent=2) + "\n")
    args.out.write_text(
        json.dumps(
            {
                "version": "rse-profile-ab/0.1",
                "prompt": args.prompt,
                "held_equal": [
                    "tempo", "meter", "duration", "track plan", "renderer",
                    "sampling settings", "seed base",
                ],
                "under_test": [
                    "declared channel roles vs inferred",
                    "explicit A-A'-B-A'' form vs interpreter form",
                    "per-section key plan vs one key",
                ],
                "seed": args.seed,
                "arms": arms,
                "listening_key": key,
            },
            indent=2,
        )
        + "\n"
    )

    print(f"rendered 2 tracks to {args.listen_dir}")
    print(f"rate them in {args.listen_dir / 'RATINGS.md'}")
    print(f"key in {key_path} -- do not open it until the answers are recorded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
