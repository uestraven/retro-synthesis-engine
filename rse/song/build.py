"""Build a multi-section NES track from a composition profile.

One implementation of the generate-arrange-assemble path, shared by
`scripts/prompt_to_song.py` and `scripts/profile_ab.py`. The alternative was a
third copy of it, and this project has already had one bug from two copies of
bar-boundary arithmetic disagreeing.

Three things this module is careful about, each because getting them wrong has
already cost a round of work:

**Row length comes from the score's tempo.** `rse.tokenizers.tokenizer`
overrides the Tempo token with `fit.bpm` whenever the fit is aligned, so a
caller leaving `period_sec` at a fixed 0.1 forces every piece to 150 BPM
however correctly the arranger placed the notes.

**Monophony and hardware repair run on the assembled track**, not only per
section. Five individually valid sections concatenated once produced five
violations: a note sustaining past its section's last bar line collides with
the next section's opening on a monophonic channel.

**Loop points are carried across the repair explicitly.** `ConstraintEngine`
returns a new composition and does not preserve them.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

from rse.analysis.restatement import ticks_per_bar
from rse.codec.benchmark import enforce_monophony
from rse.events.schema import Composition, TempoChange, TimeSignatureChange
from rse.prompt.profile import NesCompositionProfile, Section
from rse.transfer.arranger import arrange_for_nes, row_seconds
from rse.transfer.fermata import Vocab, decode, trim_to_bars

ROW_SEC = 0.1
TICKS_PER_BEAT = 480


@dataclass
class SectionResult:
    """One generated section and how it came about."""

    label: str
    bars: int
    origin: str
    composition: Composition | None = None
    token_ids: list[int] = field(default_factory=list)
    attempts: int = 0
    report: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.composition is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "bars": self.bars,
            "origin": self.origin,
            "attempts": self.attempts,
            **self.report,
        }


def build_header(
    vocab: Vocab, profile: NesCompositionProfile, *, key_offset: int = 0
) -> list[str]:
    """The composer's header for a profile, optionally transposed.

    Tokens the vocabulary lacks are dropped rather than guessed at, so a
    profile naming an unavailable key or meter degrades instead of failing.
    """
    tempo = min(vocab.tempos, key=lambda t: abs(t - profile.tempo_bpm))
    key = f"<KEY_{profile.key_root + key_offset}:{profile.key_mode}>"
    if key not in vocab.token_to_id:
        key = f"<KEY_{profile.key_root}:{profile.key_mode}>"
    meter = f"<METER_{profile.beats_per_bar}/4>"
    if meter not in vocab.token_to_id:
        meter = "<METER_4/4>"
    header = [
        "<BOS>",
        f"<GENRE_{profile.genre}>",
        f"<ENSEMBLE_{profile.ensemble}>",
        f"<TEMPO_{tempo}>",
        key,
        meter,
        "<PLAN>",
        *profile.plan_tokens(),
        "<PLAN_END>",
    ]
    return [t for t in header if t in vocab.token_to_id]


def bar_prefix(ids: list[int], *, vocab: Vocab, n_bars: int) -> list[int] | None:
    """Stream cut just before the token opening bar `n_bars + 1`, with no EOS.

    An implicit opening bar -- a stream that emits notes before its first
    `<BAR>` -- counts as the first, matching `decode`. The boundary token is
    withheld so the model's next move is its own.
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
                return ids[:index]
    return None


def has_implicit_first_bar(ids: list[int], *, vocab: Vocab) -> bool:
    bar_id = vocab.id_of("<BAR>")
    for token_id in ids:
        if token_id == bar_id:
            return False
        if vocab.id_to_token.get(token_id, "").startswith(("<PITCH_", "<DRUM_NOTE_")):
            return True
    return False


class SongBuilder:
    """Turns a profile into a rendered multi-section composition."""

    def __init__(
        self,
        *,
        vocab: Vocab,
        tokenizer: Any,
        engine: Any,
        generate: Any,
        role_for_program: dict[int, str] | None = None,
    ) -> None:
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.engine = engine
        #: `generate(prompt_ids, seed=..., bar_offset=..., n_bars=...)` ->
        #: token ids. Injected so this module never imports torch.
        self.generate = generate
        #: When set, the arranger honours these roles instead of inferring
        #: them from note activity. That inference overrode the composer's
        #: declared lead on 6 of 12 measured pieces.
        self.role_for_program = role_for_program
        self.bar_ticks = ticks_per_bar()

    def to_nes(self, ids: list[int], *, n_bars: int) -> tuple[Composition, dict[str, Any]] | None:
        trimmed = trim_to_bars(ids, vocab=self.vocab, n_bars=n_bars)
        if not (trimmed.complete and trimmed.n_bars == n_bars):
            return None
        score, decode_report = decode(trimmed.ids, vocab=self.vocab)
        arranged, arrange_report = arrange_for_nes(
            score,
            row_sec=ROW_SEC,
            divisions=self.tokenizer.config.onset_divisions,
            role_for_program=self.role_for_program,
        )
        arranged, _t = enforce_monophony(arranged)
        from rse.tokenizers.tokenizer import GridFit

        fit = GridFit(
            step_ticks=float(self.tokenizer.ticks_per_step(TICKS_PER_BEAT)),
            phase_ticks=0.0,
            aligned=True,
            period_sec=row_seconds(score.tempo_bpm),
        )
        rounded = enforce_monophony(
            self.tokenizer.decode(
                self.tokenizer.encode(arranged, fit=fit).ids
            ).composition
        )[0]
        final, _repair = self.engine.apply(rounded)
        events = final.sounding_events()
        if not events:
            return None
        ticks = [e.tick for e in events]
        if min(ticks) < 0 or max(ticks) >= n_bars * self.bar_ticks:
            return None
        return final, {
            "termination": trimmed.reason,
            "tempo_bpm": score.tempo_bpm,
            "n_events": len(events),
            "role_assignment": arrange_report.role_assignment,
            "arrangement": arrange_report.to_dict(),
            "decode": decode_report.to_dict(),
            "hardware_violations": len(self.engine.validate(final)),
        }

    def build_section(
        self,
        section: Section,
        *,
        profile: NesCompositionProfile,
        seed: int,
        prefixes: dict[str, list[int]],
        max_attempts: int = 6,
    ) -> SectionResult:
        """Generate one section, retrying on structural invalidity."""
        if section.source == "continue_from" and section.continue_from in prefixes:
            prompt_ids = prefixes[section.continue_from]
            origin = (
                f"continuation of section {section.continue_from}'s first "
                f"{profile.theme_bars} bars: those bars are that section's "
                "material, the rest is newly generated"
            )
            bar_offset = profile.theme_bars
        else:
            prompt_ids = [
                self.vocab.id_of(t)
                for t in build_header(self.vocab, profile, key_offset=section.key_offset)
            ]
            origin = "fresh generation from the profile header"
            if section.key_offset:
                origin += (
                    f", key moved by {section.key_offset} -- contrast comes from "
                    "altered conditioning, the composer was not asked for a "
                    "contrasting section"
                )
            bar_offset = 0

        result = SectionResult(label=section.label, bars=section.bars, origin=origin)
        for tries in range(max_attempts):
            ids = self.generate(
                prompt_ids, seed=seed + tries, n_bars=section.bars,
                bar_offset=1 if has_implicit_first_bar(prompt_ids, vocab=self.vocab) else 0,
            )
            if prompt_ids and ids[: len(prompt_ids)] != prompt_ids:
                continue
            built = self.to_nes(ids, n_bars=section.bars)
            if built is None:
                continue
            result.composition, result.report = built
            result.token_ids = ids
            result.attempts = tries + 1
            del bar_offset
            return result
        result.attempts = max_attempts
        return result

    def assemble(
        self,
        results: list[SectionResult],
        *,
        profile: NesCompositionProfile,
    ) -> tuple[Composition, dict[str, Any]]:
        """Lay sections on bar lines, repair the seams, mark the loop."""
        events = []
        offset_bars = 0
        placement = []
        for result in results:
            if not result.ok:
                continue
            offset = offset_bars * self.bar_ticks
            for event in result.composition.sounding_events():
                events.append(dataclasses.replace(event, tick=event.tick + offset))
            placement.append({"label": result.label, "start_bar": offset_bars})
            offset_bars += result.bars

        track = Composition(
            hardware_profile="nes", ticks_per_beat=TICKS_PER_BEAT, events=events
        )
        track.tempo_map = [
            TempoChange(0, int(round(60_000_000 / profile.tempo_bpm)))
        ]
        track.time_signatures = [
            TimeSignatureChange(0, profile.beats_per_bar, 4)
        ]
        track.normalize()

        loop_start = 0
        loop_end = offset_bars * self.bar_ticks

        # Nothing may ring past the loop point. A note that does gets cut
        # mid-sound when the track wraps, which is audible as a broken loop
        # rather than a seamless one -- the engine reports it as
        # `past_loop_end` and `apply()` does not repair it. Truncating to the
        # boundary is the musically correct answer: for the loop to join, the
        # last note has to end where the first one begins.
        clipped = 0
        trimmed_events = []
        for event in track.sounding_events():
            end = event.tick + getattr(event, "duration", 0)
            if end > loop_end and event.tick < loop_end:
                trimmed_events.append(
                    dataclasses.replace(event, duration=loop_end - event.tick)
                )
                clipped += 1
            elif event.tick < loop_end:
                trimmed_events.append(event)
            else:
                clipped += 1
        track.events = trimmed_events
        track.normalize()

        track, _truncated = enforce_monophony(track)
        track, repairs = self.engine.apply(track)
        track.loop_start_tick = loop_start
        track.loop_end_tick = loop_end
        violations = self.engine.validate(track)
        return track, {
            "placement": placement,
            "total_bars": offset_bars,
            "seam_repairs": len(repairs) if isinstance(repairs, list) else 0,
            "notes_clipped_at_loop": clipped,
            "hardware_violations": len(violations),
            # Detail, not just a count: a violation that `apply` did not
            # repair needs naming before anyone listens to the track.
            "violation_detail": [
                {
                    "kind": getattr(v, "kind", str(v)),
                    "channel": getattr(v, "channel_id", "?"),
                    "tick": getattr(v, "tick", -1),
                    "detail": getattr(v, "detail", ""),
                }
                for v in violations[:10]
            ],
            "loop": {"start_tick": loop_start, "end_tick": loop_end},
        }
