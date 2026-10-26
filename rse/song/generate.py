"""One procedural generation path, for every caller.

The CLI and the web API must not each own a copy of this. That exact mistake
has already cost this project once: a silence fix was applied to
`rse/song/build.py` while `prompt_to_song` kept its own `concatenate()`, so the
bug stayed in the shipped path and the tests passed. A second front end is
precisely when that happens again, so the generation path moves here first and
both callers import it.

Nothing in here knows about HTTP, argparse, or files on disk beyond the ones it
is handed. `compose_song` produces the composition and its report;
`write_artifacts` turns that into playable files. A caller that wants only the
score never renders audio.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rse.analysis.restatement import ticks_per_bar
from rse.events.schema import Composition
from rse.hardware.constraints import ConstraintEngine
from rse.hardware.profile import load_profile
from rse.hardware.tempo import snap as snap_tempo
from rse.prompt.interpret import Requirement, SongPlan, interpret
from rse.song.audio import PEAK_DBFS, normalise_peak, render_window
from rse.song.loopcheck import check as loop_check
from rse.song.loopcheck import pad_to_loop_length

SECTION_BARS = 8

#: How far the finished track may sit from the requested duration.
#:
#: Removing the dead air between sections also removes real time, so a form of
#: whole sections rarely lands on the requested second. The operator's call: a
#: valid loop and an intact section form matter more than the exact length,
#: within ten seconds either way.
DURATION_TOLERANCE_SEC = 10.0


@dataclass
class SongResult:
    """A finished composition and everything that can be said about it."""

    composition: Composition
    plan: SongPlan
    meta: dict[str, Any]
    violations: list[Any] = field(default_factory=list)

    @property
    def duration_sec(self) -> float:
        return float(self.meta["duration_sec"])

    @property
    def within_tolerance(self) -> bool:
        return (
            abs(self.duration_sec - self.plan.target_seconds)
            <= DURATION_TOLERANCE_SEC
        )


def _restate_form_for_procedural(plan: SongPlan, sections: list[Any]) -> None:
    """Credit the form features the composer actually built.

    The interpreter calls a contrasting section and a thematic return "built
    deterministically by the assembler, not requested of the composer", which
    described the Fermata pipeline: an assembler stitched eight-bar generations
    into a shape the model knew nothing about. The procedural composer plans
    its own form -- `plan_form` gives B its own progression and its own motif,
    and the final section restates the opening one -- so for this engine the
    claim is backwards.

    Checked against the sections that exist rather than assumed: a short
    duration yields A-A'' with no B at all, and a request for a contrasting
    middle section is then genuinely unmet.
    """
    treatments = {section.treatment for section in sections}
    labels = [section.label for section in sections]
    has_contrast = "contrast" in treatments
    # A return means a later section restating the opening material.
    has_return = len(labels) > 2 and labels[-1].startswith("A")

    for requirement in plan.requirements:
        if "built deterministically by the assembler" not in requirement.detail:
            continue
        feature = requirement.detail.split(" built", 1)[0]
        if feature == "contrast_section":
            if has_contrast:
                requirement.status = "satisfied"
                requirement.detail = (
                    "section B, composed on its own chord progression and its "
                    "own motif rather than a variation of the opening"
                )
            else:
                requirement.status = "unsupported"
                requirement.detail = (
                    f"the form came out as {'-'.join(labels)}, which has no "
                    "contrasting section; a longer duration leaves room for one"
                )
        elif feature == "theme_return":
            if has_return:
                requirement.status = "satisfied"
                requirement.detail = (
                    f"the final section ({labels[-1]}) restates the opening "
                    "theme over the opening progression"
                )
            else:
                requirement.status = "unsupported"
                requirement.detail = (
                    f"the form came out as {'-'.join(labels)}, too short to "
                    "state a theme, leave it and return to it"
                )
        elif feature == "loop":
            requirement.status = "satisfied"
            requirement.detail = (
                "loop points set on the bar line, and the boundary is checked "
                "for a timing or level discontinuity"
            )


def _no_model_wording(plan: SongPlan) -> None:
    """There is no model on this path, so the report must not mention one."""
    for requirement in plan.requirements:
        requirement.detail = requirement.detail.replace(
            "nothing in the model takes it as input",
            "nothing in the composer takes it as input",
        )


def _restate_for_procedural(plan: SongPlan, selection: Any) -> None:
    """Re-explain the requirements whose explanation is Fermata's, not ours.

    The interpreter's detail strings were written when Fermata was the only
    engine, and they say so: a scenario is "recorded as scenario 'boss_battle';
    the composer has no scenario input, so it reaches generation only through
    genre, key and tempo", and a style is "expressed to the composer as a genre
    token". Both were true then and neither is true now -- the scenario is
    precisely what picks the procedural style, and there is no genre token
    anywhere in this engine.

    Left alone, the generator page shows a user an accurate-sounding
    explanation of a mechanism that is not running. That is worse than saying
    nothing, so the procedural path restates its own. The interpreter is not
    touched: it is shared with the legacy path and has its own pinned tests.
    """
    scenario = getattr(plan, "scenario", None)
    for requirement in plan.requirements:
        detail = requirement.detail
        if "has no scenario input" in detail:
            if selection.matched_on == "scenario":
                requirement.status = "satisfied"
                requirement.detail = (
                    f"selected the {selection.style!r} composer preset, which "
                    "sets the mode, tempo, chord progressions and percussion "
                    "for this kind of scene"
                )
            else:
                requirement.detail = (
                    f"recorded as scenario {scenario!r}; it did not pick the "
                    f"preset because {selection.reason} took precedence"
                )
        elif "as a genre token and to the arranger" in detail:
            requirement.detail = (
                "the NES hardware profile: two pulse channels, triangle bass "
                "and noise percussion, enforced on every note"
            )
        elif detail.startswith("genre token "):
            # There is no genre conditioning in this engine at all.
            requirement.status = "unsupported"
            requirement.detail = (
                "the procedural composer has no genre input; the style preset "
                "is what decides the musical character"
            )


def compose_song(
    plan: SongPlan,
    *,
    seed: int,
    section_bars: int = SECTION_BARS,
    style: str | None = None,
    tempo_bpm: int | None = None,
    tempo_from_prompt: bool = True,
) -> SongResult:
    """Interpreted prompt in, complete NES composition out.

    Imports no ML dependency and loads no weights: the procedural composer is
    the whole engine. The assembler the Fermata path needs must NOT run here --
    this composer already produces its own section form, thematic return and
    loop points, and laying a second form over one that exists is what put a
    senseless transition at 0:45 the first time.

    `style` overrides selection for a caller that offers a style control. It is
    recorded as the reason, so a chosen style and an inferred one stay
    distinguishable in the report.

    `tempo_from_prompt` says whether `plan.energy` was set by the prompt or by
    a caller's control. It matters because `select()` reads a non-empty energy
    as proof that the prompt's tempo was requested, which is true of the
    interpreter and false of an energy slider. Defaults to True so existing
    callers, which only ever pass an interpreted plan, behave as before.

    `tempo_bpm` bypasses selection entirely, and must. `select()` discards a
    tempo unless the prompt also carried an energy word or a literal "bpm",
    which is right for a tempo the interpreter *guessed* from a mood -- an
    inferred 170 should not override a preset's own 150 -- and wrong for one a
    caller *asked* for. The two were indistinguishable once they both sat in
    `plan.tempo_bpm`, so a tempo slider set to 90 produced a 150 BPM song and
    a report claiming "the prompt named no tempo". Selection infers; an
    explicit control is not inference, so it does not go through it.
    """
    from rse.procedural.compose import STYLES, compose
    from rse.procedural.select import select

    selection = select(plan)
    if style is not None:
        if style not in STYLES:
            raise ValueError(
                f"unknown style {style!r}; have {sorted(STYLES)}"
            )
        if style != selection.style:
            selection.notes.append(
                f"style {style!r} chosen directly, overriding "
                f"{selection.style!r} from {selection.reason}"
            )
        selection.style = style
        selection.reason = "chosen directly"
        selection.matched_on = "explicit"
    selection.section_bars = section_bars

    if not tempo_from_prompt and tempo_bpm is None and selection.tempo_bpm:
        # `select()` applies the prompt's tempo only when the prompt carried an
        # energy word, on the reasoning that such a tempo was asked for. That
        # reasoning is sound and this caller broke it: setting `plan.energy`
        # from an energy SLIDER forges the signal, so the interpreter's
        # unrequested default of 120 BPM looked like a request and overrode
        # every preset. A boss theme asked for at "style default" played
        # 112.69 BPM instead of 150.25 -- neither the preset nor the 120.
        selection.notes.append(
            f"kept the {selection.style!r} preset tempo; the energy control "
            "chose the style, and no tempo was asked for"
        )
        selection.tempo_bpm = None

    chosen_tempo = selection.tempo_bpm
    if tempo_bpm is not None:
        chosen_tempo = int(tempo_bpm)
        selection.tempo_bpm = chosen_tempo
        # Drop the inference note, which would otherwise contradict the music.
        selection.notes = [
            note for note in selection.notes if "kept the preset tempo" not in note
        ]

    # One place reports the snap, whichever route the tempo arrived by. It was
    # only reported on the caller's route at first, so a tempo the PROMPT asked
    # for -- "a battle theme at 200 bpm" -- was snapped to 180.30 in silence.
    # Any tempo not on the frame grid is changed, so every route has to say so.
    if chosen_tempo:
        preset = STYLES[selection.style].tempo_bpm
        played = snap_tempo(chosen_tempo)
        if abs(played.bpm - chosen_tempo) > 0.01:
            selection.notes.append(
                f"tempo {chosen_tempo} BPM is not playable on NES hardware; "
                f"snapped to {played.bpm:.2f} BPM "
                f"({played.frames_per_row} frames per tracker row)"
            )
        if abs(played.bpm - preset) > 0.01:
            selection.notes.append(
                f"tempo {played.bpm:.2f} BPM overrides the "
                f"{selection.style!r} preset's {preset:.2f} BPM"
            )

    _restate_for_procedural(plan, selection)

    result = compose(
        style_name=selection.style,
        seed=seed,
        target_seconds=plan.target_seconds,
        section_bars=section_bars,
        tempo_bpm=chosen_tempo,
        tonic=selection.tonic,
    )
    _restate_form_for_procedural(plan, result.sections)
    _no_model_wording(plan)

    composition = result.composition
    end = composition.loop_end_tick or composition.end_tick or 0

    # Validation, not repair. The procedural composer writes into the four NES
    # channels directly, so a violation here would be a real defect rather
    # than something to paper over.
    violations = ConstraintEngine(load_profile("nes")).validate(composition)

    meta = {
        "engine": "procedural",
        "selection": selection.to_dict(),
        "plan": result.to_dict(),
        "form": [section.label for section in result.sections],
        "total_bars": end // ticks_per_bar(),
        "duration_sec": round(composition.time.seconds(end), 3),
        "tempo_bpm": result.style.tempo_bpm,
        "frames_per_row": result.style.frames_per_row,
        "sound": result.sound.to_dict() if result.sound else None,
        "duty_by_channel": (
            result.sound.duty_by_channel if result.sound else None
        ),
        "row_ms": round(result.style.row_sec * 1000, 3),
        "mode": result.style.mode,
        "tonic": result.style.tonic,
        "sections": [
            {
                "label": section.label,
                "bars": section.bars,
                "progression": [
                    spec if isinstance(spec, int) else list(spec)
                    for spec in section.progression
                ],
                "treatment": section.treatment,
            }
            for section in result.sections
        ],
        "unsupported": selection.unsupported,
    }
    return SongResult(
        composition=composition, plan=plan, meta=meta, violations=list(violations)
    )


def generate_song(
    prompt: str,
    *,
    seed: int,
    style: str | None = None,
    section_bars: int = SECTION_BARS,
    target_seconds: float | None = None,
    tempo_bpm: int | None = None,
    loop: bool | None = None,
    energy: str | None = None,
) -> SongResult:
    """The whole path from English to composition, overrides included.

    Each override is recorded as a requirement so the report distinguishes what
    the prompt asked for from what a control imposed.

    `energy` is set on the plan BEFORE selection, which is the only point at
    which it can do anything: it is what `select()` falls back to when the
    prompt named no scenario or mood. Recording it afterwards would have left
    an energy control that moved nothing -- and a slider the interface shows
    but the composer ignores is worse than no slider.

    `tempo_bpm` is handed to `compose_song` directly rather than written onto
    the plan, for the reason spelled out there: on the plan it becomes
    indistinguishable from a tempo the interpreter guessed, and selection
    throws guessed tempos away.
    """
    plan = interpret(prompt)
    # Read before anything overrides it. The prompt asked for a tempo if it
    # carried an energy word (which sets `energy`) OR named a literal BPM
    # (which does not) -- the same two signals `select()` itself uses. Testing
    # only for energy dropped "a battle theme at 200 bpm" on the floor.
    tempo_from_prompt = bool(plan.energy) or any(
        requirement.text.lower().endswith("bpm")
        for requirement in plan.requirements
    )
    for name, value, incoming in (
        ("target_seconds", plan.target_seconds, target_seconds),
        ("loop", plan.loop, loop),
        ("energy", plan.energy, energy),
    ):
        if incoming is None or incoming == value:
            continue
        setattr(plan, name, incoming)
        plan.requirements.append(
            Requirement(
                f"{name.replace('_', ' ')} from the prompt",
                "approximated",
                f"overridden by the caller: {value} -> {incoming}",
            )
        )
    if tempo_bpm is not None and tempo_bpm != plan.tempo_bpm:
        played = snap_tempo(tempo_bpm)
        exact = abs(played.bpm - tempo_bpm) <= 0.01
        plan.requirements.append(
            Requirement(
                f"tempo {tempo_bpm} BPM",
                "satisfied" if exact else "approximated",
                "requested directly; the composer took it"
                if exact
                else f"the NES advances a tracker row on a whole video frame, "
                f"so {tempo_bpm} BPM is not playable; the nearest that is, "
                f"{played.bpm:.2f} BPM, was used",
            )
        )
    return compose_song(
        plan,
        seed=seed,
        section_bars=section_bars,
        style=style,
        tempo_bpm=tempo_bpm,
        tempo_from_prompt=tempo_from_prompt,
    )


def finish_report(result: SongResult) -> dict[str, Any]:
    """The requirement buckets, with the two facts only the result knows.

    Appended here rather than inside `compose_song` so composing twice does not
    grow the list twice.
    """
    duration = result.duration_sec
    within = result.within_tolerance
    plan = result.plan
    already = {requirement.text for requirement in plan.requirements}
    duration_text = (
        f"duration {duration:.1f} s against {plan.target_seconds:g} s requested"
    )
    form_text = f"{len(result.meta['form'])} sections, {'-'.join(result.meta['form'])}"
    if duration_text not in already:
        plan.requirements.append(
            Requirement(
                duration_text,
                "satisfied" if within else "approximated",
                f"within the +/-{DURATION_TOLERANCE_SEC:g} s tolerance"
                if within
                else "outside tolerance; the section form was kept whole",
            )
        )
    if form_text not in already:
        plan.requirements.append(
            Requirement(
                form_text,
                "imposed",
                "section form chosen by the procedural composer",
            )
        )
    return plan.report()


def write_artifacts(
    result: SongResult,
    *,
    out_dir: Path,
    name: str,
    seed: int | None = None,
    peak_dbfs: float = PEAK_DBFS,
) -> dict[str, Any]:
    """Render, level-match, export, and check the loop. Returns the payload.

    `seed` is written into the report rather than attached by the caller
    afterwards. The caller used to do it, which meant the JSON file on disk --
    the one a listener downloads to reproduce a song they liked -- was written
    before the seed was added and therefore did not contain it.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    profile = load_profile("nes")
    duration = result.duration_sec

    wav = out_dir / f"{name}.wav"
    render_window(
        result.composition, wav, duty_by_channel=result.meta.get("duty_by_channel")
    )
    padding = pad_to_loop_length(wav, loop_seconds=duration)
    normalise_peak(wav, dbfs=peak_dbfs)
    diagnostic = loop_check(
        result.composition, bar_ticks=ticks_per_bar(), wav_path=wav
    )

    midi_path: Path | None = out_dir / f"{name}.mid"
    midi_error: str | None = None
    try:
        from rse.events.convert import composition_to_intake
        from rse.exporters.midi import record_to_midi

        record_to_midi(
            composition_to_intake(result.composition, profile=profile)
        ).save(str(midi_path))
    except Exception as error:  # noqa: BLE001 - reported, never fatal
        midi_error = f"{type(error).__name__}: {error}"
        midi_path = None

    # Before the SongSpec is serialised, not after: the duration and form
    # requirements are appended to the plan, and the recorded reports have
    # always carried them in both places. Reordering this would silently
    # change every artefact already on disk.
    report = finish_report(result)
    payload = {
        "version": "rse-prompt-to-song/0.2",
        "engine": "procedural",
        "prompt": result.plan.prompt,
        "seed": seed,
        "songspec": result.plan.to_dict(),
        "composition": result.meta,
        "duration_sec": duration,
        "requested_seconds": result.plan.target_seconds,
        "duration_tolerance_sec": DURATION_TOLERANCE_SEC,
        "duration_within_tolerance": result.within_tolerance,
        "hardware_violations": len(result.violations),
        "loop_diagnostic": diagnostic.to_dict(),
        "render_padding": padding,
        "ogg": {
            "exported": False,
            "reason": "no Vorbis encoder available: the renderer writes WAV "
            "only, and neither ffmpeg nor oggenc is installed",
        },
        "requirement_report": report,
        "files": {
            "wav": str(wav),
            "midi": str(midi_path) if midi_path else None,
        },
    }
    if midi_error:
        payload["midi_error"] = midi_error
    (out_dir / f"{name}.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload
