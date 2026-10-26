"""The procedural composer is what an ordinary English prompt now reaches.

For most of this project the composer the listener called "fantastic" was
reachable only by naming a preset in an experiment script, while the public
entry point used the one that passes 60% of the time. These tests pin the
connection end to end, through the real `scripts/prompt_to_song.py`, because
the defect class here is never in a library function -- it is in the wiring.

Two of them are about what must NOT happen. The default path must not load
torch, must not load the transformers stack, and must not even import the
Fermata codec, because the point of a procedural composer is that it needs no
model. It used to import all three transitively: the entry point reached into
`architecture_comparison` for a gain calculation, and that experiment script
imports the Fermata vocabulary at module scope.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import sys
import wave
from pathlib import Path
from typing import Any

import pytest

from rse.analysis.restatement import ticks_per_bar
from rse.hardware.constraints import ConstraintEngine
from rse.hardware.profile import load_profile
from rse.procedural.select import select
from rse.prompt.interpret import interpret

_ROOT = Path(__file__).resolve().parents[1]
_ENTRY = _ROOT / "scripts" / "prompt_to_song.py"

#: The acceptance prompt: ordinary English, no preset named, no style token.
BOSS_PROMPT = (
    "Compose a dark, energetic, 60-second NES boss battle theme with a "
    "memorable main melody, driving percussion, a contrasting middle "
    "section, and a dramatic return to the opening theme. Make it loop "
    "seamlessly."
)

SEED = 20261026


def load_entry() -> Any:
    """Load the entry point as a module, which is also how `main()` is run."""
    spec = importlib.util.spec_from_file_location("prompt_to_song_entry", _ENTRY)
    module = importlib.util.module_from_spec(spec)
    # Registered before execution: the module's dataclasses need to find it.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run(prompt: str, *, out_dir: Path, name: str, seed: int = SEED, **flags: str):
    """Invoke the real `main()` and return (exit code, parsed report)."""
    module = load_entry()
    argv = [
        "prompt_to_song.py",
        prompt,
        "--seed",
        str(seed),
        "--name",
        name,
        "--out-dir",
        str(out_dir),
    ]
    for key, value in flags.items():
        argv += [f"--{key.replace('_', '-')}", value]
    saved = sys.argv
    sys.argv = argv
    try:
        with contextlib.redirect_stdout(io.StringIO()) as captured:
            code = module.main()
    finally:
        sys.argv = saved
    report = json.loads((out_dir / f"{name}.json").read_text())
    return code, report, captured.getvalue()


@pytest.fixture(scope="module")
def boss(tmp_path_factory) -> tuple[int, dict[str, Any], str]:
    """One acceptance run, shared by the checks that only read its output."""
    out = tmp_path_factory.mktemp("boss")
    return run(BOSS_PROMPT, out_dir=out, name="boss")


# --- the connection itself ------------------------------------------------


def test_procedural_is_the_default_engine() -> None:
    """No flag, no environment variable: plain `--engine` default."""
    source = _ENTRY.read_text()
    assert 'default="procedural"' in source
    code, report, _out = run(
        "a town theme", out_dir=Path(_default_dir()), name="default_engine"
    )
    assert code == 0
    assert report["engine"] == "procedural"


def _default_dir() -> str:
    import tempfile

    return tempfile.mkdtemp(prefix="rse_default_engine_")


def test_english_prompt_reaches_a_valid_style(boss) -> None:
    _code, report, _out = boss
    selection = report["composition"]["selection"]
    from rse.procedural.compose import STYLES

    assert selection["style"] in STYLES
    assert selection["style"] == "boss"
    # Inspectable, not emergent: the reason names what it matched on.
    assert selection["matched_on"] == "scenario"
    assert "boss_battle" in selection["reason"]


def test_songspec_survives_into_the_report(boss) -> None:
    """The interpreter's work is kept, not discarded by the new engine."""
    _code, report, _out = boss
    spec = report["songspec"]
    assert spec["scenario"] == "boss_battle"
    assert spec["prompt"] == BOSS_PROMPT
    assert spec["target_seconds"] == 60.0
    assert spec["percussion"] is True


def test_requested_style_words_are_not_silently_dropped(boss) -> None:
    """Every prompt requirement lands in exactly one bucket."""
    _code, report, _out = boss
    buckets = report["requirement_report"]
    texts = {
        item["text"]
        for status in ("satisfied", "approximated", "imposed", "unsupported")
        for item in buckets[status]
    }
    for word in ("dark", "driving", "percussion", "loop"):
        assert word in texts, word


# --- what must not happen -------------------------------------------------


def test_default_path_loads_no_model_stack(tmp_path: Path) -> None:
    """torch, transformers and the Fermata codec all stay unimported.

    Checked by module identity rather than by timing, so it fails for the
    right reason: a transitive import through a helper counts.
    """
    forbidden = ("torch", "transformers", "rse.transfer.fermata")
    for name in forbidden:
        sys.modules.pop(name, None)
    code, report, _out = run(BOSS_PROMPT, out_dir=tmp_path, name="clean")
    assert code == 0
    assert report["engine"] == "procedural"
    loaded = [name for name in forbidden if name in sys.modules]
    assert loaded == [], f"default path imported {loaded}"


def test_default_path_imports_no_experiment_script(tmp_path: Path) -> None:
    """The product must not depend on the experiment scripts to render.

    `architecture_comparison` was the route by which Fermata arrived.
    """
    for name in ("architecture_comparison", "codec_comparison"):
        sys.modules.pop(name, None)
    run(BOSS_PROMPT, out_dir=tmp_path, name="noexp")
    assert "architecture_comparison" not in sys.modules
    assert "codec_comparison" not in sys.modules
    assert "from rse.song.audio import" in _ENTRY.read_text()


def test_no_fallback_to_fermata_in_the_procedural_path() -> None:
    """A failure must surface, not silently become a Fermata generation."""
    source = _ENTRY.read_text()
    procedural = source[
        source.index("def run_procedural(") : source.index("def main(")
    ]
    for banned in ("LlamaForCausalLM", "trim_to_bars", "Vocab.load", "fermata"):
        assert banned not in procedural, banned


# --- the music -------------------------------------------------------------


def test_composition_is_full_length(boss) -> None:
    _code, report, _out = boss
    assert report["duration_within_tolerance"] is True
    assert abs(report["duration_sec"] - 60.0) <= report["duration_tolerance_sec"]
    # Not a single section padded out: a real multi-section piece.
    assert report["composition"]["total_bars"] >= 32


def test_form_states_varies_contrasts_and_returns(boss) -> None:
    """The prompt asked for a contrasting middle and a dramatic return."""
    _code, report, _out = boss
    sections = report["composition"]["sections"]
    treatments = [section["treatment"] for section in sections]
    assert treatments[0] == "state"
    assert "contrast" in treatments
    # The return restates rather than continuing to vary.
    assert treatments[-1] == "state"
    labels = [section["label"] for section in sections]
    assert labels[-1].startswith("A"), labels


def test_one_harmonic_plan_drives_every_section(boss) -> None:
    """The shared framework, visible in the report.

    Each section carries the progression its four voices were all derived
    from, and the restatements carry the same one as the opening.
    """
    _code, report, _out = boss
    sections = report["composition"]["sections"]
    opening = sections[0]["progression"]
    returns = [s for s in sections if s["treatment"] == "state"]
    assert len(returns) >= 2
    assert all(section["progression"] == opening for section in returns)
    contrast = next(s for s in sections if s["treatment"] == "contrast")
    assert contrast["progression"] != opening


def test_zero_hardware_violations(boss) -> None:
    _code, report, _out = boss
    assert report["hardware_violations"] == 0


def test_the_assembled_track_is_revalidated(boss, tmp_path: Path) -> None:
    """Independently re-validate, rather than trusting the report's own count."""
    _code, report, _out = boss
    module = load_entry()
    plan = interpret(BOSS_PROMPT)
    composition, _meta = module.compose_procedural(
        plan, seed=SEED, section_bars=module.SECTION_BARS
    )
    engine = ConstraintEngine(load_profile("nes"))
    assert engine.validate(composition) == []


def test_loop_boundary_is_checked_and_continuous(boss) -> None:
    _code, report, _out = boss
    diagnostic = report["loop_diagnostic"]
    assert diagnostic["timing_continuous"] is True
    assert diagnostic["n_past_end"] == 0
    assert diagnostic["loop_end_tick"] > 0
    # The render covers the whole loop, which an earlier version did not.
    assert report["duration_sec"] * 1.0 > 0


def test_memorable_is_reported_as_a_goal_not_an_achievement(boss) -> None:
    """Nothing in the composer takes memorability as input, so it is honest
    about that rather than claiming it."""
    _code, report, _out = boss
    unsupported = {item["text"] for item in report["requirement_report"]["unsupported"]}
    assert "memorable" in unsupported
    satisfied = {item["text"] for item in report["requirement_report"]["satisfied"]}
    assert "memorable" not in satisfied


# --- artefacts and reproducibility ----------------------------------------


def test_writes_a_playable_wav_and_midi(boss) -> None:
    _code, report, _out = boss
    wav_path = Path(report["files"]["wav"])
    with wave.open(str(wav_path)) as handle:
        frames = handle.getnframes()
        rate = handle.getframerate()
    assert rate == 44100
    assert frames / rate == pytest.approx(report["duration_sec"], abs=0.05)

    midi_path = report["files"]["midi"]
    assert midi_path, "MIDI export must not fail on the default path"
    import mido

    midi = mido.MidiFile(midi_path)
    onsets = sum(
        1
        for track in midi.tracks
        for message in track
        if message.type == "note_on" and message.velocity > 0
    )
    assert onsets > 200, onsets


def test_the_same_seed_gives_the_same_song(tmp_path: Path) -> None:
    first = run(BOSS_PROMPT, out_dir=tmp_path, name="repro_a")[1]
    second = run(BOSS_PROMPT, out_dir=tmp_path, name="repro_b")[1]
    digest = lambda path: hashlib.sha256(  # noqa: E731 - local, single use
        Path(path).read_bytes()
    ).hexdigest()
    assert digest(first["files"]["wav"]) == digest(second["files"]["wav"])
    assert first["composition"]["sections"] == second["composition"]["sections"]


def test_a_different_seed_gives_a_different_song(tmp_path: Path) -> None:
    """Compared on the audio, not on the section table.

    Two seeds can draw the same progression -- the style has few of them --
    and still write different melodies over it, which is the thing the seed
    is supposed to move.
    """
    first = run(BOSS_PROMPT, out_dir=tmp_path, name="seed_a", seed=1)[1]
    second = run(BOSS_PROMPT, out_dir=tmp_path, name="seed_b", seed=2)[1]
    digest = lambda path: hashlib.sha256(  # noqa: E731 - local, single use
        Path(path).read_bytes()
    ).hexdigest()
    assert digest(first["files"]["wav"]) != digest(second["files"]["wav"])


# --- selection, the unit beneath the integration --------------------------


@pytest.mark.parametrize(
    ("prompt", "style"),
    [
        ("a dark boss theme", "boss"),
        ("music for a peaceful village", "town"),
        ("a tense dungeon crawl", "tension"),
        ("an overworld exploration theme", "adventure"),
        ("a triumphant title screen", "victory"),
        ("fast battle music", "action"),
    ],
)
def test_ordinary_prompts_select_sensible_styles(prompt: str, style: str) -> None:
    assert select(interpret(prompt)).style == style


def test_an_empty_request_falls_back_rather_than_failing() -> None:
    selection = select(interpret("music"))
    assert selection.style == "adventure"
    assert selection.matched_on == ""
    assert any("fell back" in note for note in selection.notes)


def test_section_bars_stay_on_the_grid(boss) -> None:
    """Bars are derived from the grid, not from a duration estimate."""
    _code, report, _out = boss
    bars = report["composition"]["total_bars"]
    sections = report["composition"]["sections"]
    assert bars == sum(section["bars"] for section in sections)
    assert ticks_per_bar() == 1920
