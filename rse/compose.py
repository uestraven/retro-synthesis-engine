"""One composition, start to finish.

The single implementation of "produce a piece and write it out", shared by the
CLI and the UI. Keeping one copy is not tidiness: the first time this project
had two hand-written copies of the same state machine, the sampler and the
decoder drifted apart, and nothing failed loudly.

Model output goes through the constraint engine before rendering. The
rule-based generator is compliant by construction, but a model has no such
guarantee, and the renderer should never be the thing that discovers it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rse.events.convert import composition_to_intake
from rse.exporters.midi import write_midi
from rse.exporters.wav import write_stems, write_wav
from rse.generation.loops import LoopReport, validate_loop
from rse.generation.plan import PlanSummary, make_plan
from rse.generation.procedural import generate as generate_procedural
from rse.hardware.constraints import ConstraintEngine, ValidationReport
from rse.hardware.profile import HardwareProfile, load_profile
from rse.jsonio import write_json
from rse.renderers.base import RenderSettings, notes_from_record
from rse.renderers.nes_apu import NesApuRenderer

#: Generation sources.
RULES = "rules"
MODEL = "model"
SOURCES = (RULES, MODEL)


@dataclass(frozen=True)
class ComposeRequest:
    """Everything needed to produce one piece."""

    source: str = RULES
    profile_id: str = "nes"
    # Musical intent
    mood: str = "heroic"
    energy: str = "medium"
    bars: int = 16
    tempo_bpm: float | None = None
    seed: int | None = None
    form: str | None = None
    tonic: int | None = None
    scale: str | None = None
    loop: bool = True
    # Model sampling
    checkpoint: Path | None = None
    temperature: float = 1.0
    top_k: int | None = 40
    top_p: float | None = 0.95
    constrain_to_grammar: bool = True
    # Rendering
    duty_p1: int | None = None
    duty_p2: int | None = None
    stems: bool = False

    def __post_init__(self) -> None:
        if self.source not in SOURCES:
            raise ValueError(f"source must be one of {list(SOURCES)}, got {self.source!r}")
        if self.source == MODEL and self.checkpoint is None:
            raise ValueError("the model source needs a checkpoint")
        if self.bars < 1:
            raise ValueError(f"bars must be positive, got {self.bars}")


@dataclass
class ComposeResult:
    """A finished piece and everything measured about it."""

    record: dict[str, Any]
    samples: Any
    sample_rate: int
    wav: Path | None = None
    midi: Path | None = None
    json: Path | None = None
    stems: dict[str, Path] = field(default_factory=dict)
    plan_lines: list[str] = field(default_factory=list)
    loop_report: LoopReport | None = None
    constraint_report: ValidationReport | None = None
    sample_info: dict[str, Any] | None = None
    renderer_id: str = ""
    warnings: list[str] = field(default_factory=list)

    @property
    def duration_sec(self) -> float:
        return len(self.samples) / self.sample_rate if self.sample_rate else 0.0

    @property
    def channels(self) -> list[dict[str, Any]]:
        return self.record.get("channels", [])

    @property
    def measurements(self) -> dict[str, Any]:
        return self.record.get("measurements", {})

    def summary(self) -> dict[str, Any]:
        return {
            "source": self.record.get("generation", {}).get("generator")
            or (self.sample_info or {}).get("checkpoint")
            or "unknown",
            "duration_sec": round(self.duration_sec, 2),
            "n_notes": self.measurements.get("n_notes", 0),
            "n_bars": self.measurements.get("n_bars", 0),
            "hardware_compliant": self.measurements.get("hardware_compliant"),
            "renderer_id": self.renderer_id,
            "loop": None if self.loop_report is None else self.loop_report.to_dict(),
            "constraints": (
                None if self.constraint_report is None else self.constraint_report.to_dict()
            ),
            "sample": self.sample_info,
            "files": {
                "wav": None if self.wav is None else str(self.wav),
                "midi": None if self.midi is None else str(self.midi),
                "json": None if self.json is None else str(self.json),
                "stems": {k: str(v) for k, v in self.stems.items()},
            },
            "warnings": self.warnings,
        }


def compose(
    request: ComposeRequest,
    *,
    out_base: Path | None = None,
    profile: HardwareProfile | None = None,
    configs_dir: Path | None = None,
) -> ComposeResult:
    """Produce a piece, validate it, render it, and optionally write it out."""
    profile = profile or load_profile(request.profile_id, configs_dir=configs_dir)

    if request.source == RULES:
        record, plan_lines = _from_rules(request, profile)
        sample_info = None
    else:
        record, sample_info = _from_model(request, profile)
        plan_lines = []

    # Both paths get checked. For the rule-based generator this is a no-op that
    # proves the claim; for the model it is load-bearing.
    engine = ConstraintEngine(profile)
    from rse.events.convert import composition_from_intake

    repaired, constraint_report = engine.apply(composition_from_intake(record))
    if not constraint_report.ok:
        record = composition_to_intake(repaired, profile=profile)

    renderer = NesApuRenderer(
        profile,
        duty_by_channel={
            cid: value
            for cid, value in (("P1", request.duty_p1), ("P2", request.duty_p2))
            if value is not None
        },
    )
    notes = notes_from_record(record)
    if not notes:
        raise ValueError("that composition contains no renderable notes")
    # A looping piece must not end in silence, or the repeat leaves a gap.
    rendered = renderer.render(notes, RenderSettings(tail_sec=0.0 if request.loop else 0.25))

    result = ComposeResult(
        record=record,
        samples=rendered.samples,
        sample_rate=rendered.sample_rate,
        plan_lines=plan_lines,
        constraint_report=constraint_report,
        sample_info=sample_info,
        renderer_id=rendered.renderer_id,
        warnings=list(rendered.warnings),
    )
    result.loop_report = validate_loop(
        record, samples=rendered.samples, sample_rate=rendered.sample_rate
    )

    if out_base is not None:
        _write(result, request, out_base, rendered)
    return result


def _from_rules(
    request: ComposeRequest, profile: HardwareProfile
) -> tuple[dict[str, Any], list[str]]:
    plan = make_plan(
        seed=request.seed,
        mood=request.mood,
        energy=request.energy,
        tempo_bpm=request.tempo_bpm,
        n_bars=request.bars,
        form=request.form,
        tonic=request.tonic,
        scale=request.scale,
        loop=request.loop,
    )
    composition = generate_procedural(profile, plan)
    return composition.record, PlanSummary(plan).lines()


def _from_model(
    request: ComposeRequest, profile: HardwareProfile
) -> tuple[dict[str, Any], dict[str, Any]]:
    # Imported here so the rule-based path needs no PyTorch.
    from rse.generation.sampler import SamplingConfig, sample
    from rse.tokenizers.tokenizer import Conditioning
    from rse.training.train import load_checkpoint

    model, tokenizer, payload = load_checkpoint(request.checkpoint)
    tempo = request.tempo_bpm
    if tempo is None:
        tempo = payload.get("report", {}).get("conditioning_tempo_bpm") or 120.0

    result = sample(
        model,
        tokenizer,
        conditioning=Conditioning(
            mood=request.mood,
            energy=request.energy,
            tempo_bpm=tempo,
            n_bars=request.bars,
        ),
        config=SamplingConfig(
            seed=request.seed,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            constrain_to_grammar=request.constrain_to_grammar,
        ),
        loop=request.loop,
    )
    info = {**result.to_dict(), "checkpoint": str(request.checkpoint)}
    return composition_to_intake(result.composition, profile=profile), info


def _write(result: ComposeResult, request: ComposeRequest, out_base: Path, rendered: Any) -> None:
    out_base = Path(out_base)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    result.wav = write_wav(out_base.with_suffix(".wav"), result.samples, result.sample_rate)
    result.midi = write_midi(out_base.with_suffix(".mid"), result.record)
    result.json = write_json(
        out_base.with_suffix(".json"), {**result.record, "compose": result.summary()}
    )
    if request.stems:
        result.stems = write_stems(
            out_base.parent / f"{out_base.name}_stems", rendered.stems, result.sample_rate
        )
