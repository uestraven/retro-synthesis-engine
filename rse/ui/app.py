"""The local Gradio interface (proposal 10).

A client of `rse.compose`, which the CLI also uses — the UI adds no musical
logic of its own, so the two cannot disagree about what a composition is.

Networking
----------
The server binds **127.0.0.1** and nothing else. This is a personal music tool
whose audience is the person running it, so loopback is both the private
default and the correct one, and no authentication is needed because nothing
off the machine can reach it. Gradio's `share=True` tunnel is never used: it
publishes a public URL. `--host` exists for the case where someone genuinely
needs remote access, and it refuses to bind a non-loopback address without an
explicit opt-in, because that turns a local tool into an open one.
"""

from __future__ import annotations

import ipaddress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rse import paths
from rse.compose import MODEL, RULES, ComposeRequest, ComposeResult, compose
from rse.generation.plan import ENERGIES, LENGTHS, MOODS
from rse.hardware.profile import HardwareProfile, available_profiles, load_profile

#: Private by default. Not a placeholder — the deliverable's default.
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 7860

#: Where generated files land, and the only directory the server will serve.
DEFAULT_OUTPUT_DIR = paths.REPO_ROOT / "out" / "ui"


class UiError(RuntimeError):
    """Raised for a configuration the interface will not start with."""


def is_loopback(host: str) -> bool:
    """Whether a host string names only this machine."""
    if host in ("localhost",):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def check_bind(host: str, *, allow_remote: bool = False) -> str:
    """Refuse a non-loopback bind unless it was asked for explicitly."""
    if is_loopback(host) or allow_remote:
        return host
    raise UiError(
        f"refusing to bind {host!r}: that would let anything able to reach this "
        "machine use the interface and read its generated files. Leave --host "
        f"at {DEFAULT_HOST}, or forward a port over SSH, or pass --allow-remote "
        "if exposing it is what you intend."
    )


@dataclass
class UiConfig:
    """How the interface runs."""

    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    output_dir: Path = DEFAULT_OUTPUT_DIR
    checkpoints_dir: Path = paths.CHECKPOINTS_DIR
    configs_dir: Path | None = None
    profile_id: str = "nes"
    allow_remote: bool = False
    open_browser: bool = False


def find_checkpoints(directory: Path) -> list[str]:
    """Checkpoints available to sample from, newest first."""
    if not directory.is_dir():
        return []
    found = sorted(directory.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [str(p) for p in found]


def describe(result: ComposeResult) -> str:
    """Markdown summary of what was produced."""
    measurements = result.measurements
    lines = [
        f"**{measurements.get('n_notes', 0)} notes · "
        f"{measurements.get('n_bars', 0)} bars · {result.duration_sec:.1f}s**",
        "",
    ]
    if result.plan_lines:
        lines += ["```", *result.plan_lines, "```"]

    compliant = measurements.get("hardware_compliant")
    lines.append(f"- Hardware: {'playable as written' if compliant else 'see repairs below'}")
    report = result.constraint_report
    if report is not None and not report.ok:
        lines.append(f"- Repairs applied: `{report.counts()}`")
    loop = result.loop_report
    if loop is not None:
        lines.append(f"- Loop: {'clean' if loop.ok else '; '.join(loop.issues)}")
    if result.sample_info:
        info = result.sample_info
        lines.append(f"- Sampled {info.get('n_tokens', 0)} tokens from the model")
        if info.get("hit_token_limit"):
            lines.append("- Model hit the token limit without ending on its own")
        if info.get("overran_length"):
            lines.append("- Model tried to exceed the requested length and was held to it")
    for warning in result.warnings[:5]:
        lines.append(f"- {warning}")
    return "\n".join(lines)


def channel_rows(result: ComposeResult) -> list[list[Any]]:
    """Per-channel table rows."""
    return [
        [
            channel["channel_id"],
            channel["role"],
            channel["n_notes"],
            f"{channel['value_min']}-{channel['value_max']}"
            if channel["value_min"] is not None
            else "-",
            channel["max_simultaneous"],
            channel["overlap_count"],
        ]
        for channel in result.channels
    ]


def build_app(config: UiConfig | None = None, profile: HardwareProfile | None = None):
    """Construct the Gradio interface."""
    try:
        import gradio as gr
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise UiError(
            'the interface needs Gradio; install it with `pip install -e ".[ui]"`'
        ) from exc

    config = config or UiConfig()
    profile = profile or load_profile(config.profile_id, configs_dir=config.configs_dir)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = find_checkpoints(config.checkpoints_dir)

    def run(
        source_label: str,
        checkpoint: str | None,
        mood: str,
        energy: str,
        bars: int,
        tempo: float,
        seed: float | None,
        use_seed: bool,
        temperature: float,
        top_k: int,
        stems: bool,
    ):
        source = MODEL if source_label.startswith("Model") else RULES
        if source == MODEL and not checkpoint:
            raise gr.Error(
                "No checkpoint selected. Train one with `rse train`, or choose "
                "the rule-based composer."
            )
        request = ComposeRequest(
            source=source,
            profile_id=profile.profile_id,
            mood=mood,
            energy=energy,
            bars=int(bars),
            # The model knows the tempo it was trained on; overriding it from a
            # slider is how sampled audio ends up at the wrong speed.
            tempo_bpm=float(tempo) if source == RULES else None,
            seed=int(seed) if use_seed and seed is not None else None,
            checkpoint=Path(checkpoint) if source == MODEL else None,
            temperature=float(temperature),
            top_k=int(top_k) if top_k else None,
            stems=bool(stems),
        )
        stem = f"{source}_{request.seed if request.seed is not None else 'random'}"
        try:
            result = compose(
                request,
                out_base=config.output_dir / stem,
                profile=profile,
                configs_dir=config.configs_dir,
            )
        except ValueError as exc:
            raise gr.Error(str(exc)) from exc

        downloads = [p for p in (result.wav, result.midi, result.json) if p]
        downloads += list(result.stems.values())
        return (
            str(result.wav),
            describe(result),
            channel_rows(result),
            [str(p) for p in downloads],
            result.summary(),
        )

    with gr.Blocks(title="Retro Synthesis Engine", analytics_enabled=False) as app:
        gr.Markdown(
            "# Retro Synthesis Engine\n"
            f"Generating for **{profile.display_name}** — "
            f"{', '.join(profile.channel_ids)}. Running locally; nothing leaves "
            "this machine."
        )
        with gr.Row():
            with gr.Column(scale=1):
                source_label = gr.Radio(
                    choices=[
                        "Rule-based composer",
                        "Model (needs a trained checkpoint)",
                    ],
                    value="Rule-based composer",
                    label="Source",
                )
                checkpoint = gr.Dropdown(
                    choices=checkpoints,
                    value=checkpoints[0] if checkpoints else None,
                    label="Checkpoint",
                    info="Found in checkpoints/. Only used by the model source."
                    if checkpoints
                    else "None found. Train one with `rse train`.",
                )
                mood = gr.Dropdown(choices=list(MOODS), value="heroic", label="Mood")
                energy = gr.Radio(choices=list(ENERGIES), value="medium", label="Energy")
                bars = gr.Dropdown(choices=list(LENGTHS), value=16, label="Length (bars)")
                tempo = gr.Slider(60, 220, value=120, step=1, label="Tempo (BPM)")
                with gr.Row():
                    use_seed = gr.Checkbox(value=True, label="Fix seed")
                    seed = gr.Number(value=0, precision=0, label="Seed")
                with gr.Accordion("Model sampling", open=False):
                    temperature = gr.Slider(0.1, 2.0, value=1.0, step=0.05, label="Temperature")
                    top_k = gr.Slider(0, 200, value=40, step=1, label="Top-k (0 for off)")
                stems = gr.Checkbox(value=False, label="Also write per-channel stems")
                generate_button = gr.Button("Generate", variant="primary")
            with gr.Column(scale=2):
                audio = gr.Audio(label="Result", type="filepath", interactive=False)
                summary = gr.Markdown()
                channels = gr.Dataframe(
                    headers=["Channel", "Role", "Notes", "Range", "Max poly", "Overlaps"],
                    label="Channels",
                    interactive=False,
                )
                files = gr.File(label="Downloads", file_count="multiple", interactive=False)
                with gr.Accordion("Full report", open=False):
                    report = gr.JSON()

        # Mood and energy only steer the rule-based composer until the model has
        # labelled data to condition on (proposal 6.4).
        def on_source(label: str):
            is_model = label.startswith("Model")
            return (
                gr.update(interactive=not is_model),
                gr.update(
                    info="The checkpoint's own training tempo is used."
                    if is_model
                    else "Rule-based tempo."
                ),
            )

        source_label.change(on_source, inputs=source_label, outputs=[tempo, tempo])
        generate_button.click(
            run,
            inputs=[
                source_label,
                checkpoint,
                mood,
                energy,
                bars,
                tempo,
                seed,
                use_seed,
                temperature,
                top_k,
                stems,
            ],
            outputs=[audio, summary, channels, files, report],
        )
    return app


def launch(config: UiConfig | None = None) -> None:
    """Start the server on loopback."""
    config = config or UiConfig()
    host = check_bind(config.host, allow_remote=config.allow_remote)
    app = build_app(config)
    app.launch(
        server_name=host,
        server_port=config.port,
        # Never `share=True`: it publishes a public tunnel URL.
        share=False,
        inbrowser=config.open_browser,
        # The server serves only what it generated.
        allowed_paths=[str(config.output_dir)],
        quiet=False,
    )


def available_profile_ids(configs_dir: Path | None = None) -> list[str]:
    return available_profiles(configs_dir=configs_dir)
