"""Audio synthesis adapters.

`NesApuRenderer` is a native APU synthesizer: no external binary, no
subprocess, runs offline everywhere. docs/renderer-survey.md records why the
alternatives (NES-MDB's synth package, FamiStudio CLI, Furnace CLI) were not
used, and what this one does not yet model.
"""

from rse.renderers.base import (
    RenderNote,
    RenderResult,
    RenderSettings,
    notes_from_record,
)
from rse.renderers.nes_apu import NesApuRenderer

__all__ = [
    "NesApuRenderer",
    "RenderNote",
    "RenderResult",
    "RenderSettings",
    "notes_from_record",
]
