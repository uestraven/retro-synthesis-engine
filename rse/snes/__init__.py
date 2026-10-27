"""SNES-inspired preview: sampled instruments, eight voices, stereo.

A PREVIEW renderer. It is sample-based, eight-voice limited and
hardware-informed, and it is not S-DSP-accurate: no BRR encoding, no Gaussian
interpolation, no ARAM enforcement. Those are separate verification tasks and
nothing here may be labelled authentic hardware output.
"""

from __future__ import annotations

__all__ = ["samples"]
