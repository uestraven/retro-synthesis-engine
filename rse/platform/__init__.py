"""Platform-independent musical abstractions, and per-platform realisation.

The split the SNES work needs: a musical part is an idea, a voice is a slot
that plays one note at a time, and only the second is hardware.
"""

from __future__ import annotations

__all__ = ["roles"]
