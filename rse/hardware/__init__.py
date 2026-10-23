"""Hardware profiles and the constraint engine.

The profile says what a console can play; the constraint engine prevents
invalid events during generation and repairs them afterwards, recording every
change it makes (proposal 3.2).

Only the profile is re-exported here. The constraint engine reads RSE-EF, which
reads the MIDI parser, which reads the profile — so re-exporting the engine from
this package turns that chain into an import cycle. Import it directly:

    from rse.hardware.constraints import ConstraintEngine
"""

from rse.hardware.profile import ChannelSpec, HardwareProfile, load_profile

__all__ = ["ChannelSpec", "HardwareProfile", "load_profile"]
