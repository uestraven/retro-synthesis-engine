"""The canonical RSE Event Format (RSE-EF).

Phase 1 intake records are the lossless landing zone for source data; RSE-EF is
the versioned representation the rest of the system agrees on. Conversion both
ways, and round-trip measurement against a stated tolerance, live in
`convert`.
"""

from rse.events.convert import (
    EXACT,
    RoundTripReport,
    Tolerance,
    compare_compositions,
    composition_from_intake,
    composition_from_midi,
    composition_to_intake,
    composition_to_midi,
    round_trip_midi,
    write_composition_midi,
)
from rse.events.schema import (
    RSE_EF_SCHEMA_VERSION,
    Composition,
    Controller,
    EffectEvent,
    Event,
    Noise,
    Note,
    Program,
    SchemaError,
)

__all__ = [
    "EXACT",
    "RSE_EF_SCHEMA_VERSION",
    "Composition",
    "Controller",
    "EffectEvent",
    "Event",
    "Noise",
    "Note",
    "Program",
    "RoundTripReport",
    "SchemaError",
    "Tolerance",
    "compare_compositions",
    "composition_from_intake",
    "composition_from_midi",
    "composition_to_intake",
    "composition_to_midi",
    "round_trip_midi",
    "write_composition_midi",
]
