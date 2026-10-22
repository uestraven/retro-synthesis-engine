"""Retro Synthesis Engine — symbolic music generation for vintage sound hardware."""

__version__ = "0.1.0.dev0"

# Schema identifier for the Phase 1 intake record. Phase 2 introduces the
# canonical RSE Event Format (RSE-EF); intake records are the raw, lossless
# landing zone that RSE-EF is derived from, and both are versioned separately.
INTAKE_SCHEMA_VERSION = "rse-intake/0.1.0"
