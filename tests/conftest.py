"""Shared fixtures.

Every test builds its own source files; nothing here depends on NES-MDB being
present, so the suite runs on a clean checkout.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rse import paths
from rse.data.provenance import USE_CLEARED, SourceRecord, describe_file
from rse.data.synthetic import write_nes_style_midi
from rse.hardware.profile import load_profile


@pytest.fixture(scope="session")
def nes_profile():
    """The committed NES profile, loaded from configs/hardware/nes.json."""
    return load_profile("nes", configs_dir=paths.HARDWARE_CONFIGS_DIR)


@pytest.fixture
def midi_path(tmp_path: Path) -> Path:
    return write_nes_style_midi(tmp_path / "song.mid", seed=7, bars=8)


@pytest.fixture
def source_record(midi_path: Path) -> SourceRecord:
    return describe_file(
        midi_path,
        root=midi_path.parent,
        dataset_id="test",
        dataset_version="test/v1",
        use_class=USE_CLEARED,
        acquisition_method="generated",
    )
