"""Default on-disk locations.

Every path is overridable from the CLI; these are only the defaults used when a
command is run without explicit directories.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Repository root (this file lives at <root>/rse/paths.py).
REPO_ROOT = Path(__file__).resolve().parent.parent


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else default


#: Root for all external and derived data. Git-ignored.
DATASETS_DIR = _env_path("RSE_DATASETS_DIR", REPO_ROOT / "datasets")

#: Downloaded archives, one subdirectory per dataset release.
DOWNLOADS_DIR = DATASETS_DIR / "downloads"

#: Extracted source files. Treated as read-only, untrusted input.
RAW_DIR = DATASETS_DIR / "raw"

#: Derived intake records, indexes, statistics and reports.
PROCESSED_DIR = DATASETS_DIR / "processed"

#: Hardware profiles and dataset source definitions. Committed.
CONFIGS_DIR = _env_path("RSE_CONFIGS_DIR", REPO_ROOT / "configs")
HARDWARE_CONFIGS_DIR = CONFIGS_DIR / "hardware"
DATASET_CONFIGS_DIR = CONFIGS_DIR / "datasets"

#: Model weights. Git-ignored.
CHECKPOINTS_DIR = _env_path("RSE_CHECKPOINTS_DIR", REPO_ROOT / "checkpoints")
