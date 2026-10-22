"""Loading and querying versioned hardware profiles.

Phase 1 only needs the profile to name channels, infer their roles from source
tracks, and report out-of-range events. The Phase 2 constraint engine builds on
the same objects, so the profile stays the single source of truth for what a
console can play.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rse import paths
from rse.jsonio import read_json

SUPPORTED_SCHEMA_MAJOR = "rse-hardware/0"


class ProfileError(ValueError):
    """Raised when a hardware profile is missing or malformed."""


@dataclass(frozen=True)
class ChannelSpec:
    """One hardware channel of a profile."""

    channel_id: str
    role: str
    type: str
    pitched: bool
    max_simultaneous_notes: int
    value_min: int | None = None
    value_max: int | None = None
    volume_levels: int = 16
    instrument_params: tuple[str, ...] = ()
    effects: tuple[str, ...] = ()
    track_name_patterns: tuple[str, ...] = ()
    raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChannelSpec:
        pitched = bool(data.get("pitched", True))
        # Pitched channels express limits as MIDI note numbers; unpitched ones
        # (NES noise) express them as hardware values, so accept either spelling
        # and keep the distinction in `pitched`.
        lo = data.get("pitch_min") if pitched else data.get("value_min")
        hi = data.get("pitch_max") if pitched else data.get("value_max")
        try:
            return cls(
                channel_id=data["channel_id"],
                role=data["role"],
                type=data["type"],
                pitched=pitched,
                max_simultaneous_notes=int(data.get("max_simultaneous_notes", 1)),
                value_min=None if lo is None else int(lo),
                value_max=None if hi is None else int(hi),
                volume_levels=int(data.get("volume_levels", 16)),
                instrument_params=tuple(data.get("instrument_params", ())),
                effects=tuple(data.get("effects", ())),
                track_name_patterns=tuple(data.get("track_name_patterns", ())),
                raw=data,
            )
        except KeyError as exc:  # pragma: no cover - config authoring error
            raise ProfileError(f"channel is missing required field {exc}") from exc

    def in_range(self, value: int) -> bool:
        """Whether `value` is representable on this channel."""
        if self.value_min is not None and value < self.value_min:
            return False
        if self.value_max is not None and value > self.value_max:
            return False
        return True


@dataclass(frozen=True)
class HardwareProfile:
    """A console's channel layout and representable-event rules."""

    profile_id: str
    display_name: str
    schema_version: str
    channels: tuple[ChannelSpec, ...]
    engine_tick_hz: float | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @property
    def channel_ids(self) -> tuple[str, ...]:
        return tuple(ch.channel_id for ch in self.channels)

    def channel(self, channel_id: str) -> ChannelSpec:
        for ch in self.channels:
            if ch.channel_id == channel_id:
                return ch
        raise KeyError(f"{self.profile_id} has no channel {channel_id!r}")

    def channel_by_role(self, role: str) -> ChannelSpec:
        for ch in self.channels:
            if ch.role == role:
                return ch
        raise KeyError(f"{self.profile_id} has no channel with role {role!r}")

    def match_track_name(self, track_name: str | None) -> ChannelSpec | None:
        """Find the channel whose name patterns match a source track name."""
        if not track_name:
            return None
        needle = track_name.strip().lower()
        for ch in self.channels:
            for pattern in ch.track_name_patterns:
                if re.match(pattern, needle):
                    return ch
        return None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HardwareProfile:
        schema_version = str(data.get("schema_version", ""))
        if not schema_version.startswith(SUPPORTED_SCHEMA_MAJOR):
            raise ProfileError(
                f"unsupported hardware schema {schema_version!r}; "
                f"this build understands {SUPPORTED_SCHEMA_MAJOR}.x"
            )
        channels = tuple(ChannelSpec.from_dict(ch) for ch in data.get("channels", ()))
        if not channels:
            raise ProfileError(f"profile {data.get('profile_id')!r} declares no channels")
        seen: set[str] = set()
        for ch in channels:
            if ch.channel_id in seen:
                raise ProfileError(f"duplicate channel_id {ch.channel_id!r}")
            seen.add(ch.channel_id)
        tick_hz = data.get("engine_tick_hz")
        return cls(
            profile_id=data["profile_id"],
            display_name=data.get("display_name", data["profile_id"]),
            schema_version=schema_version,
            channels=channels,
            engine_tick_hz=None if tick_hz is None else float(tick_hz),
            raw=data,
        )


def profile_path(profile_id: str, *, configs_dir: Path | None = None) -> Path:
    base = configs_dir or paths.HARDWARE_CONFIGS_DIR
    return base / f"{profile_id}.json"


def load_profile(profile_id: str = "nes", *, configs_dir: Path | None = None) -> HardwareProfile:
    """Load a profile by id from the hardware config directory."""
    path = profile_path(profile_id, configs_dir=configs_dir)
    if not path.is_file():
        raise ProfileError(f"no hardware profile at {path}")
    return HardwareProfile.from_dict(read_json(path))


def available_profiles(*, configs_dir: Path | None = None) -> list[str]:
    base = configs_dir or paths.HARDWARE_CONFIGS_DIR
    if not base.is_dir():
        return []
    return sorted(p.stem for p in base.glob("*.json"))
