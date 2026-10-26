"""NES composition profiles: decide the music's shape before generating it.

The pipeline up to now has been *compose, then constrain*: Fermata writes
unrestricted multitrack MIDI and RSE forces it onto four monophonic channels
afterwards. A profile inserts a planning step, so the composition is asked for
in terms the NES can carry.

The distinction that keeps this honest is between two kinds of constraint.

**Hardware constraints** decide whether the music can play at all: four
monophonic voices, pitch range, one note per channel per tick. Those are
already enforced, by `rse.hardware.constraints`, and they are not negotiable.

**Creative constraints** decide how music is *written* for that hardware: one
clear lead, a supporting voice that answers rather than competes, a bass that
moves the harmony, a motif worth returning to. Those are the missing layer,
and they are not rules every NES composer followed -- they are planning
targets. A profile must not force percussion into an atmospheric piece or
keep Pulse 1 sounding continuously.

**And a third category has to be stated, not hidden.** Fermata's conditioning
is a header of categorical tokens. It has no input for "keep the lead clear",
"leave room to breathe", or "don't let the pulses compete". A profile cannot
ask for those. What it *can* do is:

- choose the four roles and have the arranger honour them, instead of letting
  a behavioural heuristic guess (it overrode the composer's declared lead on
  6 of 12 measured pieces);
- choose tempo, meter and key per section, which the header does carry;
- choose the form and build it by continuation from a shared opening.

Everything else is reported as unsupported. `reachability()` returns exactly
which fields reached the composer, which were enforced downstream, and which
could not be expressed at all -- so a profile can never imply the model was
told something it was not.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rse.transfer.score import ACCOMPANIMENT, BASS, MELODY, PERCUSSION

#: Channel to role, the NES convention this profile plans against.
NES_CHANNEL_ROLES = {
    "pulse1": MELODY,
    "pulse2": ACCOMPANIMENT,
    "triangle": BASS,
    "noise": PERCUSSION,
}

#: Role to the GM program the composer's track plan will request. Fermata has
#: no NES channels, so a role is expressed to it as an instrument choice.
ROLE_PROGRAMS: dict[str, int] = {
    MELODY: 80,
    ACCOMPANIMENT: 81,
    BASS: 38,
}

#: Role names a profile may use for a pitched channel.
PITCHED_ROLES = (MELODY, ACCOMPANIMENT, BASS)

#: Creative preferences a profile may state. Every one of these is a planning
#: target with no conditioning behind it; the value records the reason, which
#: is reported rather than silently dropped.
PREFERENCE_REACH: dict[str, str] = {
    "preserve_lead_clarity": (
        "partly enforced: the arranger honours the declared melody role "
        "instead of guessing it, but the composer is not asked to keep the "
        "lead clear"
    ),
    "encourage_phrase_rests": (
        "unsupported: no conditioning expresses rests or phrase boundaries"
    ),
    "avoid_competing_melodies": (
        "unsupported: the composer cannot be told to subordinate one voice"
    ),
    "allow_arpeggios": (
        "enforced downstream: the arranger arpeggiates chords across sub-row "
        "slots rather than dropping them"
    ),
    "limit_simultaneous_activity": (
        "unsupported as a request; monophony per channel is enforced, but "
        "density is not conditioned"
    ),
}


@dataclass
class Section:
    """One block of the form."""

    label: str
    bars: int
    #: `fresh` generates from the header; `continue_from` extends an earlier
    #: section's opening bars, which is how a theme returns re-composed
    #: rather than pasted.
    source: str = "fresh"
    continue_from: str | None = None
    #: Semitone-equivalent key offset in fifths, relative to the profile key.
    key_offset: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "bars": self.bars,
            "source": self.source,
            "continue_from": self.continue_from,
            "key_offset": self.key_offset,
        }


@dataclass
class NesCompositionProfile:
    """A composition plan expressed against NES capabilities."""

    name: str = "nes_default"
    hardware: str = "nes_2a03_4voice"
    duration_seconds: float = 60.0
    tempo_bpm: int = 128
    beats_per_bar: int = 4
    genre: str = "soundtrack"
    ensemble: str = "band"
    key_root: int = 0
    key_mode: str = "minor"
    #: Channel name to role. A channel may be omitted, which means the piece
    #: does not ask for it -- an atmospheric cue need not have percussion.
    roles: dict[str, str] = field(
        default_factory=lambda: dict(NES_CHANNEL_ROLES)
    )
    sections: list[Section] = field(default_factory=list)
    primary_motifs: int = 1
    theme_bars: int = 4
    preferences: dict[str, bool] = field(default_factory=dict)

    # -- derived ----------------------------------------------------------
    @property
    def seconds_per_bar(self) -> float:
        return self.beats_per_bar * 60.0 / self.tempo_bpm

    @property
    def percussion_requested(self) -> bool:
        return PERCUSSION in self.roles.values()

    def programs(self) -> list[int]:
        """GM programs for the pitched roles, in lead/harmony/bass order."""
        wanted = {
            ROLE_PROGRAMS[r] for r in self.roles.values() if r in ROLE_PROGRAMS
        }
        return [p for p in (80, 81, 38) if p in wanted]

    def role_for_program(self) -> dict[int, str]:
        """What `arrange_for_nes` needs to honour the plan."""
        return {
            ROLE_PROGRAMS[role]: role
            for role in self.roles.values()
            if role in ROLE_PROGRAMS
        }

    def plan_tokens(self) -> list[str]:
        tokens = [f"<TRACK_{p}>" for p in self.programs()]
        if self.percussion_requested:
            tokens.append("<DRUM_TRACK>")
        return tokens

    def total_bars(self) -> int:
        return sum(s.bars for s in self.sections)

    def planned_seconds(self) -> float:
        return self.total_bars() * self.seconds_per_bar

    def default_form(self) -> list[Section]:
        """A-A'-B-A'' sized to the requested duration.

        Section length comes from `theme_bars * 2`, so a four-bar theme gives
        eight-bar sections: the theme occupies the first half and the
        continuation writes the second.
        """
        section_bars = max(2, self.theme_bars * 2)
        count = max(2, round(self.duration_seconds / (section_bars * self.seconds_per_bar)))
        labels: list[Section] = [Section("A", section_bars, "fresh")]
        for index in range(1, count):
            last = index == count - 1
            contrast = index == count - 2 and count >= 3
            if contrast:
                labels.append(
                    Section("B", section_bars, "fresh", key_offset=1)
                )
            else:
                labels.append(
                    Section(
                        "A''" if last else f"A'{index}",
                        section_bars,
                        "continue_from",
                        continue_from="A",
                    )
                )
        return labels

    def ensure_form(self) -> None:
        if not self.sections:
            self.sections = self.default_form()

    # -- honesty ----------------------------------------------------------
    def reachability(self) -> dict[str, list[dict[str, str]]]:
        """Which parts of this profile reach the composer, and which do not."""
        out: dict[str, list[dict[str, str]]] = {
            "composer_conditioning": [],
            "enforced_downstream": [],
            "assembler_imposed": [],
            "unsupported": [],
        }

        def note(bucket: str, field_name: str, detail: str) -> None:
            out[bucket].append({"field": field_name, "detail": detail})

        note("composer_conditioning", "tempo_bpm", f"<TEMPO_{self.tempo_bpm}>")
        note("composer_conditioning", "meter", f"<METER_{self.beats_per_bar}/4>")
        note("composer_conditioning", "genre", f"<GENRE_{self.genre}>")
        note("composer_conditioning", "ensemble", f"<ENSEMBLE_{self.ensemble}>")
        note(
            "composer_conditioning",
            "key",
            f"<KEY_{self.key_root}:{self.key_mode}>, per section",
        )
        note(
            "composer_conditioning",
            "roles",
            f"as a track plan: {' '.join(self.plan_tokens())}",
        )
        note(
            "enforced_downstream",
            "hardware",
            f"{self.hardware}: four monophonic voices, pitch range, "
            "one note per channel per tick",
        )
        note(
            "enforced_downstream",
            "roles",
            "the arranger honours the declared melody/harmony/bass roles "
            "rather than inferring them from note activity",
        )
        note(
            "enforced_downstream",
            "duration_seconds",
            f"{self.total_bars()} bars at {self.tempo_bpm} BPM = "
            f"{self.planned_seconds():.1f} s",
        )
        note(
            "assembler_imposed",
            "form",
            "-".join(s.label for s in self.sections)
            + "; returns are generated by continuing the opening bars, "
            "contrast by changing the key",
        )
        note(
            "assembler_imposed",
            "primary_motifs",
            f"{self.primary_motifs} principal idea, reused by continuation; "
            "the composer is not asked to limit itself to one motif",
        )
        for key, enabled in self.preferences.items():
            if not enabled:
                continue
            reach = PREFERENCE_REACH.get(key, "unsupported: unrecognised preference")
            bucket = (
                "enforced_downstream"
                if reach.startswith(("enforced", "partly enforced"))
                else "unsupported"
            )
            note(bucket, key, reach)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "hardware": self.hardware,
            "duration_seconds": self.duration_seconds,
            "tempo_bpm": self.tempo_bpm,
            "meter": f"{self.beats_per_bar}/4",
            "genre": self.genre,
            "ensemble": self.ensemble,
            "key": f"{self.key_root}:{self.key_mode}",
            "roles": dict(self.roles),
            "programs": self.programs(),
            "percussion_requested": self.percussion_requested,
            "sections": [s.to_dict() for s in self.sections],
            "total_bars": self.total_bars(),
            "planned_seconds": round(self.planned_seconds(), 3),
            "primary_motifs": self.primary_motifs,
            "theme_bars": self.theme_bars,
            "preferences": dict(self.preferences),
        }


def validate(profile: NesCompositionProfile) -> list[str]:
    """Problems that would make a profile unusable, as plain sentences."""
    problems: list[str] = []
    if profile.tempo_bpm <= 0:
        problems.append("tempo_bpm must be positive")
    if profile.beats_per_bar <= 0:
        problems.append("beats_per_bar must be positive")
    for channel, role in profile.roles.items():
        if channel not in NES_CHANNEL_ROLES:
            problems.append(
                f"unknown channel {channel!r}; expected one of "
                f"{sorted(NES_CHANNEL_ROLES)}"
            )
        elif role not in (*PITCHED_ROLES, PERCUSSION):
            problems.append(f"channel {channel!r} has unknown role {role!r}")
    if MELODY not in profile.roles.values():
        problems.append(
            "no channel is assigned the melody role; a profile with no lead "
            "cannot plan a theme"
        )
    labels = [s.label for s in profile.sections]
    if len(labels) != len(set(labels)):
        problems.append(f"duplicate section labels: {labels}")
    for section in profile.sections:
        if section.source == "continue_from" and section.continue_from not in labels:
            problems.append(
                f"section {section.label!r} continues from "
                f"{section.continue_from!r}, which is not in the form"
            )
        if section.bars <= 0:
            problems.append(f"section {section.label!r} has {section.bars} bars")
    for key in profile.preferences:
        if key not in PREFERENCE_REACH:
            problems.append(f"unrecognised preference {key!r}")
    return problems


def load(path: Path) -> NesCompositionProfile:
    """Read a profile from YAML or JSON. Unknown keys are an error.

    Silently ignoring a misspelled key would mean a profile that looks like it
    requests something and does not.
    """
    text = Path(path).read_text()
    if path.suffix in (".yaml", ".yml"):
        import yaml

        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    return from_dict(data)


def from_dict(data: dict[str, Any]) -> NesCompositionProfile:
    """Build a profile from a mapping, rejecting unknown keys."""
    known = {
        "profile",
        "name",
        "hardware",
        "duration_seconds",
        "tempo_bpm",
        "meter",
        "genre",
        "ensemble",
        "key",
        "roles",
        "musical_plan",
        "preferences",
        "sections",
    }
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"unknown profile keys: {sorted(unknown)}")

    profile = NesCompositionProfile(
        name=str(data.get("profile") or data.get("name") or "nes_default"),
        hardware=str(data.get("hardware", "nes_2a03_4voice")),
        duration_seconds=float(data.get("duration_seconds", 60.0)),
        tempo_bpm=int(data.get("tempo_bpm", 128)),
        genre=str(data.get("genre", "soundtrack")),
        ensemble=str(data.get("ensemble", "band")),
        preferences=dict(data.get("preferences") or {}),
    )
    meter = str(data.get("meter", "4/4"))
    profile.beats_per_bar = int(meter.split("/")[0])
    key = str(data.get("key", "0:minor"))
    root, _, mode = key.partition(":")
    profile.key_root = int(root)
    profile.key_mode = mode or "minor"
    if data.get("roles"):
        profile.roles = dict(data["roles"])

    plan = data.get("musical_plan") or {}
    profile.primary_motifs = int(plan.get("primary_motifs", 1))
    profile.theme_bars = int(plan.get("theme_bars", 4))
    if data.get("sections"):
        profile.sections = [
            Section(
                label=str(s["label"]),
                bars=int(s["bars"]),
                source=str(s.get("source", "fresh")),
                continue_from=s.get("continue_from"),
                key_offset=int(s.get("key_offset", 0)),
            )
            for s in data["sections"]
        ]
    profile.ensure_form()
    problems = validate(profile)
    if problems:
        raise ValueError("invalid profile: " + "; ".join(problems))
    return profile
