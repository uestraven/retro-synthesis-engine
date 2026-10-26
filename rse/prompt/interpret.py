"""Turn a free-text music request into a plan the composer can actually take.

Fermata is not language-conditioned. Its entire conditioning surface is a
header of categorical tokens: 9 genres, 7 ensembles, 105 keys (15 roots x 7
modes), 307 tempos, 264 meters, and a track plan drawn from 128 General MIDI
programs plus a drum channel. Nothing in it carries "memorable", "haunting",
or "builds tension".

So this module does one job honestly: map the phrases it recognises onto that
surface, and **say plainly what it could not use.** Every requirement lands in
exactly one of three buckets:

- `satisfied` — expressed directly in the composer's conditioning, or
  guaranteed downstream by deterministic machinery (tempo, meter, duration,
  instrumentation, loop point).
- `approximated` — a human word mapped onto a musical proxy. "Dark" becomes a
  minor or phrygian key; that is a defensible reading, not an understanding of
  the word, and the substitution is recorded so a listener can disagree with
  it.
- `unsupported` — no conditioning exists and no deterministic machinery
  substitutes. "Memorable melody" is the standing example: nothing in the
  model or the pipeline takes memorability as an input, and claiming otherwise
  would be the single easiest way to mislead about what this system does.

Structural requests ("contrasting section", "return to the main theme") are
marked `imposed`: the composer never receives them, and the song assembler
produces them deterministically. That distinction matters more than it looks.
A listener who hears a returning theme should know whether the model chose to
bring it back or whether we pasted it there.

Matching is keyword-based and deliberately shallow. A shallow matcher whose
misses are reported is safer than a clever one whose misses are silent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

#: Mode names the composer's `<KEY_root:mode>` family accepts.
MODES = ("major", "minor", "dorian", "phrygian", "lydian", "mixolydian", "none")

#: Mood words to (mode, why). Minor and phrygian for darkness, lydian and
#: major for brightness. Phrygian is the darker of the two minor-ish readings,
#: so "evil" and "sinister" get it where "sad" does not.
MOOD_MODES: dict[str, tuple[str, str]] = {
    "dark": ("minor", "minor key"),
    "ominous": ("minor", "minor key"),
    "sinister": ("phrygian", "phrygian mode, a darker minor"),
    "evil": ("phrygian", "phrygian mode, a darker minor"),
    "menacing": ("phrygian", "phrygian mode, a darker minor"),
    "haunting": ("minor", "minor key"),
    "sad": ("minor", "minor key"),
    "melancholy": ("minor", "minor key"),
    "tense": ("minor", "minor key"),
    "mysterious": ("dorian", "dorian mode"),
    "hopeful": ("major", "major key"),
    "happy": ("major", "major key"),
    "bright": ("major", "major key"),
    "cheerful": ("major", "major key"),
    "triumphant": ("major", "major key"),
    "heroic": ("mixolydian", "mixolydian mode"),
    "peaceful": ("lydian", "lydian mode"),
    "dreamy": ("lydian", "lydian mode"),
}

#: Energy words to a tempo in BPM. Game music sits high; a "battle theme"
#: default of 168 matches the existing `action` scenario.
ENERGY_TEMPO: dict[str, int] = {
    "frantic": 190,
    "frenetic": 190,
    "driving": 170,
    "energetic": 168,
    "fast": 168,
    "upbeat": 150,
    "lively": 150,
    "moderate": 120,
    "steady": 120,
    "relaxed": 100,
    "calm": 92,
    "slow": 80,
    "sparse": 76,
    "mournful": 70,
}

#: Style words to a Fermata genre.
GENRE_WORDS: dict[str, str] = {
    "jrpg": "soundtrack",
    "rpg": "soundtrack",
    "game": "soundtrack",
    "battle": "soundtrack",
    "boss": "soundtrack",
    "dungeon": "soundtrack",
    "overworld": "soundtrack",
    "town": "folk",
    "village": "folk",
    "folk": "folk",
    "chiptune": "electronic",
    "electronic": "electronic",
    "techno": "electronic",
    "orchestral": "classical",
    "classical": "classical",
    "baroque": "classical",
    "jazz": "jazz",
    "choral": "religious",
    "hymn": "religious",
}

#: Ensemble words.
ENSEMBLE_WORDS: dict[str, str] = {
    "band": "band",
    "orchestral": "orchestral",
    "orchestra": "orchestral",
    "chamber": "chamber",
    "solo": "solo",
    "duo": "duo",
    "choir": "choir",
    "choral": "choir",
}

#: Instrument words to the General MIDI program the track plan will request.
#: Restricted to programs the NES arranger can meaningfully carry.
INSTRUMENT_PROGRAMS: dict[str, tuple[int, str]] = {
    "lead": (80, "square lead"),
    "melody": (80, "square lead"),
    "square": (80, "square lead"),
    "pulse": (80, "square lead"),
    "saw": (81, "sawtooth lead"),
    "second lead": (81, "sawtooth lead"),
    "harmony": (81, "sawtooth lead"),
    "bass": (38, "synth bass"),
    "bassline": (38, "synth bass"),
}

#: Words that ask for percussion.
PERCUSSION_WORDS = (
    "percussion",
    "drums",
    "drum",
    "beat",
    "rhythmic",
    "driving percussion",
)

#: Phrases we recognise as structural requests. The composer never sees these;
#: the assembler builds them.
STRUCTURE_WORDS: dict[str, str] = {
    "contrasting": "contrast_section",
    "contrast": "contrast_section",
    "middle section": "contrast_section",
    "b section": "contrast_section",
    "bridge": "contrast_section",
    "return": "theme_return",
    "returns": "theme_return",
    "reprise": "theme_return",
    "recap": "theme_return",
    "loop": "loop",
    "loopable": "loop",
    "looping": "loop",
    "seamless": "loop",
    "intro": "intro",
    "introduction": "intro",
    "build": "build",
    "builds": "build",
    "buildup": "build",
}

#: Requests with no conditioning and no deterministic substitute. Matching one
#: of these is the point: it gets reported as unsupported rather than quietly
#: dropped.
UNSUPPORTED_PHRASES: dict[str, str] = {
    "memorable": "no conditioning expresses memorability; nothing in the model takes it as input",
    "catchy": "no conditioning expresses catchiness",
    "hummable": "no conditioning expresses this",
    "emotional": "no conditioning expresses emotional content directly",
    "epic": "not a musical parameter the composer accepts",
    "tension": "no conditioning expresses tension; dynamics and density are not controllable",
    "crescendo": "the composer has no dynamics control; velocity is not conditioned",
    "dynamics": "velocity is generated, not conditioned",
    "key change": "the header carries one key for the whole piece",
    "modulation": "the header carries one key for the whole piece",
    "time signature change": "the header carries one meter for the whole piece",
    "counterpoint": "no conditioning expresses voice-leading style",
    "chord progression": "harmony is generated, not specified",
}

#: Default when the text gives no tempo cue.
DEFAULT_TEMPO = 120
DEFAULT_GENRE = "soundtrack"
DEFAULT_ENSEMBLE = "band"
DEFAULT_MODE = "minor"
DEFAULT_BEATS_PER_BAR = 4

#: Root offset in fifths for `<KEY_root:mode>`. 0 is C.
DEFAULT_ROOT = 0


@dataclass
class Requirement:
    """One thing the prompt asked for, and what became of it."""

    text: str
    status: str  # satisfied | approximated | unsupported | imposed
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "status": self.status, "detail": self.detail}


#: Scenario words to a scenario name. These do not reach the composer -- it
#: has no scenario input -- but they are what the listener asked for, and
#: recording them lets the mapping to genre/key/tempo be audited.
SCENARIO_WORDS: dict[str, str] = {
    "boss battle": "boss_battle",
    "boss": "boss_battle",
    "battle": "battle",
    "combat": "battle",
    "fight": "battle",
    "town": "town",
    "village": "town",
    "shop": "town",
    "dungeon": "dungeon",
    "cave": "dungeon",
    "overworld": "overworld",
    "field": "overworld",
    "forest": "exploration",
    "exploration": "exploration",
    "title": "title",
    "credits": "credits",
    "game over": "game_over",
}

#: Style words to a style name, kept separate from genre for the same reason.
STYLE_WORDS: dict[str, str] = {
    "nes": "nes",
    "jrpg": "nes_jrpg",
    "rpg": "nes_jrpg",
    "chiptune": "chiptune",
    "famicom": "nes",
    "8-bit": "chiptune",
    "8 bit": "chiptune",
}

#: Energy words to a coarse level, alongside the tempo they imply.
ENERGY_LEVELS: dict[str, str] = {
    "frantic": "very_high",
    "frenetic": "very_high",
    "driving": "high",
    "energetic": "high",
    "fast": "high",
    "upbeat": "medium_high",
    "lively": "medium_high",
    "moderate": "medium",
    "steady": "medium",
    "relaxed": "low",
    "calm": "low",
    "slow": "low",
    "sparse": "very_low",
    "mournful": "very_low",
}


@dataclass
class SongPlan:
    """A composition plan expressed only in things the pipeline can deliver.

    Carries two vocabularies on purpose. `style`, `scenario`, `mood` and
    `energy` are what the request asked for; `genre`, `ensemble`, `key_mode`
    and `tempo_bpm` are what the composer can be told. Keeping both means the
    mapping between them is inspectable rather than implicit.
    """

    prompt: str = ""
    style: str = ""
    scenario: str = ""
    mood: str = ""
    energy: str = ""
    loop: bool = False
    genre: str = DEFAULT_GENRE
    ensemble: str = DEFAULT_ENSEMBLE
    tempo_bpm: int = DEFAULT_TEMPO
    key_root: int = DEFAULT_ROOT
    key_mode: str = DEFAULT_MODE
    beats_per_bar: int = DEFAULT_BEATS_PER_BAR
    target_seconds: float = 60.0
    programs: list[int] = field(default_factory=lambda: [80, 81, 38])
    percussion: bool = True
    structure: list[str] = field(default_factory=list)
    requirements: list[Requirement] = field(default_factory=list)

    @property
    def key_token(self) -> str:
        return f"{self.key_root}:{self.key_mode}"

    @property
    def seconds_per_bar(self) -> float:
        return self.beats_per_bar * 60.0 / self.tempo_bpm

    def bars_for_target(self, *, section_bars: int) -> int:
        """Bars nearest the requested duration, rounded to whole sections.

        Duration is a satisfied requirement rather than an approximated one
        because it is arithmetic: the tempo and meter are known, so the bar
        count follows. Rounding to whole sections keeps the form intact, which
        matters more than hitting the second exactly.
        """
        raw = self.target_seconds / self.seconds_per_bar
        sections = max(1, round(raw / section_bars))
        return int(sections * section_bars)

    def plan_tokens(self) -> list[str]:
        """The `<TRACK_*>`/`<DRUM_TRACK>` list for the composer's header."""
        tokens = [f"<TRACK_{p}>" for p in self.programs]
        if self.percussion:
            tokens.append("<DRUM_TRACK>")
        return tokens

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "style": self.style,
            "scenario": self.scenario,
            "mood": self.mood,
            "energy": self.energy,
            "loop": self.loop,
            "genre": self.genre,
            "ensemble": self.ensemble,
            "tempo_bpm": self.tempo_bpm,
            "key": self.key_token,
            "beats_per_bar": self.beats_per_bar,
            "target_seconds": self.target_seconds,
            "seconds_per_bar": round(self.seconds_per_bar, 4),
            "programs": list(self.programs),
            "percussion": self.percussion,
            "structure": list(self.structure),
            "requirements": [r.to_dict() for r in self.requirements],
        }

    def report(self) -> dict[str, list[dict[str, Any]]]:
        """Requirements grouped by what happened to them."""
        out: dict[str, list[dict[str, Any]]] = {
            "satisfied": [],
            "approximated": [],
            "imposed": [],
            "unsupported": [],
        }
        for requirement in self.requirements:
            out.setdefault(requirement.status, []).append(requirement.to_dict())
        return out


def _find_duration(text: str) -> tuple[float | None, str]:
    """Seconds, from "60 second", "90s", "a minute and a half"."""
    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:-|\s)?\s*(seconds?|secs?|s\b)", text)
    if match:
        return float(match.group(1)), match.group(0)
    match = re.search(r"(\d+(?:\.\d+)?)\s*minutes?", text)
    if match:
        return float(match.group(1)) * 60.0, match.group(0)
    if "minute and a half" in text:
        return 90.0, "minute and a half"
    if re.search(r"\ba minute\b", text):
        return 60.0, "a minute"
    return None, ""


def _find_tempo(text: str) -> tuple[int | None, str]:
    match = re.search(r"(\d{2,3})\s*bpm", text)
    if match:
        return int(match.group(1)), match.group(0)
    return None, ""


def _find_meter(text: str) -> tuple[int | None, str]:
    match = re.search(r"\b(\d)\s*/\s*4\b", text)
    if match:
        return int(match.group(1)), match.group(0)
    if "waltz" in text:
        return 3, "waltz"
    return None, ""


def interpret(prompt: str) -> SongPlan:
    """Read a request and return a plan plus a full accounting of it."""
    text = prompt.lower()
    plan = SongPlan(prompt=prompt)
    seen: set[str] = set()

    def add(req: Requirement) -> None:
        marker = f"{req.status}:{req.text}"
        if marker not in seen:
            seen.add(marker)
            plan.requirements.append(req)

    # -- duration: arithmetic, so satisfied rather than approximated --------
    seconds, phrase = _find_duration(text)
    if seconds is not None:
        plan.target_seconds = seconds
        add(Requirement(phrase, "satisfied", f"target {seconds:g} s"))

    # -- tempo: explicit BPM wins over an energy word ----------------------
    tempo, phrase = _find_tempo(text)
    if tempo is not None:
        plan.tempo_bpm = tempo
        add(Requirement(phrase, "satisfied", f"{tempo} BPM in the header"))
    else:
        for word, value in ENERGY_TEMPO.items():
            if re.search(rf"\b{re.escape(word)}\b", text):
                plan.tempo_bpm = value
                plan.energy = ENERGY_LEVELS.get(word, "")
                add(
                    Requirement(
                        word, "approximated", f"read as {value} BPM"
                    )
                )
                break

    meter, phrase = _find_meter(text)
    if meter is not None:
        plan.beats_per_bar = meter
        add(Requirement(phrase, "satisfied", f"{meter}/4 in the header"))

    # -- mood: a human word onto a mode. Always approximated ---------------
    for word, (mode, why) in MOOD_MODES.items():
        if re.search(rf"\b{re.escape(word)}\b", text):
            plan.key_mode = mode
            plan.mood = word
            add(Requirement(word, "approximated", f"read as {why}"))
            break

    # Scenario and style are recorded but reach the composer only through the
    # genre/key/tempo they imply; Fermata has no scenario input.
    for phrase, scenario in SCENARIO_WORDS.items():
        if phrase in text:
            plan.scenario = scenario
            add(
                Requirement(
                    phrase,
                    "approximated",
                    f"recorded as scenario {scenario!r}; the composer has no "
                    "scenario input, so it reaches generation only through "
                    "genre, key and tempo",
                )
            )
            break
    for word, style in STYLE_WORDS.items():
        if word in text:
            plan.style = style
            add(
                Requirement(
                    word,
                    "approximated",
                    f"recorded as style {style!r}; expressed to the composer "
                    "as a genre token and to the arranger as the NES profile",
                )
            )
            break

    for word, genre in GENRE_WORDS.items():
        if re.search(rf"\b{re.escape(word)}\b", text):
            plan.genre = genre
            add(Requirement(word, "approximated", f"genre token {genre!r}"))
            break

    for word, ensemble in ENSEMBLE_WORDS.items():
        if re.search(rf"\b{re.escape(word)}\b", text):
            plan.ensemble = ensemble
            add(Requirement(word, "satisfied", f"ensemble token {ensemble!r}"))
            break

    # -- instrumentation ---------------------------------------------------
    # Instrument words are ADDITIVE, never restrictive. "A memorable opening
    # melody" names a lead; it does not ask for the bass and harmony to be
    # removed. Reading it as a restriction cut the plan to a single track and
    # would have produced a one-voice arrangement from a prompt that plainly
    # wanted a band.
    requested: list[int] = []
    for word, (program, label) in INSTRUMENT_PROGRAMS.items():
        if re.search(rf"\b{re.escape(word)}\b", text):
            if program not in requested:
                requested.append(program)
                add(Requirement(word, "satisfied", f"GM program {program}, {label}"))
    for program in requested:
        if program not in plan.programs:
            plan.programs.append(program)
    # Canonical lead/harmony/bass order, so the track plan is stable
    # regardless of the order words appeared in.
    plan.programs = [p for p in (80, 81, 38) if p in plan.programs]
    for word in PERCUSSION_WORDS:
        if word in text:
            plan.percussion = True
            add(Requirement(word, "satisfied", "drum track requested in the plan"))
            break

    # -- structure: the composer never sees these; the assembler builds them
    for word, feature in STRUCTURE_WORDS.items():
        if re.search(rf"\b{re.escape(word)}\b", text):
            if feature not in plan.structure:
                plan.structure.append(feature)
            if feature == "loop":
                plan.loop = True
            add(
                Requirement(
                    word,
                    "imposed",
                    f"{feature} built deterministically by the assembler, "
                    "not requested of the composer",
                )
            )

    # -- the honest part ---------------------------------------------------
    for phrase, why in UNSUPPORTED_PHRASES.items():
        if phrase in text:
            add(Requirement(phrase, "unsupported", why))

    return plan
