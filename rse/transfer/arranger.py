"""Arranging a general score for four NES voices, with the losses counted.

A pretrained composer writes for however many tracks it likes, each possibly
polyphonic. The APU has three monophonic pitched channels and one noise
channel. Something has to give, and the only dishonest option is to drop notes
quietly and call the conversion lossless -- so every decision here is counted
and reported.

The rules, and why each is this way
-----------------------------------
**Roles are assigned by behaviour, not by track order or program number.**
`rse.conditioning.lead_channel` already established that for NES-MDB: the lead
swaps between pulse channels and sometimes sits on the triangle. A general
score's track order is even less reliable, so the arranger scores tracks on
register and movement.

**The bass goes to the triangle.** Not a preference: the triangle is the
channel NES composers use for bass, and putting a lead there would sound wrong
even though it is legal.

**Polyphony is arpeggiated, not discarded.** A chord cannot sound on one
monophonic channel at once, but an NES composer does not throw the harmony
away -- they cycle the channel through the chord tones fast enough that the ear
hears it as a chord. That is the single most characteristic sound of the
platform, and `rse.events.classify` measures it as 13.3% of all real NES-MDB
events.

So a chord is spread across the sub-row slots the frozen D4 grid already
provides: with k=4 there are four onset positions per row, so up to four
voices become an arpeggio within the row they arrived on. Notes beyond that
are counted as `n_polyphony_dropped`. The previous behaviour kept the outer
voice and dropped the rest, which cost a median 36 notes per piece and up to
94 -- harmony the composer wrote and the hardware could have played.

A chord whose notes must be reordered is rolled from the bottom up, which is
how an arpeggio is normally voiced and keeps the root first.

**Overlaps are resolved by voice stealing**, truncating the sounding note at
the newcomer's onset, which is what the hardware does. Counted separately from
polyphony drops: one is a chord the APU cannot play, the other is a sustain it
cannot hold.

**Out-of-range pitches are transposed by octaves, or dropped.** Counted either
way. Silently clamping would collapse a melody onto the range boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import fmean
from typing import Any

from rse.events.schema import Composition, Noise, Note
from rse.transfer.converters import NOISE_TO_GM, ROWS_PER_BEAT
from rse.transfer.score import (
    ACCOMPANIMENT,
    BASS,
    MELODY,
    PERCUSSION,
    Score,
    Track,
)

#: The pitched channels, and the pitch range each may sound.
PITCH_MIN, PITCH_MAX = 21, 108
#: Triangle sits an octave lower in practice; kept inside the vocabulary range.
CHANNEL_FOR_ROLE = {MELODY: "P1", ACCOMPANIMENT: "P2", BASS: "TR"}

#: GM drum key to NES noise period, inverted from the canonical mapping so the
#: arranger and the audit agree by construction rather than by coincidence.
GM_TO_NOISE = {gm: period for period, gm in NOISE_TO_GM.items()}

#: Nearest-neighbour fallback for a GM key the canonical mapping does not
#: cover. A general model uses the whole kit; refusing everything outside 16
#: chosen keys would throw away most percussion, so unmapped keys snap to the
#: closest mapped one and are counted as approximated.
_MAPPED_GM = sorted(GM_TO_NOISE)


@dataclass
class ArrangeReport:
    """Everything the arrangement changed or could not keep."""

    n_events_in: int = 0
    n_events_out: int = 0
    n_polyphony_dropped: int = 0
    n_polyphony_arpeggiated: int = 0
    n_stolen_truncated: int = 0
    n_transposed: int = 0
    n_out_of_range_dropped: int = 0
    n_percussion_approximated: int = 0
    n_percussion_dropped: int = 0
    n_tracks_merged: int = 0
    role_assignment: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def retention(self) -> float:
        return self.n_events_out / self.n_events_in if self.n_events_in else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_events_in": self.n_events_in,
            "n_events_out": self.n_events_out,
            "retention": round(self.retention, 5),
            "n_polyphony_dropped": self.n_polyphony_dropped,
            "n_polyphony_arpeggiated": self.n_polyphony_arpeggiated,
            "n_stolen_truncated": self.n_stolen_truncated,
            "n_transposed": self.n_transposed,
            "n_out_of_range_dropped": self.n_out_of_range_dropped,
            "n_percussion_approximated": self.n_percussion_approximated,
            "n_percussion_dropped": self.n_percussion_dropped,
            "n_tracks_merged": self.n_tracks_merged,
            "role_assignment": dict(self.role_assignment),
            "warnings": self.warnings[:20],
        }


def _track_register(track: Track) -> float:
    return fmean([n.pitch for n in track.notes]) if track.notes else 0.0


def _track_movement(track: Track) -> float:
    """Mean absolute interval. An accompaniment tends to sit still."""
    pitches = [n.pitch for n in sorted(track.notes, key=lambda n: n.tick)]
    steps = [abs(b - a) for a, b in zip(pitches, pitches[1:], strict=False)]
    return fmean(steps) if steps else 0.0


def row_seconds(bpm: float, *, rows_per_beat: int = ROWS_PER_BEAT) -> float:
    """Real-time length of one tracker row at `bpm`.

    The bridge between musical positions and real time, and the value a
    caller must put in `GridFit.period_sec`. `rse.tokenizers.tokenizer`
    overrides the Tempo token with `fit.bpm` whenever the fit is aligned, so
    a caller that leaves `period_sec` at a fixed 0.1 forces every piece back
    to 150 BPM however correctly the arranger placed the notes.
    """
    if bpm <= 0:
        raise ValueError(f"bpm must be positive, got {bpm}")
    return (60.0 / bpm) / rows_per_beat


def assign_roles(
    score: Score, *, role_for_program: dict[int, str] | None = None
) -> dict[int, str]:
    """Which NES role each track should take.

    Two modes, and the difference matters musically.

    **Behavioural** (the default, and the only option before composition
    profiles existed): percussion by track type, then among pitched tracks the
    lowest register becomes the bass and the most active of the rest becomes
    the melody. That is the reasoning `lead_channel` uses on real NES music,
    where track order means nothing -- appropriate when nothing reliable
    declares intent.

    **Declared**: `role_for_program` maps a GM program to a role, so a plan
    made before generation is honoured after it. This exists because the
    behavioural heuristic overrode the composer's own declared lead on 6 of 12
    measured pieces: `_track_movement` is mean absolute interval, so a fast
    arpeggio outscores a stepwise tune and takes the melody slot from it.

    Neither is simply correct. The behavioural rule was *right* on the pieces
    where the declared lead held 21 notes of 326 -- it picked the voice
    actually carrying the material. So a declared role is used when the track
    has notes, and a track declared as melody but left empty falls back to
    the behavioural choice rather than silently muting the melody channel.
    """
    roles: dict[int, str] = {}
    pitched: list[int] = []
    for index, track in enumerate(score.tracks):
        if track.role == PERCUSSION or track.percussion:
            roles[index] = PERCUSSION
        elif track.notes:
            pitched.append(index)

    if not pitched:
        return roles

    if role_for_program:
        declared: dict[int, str] = {}
        for index in pitched:
            role = role_for_program.get(score.tracks[index].program or -1)
            if role in (MELODY, BASS, ACCOMPANIMENT):
                declared[index] = role
        # Only honour the plan if it actually names a melody that has notes.
        # Otherwise the melody channel would come out empty while a busy
        # voice sat on accompaniment, which is the failure the behavioural
        # rule exists to avoid.
        if MELODY in declared.values():
            roles.update(declared)
            for index in pitched:
                roles.setdefault(index, ACCOMPANIMENT)
            return roles

    by_register = sorted(pitched, key=lambda i: _track_register(score.tracks[i]))
    bass = by_register[0]
    roles[bass] = BASS
    rest = [i for i in pitched if i != bass]
    if rest:
        lead = max(rest, key=lambda i: _track_movement(score.tracks[i]))
        roles[lead] = MELODY
        for other in rest:
            if other != lead:
                roles[other] = ACCOMPANIMENT
    return roles


def _snap_drum(key: int) -> tuple[int, bool]:
    """Noise period for a GM drum key, and whether it was approximated."""
    if key in GM_TO_NOISE:
        return GM_TO_NOISE[key], False
    nearest = min(_MAPPED_GM, key=lambda k: abs(k - key))
    return GM_TO_NOISE[nearest], True


def arrange_for_nes(
    score: Score,
    *,
    row_sec: float,
    ticks_per_beat: int = 480,
    divisions: int = 4,
    role_for_program: dict[int, str] | None = None,
) -> tuple[Composition, ArrangeReport]:
    """Collapse a general score onto P1, P2, TR and NO.

    Musical positions are preserved exactly: eight composed bars are eight
    arranged bars at any tempo, because bar positions are a property of the
    music and only their real-time length depends on tempo. The score's tempo
    is carried on the output's tempo map, so playback rate is what was asked
    for.

    `role_for_program` lets a composition profile decide the channel roles
    before generation instead of leaving them to the behavioural heuristic.
    See `assign_roles` for when each is right.

    `row_sec` no longer affects positions. It survives only as the fallback
    output tempo for a score that carries none, and callers should stop
    relying on it to fix the timebase -- use `row_seconds(bpm)` to build the
    matching `GridFit`, or the D4 encode will override the tempo with
    `fit.bpm` and reintroduce the defect this function used to have.
    """
    report = ArrangeReport()
    roles = assign_roles(score, role_for_program=role_for_program)
    # Musical positions only. A bar is a bar at any tempo, so the conversion
    # from source ticks to destination ticks is the ratio of the two
    # resolutions and nothing else: 480/24 = 20 for Fermata's 24 TPQN.
    #
    # This replaces `tick_sec / out_spt`, which mixed a hard-coded output
    # tempo into a musical conversion. `tick_sec` was
    # `row_sec * ROWS_PER_BEAT / ticks_per_quarter`, i.e. 0.4 s per beat =
    # 150 BPM, while `out_spt` used the score's tempo, so every onset and
    # duration came out scaled by `score.tempo_bpm / 150`. Eight composed
    # bars became `8 * tempo_bpm / 150` bars -- four at 76 BPM, ten at 180 --
    # and the lead appeared to decay only because a fixed eight-bar window
    # was being read over a piece that had already ended. See
    # docs/lead_continuity_diagnosis.md.
    scale = ticks_per_beat / score.ticks_per_quarter
    # Tempo is playback rate, applied once to the finished arrangement rather
    # than smeared through the positions.
    out_bpm = float(score.tempo_bpm or 60.0 / (row_sec * ROWS_PER_BEAT))

    # Group by destination channel, so merging two accompaniment tracks onto
    # P2 is visible as a merge rather than appearing as lost notes.
    per_channel: dict[str, list[tuple[int, int, int, int, bool]]] = {}
    seen_roles: dict[str, int] = {}
    for index, track in enumerate(score.tracks):
        role = roles.get(index)
        if role is None:
            continue
        channel = "NO" if role == PERCUSSION else CHANNEL_FOR_ROLE[role]
        seen_roles[channel] = seen_roles.get(channel, 0) + 1
        if seen_roles[channel] > 1:
            report.n_tracks_merged += 1
            report.warnings.append(f"track {index} merged onto {channel}")
        report.role_assignment[str(index)] = f"{role}->{channel}"

        for note in track.notes:
            report.n_events_in += 1
            pitch = note.pitch
            transposed = False
            while pitch < PITCH_MIN:
                pitch += 12
                transposed = True
            while pitch > PITCH_MAX:
                pitch -= 12
                transposed = True
            if not PITCH_MIN <= pitch <= PITCH_MAX:
                report.n_out_of_range_dropped += 1
                continue
            if transposed:
                report.n_transposed += 1
            per_channel.setdefault(channel, []).append(
                (
                    int(round(note.tick * scale)),
                    pitch,
                    max(1, int(round(note.duration * scale))),
                    note.velocity,
                    False,
                )
            )
        for hit in track.percussion:
            report.n_events_in += 1
            value, approximated = _snap_drum(hit.key)
            if approximated:
                report.n_percussion_approximated += 1
            per_channel.setdefault("NO", []).append(
                (
                    int(round(hit.tick * scale)),
                    value,
                    max(1, int(round(hit.duration * scale))),
                    hit.velocity,
                    True,
                )
            )

    events: list[Note | Noise] = []
    # One row in destination ticks, and the sub-row slot an arpeggio steps by.
    # Both are musical: a row is a quarter of a beat, so this no longer
    # depends on real time and an arpeggio steps by the same musical interval
    # at every tempo.
    row_out = max(1, ticks_per_beat // ROWS_PER_BEAT)
    slot_out = max(1, row_out // max(1, divisions))

    for channel, group in per_channel.items():
        keep_high = channel == "P1"
        by_tick: dict[int, list[tuple[int, int, int, int, bool]]] = {}
        for item in group:
            by_tick.setdefault(item[0], []).append(item)

        chosen: list[tuple[int, int, int, int, bool]] = []
        for tick in sorted(by_tick):
            candidates = by_tick[tick]
            if len(candidates) == 1:
                chosen.append(candidates[0])
                continue
            # Roll the chord across the row's sub-row slots: the harmony is
            # kept as an arpeggio instead of being thrown away.
            ordered = sorted(candidates, key=lambda c: c[1], reverse=keep_high)
            room = divisions
            fits, overflow = ordered[:room], ordered[room:]
            if overflow:
                report.n_polyphony_dropped += len(overflow)
            if len(fits) > 1:
                report.n_polyphony_arpeggiated += len(fits) - 1
            for step, item in enumerate(fits):
                at = item[0] + step * slot_out
                # An arpeggio note lasts until the next one, so the channel
                # is never asked to sound two at once.
                length = slot_out if step < len(fits) - 1 else item[2]
                chosen.append((at, item[1], max(1, length), item[3], item[4]))
        chosen.sort(key=lambda c: c[0])

        # Voice stealing: truncate at the next onset on this channel.
        for current, following in zip(chosen, chosen[1:] + [None], strict=False):
            tick, value, duration, velocity, is_drum = current
            if following is not None:
                room = following[0] - tick
                if room < 1:
                    room = 1
                if duration > room:
                    duration = room
                    report.n_stolen_truncated += 1
            if is_drum:
                events.append(
                    Noise(
                        channel_id=channel,
                        tick=tick,
                        value=value,
                        duration=duration,
                        velocity=velocity,
                    )
                )
            else:
                events.append(
                    Note(
                        channel_id=channel,
                        tick=tick,
                        pitch=value,
                        duration=duration,
                        velocity=velocity,
                    )
                )

    composition = Composition(hardware_profile="nes", ticks_per_beat=ticks_per_beat, events=events)
    from rse.events.schema import TempoChange, TimeSignatureChange

    # The score's own tempo, so the arrangement plays at the tempo that was
    # requested. Previously this wrote the `row_sec`-implied 150 BPM, which
    # combined with the tempo-scaled positions to make real duration rise
    # with tempo instead of falling.
    composition.tempo_map = [TempoChange(0, int(round(60_000_000 / out_bpm)))]
    composition.time_signatures = [TimeSignatureChange(0, score.beats_per_bar, 4)]
    composition.normalize()
    report.n_events_out = len(events)
    return composition, report
