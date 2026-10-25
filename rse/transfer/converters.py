"""D4 events to a model-native score and back.

What is being measured, and why each loss is kept separate
----------------------------------------------------------
Three different things degrade a composition on its way through an external
model, and conflating them would make the result useless for choosing an
architecture:

1. **Quantization.** RSE places onsets on `beat/16`. A tokenizer with `T`
   ticks per quarter can hold that exactly only when `T` is a multiple of 16.
   Fermata's 24 is not, so `beat/16` lands on half-integer ticks.
2. **Expressiveness.** Things the representation cannot say at any
   resolution: that a channel is monophonic, that a noise period is a timbre
   rather than a drum choice, that four rapid pitches are one chord gesture.
3. **Arrangement.** Collapsing a general score back onto four hardware voices.

This module covers 1 and 2. Arrangement is a separate stage, so that a failure
there cannot be blamed on the composer or vice versa.

The noise channel is the sharpest case. NES noise has 16 period values, which
a general model represents as General MIDI drum keys out of roughly 47. The
mapping is injective by construction here, so a clean round trip is possible
*in principle* -- which is exactly what makes Fermata's documented +1 GM key
shift measurable rather than hypothetical: write key `k`, read `k + 1`, and
the period that comes back is a different one, or none.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rse.events.schema import Composition, Noise, Note
from rse.transfer.score import (
    NES_ROLES,
    PERCUSSION,
    Score,
    ScoreNote,
    ScorePercussion,
    Track,
)

#: Rows to a beat, matching `rse.tokenizers.tokenizer.ROWS_PER_BEAT`.
ROWS_PER_BEAT = 4

#: General MIDI programs for the three pitched channels. Chosen as plausible
#: timbres, not as a claim about NES sound: the program is carried so a
#: general model has something to condition on, and `Track.role` is what the
#: arranger reads back.
NES_PROGRAMS = {"P1": 80, "P2": 81, "TR": 38}

#: NES noise period to General MIDI drum key. Injective, and spaced two keys
#: apart on purpose.
#:
#: Adjacent keys would be the natural choice -- kick, snare, toms, hats in
#: order -- but they make Fermata's documented +1 key shift *silent*: the
#: shifted key is another valid entry, so 15 of 16 periods come back as a
#: different period and nothing reports a problem. Spacing by two puts every
#: shifted key outside the mapping, turning all 16 into detectable losses.
#:
#: For a fidelity audit, percussion that is visibly absent beats percussion
#: that is quietly wrong. The cost is that a GM-trained model sees some
#: less-common kit pieces, which is the right trade while the shift is
#: unfixed.
NOISE_TO_GM = {period: 36 + 2 * (period - 1) for period in range(1, 17)}
GM_TO_NOISE = {gm: noise for noise, gm in NOISE_TO_GM.items()}


@dataclass
class ConversionLoss:
    """What a single conversion could not carry."""

    n_events_in: int = 0
    n_events_out: int = 0
    n_onset_collisions: int = 0
    n_percussion_unmapped: int = 0
    n_percussion_shifted_lost: int = 0
    n_duration_zeroed: int = 0
    notes: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.notes is None:
            self.notes = []

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_events_in": self.n_events_in,
            "n_events_out": self.n_events_out,
            "n_lost": self.n_events_in - self.n_events_out,
            "n_onset_collisions": self.n_onset_collisions,
            "n_percussion_unmapped": self.n_percussion_unmapped,
            "n_percussion_shifted_lost": self.n_percussion_shifted_lost,
            "n_duration_zeroed": self.n_duration_zeroed,
            "notes": list(self.notes[:20]),
        }


def seconds_per_tick(composition: Composition) -> float:
    end = composition.end_tick or 1
    total = composition.time.seconds(end)
    return total / end if total else 0.0


def composition_to_score(
    composition: Composition,
    *,
    ticks_per_quarter: int,
    tempo_bpm: float,
    row_sec: float,
) -> tuple[Score, ConversionLoss]:
    """Quantize a D4-decoded composition onto a model-native tick grid.

    Time is converted through *seconds*, not through the source tick scale:
    a decoded composition carries nominal ticks and recovers its real rate
    from the Tempo token, so reusing its ticks would smuggle a tempo-label
    error into a quantization measurement.
    """
    loss = ConversionLoss()
    beat_sec = row_sec * ROWS_PER_BEAT
    if beat_sec <= 0 or ticks_per_quarter < 1:
        raise ValueError(f"bad grid: beat {beat_sec}s, {ticks_per_quarter} TPQ")
    spt = seconds_per_tick(composition)
    tick_sec = beat_sec / ticks_per_quarter

    tracks: dict[str, Track] = {}
    for channel_id, role in NES_ROLES.items():
        tracks[channel_id] = Track(
            role=role,
            channel_id=channel_id,
            program=NES_PROGRAMS.get(channel_id),
            monophonic=True,
        )

    taken: dict[tuple[str, int], int] = {}
    for event in composition.sounding_events():
        channel = event.channel_id
        if channel not in tracks:
            loss.notes.append(f"channel {channel!r} has no track")
            continue
        loss.n_events_in += 1
        at = int(round(event.tick * spt / tick_sec))
        length = max(1, int(round(event.duration * spt / tick_sec)))
        if int(round(event.duration * spt / tick_sec)) < 1:
            loss.n_duration_zeroed += 1

        # A monophonic channel cannot hold two onsets at one tick. The coarser
        # grid is what creates these: two events a sub-row apart can land on
        # the same tick, and the second would overwrite the first.
        key = (channel, at)
        if key in taken:
            loss.n_onset_collisions += 1
            loss.notes.append(f"{channel} collision at tick {at}")
            continue
        taken[key] = 1

        track = tracks[channel]
        if track.role == PERCUSSION:
            value = event.value if isinstance(event, Noise) else 0
            gm = NOISE_TO_GM.get(value)
            if gm is None:
                loss.n_percussion_unmapped += 1
                loss.notes.append(f"noise period {value} has no GM key")
                continue
            track.percussion.append(
                ScorePercussion(
                    tick=at,
                    duration=length,
                    key=gm,
                    velocity=event.velocity,
                    source_value=value,
                )
            )
        else:
            pitch = event.pitch if isinstance(event, Note) else 60
            track.notes.append(
                ScoreNote(tick=at, duration=length, pitch=pitch, velocity=event.velocity)
            )

    score = Score(
        ticks_per_quarter=ticks_per_quarter,
        tempo_bpm=tempo_bpm,
        tracks=[t for t in tracks.values() if t.n_events],
        provenance={"row_sec": row_sec, "source": "d4"},
    )
    loss.n_events_out = score.n_events
    score.warnings = list(loss.notes)
    return score, loss


def score_to_composition(
    score: Score,
    *,
    row_sec: float,
    ticks_per_beat: int = 480,
    drum_key_shift: int = 0,
) -> tuple[Composition, ConversionLoss]:
    """Rebuild NES events from a model-native score.

    `drum_key_shift` models Fermata's documented defect: its MusicXML
    `midi-unpitched` values are 1-based but were read as 0-based, so drum
    notes come out one GM key high. Passing 1 reproduces it, which is how the
    audit measures its effect on percussion identity instead of assuming it is
    recoverable.
    """
    loss = ConversionLoss()
    beat_sec = row_sec * ROWS_PER_BEAT
    tick_sec = beat_sec / score.ticks_per_quarter
    # Nominal destination scale, as `decode` uses.
    out_spt = (60.0 / (score.tempo_bpm or 120.0)) / ticks_per_beat

    events: list[Note | Noise] = []
    for track in score.tracks:
        channel = track.channel_id or _channel_for_role(track.role)
        for note in track.notes:
            loss.n_events_in += 1
            events.append(
                Note(
                    channel_id=channel,
                    tick=int(round(note.tick * tick_sec / out_spt)),
                    pitch=note.pitch,
                    duration=max(1, int(round(note.duration * tick_sec / out_spt))),
                    velocity=note.velocity,
                )
            )
        for hit in track.percussion:
            loss.n_events_in += 1
            observed = hit.key + drum_key_shift
            value = GM_TO_NOISE.get(observed)
            if value is None:
                # The shift moved the key outside the kit we mapped, so the
                # hit cannot be read back at all.
                loss.n_percussion_shifted_lost += 1
                loss.notes.append(f"GM key {observed} (shift {drum_key_shift}) is unmapped")
                continue
            events.append(
                Noise(
                    channel_id=channel,
                    tick=int(round(hit.tick * tick_sec / out_spt)),
                    value=value,
                    duration=max(1, int(round(hit.duration * tick_sec / out_spt))),
                    velocity=hit.velocity,
                )
            )

    composition = Composition(hardware_profile="nes", ticks_per_beat=ticks_per_beat, events=events)
    from rse.events.schema import TempoChange, TimeSignatureChange

    composition.tempo_map = [TempoChange(0, int(round(60_000_000 / (score.tempo_bpm or 120.0))))]
    composition.time_signatures = [TimeSignatureChange(0, score.beats_per_bar, 4)]
    composition.normalize()
    loss.n_events_out = len(events)
    return composition, loss


def _channel_for_role(role: str) -> str:
    for channel, mapped in NES_ROLES.items():
        if mapped == role:
            return channel
    return "P1"


def percussion_round_trip(drum_key_shift: int = 0) -> dict[str, Any]:
    """Which noise periods survive a GM drum round trip under a key shift.

    Reported on its own because percussion is 27.2% of NES-MDB note-ons, and
    a defect that silently rewrites which drum plays would not show up in any
    onset or duration measurement.
    """
    recovered: dict[int, int | None] = {}
    for period, gm in sorted(NOISE_TO_GM.items()):
        recovered[period] = GM_TO_NOISE.get(gm + drum_key_shift)
    exact = sum(1 for p, back in recovered.items() if back == p)
    wrong = sum(1 for p, back in recovered.items() if back is not None and back != p)
    lost = sum(1 for back in recovered.values() if back is None)
    return {
        "drum_key_shift": drum_key_shift,
        "n_periods": len(NOISE_TO_GM),
        "n_exact": exact,
        "n_wrong_period": wrong,
        "n_lost": lost,
        "mapping": {str(p): back for p, back in recovered.items()},
    }
