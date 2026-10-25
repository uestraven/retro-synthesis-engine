"""A **reconstructed** codec for Fermata 0.2B's native token stream.

This is not the author's implementation. `haster/Fermata-0.2B-v1` ships
`tokenizer/vocab.json` and `tokenizer/bpe.json` but **no Python encoder or
decoder** -- the repository contains no `.py` files at all, and the model card
says only that decoding uses "this project's pipeline", without linking one.
Every claim made with this module must therefore be reported as measured
through a reconstruction, never as the model's own fidelity.

How the grammar was recovered
-----------------------------
Not guessed. The checkpoint was sampled from `<BOS>` with the dead BPE tokens
suppressed, and it emitted its own structure:

    <BAR> <TRK_n> ( <BEAT_b> <SUBPOS_s> <PITCH_p> <DURATION_d> <VELOCITY_v> )*
          <TRK_m> ( ... )*  <BAR> ...

Two details that a reasonable guess would have got wrong:

- **DURATION precedes VELOCITY.** Most REMI variants put velocity first.
- **`<BEAT_>` is the beat *within the bar*, not an absolute beat index.** The
  sampled stream runs `BEAT_0 … BEAT_1 … <BAR> BEAT_0`, so it resets. The
  0..223 range exists to allow very long bars, not long pieces.

The header order is the card's: `BOS [GENRE] [ENSEMBLE] TEMPO KEY METER
<PLAN> {TRACK|DRUM}* <PLAN_END> (BAR ...)* EOS`, and `<TRK_i>` indexes the
plan's declaration order.

The BPE defect
--------------
7,348 of 9,507 tokens are `<BPE_*>`, and the model card states the merges were
never applied to the training shards, so the model trained on the atomic
stream. This module therefore **never emits a BPE token and suppresses them at
sampling**: the effective vocabulary is 2,159. Applying the saved merges would
feed the model ids it has never seen.

Time base
---------
`<SUBPOS_>` has exactly 24 values, which independently confirms the 24
ticks-per-quarter figure from the model card. Durations are counted in the
same 1/24-beat unit, and only 811 of the 14,400 possible lengths exist as
tokens, so a duration must be snapped to the nearest available one rather than
assumed representable.
"""

from __future__ import annotations

import json
import re
from bisect import bisect_left
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rse.transfer.score import (
    PERCUSSION,
    Score,
    ScoreNote,
    ScorePercussion,
    Track,
)

#: Sub-beat divisions in the native grid, from the `<SUBPOS_*>` family.
SUBPOS = 24

#: Marks every artifact produced through this module, so a result can never be
#: mistaken for one obtained with the author's own pipeline.
RECONSTRUCTION_ID = "rse-fermata-reconstruction/1.0.0"


@dataclass
class Vocab:
    """Fermata's token table, with the families this codec needs."""

    token_to_id: dict[str, int]
    id_to_token: dict[int, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}
        self.bpe_ids = sorted(i for t, i in self.token_to_id.items() if t.startswith("<BPE_"))
        self.durations = self._numeric("<DURATION_")
        self.beats = set(self._numeric("<BEAT_"))
        self.pitches = set(self._numeric("<PITCH_"))
        self.drum_notes = set(self._numeric("<DRUM_NOTE_"))
        self.velocities = self._numeric("<VELOCITY_")
        self.tempos = self._numeric("<TEMPO_")
        self.programs = set(self._numeric("<TRACK_"))
        self.trk_slots = set(self._numeric("<TRK_"))

    def _numeric(self, prefix: str) -> list[int]:
        out: list[int] = []
        for token in self.token_to_id:
            match = re.fullmatch(re.escape(prefix) + r"(-?\d+)>", token)
            if match:
                out.append(int(match.group(1)))
        return sorted(out)

    @classmethod
    def load(cls, path: Path) -> Vocab:
        return cls(token_to_id=json.loads(Path(path).read_text()))

    def id_of(self, token: str) -> int:
        try:
            return self.token_to_id[token]
        except KeyError as exc:
            raise KeyError(f"{token!r} is not in Fermata's vocabulary") from exc

    def has(self, token: str) -> bool:
        return token in self.token_to_id

    def nearest_duration(self, units: int) -> tuple[int, int]:
        """Closest representable duration, and the error in 1/24-beat units.

        Only 811 of 14,400 lengths have tokens, so snapping is required and
        the error is returned rather than discarded.
        """
        if not self.durations:
            raise ValueError("vocabulary has no DURATION tokens")
        units = max(1, units)
        index = bisect_left(self.durations, units)
        candidates = [self.durations[i] for i in (index - 1, index) if 0 <= i < len(self.durations)]
        best = min(candidates, key=lambda d: abs(d - units))
        return best, abs(best - units)


@dataclass
class EncodeReport:
    """What the native representation could not carry."""

    n_notes: int = 0
    n_percussion: int = 0
    n_pitch_out_of_range: int = 0
    n_drum_unmapped: int = 0
    n_duration_snapped: int = 0
    duration_error_units: int = 0
    n_beat_out_of_range: int = 0
    n_tracks_over_limit: int = 0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        total = self.n_notes + self.n_percussion
        return {
            "reconstruction": RECONSTRUCTION_ID,
            "n_events": total,
            "n_notes": self.n_notes,
            "n_percussion": self.n_percussion,
            "n_pitch_out_of_range": self.n_pitch_out_of_range,
            "n_drum_unmapped": self.n_drum_unmapped,
            "n_duration_snapped": self.n_duration_snapped,
            "mean_duration_error_units": (
                round(self.duration_error_units / total, 4) if total else 0.0
            ),
            "n_beat_out_of_range": self.n_beat_out_of_range,
            "n_tracks_over_limit": self.n_tracks_over_limit,
            "warnings": self.warnings[:20],
        }


def _meter_token(vocab: Vocab, beats_per_bar: int) -> str:
    for candidate in (f"<METER_{beats_per_bar}/4>", "<METER_4/4>"):
        if vocab.has(candidate):
            return candidate
    raise KeyError(f"no METER token for {beats_per_bar}/4")


def _tempo_token(vocab: Vocab, bpm: float) -> str:
    nearest = min(vocab.tempos, key=lambda t: abs(t - bpm))
    return f"<TEMPO_{nearest}>"


def encode(
    score: Score,
    *,
    vocab: Vocab,
    genre: str = "soundtrack",
    ensemble: str = "band",
    key: str = "0:major",
    n_bars: int | None = None,
) -> tuple[list[int], EncodeReport]:
    """Encode a score into Fermata's native atomic token stream.

    `score.ticks_per_quarter` must be 24: the native grid has exactly that
    many `<SUBPOS_>` values, and silently resampling here would hide the
    quantization this whole study is measuring.
    """
    if score.ticks_per_quarter != SUBPOS:
        raise ValueError(
            f"Fermata's grid is {SUBPOS} ticks per quarter, got "
            f"{score.ticks_per_quarter}; convert before encoding so the "
            f"quantization is measured rather than hidden"
        )
    report = EncodeReport()
    out: list[str] = ["<BOS>"]
    for token in (f"<GENRE_{genre}>", f"<ENSEMBLE_{ensemble}>"):
        if not vocab.has(token):
            raise KeyError(f"{token!r} is not a Fermata conditioning value")
        out.append(token)
    out.append(_tempo_token(vocab, score.tempo_bpm))
    key_token = f"<KEY_{key}>"
    out.append(key_token if vocab.has(key_token) else "<KEY_0:none>")
    out.append(_meter_token(vocab, score.beats_per_bar))

    # The plan declares each track once, and <TRK_i> later refers to its
    # position here.
    tracks = [t for t in score.tracks if t.n_events]
    if len(tracks) > len(vocab.trk_slots):
        report.n_tracks_over_limit = len(tracks) - len(vocab.trk_slots)
        report.warnings.append(f"{len(tracks)} tracks but only {len(vocab.trk_slots)} TRK slots")
        tracks = tracks[: len(vocab.trk_slots)]
    out.append("<PLAN>")
    for track in tracks:
        if track.role == PERCUSSION:
            out.append("<DRUM_TRACK>")
        else:
            program = track.program if track.program in vocab.programs else 0
            out.append(f"<TRACK_{program}>")
    out.append("<PLAN_END>")

    ticks_per_bar = SUBPOS * score.beats_per_bar
    # Group every event by bar, then by track, which is the bar-major order
    # the model's own sampled output showed.
    per_bar: dict[int, dict[int, list[tuple[int, int, int, int, bool]]]] = {}
    for slot, track in enumerate(tracks):
        for note in track.notes:
            if note.pitch not in vocab.pitches:
                report.n_pitch_out_of_range += 1
                continue
            report.n_notes += 1
            bar, within = divmod(note.tick, ticks_per_bar)
            per_bar.setdefault(bar, {}).setdefault(slot, []).append(
                (within, note.pitch, note.duration, note.velocity, False)
            )
        for hit in track.percussion:
            if hit.key not in vocab.drum_notes:
                report.n_drum_unmapped += 1
                report.warnings.append(f"drum key {hit.key} has no DRUM_NOTE token")
                continue
            report.n_percussion += 1
            bar, within = divmod(hit.tick, ticks_per_bar)
            per_bar.setdefault(bar, {}).setdefault(slot, []).append(
                (within, hit.key, hit.duration, hit.velocity, True)
            )

    last_bar = max(per_bar, default=-1)
    total_bars = n_bars if n_bars is not None else last_bar + 1
    for bar in range(total_bars):
        out.append("<BAR>")
        for slot in sorted(per_bar.get(bar, {})):
            out.append(f"<TRK_{slot}>")
            for within, value, duration, velocity, is_drum in sorted(per_bar[bar][slot]):
                beat, subpos = divmod(within, SUBPOS)
                if beat not in vocab.beats:
                    report.n_beat_out_of_range += 1
                    continue
                snapped, error = vocab.nearest_duration(duration)
                if error:
                    report.n_duration_snapped += 1
                    report.duration_error_units += error
                bucket = min(
                    vocab.velocities,
                    key=lambda v: abs(v - velocity * len(vocab.velocities) // 128),
                )
                out.extend(
                    (
                        f"<BEAT_{beat}>",
                        f"<SUBPOS_{subpos}>",
                        f"<DRUM_NOTE_{value}>" if is_drum else f"<PITCH_{value}>",
                        f"<DURATION_{snapped}>",
                        f"<VELOCITY_{bucket}>",
                    )
                )
    out.append("<EOS>")
    return [vocab.id_of(t) for t in out], report


@dataclass
class DecodeReport:
    n_tokens: int = 0
    n_skipped: int = 0
    n_events: int = 0
    n_bars: int = 0
    hit_eos: bool = False
    #: A stream that never declared a plan cannot have its tracks typed, so
    #: percussion and pitched tracks become indistinguishable.
    had_plan: bool = False
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "reconstruction": RECONSTRUCTION_ID,
            "n_tokens": self.n_tokens,
            "n_skipped": self.n_skipped,
            "n_events": self.n_events,
            "n_bars": self.n_bars,
            "hit_eos": self.hit_eos,
            "had_plan": self.had_plan,
            "warnings": self.warnings[:20],
        }


def decode(ids: list[int], *, vocab: Vocab) -> tuple[Score, DecodeReport]:
    """Rebuild a score from a native token stream.

    Forgiving, like RSE's own decoder: model output is malformed before it is
    musical, and a decoder that raised would make a checkpoint impossible to
    listen to. Skipped tokens are counted.
    """
    report = DecodeReport(n_tokens=len(ids))
    tokens = [vocab.id_to_token.get(i, "<UNK>") for i in ids]

    tempo, beats_per_bar = 120.0, 4
    plan: list[str] = []
    in_plan = False
    bar = -1
    slot = 0
    beat = subpos = 0
    pending: list[str] = []
    events: dict[int, list[tuple[int, int, int, int, bool]]] = {}

    def flush() -> None:
        """Emit a note once its five fields have arrived."""
        nonlocal pending
        if len(pending) < 3:
            pending = []
            return
        kind = pending[0]
        is_drum = kind.startswith("<DRUM_NOTE_")
        value = int(re.findall(r"-?\d+", kind)[0])
        duration = int(re.findall(r"-?\d+", pending[1])[0]) if len(pending) > 1 else 6
        velocity = int(re.findall(r"-?\d+", pending[2])[0]) if len(pending) > 2 else 8
        tick = bar * SUBPOS * beats_per_bar + beat * SUBPOS + subpos
        events.setdefault(slot, []).append((tick, value, duration, velocity, is_drum))
        report.n_events += 1
        pending = []

    for token in tokens:
        if token == "<EOS>":
            flush()
            report.hit_eos = True
            break
        if token.startswith("<BPE_"):
            # Never legitimate: the model was trained on the atomic stream.
            report.n_skipped += 1
            continue
        if token == "<PLAN>":
            in_plan, report.had_plan = True, True
        elif token == "<PLAN_END>":
            in_plan = False
        elif in_plan and (token.startswith("<TRACK_") or token == "<DRUM_TRACK>"):
            plan.append(token)
        elif token.startswith("<TEMPO_"):
            tempo = float(re.findall(r"\d+", token)[0])
        elif token.startswith("<METER_"):
            found = re.findall(r"(\d+)/(\d+)", token)
            if found:
                beats_per_bar = int(found[0][0])
        elif token == "<BAR>":
            flush()
            bar += 1
            report.n_bars = bar + 1
        elif token.startswith("<TRK_"):
            flush()
            slot = int(re.findall(r"\d+", token)[0])
        elif token.startswith("<BEAT_"):
            flush()
            beat = int(re.findall(r"\d+", token)[0])
        elif token.startswith("<SUBPOS_"):
            subpos = int(re.findall(r"\d+", token)[0])
        elif token.startswith(("<PITCH_", "<DRUM_NOTE_")):
            flush()
            pending = [token]
        elif token.startswith("<DURATION_") and pending:
            pending.append(token)
        elif token.startswith("<VELOCITY_") and len(pending) == 2:
            pending.append(token)
            flush()
        else:
            report.n_skipped += 1
    flush()

    if bar < 0:
        report.warnings.append("no <BAR> token: the stream carries no music")

    tracks: list[Track] = []
    for index in sorted(events):
        declared = plan[index] if index < len(plan) else None
        is_drum_track = declared == "<DRUM_TRACK>"
        if declared is None:
            report.warnings.append(
                f"TRK_{index} was never declared in the plan; track type unknown"
            )
        program = (
            int(re.findall(r"\d+", declared)[0])
            if declared and declared.startswith("<TRACK_")
            else None
        )
        track = Track(
            role=PERCUSSION if is_drum_track else "melody",
            program=program,
            monophonic=False,
        )
        for tick, value, duration, velocity, was_drum in events[index]:
            loud = int((velocity + 0.5) * 128 / 16)
            if is_drum_track or was_drum:
                track.percussion.append(
                    ScorePercussion(tick=tick, duration=duration, key=value, velocity=loud)
                )
            else:
                track.notes.append(
                    ScoreNote(tick=tick, duration=duration, pitch=value, velocity=loud)
                )
        tracks.append(track)

    score = Score(
        ticks_per_quarter=SUBPOS,
        tempo_bpm=tempo,
        beats_per_bar=beats_per_bar,
        tracks=tracks,
        provenance={"source": RECONSTRUCTION_ID, "plan": list(plan)},
    )
    score.warnings = list(report.warnings)
    return score, report
