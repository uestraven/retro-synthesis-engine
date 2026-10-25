"""Encoding RSE-EF to tokens and back.

Tokenization is where quantization finally becomes destructive, and that is a
deliberate change of contract. The intake stage keeps source ticks untouched
and treats quantization as additive (proposal 5.3); here, note starts and
durations are snapped to the grid because a model needs a finite vocabulary.
`encode` therefore reports the error it introduced, so the cost is visible
rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rse.data.timing import TempoChange, TimeSignatureChange
from rse.events.pulse import detect_pulse
from rse.events.schema import Composition, Noise, Note, SectionSpan
from rse.jsonio import read_json, write_json
from rse.tokenizers.grammar import TokenGrammar
from rse.tokenizers.vocab import (
    BAR,
    BOS,
    CHANNEL,
    DURATION,
    ENERGY,
    EOS,
    HW,
    LENGTH,
    LOOP_END,
    LOOP_START,
    MOOD,
    MOTIF,
    NOISE,
    OFFSET,
    PITCH,
    POSITION,
    SECTION,
    TEMPO,
    VELOCITY,
    TokenizerConfig,
    Vocabulary,
    build_vocabulary,
)

DEFAULT_TICKS_PER_BEAT = 480

#: The tracker convention NES music is authored under. A music engine advances
#: one row every `Tick Speed` video frames, and a row is a sixteenth: 4 rows to
#: a beat, 16 to a bar in 4/4. FamiTracker's defaults are Tick Speed 6 at 60 Hz,
#: giving a 0.1 s row and 150 BPM, and the corpus agrees — of 160 NES-MDB pieces
#: with a whole-frame grid, the modal row is 6 frames (150.2 BPM) in 43 of them,
#: then 8 (112.7 BPM), 10 (90.1), 7 (128.8), 5 (180.3), 12 (75.1), 4 (225.4).
#: Those are Tick Speed values, not arbitrary periods.
#:
#: So the bar is not a quantity to detect. `rse.events.bar` was written to
#: measure it and was weak at exactly that; its common 32-row repeat is a
#: two-bar phrase, which is a form signal rather than a bar length.
ROWS_PER_BEAT = 4
ROWS_PER_BAR = 16


@dataclass
class Conditioning:
    """The musical intent prefixed to a sequence (proposal 6.4)."""

    mood: str = "unknown"
    energy: str = "unknown"
    tempo_bpm: float = 120.0
    n_bars: int = 16
    #: Requested motif recurrence. `unspecified` is the neutral value, so
    #: an unconditioned sequence has the same prefix length as a
    #: conditioned one and the arms differ only in information content.
    motif: str = "unspecified"

    def to_dict(self) -> dict[str, Any]:
        return {
            "mood": self.mood,
            "energy": self.energy,
            "tempo_bpm": self.tempo_bpm,
            "n_bars": self.n_bars,
            "motif": self.motif,
        }


@dataclass(frozen=True)
class GridFit:
    """The row grid one composition is encoded against."""

    #: Ticks in one row. Fractional: a fitted row is not a whole number of
    #: ticks, and rounding it accumulates drift across a piece.
    step_ticks: float
    #: Where the first row starts, in ticks.
    phase_ticks: float
    #: Whether the row was fitted, or fell back to one step per source frame.
    aligned: bool
    #: Row length in seconds, which fixes the tempo.
    period_sec: float = 0.0

    def row_of(self, tick: int) -> int:
        return int(round((tick - self.phase_ticks) / self.step_ticks))

    def tick_of(self, row: int) -> int:
        return int(round(self.phase_ticks + row * self.step_ticks))

    def slot_of(self, tick: int, divisions: int) -> int:
        """Nearest sub-row slot, counting `divisions` slots to the row.

        `divisions=1` is `row_of` exactly, so codec A's arithmetic is
        unchanged rather than merely equivalent.
        """
        if divisions < 1:
            raise ValueError(f"divisions must be at least 1, got {divisions}")
        if divisions == 1:
            return self.row_of(tick)
        return int(round((tick - self.phase_ticks) * divisions / self.step_ticks))

    def tick_of_slot(self, slot: int, divisions: int) -> int:
        if divisions < 1:
            raise ValueError(f"divisions must be at least 1, got {divisions}")
        if divisions == 1:
            return self.tick_of(slot)
        return int(round(self.phase_ticks + slot * self.step_ticks / divisions))

    @property
    def bpm(self) -> float:
        """Quarter-note tempo implied by the row length.

        Four rows to a beat, so `BPM = 15 / row_seconds`. This matches
        FamiTracker's `BPM = 6 * Module Tempo / Tick Speed` at its defaults.
        """
        return 60.0 / (self.period_sec * ROWS_PER_BEAT) if self.period_sec > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_ticks": round(self.step_ticks, 4),
            "phase_ticks": round(self.phase_ticks, 4),
            "aligned": self.aligned,
            "row_sec": round(self.period_sec, 5),
            "bpm": round(self.bpm, 2),
        }


@dataclass
class EncodedComposition:
    """A token sequence plus what encoding it cost."""

    ids: list[int]
    tokens: list[str]
    conditioning: Conditioning
    n_events: int = 0
    n_encoded: int = 0
    max_start_error_ticks: int = 0
    total_start_error_ticks: int = 0
    n_duration_clamped: int = 0
    n_out_of_range: int = 0
    #: Events folded into an earlier note on the same row and channel,
    #: which are vibrato and arpeggio writes rather than notes.
    n_rows_collapsed: int = 0
    warnings: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.ids)

    @property
    def mean_start_error_ticks(self) -> float:
        return self.total_start_error_ticks / self.n_encoded if self.n_encoded else 0.0

    def stats(self) -> dict[str, Any]:
        return {
            "n_tokens": len(self.ids),
            "n_events": self.n_events,
            "n_encoded": self.n_encoded,
            "n_dropped": self.n_events - self.n_encoded,
            "max_start_error_ticks": self.max_start_error_ticks,
            "mean_start_error_ticks": round(self.mean_start_error_ticks, 3),
            "n_duration_clamped": self.n_duration_clamped,
            "n_out_of_range": self.n_out_of_range,
            "n_rows_collapsed": self.n_rows_collapsed,
            "conditioning": self.conditioning.to_dict(),
            "warnings": self.warnings[:20],
        }


@dataclass
class DecodedComposition:
    composition: Composition
    n_tokens_read: int = 0
    n_tokens_skipped: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return self.n_tokens_skipped == 0


class RseTokenizer:
    """REMI-inspired tokenizer with RSE's channel and noise semantics."""

    def __init__(
        self, config: TokenizerConfig | None = None, vocabulary: Vocabulary | None = None
    ) -> None:
        if vocabulary is not None:
            self.vocabulary = vocabulary
            self.config = vocabulary.config
        else:
            self.config = config or TokenizerConfig()
            self.vocabulary = build_vocabulary(self.config)

    def __len__(self) -> int:
        return len(self.vocabulary)

    # -- grid helpers -----------------------------------------------------

    def ticks_per_step(self, ticks_per_beat: int) -> int:
        """Ticks in one grid step."""
        per_bar = ticks_per_beat * self.config.beats_per_bar
        step = per_bar // self.config.steps_per_bar
        if step <= 0:
            raise ValueError(
                f"resolution {ticks_per_beat} ticks/beat is too coarse for "
                f"{self.config.steps_per_bar} steps per bar"
            )
        return step

    def fit_grid(self, composition: Composition) -> GridFit:
        """Find the tracker row a composition is written on.

        Falls back to one step per source frame when no row is found, which is
        the old fixed-grid encoding. The fallback matters: NES-MDB's exprsco
        release samples at 24 Hz, and a row is exactly representable there only
        when it is a multiple of 2.5 frames, which 82% of rows are not — so on
        that release the fit often declines and the legacy grid stands in.
        """
        legacy = GridFit(
            step_ticks=float(self.ticks_per_step(composition.ticks_per_beat)),
            phase_ticks=0.0,
            aligned=False,
        )
        if not self.config.aligned_grid:
            return legacy

        pulse = detect_pulse(composition)
        if not pulse.found or pulse.period_ticks <= 0:
            return legacy

        fit = GridFit(
            step_ticks=pulse.period_ticks,
            phase_ticks=pulse.phase_ticks,
            aligned=True,
            period_sec=pulse.period_sec,
        )
        # The Tempo token is the only thing `decode` has to rebuild absolute
        # time from rows, so a tempo the vocabulary cannot express would play
        # the piece at the wrong speed. Declining is better than clamping.
        if not self.config.tempo_min <= fit.bpm <= self.config.tempo_max:
            return legacy
        return fit

    def velocity_bucket(self, velocity: int) -> int:
        buckets = self.config.velocity_buckets
        return max(0, min(buckets - 1, velocity * buckets // 128))

    def bucket_velocity(self, bucket: int) -> int:
        buckets = self.config.velocity_buckets
        return max(1, min(127, int((bucket + 0.5) * 128 / buckets)))

    def tempo_token_value(self, bpm: float) -> int:
        values = self.config.tempo_values
        return min(values, key=lambda value: abs(value - bpm))

    def motif_token_value(self, motif: str) -> str:
        """A valid motif value, falling back to `unspecified`.

        Never falls back to `low`: an unrecognised request is an absent
        request, not a request for little recurrence.
        """
        return motif if motif in self.config.motif_values else "unspecified"

    def length_token_value(self, n_bars: int) -> int:
        return min(self.config.lengths, key=lambda length: abs(length - n_bars))

    # -- encode -----------------------------------------------------------

    def encode(
        self,
        composition: Composition,
        *,
        conditioning: Conditioning | None = None,
        fit: GridFit | None = None,
    ) -> EncodedComposition:
        """Turn a composition into a token sequence.

        `fit` lets a caller supply the grid instead of having it refitted.
        A training window should pass its parent composition's row: a short
        slice can fit a different row from the piece it came from, and a
        corpus whose windows disagree about the grid is the confusion this
        change exists to remove.
        """
        fit = fit if fit is not None else self.fit_grid(composition)
        step_ticks = fit.step_ticks
        steps_per_bar = self.config.steps_per_bar
        onset_divisions = self.config.onset_divisions
        duration_divisions = self.config.duration_divisions
        # Slots per bar at onset resolution. Codec A has one per row, so this
        # is `steps_per_bar` and every index below is the row index it was.
        slots_per_bar = steps_per_bar * onset_divisions
        sounding = composition.sounding_events()

        if conditioning is None:
            conditioning = conditioning_from_composition(composition, self.config, fit=fit)

        result = EncodedComposition(
            ids=[], tokens=[], conditioning=conditioning, n_events=len(sounding)
        )

        channel_order = {cid: i for i, cid in enumerate(self.config.channels)}
        placed: list[tuple[int, str, Note | Noise]] = []
        for event in sounding:
            if event.channel_id not in channel_order:
                result.warnings.append(
                    f"channel {event.channel_id!r} is not in the vocabulary; dropped"
                )
                continue
            value = event.pitch if isinstance(event, Note) else event.value
            if not self._value_in_vocab(event.channel_id, value):
                result.n_out_of_range += 1
                result.warnings.append(
                    f"{event.channel_id} value {value} is outside the vocabulary; dropped"
                )
                continue
            step = fit.slot_of(event.tick, onset_divisions)
            error = abs(event.tick - fit.tick_of_slot(step, onset_divisions))
            result.max_start_error_ticks = max(result.max_start_error_ticks, int(round(error)))
            result.total_start_error_ticks += int(round(error))
            placed.append((step, event.channel_id, event))

        # Anchor the first sounding row at bar 0, position 0.
        #
        # Counting rows from the fitted phase instead is not stable: the phase
        # is only defined modulo the row, so a fit can return a value just
        # under a full row where another returns just over zero, shifting every
        # position by one. Encoding the same music twice then gives two
        # different sequences, which the round-trip check caught.
        #
        # The cost is that an anacrusis is lost — a piece starting on beat four
        # is encoded as starting on a downbeat. That is acceptable because
        # `detect_pulse` finds the row grid and not the downbeat, so bar 0
        # position 0 already means "where the music starts" rather than "the
        # notated downbeat". Pinning it to the first event at least makes that
        # meaning consistent.
        if placed:
            lowest = min(step for step, _, _ in placed)
            placed = [(step - lowest, cid, event) for step, cid, event in placed]

        # Collapse to one note per channel per row. A tracker row holds a single
        # note per channel, so several source events inside one row are a
        # vibrato or arpeggio effect rather than several notes — NES-MDB is a
        # register-write transcription and carries them as separate events.
        # Keeping the first and counting the rest is correct transcription, and
        # it stops the encoder emitting a chord on a monophonic channel, which
        # the hardware cannot play.
        #
        # A sub-row codec collapses per sub-row slot instead, which is what
        # lets it keep an arpeggio's frames as separate events. The invariant
        # that matters for the hardware is unchanged either way: at most one
        # event per channel per slot, so a monophonic channel never receives
        # two simultaneous voices however fine the grid gets.
        grid: dict[tuple[int, int], list[tuple[int, Note | Noise]]] = {}
        taken: set[tuple[int, str]] = set()
        for step, cid, event in sorted(
            placed, key=lambda item: (item[0], channel_order[item[1]], item[2].tick)
        ):
            if (step, cid) in taken:
                result.n_rows_collapsed += 1
                continue
            taken.add((step, cid))
            key = (step // slots_per_bar, step % slots_per_bar)
            grid.setdefault(key, []).append((channel_order[cid], event))

        tokens: list[str] = [
            BOS,
            f"{HW}{self.config.profile_id}",
            f"{MOOD}{conditioning.mood}",
            f"{ENERGY}{conditioning.energy}",
            f"{TEMPO}{self.tempo_token_value(conditioning.tempo_bpm)}",
            f"{LENGTH}{self.length_token_value(conditioning.n_bars)}",
            f"{MOTIF}{self.motif_token_value(conditioning.motif)}",
        ]
        if composition.loop_start_tick is not None:
            tokens.append(LOOP_START)

        n_bars = max((bar for bar, _ in grid), default=-1) + 1
        n_bars = max(n_bars, conditioning.n_bars)
        # Sections are announced at the bar they open. Empty means the form is
        # unknown, and the stream simply carries no Section tokens.
        section_at_bar = (
            {
                span.start_bar: span.label
                for span in composition.sections
                if span.label in self.config.section_labels
            }
            if self.config.emit_sections
            else {}
        )
        for bar in range(n_bars):
            if bar in section_at_bar:
                tokens.append(f"{SECTION}{section_at_bar[bar]}")
            # A Bar token for every bar, including empty ones, so bar count and
            # phrase position stay recoverable from the sequence alone.
            tokens.append(BAR)
            for slot in range(slots_per_bar):
                events = grid.get((bar, slot))
                if not events:
                    continue
                # Where the four codecs diverge, and the only place they do.
                #
                # A: one Position per row.
                # B: a row Position followed by an Offset token. Both are
                #    re-emitted per offset, which is the one extra token per
                #    group that makes B strictly costlier than C.
                # C and D: a single Position spanning row * k + offset.
                if self.config.codec in ("C", "D"):
                    tokens.append(f"{POSITION}{slot}")
                else:
                    tokens.append(f"{POSITION}{slot // onset_divisions}")
                    if self.config.emits_offset_tokens:
                        tokens.append(f"{OFFSET}{slot % onset_divisions}")
                duration_unit = step_ticks / duration_divisions
                for _order, event in sorted(events, key=lambda pair: pair[0]):
                    duration_steps = int(round(event.duration / duration_unit))
                    if duration_steps < 1:
                        duration_steps = 1
                    if duration_steps > self.config.duration_token_max:
                        duration_steps = self.config.duration_token_max
                        result.n_duration_clamped += 1
                    value = event.pitch if isinstance(event, Note) else event.value
                    prefix = PITCH if isinstance(event, Note) else NOISE
                    tokens.extend(
                        (
                            f"{CHANNEL}{event.channel_id}",
                            f"{prefix}{value}",
                            f"{VELOCITY}{self.velocity_bucket(event.velocity)}",
                            f"{DURATION}{duration_steps}",
                        )
                    )
                    result.n_encoded += 1

        if composition.loop_end_tick is not None:
            tokens.append(LOOP_END)
        tokens.append(EOS)

        result.tokens = tokens
        result.ids = [self.vocabulary.id_of(token) for token in tokens]
        return result

    def _value_in_vocab(self, channel_id: str, value: int) -> bool:
        if self.config.is_pitched(channel_id):
            return self.config.pitch_min <= value <= self.config.pitch_max
        return self.config.noise_min <= value <= self.config.noise_max

    # -- decode -----------------------------------------------------------

    def decode(
        self,
        ids: list[int] | list[str],
        *,
        ticks_per_beat: int = DEFAULT_TICKS_PER_BEAT,
    ) -> DecodedComposition:
        """Rebuild a composition from tokens.

        Forgiving by necessity: model output is malformed long before it is
        musical, and a decoder that raised would make early checkpoints
        impossible to listen to. Illegal tokens are skipped and counted.
        """
        tokens = [
            token if isinstance(token, str) else self.vocabulary.token_of(token) for token in ids
        ]
        step_ticks = self.ticks_per_step(ticks_per_beat)
        onset_divisions = self.config.onset_divisions
        duration_divisions = self.config.duration_divisions
        # Fractional, and rounded only at the final tick. Rounding the unit
        # first would accumulate drift across a piece, which is the error the
        # sub-row codecs exist to remove.
        fine_ticks = step_ticks / onset_divisions
        grammar = TokenGrammar(self.vocabulary)
        result = DecodedComposition(
            composition=Composition(
                hardware_profile=self.config.profile_id, ticks_per_beat=ticks_per_beat
            )
        )

        bar = -1
        pending_section: str | None = None
        sections: list[SectionSpan] = []
        position = 0
        offset = 0
        channel: str | None = None
        value: int | None = None
        velocity = 96
        tempo_bpm: float | None = None
        motif = "unspecified"
        has_loop_start = False
        has_loop_end = False

        for token in tokens:
            if not grammar.advance(token):
                result.n_tokens_skipped += 1
                continue
            result.n_tokens_read += 1

            if token == BAR:
                bar += 1
                position = 0
                offset = 0
                if pending_section is not None:
                    sections.append(SectionSpan(start_bar=bar, label=pending_section))
                    pending_section = None
            elif token.startswith(SECTION):
                pending_section = token[len(SECTION) :]
            elif token == LOOP_START:
                has_loop_start = True
            elif token == LOOP_END:
                has_loop_end = True
            elif token == EOS:
                break
            elif token.startswith(TEMPO):
                tempo_bpm = float(token[len(TEMPO) :])
            elif token.startswith(MOTIF):
                motif = token[len(MOTIF) :]
            elif token.startswith(POSITION):
                position = int(token[len(POSITION) :])
                # A new Position cancels any offset from the previous one, or
                # codec B would carry a stale offset onto the next row.
                offset = 0
            elif token.startswith(OFFSET):
                offset = int(token[len(OFFSET) :])
            elif token.startswith(CHANNEL):
                channel = token[len(CHANNEL) :]
                value = None
            elif token.startswith(PITCH) or token.startswith(NOISE):
                value = int(token.split("_", 1)[1])
            elif token.startswith(VELOCITY):
                velocity = self.bucket_velocity(int(token[len(VELOCITY) :]))
            elif token.startswith(DURATION):
                steps = int(token[len(DURATION) :])
                if channel is None or value is None or bar < 0:
                    result.warnings.append(
                        f"a duration arrived with no note to attach it to at bar {bar}"
                    )
                    continue
                # Codecs C and D carry the sub-row offset inside Position, so
                # their `position` is already a fine slot; A and B carry a row
                # Position with the offset alongside (zero for A).
                if self.config.codec in ("C", "D"):
                    slot = bar * self.config.steps_per_bar * onset_divisions + position
                else:
                    slot = (bar * self.config.steps_per_bar + position) * onset_divisions + offset
                tick = int(round(slot * fine_ticks))
                duration = int(round(steps * step_ticks / duration_divisions))
                if self.config.is_pitched(channel):
                    event = Note(
                        channel_id=channel,
                        tick=tick,
                        pitch=value,
                        duration=duration,
                        velocity=velocity,
                    )
                else:
                    event = Noise(
                        channel_id=channel,
                        tick=tick,
                        value=value,
                        duration=duration,
                        velocity=velocity,
                    )
                result.composition.events.append(event)
                value = None

        composition = result.composition
        composition.sections = sections
        composition.tempo_map = [TempoChange(0, int(round(60_000_000 / (tempo_bpm or 120.0))))]
        composition.time_signatures = [TimeSignatureChange(0, self.config.beats_per_bar, 4)]
        if has_loop_start:
            composition.loop_start_tick = 0
        if has_loop_end:
            composition.loop_end_tick = max(bar + 1, 1) * self.config.steps_per_bar * step_ticks
        composition.normalize()
        composition.source_metadata["requested_motif"] = motif
        return result

    # -- persistence ------------------------------------------------------

    def save(self, path: Path) -> Path:
        return write_json(Path(path), self.vocabulary.to_dict())

    @classmethod
    def load(cls, path: Path) -> RseTokenizer:
        return cls(vocabulary=Vocabulary.from_dict(read_json(Path(path))))


def _bar_count(composition: Composition, config: TokenizerConfig, fit: GridFit | None) -> int:
    """How many bars a composition occupies.

    Counted from the last row carrying an onset rather than from the end tick.
    An end tick includes the final note's duration, which quantizes
    independently, so rounding it up gave a bar count that moved by one between
    a piece and its own decoded copy — and since `encode` pads out to this
    count, that put a trailing `Bar` in one encoding and not the other.
    """
    sounding = composition.sounding_events()
    if not sounding:
        return 1
    if fit is None:
        ticks_per_bar = composition.ticks_per_beat * config.beats_per_bar
        return max(1, int(-(-composition.end_tick // ticks_per_bar)))
    rows = [fit.row_of(event.tick) for event in sounding]
    # `encode` anchors the first sounding row at bar 0, so measure the span.
    return max(1, (max(rows) - min(rows)) // config.steps_per_bar + 1)


def conditioning_from_composition(
    composition: Composition, config: TokenizerConfig, *, fit: GridFit | None = None
) -> Conditioning:
    """Read conditioning out of a composition's own metadata.

    Generated pieces carry their plan, so mood and energy are known exactly.
    Dataset files do not, and get `unknown` rather than a guess — proposal 6.4
    is explicit that mood conditioning needs labels before it means anything.
    """
    plan = (composition.source_metadata.get("generation") or {}).get("plan") or {}
    tempo_bpm = 120.0
    if composition.tempo_map:
        tempo_bpm = 60_000_000 / composition.tempo_map[0].us_per_beat
    tempo_bpm = plan.get("tempo_bpm", tempo_bpm)
    # On a fitted row the Tempo token is load-bearing: it is all `decode`
    # has to turn rows back into seconds. NES-MDB's own tempo map is
    # worthless here anyway, since every file claims 120 BPM.
    if fit is not None and fit.aligned and fit.bpm > 0:
        tempo_bpm = fit.bpm

    n_bars = plan.get("n_bars") or _bar_count(composition, config, fit)
    return Conditioning(
        mood=plan.get("mood", "unknown"),
        energy=plan.get("energy", "unknown"),
        tempo_bpm=tempo_bpm,
        n_bars=int(n_bars),
    )
