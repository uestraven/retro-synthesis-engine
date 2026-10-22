"""Dataset-level statistics accumulated over intake records.

This is the Phase 1 "dataset statistics" deliverable, and it doubles as the
evidence for several decisions deferred in the proposal: whether the provisional
pitch ranges in the hardware profile are right, how often source data already
violates one-note-per-channel, and which controllers actually carry expression.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from statistics import median
from typing import Any

from rse.data.analysis import duration_bucket

#: Tempo histogram bucket width in BPM.
TEMPO_BUCKET = 5


class DatasetStats:
    """Streaming accumulator — never holds more than one record at a time."""

    def __init__(self) -> None:
        self.n_records = 0
        self.n_notes = 0
        self.n_controls = 0
        self.n_dropped_notes = 0
        self.n_dropped_controls = 0
        self.n_hardware_compliant = 0
        self.n_with_loop_markers = 0
        self.n_constant_tempo = 0
        self.n_constant_meter = 0

        self.notes_per_channel: Counter[str] = Counter()
        self.records_using_channel: Counter[str] = Counter()
        self.assignment_methods: Counter[str] = Counter()
        self.value_range: dict[str, tuple[int, int]] = {}
        self.value_histogram: dict[str, Counter[int]] = defaultdict(Counter)
        self.duration_histogram: dict[str, Counter[str]] = defaultdict(Counter)
        self.overlap_records: Counter[str] = Counter()
        self.overlap_events: Counter[str] = Counter()
        self.out_of_range_values: dict[str, Counter[int]] = defaultdict(Counter)
        self.max_simultaneous_seen: Counter[str] = Counter()
        self.controllers_used: Counter[int] = Counter()
        self.programs_used: Counter[int] = Counter()

        self.ticks_per_beat: Counter[int] = Counter()
        self.midi_format: Counter[int] = Counter()
        self.time_signatures: Counter[str] = Counter()
        self.tempo_buckets: Counter[int] = Counter()
        self.channels_used_histogram: Counter[int] = Counter()
        self.use_class: Counter[str] = Counter()
        self.warning_kinds: Counter[str] = Counter()

        self._bars: list[int] = []
        self._seconds: list[float] = []
        self._notes_per_record: list[int] = []

    def add(self, record: dict[str, Any]) -> None:
        self.n_records += 1
        measurements = record.get("measurements", {})
        self.n_notes += measurements.get("n_notes", 0)
        self.n_controls += measurements.get("n_controls", 0)
        self.n_dropped_notes += measurements.get("dropped_notes", 0)
        self.n_dropped_controls += measurements.get("dropped_controls", 0)
        self.n_hardware_compliant += bool(measurements.get("hardware_compliant"))
        self.n_constant_tempo += bool(measurements.get("constant_tempo"))
        self.n_constant_meter += bool(measurements.get("constant_meter"))
        self.n_with_loop_markers += bool(record.get("loop_markers"))

        self._bars.append(measurements.get("n_bars", 0))
        self._seconds.append(measurements.get("end_sec", 0.0))
        self._notes_per_record.append(measurements.get("n_notes", 0))
        self.channels_used_histogram[measurements.get("channels_used", 0)] += 1

        midi = record.get("midi", {})
        self.ticks_per_beat[midi.get("ticks_per_beat", 0)] += 1
        self.midi_format[midi.get("format", -1)] += 1
        self.use_class[record.get("source", {}).get("use_class", "unknown")] += 1

        for sig in record.get("time_signatures", ()):
            self.time_signatures[f"{sig['numerator']}/{sig['denominator']}"] += 1
        for tempo in record.get("tempo_map", ()):
            bucket = int(round(tempo.get("bpm", 0) / TEMPO_BUCKET) * TEMPO_BUCKET)
            self.tempo_buckets[bucket] += 1

        for channel in record.get("channels", ()):
            cid = channel["channel_id"]
            self.assignment_methods[channel.get("assignment_method", "unassigned")] += 1
            if channel.get("n_notes", 0) == 0:
                continue
            self.records_using_channel[cid] += 1
            self.notes_per_channel[cid] += channel["n_notes"]
            self.max_simultaneous_seen[cid] = max(
                self.max_simultaneous_seen[cid], channel.get("max_simultaneous", 0)
            )
            if channel.get("overlap_count"):
                self.overlap_records[cid] += 1
                self.overlap_events[cid] += channel["overlap_count"]
            for value in channel.get("out_of_range_values", ()):
                self.out_of_range_values[cid][value] += 1
            lo, hi = channel.get("value_min"), channel.get("value_max")
            if lo is not None and hi is not None:
                prev = self.value_range.get(cid)
                self.value_range[cid] = (
                    min(lo, prev[0]) if prev else lo,
                    max(hi, prev[1]) if prev else hi,
                )
            self.controllers_used.update(channel.get("controllers_used", ()))
            self.programs_used.update(channel.get("programs", ()))

        for event in record.get("events", ()):
            etype = event.get("type")
            if etype not in ("note", "noise"):
                continue
            cid = event["ch"]
            value = event.get("pitch", event.get("value"))
            if value is not None:
                self.value_histogram[cid][value] += 1
            self.duration_histogram[cid][duration_bucket(event.get("dur_beats", 0.0))] += 1

        for warning in record.get("warnings", ()):
            self.warning_kinds[_warning_kind(warning)] += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_records": self.n_records,
            "n_notes": self.n_notes,
            "n_controls": self.n_controls,
            "n_dropped_notes": self.n_dropped_notes,
            "n_dropped_controls": self.n_dropped_controls,
            "n_hardware_compliant": self.n_hardware_compliant,
            "n_with_loop_markers": self.n_with_loop_markers,
            "n_constant_tempo": self.n_constant_tempo,
            "n_constant_meter": self.n_constant_meter,
            "notes_per_record": _summary(self._notes_per_record),
            "bars_per_record": _summary(self._bars),
            "seconds_per_record": _summary(self._seconds),
            "channels_used_histogram": _sorted_counter(self.channels_used_histogram),
            "per_channel": {
                cid: {
                    "n_notes": self.notes_per_channel[cid],
                    "n_records_using": self.records_using_channel[cid],
                    "value_min": self.value_range.get(cid, (None, None))[0],
                    "value_max": self.value_range.get(cid, (None, None))[1],
                    "max_simultaneous_seen": self.max_simultaneous_seen[cid],
                    "records_with_overlaps": self.overlap_records[cid],
                    "overlap_events": self.overlap_events[cid],
                    "out_of_range_values": _sorted_counter(self.out_of_range_values[cid]),
                    "value_histogram": _sorted_counter(self.value_histogram[cid]),
                    "duration_histogram": dict(self.duration_histogram[cid]),
                }
                for cid in sorted(
                    set(self.notes_per_channel)
                    | set(self.records_using_channel)
                    | set(self.value_histogram)
                )
            },
            "assignment_methods": dict(self.assignment_methods.most_common()),
            "controllers_used": _sorted_counter(self.controllers_used),
            "programs_used": _sorted_counter(self.programs_used),
            "ticks_per_beat": _sorted_counter(self.ticks_per_beat),
            "midi_format": _sorted_counter(self.midi_format),
            "time_signatures": dict(self.time_signatures.most_common()),
            "tempo_bpm_histogram": _sorted_counter(self.tempo_buckets),
            "use_class": dict(self.use_class.most_common()),
            "warning_kinds": dict(self.warning_kinds.most_common()),
        }


def _summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "min": round(min(values), 3),
        "median": round(median(values), 3),
        "max": round(max(values), 3),
        "mean": round(sum(values) / len(values), 3),
    }


def _sorted_counter(counter: Counter) -> dict[str, int]:
    return {str(key): counter[key] for key in sorted(counter)}


def _warning_kind(warning: str) -> str:
    """Collapse a warning message to a countable category."""
    text = warning.lower()
    for needle, kind in (
        ("note_off without note_on", "orphan_note_off"),
        ("note left open", "unterminated_note"),
        ("zero-length note", "zero_length_note"),
        ("already taken", "channel_name_conflict"),
        ("no free", "no_free_channel"),
        ("no source voice mapped", "unused_channel"),
    ):
        if needle in text:
            return kind
    return "other"
