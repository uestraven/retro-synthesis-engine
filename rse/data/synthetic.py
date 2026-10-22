"""Synthetic four-channel MIDI generation for testing the pipeline.

NES-MDB cannot be redistributed with this repository, so the test suite and the
end-to-end smoke test need source files of their own. These files imitate the
*shape* of the dataset — one named track per NES channel, expressive
controllers, loop markers, composer-disjoint split directories — without
imitating anybody's music.

This is not the Phase 3 procedural composer; it exists only to exercise
parsing, normalization, measurement and de-duplication.
"""

from __future__ import annotations

import random
import tarfile
from pathlib import Path

import mido

#: Track names matched by the NES profile's `track_name_patterns`.
CHANNEL_TRACK_NAMES = ("P1", "P2", "TR", "NO")

#: A minor pentatonic scale, as semitone offsets.
_SCALE = (0, 3, 5, 7, 10)

#: MIDI CC 11 (expression) stands in for the per-channel volume envelope that
#: NES-MDB encodes as controller data.
EXPRESSION_CC = 11


def write_nes_style_midi(
    path: Path,
    *,
    seed: int = 0,
    bars: int = 8,
    tempo_bpm: float = 120.0,
    ticks_per_beat: int = 480,
    root: int = 57,
    include_controllers: bool = True,
    include_loop_markers: bool = True,
) -> Path:
    """Write one four-channel, monophonic-per-channel MIDI file."""
    rng = random.Random(seed)
    midi = mido.MidiFile(type=1, ticks_per_beat=ticks_per_beat)

    meta = mido.MidiTrack()
    meta.append(mido.MetaMessage("track_name", name="meta", time=0))
    meta.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(tempo_bpm), time=0))
    meta.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    if include_loop_markers:
        meta.append(mido.MetaMessage("marker", text="loop_start", time=0))
        meta.append(mido.MetaMessage("marker", text="loop_end", time=bars * 4 * ticks_per_beat))
    midi.tracks.append(meta)

    sixteenth = ticks_per_beat // 4
    melody = _phrase(rng, bars, root + 12, steps_per_bar=8)
    harmony = _phrase(rng, bars, root + 7, steps_per_bar=4)
    bass = _phrase(rng, bars, root - 12, steps_per_bar=2)
    percussion = [(i, rng.randint(1, 16)) for i in range(bars * 4)]

    midi.tracks.append(
        _voice_track(
            "P1",
            0,
            melody,
            step_ticks=sixteenth * 2,
            include_controllers=include_controllers,
            rng=rng,
        )
    )
    midi.tracks.append(
        _voice_track(
            "P2",
            1,
            harmony,
            step_ticks=sixteenth * 4,
            include_controllers=include_controllers,
            rng=rng,
        )
    )
    midi.tracks.append(
        _voice_track("TR", 2, bass, step_ticks=sixteenth * 8, include_controllers=False, rng=rng)
    )
    midi.tracks.append(
        _voice_track(
            "NO",
            3,
            percussion,
            step_ticks=ticks_per_beat,
            note_ticks=sixteenth,
            include_controllers=include_controllers,
            rng=rng,
        )
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    midi.save(str(path))
    return path


def _phrase(
    rng: random.Random, bars: int, root: int, *, steps_per_bar: int
) -> list[tuple[int, int]]:
    """`(step index, pitch)` pairs, with the first bar repeated to create structure."""
    first_bar = [
        (step, root + rng.choice(_SCALE) + 12 * rng.choice((0, 0, 0, 1)))
        for step in range(steps_per_bar)
    ]
    notes: list[tuple[int, int]] = []
    for bar in range(bars):
        # Repeat the opening bar every other bar so the fixtures contain real
        # repetition for the repetition-balance measurements to find.
        source = (
            first_bar
            if bar % 2 == 0
            else [(step, pitch + rng.choice((-2, 0, 2))) for step, pitch in first_bar]
        )
        notes.extend((bar * steps_per_bar + step, pitch) for step, pitch in source)
    return notes


def _voice_track(
    name: str,
    midi_channel: int,
    notes: list[tuple[int, int]],
    *,
    step_ticks: int,
    rng: random.Random,
    note_ticks: int | None = None,
    include_controllers: bool = True,
) -> mido.MidiTrack:
    track = mido.MidiTrack()
    track.append(mido.MetaMessage("track_name", name=name, time=0))
    cursor = 0
    for index, (step, value) in enumerate(notes):
        start = step * step_ticks
        length = note_ticks or step_ticks
        if include_controllers and index % 4 == 0:
            track.append(
                mido.Message(
                    "control_change",
                    channel=midi_channel,
                    control=EXPRESSION_CC,
                    value=rng.randint(40, 127),
                    time=start - cursor,
                )
            )
            cursor = start
        track.append(
            mido.Message(
                "note_on",
                channel=midi_channel,
                note=max(0, min(127, value)),
                velocity=rng.randint(64, 127),
                time=start - cursor,
            )
        )
        track.append(
            mido.Message(
                "note_off",
                channel=midi_channel,
                note=max(0, min(127, value)),
                velocity=0,
                time=length,
            )
        )
        cursor = start + length
    return track


def make_sample_release(
    dest_dir: Path,
    *,
    n_per_split: int = 4,
    seed: int = 0,
    archive_name: str = "rse_sample_midi.tar.gz",
    duplicate_one: bool = True,
) -> Path:
    """Build a tar.gz that looks like a split dataset release.

    With `duplicate_one`, the train split contains two byte-identical copies of
    one composition so the de-duplication report has something to find.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    staging = dest_dir / "sample_src"
    # Seeds are derived from positional indexes, not from hash(), so the same
    # call produces byte-identical files on every run and every interpreter.
    for split_index, split in enumerate(("train", "valid", "test")):
        for i in range(n_per_split):
            write_nes_style_midi(
                staging / split / f"{split}_{i:03d}.mid",
                seed=seed + 1000 * split_index + i,
                bars=8 if i % 2 == 0 else 16,
                tempo_bpm=100 + 10 * i,
            )
    if duplicate_one:
        original = staging / "train" / "train_000.mid"
        (staging / "train" / "train_000_copy.mid").write_bytes(original.read_bytes())

    archive = dest_dir / archive_name
    with tarfile.open(archive, "w:gz") as tf:
        for path in sorted(staging.rglob("*.mid")):
            tf.add(path, arcname=path.relative_to(staging).as_posix())
    return archive
