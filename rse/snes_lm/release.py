"""Reading a published snesmdb release.

The seam between the two projects is a FILE FORMAT, not an import. snesmdb has
no code dependency on this engine and this engine has none on snesmdb; it reads
the release's documented artifacts. Nothing here reaches into the dataset
repository's source.

Traps handled here because the release documents them and walking into one
would invalidate a result:

* **Parse the 6-dp decimals as `Decimal`, not `float`.** The literals are not
  float64-representable, and parsing as float manufactures about 3,640
  overlaps of ~1e-16 s that do not exist in the data.
* **`track_class` is not all music.** 488 of 2,228 tracks are jingle, drone or
  effect; the registered population is `music` only.
* **Driver labels are not measurements.** The release calls them "curated
  external knowledge… a hypothesis to test rather than a measurement", so they
  are carried for description and never used as a feature.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterator

SPLITS = ("train", "valid", "test")


@dataclass
class Track:
    track_id: str
    game: str
    title: str
    split: str
    track_class: str
    composers: tuple[str, ...]
    composer_truncated: bool
    driver: str | None
    n_notes: int
    render_seconds: float
    events_path: str

    @property
    def descriptive_driver(self) -> str | None:
        """The driver label, for description only -- never a model feature."""
        return self.driver


class Release:
    """A published snesmdb release directory."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser()
        index_path = self.root / "index.json"
        if not index_path.is_file():
            raise FileNotFoundError(
                f"{index_path} not found; point this at a published snesmdb "
                f"release directory"
            )
        self.index = json.loads(index_path.read_text())
        self.schema = self.index.get("schema")

    def tracks(
        self, *, split: str | None = None, track_class: str | None = "music"
    ) -> list[Track]:
        out: list[Track] = []
        for row in self.index["tracks"]:
            if split is not None and row["split"] != split:
                continue
            if track_class is not None and row["track_class"] != track_class:
                continue
            out.append(
                Track(
                    track_id=row["track_id"],
                    game=row["game"],
                    title=row.get("title", ""),
                    split=row["split"],
                    track_class=row["track_class"],
                    composers=tuple(row.get("composers") or ()),
                    composer_truncated=bool(row.get("composer_truncated")),
                    driver=row.get("driver"),
                    n_notes=int(row["n_notes"]),
                    render_seconds=float(row["render_seconds"]),
                    events_path=row["events"],
                )
            )
        out.sort(key=lambda t: t.track_id)
        return out

    def events(self, track: Track) -> list[dict[str, Any]]:
        """A track's events, with onsets and durations parsed as Decimal.

        Decimal rather than float because the release says so: the published
        6-dp literals are not representable in binary float64, and parsing them
        as float invents overlaps of order 1e-16 s. Callers that want floats
        should convert AFTER any interval arithmetic.
        """
        path = self.root / track.events_path
        with gzip.open(path, "rt") as handle:
            payload = json.load(handle, parse_float=Decimal)
        return payload["events"]

    def iter_events(
        self, *, split: str | None = None, track_class: str | None = "music"
    ) -> Iterator[tuple[Track, list[dict[str, Any]]]]:
        for track in self.tracks(split=split, track_class=track_class):
            yield track, self.events(track)

    def census(self) -> dict[str, Any]:
        """Counts straight off the index, for checking a claim against source."""
        rows = self.index["tracks"]
        by_split: dict[str, dict[str, Any]] = {}
        for split in SPLITS:
            everything = [r for r in rows if r["split"] == split]
            music = [r for r in everything if r["track_class"] == "music"]
            by_split[split] = {
                "tracks": len(everything),
                "music_tracks": len(music),
                "music_events": sum(r["n_notes"] for r in music),
                "games": len({r["game"] for r in music}),
            }
        return {
            "schema": self.schema,
            "tracks": len(rows),
            "per_split": by_split,
        }
