"""Exact and near-duplicate detection across a dataset.

Duplicates are a memorization risk (proposal 14), so they are detected on the
*music* rather than only on file bytes: the same tune re-saved with different
tempo metadata has identical bytes nowhere and identical content everywhere.

Detection flags; it never deletes. The pipeline records `duplicate_of` on the
index entry and the caller decides whether to exclude those records.
"""

from __future__ import annotations

import hashlib
import zlib
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

#: Length, in intervals, of the melodic n-grams used for the fingerprint.
NGRAM = 6

#: Jaccard similarity at or above which two compositions are called near-duplicates.
DEFAULT_THRESHOLD = 0.85

#: Fingerprints are reduced to the k smallest hashes (a bottom-k sketch) so that
#: a few thousand compositions' worth of grams stay in memory at once. Jaccard
#: over two sketches estimates Jaccard over the full gram sets; the estimate is
#: what makes this a review-flagging tool rather than a verdict.
DEFAULT_SKETCH_SIZE = 256


def content_hash(record: dict[str, Any]) -> str:
    """Hash of the musical content of an intake record.

    Covers channel, position, duration and value of every note/noise event, and
    deliberately ignores file metadata, so byte-different re-saves of the same
    music collide.
    """
    digest = hashlib.sha256()
    for event in record.get("events", ()):
        if event.get("type") not in ("note", "noise"):
            continue
        value = event.get("pitch", event.get("value"))
        digest.update(
            f"{event['ch']}|{event['tick']}|{event.get('dur_ticks', 0)}|{value}\n".encode()
        )
    return digest.hexdigest()


def fingerprint(
    record: dict[str, Any], *, sketch_size: int | None = DEFAULT_SKETCH_SIZE
) -> set[int]:
    """Transposition-invariant melodic n-gram fingerprint.

    Only pitched channels contribute: noise-channel values are hardware periods,
    and intervals between them are not musically meaningful. With `sketch_size`
    set, the result is the k smallest gram hashes rather than all of them.
    """
    by_channel: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for event in record.get("events", ()):
        if event.get("type") != "note":
            continue
        by_channel[event["ch"]].append((event["tick"], event["pitch"]))

    grams: set[int] = set()
    for channel, notes in by_channel.items():
        notes.sort()
        pitches = [p for _tick, p in notes]
        intervals = [b - a for a, b in zip(pitches, pitches[1:], strict=False)]
        for i in range(len(intervals) - NGRAM + 1):
            window = intervals[i : i + NGRAM]
            key = f"{channel}:" + ",".join(str(v) for v in window)
            grams.add(zlib.crc32(key.encode()))
    if sketch_size is not None and len(grams) > sketch_size:
        return set(sorted(grams)[:sketch_size])
    return grams


def jaccard(a: set[int], b: set[int]) -> float:
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    if not intersection:
        return 0.0
    return intersection / len(a | b)


@dataclass
class DuplicateReport:
    exact_groups: list[list[str]] = field(default_factory=list)
    near_pairs: list[tuple[str, str, float]] = field(default_factory=list)
    #: Maps a flagged composition id to the representative it duplicates.
    duplicate_of: dict[str, str] = field(default_factory=dict)

    @property
    def n_exact(self) -> int:
        return sum(len(group) - 1 for group in self.exact_groups)

    @property
    def n_near(self) -> int:
        """Compositions flagged by similarity alone, excluding exact matches."""
        return max(0, len(self.duplicate_of) - self.n_exact)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_exact_duplicates": self.n_exact,
            "n_near_duplicates": self.n_near,
            "exact_groups": self.exact_groups,
            "near_pairs": [
                {"a": a, "b": b, "similarity": round(sim, 4)} for a, b, sim in self.near_pairs
            ],
            "duplicate_of": self.duplicate_of,
        }


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        self.parent.setdefault(item, item)
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != root:  # path compression
            self.parent[item], item = root, self.parent[item]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # Lowest id wins so the representative is stable across runs.
            lo, hi = sorted((ra, rb))
            self.parent[hi] = lo


def find_duplicates(
    entries: list[tuple[str, str, set[int]]],
    *,
    threshold: float = DEFAULT_THRESHOLD,
) -> DuplicateReport:
    """Group `(composition_id, content_hash, fingerprint)` triples.

    Exact matches come from the content hash. Near matches are found by sharing
    at least one n-gram and then clearing the Jaccard threshold, which keeps the
    comparison count near-linear instead of quadratic in dataset size.
    """
    report = DuplicateReport()
    uf = _UnionFind()

    by_hash: dict[str, list[str]] = defaultdict(list)
    for comp_id, chash, _grams in entries:
        by_hash[chash].append(comp_id)
    for group in by_hash.values():
        if len(group) > 1:
            group = sorted(group)
            report.exact_groups.append(group)
            for other in group[1:]:
                uf.union(group[0], other)

    index: dict[int, list[str]] = defaultdict(list)
    grams_by_id: dict[str, set[int]] = {}
    for comp_id, _chash, grams in entries:
        grams_by_id[comp_id] = grams
        for gram in grams:
            index[gram].append(comp_id)

    candidates: set[tuple[str, str]] = set()
    for ids in index.values():
        if len(ids) < 2 or len(ids) > 64:
            # A gram shared by very many compositions is a common idiom, not
            # evidence of duplication, and would dominate the candidate set.
            continue
        ordered = sorted(set(ids))
        for i, a in enumerate(ordered):
            for b in ordered[i + 1 :]:
                candidates.add((a, b))

    for a, b in sorted(candidates):
        if uf.find(a) == uf.find(b):
            continue
        similarity = jaccard(grams_by_id[a], grams_by_id[b])
        if similarity >= threshold:
            report.near_pairs.append((a, b, similarity))
            uf.union(a, b)

    for comp_id, _chash, _grams in entries:
        rep = uf.find(comp_id)
        if rep != comp_id:
            report.duplicate_of[comp_id] = rep
    return report
