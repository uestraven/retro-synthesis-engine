"""Caps: one caller, one instance, one disk.

A public generator that will run on a small instance needs all three. None of
this is security -- an anonymous public API has no credential to check -- it is
cost and availability control, which is the actual risk a free retro-music
generator carries.

The rate limiter is an in-process token bucket, which is the honest shape for a
single instance. Behind more than one instance it becomes per-instance, so the
real limit is the platform's; this is deliberately the simple version and the
docstring says so rather than implying a distributed guarantee.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


@dataclass
class _Bucket:
    tokens: float
    updated: float


class RateLimiter:
    """Token bucket per key, refilling at `per_minute` with burst `burst`."""

    def __init__(self, *, per_minute: int, burst: int) -> None:
        self.per_second = per_minute / 60.0
        self.burst = float(max(1, burst))
        self._buckets: dict[str, _Bucket] = {}
        self._lock = threading.Lock()

    def check(self, key: str, *, now: float | None = None) -> float | None:
        """Consume a token. Returns None when allowed, else seconds to wait."""
        moment = time.monotonic() if now is None else now
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                self._buckets[key] = _Bucket(tokens=self.burst - 1.0, updated=moment)
                return None
            elapsed = max(0.0, moment - bucket.updated)
            bucket.tokens = min(
                self.burst, bucket.tokens + elapsed * self.per_second
            )
            bucket.updated = moment
            if bucket.tokens >= 1.0:
                bucket.tokens -= 1.0
                return None
            deficit = 1.0 - bucket.tokens
            return deficit / self.per_second if self.per_second else 60.0

    def forget_before(self, cutoff: float) -> int:
        """Drop idle buckets so the table cannot grow without bound."""
        with self._lock:
            stale = [
                key for key, bucket in self._buckets.items() if bucket.updated < cutoff
            ]
            for key in stale:
                del self._buckets[key]
            return len(stale)


@dataclass
class SongStore:
    """Generated songs on local disk, with age and count limits.

    Temporary by design: the roadmap puts these in object storage later, and
    nothing here assumes the files survive a restart.
    """

    root: object  # Path, kept loose so this module imports nothing heavy
    retention_seconds: float
    max_songs: int
    _order: list[str] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def record(self, song_id: str) -> list[str]:
        """Register a new song and return the ids evicted to stay in budget."""
        with self._lock:
            self._order.append(song_id)
            evicted = []
            while len(self._order) > self.max_songs:
                evicted.append(self._order.pop(0))
            return evicted

    def expired(self, *, now: float) -> list[str]:
        """Ids whose directory is older than the retention window."""
        from pathlib import Path

        root = Path(str(self.root))
        stale = []
        for child in root.iterdir() if root.exists() else []:
            if not child.is_dir():
                continue
            try:
                age = now - child.stat().st_mtime
            except OSError:
                continue
            if age > self.retention_seconds:
                stale.append(child.name)
        return stale

    def forget(self, song_id: str) -> None:
        with self._lock:
            if song_id in self._order:
                self._order.remove(song_id)
