"""Deployment settings, with defaults safe enough to be the committed ones.

Two rules shape the defaults here.

**The bind defaults to loopback.** `RSE_API_HOST` exists because a platform
like Render routes to a container and needs a wildcard bind to do it, but the
value that ships in the repository is `127.0.0.1`, so running the service on a
laptop or a shared build host does not put it on the network. The override is
deliberate and visible in the deploy configuration, not a default nobody chose.

**Every limit has a number.** A public generator with no caps is an open
invitation to spend someone's compute budget: the duration cap bounds one
request, the concurrency cap bounds the instance, and the rate limit bounds one
caller. Measured generation cost is about 1.2 s of CPU for a 60-second song,
97% of it in the audio renderer, so these are set against that figure rather
than guessed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

#: Loopback. A deploy that needs otherwise sets RSE_API_HOST explicitly.
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000

#: One request's ceiling. Generation cost scales with duration, so this is the
#: single most effective cap: at ~20 ms of CPU per rendered second, 180 s of
#: music is ~3.6 s of CPU.
DEFAULT_MAX_SECONDS = 180.0
DEFAULT_MIN_SECONDS = 8.0

#: The instance's ceiling. Renders are CPU-bound and run in a worker thread, so
#: more concurrency than cores buys nothing and makes every caller wait longer.
DEFAULT_MAX_CONCURRENT = 2

#: One caller's ceiling, as a token bucket: `rate` songs per minute, burst of
#: `burst`. Generous for a person clicking Generate, useless for a script.
DEFAULT_RATE_PER_MINUTE = 20
DEFAULT_RATE_BURST = 5

#: How long a generated song stays fetchable. Long enough to download, short
#: enough that a small instance's disk cannot fill.
DEFAULT_RETENTION_SECONDS = 1800.0

#: Hard ceiling on stored songs regardless of age, in case of a burst.
DEFAULT_MAX_STORED_SONGS = 200


def _float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError as error:
        raise ValueError(f"{name}={raw!r} is not a number") from error


def _int(name: str, default: int) -> int:
    return int(_float(name, float(default)))


@dataclass(frozen=True)
class Settings:
    """Resolved configuration for one process."""

    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    max_seconds: float = DEFAULT_MAX_SECONDS
    min_seconds: float = DEFAULT_MIN_SECONDS
    max_concurrent: int = DEFAULT_MAX_CONCURRENT
    rate_per_minute: int = DEFAULT_RATE_PER_MINUTE
    rate_burst: int = DEFAULT_RATE_BURST
    retention_seconds: float = DEFAULT_RETENTION_SECONDS
    max_stored_songs: int = DEFAULT_MAX_STORED_SONGS
    storage_dir: Path | None = None
    #: Browser origins allowed to call this API. Empty means same-origin only,
    #: which is the right default: the public site's origin is named in the
    #: deploy configuration rather than guessed here, and `*` is never a
    #: default we ship.
    allowed_origins: tuple[str, ...] = ()

    @classmethod
    def from_env(cls) -> Settings:
        storage = os.environ.get("RSE_API_STORAGE_DIR")
        origins = os.environ.get("RSE_API_ALLOWED_ORIGINS", "")
        return cls(
            host=os.environ.get("RSE_API_HOST", DEFAULT_HOST),
            # Render and most platforms inject PORT; honour it, since the
            # platform owns the port it routes to.
            port=_int("PORT", _int("RSE_API_PORT", DEFAULT_PORT)),
            max_seconds=_float("RSE_API_MAX_SECONDS", DEFAULT_MAX_SECONDS),
            min_seconds=_float("RSE_API_MIN_SECONDS", DEFAULT_MIN_SECONDS),
            max_concurrent=_int("RSE_API_MAX_CONCURRENT", DEFAULT_MAX_CONCURRENT),
            rate_per_minute=_int(
                "RSE_API_RATE_PER_MINUTE", DEFAULT_RATE_PER_MINUTE
            ),
            rate_burst=_int("RSE_API_RATE_BURST", DEFAULT_RATE_BURST),
            retention_seconds=_float(
                "RSE_API_RETENTION_SECONDS", DEFAULT_RETENTION_SECONDS
            ),
            max_stored_songs=_int(
                "RSE_API_MAX_STORED_SONGS", DEFAULT_MAX_STORED_SONGS
            ),
            storage_dir=Path(storage) if storage else None,
            allowed_origins=tuple(
                origin.strip()
                for origin in origins.split(",")
                if origin.strip()
            ),
        )
