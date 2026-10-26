"""HTTP interface to the procedural composer.

Separate from `rse.song.generate`, which does the work: the library has no idea
it is being served over HTTP, and this package adds no musical behaviour.
"""

from __future__ import annotations

__all__ = ["create_app"]


def create_app(*args, **kwargs):
    """Imported lazily so `import rse.api` costs nothing without FastAPI."""
    from rse.api.app import create_app as factory

    return factory(*args, **kwargs)
