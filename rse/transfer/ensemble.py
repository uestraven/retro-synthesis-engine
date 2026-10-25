"""Did the generation deliver the instrumentation it was asked for?

This replaces a blanket "at least three channels" gate, which was wrong. The
`sparse` scenario plans exactly two tracks, so its two-channel output was
correct behaviour — and the gate discarded seven of sixty pieces as failures
for obeying their prompt. A two-channel atmospheric piece and a two-channel
battle theme are not the same result, and only the request distinguishes them.

So the requested plan is the authority. Validation reports what was asked for,
what the composer produced, what the arranger mapped, and what finally sounds,
as four separate numbers. They diverge for different reasons and a single
"channels used" figure hides all of them:

- the composer can ignore a planned track,
- the arranger can merge two tracks onto one channel,
- the hardware repair can silence something.

Nothing here forces percussion or invents notes to pass a threshold. A piece
that genuinely was not asked for drums is not improved by adding them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rse.events.schema import Composition
from rse.transfer.score import PERCUSSION, Score

#: NES channels, in the fixed order the profile declares them.
CHANNELS = ("P1", "P2", "TR", "NO")


@dataclass
class EnsembleCheck:
    """Requested against delivered instrumentation, at every stage."""

    scenario: str = ""
    #: Tracks named in the Fermata header's plan.
    n_requested_tracks: int = 0
    requested_percussion: bool = False
    #: Tracks the composer actually produced events for.
    n_generated_tracks: int = 0
    generated_percussion: bool = False
    #: Tracks the arranger mapped onto a channel.
    n_mapped_tracks: int = 0
    #: Channels sounding in the final, hardware-valid composition.
    channels_sounding: list[str] = field(default_factory=list)
    percussion_sounding: bool = False
    n_tracks_merged: int = 0

    @property
    def missing_tracks(self) -> int:
        """Planned tracks the composer never wrote anything for."""
        return max(0, self.n_requested_tracks - self.n_generated_tracks)

    @property
    def percussion_as_requested(self) -> bool:
        """Drums present exactly when they were asked for."""
        return self.requested_percussion == self.percussion_sounding

    @property
    def honours_request(self) -> bool:
        """Every planned track reached a sounding channel.

        Deliberately not "used at least N channels": the question is whether
        the piece delivered its own brief, which is the only criterion that
        treats a sparse scenario and a battle theme correctly.
        """
        return (
            self.missing_tracks == 0
            and self.percussion_as_requested
            and len(self.channels_sounding) >= self.n_mapped_tracks - self.n_tracks_merged
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "n_requested_tracks": self.n_requested_tracks,
            "requested_percussion": self.requested_percussion,
            "n_generated_tracks": self.n_generated_tracks,
            "generated_percussion": self.generated_percussion,
            "n_mapped_tracks": self.n_mapped_tracks,
            "channels_sounding": list(self.channels_sounding),
            "percussion_sounding": self.percussion_sounding,
            "n_tracks_merged": self.n_tracks_merged,
            "missing_tracks": self.missing_tracks,
            "percussion_as_requested": self.percussion_as_requested,
            "honours_request": self.honours_request,
        }


def check_ensemble(
    *,
    scenario: str,
    plan: list[str],
    score: Score,
    arranged: Composition,
    n_tracks_merged: int = 0,
) -> EnsembleCheck:
    """Compare the header's plan with what actually sounds.

    `plan` is the list of `<TRACK_*>` and `<DRUM_TRACK>` tokens from the
    prompt, which is the authoritative statement of what was asked for.
    """
    requested_percussion = any(t == "<DRUM_TRACK>" for t in plan)
    generated = [t for t in score.tracks if t.n_events]
    sounding = sorted(
        {e.channel_id for e in arranged.sounding_events()},
        key=lambda c: CHANNELS.index(c) if c in CHANNELS else len(CHANNELS),
    )
    return EnsembleCheck(
        scenario=scenario,
        n_requested_tracks=len(plan),
        requested_percussion=requested_percussion,
        n_generated_tracks=len(generated),
        generated_percussion=any(t.role == PERCUSSION or t.percussion for t in generated),
        n_mapped_tracks=len(generated),
        channels_sounding=sounding,
        percussion_sounding="NO" in sounding,
        n_tracks_merged=n_tracks_merged,
    )
