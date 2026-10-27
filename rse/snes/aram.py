"""The 64 KiB audio RAM budget, counted rather than estimated.

Everything the SPC700 and S-DSP need lives in one 65,536-byte space: the
driver's own code, the sample directory, the BRR sample data, the sequence,
the DSP's echo buffer, and whatever runtime state the driver keeps. There is
no second pool and nothing is paged in.

That is why the echo buffer matters so much more than it looks. It is charged
against the same 64 KiB as the instruments, so a generous reverb is paid for
in samples -- and at 2 KiB per 16 ms of delay, a 240 ms echo costs 30 KiB,
which is most of the budget. Composers chose between a big instrument set and
a big room, and could not have both.

Previous reports here said "BRR estimate" and carried a caveat that nothing
was encoded. That is no longer true: `rse.snes.brr` encodes for real, so these
figures are measured sizes of actual encoded data. What is still NOT claimed
is hardware compatibility -- a correct byte count is not an SPC file, and no
emulator has run any of this.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: The whole audio RAM.
ARAM_BYTES = 65536

#: Bytes per directory entry: a start address and a loop address, two bytes
#: each. The directory is indexed by sample number, so it is sized by the
#: highest index used rather than by the count.
DIRECTORY_ENTRY_BYTES = 4

#: The echo buffer costs 2 KiB per 16 ms of delay, and the DSP's delay
#: register steps in 16 ms units.
ECHO_BYTES_PER_STEP = 2048
ECHO_STEP_MS = 16

#: A representative driver footprint. Declared as an assumption rather than a
#: measurement: RSE has no SPC700 driver, so this is the space one would need
#: and not the size of code that exists.
ASSUMED_DRIVER_BYTES = 2048
ASSUMED_RUNTIME_BYTES = 512


@dataclass
class AramBudget:
    """What a track would occupy, and what is left."""

    sample_bytes: int = 0
    directory_bytes: int = 0
    sequence_bytes: int = 0
    echo_bytes: int = 0
    driver_bytes: int = ASSUMED_DRIVER_BYTES
    runtime_bytes: int = ASSUMED_RUNTIME_BYTES
    #: Per-instrument encoded sizes, for the report.
    per_sample: dict[str, int] = field(default_factory=dict)
    #: Which figures are measured and which are assumed.
    assumed: tuple[str, ...] = ("driver_bytes", "runtime_bytes")

    @property
    def total(self) -> int:
        return (
            self.sample_bytes
            + self.directory_bytes
            + self.sequence_bytes
            + self.echo_bytes
            + self.driver_bytes
            + self.runtime_bytes
        )

    @property
    def remaining(self) -> int:
        return ARAM_BYTES - self.total

    @property
    def fits(self) -> bool:
        return self.total <= ARAM_BYTES

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_bytes": self.total,
            "aram_bytes": ARAM_BYTES,
            "remaining_bytes": self.remaining,
            "fits": self.fits,
            "used_fraction": round(self.total / ARAM_BYTES, 4),
            "breakdown": {
                "samples": self.sample_bytes,
                "directory": self.directory_bytes,
                "sequence": self.sequence_bytes,
                "echo": self.echo_bytes,
                "driver": self.driver_bytes,
                "runtime": self.runtime_bytes,
            },
            "per_sample": dict(self.per_sample),
            "assumed": list(self.assumed),
            "measured": [
                "samples (real BRR encoding)",
                "directory",
                "echo (from the delay setting)",
            ],
            "caveat": (
                "Byte counts are real: the samples are BRR-encoded by "
                "rse.snes.brr. Driver and runtime figures are ASSUMPTIONS -- "
                "RSE has no SPC700 driver. A correct byte count is not an SPC "
                "file and no emulator has run any of this, so no hardware "
                "compatibility is claimed."
            ),
        }

    def report(self) -> str:
        lines = [
            f"ARAM {self.total} / {ARAM_BYTES} bytes "
            f"({self.total / ARAM_BYTES * 100:.1f}%), "
            f"{self.remaining} free" + ("" if self.fits else "  OVER BUDGET"),
            f"  samples   {self.sample_bytes:>6}  (BRR, measured)",
            f"  directory {self.directory_bytes:>6}",
            f"  sequence  {self.sequence_bytes:>6}",
            f"  echo      {self.echo_bytes:>6}",
            f"  driver    {self.driver_bytes:>6}  (assumed)",
            f"  runtime   {self.runtime_bytes:>6}  (assumed)",
        ]
        return "\n".join(lines)


def echo_bytes_for(delay_ms: int) -> int:
    """Echo buffer cost for a delay, rounded up to the DSP's 16 ms step.

    Charged against the same 64 KiB as the samples, which is the tradeoff the
    hardware imposes: a 240 ms echo is 30 KiB, nearly half the budget.
    """
    if delay_ms <= 0:
        return 0
    steps = -(-delay_ms // ECHO_STEP_MS)
    return steps * ECHO_BYTES_PER_STEP


def budget_for(
    encoded: dict[str, Any],
    *,
    sequence_bytes: int = 0,
    echo_delay_ms: int = 0,
    directory_slots: int | None = None,
) -> AramBudget:
    """Add up a track's ARAM usage from real encoded sizes.

    `encoded` maps instrument name to a `BrrSample`.
    """
    per_sample = {name: sample.size_bytes for name, sample in encoded.items()}
    slots = directory_slots if directory_slots is not None else len(encoded)
    return AramBudget(
        sample_bytes=sum(per_sample.values()),
        directory_bytes=slots * DIRECTORY_ENTRY_BYTES,
        sequence_bytes=sequence_bytes,
        echo_bytes=echo_bytes_for(echo_delay_ms),
        per_sample=per_sample,
    )


def largest_echo_that_fits(budget: AramBudget) -> int:
    """The longest echo delay, in ms, that the remaining space allows.

    Useful in the other direction: given an instrument set, how much room is
    there for a room?
    """
    spare = ARAM_BYTES - (budget.total - budget.echo_bytes)
    steps = max(0, spare // ECHO_BYTES_PER_STEP)
    return int(steps * ECHO_STEP_MS)
