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
was encoded. That is no longer true: `rse.snes.brr` encodes for real, so the
sample figures are measured sizes of actual encoded data.

But "measured" is a word worth spending care on, because only some of these
numbers earn it. Four classes, and the report labels every line with one:

**MEASURED** -- the number is the size of something that exists. Sample bytes
are `len()` of real encoded data. The echo buffer follows exactly from the
DSP's delay register, 2 KiB per 16 ms step, with no freedom in it. Directory
entries are 4 bytes by the format's definition.

**MODELLED** -- a defensible number for something RSE does not actually
produce. Sequence data is 4 bytes per note event: a reasonable driver
encoding, but RSE emits no sequence bytes, so this is arithmetic about a
hypothetical.

**ASSUMED** -- a stated allowance. Driver code and runtime state. RSE has no
SPC700 driver, so these are the space one would need.

**UNMODELLED** -- named, counted as zero, and the reason this module must not
claim hardware-compatible budgeting. See `UNMODELLED`.

Fitting inside 65,536 bytes here means the represented allocations fit. It
does not mean a real SPC would load. `hardware_budgeted` is False and stays
False until the unmodelled list is empty.
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

#: The sample directory and the echo buffer are both located by a PAGE number
#: (the DIR and ESA registers), so each must start on a 256-byte boundary.
#: Real layouts therefore waste some padding, bounded by this per aligned
#: region. Reported as a worst case rather than computed, because computing it
#: needs a real address map and RSE has none.
ALIGNMENT_BYTES = 256

#: What a working SPC image needs that this budget does NOT represent. Every
#: entry is a reason `hardware_budgeted` is False. Counted as zero -- not
#: because it is free, but because inventing a number for it would make the
#: total look authoritative when it is not.
#:
#: Note that several of these are small. That is not reassuring: being short
#: by 200 bytes and being short by 2,000 both fail to load, and the point of
#: naming them is that the total is not yet a load test.
UNMODELLED: tuple[tuple[str, str], ...] = (
    (
        "instrument table",
        "per-instrument ADSR/GAIN settings, base pitch and directory index; "
        "a few bytes per instrument that the driver reads at key-on",
    ),
    (
        "pattern and order tables",
        "the sequence's own structure -- pattern pointers, order list, loop "
        "points -- as distinct from the note events counted above",
    ),
    (
        "page alignment padding",
        f"up to {ALIGNMENT_BYTES} bytes before each of the directory and the "
        "echo buffer, since both are addressed by page number",
    ),
    (
        "SPC700 stack and direct page",
        "the CPU's own working memory, which a driver's code size does not "
        "include",
    ),
    (
        "DSP register shadow",
        "most drivers keep a RAM copy of the 128 DSP registers to write them "
        "coherently",
    ),
)


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

    #: Every line's evidence class, so no caller has to guess which numbers
    #: are real. Keyed by the breakdown's own names.
    EVIDENCE: dict[str, str] = field(
        default_factory=lambda: {
            "samples": "measured",
            "directory": "measured",
            "echo": "measured",
            "sequence": "modelled",
            "driver": "assumed",
            "runtime": "assumed",
        },
        init=False,
        repr=False,
    )

    @property
    def hardware_budgeted(self) -> bool:
        """False while anything in `UNMODELLED` is still unmodelled.

        Deliberately not a function of whether the total fits. Fitting means
        the represented allocations fit; this means the representation is
        complete. Only the second licenses the phrase "hardware-compatible
        memory budgeting", and it is not true yet.
        """
        return not UNMODELLED

    @property
    def measured_bytes(self) -> int:
        """The part of the total that is the size of something real."""
        return self.sample_bytes + self.directory_bytes + self.echo_bytes

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
            "evidence": dict(self.EVIDENCE),
            "assumed": list(self.assumed),
            "measured": [
                "samples (real BRR encoding)",
                "directory",
                "echo (from the delay setting)",
            ],
            "measured_bytes": self.measured_bytes,
            "modelled": ["sequence (4 bytes per note event, not RSE's own format)"],
            "unmodelled": [
                {"name": name, "why_it_matters": why} for name, why in UNMODELLED
            ],
            "hardware_budgeted": self.hardware_budgeted,
            "caveat": (
                "Of this total, "
                f"{self.measured_bytes} bytes are MEASURED (BRR sample data, "
                "the directory, and the echo buffer, which follows exactly "
                f"from the delay register); {self.sequence_bytes} are "
                "MODELLED (RSE emits no sequence bytes); and "
                f"{self.driver_bytes + self.runtime_bytes} are ASSUMED (RSE "
                "has no SPC700 driver). The echo buffer IS included in the "
                "total -- do not add it again. Several allocations a working "
                "SPC image needs are UNMODELLED and counted as zero, so "
                "fitting means the represented allocations fit, NOT that a "
                "real SPC would load. hardware_budgeted is False and no "
                "emulator has run any of this."
            ),
        }

    def report(self) -> str:
        def line(name: str, value: int) -> str:
            return f"  {name:<10}{value:>6}  ({self.EVIDENCE[name]})"

        lines = [
            f"ARAM {self.total} / {ARAM_BYTES} bytes "
            f"({self.total / ARAM_BYTES * 100:.1f}%), "
            f"{self.remaining} free" + ("" if self.fits else "  OVER BUDGET"),
            line("samples", self.sample_bytes),
            line("directory", self.directory_bytes),
            line("sequence", self.sequence_bytes),
            line("echo", self.echo_bytes),
            line("driver", self.driver_bytes),
            line("runtime", self.runtime_bytes),
            f"  {'-- of which measured':<10} {self.measured_bytes} bytes",
            f"  UNMODELLED, counted as zero: "
            f"{', '.join(name for name, _ in UNMODELLED)}",
            "  hardware_budgeted: False (the list above is not empty)",
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
