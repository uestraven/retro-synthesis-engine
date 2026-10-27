"""BRR: the SNES sample format, encoded and decoded for real.

Bit Rate Reduction is the S-DSP's only sample format. Nine bytes hold sixteen
samples: one header byte, then sixteen 4-bit nibbles. The header carries a
4-bit range (a left shift), a 2-bit filter selector, and the loop and end
flags.

    byte 0   RRRR FF L E
    bytes 1-8   16 nibbles, signed 4-bit

The four filters are linear predictors over the previous two decoded samples,
which is where the compression comes from -- a waveform that moves smoothly is
cheap to predict and the nibble only has to carry the error:

    0:  s = d
    1:  s = d + p1 * 15/16
    2:  s = d + p1 * 61/32 - p2 * 15/16
    3:  s = d + p1 * 115/64 - p2 * 13/16

Why this matters beyond file size: BRR's quantisation is part of the console's
recognisable sound. A 4-bit residual over a 2-tap predictor is lossy in a
specific way, and a preview that plays clean PCM does not sound like a SNES
even when the samples and the mixing are right.

Two hardware constraints this enforces, both of which a naive encoder gets
wrong:

**Block alignment.** A sample's length must be a multiple of 16, because the
hardware reads whole blocks. A loop point must also land on a block boundary,
since the loop address is a block address.

**The predictor runs across the loop.** Filters 1-3 depend on the previous two
decoded samples, so the first block after a loop wrap predicts from whatever
preceded it. The hardware convention is that the block at the loop point uses
filter 0 -- which needs no history -- so the wrap is seamless regardless of
where the loop came from. An encoder that leaves a predictive filter there
produces a click on every repeat.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

#: Samples per BRR block, and bytes per block.
SAMPLES_PER_BLOCK = 16
BYTES_PER_BLOCK = 9

#: Filter coefficients as (p1, p2) multipliers. Expressed as the exact
#: fractions the hardware uses rather than decimals, because the quantisation
#: is the point.
FILTERS: tuple[tuple[float, float], ...] = (
    (0.0, 0.0),
    (15 / 16, 0.0),
    (61 / 32, -15 / 16),
    (115 / 64, -13 / 16),
)

#: Range is a 4-bit left shift, 0-12 usable. 13-15 behave oddly on hardware
#: and are not emitted.
MAX_RANGE = 12

#: The decoded sample range. BRR decodes to 15-bit signed.
SAMPLE_MIN, SAMPLE_MAX = -32768, 32767


@dataclass
class BrrSample:
    """An encoded BRR sample and its block geometry."""

    data: bytes
    #: Block index the loop returns to, or None for a one-shot.
    loop_block: int | None
    #: Decoded length in samples.
    frames: int

    @property
    def blocks(self) -> int:
        return len(self.data) // BYTES_PER_BLOCK

    @property
    def size_bytes(self) -> int:
        return len(self.data)

    @property
    def loop_offset_bytes(self) -> int | None:
        """Byte offset of the loop point, which is what ARAM stores."""
        if self.loop_block is None:
            return None
        return self.loop_block * BYTES_PER_BLOCK

    def to_dict(self) -> dict[str, Any]:
        return {
            "blocks": self.blocks,
            "size_bytes": self.size_bytes,
            "frames": self.frames,
            "loop_block": self.loop_block,
            "loop_offset_bytes": self.loop_offset_bytes,
        }


def _clamp(value: float) -> int:
    return int(max(SAMPLE_MIN, min(SAMPLE_MAX, round(value))))


def _encode_block(
    samples: np.ndarray,
    previous: tuple[int, int],
    *,
    allowed_filters: tuple[int, ...] = (0, 1, 2, 3),
) -> tuple[bytes, tuple[int, int], float]:
    """Encode sixteen samples, choosing the cheapest filter and range.

    Brute force over every filter and range, scored by the error the DECODER
    would produce -- which is the only error that matters, and it differs from
    the quantisation error because the predictor feeds back. An encoder that
    scores the nibble error instead drifts.
    """
    best: tuple[bytes, tuple[int, int], float] | None = None
    for filter_index in allowed_filters:
        c1, c2 = FILTERS[filter_index]
        for shift in range(MAX_RANGE + 1):
            p1, p2 = previous
            nibbles: list[int] = []
            error = 0.0
            for target in samples:
                prediction = c1 * p1 + c2 * p2
                residual = float(target) - prediction
                # The nibble is the residual shifted down, clamped to 4 bits.
                quantised = int(round(residual / (1 << shift))) if shift else int(
                    round(residual)
                )
                quantised = max(-8, min(7, quantised))
                decoded = _clamp(prediction + (quantised << shift))
                error += (decoded - float(target)) ** 2
                nibbles.append(quantised & 0xF)
                p2, p1 = p1, decoded
            if best is None or error < best[2]:
                header = (shift << 4) | (filter_index << 2)
                payload = bytes(
                    (nibbles[i] << 4) | nibbles[i + 1]
                    for i in range(0, SAMPLES_PER_BLOCK, 2)
                )
                best = (bytes([header]) + payload, (p1, p2), error)
    assert best is not None
    return best


def encode(
    samples: np.ndarray,
    *,
    loop_start: int | None = None,
    scale: float = 32767.0,
) -> BrrSample:
    """Encode float samples in [-1, 1] as BRR.

    `loop_start` is in samples and is rounded DOWN to a block boundary, since
    the hardware's loop address is a block address and rounding up would skip
    audio. The block at the loop point is forced to filter 0, which needs no
    predictor history, so the wrap is seamless whatever preceded it.
    """
    pcm = np.clip(np.asarray(samples, dtype=np.float64) * scale, SAMPLE_MIN, SAMPLE_MAX)
    # Pad to a whole number of blocks: the hardware reads blocks, not samples.
    remainder = pcm.size % SAMPLES_PER_BLOCK
    if remainder:
        pcm = np.concatenate([pcm, np.zeros(SAMPLES_PER_BLOCK - remainder)])

    loop_block: int | None = None
    if loop_start is not None:
        loop_block = max(0, int(loop_start) // SAMPLES_PER_BLOCK)

    out = bytearray()
    previous = (0, 0)
    total_blocks = pcm.size // SAMPLES_PER_BLOCK
    for index in range(total_blocks):
        block = pcm[index * SAMPLES_PER_BLOCK : (index + 1) * SAMPLES_PER_BLOCK]
        # Filter 0 at the loop point and at the very start: neither has valid
        # predictor history.
        allowed = (0,) if index == 0 or index == loop_block else (0, 1, 2, 3)
        encoded, previous, _error = _encode_block(
            block, previous, allowed_filters=allowed
        )
        header = encoded[0]
        if index == total_blocks - 1:
            header |= 0x01  # end
            if loop_block is not None:
                header |= 0x02  # loop
        elif loop_block is not None:
            header |= 0x02
        out.append(header)
        out.extend(encoded[1:])

    return BrrSample(
        data=bytes(out), loop_block=loop_block, frames=int(pcm.size)
    )


def decode(sample: BrrSample, *, frames: int | None = None) -> np.ndarray:
    """Decode BRR to floats in [-1, 1].

    Decodes straight through; looping is the player's job, not the decoder's.
    """
    wanted = frames if frames is not None else sample.frames
    out = np.zeros(wanted, dtype=np.float64)
    p1 = p2 = 0
    position = 0
    for index in range(sample.blocks):
        offset = index * BYTES_PER_BLOCK
        header = sample.data[offset]
        shift = (header >> 4) & 0xF
        filter_index = (header >> 2) & 0x3
        c1, c2 = FILTERS[filter_index]
        for byte in sample.data[offset + 1 : offset + BYTES_PER_BLOCK]:
            for nibble in ((byte >> 4) & 0xF, byte & 0xF):
                if position >= wanted:
                    return out / 32767.0
                value = nibble - 16 if nibble >= 8 else nibble
                decoded = _clamp(c1 * p1 + c2 * p2 + (value << shift))
                out[position] = decoded
                p2, p1 = p1, decoded
                position += 1
    return out / 32767.0


def round_trip_error(
    samples: np.ndarray, *, loop_start: int | None = None
) -> float:
    """RMS error of encode-then-decode, as a fraction of full scale.

    The honest measure of what BRR costs. Reported rather than hidden: a 4-bit
    residual over a 2-tap predictor is lossy, and how lossy depends entirely
    on the material.
    """
    encoded = encode(samples, loop_start=loop_start)
    decoded = decode(encoded, frames=len(samples))
    reference = np.asarray(samples, dtype=np.float64)[: len(decoded)]
    return float(np.sqrt(np.mean((decoded - reference) ** 2)))


def blocks_for(frames: int) -> int:
    """Blocks a sample of `frames` samples needs."""
    return (frames + SAMPLES_PER_BLOCK - 1) // SAMPLES_PER_BLOCK


def bytes_for(frames: int) -> int:
    """ARAM bytes a sample of `frames` samples costs, encoded."""
    return blocks_for(frames) * BYTES_PER_BLOCK
