"""The frozen representation, named and versioned.

Branch B selected this configuration by measurement on the training split, with
validation confirming every ordering and the official test split untouched. The
numbers behind each choice are in `docs/codec_comparison.md`.

It lives in its own module, and the historical defaults in `TokenizerConfig`
are left exactly as they were, because the previous foundation and every Gate 3
artifact index a 272-token vocabulary. Those checkpoints are historical
baselines, not loadable weights for this representation: the vocabulary, the
context and the number of events in a window all differ.

Why each setting:

- **codec D at k=4** -- sub-row onsets *and* sub-row durations. C's row-unit
  durations are structurally incompatible with sub-row onsets, so 35% of its
  events had their length overridden by voice stealing against D's 1%. k=4
  rather than k=8 because k=8 pushes 12% of windows past a reasonable context
  against k=4's 8%, for a recall gain of 0.056.
- **tempo ratio 1.01** -- cuts absolute onset error 39.9 ms to 16.1 ms and
  note-to-note jitter 12.2 ms to 4.8 ms, within 14% of the unquantizable
  floor. 1.005 was measured and rejected: `int(round(...))` makes integer BPM
  the floor below about ratio 1.025, so a finer ratio buys nothing.
- **tempo range 20-380 BPM** -- fitted row rates span 21.5 to 374.9, and
  `fit_grid` *declines* a piece whose rate falls outside the token range
  rather than clipping it. The old 40-360 therefore rejected pieces outright.
- **8-bar windows at 3072 tokens** -- 8 bars because the melodic structure
  being learned needs the context, and 3072 because at 2304 the 7.9% of
  windows that overflow carry three times the events of those that fit.
  Excluding them would clip the densest music out of the corpus.
"""

from __future__ import annotations

from rse.tokenizers.vocab import TokenizerConfig

#: Bump when any field below changes. Caches and checkpoints record it, and a
#: mismatch is refused rather than reconciled.
REPRESENTATION_ID = "rse-repr/1.0.0-d4k4-tempo1.01"

#: Grammar revision the conditioning prefix follows. Separate from the
#: representation id because a prefix change can invalidate a cache without
#: touching the codec.
CONDITIONING_GRAMMAR_VERSION = "rse-prefix/1.0.0-motif"

CODEC = "D"
SUB_ROW_DIVISIONS = 4
TEMPO_RATIO = 1.01
TEMPO_MIN = 20
TEMPO_MAX = 380

#: Training context, in tokens.
CONTEXT_TOKENS = 3072
#: Default training unit, in bars.
WINDOW_BARS = 8


def frozen_config(**overrides: object) -> TokenizerConfig:
    """The frozen tokenizer configuration.

    `overrides` exist for measurement scripts that need to vary one field
    while holding the rest; training uses the defaults.
    """
    settings: dict[str, object] = {
        "codec": CODEC,
        "sub_row_divisions": SUB_ROW_DIVISIONS,
        "tempo_ratio": TEMPO_RATIO,
        "tempo_min": TEMPO_MIN,
        "tempo_max": TEMPO_MAX,
    }
    settings.update(overrides)
    return TokenizerConfig(**settings)  # type: ignore[arg-type]
