# The procedural composer beats the pretrained model

Listener verdict on the 24-piece procedural batch: **"fantastic. no
complaints."**

Against the same question and the same renderer:

| composer | would pass as NES game music |
|---|---|
| Fermata 0.2B + RSE arrangement | 12/20 — **60%** |
| procedural (rules + seeded randomness) | **24/24 reported with no complaints** |

The procedural composer was written in one working session. No model, no
training, no GPU, no dataset. Fermata is 201 million pretrained parameters
with a reconstructed codec, a corrected arranger, two fixed timing defects and
several weeks of work behind it.

## Why rules won here, as far as the evidence shows

The failure the listener kept reporting for Fermata was harmonic and
directional: "harmonies make no sense", "too random", "no melody", "random
notes". One of those pieces had been arpeggiated 0%, truncated 0%, lost
nothing and merged nothing — the arranger never touched it, so the harmony was
the composer's own.

A sampled model produces four individually plausible lines that can disagree
with each other. Four lines derived from one chord progression cannot. That is
the whole mechanism, and it needed no intelligence.

The procedural composer also writes *into* the four monophonic channels
instead of onto them, so three of the transformations that degraded Fermata's
output never happen: no polyphony reduction, no role inference, no chord
arpeggiation.

## What this does not establish

- **n=24, one listener, one session.** No per-piece data was recorded, so
  which styles worked is unknown.
- **Six hand-written presets.** The generator can only produce what its rules
  describe. It has no prompt interpretation yet.
- **Diversity is bounded by the rule set, and measurably so.** Across 200
  seeds per style: all 200 melodies are distinct, but only 13–25 distinct
  motifs appear and exactly **3 distinct progressions**, because each style
  declares three. Every `boss` piece is one of three harmonic plans. 24 pieces
  hid that; 100 would not.
- **It cannot be surprising.** A rule-based composer has no capacity to write
  something its author did not anticipate, which is the one thing a pretrained
  model offers.

## Architecture consequence

The NES adaptation milestone was justified by Fermata being the best composer
available. On this evidence it is not, so the LoRA pilot should be
deprioritised rather than cancelled — the representation audit says the
vocabulary is lossless relative to D4 (recall 1.0000 at 10 ms), so that option
stays open and cheap to revisit.

The immediate work is to make the procedural composer the product:

1. **Wire it into Prompt-to-Song** as the composer, so a natural-language
   prompt selects and parameterises a style instead of conditioning Fermata.
2. **Widen the rule set**, because three progressions per style is the
   measured ceiling on harmonic variety.
3. **Then** consider the hybrid: procedural harmony and form, with Fermata
   supplying candidate melodies over a fixed progression. That is the one
   arrangement in which the pretrained model's unpredictability is an asset
   rather than a liability, and it should not be built until the procedural
   baseline is the product.
