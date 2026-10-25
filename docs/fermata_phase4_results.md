# Phase 4: Fermata 0.2B through the NES pipeline

CPU only, no fine-tuning, frozen D4 codec untouched. Reproduce with

```bash
python scripts/fermata_generate.py --seeds 0 1 --max-new-tokens 1200 \
  --out fermata_generations
python scripts/fermata_to_nes.py --render 6 --out fermata_nes_results.json
```

Checkpoint `haster/Fermata-0.2B-v1`, revision
`18b647098f579ee41db67c0ad4c7f7f39ce0eac1`, weights sha256
`4ca6a5db3a2d1510bad7acde4d313a9ad732fc91dcda4776f044c5b5bd8ea9b7`,
201,398,272 parameters. Sampling at the card's defaults: temperature 0.7,
top-k 24, repetition penalty 1.12, 1,200 new tokens, seeds 0 and 1.

## The tokenizer had to be reconstructed

The repository contains **no Python at all** — nine files, zero `.py`, only
`tokenizer/vocab.json` and `tokenizer/bpe.json` as data. The card refers to
"this project's pipeline" without linking one, and no newer release exists.
So `rse/transfer/fermata.py` is a **reconstruction**, marked
`rse-fermata-reconstruction/1.0.0` on every artifact it produces.

The grammar was recovered from the checkpoint rather than guessed: sampling
from `<BOS>` with the dead BPE tokens suppressed made it emit its own
structure.

```
<BAR> <TRK_n> ( <BEAT_b> <SUBPOS_s> <PITCH_p> <DURATION_d> <VELOCITY_v> )*
```

Two details a reasonable guess gets wrong: **DURATION precedes VELOCITY**, and
**`<BEAT_>` is the beat within the bar**, not an absolute index.

Validated by perplexity on the repository's own promo samples:

| | correct | shuffled body | DURATION/VELOCITY swapped |
|---|---|---|---|
| promo_rock | **0.2254** | 10.7501 | 9.2196 |
| promo_piano_concerto | **0.4058** | 11.3553 | 9.6820 |

Swapping two tokens of five raises NLL fortyfold, which pins that specific
field order. Round trips preserved every event (2,490/2,490 and 1,322/1,322).

**The effective vocabulary is 2,159, not 9,507.** 7,348 entries are `<BPE_*>`
which the documented builder bug means the model never saw, so three quarters
of its embedding table and output softmax are dead. The codec never emits one
and suppresses them at sampling.

## Generation: 12 samples, 6 scenarios, 2 seeds

Scenarios use the controls Fermata actually has — genre and ensemble labels,
tempo, key, track plan. It has no free-text conditioning.

| | median | min | max |
|---|---|---|---|
| tokens generated | 1,212 | 1,210 | 1,212 |
| bars | 10.5 | 4 | 19 |
| notes (pre-arrangement) | 130 | 62 | 234 |
| percussion (pre-arrangement) | 100 | 0 | 170 |
| distinct pitches | 27 | 20 | 48 |

**No generation emitted `<EOS>`.** All twelve were cut off by the 1,200-token
budget, so the model never chose to stop. That is not "eight bars completed" —
it is eight bars never asked for and never concluded, and bar counts of 4 to 19
mean length is uncontrolled. Recorded, not repaired.

Generation runs at 17–34 tokens/second on this CPU, about 60 seconds per
sample.

## NES arrangement

`rse/transfer/arranger.py` assigns roles by behaviour (lowest register becomes
the triangle bass, most active of the rest becomes the melody), resolves
chords on a monophonic channel by keeping the outer voice, steals voices on
overlap, and counts every decision.

| | median | min | max |
|---|---|---|---|
| event retention | **0.844** | 0.597 | 1.000 |
| polyphony dropped | 36 | 0 | 94 |
| voice-stolen | 0 | 0 | 0 |

**All 12 arrangements are hardware-compliant** after the constraint engine,
with no violations remaining. **All four channels are used in 9 of 12**, and
the three exceptions are explainable: `sparse` (both seeds) had no drum track
in its plan, and `action_s0` left P2 silent. **Percussion is present in 10 of
10 generations whose plan included a drum track.**

Arrangement costs 16% of events at the median and up to 40% in the worst case,
almost entirely to polyphony the APU cannot play. That is the real cost of
path C, and it is measured rather than assumed.

## Musical quality after NES conversion

| metric | corpus | native 10k (t=0.7) | native 10k (t=0.85) | **Fermata → NES** |
|---|---|---|---|---|
| pitch-class entropy | — | **0.23** | 1.71 | **2.62** |
| stepwise fraction | 0.370 | 0.032 | 0.206 | 0.536 |
| notes per bar | ~24 | 38.4 | 24.2 | 18.4 |
| percussion | 27% of note-ons | **0** | **0** | **68 median** |
| channels used | 4 | 2–3 | 2–3 | **4 in 9/12** |
| n-gram repetition | 0.393 | 0.936 | 0.000 | 0.385 median |

The two findings that matter:

**Fermata does not collapse to a single pitch.** Pitch-class entropy 2.62
median against the native model's 0.23 at the temperature where it produced
most notes. The native model's t=0.7 output was a hammered single pitch at 16
onsets per second; nothing in these twelve looks like that, and the worst case
(ominous_s1, 0.81) is still three times better.

**Fermata generates percussion, reliably.** Every generation with a drum track
in its plan produced noise-channel events. The native model produced zero from
a corpus that is 27.2% percussion.

### What the repetition median does *not* show

I need to correct my own first reading. The n-gram repetition median of 0.385
sits almost exactly on the corpus's 0.393, which looks like a clean result. It
is not: the distribution is **bimodal**, the same failure shape the native
model had.

```
0.000  action_s0        0.556  exploration_s0
0.000  fast_rhythmic_s1 0.593  town_s0
0.000  ominous_s1       0.769  ominous_s0
0.000  town_s1          0.779  sparse_s0
0.089  sparse_s1        0.867  fast_rhythmic_s0
0.214  exploration_s1   1.000  action_s1
```

Only **3 of 12** land in the corpus middle band (0.15–0.65); four are at
exactly zero and one at 1.000. The median is the midpoint of a split, not a
central tendency.

In fairness the corpus is itself bimodal (22% zero, 23% middle, 17% at or
above 0.8) and Fermata's split (33% / 25% / 17%) is not obviously different —
but **12 samples cannot distinguish those distributions**, and no claim either
way is supported.

### A metric correction

I previously reported that `unique_bar_fraction` was broken under codec D
because sub-row offsets prevent identical bars from hashing together, and said
the loop detector was blind. `bar_signatures` now takes `quantize_ticks`, and a
controlled test confirms it works as intended: a bar repeated one sub-row slot
later reads as repeated (0.50) with row tolerance and is missed (0.00) without.

But applied to this output it **changes nothing** — unique stays 1.000 for all
twelve, because no two bars are near-identical. So my diagnosis of *why* the
metric read 1.000 on the native model's degenerate output was wrong: that was a
hammered note with jittered timing, which is not bar-aligned repetition at all.
Neither bar metric detects that failure, and pitch-class entropy is the right
detector for it. The quantization option is correct and worth keeping; it is
not the fix I claimed it was.

## Recommendation

**Not yet a recommendation between paths — one control is missing.** The native
100,000-step baseline has not run (its pod lost `mido` to a container-disk
reset), so every comparison above is against a model that had 10,000 steps on a
harder representation and was still improving when it stopped. Calling Fermata
better on that basis would be comparing a trained model to an undertrained one.

What the evidence does support:

- **Path C is demonstrated end to end.** Fermata composes, the arranger
  collapses it to four voices at 0.844 median retention, the frozen codec
  encodes it, the hardware validates, and it renders. That pipeline works.
- **Path B is viable representationally** (Phase 2: zero event loss at 24
  TPQN) but unattractive for a different reason now visible: 77% of Fermata's
  vocabulary is dead, so fine-tuning inherits an embedding table three
  quarters of which has never been trained.
- **Fermata's weakness is control, not content.** It never terminates, bar
  counts vary 4 to 19, and repetition is bimodal. Those are the problems a
  conditioning layer or an arranger-side length policy would address — and
  they are precisely what RSE's own machinery was built for.

So the architecture question remains open, and the next step is the native
baseline, not more Fermata work.

## Listening samples

`fermata_nes_audio/` holds six NES renders through the same RSE renderer used
for every other comparison in this project. `transfer_audio/` holds the Phase 2
reference/24/48-TPQN sets. Both are gitignored: the corpus is research-only.

## Licensing

Fermata's weights and card are Apache-2.0, so a derivative is unencumbered —
the one candidate of three without that problem. MIDI-GPT's weights remain
non-commercial research only. Contamination is unresolved for all three; a
memorization screen against NES-MDB is a prerequisite before any quality claim
is published, and the evaluation already has corpus fingerprinting for it.
