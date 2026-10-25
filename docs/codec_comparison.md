# Branch B: codec comparison

Which representation preserves the most musically useful information for its
vocabulary, sequence length and processing cost. Deterministic throughout: no
training, no sampling, no GPU. Reproduce with

```bash
python scripts/codec_comparison.py --split train --limit 600 --k 2 4 8 \
  --out codec_comparison_train.json
python scripts/codec_comparison.py --split valid --limit 400 --k 2 4 8 \
  --out codec_comparison_valid.json
```

Train selected; validation confirmed. The official test split is untouched and
the script refuses to read it without `--allow-test`, reserved for one frozen
candidate.

Scale: **600 train pieces / 1,173 windows / 376,720 events**, and **398
validation pieces / 691 windows**. Of 600 train pieces, 70 had no fitted row
and were skipped; 53 of 398 in validation.

## The codecs

| | onset resolution | duration resolution |
|---|---|---|
| A | 1 tracker row | 1 row |
| B | 1/k row, via a separate `Offset_` token | 1 row |
| C | 1/k row, folded into `Position_` | 1 row |
| D | 1/k row, folded into `Position_` | 1/k row |

## Results, train split

| codec | vocab | recall | spurious | grid MAE | abs MAE | dur MAE | IOI kept | tokens (med) | p90 | >2304 | trunc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 272 | 0.636 | 0.032 | 15.60 ms | 36.9 ms | 38.1 ms | 20% | 954 | 1648 | 0% | 0% |
| B2 | 274 | 0.700 | 0.038 | 9.74 ms | 36.5 ms | 31.5 ms | 43% | 1120 | 2078 | 5% | 15% |
| B4 | 276 | 0.758 | 0.041 | 5.71 ms | 36.1 ms | 24.8 ms | 65% | 1202 | 2407 | 12% | 25% |
| B8 | 280 | 0.814 | 0.044 | 2.91 ms | 36.5 ms | 20.0 ms | 76% | 1226 | 2743 | 16% | 35% |
| C2 | 288 | 0.700 | 0.038 | 9.74 ms | 36.5 ms | 31.5 ms | 43% | 1007 | 1887 | 3% | 15% |
| C4 | 320 | 0.758 | 0.041 | 5.71 ms | 36.1 ms | 24.8 ms | 65% | 1071 | 2141 | 8% | 25% |
| C8 | 384 | 0.814 | 0.044 | 2.91 ms | 36.5 ms | 20.0 ms | 76% | 1086 | 2424 | 12% | 35% |
| D2 | 320 | 0.700 | 0.038 | 9.74 ms | 36.5 ms | **18.2 ms** | 43% | 1007 | 1887 | 3% | **1%** |
| **D4** | 416 | 0.758 | 0.041 | 5.71 ms | 36.1 ms | **8.1 ms** | 65% | 1071 | 2141 | 8% | **1%** |
| D8 | 608 | 0.814 | 0.044 | 2.91 ms | 36.5 ms | **4.1 ms** | 76% | 1086 | 2424 | 12% | **2%** |

`trunc` is the share of events whose duration the hardware must override to
keep a monophonic channel monophonic.

### Event recall by class

| codec | attack | arpeggio | vibrato | retrigger | passage |
|---|---|---|---|---|---|
| A | 0.895 | 0.094 | 0.183 | 0.336 | 0.174 |
| k=2 | 0.926 | 0.198 | 0.326 | 0.560 | 0.314 |
| k=4 | 0.932 | 0.311 | 0.507 | 0.827 | 0.501 |
| k=8 | 0.934 | 0.398 | 0.646 | 0.863 | 0.725 |

Identical for B, C and D at each k, as they must be: all three quantize onsets
the same way and differ only downstream of that.

### Validation confirmation

Every ordering reproduces. Recall 0.674 → 0.839 across A → k=8 (train 0.636 →
0.814), grid MAE 10.84 → 2.50 ms, D8 duration MAE 3.5 ms against C8's 25.0 ms,
IOI retained 25% → 67%.

## Four findings

**1. Codec A discards most non-attack events, and attacks were never the
problem.** Attack recall is 0.895 under A while overall recall is 0.636. What
the row grid drops is arpeggio frames (0.094 kept), vibrato (0.183) and
`passage` events (0.174) — and `passage` is the class whose musical status is
genuinely uncertain, 13.8% of all source events, possibly real fast melody.
At k=8 those become 0.398, 0.646 and 0.725.

**2. Sub-row onsets recover rhythmic diversity, measured directly.** IOI
entropy retained goes from 20% to 76% end-to-end through encode and decode.
This is measured, not inferred from the earlier 97% entropy-loss
decomposition.

**3. C is the efficient way to buy onsets; B is not.** Identical fidelity at
every k, but C8 needs 1,086 median tokens against B8's 1,226 (−11%) and keeps
12% of windows inside context against B8's 16%. B's compensating advantage is a
smaller vocabulary (280 vs 384), which is the wrong thing to optimize:
vocabulary costs one embedding row each, while sequence length costs attention
quadratically.

**4. D is the only codec whose durations are real.** C's row-unit durations are
structurally incompatible with sub-row onsets — the shortest expressible note
is a whole row while onsets are 1/8 of one, so **35% of C8's events have their
duration overridden** by voice stealing. D8 needs it for 2%. Note-length
variability tells the same story: the corpus median CV is 1.32, A reconstructs
0.80, C8 reaches 0.98, D8 reaches 1.25.

## Recommendation: codec D at k=4

D is the fidelity winner at every k. The choice of k is a context-window
trade:

- **D4** — recall 0.758, grid MAE 5.71 ms, duration MAE 8.1 ms, 1% truncation,
  note-length CV 1.20 of 1.32, median 1,071 tokens, **8% of windows over
  2,304**, vocabulary 416.
- **D8** — recall 0.814, grid 2.91 ms, duration 4.1 ms, CV 1.25, median 1,086
  tokens, but **12% over context**.

D4 is the recommendation because context overflow is not a graceful
degradation: a cropped window loses music, and the Gate 3A measurement found
that cropping also changes the structural label a window carries, so the model
trains against labels describing music it cannot see. D8 becomes preferable if
the context is raised to about 2,560, which the p90 of 2,141 (D4) and 2,424
(D8) suggests would clear almost all of it.

## The finding that outranks the codec choice

**Absolute onset error is flat at ~36.5 ms for every codec**, including A,
while grid error falls from 15.60 ms to 2.91 ms. The difference never reaches
the output because the `Tempo_` token's geometric buckets (`tempo_ratio=1.03`)
mislabel a fitted row rate by a median **0.34%**, which accumulates across a
window — roughly 19 ms over 128 rows — and is identical for every codec.

So sub-row onsets buy an eightfold improvement in grid placement that is
currently inaudible, because a larger error sits downstream of it. Fixing the
tempo resolution is cheap in a way the codec choice is not: there is **one
Tempo token per sequence**, so a finer ratio costs vocabulary and no sequence
length at all. At ratio 1.01 the bucket error falls roughly threefold for about
150 extra tokens and zero additional context.

That should be done before or alongside adopting any sub-row codec. Otherwise
the codec's main measurable gain stays bottled up behind a label error.

## Caveats

- The event classes are heuristic interpretations of register-derived events,
  not independent musical annotations. They interpret what survived; they do
  not decide what counts, and every retained event is credited regardless of
  class.
- The half-row grouping threshold separates rapid activity from separate rows.
  It does not prove two close events cannot be independent musical attacks;
  NES music can articulate more than one note within a row.
- Melodic density here counts all pitched channels per 8-bar window (corpus
  median 195), which is a different definition from the lead-channel figure
  used in the generation analysis (median 50). The two are not comparable.
- 70 of 600 train pieces have no fitted row and are excluded from every column,
  so these figures describe the row-fitting subset of the corpus.
