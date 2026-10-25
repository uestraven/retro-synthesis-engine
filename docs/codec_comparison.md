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

---

# Tempo precision, and what it does not fix

Run with `scripts/tempo_precision.py --split train --limit 600`. Same source
pieces, window boundaries, event matching and monophony repair throughout.

| config | vocab | recall | grid MAE | abs MAE | dur MAE | jitter | drift | IOI |
|---|---|---|---|---|---|---|---|---|
| A/1.03 | 272 | 0.636 | 15.57 ms | 40.97 ms | 49.39 ms | 18.52 ms | −1.69 ms/s | 20% |
| A/1.01 | 386 | 0.653 | 15.57 ms | 19.76 ms | 49.38 ms | 8.39 ms | 1.39 ms/s | 20% |
| D4/1.03 | 416 | 0.758 | 5.70 ms | 39.88 ms | 11.01 ms | 12.22 ms | −1.71 ms/s | 65% |
| **D4/1.01** | 530 | 0.779 | 5.70 ms | 16.05 ms | 9.54 ms | **4.79 ms** | 1.54 ms/s | 65% |
| A/exact | 386 | 0.655 | 15.57 ms | 12.08 ms | 49.70 ms | 7.80 ms | 0.01 ms/s | 20% |
| D4/exact | 530 | 0.783 | 5.70 ms | 6.81 ms | 9.23 ms | 4.21 ms | 0.01 ms/s | 65% |

`exact` is an oracle: the true fitted row rate, unquantized. It is a ceiling
measurement, not a deployable configuration — a decoder has only the Tempo
token to rebuild time from.

Absolute error decomposes into three things a listener experiences very
differently, fitted per window:

- **offset** — every onset shifted equally, because the encoder anchors the
  first sounding row at bar 0. Inaudible.
- **drift** — error growing with elapsed time, from a mislabelled global row
  rate. A sub-0.2% playback-speed change, not a rhythmic defect.
- **jitter** — what remains: note-to-note timing error. This is what sounds
  sloppy, and what a finer grid fixes.

## The drift mechanism is confirmed, not assumed

Signed error against elapsed time, which is the discriminating evidence:

| elapsed | A/1.03 | A/1.01 | D4/1.01 | D4/exact |
|---|---|---|---|---|
| 0 s+ | −7.02 | −7.37 | −9.02 | −9.04 |
| 2 s+ | −10.38 | −7.54 | −8.92 | −9.19 |
| 4 s+ | −14.16 | −7.90 | −8.51 | −9.79 |
| 6 s+ | −15.91 | −8.26 | −7.88 | −9.13 |

Under ratio 1.03 the error grows with time; at 1.01 and with an exact rate it
is flat. Per window the fitted slope matches the label error arithmetically: a
0.165% rate error is 1.65 ms per second, and 1.54 was measured.

## Answers to the three questions

**How much absolute error is tempo quantization?** 70% of codec A's
(40.97 against a 12.08 oracle) and 83% of D4/1.03's (39.88 against 6.81).

**What does sub-row resolution contribute independently?** Holding tempo
exact, A → D4 cuts absolute error 12.08 → 6.81, jitter 7.80 → 4.21, grid error
15.57 → 5.70, duration error 49.70 → 9.23, and raises IOI retention 20% → 65%.

**How close does D4/1.01 get to achievable?** On jitter, 4.79 against the
oracle's 4.21 — within 14% of the floor. On absolute error, 16.05 against
6.81. The 9 ms residual is almost entirely drift, so it is a global speed
error of under 0.2% rather than rhythmic damage.

## Why 1.01 and not finer

Ratio 1.005 was measured and rejected: median label error 0.169% against
1.01's 0.176%. `int(round(...))` makes **integer BPM** the floor below about
ratio 1.025, so a finer ratio buys nothing at the low end of the range.
Non-integer labels were measured too and do not improve the median.

**Coverage gap found.** Fitted rates span 21.5 to 374.9 BPM, and 8 of 181
sampled pieces fall outside the `[40, 360]` token range. `fit_grid` declines
those rather than clipping them, which is part of why 70 of 600 pieces have no
fitted row. Widening the range to about `[20, 380]` would let roughly 4% more
of the corpus use the row grid at all — independent of the ratio question.

# Context policy: no silent truncation

`scripts/context_policy.py`, D4/1.01, train split.

8-bar windows (1,173 windows): median 1,071 tokens, p90 2,141, p95 2,435,
p99 3,153, **max 4,757**.

| context | over | share | density inside | density excluded | bias |
|---|---|---|---|---|---|
| 2048 | 135 | 11.5% | 247 | 708 | 2.87× |
| 2304 | 93 | 7.9% | 254 | 777 | **3.06×** |
| 2560 | 43 | 3.7% | 263 | 951 | 3.62× |
| 3072 | 15 | 1.3% | 268 | 1150 | 4.29× |
| 4096 | 2 | 0.2% | 269 | 1445 | 5.37× |

**Excluding overflow is not a neutral choice.** The windows that do not fit
carry about three times the events of those that do, and the bias worsens as
the context tightens. Dropping them would systematically discard the densest
music — exactly the arpeggio and passage material D4 was adopted to preserve.

4-bar windows (2,274 windows) overflow 3 at 2,304 and **none at 2,560**.

**Recommendation: 8-bar windows at 3,072 context, splitting the 1.3% that
overflow at bar boundaries.** This keeps the 8-bar unit the motif analysis
depends on, loses no music, and introduces no density bias. 4-bar windows at
2,560 is the cheaper alternative but halves the musical context per example.

# Does D4 make conditioning viable?

`scripts/conditioning_gate2.py --codec D --k 4 --tempo-ratio 1.01`, 300 train
pieces, against codec A at 1.03. Label preservation through encode/decode:

| axis | A | D4/1.01 |
|---|---|---|
| motif | 0.738 | **0.840** |
| rhythm | 0.650 | 0.773 |
| phrase | 0.709 | 0.739 |

Motif eligibility lost on round-trip falls from 21 pieces to 12.

| label | prec A | prec D4 | recall A | recall D4 |
|---|---|---|---|---|
| motif `high` | 0.954 | 0.966 | 0.586 | **0.800** |
| rhythm `driving` | 0.481 | **0.706** | 0.342 | **0.316** |
| rhythm `moderate` | 0.765 | 0.821 | 0.503 | 0.742 |
| phrase `variation` | 0.717 | 0.667 | 0.745 | **0.588** |

**Motif conditioning is now substantially more viable**: preservation 0.84 and
`high` recall up from 0.586 to 0.800 at essentially unchanged precision.

**Repairing IOI entropy did not repair the `driving` label.** Its precision
improves sharply (0.481 → 0.706, so far fewer false positives) but recall does
not move (0.342 → 0.316). Two thirds of genuinely driving pieces still fail to
read as driving after a round trip, even with 65% of IOI entropy retained. So
rhythmic conditioning remains unsupported, and the cause is not the entropy
loss that was assumed to explain it.

**Phrase is mixed and should not be read as improved**: preservation barely
moves, `stable` recall rises 0.475 → 0.850 while `variation` recall falls
0.745 → 0.588, on only 86–92 eligible pieces.
