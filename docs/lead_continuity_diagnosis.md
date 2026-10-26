# Where the melody gets lost: a unit bug in the arranger

**Answer to the Phase 1 question: neither Fermata nor the arranger weakens the
lead. The arranger rescales time, so eight Fermata bars become a different
number of NES bars, and every "second-half decay" measurement was reading an
8-bar window over a piece that had already finished.**

## Fermata's lead is healthy before arrangement

Measured on all 12 Fermata pieces from the blind architecture comparison, each
reproduced from its recorded seed (12/12 reproduced, checked against the event
count the comparison recorded). The intended lead is the track declared
`<TRACK_80>` — GM program 80, Lead 1 (square) — plan index 0 in every
scenario, rather than whichever channel happens to look active.

| measure | Fermata's own score | corpus reference |
|---|---|---|
| lead second-half / first-half note count | **median 1.143** (min 0.41, max 1.91) | ~1.000 |
| bars with no lead note | **0 in 11 of 12 pieces** | — |
| lead notes per piece | 13–105 | — |

There is no composition-side continuity defect to fix.

## The arranger rescales the music by `tempo_bpm / 150`

`rse/transfer/arranger.py`:

```python
tick_sec = (row_sec * ROWS_PER_BEAT) / score.ticks_per_quarter
out_spt  = (60.0 / (score.tempo_bpm or 120.0)) / ticks_per_beat
scale    = tick_sec / out_spt
```

`tick_sec` hard-codes the output tempo: `row_sec=0.1` with `ROWS_PER_BEAT=4`
is 0.4 s per beat, i.e. **150 BPM**. `out_spt` uses the *score's* tempo. So
`scale` stretches or squashes every onset and duration by exactly
`score.tempo_bpm / 150`.

The docstring says `row_sec` "fixes the output tempo", and it does — but the
arranged composition keeps `score.tempo_bpm` in its time map, so the two
disagree and the music is rescaled by the ratio between them. Eight bars of
Fermata become `8 × tempo_bpm / 150` NES bars. Only at 150 BPM is it eight.

Predicted against observed, all 12 pieces:

| scenario | BPM | predicted bars | observed bars | notes/bar |
|---|---|---|---|---|
| sparse | 76 | 4.1 | **4** | 21.0 / 46.2 |
| ominous | 92 | 4.9 | **5** | 55.4 / 30.2 |
| exploration | 120 | 6.4 | **7** | 24.7 / 30.6 |
| town | 132 | 7.0 | **7** | 30.3 / 21.0 |
| action | 168 | 9.0 | **9** | 25.4 / 28.0 |
| fast_rhythmic | 180 | 9.6 | **11** | 21.6 / 19.7 |

## What this explains, all previously misattributed

- **"Second-half lead activity falls to 78.8%"** (16-bar paired analysis). A
  fixed 16-bar window over a piece that ends at bar 8 reads as decay. The
  pre-arrangement lead does not decay.
- **`sparse` "lead voice stops playing"** — 270→3, 238→0, 135→0, 86→1 notes
  across halves. At 76 BPM a 16-bar generation occupies ~8 NES bars, so the
  second half of the window is empty *by construction*.
- **Empty trailing bars, 8 of 12 Fermata pieces** in the architecture
  comparison, against 2 of 12 native. The native model writes D4 directly and
  never passes through the arranger.
- **"Too fast. Too chaotic."** The same notes compressed into fewer bars
  inflates density: the two pieces the listener called chaotic are the two
  densest at 55.4 and 46.2 notes/bar, both from slow scenarios.
- **Fermata's 8-bar pieces rendering 10.8 s** where 8 bars at `row_sec=0.1`
  is 12.8 s.

## Secondary finding: the arranger ignores the declared plan

`assign_roles` picks the lead behaviourally — lowest register becomes bass,
then the highest mean absolute interval among the rest becomes melody. On 6 of
12 pieces that disagrees with the header, and track 0 (program 80) is demoted
to accompaniment while track 1 (program 81) is promoted. Twice the promoted
track is nearly empty: 3 notes beating 40, and 13 beating 64.

It is a real defect but **not** what makes pieces unlistenable. Plan
agreement does not predict the blind listening tier:

| arranger agrees with plan | "not bad" | partial | "not musical" |
|---|---|---|---|
| yes (6) | 2 | 1 | 3 |
| no (6) | 2 | 2 | 2 |

The likely reason it is perceptually inert: melody maps to P1 and
accompaniment to P2, and both are pulse channels with the same timbre. The
swap moves the tune between two near-identical voices. It would matter if the
lead landed on the triangle or were dropped — neither happens here.

## What is now in doubt

The time bug contaminates any measurement taken over a fixed bar window on
arranged Fermata output:

- the 78.8% lead-decay figure
- the `sparse` lead-collapse diagnosis
- `new_pitches` / `lost_pitches` in the restatement comparison, where a
  partly-empty second half inflates "lost"
- `empty_trailing_bars` as a Fermata quality signal

Contour recurrence (0.739 against a corpus median of 0.667) is less affected,
since it is computed over melody notes rather than bar positions, but it
should be re-measured after the fix.

The **architectural decision is unaffected in direction**: Fermata won 7/12
against 1/12 *while* 8 of its 12 pieces were being time-rescaled. The margin
understates it. Nothing about the native model's result depended on the
arranger.

## Recommended intervention

One change, in the arranger: make the output time base consistent so eight
composed bars become eight NES bars regardless of the scenario tempo. Either
write the implied 150 BPM into the arranged composition's time map, or map
score ticks to destination ticks without the tempo term — the two are
equivalent, and the second keeps `row_sec` as the single source of output
tempo it is documented to be.

Then re-measure lead continuity and re-run the length comparison, because both
prior results were reading this bug.

The role-assignment defect should be fixed separately and afterwards, so the
two interventions stay distinguishable.

---

# Correction applied, and the research record reclassified

The fix is in `rse/transfer/arranger.py` plus `row_seconds()` and the six
callers that built a `GridFit`. Regression coverage is
`tests/test_arranger_timing.py` (50 tests at 90/120/150/180/210 BPM).

## The second half of the bug

Fixing the arranger alone would have passed every unit test and still produced
wrongly timed music. `rse/tokenizers/tokenizer.py` overrides the Tempo token
with `fit.bpm` whenever the fit is aligned, and every caller passed
`GridFit(period_sec=0.1)` — which *is* 150 BPM. The encode put back exactly
what the arranger had stopped doing. `row_seconds(bpm)` now derives the row
length from the score's tempo, and the callers use it.

## Historical versus corrected, same twelve scores

The historical branch runs the real pre-fix arranger, extracted from commit
`5a6c456` with `git show` and loaded as a module, together with
`GridFit(period_sec=0.1)` so both defects fire. **It reproduces all twelve
archived WAVs byte-for-byte** (SHA-256 after identical peak normalisation), so
this compares against the audio that was actually judged rather than a
reconstruction of it.

| measure | historical | corrected |
|---|---|---|
| encoded BPM | **149.0 on every piece** | source BPM, 0.00% error on all 12 |
| bars spanned (median) | 6.73 | **8.02** |
| span not ≈8 bars | **12 / 12** | 3 / 12 |
| playback duration (median) | 10.8 s | 15.3 s |
| events | 213 | 215.5 |
| empty bars in window | 1 | **0** |
| lead notes | 46.5 | 46.5 |
| **lead second/first-half ratio** | **0.730** | **1.143** |
| lead distinct pitches | 10.5 | 10.5 |
| lead contour recurrence | 0.400 | 0.389 |
| hardware violations | **0** | **0** |

**The lead-activity decline disappears.** 0.730 → 1.143 against a corpus
reference of ~1.000, with the same number of lead notes and the same distinct
pitches — the notes were always there; the measurement window was wrong.

Contour recurrence barely moves (0.400 → 0.389), which is consistent: it is
computed over melody notes rather than bar positions, so it was never
sensitive to this bug.

The three corrected pieces that are not ≈8 bars are Fermata's own output, not
conversion:

- both `fast_rhythmic` pieces have a full 8-bar window plus 29 and 19 events
  beyond it. Fermata's `<BEAT_b>` spans 0–223 "to allow very long bars", and
  the model wrote bars longer than four beats. A separate finding.
- `sparse_s108301` has per-bar counts `[16, 23, 46, 45, 36, 32, 8, 0]`: its
  own final bar is empty.

## Reclassified findings

**Invalidated** — read a fixed bar window over arranged Fermata output:

- second-half lead activity at 78.8% of the first half
- the `sparse` "lead voice stops playing" diagnosis (270→3, 238→0, 135→0,
  86→1 notes); at 76 BPM a 16-bar generation occupied ~8 NES bars
- `empty_trailing_bars` as a signal of Fermata quality (8/12 vs native 2/12)
- `new_pitches` / `lost_pitches` against the corpus restatement reference
- notes-per-bar for slow scenarios, which the compression inflated to 46–55

**Requires re-evaluation** — judgements of affected audio. The ratings stand
as valid evidence about the audio that was played; what they can no longer
support is a claim about Fermata's composition:

- the 30-piece blind batch (`listen_batch2`)
- the paired 8-vs-16-bar length test (`listen_paired`)
- the Fermata arm of the architecture comparison, including its 7/12
- "chaotic", "too fast" and pacing complaints on slow-scenario pieces

**Still valid** — never passed through the arranger, or measured before it:

- the native model's training behaviour, its IOI entropy of 0.000, and its
  1/12 result. The native path writes D4 directly.
- the codec comparison and the tempo-precision work
- the hardware constraint tests, and `hardware_violations == 0` throughout
- the corpus restatement reference (train split, corpus-side only)
- every Phase 1 measurement on Fermata's **symbolic scores**, including the
  lead staying active in 11 of 12 pieces — the finding that made the bug
  findable
- the architectural decision itself. Fermata won 7/12 to 1/12 while 8 of its
  12 pieces were being time-rescaled, so the margin understates it.

## Open question

Whether correcting the timeline makes the music better is a listening
question, and `scripts/timing_ab_listen.py` builds the twelve blind A/B pairs
to answer it. No claim of subjective improvement is made here.

## Blind A/B result: the correction is preferred exactly where the bug bit

Twelve pairs, one composition each, judged A / B / = with the treatment
hidden and A/B assigned by an independent coin per pair.

**Corrected 6, historical 2, no preference 4.**

The raw tally is not the strong evidence. This is:

| the bug's effect on the piece | n | corrected | historical | tie |
|---|---|---|---|---|
| **compressed** (historical < 8 bars, 76–132 BPM) | 8 | **6** | 1 | 1 |
| **expanded** (historical > 8 bars, 168–180 BPM) | 4 | **0** | 1 | 3 |

Preference for the correction appears where the conversion shortened the
music and disappears where it lengthened it. A dose-response that tracks the
defect's own magnitude is much harder to explain away than a 6–2 margin,
because it is a prediction the bug makes and a bare preference does not.

Ordered by compression, the pattern is nearly monotone:

| BPM | historical bars | outcome |
|---|---|---|
| 76 | 4.06 | historical |
| 76 | 3.17 | corrected |
| 92 | 4.91 | corrected |
| 92 | 4.91 | corrected |
| 120 | 6.41 | corrected |
| 120 | 6.41 | corrected |
| 132 | 7.06 | corrected |
| 132 | 7.05 | tie |
| 168 | 8.95 | tie |
| 168 | 9.00 | historical |
| 180 | 10.83 | tie |
| 180 | 10.81 | tie |

### Caveats, including one against the result

**There is an order effect.** Six of the eight decisive judgements chose A,
and the corrected version sat in A in 7 of 12 pairs, so some of the margin is
position rather than treatment. Broken out: with corrected in A it won 4 and
lost 0 (3 ties); with corrected in B it won 2 and lost 2 (1 tie).

The dose-response survives this, because position is not confounded with
compression — corrected sat in A in 5 of 8 compressed pairs and 2 of 4
expanded ones. Two compressed pairs were won by the corrected version from
the **B** slot, against the order effect (`pair_01` at 132 BPM, `pair_10` at
92 BPM), and both expanded pairs where corrected held the favourable A slot
came back ties.

**`sparse` at 76 BPM split 1–1**, the only compressed scenario that did not
favour the correction. `pair_02` preferred historical despite the corrected
version restoring a dead lead (half-ratio 0.0 → 1.25) and reaching 8.02 bars.
At 76 BPM the corrected piece runs about 25 seconds where the historical one
ran 6.5, and the honest reading is that correctly-timed slow music can be
merely slow. `sparse` remains suspended.

**No comments were recorded**, so there is no evidence here about pacing,
melody or endings specifically — only about overall preference.

**Preference is not quality.** This establishes that corrected beats
historical on compressed pieces. It does not establish that the corrected
audio is enjoyable: the same listener called the batch these came from "99%
not good". Four ties and the whole fast-tempo half say the correction is a
real repair with a bounded effect, not a transformation.

---

# Phase 1 audit: the dropped opening `<BAR>`

The two `fast_rhythmic` pieces carrying 29 and 19 events past their eight-bar
window were not writing extended measures. Both earlier guesses were wrong.

## `<BEAT_b>` is interpreted correctly

Across 4,769 events in the 30 stored `listen_batch2` streams, **zero** events
sit at `BEAT >= beats_per_bar`. Max beat is 3, meter is `4/4` throughout. The
vocabulary does carry 215 `<BEAT_>` values up to 223, and ~260 `<METER_>`
tokens including compound ones like `<METER_2+3/4>`, but the model uses none
of that here. Units: `<BEAT_b>` is the beat within the bar, `<SUBPOS_s>` is
one of 24 ticks per quarter. Both confirmed against the decoder.

## The actual mechanism

Fermata sometimes emits notes **before its first `<BAR>`**. The grammar is
`<BAR> <TRK_n> (events)*`, so the opening `<BAR>` was dropped. `decode`
tracked bars from `-1`, so those notes took negative ticks:

    tick = bar * SUBPOS * beats_per_bar + beat * SUBPOS + subpos
         = -96 + beat * 24 + subpos

The D4 round trip then shifts the minimum tick to zero. Confirmed in
isolation: `[-1920, 0, 1920]` comes back as `[0, 1920, 3840]`. The whole piece
moves one bar later, so eight requested bars span nine and the final bar's
onsets land past the window — 29 and 19 events, matching those pieces' bar-7
counts exactly.

## Frequency

4 of 30 pieces, 61 of 4,769 events (1.28%), in four of six scenarios:

| scenario | pre-`<BAR>` events | pieces hit |
|---|---|---|
| exploration | 24 | 1 / 5 |
| sparse | 22 | 1 / 5 |
| fast_rhythmic | 12 | 1 / 5 |
| town | 3 | 1 / 5 |
| action | 0 | 0 / 5 |
| ominous | 0 | 0 / 5 |

The same 61 events across 4 pieces appear in the paired 16-bar streams.

## Nothing downstream caught it

Hardware validation reported **zero violations** on both affected pieces, and
arrangement retention was 1.000. Validation checks channels, pitches and
polyphony; no layer in the pipeline ever validated bar structure. The defect
was invisible to every existing check, which is why it survived three
listening tests.

## Correction

A leading note is read as an **implicit opening bar** — the only reading
consistent with the grammar — rather than as bar `-1`. `trim_to_bars` counts
that implicit bar, so a request for eight bars returns eight bars of music
instead of nine.

Nothing is clamped, truncated or rewritten: all 61 events survive, cross-bar
sustains keep their full duration, and `DecodeReport` now carries
`implicit_first_bar` and `n_events_before_first_bar` so the condition is
reported rather than silent. The frozen D4 representation is untouched, and
the meter still comes from the `<METER_>` token rather than an assumption.

On the two affected pieces, after the fix: minimum tick 0, span 7.88 bars,
**0 onsets beyond bar 8**, with 14 and 22 pre-`<BAR>` events correctly
attributed.

Regression coverage is `tests/test_fermata_implicit_bar.py` (11 tests),
including a guard that a bare `<TRK_>` before the first `<BAR>` does *not*
imply a bar, so well-formed streams do not gain a phantom one.
