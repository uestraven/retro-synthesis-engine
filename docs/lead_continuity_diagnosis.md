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
