# Variety — 6 songs per arm. NULL RESULT.

Rated 2026-10-26. Arm A was v1; arm B had `variety=full` (per-song key, tempo
rung, form archetype, accompaniment figure, bass figure, re-orchestrated A'
sections, and the extra progressions in the pool).

## Result

**All 12 files pass as NES game music, including all six variety songs.**
So the change broke nothing — that gate is clear.

**"Which arm had more variety?" — no real difference.** The batch still
sounded like variations of one song.

That is a null result on four axes at once, and it is the third
arrangement-level experiment to return one:

| experiment | pairs | outcome |
|---|---|---|
| harmonic rhythm | 24 | no preference |
| harmonic vocabulary | 24 | no preference |
| key / tempo / form / arrangement | 6 per arm | no difference |

`variety=full` stays implemented and **off**.

## What this means

The measurements said the change worked: harmony and bass went from 5–6
unique across 20 seeds to 17–20, forms from 1 to 2–5, keys from 1 to 5,
tempos from 1 to 2. Every number moved and the listener heard no difference.

So the metric was wrong, not the music. **Byte-uniqueness is not
perceptual distinctness.** The melody already measured 20 unique out of 20
seeds while sounding the same every time, because all 20 are built from the
same five motif shapes re-anchored on different chord tones. Counting
distinct note sequences cannot detect that; it counts the anchoring, not the
idea.

The next thing to measure is the melodic vocabulary itself — distinct
contours and rhythmic profiles rather than distinct byte strings — and the
per-voice material that never varies at all: one drum kit per style, no
counter-melody by default, 4/4 only, and no duty-cycle or dynamic changes.
