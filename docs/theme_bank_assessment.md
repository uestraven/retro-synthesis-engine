# The opening bank is not a theme bank

Phase 1 asked whether the first two bars of the five Best/Good pieces contain
a recognisable musical idea. They mostly do not.

| theme | source | rated | idea? | pitched notes | drum hits | what plays in bars 1–2 |
|---|---|---|---|---|---|---|
| 1 | fast_rhythmic_s11032 | BEST | **YES** | 7 | 39 | bass (prog 38) + drums |
| 2 | action_s13662 | GOOD | WEAK | 5 | 10 | bass (prog 38) + drums |
| 3 | ominous_s7261 | GOOD | **NO** "just drums" | **0** | 24 | **drums only** |
| 4 | town_s15040 | GOOD | WEAK "potential but not long enough" | 26 | 23 | all four tracks |
| 5 | town_s15041 | GOOD | **NO** | 10 | 24 | trk0 = 1 note; trk2 = 4 notes all on pitch 50 |

**1 YES, 2 WEAK, 2 NO.** The listening judgement and the token data agree
exactly: the piece called "just drums" contains zero pitched notes.

## Why: Fermata often does not open with melody

In three of the five, bars 1–2 hold no melodic voice at all — only program 38
(Synth Bass) and percussion. The declared leads enter later. Across all 20
baseline pieces:

| voice | enters in bar 0 | first-entry bars observed |
|---|---|---|
| drums | 18/20 | 0, 1 |
| program 80 (Lead 1) | 12/20 | 0, 1, 2, 5, 6, 7 |
| program 81 (Lead 2) | 9/20 | 0, 1, 2, 3, 4, 5 |
| program 38 (bass) | 12/20 | 0, 1, 2, 3 |

5 of 20 pieces have **both** declared leads absent from bar 0, and three of
those five are in this bank. `action_s13662`, rated GOOD, does not bring in
either lead until **bar 5** — its first five bars are bass and drums.

This is not a defect. Starting on a drum-and-bass vamp and bringing the tune
in afterwards is ordinary game-music practice. But it does mean the premise of
the experiment — supply a coherent two-bar musical idea — cannot be tested
with these prefixes, because in most of them there is no melodic idea present
to supply.

## Consequence

Running 15 guided continuations from this bank would test whether a
bass-and-drum vamp helps, not whether a musical theme helps. The per-theme
breakdown would rest on a single YES opening.

Phase 1 existed to catch exactly this before Phase 3 spent 30 generations on
it. Nothing has been generated.

The listener's own note on theme 4 — "potential but not long enough" — points
at the fix: two bars is 2.7–5.2 seconds, which is short for a theme at any
tempo, and program 80 enters by bar 2 in 16 of 20 pieces. A four-bar window
would contain melodic material in nearly every case while still being an
opening rather than an excerpt from the middle.

No openings were replaced. All five stay as recorded.
