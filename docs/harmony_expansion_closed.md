# Harmony expansion closed. Two experiments, 48 pairs, zero preference.

| experiment | pairs | control | treatment | ties |
|---|---|---|---|---|
| Stage 1 — harmonic rhythm | 24 | 7 | **0** | 17 |
| Stage 2 — harmonic vocabulary | 24 | 0 | **0** | **24** |

Listener verdict on Stage 2: *"they are all ties. i can barely tell them apart
and when i can, they just sound like two valid arrangements."*

That second sentence is the finding. Not "both are bad" — **both are valid.**
The harmony is already doing its job, so making it more sophisticated changes
nothing a listener can act on.

Per the operator's pre-stated stopping rule, harmony expansion stops here and
the next milestone is melodic phrasing.

## What the two experiments together establish

Stage 2 was the stronger test and it produced the weaker signal. Its pairs
shared motif, rhythm, tempo, percussion and chord-change rate, so harmony was
nearly the only variable — and it produced 24 ties against Stage 1's 17. The
more precisely harmony was isolated, the less it mattered.

The property that made the procedural composer beat Fermata was never
harmonic *sophistication*. It was harmonic *coordination*: all four voices
derived from one progression, so they cannot disagree. That was already true
in v1 with three progressions per style. Adding borrowed dominants, five
harmonic rhythms and nine progression families added real theory and no
audible benefit.

Both expansions stay in the codebase, opt-in, defaulting off. They cost
nothing and `two_bar_hold` and the borrowed V remain plausible tools for a
specific request. Neither is promoted.

## Where the headroom actually is, measured

The melody restates the same motif on every chord. Section A of
`action_s900`, eight bars:

| bar | onset rows | intervals |
|---|---|---|
| 0 | 0, 2, 4, 8, 10, 12 | 1, -1, 3, -3, 1 |
| 1 | 0, 2, 4, 8, 10, 12 | 2, 1, 0, -3, 0 |
| 2 | 0, 2, 4, 8, 10, 12 | 1, 2, 0, -3, 0 |
| … | identical | … |
| 7 | 0, 2, 4, 8, 10, 12 | 1, 2, 0, -3, 0 |

**One distinct rhythmic placement across all eight bars.** Every bar attacks
on rows 0, 2, 4, 8, 10, 12. The pitches move because the chord moves; the
rhythm never does.

So there is no antecedent and consequent, no difference between a mid-phrase
cadence and a final one, no four- or eight-bar arc. A tune that asks a
question in bars 1–4 and answers it in bars 5–8 is not currently expressible,
because the generator has no unit larger than one bar.

That is consistent with the operator's own earlier assessment of the melody
generator — "coherent, limited vocabulary" — and with the one complaint the
procedural composer has attracted in its own right: a riff that is fine the
first time and wearing by the fourth.

## Next milestone: phrase-level melody

The concrete gap is a unit larger than the bar. What that needs, in rough
order of likely value:

1. **Antecedent/consequent pairs.** Generate a 2-bar idea and a 2-bar answer
   that ends differently, then treat the 4-bar pair as the unit that repeats.
2. **Cadence-aware phrase endings.** A phrase ending mid-section should land
   unresolved; one ending a section should land on a chord tone, ideally the
   tonic. The harmonic information to do this already exists in `Voicing`.
3. **Rhythmic variation within a phrase.** At minimum, the answer should not
   attack on exactly the same rows as the question.
4. **Call and response between voices.** P1 and P2 are both pulse channels;
   handing the figure between them is idiomatic NES writing and costs no new
   theory.

Same discipline as before: v1 stays the default, every addition is opt-in,
and nothing is promoted without a blind paired listening result. Given two
consecutive nulls, the next experiment should also be prepared to return one —
and if phrasing also ties, the honest conclusion is that the procedural
composer is at its ceiling and the remaining gap is not reachable by rules of
this kind.
