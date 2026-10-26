# The question was wrong: 60%, not 25%

Every quality figure reported for RSE up to this point answered "is this good
music" — Best/Good/OK/Bad, "is either one good", a 1–6 scale. The product
question is **"would this pass as NES game music"**, and the NES carried a
very wide range of styles. Those are different bars, and only the second one
is the deliverable.

Re-asked on the *same twenty files* of the corrected baseline, with no
regeneration:

| question | result |
|---|---|
| Best or Good | 5/20 — **25%** |
| **Would pass as NES game music** | **12/20 — 60%** |

## The two questions relate cleanly

| rating | passes | fails |
|---|---|---|
| BEST | 1 | 0 |
| GOOD | 4 | 0 |
| OK | **7** | 1 |
| BAD | 0 | 7 |

Every Best and Good piece passes. **Seven of eight OK pieces pass too** — that
is the entire gap. No Bad piece passes, so the two judgements agree at both
extremes and diverge exactly in the middle band, where "not the best music I
have heard" and "would work in a game" are both true at once.

This is not a rescoring or a relaxed threshold. It is a different question,
and the one that matters.

## What this invalidates

Reported figures that answered the wrong question, and so understate the
system by an unknown amount:

- "25% Best-or-Good" as the corrected baseline headline
- "the batch was 99% not good" for the architecture comparison
- "only 2 of 6 are any good" in the source-versus-NES audit
- every conclusion of the form "composition quality is the binding
  constraint", which was inferred from those numbers

The *relative* findings survive, because both arms of each comparison were
asked the same wrong question: Fermata beating native RSE 7/12 to 1/12, the
NES render beating the GM source 5 of 6, the timing correction winning 6–2 on
compressed pieces. Those are unaffected.

What does not survive is the absolute claim. "Composition quality is the main
problem" was built on 25%. At 60% the picture is a system that already
produces usable game music in the majority of attempts.

## Pass rate by scenario

| scenario | passes | rate |
|---|---|---|
| ominous | 4/4 | **100%** |
| town | 3/4 | 75% |
| fast_rhythmic | 2/4 | 50% |
| action | 2/4 | 50% |
| exploration | 1/4 | **25%** |

Four pieces each, so exploratory. But `ominous` passing 4/4 and `exploration`
1/4 is a 4x spread, and `exploration` has now been the weakest scenario in
three consecutive batches. That is a scenario-definition problem, not a
composer problem, and it is cheap to act on.

## One metric finally separates something

Motif participation: **0.500 for passing pieces against 0.123 for failing
ones**, a 4x gap. Every other measure is flat (notes/bar 23.2 vs 21.1, IOI
entropy 1.68 vs 1.73, pitch-class entropy 2.65 vs 2.64).

This is worth stating carefully, because the same metric has failed three
times before — against Best/Good/OK/Bad it was *inverted*, with a median of
0.00 in the Best/Good band. It separates "passes as game music" and does not
separate "is good music". With n=20 and a history of this measure misleading
us, it is a hypothesis for a future prospective test, **not** a selection
rule, and nothing should be filtered on it.

## Correction to the record

The system does not have a composition-quality crisis. It has a 60% pass rate
on eight-bar pieces, a working full-song pipeline, and one weak scenario.
Earlier reports said otherwise because of how the listener was asked.
