# Harmonic rhythm: no improvement. v1 stays the default.

24 matched pairs, same style/seed/key/tempo/progression/motif, chord-change
rate the only intended difference. Blind, A/B shuffled per pair.

**Treatment won 0 of 24.**

| outcome | pairs |
|---|---|
| control (one chord per bar) preferred | **7** |
| treatment (new pacing) preferred | **0** |
| no preference | 17 |

Pass rate is unchanged: **24/24 for both versions**. The listener's summary was
"this was a tough one. they're all passable" — so the new pacing does not
break anything, it just never earns a preference.

Against the stated success criterion — *"simply matching that result is not an
improvement"* — this is a clear negative. Harmonic rhythm is **not** promoted
to a default and should not enter Prompt-to-Song.

## The position check matters here

The listener chose A 6 times and B once, with 17 ties, and the treatment sat
in A for 15 of 24 pairs. Broken out:

| treatment position | n | control won | treatment won | tie |
|---|---|---|---|---|
| A | 15 | 1 | 0 | 14 |
| B | 9 | **6** | 0 | 3 |

So nearly every decisive judgement came from pairs where the treatment was in
B — i.e. the listener picked the *first* thing they heard. That is consistent
with an order effect rather than a judgement about pacing, and it makes the
7–0 margin weaker than it looks.

But it does not rescue the treatment: **it won zero pairs from either
position.** An order effect would have given it some wins from the A slot and
it got none.

## By rhythm, with the sample sizes stated

| rhythm | n | control | treatment | tie |
|---|---|---|---|---|
| two_bar_hold | 8 | 0 | 0 | **8** |
| accelerating | 7 | 3 | 0 | 4 |
| two_per_bar | 6 | 3 | 0 | 3 |
| tonic_pedal | 3 | 1 | 0 | 2 |

`two_bar_hold` is the only pacing that never lost — 8 ties, no losses. That is
the slowest alternative, closest to the control, which is the least
interesting possible reading: the further pacing moves from one chord per bar,
the worse it does. `accelerating` and `two_per_bar`, the two that change
chords faster, account for 6 of the 7 control wins.

At n=3 to 8 per rhythm, none of this supports a per-rhythm claim.

## Truncation is exonerated

The two truncating pairs — `tension` under `tonic_pedal`, each losing 16
melody notes and 128 rows, 8 bars of duration out of 24 — both came back
**ties**. Neither lost.

So the hard-truncation rule I imposed is not what cost the treatment its
preference, and the harmony-aware sustain policy would not have rescued it.
The correction to the music theory stands on its own merits — a common tone
held across a chord change really is not a clash — but implementing it is not
the fix for this result.

`tension` and `town` are the two styles where the treatment never lost at all
(4 ties each), and they are also the two slowest and sparsest.

## Conclusion

Faster harmonic movement does not make this music better. The property that
made the procedural composer work was that all voices share one harmonic
plan, and changing chords more often does not add to that — it slightly
degrades it, in the direction of busier.

Per the operator's decision table, the result is "no improvement": **preserve
v1 as default and proceed to progression expansion cautiously.** The rhythm
code stays in place and opt-in, since it costs nothing and `two_bar_hold`
remains a plausible tool for a slow style, but nothing selects it
automatically.

The more useful signal for Stage 2 is that 17 of 24 pairs were
indistinguishable. Harmonic *pacing* is not where the remaining headroom is.
Whether harmonic *vocabulary* is remains untested — that is Stage 2, and this
result does not predict it either way.
