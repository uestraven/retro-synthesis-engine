# Corrected-pipeline baseline: the quality answer

20 eight-bar pieces, four each from `action`, `town`, `exploration`,
`ominous`, `fast_rhythmic`. `sparse` suspended. First batch where both
conversion defects are fixed and every piece passed an eight-point structural
gate. Nothing ranked, filtered or reordered.

## Result

| rating | n | share |
|---|---|---|
| Best | 1 | 5% |
| Good | 4 | 20% |
| OK | 8 | 40% |
| Bad | 7 | 35% |

**Best or Good: 5/20 (25%). OK or better: 13/20 (65%).**

## Correctness is solved; quality is not

All 20 pieces pass eight complete bars, declared meter, tempo preserved
(0.000% error on every piece), no negative ticks, onsets inside the window,
requested instrumentation, and zero hardware violations. Duration matches
`1920/bpm` per scenario to a tenth of a second.

And a third of the batch is still Bad, with the same vocabulary as every
previous round: "too random" ×2, "random notes" ×2, "no melody", "harmonies
make no sense". **Fixing the pipeline did not fix the music.** That was the
hypothesis worth testing and it is now answered.

## The correction did not rescue the pieces it touched

6 of 20 pieces had dropped their opening `<BAR>` and were repaired by
`d40fa80`. They rated *worse* than the well-formed ones:

| | n | Best | Good | OK | Bad | mean rank (0 = Best) |
|---|---|---|---|---|---|---|
| had implicit bar | 6 | 0 | 1 | 1 | **4** | 2.50 |
| well-formed | 14 | 1 | 3 | 7 | 3 | 1.86 |

Two readings, and the data cannot separate them at n=6: either dropping the
opening bar correlates with a weaker composition — the model losing the
grammar and losing the thread together — or it is noise. Worth watching, not
worth acting on.

## By scenario

| scenario | Best | Good | OK | Bad | mean rank |
|---|---|---|---|---|---|
| town | 0 | 2 | 2 | 0 | 1.50 |
| ominous | 0 | 1 | 3 | 0 | 1.75 |
| fast_rhythmic | 1 | 0 | 1 | 2 | 2.00 |
| action | 0 | 1 | 1 | 2 | 2.25 |
| exploration | 0 | 0 | 1 | 3 | 2.75 |

`town` and `ominous` produced no Bad pieces; `exploration` produced three and
no Good ones. With four pieces per scenario this is exploratory. It is also
the third batch in which `town` has sat at the top, which is at least
consistent.

## No metric separates the bands

| band | n | notes/bar | IOI H | motif | stepwise | PC entropy | n-gram rep |
|---|---|---|---|---|---|---|---|
| Best/Good | 5 | 21.1 | 1.71 | **0.00** | 0.69 | 2.53 | 0.48 |
| OK | 8 | 26.2 | 1.82 | 0.50 | 0.31 | 2.68 | 0.51 |
| Bad | 7 | 22.8 | 1.68 | 0.25 | 0.17 | 2.62 | 0.39 |

Motif participation has a **median of 0.00 in the Best/Good band** and 0.25 in
Bad — the reverse of what the retired corpus-distance ranking weighted most
heavily. Stepwise fraction is the only measure that orders the bands
monotonically (0.69 / 0.31 / 0.17), and the Best piece has a stepwise fraction
of 0.05 while a Good piece has 1.00, so even that does not survive contact
with individual pieces. This is now the third independent failure of
statistical typicality to predict this listener's judgement.

## Comparison with earlier batches

Not directly comparable: `listen_batch2` used a 1–6 scale, in a different
session, with `sparse` in the pool. The coarse split, `sparse` excluded:

| | good-or-better | OK-or-better | bad |
|---|---|---|---|
| batch 2, non-sparse (n=25) | 24% | 56% | 24% |
| corrected baseline (n=20) | 25% | 65% | 35% |

No improvement in the top band. Any claim that the timing fix raised absolute
quality is unsupported — and the A/B test already said its effect was bounded.
What the fix bought is a baseline that means what it says.

## Where this leaves the three questions

- **Correctness**: evidenced. Two deterministic defects found, fixed, and
  regression-tested; 20/20 structurally valid.
- **Quality**: 25% Best-or-Good is the honest number, on a pipeline that no
  longer distorts the composer. The ceiling is Fermata's, not ours.
- **Control**: untouched. No scenario reliably produces good music; the best
  scenario produced no Bad pieces out of four, which is not control.
